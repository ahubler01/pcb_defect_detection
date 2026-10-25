"""The three building blocks of the PCB defect pipeline.

PCBDataset downloads the data from Roboflow, splits it by photo and describes it.
DefectDetector trains, evaluates and runs a YOLO model.
CostModel turns the model scores into euros per 1000 boards.
"""

import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

from api_key import load_api_key


def read_labels(file):
    """Reads a YOLO label file into rows of class, x, y, w, h."""
    if not file.exists():
        return np.zeros((0, 5))
    rows = [line.split()[:5] for line in file.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5)


def photo_of(path):
    """Name of the photo an image was cut from, e.g. 01_missing_hole_02."""
    return re.search(r"(\d{2}_[a-z_]+_\d{2})_\d+_600", path.name).group(1)


def pick_device():
    """Uses a GPU when there is one."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PCBDataset:
    """Downloads the dataset from Roboflow and describes what is inside."""

    SPLITS = ["train", "valid", "test"]

    def __init__(self, workspace, project, version, folder):
        self.workspace = workspace
        self.project = project
        self.version = version
        self.folder = Path(folder).resolve()
        self.yaml_path = self.folder / "data.yaml"

    def download(self):
        # Roboflow skips the download when the folder already exists
        rf = Roboflow(api_key=load_api_key())
        project = rf.workspace(self.workspace).project(self.project)
        project.version(self.version).download(
            model_format="yolov11",
            location=str(self.folder),
        )
        self.names = yaml.safe_load(self.yaml_path.read_text())["names"]
        return self

    def split_by_photo(self, folder, shares, seed):
        """Splits the data again so all images of a photo stay in the same split.

        Each defect type keeps the same train, valid and test shares.
        """
        photos = {}
        for image in self.folder.glob("*/images/*"):
            photo = photo_of(image)
            defect = photo[3:-3]  # 01_missing_hole_02 gives missing_hole
            photos.setdefault(defect, set()).add(photo)

        rng = random.Random(seed)
        split_of = {}
        for defect in sorted(photos):
            group = sorted(photos[defect])
            rng.shuffle(group)
            n_train = round(len(group) * shares[0])
            n_valid = round(len(group) * shares[1])
            for i, photo in enumerate(group):
                if i < n_train:
                    split_of[photo] = "train"
                elif i < n_train + n_valid:
                    split_of[photo] = "valid"
                else:
                    split_of[photo] = "test"

        # start from an empty folder so old files never mix in
        folder = Path(folder).resolve()
        shutil.rmtree(folder, ignore_errors=True)
        for split in self.SPLITS:
            (folder / split / "images").mkdir(parents=True)
            (folder / split / "labels").mkdir(parents=True)
        for image in self.folder.glob("*/images/*"):
            split = split_of[photo_of(image)]
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            shutil.copy(image, folder / split / "images")
            shutil.copy(label, folder / split / "labels")

        data = {
            "path": str(folder),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": self.names,
        }
        self.folder = folder
        self.yaml_path = folder / "data.yaml"
        self.yaml_path.write_text(yaml.safe_dump(data))
        return self

    def describe(self):
        """Counts images and defects per split, and the median defect size."""
        boxes = []
        for split in self.SPLITS:
            for file in (self.folder / split / "labels").glob("*.txt"):
                for cls, x, y, w, h in read_labels(file):
                    boxes.append({
                        "split": split,
                        "image": file.stem,
                        "class": self.names[int(cls)],
                        "size": np.sqrt(w * h),
                    })
        boxes = pd.DataFrame(boxes)

        # defects of each class in each split
        defects = pd.crosstab(boxes["class"], boxes["split"])[self.SPLITS]
        # median box side as a share of the image side
        defects["median_size"] = boxes.groupby("class")["size"].median().round(3)

        images = pd.DataFrame(
            {
                "images": [len(list((self.folder / s / "images").glob("*"))) for s in self.SPLITS],
                "with_defect": [boxes.loc[boxes["split"] == s, "image"].nunique() for s in self.SPLITS],
            },
            index=self.SPLITS,
        )
        return defects, images

    def leakage(self):
        """Share of test images cut from a photo that is also in train."""
        train = {photo_of(p) for p in (self.folder / "train" / "images").iterdir()}
        test = [photo_of(p) in train for p in (self.folder / "test" / "images").iterdir()]
        return np.mean(test)


class DefectDetector:
    """A YOLO model we can finetune, evaluate and use to score boards."""

    def __init__(self, weights):
        self.model = YOLO(weights)
        self.device = pick_device()

    def train(self, data, epochs, imgsz, batch, fraction, save_to):
        """Finetunes the model, saves the best weights and returns the run folder."""
        self.model.train(
            data=str(data),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
            device=self.device,
            name="pcb",
            exist_ok=True,
            seed=0,
        )
        save_to = Path(save_to)
        save_to.parent.mkdir(exist_ok=True)
        shutil.copy(self.model.trainer.best, save_to)
        return Path(self.model.trainer.save_dir)

    def evaluate(self, data, conf=0.001, any_defect=False):
        """Validates on the test split. With any_defect the class is ignored."""
        return self.model.val(
            data=str(data),
            split="test",
            conf=conf,
            single_cls=any_defect,
            device=self.device,
            plots=True,
            verbose=False,
        )

    def matches(self, split_dir):
        """Yields each image of a split with its labels, box confidences and IoUs."""
        results = self.model.predict(
            source=str(split_dir / "images"),
            conf=0.01,
            device=self.device,
            stream=True,
            verbose=False,
        )
        for result in results:
            image = Path(result.path)
            labels = read_labels(split_dir / "labels" / f"{image.stem}.txt")
            truth = torch.tensor(labels[:, 1:], dtype=torch.float32)
            boxes = result.boxes.xywhn.cpu()
            conf = result.boxes.conf.cpu().numpy()
            iou = box_iou(xywh2xyxy(truth), xywh2xyxy(boxes)).numpy()
            yield image, labels, conf, iou

    def board_scores(self, split_dir):
        """Top box confidence on a real defect and on clean copper, for each board.

        The dataset has no clean boards, so clean_score stands in for one.
        """
        rows = []
        for image, labels, conf, iou in self.matches(split_dir):
            # a box hits a defect when its IoU with a true box is at least 0.5
            hits = (iou >= 0.5).any(axis=0)
            rows.append({
                "image": image.name,
                "defect_score": conf[hits].max(initial=0) if len(labels) else np.nan,
                "clean_score": conf[~hits].max(initial=0),
            })
        return pd.DataFrame(rows)

    def defect_scores(self, split_dir, names):
        """Top confidence for every real defect."""
        rows = []
        for image, labels, conf, iou in self.matches(split_dir):
            for (cls, x, y, w, h), hits in zip(labels, iou >= 0.5):
                rows.append({
                    "image": image.name,
                    "class": names[int(cls)],
                    "size": np.sqrt(w * h),
                    "score": conf[hits].max(initial=0),
                })
        return pd.DataFrame(rows)


class CostModel:
    """Turns board scores into outcomes and euros per 1000 boards.

    Below low a board passes, from high it fails, in between a person checks it.
    """

    def __init__(self, costs, base_rate, review_cost=0.0):
        self.costs = costs  # euros for each of tp, fn, fp, tn
        self.base_rate = base_rate  # share of boards that really have a defect
        self.review_cost = review_cost  # euros for a person to check one board

    def outcomes(self, scores, low, high):
        """Boards per 1000 that pass, go to review or fail, split by the truth."""
        high = max(low, high)
        defect = scores["defect_score"].dropna().to_numpy()
        clean = scores["clean_score"].to_numpy()
        defective = 1000 * self.base_rate
        good = 1000 - defective
        return pd.DataFrame(
            {
                "pass": [defective * np.mean(defect < low), good * np.mean(clean < low)],
                "review": [
                    defective * np.mean((defect >= low) & (defect < high)),
                    good * np.mean((clean >= low) & (clean < high)),
                ],
                "fail": [defective * np.mean(defect >= high), good * np.mean(clean >= high)],
            },
            index=["defective", "clean"],
        )

    def breakdown(self, table):
        """Cost per 1000 boards by type. Assumes reviewers are always right."""
        return {
            "escaped defects": table.loc["defective", "pass"] * self.costs["fn"],
            "good boards rejected": table.loc["clean", "fail"] * self.costs["fp"],
            "defects handled": (table.loc["defective", "review"] + table.loc["defective", "fail"]) * self.costs["tp"],
            "good boards shipped": (table.loc["clean", "review"] + table.loc["clean", "pass"]) * self.costs["tn"],
            "review time": table["review"].sum() * self.review_cost,
        }

    def cost(self, table):
        return sum(self.breakdown(table).values())

    def sweep(self, scores, thresholds):
        """Cost of every threshold when the model decides alone."""
        rows = []
        for threshold in thresholds:
            table = self.outcomes(scores, threshold, threshold)
            tp, fn = table.loc["defective", "fail"], table.loc["defective", "pass"]
            fp, tn = table.loc["clean", "fail"], table.loc["clean", "pass"]
            rows.append({
                "threshold": threshold,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "recall": tp / (tp + fn),
                "false_alarm_rate": fp / (fp + tn),
                "cost": self.cost(table),
            })
        return pd.DataFrame(rows)
