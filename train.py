"""Training pipeline for PCB defect detection.

Run it with: uv run python main.py train
Run it with QUICK=1 in front to test the whole pipeline in a few minutes.
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcb import CostModel, DefectDetector, PCBDataset

# Dataset on Roboflow Universe
WORKSPACE = "ai-dyci3"
PROJECT = "pcb-defect-eosje"
VERSION = 2

# Roboflow put crops of the same photo in train and test, so we split again by photo
SPLIT_SHARES = (0.7, 0.15, 0.15)
SPLIT_SEED = 0

# QUICK runs 1 epoch on 5 percent of the train images, only to test the code
QUICK = os.environ.get("QUICK") == "1"
BASE_MODEL = "yolo11n.pt"
EPOCHS = 1 if QUICK else 50
FRACTION = 0.05 if QUICK else 1.0
IMAGE_SIZE = 640
BATCH = 16

COSTS = {"tp": 10, "fn": 200, "fp": 2, "tn": 0}
BASE_RATE = 0.03  # share of boards that really have a defect
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

DATA_DIR = Path("datasets/pcb")
SPLIT_DIR = Path("datasets/pcb_by_photo")
WEIGHTS = Path("weights/pcb_yolo11n.pt")
RESULTS = Path("results")


def section(title):
    print()
    print("=" * 10, title, "=" * 10)


def plot_sweep(sweep, threshold, path):
    """Saves the cost curve with the chosen threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["cost"])
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"chosen threshold {threshold:.2f}",
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Expected cost per 1000 boards (EUR)")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_examples(scores, split_dir, folder, n=4):
    """Copies a few missed, false alarm and caught boards for the dashboard."""
    folder.mkdir(parents=True, exist_ok=True)
    picks = {
        "missed": scores.nsmallest(n, "defect_score"),
        "false_alarm": scores.nlargest(n, "clean_score"),
        "caught": scores.nlargest(n, "defect_score"),
    }
    for kind, rows in picks.items():
        for i, image in enumerate(rows["image"], start=1):
            shutil.copy(split_dir / "images" / image, folder / f"{kind}_{i}.jpg")
            shutil.copy(split_dir / "labels" / f"{Path(image).stem}.txt", folder / f"{kind}_{i}.txt")


def main():
    RESULTS.mkdir(exist_ok=True)

    section("1. Dataset")
    dataset = PCBDataset(
        workspace=WORKSPACE,
        project=PROJECT,
        version=VERSION,
        folder=DATA_DIR,
    ).download()
    roboflow_leakage = dataset.leakage()
    dataset.split_by_photo(
        folder=SPLIT_DIR,
        shares=SPLIT_SHARES,
        seed=SPLIT_SEED,
    )
    leakage = dataset.leakage()
    defects, images = dataset.describe()
    print(images)
    print(defects)
    print(f"Test images cut from a photo also in train: {roboflow_leakage:.1%} with the Roboflow split")
    print(f"Test images cut from a photo also in train: {leakage:.1%} with our split by photo")
    defects.to_csv(RESULTS / "dataset_defects.csv")
    images.to_csv(RESULTS / "dataset_images.csv")

    section("2. Finetune " + BASE_MODEL)
    run_dir = DefectDetector(BASE_MODEL).train(
        data=dataset.yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        fraction=FRACTION,
        save_to=WEIGHTS,
    )
    # keep the loss curves and plots, the weights are already in weights/
    shutil.copytree(
        run_dir,
        RESULTS / "training",
        ignore=shutil.ignore_patterns("weights"),
        dirs_exist_ok=True,
    )
    print("Best weights saved to", WEIGHTS)

    section("3. Per class results on the test split")
    finetuned = DefectDetector(WEIGHTS)
    metrics = finetuned.evaluate(dataset.yaml_path)
    per_class = pd.DataFrame(metrics.summary())
    print(per_class.round(3).to_string(index=False))
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    per_class.to_csv(RESULTS / "per_class.csv", index=False)

    section("4. Cost sweep, finetuned model against the baselines")
    coco = DefectDetector(BASE_MODEL)
    test_dir = dataset.folder / "test"
    scores = {
        "finetuned": finetuned.board_scores(test_dir),
        "coco_zero_shot": coco.board_scores(test_dir),
    }
    # the trivial baseline never draws a box, so every board passes
    scores["predict_nothing"] = scores["finetuned"].assign(defect_score=0.0, clean_score=0.0)
    # AP50 where any box on any defect counts, the only fair way to score a COCO model
    ap50 = {
        "finetuned": finetuned.evaluate(dataset.yaml_path, any_defect=True).box.map50,
        "coco_zero_shot": coco.evaluate(dataset.yaml_path, any_defect=True).box.map50,
        "predict_nothing": 0.0,
    }

    cost_model = CostModel(costs=COSTS, base_rate=BASE_RATE)
    sweeps, rows = {}, []
    for name in scores:
        sweeps[name] = cost_model.sweep(scores[name], THRESHOLDS)
        best = sweeps[name].loc[sweeps[name]["cost"].idxmin()]
        rows.append({"model": name, "ap50_any_defect": ap50[name], **best.to_dict()})
    comparison = pd.DataFrame(rows)
    print(comparison.round(3).to_string(index=False))

    section("5. Confusion matrix at the chosen threshold")
    threshold = float(comparison.loc[0, "threshold"])
    cost = float(comparison.loc[0, "cost"])
    matrix = finetuned.evaluate(dataset.yaml_path, conf=threshold).confusion_matrix.matrix
    labels = dataset.names + ["background"]
    confusion = pd.DataFrame(matrix, index=labels, columns=labels).astype(int)
    print("Rows are what the model predicted, columns are the truth")
    print(confusion.to_string())

    section("6. Save everything the dashboard needs")
    confusion.to_csv(RESULTS / "confusion_matrix.csv")
    comparison.to_csv(RESULTS / "baselines.csv", index=False)
    scores["finetuned"].to_csv(RESULTS / "test_scores.csv", index=False)
    finetuned.defect_scores(test_dir, dataset.names).to_csv(RESULTS / "defect_scores.csv", index=False)
    save_examples(scores["finetuned"], test_dir, RESULTS / "examples")
    sweeps["finetuned"].to_csv(RESULTS / "cost_sweep.csv", index=False)
    plot_sweep(sweeps["finetuned"], threshold, RESULTS / "cost_curve.png")
    summary = {
        "dataset": f"{WORKSPACE}/{PROJECT}/{VERSION}",
        "split_shares": SPLIT_SHARES,
        "split_seed": SPLIT_SEED,
        "test_leakage_roboflow": float(roboflow_leakage),
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "fraction": FRACTION,
        "image_size": IMAGE_SIZE,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "test_leakage": float(leakage),
        "costs": COSTS,
        "base_rate": BASE_RATE,
        "threshold": threshold,
        "cost_per_1000": cost,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved results to", RESULTS)
    print(f"Chosen threshold: {threshold:.2f} | expected cost: {cost:.0f} EUR per 1000 boards")


if __name__ == "__main__":
    main()
