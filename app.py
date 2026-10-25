"""Gradio dashboard for PCB defect inspection.

Run it with: uv run python main.py app
It only reads what train.py saved in weights/ and results/, it never trains.
"""

import functools
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import charts
from pcb import CostModel, DefectDetector, read_labels

RESULTS = Path("results")
EXAMPLES = RESULTS / "examples"
WEIGHTS = Path("weights/pcb_yolo11n.pt")

BOARDS_PER_DAY = 2000
REVIEW_MINUTES = 1.5
HOURLY_COST = 30  # euros per hour for one reviewer, overheads included
HOURS_PER_REVIEWER = 7  # hours one person can spend checking boards each day
REVIEWERS = 1

FONT = ImageFont.load_default(size=15)
TRUTH = "#00e5ff"  # cyan stands out on a green board
CSS = """
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;}
.kpi {border: 1px solid var(--border-color-primary); border-left: 6px solid #607d8b;
      border-radius: 10px; padding: 10px 14px; background: var(--block-background-fill);}
.kpi.good {border-left-color: #2e7d32;}
.kpi.bad {border-left-color: #c62828;}
.kpi .label {font-size: 13px; opacity: 0.7;}
.kpi .value {font-size: 26px; font-weight: 700;}
.kpi .note {font-size: 12px; opacity: 0.7;}
.verdict {color: white; font-size: 30px; font-weight: 700; text-align: center;
          padding: 14px; border-radius: 10px;}
"""
# the charts are drawn on white, so we always open the page in light mode
LIGHT_MODE = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.href = url.href;
    }
}
"""


class Dashboard:
    """Loads what training saved and computes every number the page shows."""

    THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
    GRID = np.round(np.arange(0, 1.0001, 0.02), 2)

    def __init__(self):
        self.summary = json.loads((RESULTS / "summary.json").read_text())
        self.scores = pd.read_csv(RESULTS / "test_scores.csv")
        self.defects = pd.read_csv(RESULTS / "defect_scores.csv")
        self.per_class = pd.read_csv(RESULTS / "per_class.csv")
        self.confusion = pd.read_csv(RESULTS / "confusion_matrix.csv", index_col=0)
        self.baselines = pd.read_csv(RESULTS / "baselines.csv")
        self.dataset = pd.read_csv(RESULTS / "dataset_defects.csv")
        self.training = pd.read_csv(RESULTS / "training" / "results.csv")
        self.detector = DefectDetector(WEIGHTS)

    @staticmethod
    def reviewers_needed(table, boards, minutes, hours):
        """Reviewers needed per day."""
        reviews_per_day = table["review"].sum() / 1000 * boards
        return reviews_per_day * minutes / 60 / hours

    @functools.lru_cache(maxsize=16)
    def analysis(self, fn, fp, tp, base_rate, boards, minutes, hourly, hours):
        """Cost sweep and cost of every threshold pair, cached per set of assumptions."""
        model = CostModel(
            costs={"tp": tp, "fn": fn, "fp": fp, "tn": 0},
            base_rate=base_rate / 100,
            review_cost=minutes / 60 * hourly,
        )
        sweep = model.sweep(self.scores, self.THRESHOLDS)
        rows = []
        for low in self.GRID:
            for high in self.GRID[self.GRID >= low]:
                table = model.outcomes(self.scores, low, high)
                rows.append({
                    "low": low,
                    "high": high,
                    "cost": model.cost(table),
                    "reviewers": self.reviewers_needed(table, boards, minutes, hours),
                })
        return model, sweep, pd.DataFrame(rows)

    def best_band(self, reviewers, fn, fp, tp, base_rate, boards, minutes, hourly, hours):
        """Cheapest thresholds the available reviewers can handle."""
        _, _, grid = self.analysis(fn, fp, tp, base_rate, boards, minutes, hourly, hours)
        feasible = grid[grid["reviewers"] <= reviewers]
        best = feasible.loc[feasible["cost"].idxmin()]
        return float(best["low"]), float(best["high"])

    def update(self, low, high, reviewers, fn, fp, tp, base_rate, boards, minutes, hourly, hours):
        """Recomputes the numbers and charts when a control changes."""
        model, sweep, grid = self.analysis(fn, fp, tp, base_rate, boards, minutes, hourly, hours)
        high = max(low, high)
        alone = sweep.loc[sweep["cost"].idxmin(), "threshold"]
        yours = "Model plus people, your thresholds"
        tables = {
            "Do nothing, ship every board": model.outcomes(self.scores, 1.01, 1.01),
            "People check every board": model.outcomes(self.scores, 0, 1.01),
            f"Model alone at {alone:.2f}": model.outcomes(self.scores, alone, alone),
            yours: model.outcomes(self.scores, low, high),
        }
        costs = {name: model.breakdown(table) for name, table in tables.items()}
        staff = {name: self.reviewers_needed(table, boards, minutes, hours) for name, table in tables.items()}
        totals = {name: sum(parts.values()) for name, parts in costs.items()}

        others = min(total for name, total in totals.items() if name != yours)
        table = tables[yours]
        cards = [
            ("Cost per 1000 boards", f"EUR {totals[yours]:,.0f}", f"best alternative EUR {others:,.0f}", None),
            ("Saving per day", f"EUR {(others - totals[yours]) / 1000 * boards:,.0f}", "against the best alternative", totals[yours] <= others),
            ("Sent to a person", f"{table['review'].sum() / 1000:.1%}", f"{table['review'].sum() / 1000 * boards:,.0f} boards a day", None),
            ("Reviewers needed", f"{staff[yours]:.1f}", f"{reviewers:g} available", staff[yours] <= reviewers),
            ("Escaped defects per day", f"{table.loc['defective', 'pass'] / 1000 * boards:.1f}", f"{table.loc['clean', 'fail'] / 1000 * boards:.0f} good boards rejected", None),
        ]
        return (
            kpi_cards(cards),
            charts.outcome_matrix(table),
            charts.board_flow(table),
            charts.strategy_costs(costs, staff),
            charts.score_distribution(self.scores, low, high),
            charts.cost_curve(sweep, self.summary["threshold"], low, high, totals[yours]),
            charts.cost_landscape(grid, reviewers, low, high),
            charts.recall_by_size(self.defects, low),
        )

    @functools.lru_cache(maxsize=32)
    def predict(self, path):
        """All boxes on an image with a confidence of at least 0.01."""
        result = self.detector.model.predict(
            path,
            conf=0.01,
            device=self.detector.device,
            verbose=False,
        )[0]
        boxes = pd.DataFrame(result.boxes.xyxy.cpu().numpy(), columns=["x1", "y1", "x2", "y2"])
        boxes["defect"] = [result.names[int(c)] for c in result.boxes.cls]
        boxes["confidence"] = result.boxes.conf.cpu().numpy().astype(float)
        return boxes

    def inspect(self, path, low, high):
        """Draws the boxes and gives the board its outcome."""
        if not path:
            return None, "", None
        high = max(low, high)
        boxes = self.predict(path)
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # true defects in cyan and a bit larger, we only know them for the example boards
        for cls, x, y, w, h in read_labels(EXAMPLES / f"{Path(path).stem}.txt"):
            draw.rectangle(
                [(x - w / 2) * width - 5, (y - h / 2) * height - 5, (x + w / 2) * width + 5, (y + h / 2) * height + 5],
                outline=TRUTH,
                width=3,
            )
        shown = boxes[boxes["confidence"] >= low].copy()
        shown["zone"] = np.where(shown["confidence"] >= high, "fail", "review")
        for box in shown.itertuples():
            color = charts.FAIL if box.zone == "fail" else charts.REVIEW
            label = f"{box.defect} {box.confidence:.2f}"
            corner = (box.x1, max(0, box.y1 - 18))
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=3)
            draw.rectangle(draw.textbbox(corner, label, font=FONT), fill=color)
            draw.text(corner, label, fill="white", font=FONT)

        score = boxes["confidence"].max() if len(boxes) else 0.0
        verdict, color = ("FAIL", charts.FAIL) if score >= high else ("REVIEW", charts.REVIEW) if score >= low else ("PASS", charts.PASS)
        html = (
            f'<div class="verdict" style="background:{color}">{verdict}</div>'
            f"<p>Board score {score:.2f}, the confidence of its most confident box. "
            f"Review from {low:.2f}, fail from {high:.2f}.</p>"
        )
        return image, html, shown[["defect", "confidence", "zone"]].round(3)


def kpi_cards(cards):
    """HTML cards for the headline numbers."""
    html = ""
    for label, value, note, good in cards:
        mood = "" if good is None else "good" if good else "bad"
        html += (
            f'<div class="kpi {mood}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>'
        )
    return f'<div class="kpis">{html}</div>'


def build(dash):
    """Builds the page and connects the controls to the charts."""
    costs = dash.summary["costs"]
    assumptions = [costs["fn"], costs["fp"], costs["tp"], dash.summary["base_rate"] * 100]
    assumptions += [BOARDS_PER_DAY, REVIEW_MINUTES, HOURLY_COST, HOURS_PER_REVIEWER]
    low_start, high_start = dash.best_band(REVIEWERS, *assumptions)
    examples = sorted(EXAMPLES.glob("*.jpg"))

    with gr.Blocks(title="PCB defect inspection") as demo:
        gr.Markdown(
            "# PCB defect inspection\n"
            "Every board gets a score, the confidence of its most confident box. "
            "Below the review threshold it ships, from the fail threshold on it is rejected, "
            "and in between a person checks it. The numbers come from "
            f"{len(dash.scores)} test boards the model never saw, scaled to your defect rate, "
            "and assume a person who checks a board gets it right."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Thresholds")
                low = gr.Slider(0, 1, value=low_start, step=0.01, label="Send to a person from")
                high = gr.Slider(0, 1, value=high_start, step=0.01, label="Fail automatically from")
                reviewers = gr.Number(REVIEWERS, label="Reviewers available", minimum=0)
                cheapest = gr.Button("Find the cheapest thresholds you can staff", variant="primary")
                with gr.Accordion("Business assumptions", open=True):
                    fn = gr.Number(assumptions[0], label="Escaped defect, EUR per board")
                    fp = gr.Number(assumptions[1], label="Good board rejected, EUR per board")
                    tp = gr.Number(assumptions[2], label="Repairing a caught defect, EUR per board")
                    base_rate = gr.Slider(0.5, 20, value=assumptions[3], step=0.5, label="Boards with a defect (%)")
                    boards = gr.Number(assumptions[4], label="Boards per day")
                    minutes = gr.Number(assumptions[5], label="Minutes to check one board")
                    hourly = gr.Number(assumptions[6], label="Reviewer cost, EUR per hour")
                    hours = gr.Number(assumptions[7], label="Review hours per person per day")

            with gr.Column(scale=3):
                kpis = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("Business impact"):
                        with gr.Row():
                            matrix = gr.Plot()
                            flow = gr.Plot()
                        strategy = gr.Plot()
                    with gr.Tab("Thresholds"):
                        distribution = gr.Plot()
                        with gr.Row():
                            curve = gr.Plot()
                            landscape = gr.Plot()
                    with gr.Tab("Where it fails"):
                        recall = gr.Plot()
                        with gr.Row():
                            gr.Plot(charts.box_confusion(dash.confusion))
                            gr.Plot(charts.per_class_bars(dash.per_class))
                        gr.Dataframe(dash.per_class.round(3), label="Per class results on the test split")
                    with gr.Tab("Inspect a board"):
                        gr.Markdown(
                            "Upload a board or pick an example. Cyan boxes are the true defects, "
                            "known only for the examples. Amber boxes go to a person, red boxes fail the board."
                        )
                        with gr.Row():
                            with gr.Column():
                                image = gr.Image(type="filepath", label="Board")
                                gallery = gr.Gallery(
                                    value=[(str(path), path.stem.replace("_", " ")) for path in examples],
                                    label="Test boards the model missed, raised a false alarm on, or caught",
                                    columns=4,
                                    height=420,
                                    allow_preview=False,
                                )
                            with gr.Column():
                                annotated = gr.Image(label="What the model sees")
                                verdict = gr.HTML()
                                detections = gr.Dataframe(label="Boxes above the review threshold")
                    with gr.Tab("Model and data"):
                        gr.Markdown(
                            f"Finetuned **{dash.summary['base_model']}** for {dash.summary['epochs']} epochs on "
                            f"`{dash.summary['dataset']}`, split again by photo. Test mAP50 "
                            f"**{dash.summary['map50']:.3f}**, mAP50-95 **{dash.summary['map50_95']:.3f}**. "
                            f"With the Roboflow split {dash.summary['test_leakage_roboflow']:.1%} of test "
                            "images came from a photo also in train, with ours none do."
                        )
                        gr.Dataframe(dash.baselines.round(3), label="Baselines on the same test split, with the costs used in training")
                        gr.Plot(charts.training_curves(dash.training))
                        gr.Dataframe(dash.dataset, label="Defects per class and split, median size as a share of the image side")

        controls = [low, high, reviewers, fn, fp, tp, base_rate, boards, minutes, hourly, hours]
        outputs = [kpis, matrix, flow, strategy, distribution, curve, landscape, recall]
        for control in controls:
            control.change(dash.update, controls, outputs)
        demo.load(dash.update, controls, outputs)
        cheapest.click(dash.best_band, controls[2:], [low, high])

        def pick_example(evt: gr.SelectData):
            return str(examples[evt.index])

        gallery.select(pick_example, None, image)
        for control in [image, low, high]:
            control.change(dash.inspect, [image, low, high], [annotated, verdict, detections])
    return demo


if __name__ == "__main__":
    demo = build(Dashboard())
    demo.launch(
        theme=gr.themes.Soft(),
        css=CSS,
        js=LIGHT_MODE,
    )
