"""Plotly charts for the dashboard. Each function takes plain data and returns a figure."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PASS, REVIEW, FAIL = "#2e7d32", "#f9a825", "#c62828"
DEFECTIVE, CLEAN = "#c62828", "#1565c0"
LAYOUT = dict(
    template="plotly_white",
    height=380,
    margin=dict(l=50, r=20, t=60, b=50),
    font=dict(size=13),
)


def outcome_matrix(table):
    """Pass, review and fail per 1000 boards."""
    # 0 is a good outcome, 1 a person checks it, 2 a costly mistake
    quality = [[2, 1, 0], [0, 1, 2]]
    meaning = [
        ["escaped", "found by a person", "caught by model"],
        ["shipped", "cleared by a person", "wrongly rejected"],
    ]
    text = [
        [f"<span style='font-size:18px'><b>{table.iloc[r, c]:.1f}</b></span><br>{meaning[r][c]}" for c in range(3)]
        for r in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=quality,
        x=["Pass", "Review", "Fail"],
        y=["Defective board", "Clean board"],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale=[[0, "#c8e6c9"], [0.5, "#fff3c4"], [1, "#ffcdd2"]],
        showscale=False,
        hoverinfo="skip",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**LAYOUT, title="What happens to 1000 boards")
    return fig


def board_flow(table):
    """Sankey diagram of where 1000 boards end up."""
    d, c = table.loc["defective"], table.loc["clean"]
    fig = go.Figure(go.Sankey(
        node=dict(
            label=["1000 boards", "Defective", "Clean", "Pass", "Review", "Fail"],
            color=["#607d8b", DEFECTIVE, CLEAN, PASS, REVIEW, FAIL],
            pad=20,
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 2, 2, 2],
            target=[1, 2, 3, 4, 5, 3, 4, 5],
            value=[d.sum(), c.sum(), d["pass"], d["review"], d["fail"], c["pass"], c["review"], c["fail"]],
            # red links are costly mistakes, green links are right
            color=[
                "rgba(198,40,40,0.25)", "rgba(21,101,192,0.25)",
                "rgba(198,40,40,0.6)", "rgba(249,168,37,0.5)", "rgba(46,125,50,0.4)",
                "rgba(46,125,50,0.4)", "rgba(249,168,37,0.5)", "rgba(198,40,40,0.6)",
            ],
        ),
    ))
    fig.update_layout(**LAYOUT, title="Board flow")
    return fig


def strategy_costs(strategies, reviewers):
    """Cost per 1000 boards for each strategy."""
    parts = {
        "escaped defects": FAIL,
        "good boards rejected": "#ef9a9a",
        "defects handled": "#90a4ae",
        "good boards shipped": "#cfd8dc",
        "review time": REVIEW,
    }
    names = list(strategies)
    fig = go.Figure()
    for part, color in parts.items():
        if all(strategies[name][part] == 0 for name in names):
            continue
        fig.add_bar(
            y=names,
            x=[strategies[name][part] for name in names],
            name=part,
            orientation="h",
            marker_color=color,
        )
    for name in names:
        total = sum(strategies[name].values())
        fig.add_annotation(
            y=name,
            x=total,
            text=f"  EUR {total:,.0f} | {reviewers[name]:.1f} reviewers",
            showarrow=False,
            xanchor="left",
        )
    fig.update_layout(
        **LAYOUT,
        barmode="stack",
        title="Cost per 1000 boards of each strategy",
        xaxis_title="EUR per 1000 boards",
        legend=dict(orientation="h", y=-0.25),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(sum(s.values()) for s in strategies.values()) * 1.45])
    return fig


def add_zones(fig, low, high):
    """Shades the pass, review and fail zones."""
    fig.add_vrect(x0=0, x1=low, fillcolor=PASS, opacity=0.08, line_width=0)
    fig.add_vrect(x0=low, x1=high, fillcolor=REVIEW, opacity=0.15, line_width=0)
    fig.add_vrect(x0=high, x1=1, fillcolor=FAIL, opacity=0.08, line_width=0)
    for x, text in [(low / 2, "PASS"), ((low + high) / 2, "REVIEW"), ((high + 1) / 2, "FAIL")]:
        fig.add_annotation(x=x, y=1.06, yref="paper", text=text, showarrow=False)


def score_distribution(scores, low, high):
    """Histogram of the scores of defective and clean boards."""
    fig = go.Figure()
    fig.add_histogram(
        x=scores["defect_score"].dropna(),
        name="defective boards",
        marker_color=DEFECTIVE,
        xbins=dict(start=0, end=1, size=0.02),
        histnorm="percent",
        opacity=0.6,
    )
    fig.add_histogram(
        x=scores["clean_score"],
        name="clean boards (false alarm score)",
        marker_color=CLEAN,
        xbins=dict(start=0, end=1, size=0.02),
        histnorm="percent",
        opacity=0.6,
    )
    add_zones(fig, low, high)
    fig.update_layout(
        **LAYOUT,
        barmode="overlay",
        title="Board scores of the test boards",
        xaxis_title="Board score (top box confidence)",
        yaxis_title="Share of boards (%, log scale)",
        yaxis_type="log",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def cost_curve(sweep, chosen, low, high, band_cost):
    """Cost against threshold when the model decides alone."""
    best = sweep.loc[sweep["cost"].idxmin()]
    fig = go.Figure()
    fig.add_scatter(
        x=sweep["threshold"],
        y=sweep["cost"],
        name="model alone",
        line=dict(color="#37474f", width=3),
    )
    fig.add_vline(
        x=chosen,
        line_dash="dash",
        line_color=FAIL,
        annotation_text=f"chosen in training {chosen:.2f}",
    )
    fig.add_scatter(
        x=[best["threshold"]],
        y=[best["cost"]],
        mode="markers",
        name=f"cheapest now {best['threshold']:.2f}",
        marker=dict(size=14, symbol="star", color=PASS),
    )
    fig.add_hline(
        y=band_cost,
        line_dash="dot",
        line_color=REVIEW,
        annotation_text=f"your band {low:.2f} to {high:.2f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        **LAYOUT,
        title="Cost against threshold, model alone",
        xaxis_title="Threshold",
        yaxis_title="EUR per 1000 boards",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def cost_landscape(grid, reviewers, low, high):
    """Cost of every threshold pair. Pairs we can't staff are blank."""
    feasible = grid.assign(cost=grid["cost"].where(grid["reviewers"] <= reviewers))
    table = feasible.pivot(index="high", columns="low", values="cost")
    best = feasible.loc[feasible["cost"].idxmin()]
    fig = go.Figure(go.Heatmap(
        x=table.columns,
        y=table.index,
        z=table.values,
        colorscale="RdYlGn_r",
        zmin=best["cost"],
        zmax=best["cost"] * 3,
        colorbar=dict(title="EUR"),
        hovertemplate="review from %{x}<br>fail from %{y}<br>EUR %{z:,.0f} per 1000<extra></extra>",
    ))
    fig.add_scatter(
        x=[low],
        y=[high],
        mode="markers",
        name="your thresholds",
        marker=dict(size=14, color="white", line=dict(width=3, color="black")),
    )
    fig.add_scatter(
        x=[best["low"]],
        y=[best["high"]],
        mode="markers",
        name="cheapest you can staff",
        marker=dict(size=16, symbol="star", color="#1b5e20"),
    )
    fig.update_layout(
        **LAYOUT,
        title="Cost landscape of the two thresholds",
        xaxis_title="Send to review from",
        yaxis_title="Fail automatically from",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def recall_by_size(defects, threshold):
    """Share of real defects found, by class and size."""
    edges = defects["size"].quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
    # sizes are a share of the 640 pixel image side
    labels = [
        f"small, under {edges[1] * 640:.0f} px",
        f"medium, {edges[1] * 640:.0f} to {edges[2] * 640:.0f} px",
        f"large, over {edges[2] * 640:.0f} px",
    ]
    defects = defects.assign(
        bucket=pd.cut(defects["size"], edges, labels=labels, include_lowest=True).astype(str),
        found=defects["score"] >= threshold,
        missed=defects["score"] < threshold,
    )
    found = defects.pivot_table(index="class", columns="bucket", values="found", aggfunc="mean")[labels]
    missed = defects.pivot_table(index="class", columns="bucket", values="missed", aggfunc="sum")[labels]
    found["all sizes"] = defects.groupby("class")["found"].mean()
    missed["all sizes"] = defects.groupby("class")["missed"].sum()
    text = found.map(lambda v: f"{v:.0%}") + "<br>" + missed.astype(int).astype(str) + " missed"
    fig = go.Figure(go.Heatmap(
        x=found.columns,
        y=found.index,
        z=found.values,
        text=text.values,
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorscale="RdYlGn",
        zmin=min(0.8, np.nanmin(found.values)),
        zmax=1,
        colorbar=dict(title="found", tickformat=".0%"),
        hoverinfo="skip",
    ))
    fig.update_layout(**LAYOUT, title=f"Real defects flagged at your review threshold {threshold:.2f}")
    return fig


def box_confusion(confusion):
    """Confusion matrix of the boxes, as shares of each true class."""
    shares = confusion / confusion.sum(axis=0)
    fig = go.Figure(go.Heatmap(
        x=confusion.columns,
        y=confusion.index,
        z=shares.values,
        text=confusion.values,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        hovertemplate="true %{x}<br>predicted %{y}<br>%{z:.0%} of the column<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        **LAYOUT,
        title="Box confusion matrix at the training threshold",
        xaxis_title="True class",
        yaxis_title="Predicted class",
    )
    return fig


def per_class_bars(per_class):
    """Precision, recall and AP50 for each defect class."""
    fig = go.Figure()
    for metric, color in [("Box-P", CLEAN), ("Box-R", DEFECTIVE), ("mAP50", "#37474f")]:
        fig.add_bar(
            x=per_class["Class"],
            y=per_class[metric],
            name=metric,
            marker_color=color,
            text=per_class[metric].round(3),
            textposition="outside",
        )
    fig.update_layout(
        **LAYOUT,
        title="Per class results on the test split",
        yaxis_range=[0.85, 1.01],
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def training_curves(log):
    """Losses and validation mAP for every epoch."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Losses", "Validation mAP"])
    for column, color in [("box_loss", "#37474f"), ("cls_loss", DEFECTIVE)]:
        fig.add_scatter(x=log["epoch"], y=log[f"train/{column}"], name=f"train {column}", line_color=color, row=1, col=1)
        fig.add_scatter(x=log["epoch"], y=log[f"val/{column}"], name=f"val {column}", line=dict(color=color, dash="dot"), row=1, col=1)
    for column, color in [("mAP50(B)", PASS), ("mAP50-95(B)", CLEAN)]:
        fig.add_scatter(x=log["epoch"], y=log[f"metrics/{column}"], name=column, line_color=color, row=1, col=2)
    fig.update_xaxes(title_text="Epoch")
    fig.update_layout(**LAYOUT, title="Training", legend=dict(orientation="h", y=-0.3))
    return fig
