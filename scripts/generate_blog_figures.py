#!/usr/bin/env python3
"""Generate the article's publication figures from sanitized artifacts.

Statistical plots use Matplotlib and Seaborn. System diagrams use Graphviz.
The renderer reads committed evidence and never rewrites scientific results.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "artifacts/public_router_v1_quality_cost_curves.json"
AGENT = ROOT / "artifacts/agent_step_router_v2_results.json"
ASSETS = ROOT / "docs/assets"
RASTER_ASSETS = ASSETS / "medium"
DIAGRAM_SOURCES = ASSETS / "source"

INK = "#172033"
MUTED = "#5f6b7a"
GRID = "#d9dee8"
PANEL = "#f6f8fb"
BLUE = "#2563eb"
TEAL = "#0f8f83"
ORANGE = "#d97706"
PURPLE = "#7656b3"
GREEN = "#178a55"
RED = "#c43d4b"


def load_curves() -> dict[str, Any]:
    data = json.loads(CURVES.read_text(encoding="utf-8"))
    expected_methods = {
        "shared_task_model_hashed_classifier",
        "independent_hashed_logistic",
        "calibrated_gradient_boosting",
        "text_knn",
    }
    if set(data["curves"]) != expected_methods:
        raise ValueError("unexpected prompt-routing curve methods")
    if any(len(points) != 10 for points in data["curves"].values()):
        raise ValueError("every prompt-routing method must have ten frozen points")

    balanced = data["controls"]["published_balanced_artifact"]
    primary = data["controls"]["confirmatory_primary"]
    if (
        balanced["method"] != "shared_task_model_hashed_classifier"
        or float(balanced["lambda"]) != 0.10
    ):
        raise ValueError("published balanced point does not match the frozen artifact")
    if primary["method"] != "text_knn" or float(primary["lambda"]) != 0.20:
        raise ValueError("predeclared primary point does not match the frozen study")
    return data


def load_agent() -> dict[str, Any]:
    data = json.loads(AGENT.read_text(encoding="utf-8"))
    audit = data["active_router_audit"]
    reasons = audit["decision_reasons"]
    router = data["full_60_task_analysis"]["policies"][
        "router:guarded-live-calibrated-v2"
    ]
    if sum(reasons.values()) != audit["total_calls"]:
        raise ValueError("decision-reason counts do not sum to routed calls")
    if reasons["classifier_cheap"] != router["cheap_calls"]:
        raise ValueError("classifier cheap decisions do not match routed cheap calls")
    if audit["total_calls"] - reasons["classifier_cheap"] != router["strong_calls"]:
        raise ValueError("strong decisions do not match routed strong calls")
    return data


def configure_style() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        },
    )
    matplotlib.rcParams["svg.hashsalt"] = "budget-router-figures-v1"
    matplotlib.rcParams["svg.fonttype"] = "none"


def normalize_svg(path: Path) -> None:
    """Remove generator whitespace that fails repository diff checks."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def save_figure(figure: plt.Figure, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    RASTER_ASSETS.mkdir(parents=True, exist_ok=True)
    svg_path = ASSETS / f"{stem}.svg"
    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Matplotlib", "Date": None},
    )
    figure.savefig(
        RASTER_ASSETS / f"{stem}.png",
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Matplotlib"},
    )
    normalize_svg(svg_path)
    plt.close(figure)


def render_graphviz(source: Path, stem: str) -> None:
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError("Graphviz 'dot' is required to render article diagrams")
    graph = source.read_text(encoding="utf-8")
    ASSETS.mkdir(parents=True, exist_ok=True)
    RASTER_ASSETS.mkdir(parents=True, exist_ok=True)
    for output, output_format in (
        (ASSETS / f"{stem}.svg", "svg"),
        (RASTER_ASSETS / f"{stem}.png", "png"),
    ):
        subprocess.run(
            [dot, f"-T{output_format}", "-o", str(output)],
            input=graph,
            text=True,
            check=True,
        )
        if output_format == "svg":
            normalize_svg(output)


def render_diagrams() -> None:
    render_graphviz(DIAGRAM_SOURCES / "router-architecture.dot", "router-architecture")
    render_graphviz(DIAGRAM_SOURCES / "agent-state-loop.dot", "agent-state-loop")


def public_quality_cost(data: dict[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    methods = {
        "shared_task_model_hashed_classifier": ("Shared classifier", BLUE),
        "independent_hashed_logistic": ("Independent heads", TEAL),
        "calibrated_gradient_boosting": ("Gradient boosting", ORANGE),
        "text_knn": ("Text kNN", PURPLE),
    }
    for method, (label, color) in methods.items():
        points = sorted(data["curves"][method], key=lambda row: row["cost_usd"])
        axis.plot(
            [point["cost_usd"] for point in points],
            [point["score"] for point in points],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=label,
            color=color,
        )

    controls = data["controls"]
    gpt = controls["gpt_5"]
    oracle = controls["hindsight_oracle"]
    balanced = controls["published_balanced_artifact"]
    primary = controls["confirmatory_primary"]
    axis.scatter(
        gpt["cost_usd"],
        gpt["score"],
        s=70,
        color=INK,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="Always GPT-5",
    )
    axis.scatter(
        oracle["cost_usd"],
        oracle["score"],
        s=90,
        marker="D",
        color=GREEN,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label="Hindsight oracle",
    )
    axis.scatter(
        balanced["cost_usd"],
        balanced["score"],
        s=150,
        facecolor="none",
        edgecolor=BLUE,
        linewidth=2.2,
        zorder=6,
    )
    axis.scatter(
        primary["cost_usd"],
        primary["score"],
        s=150,
        facecolor="white",
        edgecolor=RED,
        linewidth=2.2,
        zorder=6,
    )
    axis.annotate(
        "secondary balanced point",
        (balanced["cost_usd"], balanced["score"]),
        xytext=(-12, 16),
        textcoords="offset points",
        ha="right",
        color=BLUE,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 0.8},
    )
    axis.annotate(
        "predeclared text-kNN point",
        (primary["cost_usd"], primary["score"]),
        xytext=(10, -18),
        textcoords="offset points",
        color=RED,
        fontsize=9,
    )
    axis.annotate(
        "oracle sees terminal outcomes",
        (oracle["cost_usd"], oracle["score"]),
        xytext=(15, -24),
        textcoords="offset points",
        color=MUTED,
        fontsize=8.5,
    )
    axis.set(
        title="Held-out prompt routing: discrete quality-cost operating points",
        xlabel="Observed cost across 2,184 prompts (USD)",
        ylabel="Macro-average score",
        xlim=(0, 72),
        ylim=(0.50, 0.90),
    )
    axis.text(
        0.01,
        0.02,
        "Y-axis shown from 50% to 90%. Each dot is one frozen λ.",
        transform=axis.transAxes,
        fontsize=8.5,
        color=MUTED,
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="lower right", ncol=2, fontsize=8.5)
    sns.despine(ax=axis)
    save_figure(figure, "public-quality-cost")


def agent_quality_cost(agent: dict[str, Any]) -> None:
    policies = agent["full_60_task_analysis"]["policies"]
    rows = [
        (
            "Fixed GPT-OSS 20B",
            policies["fixed:openai-gpt-oss-20b"],
            ORANGE,
            (10, 9),
            "left",
            "bottom",
        ),
        (
            "Fixed Qwen3.6 35B",
            policies["fixed:qwen3.6-35b"],
            BLUE,
            (-10, -10),
            "right",
            "top",
        ),
        (
            "Guarded router",
            policies["router:guarded-live-calibrated-v2"],
            GREEN,
            (-10, 10),
            "right",
            "bottom",
        ),
    ]
    comparison = agent["full_60_task_analysis"]["router_vs_fixed_strong"]
    figure, axis = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    for label, row, color, offset, horizontal, vertical in rows:
        cost = Decimal(str(row["total_conservative_cost_usd"]))
        resolved = int(row["resolved_count"])
        axis.scatter(
            float(cost),
            resolved,
            s=145,
            color=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        axis.annotate(
            f"{label}\n{resolved}/60 · ${cost:.2f}",
            (float(cost), resolved),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=9.5,
            fontweight="bold",
        )

    quality_low, quality_high = comparison["paired_task_bootstrap"][
        "quality_difference_95ci"
    ]
    cost_low, cost_high = comparison["paired_task_bootstrap"][
        "cost_saving_fraction_95ci"
    ]
    note = (
        "Router vs fixed Qwen\n"
        f"observed cost: {comparison['routed_cost_saving_fraction']:.1%} lower\n"
        f"cost-saving 95% CI: {cost_low:.1%} to {cost_high:.1%}\n"
        "quality difference 95% CI\n"
        f"{quality_low:+.2%} to {quality_high:+.2%}"
    )
    axis.text(
        0.98,
        0.05,
        note,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": PANEL,
            "edgecolor": GRID,
        },
    )
    axis.set(
        title="Held-out coding agent: official resolutions versus cost",
        xlabel="Conservative cost across 60 tasks (USD)",
        ylabel="Officially resolved SWE-bench tasks",
        xlim=(0, 35),
        ylim=(0, 30),
    )
    sns.despine(ax=axis)
    save_figure(figure, "agent-quality-cost")


def decision_audit(agent: dict[str, Any]) -> None:
    audit = agent["active_router_audit"]
    reasons = audit["decision_reasons"]
    labels = [
        "Classifier selected Qwen",
        "Context guard forced Qwen",
        "Cheap-burst guard forced Qwen",
        "Initial-call guard forced Qwen",
        "Dwell guard forced Qwen",
        "Classifier selected GPT-OSS",
    ]
    values = [
        reasons["classifier_strong"],
        reasons["force_strong_context_guard"],
        reasons["force_strong_cheap_burst"],
        reasons["force_strong_initial_calls"],
        reasons["force_strong_dwell"],
        reasons["classifier_cheap"],
    ]
    colors = [BLUE, MUTED, MUTED, MUTED, MUTED, GREEN]
    total_calls = int(audit["total_calls"])

    figure, axis = plt.subplots(figsize=(9.2, 5.25), constrained_layout=True)
    bars = axis.barh(labels, values, color=colors, height=0.62)
    axis.invert_yaxis()
    axis.bar_label(
        bars,
        labels=[f"{value:,} ({value / total_calls:.1%})" for value in values],
        padding=6,
    )
    axis.set(
        title="Activation audit: one recorded reason for every model call",
        xlabel="Calls",
        ylabel="",
        xlim=(0, max(values) * 1.19),
    )
    axis.xaxis.grid(True)
    axis.yaxis.grid(False)
    axis.text(
        0.0,
        -0.21,
        (
            f"{total_calls:,} calls  ·  {reasons['classifier_cheap']:,} GPT-OSS  ·  "
            f"{total_calls - reasons['classifier_cheap']:,} Qwen  ·  "
            f"{audit['trajectories_with_cheap_call']}/{audit['trajectories']} "
            "trajectories used GPT-OSS\n"
            "400 switches  ·  60/60 trajectories satisfied every guard"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )
    sns.despine(ax=axis, left=True)
    save_figure(figure, "agent-decision-audit")


def main() -> None:
    configure_style()
    curves = load_curves()
    agent = load_agent()
    render_diagrams()
    public_quality_cost(curves)
    agent_quality_cost(agent)
    decision_audit(agent)
    for name in (
        "router-architecture",
        "agent-state-loop",
        "public-quality-cost",
        "agent-quality-cost",
        "agent-decision-audit",
    ):
        print(ASSETS / f"{name}.svg")
        print(RASTER_ASSETS / f"{name}.png")


if __name__ == "__main__":
    main()
