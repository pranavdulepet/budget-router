#!/usr/bin/env python3
"""Generate the article's sanitized aggregate data and SVG figures."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/public_router_v1/study/results.json"
CURVES = ROOT / "artifacts/public_router_v1_quality_cost_curves.json"
AGENT = ROOT / "artifacts/agent_step_router_v2_results.json"
ASSETS = ROOT / "docs/assets"

INK = "#172033"
MUTED = "#5f6b7a"
GRID = "#d9dee8"
PANEL = "#f6f8fb"
BLUE = "#2563eb"
TEAL = "#0f9f8f"
ORANGE = "#e17b21"
PURPLE = "#7c4dff"
GREEN = "#178a55"
RED = "#c43d4b"


def metric(value: dict[str, Any]) -> dict[str, float]:
    row = value.get("metrics", value)
    return {
        "score": float(row["macro_score"]),
        "cost_usd": float(row["total_cost_usd"]),
    }


def export_curves() -> dict[str, Any]:
    if STUDY.exists():
        study = json.loads(STUDY.read_text(encoding="utf-8"))
        test = study["test"]["id_test"]
        controls = test["controls"]
        data = {
            "schema_version": "public-router-quality-cost-curves-v1",
            "source": "outputs/public_router_v1/study/results.json",
            "prompt_count": 2184,
            "curves": {
                method: [
                    {
                        "lambda": float(point["lambda"]),
                        **metric(point),
                    }
                    for point in points
                ]
                for method, points in test["learned_cost_quality_curves"].items()
            },
            "controls": {
                "gpt_5": metric(controls["fixed_each"]["gpt-5"]),
                "cheapest_fixed": {
                    "model": controls["cheapest_fixed"]["model"],
                    **metric(controls["cheapest_fixed"]),
                },
                "training_dataset_lookup": metric(
                    controls["training_dataset_lookup"]
                ),
                "hindsight_oracle": metric(controls["hindsight_oracle"]),
                "confirmatory_primary": {
                    "method": test["primary"]["method"],
                    "lambda": float(test["primary"]["lambda"]),
                    **metric(test["primary"]),
                },
                "published_balanced_artifact": {
                    "method": test["descriptive_test_matched_quality_best"]["method"],
                    "lambda": float(
                        test["descriptive_test_matched_quality_best"]["lambda"]
                    ),
                    **metric(test["descriptive_test_matched_quality_best"]),
                },
            },
        }
        CURVES.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return data
    return json.loads(CURVES.read_text(encoding="utf-8"))


def svg_document(title: str, description: str, body: str, *, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">
  <title id="title">{escape(title)}</title>
  <desc id="desc">{escape(description)}</desc>
  <rect width="1120" height="{height}" fill="#ffffff"/>
  <style>
    text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: {INK}; }}
    .label {{ font-size: 18px; font-weight: 600; }}
    .small {{ font-size: 15px; fill: {MUTED}; }}
    .tick {{ font-size: 14px; fill: {MUTED}; }}
    .number {{ font-size: 24px; font-weight: 700; }}
  </style>
{body}
</svg>
"""


def box(x: int, y: int, width: int, height: int, title: str, lines: list[str]) -> str:
    text = [
        f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="{PANEL}" stroke="{GRID}"/>',
        f'  <text x="{x + 22}" y="{y + 34}" class="label">{escape(title)}</text>',
    ]
    for index, line in enumerate(lines):
        text.append(
            f'  <text x="{x + 22}" y="{y + 62 + index * 22}" class="small">{escape(line)}</text>'
        )
    return "\n".join(text)


def architecture() -> str:
    arrow = (
        f'<path d="M {{x1}} {{y}} H {{x2}}" stroke="{MUTED}" stroke-width="2" '
        f'marker-end="url(#arrow)" fill="none"/>'
    )
    body = f"""
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{MUTED}"/>
    </marker>
  </defs>
  <text x="40" y="45" class="label">Offline: learn one immutable routing artifact</text>
{box(40, 70, 220, 112, "Outcome matrix", ["Requests × candidate models", "Train and calibration splits"])}
{arrow.format(x1=260, x2=302, y=126)}
{box(302, 70, 220, 112, "Classifier", ["P(success | request, model)", "Shared features + interactions"])}
{arrow.format(x1=522, x2=564, y=126)}
{box(564, 70, 220, 112, "Calibration", ["Per-model probabilities", "Non-inferiority gate"])}
{arrow.format(x1=784, x2=826, y=126)}
{box(826, 70, 254, 112, "Versioned artifact", ["Weights, calibrators, costs", "Cards, policy, content hash"])}

  <line x1="40" y1="225" x2="1080" y2="225" stroke="{GRID}"/>
  <text x="40" y="270" class="label">Online: select before inference</text>
{box(40, 295, 190, 122, "Visible state", ["Prompt or agent prefix", "Token estimate"])}
{arrow.format(x1=230, x2=270, y=356)}
{box(270, 295, 190, 122, "Hard filters", ["Capability and context", "Provider and cost limits"])}
{arrow.format(x1=460, x2=500, y=356)}
{box(500, 295, 200, 122, "Success surface", ["Calibrated probability", "for each eligible model"])}
{arrow.format(x1=700, x2=740, y=356)}
{box(740, 295, 160, 122, "Policy", ["Quality, balanced,", "or cost mode"])}
{arrow.format(x1=900, x2=940, y=356)}
{box(940, 295, 140, 122, "Selection", ["Model ID", "No API call"])}

  <path d="M1010 417 V474 H760" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)" fill="none"/>
  <rect x="430" y="448" width="330" height="68" rx="14" fill="#fff7ed" stroke="#fed7aa"/>
  <text x="452" y="478" class="label">Provider/runtime layer</text>
  <text x="452" y="501" class="small">Endpoint, health, latency, retry, session cache</text>
"""
    return svg_document(
        "Budget Router architecture",
        "Offline training produces a calibrated artifact. Online routing applies hard constraints, predicts success per model, and selects a model. Provider execution remains separate.",
        body,
        height=550,
    )


def public_curve(data: dict[str, Any]) -> str:
    left, top, width, height = 92, 55, 760, 500
    x_min, x_max = 0.0, 72.0
    y_min, y_max = 0.50, 0.90

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * width

    def sy(value: float) -> float:
        return top + height - (value - y_min) / (y_max - y_min) * height

    parts: list[str] = []
    for value in range(0, 73, 10):
        x = sx(float(value))
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + height + 28}" text-anchor="middle" class="tick">${value}</text>'
        )
    for value in (0.50, 0.60, 0.70, 0.80, 0.90):
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.0%}</text>'
        )
    names = {
        "shared_task_model_hashed_classifier": ("Shared classifier", BLUE),
        "independent_hashed_logistic": ("Independent heads", TEAL),
        "calibrated_gradient_boosting": ("Gradient boosting", ORANGE),
        "text_knn": ("Text kNN", PURPLE),
    }
    for method, (label, color) in names.items():
        points = sorted(data["curves"][method], key=lambda row: row["cost_usd"])
        coords = " ".join(
            f'{sx(point["cost_usd"]):.1f},{sy(point["score"]):.1f}'
            for point in points
        )
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        parts.extend(
            f'<circle cx="{sx(point["cost_usd"]):.1f}" cy="{sy(point["score"]):.1f}" r="4.5" fill="{color}"/>'
            for point in points
        )
    controls = data["controls"]
    for key, label, color, shape in (
        ("gpt_5", "Always GPT-5", INK, "circle"),
        ("hindsight_oracle", "Hindsight oracle", GREEN, "diamond"),
    ):
        row = controls[key]
        x, y = sx(row["cost_usd"]), sy(row["score"])
        if shape == "diamond":
            parts.append(
                f'<path d="M{x:.1f},{y-8:.1f} L{x+8:.1f},{y:.1f} L{x:.1f},{y+8:.1f} L{x-8:.1f},{y:.1f} Z" fill="{color}"/>'
            )
        else:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/>'
            )
        dx = -14 if key == "gpt_5" else 12
        anchor = "end" if key == "gpt_5" else "start"
        parts.append(
            f'<text x="{x + dx:.1f}" y="{y - 18:.1f}" text-anchor="{anchor}" class="small">{label}</text>'
        )
    balanced = controls["published_balanced_artifact"]
    bx, by = sx(balanced["cost_usd"]), sy(balanced["score"])
    parts.append(
        f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="10" fill="none" stroke="{BLUE}" stroke-width="3"/>'
    )
    parts.append(
        f'<path d="M{bx - 7:.1f},{by - 7:.1f} L{bx - 18:.1f},{by - 25:.1f}" stroke="{MUTED}" fill="none"/>'
    )
    parts.append(
        f'<text x="{bx - 22:.1f}" y="{by - 29:.1f}" text-anchor="end" class="small">published λ=0.10</text>'
    )
    primary = controls["confirmatory_primary"]
    px, py = sx(primary["cost_usd"]), sy(primary["score"])
    parts.append(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="#ffffff" stroke="{RED}" stroke-width="3"/>'
    )
    parts.append(
        f'<text x="{px + 14:.1f}" y="{py + 5:.1f}" class="small">confirmatory primary</text>'
    )
    legend_y = 110
    for index, (_, (label, color)) in enumerate(names.items()):
        y = legend_y + index * 38
        parts.append(
            f'<line x1="890" y1="{y}" x2="930" y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(f'<text x="944" y="{y + 5}" class="small">{label}</text>')
    parts.extend(
        [
            f'<text x="{left + width / 2}" y="625" text-anchor="middle" class="label">Observed cost across 2,184 prompts</text>',
            f'<text x="24" y="{top + height / 2}" transform="rotate(-90 24 {top + height / 2})" text-anchor="middle" class="label">Macro-average score</text>',
            '<text x="890" y="340" class="small">Each line varies the frozen</text>',
            '<text x="890" y="362" class="small">cost penalty λ. Lower cost</text>',
            '<text x="890" y="384" class="small">moves left.</text>',
            '<text x="890" y="438" class="small">The oracle sees terminal</text>',
            '<text x="890" y="460" class="small">outcomes and is not deployable.</text>',
        ]
    )
    return svg_document(
        "Prompt routing quality-cost curves",
        "Four learned routing methods trace quality-cost curves on 2,184 held-out prompts. The published balanced shared classifier is highlighted, along with the failed confirmatory primary, GPT-5, and a non-deployable hindsight oracle.",
        "\n  ".join(parts),
        height=660,
    )


def agent_plot(agent: dict[str, Any]) -> str:
    policies = agent["full_60_task_analysis"]["policies"]
    rows = [
        (
            "Fixed GPT-OSS 20B",
            policies["fixed:openai-gpt-oss-20b"],
            ORANGE,
        ),
        ("Fixed Qwen3.6 35B", policies["fixed:qwen3.6-35b"], BLUE),
        (
            "Guarded router",
            policies["router:guarded-live-calibrated-v2"],
            GREEN,
        ),
    ]
    left, top, width, height = 105, 55, 820, 465

    def sx(value: float) -> float:
        return left + value / 35.0 * width

    def sy(value: float) -> float:
        return top + height - value / 30.0 * height

    parts: list[str] = []
    for value in range(0, 36, 5):
        x = sx(float(value))
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + height + 28}" text-anchor="middle" class="tick">${value}</text>'
        )
    for value in range(0, 31, 5):
        y = sy(float(value))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value}</text>'
        )
    offsets = {
        "Fixed GPT-OSS 20B": (14, 27),
        "Fixed Qwen3.6 35B": (-16, 28),
        "Guarded router": (-14, -20),
    }
    for label, row, color in rows:
        cost = float(row["total_conservative_cost_usd"])
        resolved = int(row["resolved_count"])
        x, y = sx(cost), sy(float(resolved))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{color}" stroke="#ffffff" stroke-width="3"/>'
        )
        dx, dy = offsets[label]
        anchor = "end" if dx < 0 else "start"
        parts.append(
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" class="label">{label}</text>'
        )
        parts.append(
            f'<text x="{x + dx:.1f}" y="{y + dy + 22:.1f}" text-anchor="{anchor}" class="small">{resolved}/60 · ${cost:.2f}</text>'
        )
    comparison = agent["full_60_task_analysis"]["router_vs_fixed_strong"]
    parts.extend(
        [
            f'<text x="{left + width / 2}" y="590" text-anchor="middle" class="label">Conservative cost across 60 tasks</text>',
            f'<text x="25" y="{top + height / 2}" transform="rotate(-90 25 {top + height / 2})" text-anchor="middle" class="label">Officially resolved SWE-bench tasks</text>',
            f'<rect x="955" y="90" width="140" height="248" rx="14" fill="{PANEL}" stroke="{GRID}"/>',
            '<text x="975" y="124" class="label">Router vs Qwen</text>',
            '<text x="975" y="168" class="number">20.79%</text>',
            '<text x="975" y="191" class="small">lower cost</text>',
            '<text x="975" y="235" class="small">Quality difference</text>',
            '<text x="975" y="257" class="small">95% interval:</text>',
            '<text x="975" y="281" class="label">−6.67 to +13.33</text>',
            '<text x="975" y="303" class="small">percentage points</text>',
            '<text x="955" y="390" class="small">Observed quality:</text>',
            '<text x="955" y="412" class="small">26 vs 24. The interval</text>',
            '<text x="955" y="434" class="small">does not establish</text>',
            '<text x="955" y="456" class="small">quality superiority.</text>',
        ]
    )
    assert abs(float(comparison["routed_cost_saving_fraction"]) - 0.20790118867960394) < 1e-12
    return svg_document(
        "Agent routing quality and cost",
        "On 60 held-out SWE-bench tasks, fixed GPT-OSS resolved 13 at 9.53 dollars, fixed Qwen resolved 24 at 30.98 dollars, and the guarded router resolved 26 at 24.54 dollars.",
        "\n  ".join(parts),
        height=625,
    )


def decision_funnel(agent: dict[str, Any]) -> str:
    reasons = agent["active_router_audit"]["decision_reasons"]
    rows = [
        ("Classifier selected Qwen", reasons["classifier_strong"], BLUE),
        ("Initial-call guard forced Qwen", reasons["force_strong_initial_calls"], MUTED),
        ("Context guard forced Qwen", reasons["force_strong_context_guard"], MUTED),
        ("Dwell guard forced Qwen", reasons["force_strong_dwell"], MUTED),
        ("Cheap-burst guard forced Qwen", reasons["force_strong_cheap_burst"], MUTED),
        ("Classifier selected GPT-OSS", reasons["classifier_cheap"], GREEN),
    ]
    max_value = max(value for _, value, _ in rows)
    parts: list[str] = [
        '<text x="40" y="44" class="small">Every visible prefix produced one auditable reason.</text>'
    ]
    for index, (label, value, color) in enumerate(rows):
        y = 82 + index * 68
        width = 640 * value / max_value
        parts.extend(
            [
                f'<text x="40" y="{y + 18}" class="label">{label}</text>',
                f'<rect x="390" y="{y}" width="640" height="28" rx="6" fill="{GRID}"/>',
                f'<rect x="390" y="{y}" width="{width:.1f}" height="28" rx="6" fill="{color}"/>',
                f'<text x="1045" y="{y + 20}" class="label">{value:,}</text>',
            ]
        )
    parts.extend(
        [
            '<line x1="40" y1="505" x2="1080" y2="505" stroke="#d9dee8"/>',
            '<text x="40" y="548" class="number">2,866 calls</text>',
            '<text x="240" y="548" class="number">364 cheap calls</text>',
            '<text x="515" y="548" class="number">56/60 trajectories</text>',
            '<text x="805" y="548" class="number">400 switches</text>',
            '<text x="40" y="575" class="small">total</text>',
            '<text x="240" y="575" class="small">12.70% of all calls</text>',
            '<text x="515" y="575" class="small">used GPT-OSS at least once</text>',
            '<text x="805" y="575" class="small">between model tiers</text>',
        ]
    )
    return svg_document(
        "Guarded agent router decisions",
        "A bar chart partitions 2,866 agent calls by the classifier or guard reason that selected Qwen or GPT-OSS.",
        "\n  ".join(parts),
        height=610,
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    curves = export_curves()
    agent = json.loads(AGENT.read_text(encoding="utf-8"))
    files = {
        "router-architecture.svg": architecture(),
        "public-quality-cost.svg": public_curve(curves),
        "agent-quality-cost.svg": agent_plot(agent),
        "agent-decision-audit.svg": decision_funnel(agent),
    }
    for name, content in files.items():
        (ASSETS / name).write_text(content, encoding="utf-8")
    print("\n".join(str(ASSETS / name) for name in files))


if __name__ == "__main__":
    main()
