from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    mean_prediction: float
    observed_rate: float
    count: int


def reliability_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities and outcomes need equal non-zero length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    result: list[ReliabilityBin] = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        rows = [
            (float(probability), bool(outcome))
            for probability, outcome in zip(probabilities, outcomes, strict=True)
            if lower <= probability <= upper
            and (index == bins - 1 or probability < upper)
        ]
        if rows:
            result.append(
                ReliabilityBin(
                    lower=lower,
                    upper=upper,
                    mean_prediction=sum(row[0] for row in rows) / len(rows),
                    observed_rate=sum(row[1] for row in rows) / len(rows),
                    count=len(rows),
                )
            )
    return tuple(result)


def write_reliability_svg(
    path: str | Path,
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    *,
    title: str = "Reliability",
) -> None:
    points = reliability_bins(probabilities, outcomes)
    width, height, margin = 640, 480, 60

    def x(value: float) -> float:
        return margin + value * (width - 2 * margin)

    def y(value: float) -> float:
        return height - margin - value * (height - 2 * margin)

    circles = "\n".join(
        f'<circle cx="{x(point.mean_prediction):.2f}" cy="{y(point.observed_rate):.2f}" '
        f'r="{max(4, min(12, point.count ** 0.5)):.2f}" fill="#2368a2">'
        f"<title>n={point.count}</title></circle>"
        for point in points
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle"
 font-family="sans-serif" font-size="18">{html.escape(title)}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{margin}"
 stroke="#999" stroke-dasharray="5 5"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}"
 y2="{height-margin}" stroke="black"/>
<line x1="{margin}" y1="{height-margin}" x2="{margin}" y2="{margin}"
 stroke="black"/>
<text x="{width / 2}" y="{height-14}" text-anchor="middle"
 font-family="sans-serif">Predicted probability</text>
<text x="18" y="{height / 2}" text-anchor="middle" font-family="sans-serif"
 transform="rotate(-90 18 {height / 2})">Observed success rate</text>
{circles}
</svg>
"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")


def markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])
