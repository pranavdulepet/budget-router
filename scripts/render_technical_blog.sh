#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_file="$repository_root/docs/technical_blog.md"
output_file="${1:-$repository_root/output/pdf/model-router-deep-dive.pdf}"

mkdir -p "$(dirname -- "$output_file")"

# Fix human-readable PDF dates; XeLaTeX may still vary the internal trailer ID.
export SOURCE_DATE_EPOCH=1785643200

sed '1,8d' "$source_file" |
  pandoc \
    --from=markdown+tex_math_single_backslash \
    --to=pdf \
    --pdf-engine=xelatex \
    --resource-path="$repository_root/docs" \
    --metadata title="What I learned building a model router for coding agents" \
    --metadata subtitle="A technical guide to model selection, quality-cost curves, calibration, and per-call agent routing—backed by an open-source implementation and held-out SWE-bench evaluation" \
    --metadata author="Pranav Dulepet" \
    --toc \
    --toc-depth=2 \
    --number-sections \
    --shift-heading-level-by=-1 \
    -V papersize=letter \
    -V geometry:margin=0.78in \
    -V fontsize=10pt \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V toccolor=black \
    -V linestretch=1.08 \
    -o "$output_file"

printf '%s\n' "$output_file"
