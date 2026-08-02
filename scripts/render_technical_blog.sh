#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
source_file="$repository_root/docs/technical_blog.md"
output_file="${1:-$repository_root/output/pdf/model-router-deep-dive.pdf}"

mkdir -p "$(dirname -- "$output_file")"

# Fix human-readable PDF dates; XeLaTeX may still vary the internal trailer ID.
export SOURCE_DATE_EPOCH=1785643200

awk '
  found_byline { print }
  /^\*Pranav Dulepet\*$/ { found_byline = 1; next }
' "$source_file" |
  pandoc \
    --from=markdown-implicit_figures+tex_math_single_backslash \
    --to=pdf \
    --pdf-engine=xelatex \
    --resource-path="$repository_root/docs" \
    --metadata title="Building a model router that knows when to spend" \
    --metadata subtitle="A practical deep dive into model selection, calibration, quality-cost curves, and per-call routing inside coding agents" \
    --metadata author="Pranav Dulepet" \
    -V papersize=letter \
    -V geometry:margin=0.78in \
    -V fontsize=10pt \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    -V toccolor=black \
    -V linestretch=1.04 \
    -o "$output_file"

printf '%s\n' "$output_file"
