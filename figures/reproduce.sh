#!/bin/bash
# Regenerate every figure and table from the experiments' results (style shared via figures/style.py).
set -e
cd "$(dirname "$0")/.."
RUN="uv run --no-project --with pyyaml --with matplotlib python"
for f in reasoning_modes method_schematic theory_frontier method_comparison controls; do
  $RUN figures/$f/plot.py
done
$RUN figures/tables/make_tables.py
$RUN figures/tasks/make_examples.py
