#!/usr/bin/env bash
# Rebuild every figure in BOTH styles, in the pinned environment.
#
# This exists because "remember to also run --style science" is a rule that gets
# missed, and when it is missed the repo keeps a paper-style figure showing new
# numbers next to a science-style figure showing old ones. One command, no
# half-updates. Run it after touching any results/*.csv or plot_*.py.
#
#   ./build_figures.sh            # everything, both styles
#   ./build_figures.sh probing    # just plot_probing.py, both styles
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME=vpro-plots
# An already-activated vpro-plots wins; otherwise find it by conda's env dir, so
# this works whether or not the caller activated anything.
if [[ "${CONDA_PREFIX:-}" == *"/$ENV_NAME" ]]; then
  PY="$CONDA_PREFIX/bin/python"
elif PY_BASE=$(conda info --base 2>/dev/null) && [[ -x "$PY_BASE/envs/$ENV_NAME/bin/python" ]]; then
  PY="$PY_BASE/envs/$ENV_NAME/bin/python"
else
  echo "error: environment '$ENV_NAME' not found." >&2
  echo "  conda env create -f environment.yml && conda activate $ENV_NAME" >&2
  exit 1
fi

# Fail before writing anything rather than after half the figures are rebuilt:
# a missing SciencePlots would otherwise leave every _science.pdf stale, which
# is the exact failure this script exists to prevent.
"$PY" - <<'PYCHECK'
import sys
missing = []
for mod in ("matplotlib", "pandas", "numpy", "scienceplots"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit("error: environment is incomplete, missing: " + ", ".join(missing))
PYCHECK

if [[ $# -gt 0 ]]; then
  SCRIPTS=()
  for n in "$@"; do SCRIPTS+=("plot_${n#plot_}"); done
  SCRIPTS=("${SCRIPTS[@]/%/.py}")
  SCRIPTS=("${SCRIPTS[@]/%.py.py/.py}")
else
  SCRIPTS=(plot_probing.py plot_realworld.py plot_realworld_bars.py
           plot_realworld_multiview_vs_side.py
           plot_libero_radar.py plot_libero_plus_radar.py plot_radar_row.py
           plot_realworld_row.py)
fi

# Scripts that produce only the science variant and take no --style switch. The
# combined row figure is built to be pasted into the paper, which uses that
# style; a paper-style twin nobody includes is one more thing to keep in sync.
SCIENCE_ONLY=(plot_realworld_row.py)

for s in "${SCRIPTS[@]}"; do
  science_only=false
  for so in "${SCIENCE_ONLY[@]}"; do
    [[ "$s" == "$so" ]] && science_only=true
  done

  if $science_only; then
    echo "=== $s (science only)"
    "$PY" "$s"
    continue
  fi

  for style in paper science; do
    echo "=== $s --style $style"
    "$PY" "$s" --style "$style"
  done
done

echo
echo "done. Both styles rebuilt for: ${SCRIPTS[*]}"
