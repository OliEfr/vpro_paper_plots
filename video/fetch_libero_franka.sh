#!/bin/bash
# Pull the LIBERO Franka rollout renders for slide 1 from MN5 (via a login or transfer node) once
# GPFS is reachable again, then rebuild slide 1. Run from vpro_paper_plots/video/.
#
#   ./fetch_libero_franka.sh [host]      # default host: transfer1.bsc.es (alias in ~/.ssh/config)
#
# Source: the paper's dual-view "ours" arm (policy 43582857, ckpt 070000), render5 group.
# recorded_successes.json says libero_goal task 1 ("put the bowl on the stove") episodes 0-4 all
# succeeded, so the first four rendered episodes are taken.
set -euo pipefail
HOST=${1:-transfer1.bsc.es}
G=/gpfs/projects/ehpc637/felix_minzenmay2/runs_root/runs_lerobot/outputs/eval_split/2026-07-21_rollout_sqsh_43582857_rlfvxemb_withvideo_lam1_policy5_nact10_ep100_bs25_async_osmesa_render5_ckpt070000
OUT=clips/libero/franka
mkdir -p "$OUT"
# The per-task run dir name follows the worker's RUN_NAME pattern (…_libero_goal_t1_…); list, then copy.
RUNDIR=$(ssh "$HOST" "ls -d $G/*libero_goal_t1* | head -1")
echo "run dir: $RUNDIR"
ssh "$HOST" "find $RUNDIR -name '*.mp4' | sort" | tee "$OUT/_remote_files.txt"
for i in 0 1 2 3; do
  f=$(grep -E "eval_episode_${i}\.mp4$" "$OUT/_remote_files.txt" | head -1)
  [ -n "$f" ] && scp "$HOST:$f" "$OUT/goal_t1_ep${i}.mp4"
done
ls -la "$OUT"
echo "now: replace the four placeholder entries in sources.yaml slide 1 with"
echo "  {clip: video/$OUT/goal_t1_ep<i>.mp4}   and run:  python3 compose.py 1"
