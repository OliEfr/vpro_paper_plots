#!/usr/bin/env python
"""Write results/probe_perdim_realworld.csv from Felix's DK1 probe outputs on MN5.

Source directories (read-only, Felix's account `tum205310`, see the brain page
`experiments-account` / `dk1-p19-sweep-runs-20260719`):

  /gpfs/scratch/ehpc637/felix_minzenmay2/dk1_postprocessed_sweep_run_20260719/latent_analysis/
      sharedlam2full_dino768d6_ckpt030000_idx1_robot3cam_h5_probe      multi-view (front+side), LAM 43524549
      sharedlam3sidefull_dino768d6_ckpt030000_idx1_robot3cam_h5_probe  single-view (side only), LAM 44160743
      sharedlam4full_dino768d6_ckpt030000_idx1_robot3cam_h5_probe      multi-view on the NEW-TASK dataset, LAM 44227723

Each holds `action_probe_r2.csv` with rows (probe_model, target, action_dim, r2,
avg_r2_for_target, n_train, n_test, ...), produced by the standard probe CLI
(`--probe-model both --probe-split episode --future-frames 5 --probe-max-samples 0`)
on the robot_3cam episodes only (the human video has no actions). The first two were
scored on identical rows (119,983 train / 36,067 test), so they are directly comparable;
sharedlam4 was trained and probed on a different dataset and is kept for reference only.

Mirror the three directories to <mirror>/ and run:
    python experiments/extract_probe_realworld.py --mirror <mirror>

The cluster probes target the stored DK1 action, which is an ABSOLUTE end-effector pose
target (~ the measured pose 5 frames ahead), so they score pose decodability. `--local`
adds our own fit with the motion target state[t+5]-state[t] over the LAM horizon, the
real-world analogue of the per-step delta actions in the simulation suites.

Action dims are the DK1 end-effector action ['ee.x','ee.y','ee.z','ee.wx','ee.wy','ee.wz',
'ee.gripper_pos'] (dataset meta/info.json), named dx..gripper here to match probe_perdim.csv.
No axis relabelling: both LAMs see the same cameras of the same robot.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "probe_perdim_realworld.csv"
DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

ARMS = {
    "ours_multi": ("sharedlam2full_dino768d6_ckpt030000_idx1_robot3cam_h5_probe", "43524549",
                   "front+side (multi-view) DINO768d6 LAM, 30k, sharedlam2"),
    "ours_single": ("sharedlam3sidefull_dino768d6_ckpt030000_idx1_robot3cam_h5_probe", "44160743",
                    "side-only (single-view) DINO768d6 LAM, 30k, sharedlam3"),
    "ours_multi_newtasks": ("sharedlam4full_dino768d6_ckpt030000_idx1_robot3cam_h5_probe", "44227723",
                            "front+side LAM on the new-task dataset, 30k, sharedlam4 (different data; not matched)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", type=Path, required=True)
    ap.add_argument("--local", type=Path, default=None,
                    help="probe_realworld_local.csv from fit_decode_realworld.py: adds target "
                         "'state_delta_h5' (EE motion state[t+5]-state[t]) and 'abs_action_local' rows")
    a = ap.parse_args()
    rows = []
    for method, (d, job, note) in ARMS.items():
        r = pd.read_csv(a.mirror / d / "action_probe_r2.csv")
        man = json.load(open(a.mirror / d / "analysis_manifest.json"))
        assert man["headline_metrics"]["probe_split"] == "episode"
        base = dict(benchmark="realworld", method=method, teacher_job=job, ckpt_step=30000, source_dir=d, note=note)
        for _, x in r.iterrows():
            rows.append(dict(base, probe=x.probe_model, target=x.target, action_dim=int(x.action_dim),
                             dim_name=DIM_NAMES[int(x.action_dim)], r2=round(x.r2, 6),
                             n_train=int(x.n_train), n_test=int(x.n_test)))
        for (p, t), g in r.groupby(["probe_model", "target"]):
            assert abs(g.r2.mean() - g.avg_r2_for_target.iloc[0]) < 1e-9
            rows.append(dict(base, probe=p, target=t, action_dim=-1, dim_name="mean", r2=round(g.r2.mean(), 6),
                             n_train=int(g.n_train.iloc[0]), n_test=int(g.n_test.iloc[0])))
    if a.local:
        loc = pd.read_csv(a.local)
        for _, x in loc.iterrows():
            d, job, note = ARMS[x.method]
            rows.append(dict(benchmark="realworld", method=x.method, teacher_job=job, ckpt_step=30000, probe=x.probe,
                             target={"abs": "abs_action_local", "delta_h5": "state_delta_h5"}[x.target],
                             action_dim=int(x.action_dim), dim_name=DIM_NAMES[int(x.action_dim)] if x.action_dim >= 0 else "mean",
                             r2=round(float(x.r2), 6), n_train=int(x.n_train), n_test=int(x.n_test),
                             source_dir="local fit, experiments/fit_decode_realworld.py on the full exports",
                             note=note + "; local 20%-episode holdout, ridge alpha=1 / MLP(512,256)"))
    cols = ["benchmark", "method", "teacher_job", "ckpt_step", "probe", "target", "action_dim", "dim_name", "r2",
            "n_train", "n_test", "source_dir", "note"]
    pd.DataFrame(rows)[cols].to_csv(OUT, index=False)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
