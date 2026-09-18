#!/usr/bin/env python
"""Per-episode reliability proxies for the decoded EE motion, and the episode pick.

Human video has no ground truth, so how well the decoded Delta-x tracks the true motion
cannot be measured there. This script tests label-free proxies on ROBOT episodes, where
the per-episode Delta-x R^2 of the ridge probe IS known, and then ranks the human
episodes of the four eval tasks by the proxy that predicts it best.

Proxies (all per episode, all computable without actions):
  ood_maha     mean Mahalanobis distance of the episode's latents to the robot-latent
               distribution (mean/cov on robot frames) -- OOD-ness for the probe; lower = better
  agree_ridge_mlp  Pearson r between the ridge and the MLP decoding of Delta-x (same LAM)
  agree_multi_side Pearson r between the multi-view and the side-only ridge decodings of Delta-x
  amp_dx       std of the decoded Delta-x (a flat decoding cannot track anything)

Validation: Spearman rho between each proxy and the per-episode ridge Delta-x R^2 on the
robot episodes that were HELD OUT of the probe fit (the same 20% episode split as
fit_decode_realworld.py, seed 42). Printed; the winner is used for the ranking.

Writes results/decoded_motion_realworld_selection.csv: one row per eval-task episode
(robot and human) with the proxies, the R^2 where it exists, and `rank_in_task` by the
chosen proxy. extract_decoded_motion_realworld.py --episodes <csv> then builds the figure
from the top-k rows.

Run in the staging dir (decoded_*.parquet from fit_decode_realworld.py, exports/):
    python experiments/select_decoded_motion_realworld.py --stage <stage> [--proxy auto]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "decoded_motion_realworld_selection.csv"
EVAL_TASKS = ["milk on pink plate", "salt on pink plate", "banana in cardboard box", "banana in black bowl"]
# LAM-training tasks with the highest median ridge/MLP agreement on human video (2026-09-18 ranking
# over all 36 tasks: 0.97, 0.94, 0.94, 0.92). Each has 10 robot + 20 human episodes.
TRAIN_BEST = ["yellow brick in orange bowl", "blue brick in blue bowl", "yellow cup next to stove", "yellow brick on pink plate"]
PROXIES = ["ood_maha", "agree_ridge_mlp", "agree_multi_side", "amp_dx"]
HIGHER_IS_BETTER = {"ood_maha": False, "agree_ridge_mlp": True, "agree_multi_side": True, "amp_dx": True}


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def latents(root):
    t = ds.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "frame_index", "latent_labels.continuous_vector_latents", "latent_labels.valid"]).to_pandas()
    t = t.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    Z = np.stack([np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in r])
                  for r in t["latent_labels.continuous_vector_latents"].values])
    return t, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--proxy", default="auto", choices=["auto"] + PROXIES)
    ap.add_argument("--tasks", default="eval", choices=["eval", "train-best"],
                    help="eval: the four held-out eval tasks -> *_selection.csv; train-best: the TRAIN_BEST "
                         "tasks (highest median human-video proxy among LAM-training tasks) -> *_selection_train.csv")
    a = ap.parse_args()
    multi = pd.read_parquet(a.stage / "decoded_sharedlam2.parquet").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    side = pd.read_parquet(a.stage / "decoded_sharedlam3side.parquet").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    tz, Z = latents(a.stage / "exports" / "sharedlam2")
    assert (tz.episode_index.values == multi.episode_index.values).all()
    ok = multi.valid.values & multi.gt_dx.notna().values | (multi.valid.values & (multi.source_kind.values == "video_2cam"))
    rob = (multi.source_kind.values == "robot_3cam") & multi.valid.values
    mu, cov = Z[rob].mean(0), np.cov(Z[rob].T)
    icov = np.linalg.inv(cov)
    dz = Z - mu
    maha = np.sqrt(np.einsum("ij,jk,ik->i", dz, icov, dz))

    # same held-out split as fit_decode_realworld.py
    eps = multi.episode_index.values
    rng = np.random.RandomState(42)
    reps = np.unique(eps[rob])
    te_eps = set(rng.choice(reps, int(0.2 * len(reps)), replace=False).tolist())

    rows = []
    for ep, g in multi.groupby("episode_index"):
        i = g.index.values
        v = g.valid.values & g.ridge_dx.notna().values
        if v.sum() < 30:
            continue
        gs = side.loc[i]
        d = dict(episode_index=int(ep), task=g.task.iloc[0], source_kind=g.source_kind.iloc[0], n_valid=int(v.sum()),
                 ood_maha=float(maha[i][v].mean()),
                 agree_ridge_mlp=float(np.corrcoef(g.ridge_dx.values[v], g.mlp_dx.values[v])[0, 1]) if g.mlp_dx.notna().any() else np.nan,
                 agree_multi_side=float(np.corrcoef(g.ridge_dx.values[v], gs.ridge_dx.values[v])[0, 1]),
                 amp_dx=float(g.ridge_dx.values[v].std()),
                 heldout=bool(ep in te_eps))
        if g.source_kind.iloc[0] == "robot_3cam":
            gt = g.gt_dx.values[v]
            d["r2_dx_ridge_multi"] = float(r2(gt, g.ridge_dx.values[v]))
            d["r2_dx_ridge_side"] = float(r2(gt, gs.ridge_dx.values[v]))
            d["r2_xyz_ridge_multi"] = float(np.mean([r2(g[f"gt_d{k}"].values[v], g[f"ridge_d{k}"].values[v]) for k in "xyz"]))
        rows.append(d)
    df = pd.DataFrame(rows)

    val = df[(df.source_kind == "robot_3cam") & df.heldout]
    print(f"proxy validation on {len(val)} held-out robot episodes (Spearman rho vs per-episode ridge Delta-x R^2, multi-view):")
    scores = {}
    for p in PROXIES:
        rho, pv = spearmanr(val[p], val.r2_dx_ridge_multi)
        rho = rho if HIGHER_IS_BETTER[p] else -rho
        scores[p] = rho
        print(f"  {p:18s} rho = {rho:+.3f}  (p = {pv:.3g}; sign flipped for lower-is-better)")
    proxy = a.proxy if a.proxy != "auto" else max(scores, key=scores.get)
    print("chosen proxy:", proxy)

    tasks = EVAL_TASKS if a.tasks == "eval" else TRAIN_BEST
    out = OUT if a.tasks == "eval" else OUT.with_name(OUT.stem + "_train.csv")
    sel = df[df.task.isin(tasks)].copy()
    sign = 1 if HIGHER_IS_BETTER[proxy] else -1
    sel["score"] = sign * sel[proxy]
    sel["proxy"] = proxy
    sel["rank_in_task"] = sel.groupby(["task", "source_kind"]).score.rank(ascending=False, method="first").astype(int)
    sel = sel.sort_values(["task", "source_kind", "rank_in_task"], ascending=[True, False, True])
    sel.to_csv(out, index=False, float_format="%.4f")
    print("wrote", out, len(sel), "episodes of", tasks)
    print(sel[sel.rank_in_task <= 2][["task", "source_kind", "episode_index", proxy, "r2_dx_ridge_multi"]].to_string(index=False))


if __name__ == "__main__":
    main()
