#!/usr/bin/env python
"""Write results/tsne_hardware_motion_timeline.csv (+ _decodability.csv).

Robot-vs-human t-SNE of the MULTI-VIEW hardware LAM (sharedlam2, 43524549) at the
first and the last checkpoint, in the three views of fit_tsne_hardware_motion.py:

  raw           the 8-D latent
  motion        projection onto the 3 directions a ridge probe (latent -> EE motion
                state[t+5]-state[t], xyz, robot rows of THAT checkpoint) uses
  pose_removed  the latent minus its 3 directions most predictive of the absolute EE
                pose (ridge latent -> observation.state, robot rows of THAT checkpoint)

Checkpoints: 1k comes from the 3k screening run screen_c (identical config, seed, data
and LR schedule as the production run, so it lies on the same trajectory -- see
plot_tsne_hardware.py); 30k is the production checkpoint. Both are the 55-episode
stratified subset exports (13,127 valid rows: 41 robot_3cam + 14 video_2cam episodes),
so the two panels see identical frames:

  <stage>/exports_sub/screen_c_ckpt001000_idx1
  <stage>/exports_sub/sharedlam2_dino768d6_ckpt030000_idx1

Every valid frame is used (no stride: the subset is small), balanced to the smaller
source. Probes and projections are fitted per checkpoint; t-SNE perplexity 30,
init="pca", seed 42, one embedding per (checkpoint, view). kNN-15 5-fold source
decodability is fitted on the projected latent, never on the embedding.

    python experiments/fit_tsne_hardware_motion_timeline.py --stage <stage>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
H = 5
CKPTS = [("001000", "screen_c_ckpt001000_idx1"), ("030000", "sharedlam2_dino768d6_ckpt030000_idx1")]
VIEWS = ["raw", "motion", "pose_removed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    out, dec = [], []
    for ck, name in CKPTS:
        root = a.stage / "exports_sub" / name
        src = pd.read_parquet(root / "meta/source_episodes.parquet")[["episode_index", "source_kind"]]
        t = ds.dataset(root / "data", format="parquet").to_table(
            columns=["episode_index", "frame_index", "observation.state", "latent_labels.continuous_vector_latents",
                     "latent_labels.valid"]).to_pandas().merge(src, on="episode_index")
        t = t.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        v = t["latent_labels.valid"].values.astype(bool)
        Z = np.stack([np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in r])
                      for r in t["latent_labels.continuous_vector_latents"].values])
        S = np.stack(t["observation.state"].values)
        eps = t.episode_index.values
        rob = (t.source_kind.values == "robot_3cam") & v
        hum = (t.source_kind.values == "video_2cam") & v
        Zs = StandardScaler().fit(Z[v]).transform(Z)
        same = np.roll(eps, -H) == eps
        D = np.roll(S, -H, axis=0) - S
        okm = rob & same
        Wm = Ridge(1.0).fit(Zs[okm], D[okm][:, :3]).coef_
        Um, _, _ = np.linalg.svd(Wm.T, full_matrices=False)
        Wp = Ridge(1.0).fit(Zs[rob], S[rob]).coef_
        Up, _, _ = np.linalg.svd(Wp.T, full_matrices=False)
        rng = np.random.RandomState(42)
        ir, ih = np.where(rob)[0], np.where(hum)[0]
        n = min(len(ir), len(ih))
        idx = np.concatenate([rng.choice(ir, n, replace=False), rng.choice(ih, n, replace=False)])
        y = np.r_[np.ones(n), np.zeros(n)]
        views = {"raw": Zs, "motion": Zs @ Um[:, :3], "pose_removed": Zs - (Zs @ Up[:, :3]) @ Up[:, :3].T}
        for vname in VIEWS:
            X = views[vname]
            knn = cross_val_score(KNeighborsClassifier(15), X[idx], y, cv=5).mean()
            E = TSNE(2, perplexity=30, random_state=42, init="pca").fit_transform(X[idx])
            dec.append(dict(model="sharedlam2", checkpoint=ck, view=vname, dims=X.shape[1], n_per_source=n, knn15=round(knn, 3)))
            out.append(pd.DataFrame(dict(model="sharedlam2", checkpoint=ck, view=vname,
                                         source=np.where(y == 1, "robot_3cam", "video_2cam"),
                                         tsne_x=E[:, 0].round(2), tsne_y=E[:, 1].round(2))))
            print(ck, vname, f"n/source {n} kNN15 {knn:.3f}", flush=True)
    pd.concat(out).to_csv(HERE / "results" / "tsne_hardware_motion_timeline.csv", index=False)
    pd.DataFrame(dec).to_csv(HERE / "results" / "tsne_hardware_motion_timeline_decodability.csv", index=False)


if __name__ == "__main__":
    main()
