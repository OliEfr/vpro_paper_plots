#!/usr/bin/env python
"""Write results/tsne_hardware_motion.csv and results/tsne_hardware_motion_decodability.csv.

Same latents as fit_tsne_dumps.py's DK1 part (30k checkpoints of the multi-view LAM
sharedlam2 / 43524549 and the side-only LAM sharedlam3 / 44160743, full-dataset exports
dk1_postproc19_*_ckpt030000_idx1), but embedded in three views of the same latent:

  raw latent (8-D)
  motion (xyz) subspace (3-D)   directions used by a ridge probe latent -> state[t+5]-state[t]
                                (xyz), fitted on robot rows only; = the action-relevant part
  minus top-3 pose directions   latent with the 3 directions most predictive of the ABSOLUTE
                                EE pose (ridge latent -> observation.state, robot rows) removed

Balanced 3,000 frames per source at a 1 s stride (frame_index % 20 == 0), seed 42,
StandardScaler on all valid rows, t-SNE perplexity 30 init="pca". kNN-15 5-fold
decodability is fitted on the projected latent, not on the 2-D embedding.

Run in the staging dir with exports/<lam>/ present (needs scikit-learn + pyarrow):
    python experiments/fit_tsne_hardware_motion.py --stage <stage>
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
MODELS = [("sharedlam2", "front+side"), ("sharedlam3side", "side only")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    a = ap.parse_args()
    src = pd.read_parquet(a.stage / "exports/sharedlam2/meta/source_episodes.parquet")[["episode_index", "task", "source_kind"]]
    out, dec = [], []
    for m, mname in MODELS:
        t = ds.dataset(a.stage / f"exports/{m}/data", format="parquet").to_table(
            columns=["episode_index", "frame_index", "observation.state", "latent_labels.continuous_vector_latents",
                     "latent_labels.valid"]).to_pandas().merge(src, on="episode_index")
        t = t.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        Z = np.stack([np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in r])
                      for r in t["latent_labels.continuous_vector_latents"].values])
        S = np.stack(t["observation.state"].values)
        v = t["latent_labels.valid"].values.astype(bool)
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
        stride = t.frame_index.values % 20 == 0
        rng = np.random.RandomState(42)
        ir, ih = np.where(rob & stride)[0], np.where(hum & stride)[0]
        n = min(len(ir), len(ih), 3000)
        idx = np.concatenate([rng.choice(ir, n, replace=False), rng.choice(ih, n, replace=False)])
        y = np.r_[np.ones(n), np.zeros(n)]
        views = {"raw latent (8-D)": Zs,
                 "motion (xyz) subspace (3-D)": Zs @ Um[:, :3],
                 "minus top-3 pose directions (5-D)": Zs - (Zs @ Up[:, :3]) @ Up[:, :3].T}
        for vname, X in views.items():
            knn = cross_val_score(KNeighborsClassifier(15), X[idx], y, cv=5).mean()
            E = TSNE(2, perplexity=30, random_state=42, init="pca").fit_transform(X[idx])
            dec.append(dict(model=m, model_label=mname, view=vname, dims=X.shape[1], n_per_source=n, knn15=round(knn, 3)))
            out.append(pd.DataFrame(dict(model=m, model_label=mname, view=vname,
                                         source=np.where(y == 1, "robot_3cam", "video_2cam"),
                                         tsne_x=E[:, 0].round(2), tsne_y=E[:, 1].round(2))))
            print(m, vname, f"kNN15 {knn:.3f}", flush=True)
    pd.concat(out).to_csv(HERE / "results" / "tsne_hardware_motion.csv", index=False)
    pd.DataFrame(dec).to_csv(HERE / "results" / "tsne_hardware_motion_decodability.csv", index=False)


if __name__ == "__main__":
    main()
