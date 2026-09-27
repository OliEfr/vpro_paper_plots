#!/usr/bin/env python
r"""Cross-embodiment geometry of the DK1 hardware latent: DINO encoder vs our learned tokenizer.

Writes ``results/tsne_hardware_encoder.csv`` (t-SNE + UMAP point coordinates) and
``results/tsne_hardware_encoder_metrics.csv`` (source decodability, action R^2, latent norms),
which ``plot_tsne_hardware_encoder.py`` turns into the figure.

WHAT IS COMPARED
Two DK1 dual-view LAM teachers that differ in ONE config key, ``policy.dino_model_name``:

  ours_dino     sharedlam4, frozen DINOv3-S/16 encoder     (job 44227723 on MN5, ckpt 030000)
  ours_learned  sharedlam6, our learned patch tokenizer    (Leonardo, ckpt 030000)

Everything else is identical: dim 768, spatial/temporal depth 6, patch 32, image 256,
``quant_dim`` 8, future offsets [1,5,9], cameras front+side, dual-view augmentation on,
seed 1003, 30k steps, batch 32/GPU, LR 1e-4 cosine with 1000 warmup, and the same
``experiments/dk1-postprocessed-newtasks-3cam-placeholder-20260729`` episode list. They even
run the same code package: with ``dino_model_name=null`` the DINO-port package takes its
learned ``pixel_projection`` tokenizer, which is the unmodified learned package's path.

THE EPISODE SET
Both exports cover the SAME task-balanced 432 episodes: 36 tasks x 6 robot_3cam + 6
video_2cam. This is deliberate. The published DK1 source-separability figures sample an
index-stride episode subset in which only 1-10 of 32-34 tasks carry both sources, so "source"
is nearly collinear with "task" and the separability is partly a task effect. The balanced set
removes that confound; the dataset itself was always task-matched.

PROTOCOL (kept identical to the eight previously measured DK1 arms so the numbers line up)
  sampling      <=140 rows per (task, source) cell -> 10,080 rows, 5,040 per source
  split         EPISODE-grouped 70/30 (a row split leaks temporal autocorrelation)
  variants      raw              the latent as trained
                unitnorm         z/||z||, a label-free scale control: latent norm alone
                                 separates the sources on the dual-view teachers
                srcstd           per-source standardisation applied POST HOC at analysis
                                 time. It uses the source label, so it shows the two
                                 sources are the same manifold offset in mean and scale --
                                 it is NOT evidence of a learned invariant latent.
  decodability  logistic regression, kNN-15, silhouette; chance = 0.5
  action R^2    episode-split ridge (alpha=1) latent -> 7-D EE action, ROBOT rows only.
                Guards against reading "the sources overlap" off a latent that merely
                got destroyed.
  embeddings    t-SNE (perplexity 30, init pca) and UMAP (n_neighbors 30, min_dist 0.05),
                seed 42, fitted per (model, variant) on standardised features.

Protocol, sampler and metric definitions follow
``rlfv/repo/experiments/srcinv-tsne-20260904/analysis/fit_tsne_arm.py`` on Leonardo, so this
arm's numbers are directly comparable to the baseline, the CORAL/GRL/dim-4/offset arms and
the motion-filtered dataset arm measured there.

INPUTS
Each export is a LeRobot label root written by the balanced-432 export job, containing
``data/**.parquet`` with ``episode_index, frame_index, task_index, action,
latent_labels.continuous_vector_latents, latent_labels.valid`` plus
``meta/source_episodes.parquet`` with ``episode_index, source_kind, task``.
(The label exporter does not preserve ``source_episodes.parquet`` by itself; the export
sbatch copies it back. Without it there are no source labels and nothing here works.)

Needs scikit-learn, pyarrow and umap-learn, so run it on tueilsy-st-022 like the other
hardware fits:

    /mnt/data/workspace/.conda/rlfv/bin/python experiments/fit_tsne_hardware_encoder.py \
        --export ours_dino=/path/balanced432_sharedlam4_base_ckpt030000 \
        --export ours_learned=/path/balanced432_sharedlam6_learned_ckpt030000 \
        --out-dir results
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

LAT = "latent_labels.continuous_vector_latents"
VALID = "latent_labels.valid"
SOURCES = ["robot_3cam", "video_2cam"]
SEED = 42
PER_CELL = 140          # rows per (task, source) cell
VARIANTS = ["raw", "unitnorm", "srcstd"]
LABELS = {"ours_dino": "DINOv3 encoder", "ours_learned": "Ours (learned tokenizer)"}
H_MOTION = 5            # the LAM horizon: state[t+5]-state[t], as in fit_tsne_hardware_motion.py
# spaces already expressed in standardised units -- re-standardising them would rescale the
# subspace axes and stop matching the existing motion figure
NO_RESTANDARDISE = {"motion", "pose_removed"}


def load(root):
    """Valid latent rows of one export, with source/task labels attached per episode."""
    root = Path(root)
    tb = ds.dataset(str(root / "data"), format="parquet").to_table(
        columns=["episode_index", "frame_index", "task_index", "action", "observation.state",
                 LAT, VALID])
    n = tb.num_rows
    z = tb[LAT].combine_chunks().flatten()
    try:
        z = z.flatten().to_numpy(zero_copy_only=False)
    except Exception:
        z = z.to_numpy(zero_copy_only=False)
    d = z.size // n
    z = z.reshape(n, d).astype(np.float32)
    ep = tb["episode_index"].to_numpy()
    fr = tb["frame_index"].to_numpy()
    act = np.asarray(tb["action"].combine_chunks().flatten().to_numpy(zero_copy_only=False)).reshape(n, 7)
    st = np.asarray(tb["observation.state"].combine_chunks().flatten()
                    .to_numpy(zero_copy_only=False)).reshape(n, 7)
    keep = (tb[VALID].to_numpy() == 1) & np.isfinite(z).all(1)

    se = pd.read_parquet(root / "meta" / "source_episodes.parquet",
                         columns=["episode_index", "source_kind", "task"])
    smap = dict(zip(se["episode_index"], se["source_kind"]))
    tmap = dict(zip(se["episode_index"], se["task"]))
    ep, fr, z, act, st = ep[keep], fr[keep], z[keep], act[keep], st[keep]
    src = np.array([smap.get(int(e), "?") for e in ep], dtype=object)
    tsk = np.array([tmap.get(int(e), "?") for e in ep], dtype=object)
    m = np.isin(src, SOURCES)
    return z[m], ep[m], fr[m], src[m], tsk[m], act[m], st[m], int(d)


def balanced_sample(ep, src, tsk, per_cell=PER_CELL, seed=SEED):
    """<= per_cell rows per (task, source) cell: task- AND source-balanced."""
    rng = np.random.default_rng(seed)
    keep = []
    for t in np.unique(tsk):
        for s in SOURCES:
            idx = np.where((tsk == t) & (src == s))[0]
            if len(idx) == 0:
                continue
            keep.append(rng.choice(idx, size=min(per_cell, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def decodability(X, y, groups):
    """Source decodability under an EPISODE-grouped 70/30 split. Chance = 0.5."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(X, y, groups))
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    kn = KNeighborsClassifier(n_neighbors=15).fit(sc.transform(X[tr]), y[tr])
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    return {"logreg_acc": round(float(accuracy_score(y[te], lr.predict(sc.transform(X[te])))), 4),
            "knn15_acc": round(float(accuracy_score(y[te], kn.predict(sc.transform(X[te])))), 4),
            "silhouette": round(float(silhouette_score(Xs[idx], y[idx])), 4),
            "n_train": int(len(tr)), "n_test": int(len(te))}


def action_r2(Z, A, eps, seed=SEED):
    """Episode-split ridge latent -> 7-D EE action on robot rows. How much the latent kept."""
    u = np.unique(eps)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    te = set(u[:max(1, int(0.3 * len(u)))].tolist())
    m = np.array([e in te for e in eps])
    if m.sum() < 200 or (~m).sum() < 200:
        return None
    sc = StandardScaler().fit(Z[~m])
    r = Ridge(alpha=1.0).fit(sc.transform(Z[~m]), A[~m])
    return round(float(r2_score(A[m], r.predict(sc.transform(Z[m])), multioutput="variance_weighted")), 4)


def variants_of(Z, S):
    """raw, unit-norm (scale control), per-source-standardised (post-hoc alignment)."""
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-8)
    Zs = Z.copy()
    for s in SOURCES:
        m = S == s
        Zs[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-8)
    return {"raw": Z, "unitnorm": Zn, "srcstd": Zs}


def motion_pose_bases(Z, ST, ep, src, h=H_MOTION):
    """The two subspaces the existing hardware motion figure decomposes the latent into.

    motion:  top-3 left singular directions of the ridge map latent -> (state[t+h]-state[t])[:3],
             i.e. the part of the latent that predicts end-effector translation.
    pose:    top-3 of the ridge map latent -> state, i.e. absolute arm configuration.

    Both are fitted on ROBOT rows only (human video has no usable state) and on all valid rows
    before sampling, exactly as experiments/fit_tsne_hardware_motion.py does. Returns the two
    orthonormal bases, or (None, None) when there is not enough robot data to fit them.
    """
    rob = src == SOURCES[0]
    if rob.sum() < 500:
        return None, None
    Zs = StandardScaler().fit(Z).transform(Z)
    same = np.roll(ep, -h) == ep                      # the +h row must be the same episode
    D = np.roll(ST, -h, axis=0) - ST
    okm = rob & same
    if okm.sum() < 500:
        return None, None
    Wm = Ridge(1.0).fit(Zs[okm], D[okm][:, :3]).coef_
    Um, _, _ = np.linalg.svd(Wm.T, full_matrices=False)
    Wp = Ridge(1.0).fit(Zs[rob], ST[rob]).coef_
    Up, _, _ = np.linalg.svd(Wp.T, full_matrices=False)
    return Um, Up


def checkpoint_of(export, given=None):
    """Training step behind an export. Taken from the spec if given, else read off the
    directory name (the export jobs write ..._ckpt030000), else left blank rather than
    guessed -- a wrong checkpoint label would silently mislabel a panel."""
    if given:
        return given
    m = re.search(r"ckpt(\d+)", Path(export).name)
    return m.group(1) if m else ""


def run_one(model, export, coords, metrics, summaries, ckpt=""):
    import umap  # imported here so a coordinate-only rerun does not need it installed

    z, ep, fr, src, tsk, act, st, d = load(export)
    # subspaces are fitted on ALL valid robot rows, before sampling
    Um, Up = motion_pose_bases(z, st, ep, src)
    take = balanced_sample(ep, src, tsk)
    Z, E, S, T, A = z[take], ep[take], src[take], tsk[take], act[take]
    shared = set(np.unique(T[S == SOURCES[0]])) & set(np.unique(T[S == SOURCES[1]]))
    r2 = action_r2(Z[S == SOURCES[0]], A[S == SOURCES[0]], E[S == SOURCES[0]])
    norms = {s: round(float(np.linalg.norm(Z[S == s], axis=1).mean()), 4) for s in SOURCES}
    summ = {"model": model, "model_label": LABELS.get(model, model), "checkpoint": ckpt,
            "export": str(export),
            "latent_dim": d, "rows_total": int(len(z)), "n_sampled": int(len(Z)),
            "episodes": {s: int(len(np.unique(ep[src == s]))) for s in SOURCES},
            "tasks_with_both_sources": len(shared), "tasks_total": int(len(np.unique(tsk))),
            "norm_mean": norms, "action_r2_robot_raw": r2,
            "latent_std_per_dim_mean": round(float(Z.std(axis=0).mean()), 5),
            "decodability": {}}
    print(f"[{model} ckpt{ckpt or '?'}] {len(z)} valid rows, sampled {len(Z)} "
          f"({(S == SOURCES[0]).sum()} robot / {(S == SOURCES[1]).sum()} video), "
          f"{len(shared)} tasks with both sources, action R2 {r2}", flush=True)

    spaces = variants_of(Z, S)
    if Um is not None:
        Zstd = StandardScaler().fit(z).transform(Z)      # same scaler basis as the subspace fit
        spaces["motion"] = Zstd @ Um[:, :3]                       # 3-D, predicts EE translation
        spaces["pose_removed"] = Zstd - (Zstd @ Up[:, :3]) @ Up[:, :3].T   # top-3 pose dirs out
    else:
        print(f"  [{model}] too few robot rows for the motion/pose subspaces; skipping those spaces",
              flush=True)

    for vname, V in spaces.items():
        m = decodability(V, S, E)
        summ["decodability"][vname] = m
        Vs = V if vname in NO_RESTANDARDISE else StandardScaler().fit_transform(V)
        xy_t = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED,
                    max_iter=1000).fit_transform(Vs)
        xy_u = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="euclidean",
                         random_state=SEED).fit_transform(Vs)
        coords.append(pd.DataFrame({
            "model": model, "model_label": LABELS.get(model, model), "checkpoint": ckpt,
            "variant": vname, "source": S, "task": T, "episode": E,
            "tsne_x": np.round(xy_t[:, 0], 3), "tsne_y": np.round(xy_t[:, 1], 3),
            "umap_x": np.round(xy_u[:, 0], 3), "umap_y": np.round(xy_u[:, 1], 3)}))
        metrics.append({"model": model, "model_label": LABELS.get(model, model),
                        "checkpoint": ckpt, "variant": vname, "latent_dim": d,
                        "space_dim": int(V.shape[1]), "logreg_acc": m["logreg_acc"], "knn15_acc": m["knn15_acc"],
                        "silhouette": m["silhouette"], "action_r2": r2,
                        "norm_robot": norms["robot_3cam"], "norm_video": norms["video_2cam"],
                        "latent_std_mean": summ["latent_std_per_dim_mean"],
                        "n_sampled": int(len(Z)), "n_per_source": int((S == SOURCES[0]).sum()),
                        "tasks_with_both": len(shared),
                        "episodes_robot": summ["episodes"]["robot_3cam"],
                        "episodes_video": summ["episodes"]["video_2cam"]})
        print(f"  {vname:9s} logreg {m['logreg_acc']:.4f}  knn15 {m['knn15_acc']:.4f}  "
              f"silhouette {m['silhouette']:+.4f}", flush=True)
    summaries.append(summ)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="append", required=True, metavar="MODEL[:CKPT]=PATH",
                   help="repeatable, e.g. --export ours_dino=/path/balanced432_..._ckpt030000 . The "
                        "checkpoint is read off the directory name unless given explicitly as "
                        "MODEL:CKPT=PATH, so one call can fit several checkpoints of the same model.")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    a = p.parse_args()

    coords, metrics, summaries = [], [], []
    for spec in a.export:
        key, _, path = spec.partition("=")
        assert path, f"--export needs MODEL=PATH or MODEL:CKPT=PATH, got {spec!r}"
        model, _, given_ckpt = key.partition(":")
        run_one(model, Path(path), coords, metrics, summaries, ckpt=checkpoint_of(path, given_ckpt))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(coords, ignore_index=True).to_csv(a.out_dir / "tsne_hardware_encoder.csv", index=False)
    pd.DataFrame(metrics).to_csv(a.out_dir / "tsne_hardware_encoder_metrics.csv", index=False)
    (a.out_dir / "tsne_hardware_encoder_summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    print("wrote", a.out_dir / "tsne_hardware_encoder.csv")
    print("wrote", a.out_dir / "tsne_hardware_encoder_metrics.csv")
    print(pd.DataFrame(metrics)[["model", "checkpoint", "variant", "logreg_acc", "knn15_acc",
                                 "silhouette", "action_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
