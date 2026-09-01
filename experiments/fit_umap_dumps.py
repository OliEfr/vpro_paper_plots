#!/usr/bin/env python
"""Stage A: fit the UMAP embeddings and dump coordinates for vpro_paper_plots.

This is the *experiment* half of the repo's results/ contract: it reads the
labelled latent exports, fits one joint UMAP per model across its checkpoints,
and writes plain CSVs of coordinates. Nothing here imports matplotlib -- the
figure is drawn separately in the pinned vpro-plots env from these CSVs alone.

Run on tueilsy-st-022 with /mnt/data/workspace/.conda/rlfv/bin/python
(umap-learn 0.5.12, the version that produced the reference figures).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import umap

LAT = "latent_labels.continuous_vector_latents"
VALID = "latent_labels.valid"
LABELS = Path("/mnt/data/workspace/runs_root/runs_lerobot/latent_labels")
DK1 = LABELS / "dk1_p19_sweep_20260719"
EPS_FILE = LABELS / "subset300_eps_allemb.json"
OUT = Path("/mnt/data/robot-learning/umap_dumps")
SEED, STRIDE, NN, MIND = 42, 20, 30, 0.05

RANGES = [("franka", 0, 5657), ("ur5e", 5658, 12157), ("kinova3", 12158, 18657),
          ("iiwa", 18658, 25157), ("sawyer", 25158, 31657)]

TEACHERS = [
    ("xemb1iiwa", 2, ["franka", "iiwa"], "44965301", "44222277"),
    ("lib90t71", 3, ["franka", "iiwa", "kinova3"], "44965302", "44161039"),
    ("xemb4all", 5, ["franka", "iiwa", "kinova3", "ur5e", "sawyer"], "44965304", "44222449"),
]
CKPTS = ["001000", "003000", "010000", "030000", "070000"]

DK1_MODELS = {
    "sharedlam2": [("001000", DK1 / "screen_c_ckpt001000_idx1"),
                   ("003000", DK1 / "screen_c_ckpt003000_idx1")] +
                  [(c, DK1 / f"sharedlam2_dino768d6_ckpt{c}_idx1")
                   for c in ["005000", "010000", "015000", "020000", "025000", "030000"]],
    "sharedlam3side": [(c, DK1 / f"sharedlam3side_dino768d6_ckpt{c}_idx1")
                       for c in ["005000", "010000", "015000", "020000", "025000", "030000"]],
    "sharedlam4tl": [(c, DK1 / f"sharedlam4tl_ckpt{c}_idx1")
                     for c in ["005000", "010000", "015000", "020000", "025000", "030000"]],
}
SOURCES = ["robot_3cam", "video_2cam"]


def libero_root(key, early, job, ck):
    if ck in ("001000", "003000"):
        return LABELS / f"libero_allemb_fourcam_fullfail_lam1o5_{key}early{early}_sub300e_ckpt{ck}_idx0_labeled"
    if ck == "070000":
        return LABELS / f"libero_allemb_fourcam_fullfail_lam1o5_{key}_{job}_ckpt{ck}_idx0_labeled"
    return LABELS / f"libero_allemb_fourcam_fullfail_lam1o5_{key}_{job}_sub300e_ckpt{ck}_idx0_labeled"


def read_latents(root, cols):
    table = ds.dataset(Path(root) / "data", format="parquet").to_table(columns=cols)
    n = table.num_rows
    flat = table[LAT].combine_chunks().flatten()
    try:
        lat = flat.flatten().to_numpy(zero_copy_only=False).reshape(n, 8)
    except Exception:
        lat = flat.to_numpy(zero_copy_only=False).reshape(n, 8)
    return table, lat.astype(np.float32), n


def emb_of(ep):
    out = np.full(ep.shape[0], "", dtype=object)
    for name, lo, hi in RANGES:
        out[(ep >= lo) & (ep <= hi)] = name
    return out


def metrics(X, y, seed=SEED, max_sil=5000):
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    knn = KNeighborsClassifier(n_neighbors=15).fit(sc.transform(Xtr), ytr)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(max_sil, X.shape[0]), replace=False)
    return {"chance": round(1.0 / len(classes), 4),
            "logreg_acc": round(float(accuracy_score(yte, lr.predict(sc.transform(Xte)))), 4),
            "knn15_acc": round(float(accuracy_score(yte, knn.predict(sc.transform(Xte)))), 4),
            "silhouette": round(float(silhouette_score(
                StandardScaler().fit_transform(X)[idx], y[idx])), 4)}


def joint_umap(df):
    X = df[[f"latent_{i}" for i in range(8)]].to_numpy(np.float32)
    xy = umap.UMAP(n_neighbors=NN, min_dist=MIND, metric="euclidean",
                   random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
    df = df.copy()
    df["umap_x"] = np.round(xy[:, 0], 4)
    df["umap_y"] = np.round(xy[:, 1], 4)
    return df


def run_libero(eps, recs):
    rows = []
    for key, n_emb, seen, early, job in TEACHERS:
        rng = np.random.default_rng(SEED)
        frames, avail = [], []
        for ck in CKPTS:
            root = libero_root(key, early, job, ck)
            tb, lat, _ = read_latents(root, ["episode_index", "frame_index", LAT, VALID])
            ep, fr = tb["episode_index"].to_numpy(), tb["frame_index"].to_numpy()
            keep = ((tb[VALID].to_numpy() == 1) & np.isfinite(lat).all(1)
                    & (fr % STRIDE == 0) & np.isin(ep, eps))
            ep, lat = ep[keep], lat[keep]
            emb = emb_of(ep)
            counts = {n: int((emb == n).sum()) for n in seen}
            avail.append(min(counts.values()))
            frames.append((ck, ep, emb, lat, counts))
        per = min([2341] + avail)
        for ck, ep, emb, lat, counts in frames:
            take = np.concatenate([rng.choice(np.where(emb == n)[0], size=per, replace=False)
                                   for n in seen])
            rows.append(pd.DataFrame({
                "teacher": key, "n_emb": n_emb, "checkpoint": ck,
                "embodiment": emb[take],
                **{f"latent_{i}": lat[take, i] for i in range(8)}}))
            recs.append({"family": "libero", "model": key, "checkpoint": ck,
                         "group": "embodiment", "n_per_group": per, **metrics(lat[take], emb[take])})
            print(f"  [{key} {ck}] {counts} -> {per}/emb", flush=True)
        big = joint_umap(pd.concat(rows[-len(CKPTS):], ignore_index=True))
        rows[-len(CKPTS):] = [big]
    out = pd.concat(rows, ignore_index=True)
    return out[["teacher", "n_emb", "checkpoint", "embodiment", "umap_x", "umap_y"]]


def run_dk1(recs):
    out = []
    for model, ckpts in DK1_MODELS.items():
        rng = np.random.default_rng(SEED)
        frames, avail = [], []
        for ck, root in ckpts:
            tb, lat, _ = read_latents(root, ["episode_index", LAT, VALID])
            ep = tb["episode_index"].to_numpy()
            keep = (tb[VALID].to_numpy() == 1) & np.isfinite(lat).all(1)
            ep, lat = ep[keep], lat[keep]
            smap = pd.read_parquet(Path(root) / "meta" / "source_episodes.parquet",
                                   columns=["episode_index", "source_kind"])
            smap = dict(zip(smap["episode_index"], smap["source_kind"]))
            src = np.array([smap.get(int(e), "unknown") for e in ep], dtype=object)
            counts = {s: int((src == s).sum()) for s in SOURCES}
            avail.append(min(counts.values()))
            frames.append((ck, src, lat, counts))
        per = min([6000] + avail)
        parts = []
        for ck, src, lat, counts in frames:
            take = np.concatenate([rng.choice(np.where(src == s)[0], size=per, replace=False)
                                   for s in SOURCES])
            parts.append(pd.DataFrame({
                "model": model, "checkpoint": ck, "source": src[take],
                **{f"latent_{i}": lat[take, i] for i in range(8)}}))
            recs.append({"family": "dk1", "model": model, "checkpoint": ck,
                         "group": "source", "n_per_group": per, **metrics(lat[take], src[take])})
            print(f"  [{model} {ck}] {counts} -> {per}/source", flush=True)
        out.append(joint_umap(pd.concat(parts, ignore_index=True)))
    res = pd.concat(out, ignore_index=True)
    return res[["model", "checkpoint", "source", "umap_x", "umap_y"]]


if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    eps = np.array(json.load(open(EPS_FILE)))
    print("LIBERO teachers:", flush=True)
    run_libero(eps, recs).to_csv(OUT / "umap_xemb_teachers.csv", index=False)
    print("DK1 hardware:", flush=True)
    run_dk1(recs).to_csv(OUT / "umap_dk1_hardware.csv", index=False)
    pd.DataFrame(recs).to_csv(OUT / "umap_decodability.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}", flush=True)
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB", flush=True)
