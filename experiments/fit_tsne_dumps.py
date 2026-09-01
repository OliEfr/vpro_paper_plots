#!/usr/bin/env python
"""Stage A, t-SNE variant: dump t-SNE coordinates for the same latents as UMAP.

This is `fit_umap_dumps.py` with one thing changed -- the embedder. Data
selection is byte-identical on purpose: same models, same checkpoints, same
`subset300_eps_allemb.json` episode subset, same 1 s frame stride, same balanced
per-group draw off the same seeded generator, same StandardScaler. If the two
methods disagree about the picture, the disagreement is the method and nothing
else, which is the only reason to run this at all.

Decodability is NOT recomputed here. Those numbers are fitted on the raw 8-D
latent, never on the 2-D projection, so they do not depend on which embedder
drew the scatter -- `results/umap_decodability.csv` covers this run too, and a
second copy would only invite the two to drift apart.

Run on tueilsy-st-022 with /mnt/data/workspace/.conda/rlfv/bin/python.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

LAT = "latent_labels.continuous_vector_latents"
VALID = "latent_labels.valid"
LABELS = Path("/mnt/data/workspace/runs_root/runs_lerobot/latent_labels")
DK1 = LABELS / "dk1_p19_sweep_20260719"
EPS_FILE = LABELS / "subset300_eps_allemb.json"
OUT = Path("/mnt/data/robot-learning/tsne_dumps")
SEED, STRIDE = 42, 20
# perplexity 30 is the analogue of the UMAP fit's n_neighbors=30: both set how
# many neighbours define "local". sklearn requires n_samples > 3*perplexity;
# the smallest pooled fit here is 23410 points, so there is no risk of that.
PERPLEXITY = 30.0

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


def joint_tsne(df, tag):
    """One t-SNE per model, fitted jointly over that model's checkpoints.

    t-SNE has no transform(), so "joint" cannot mean fit-then-project the way it
    does for UMAP: the checkpoints are pooled into one array and embedded in a
    single fit. Rows come back in input order, so the checkpoint column already
    travels with them and splitting the result apart again is just a groupby.

    That makes a model's panels commensurate with each other and with nothing
    else -- across models the latent bases differ, as with UMAP, and on top of
    that t-SNE fixes no global scale at all.
    """
    X = df[[f"latent_{i}" for i in range(8)]].to_numpy(np.float32)
    t0 = time.time()
    xy = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca",
              learning_rate="auto", metric="euclidean", random_state=SEED,
              n_jobs=-1, verbose=1).fit_transform(StandardScaler().fit_transform(X))
    print(f"  [{tag}] t-SNE on {X.shape[0]} pts in {time.time()-t0:.0f}s", flush=True)
    df = df.copy()
    df["tsne_x"] = np.round(xy[:, 0], 4)
    df["tsne_y"] = np.round(xy[:, 1], 4)
    return df


def run_libero(eps):
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
            print(f"  [{key} {ck}] {counts} -> {per}/emb", flush=True)
        big = joint_tsne(pd.concat(rows[-len(CKPTS):], ignore_index=True), key)
        rows[-len(CKPTS):] = [big]
    out = pd.concat(rows, ignore_index=True)
    return out[["teacher", "n_emb", "checkpoint", "embodiment", "tsne_x", "tsne_y"]]


def run_dk1():
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
            print(f"  [{model} {ck}] {counts} -> {per}/source", flush=True)
        out.append(joint_tsne(pd.concat(parts, ignore_index=True), model))
    res = pd.concat(out, ignore_index=True)
    return res[["model", "checkpoint", "source", "tsne_x", "tsne_y"]]


if __name__ == "__main__":
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    eps = np.array(json.load(open(EPS_FILE)))
    print("LIBERO teachers:", flush=True)
    run_libero(eps).to_csv(OUT / "tsne_xemb_teachers.csv", index=False)
    print("DK1 hardware:", flush=True)
    run_dk1().to_csv(OUT / "tsne_dk1_hardware.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}", flush=True)
    for f in sorted(OUT.glob("*.csv")):
        print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB", flush=True)
