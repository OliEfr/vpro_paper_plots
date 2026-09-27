#!/usr/bin/env python
r"""Collect the DK1 embodiment-overlap trade-off into ``results/xemb_hardware_frontier.csv``.

Every intervention tried so far on the DK1 hardware latent sits on one trade-off: the more
robot demos and human video OVERLAP, the less the latent knows about the action. This script
gathers the measured arms into one table so ``plot_xemb_hardware_frontier.py`` can show where
a new arm lands on it.

Each input is a ``summary_<arm>.json`` written by the shared analysis protocol (the same one
``fit_tsne_hardware_encoder.py`` reimplements and reproduces digit-for-digit): task-balanced
432-episode export, <=140 rows per (task, source) cell, episode-grouped split, source
decodability by logistic regression on the raw latent, and an episode-split ridge action R^2
on robot rows.

The two axes:
  source_logreg  how separable robot and human video are, chance 0.5. LOWER is better.
  action_r2      how much of the 7-D end-effector action survives in the latent. HIGHER is
                 better. Without this axis the plot is meaningless: a collapsed latent mixes
                 the sources perfectly and is worth nothing.

Arms fall in three classes, kept apart in the ``kind`` column because they are not comparable
claims:
  teacher   a latent as some teacher actually trained it
  posthoc   the baseline latent with per-source standardisation applied at ANALYSIS time.
            It uses the source label, so it is not a learned invariant latent; it shows the
            two sources are the same manifold offset in mean and scale.
  ours      the arm this study adds: our learned tokenizer in place of the DINO encoder

Sources of the summaries, all on Leonardo under
``/leonardo_work/EUHPC_B38_106/rlfv/repo/experiments/srcinv-tsne-20260904/results/``:
  base_balanced   the DINO dual-view teacher at 30k, the hardware baseline
  base_5k         the same teacher at 5k, an unregularised control
  activeruns      the same recipe on a motion-filtered derivative dataset
  dim4            quant_dim 4 instead of 8, a capacity ablation
  srcoffsets      per-source frame offsets, a retiming arm
  coral3k         CORAL source-invariance penalty, weight 1
  coralstd3k      scale-invariant CORAL
  coralstd_w003   scale-invariant CORAL, weight 0.03
The ``old_stride_*`` summaries are deliberately EXCLUDED: they are the demonstration that the
published index-stride subset confounds source with task, not arms of this study, and they
carry no action R^2.

The new arm does not arrive as a ``summary_<arm>.json``; it comes out of
``fit_tsne_hardware_encoder.py`` as one list keyed by ``model``. Pass that file to
``--encoder-summary`` and its ``ours_learned`` entry becomes the 'ours' point. The fields are
the same because the protocol is the same.

Usage:
    python experiments/collect_xemb_hardware_frontier.py --summaries <dir-of-summary-json>
        [--summaries <another-dir>] [--encoder-summary results/tsne_hardware_encoder_summary.json]
        [--out results/xemb_hardware_frontier.csv]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# arm key -> (display label, kind). Order is the plotting order.
ARMS = {
    "base_balanced": ("DINOv3 encoder, 30k", "teacher"),
    "base_5k": ("DINOv3 encoder, 5k", "teacher"),
    "activeruns": ("motion-filtered data", "teacher"),
    "dim4": ("latent dim 4", "teacher"),
    "srcoffsets": ("per-source offsets", "teacher"),
    "coralstd3k": ("CORAL, scale-inv.", "teacher"),
    "coral3k": ("CORAL, w=1", "teacher"),
    "coralstd_w003": ("CORAL, scale-inv., w=0.03", "teacher"),
    "learned": ("Ours (learned tokenizer)", "ours"),
    "sharedlam6_learned": ("Ours (learned tokenizer)", "ours"),
}
POSTHOC_FROM = "base_balanced"   # the srcstd variant of the baseline becomes its own point


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", action="append", required=True, type=Path,
                   help="directory holding summary_<arm>.json files; repeatable")
    p.add_argument("--encoder-summary", type=Path, default=None,
                   help="results/tsne_hardware_encoder_summary.json from fit_tsne_hardware_encoder.py; "
                        "its ours_learned entry is added as the 'ours' point (same protocol, same fields)")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results" / "xemb_hardware_frontier.csv")
    a = p.parse_args()

    found = {}
    for d in a.summaries:
        for f in sorted(d.glob("summary_*.json")):
            found.setdefault(f.name[len("summary_"):-len(".json")], f)

    # the new arm arrives in this study's own summary format: one list, keyed by `model`
    extra = {}
    if a.encoder_summary and a.encoder_summary.exists():
        # The summary carries every checkpoint that was fitted. Take the LAST one by step, so
        # the frontier compares our arm against base_balanced (30k) at the same training length
        # rather than against whichever entry happened to be written last.
        cands = [s for s in json.loads(a.encoder_summary.read_text())
                 if s.get("model") == "ours_learned"]
        if cands:
            s = max(cands, key=lambda s: int(str(s.get("checkpoint") or 0)))
            extra["learned"] = s
            print(f"read ours_learned @ ckpt {s.get('checkpoint')} from {a.encoder_summary} "
                  f"({len(cands)} checkpoint(s) available)")

    rows = []
    for arm, (label, kind) in ARMS.items():
        f = found.get(arm)
        if f is None and arm not in extra:
            continue
        s = extra[arm] if arm in extra else json.loads(f.read_text())
        raw = s["decodability"]["raw"]
        rows.append({"arm": arm, "label": label, "kind": kind,
                     "source_logreg": raw["logreg_acc"], "source_knn15": raw["knn15_acc"],
                     "silhouette": raw["silhouette"], "action_r2": s.get("action_r2_robot_raw"),
                     "latent_std": s.get("latent_std_per_dim_mean"),
                     "norm_robot": s["norm_mean"]["robot_3cam"],
                     "norm_video": s["norm_mean"]["video_2cam"],
                     "n_sampled": s.get("n_sampled"),
                     "tasks_with_both": s.get("tasks_with_both_sources")})
        if arm == POSTHOC_FROM:
            # same latent, same action content, source label used at analysis time
            sstd = s["decodability"]["srcstd"]
            rows.append({"arm": f"{arm}_srcstd", "label": "post-hoc alignment",
                         "kind": "posthoc",
                         "source_logreg": sstd["logreg_acc"], "source_knn15": sstd["knn15_acc"],
                         "silhouette": sstd["silhouette"], "action_r2": s.get("action_r2_robot_raw"),
                         "latent_std": s.get("latent_std_per_dim_mean"),
                         "norm_robot": s["norm_mean"]["robot_3cam"],
                         "norm_video": s["norm_mean"]["video_2cam"],
                         "n_sampled": s.get("n_sampled"),
                         "tasks_with_both": s.get("tasks_with_both_sources")})

    df = pd.DataFrame(rows)
    missing = [k for k in ARMS if k not in found and k not in extra]
    assert not df.empty, f"no summaries found in {[str(d) for d in a.summaries]}"
    assert df.action_r2.notna().all(), f"arms without action R2: {df[df.action_r2.isna()].arm.tolist()}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(df)} points)")
    if missing:
        print("not present (fine if not yet measured):", ", ".join(missing))
    print(df[["arm", "kind", "source_logreg", "action_r2", "latent_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
