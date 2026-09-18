"""Write results/probe_perdim.csv -- per-action-dimension latent->action probe R^2.

Independent of plot_probing.py and results/probing_*.csv: this dump was re-pulled
from the probe artifacts on MN5 on 2026-09-17, and nothing here reads or writes
the older probing files.

WHAT IT READS. Every LAM teacher in the project is followed by a latent-label
export and a probe job (ridge alpha=1 + MLP [512,256], episode split, seed 42,
`--probe-max-samples 0`, targets `current_action` and `future_action_mean`
= mean over the next 5 frames). The probe writes one directory holding
`action_probe_r2.csv`, whose rows are (probe_model, target, action_dim, r2,
avg_r2_for_target, ...). Those directories are the source of truth; the mean
R^2 quoted in the brain pages is `avg_r2_for_target` from these files.

Cluster locations (Oliver's account, `ssh alogin1.bsc.es`):

    /gpfs/scratch/ehpc637/oliver_hausdorfer/runs_lerobot/probe_analysis/<dir>
    /gpfs/scratch/ehpc637/oliver_hausdorfer/runs_lerobot/latent_analysis/2026-06-30/mimicgen_xemb_probes/{single,multi}

Mirror them locally (the scores CSV is large and not needed):

    rsync -a --exclude action_probe_scores.csv \
        alogin1.bsc.es:/gpfs/scratch/ehpc637/oliver_hausdorfer/runs_lerobot/probe_analysis/ \
        <mirror>/probe_analysis/
    rsync -a --exclude action_probe_scores.csv \
        alogin1.bsc.es:/gpfs/scratch/ehpc637/oliver_hausdorfer/runs_lerobot/latent_analysis/2026-06-30/mimicgen_xemb_probes/ \
        <mirror>/mimicgen_xemb_probes_20260630/

then `python experiments/extract_probe_perdim.py --mirror <mirror>`. That copies
each arm's per-dim CSV verbatim into results/probe_perdim_raw/<arm_key>.csv
(provenance, small) and rebuilds results/probe_perdim.csv from those copies.
Without `--mirror` it rebuilds from the raw copies already in the repo.

ARM CHOICE. One canonical arm per (benchmark, method), chosen so that within a
benchmark the four teachers differ only in the lever under test (view count,
action grounding, flow supervision). Alternates are kept in the dump with
role=alternate so the choice can be flipped in the plot script without
re-pulling anything. Rationale per cell is in ARMS below.

The LIBERO dual-view teacher (t71) has no plain probe directory under Oliver's
account; its numbers come from the slice-probe run on the same labelled export,
whose `all` slice is the identical 4,707,223-row pool with the identical
episode split (train 3,289,126 / val 476,711 / test 941,386) as every other
LIBERO arm, so it is directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "results" / "probe_perdim_raw"
OUT_CSV = REPO / "results" / "probe_perdim.csv"

# All three benchmarks use robosuite OSC_POSE deltas: translation (0-2),
# axis-angle rotation (3-5), gripper (6), normalised to [-1, 1].
DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

# Per-benchmark relabelling of action columns. On plain LIBERO the cameras
# were mounted swapped relative to the other two benchmarks, so what the probe
# file calls action_dim 0 is the paper's dy and action_dim 1 its dx (Oliver,
# 2026-09-17). The raw probe files are left as written; only `dim_name` in the
# dump is exchanged. `action_dim` keeps the raw column index for traceability.
DIM_RELABEL = {"libero": {0: 1, 1: 0}}

# (benchmark, method, role, arm_key, teacher_job, probe_job, ckpt_step,
#  valid_rows, source relative to the mirror, reader, note)
# probe_job is the Slurm ID of the probe stage as recorded in the brain;
# "" where the brain does not record it (the probe manifest carries no job ID).
ARMS = [
    # ---------------- LIBERO (plain), pool 4,707,223 valid rows, ckpt070000 ----
    ("libero", "ours_single", "canonical", "libero_side_45408556",
     "45408556", "45408561", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_45408556_ckpt070000", "standard",
     "lam1o5_side: sideview-only single-view teacher, seed 1003. The reference every "
     "plain-LIBERO baseline (CLAM, LAOF, DINO, villa-X, LAPA) was built from, so it is "
     "the matched single-view comparator."),
    ("libero", "ours_single", "alternate", "libero_frontsv_45351533",
     "45351533", "45351535", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_sv_45351533_ckpt070000", "standard",
     "lam1o5_sv: frontview-only single-view teacher, seed 1003 (SR 62.65)."),
    ("libero", "ours_multi", "canonical", "libero_dual_t71_44161039",
     "44161039", "", 70000, 4707223,
     "probe_analysis/libero_playdata_slices_lam1o5_lib90t71_44161039_ckpt070000", "slice_all",
     "lam1o5_lib90t71: front+side dual-view 3-embodiment teacher, seed 1000 (the paper's "
     "LIBERO xemb arm, SR 62.38). Slice probe, `all` slice = full pool, same split."),
    ("libero", "clam", "canonical", "libero_clam_45462812",
     "45462812", "45462814", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_clam_45462812_ckpt070000", "standard",
     "lam1o5_side_clam: sideview reference + CLAM action-grounding head (7-d mean target)."),
    ("libero", "laof", "canonical", "libero_flowbmse_45468967",
     "45468967", "45468969", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_flowbmse_45468967_ckpt070000", "standard",
     "lam1o5_flowbmse: sideview reference + LAOF flow decoder, bounded-MSE objective."),
    ("libero", "dino", "canonical", "libero_dino_45461745",
     "45461745", "45461747", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_dino_45461745_ckpt070000", "standard",
     "lam1o5_side_dino: sideview reference with frozen DINOv3-S/16 tokenizer (UniVLA-style)."),
    ("libero", "lapa", "canonical", "libero_lapaft_45477499",
     "45477499", "45477501", 70000, 4707223,
     "probe_analysis/rlfv_libero_xemb_v1_lapaft_side_45477499_ckpt070000", "standard",
     "lapaft_side: sideview reference initialised from LAPA's Open-X checkpoint and finetuned."),
    # ---------------- LIBERO-plus, pool 7,228,830 valid rows, ckpt100000 --------
    ("libero_plus", "ours_single", "canonical", "lp_single_run3_44001862",
     "44001862", "44014454", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_single_44001862_ckpt100000", "standard",
     "run3: single-view 2-frame teacher, seed 1000. The matched reference of the "
     "LIBERO-plus baseline arms."),
    ("libero_plus", "ours_multi", "canonical", "lp_dualnoaug_run11_44809827",
     "44809827", "44809829", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_dualnoaug_dualnoaug_44809827_ckpt100000", "standard",
     "run11: dual-view teacher, dual_view_augmentation=false, seed 1003. The paper's "
     "LIBERO-plus dual-view arm (SR 66.27 @60k)."),
    ("libero_plus", "ours_multi", "alternate", "lp_dual_run4_44056649",
     "44056649", "44064423", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_multi_44056649_ckpt100000", "standard",
     "run4: dual-view teacher with augmentation on, seed 1000 (SR 59.23, below run3)."),
    ("libero_plus", "clam", "canonical", "lp_clam7d_run10_44815478",
     "44815478", "44815480", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_clam7d_44815478_ckpt100000", "standard",
     "run10: CLAM with the 7-d mean-action target at beta=balanced/10 -- the recipe the "
     "MimicGen and plain-LIBERO CLAM arms use, so the three CLAM cells match."),
    ("libero_plus", "clam", "alternate", "lp_clam35d_run6_44424197",
     "44424197", "44424199", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_clam_44424197_ckpt100000", "standard",
     "run6: first CLAM arm, 35-d per-step chunk target, beta=5e-4 (SR 51.29)."),
    ("libero_plus", "laof", "canonical", "lp_flowbmse_run13_45415659",
     "45415659", "", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_flowbmse_flowbmse_45415659_ckpt100000", "standard",
     "run13 (lp_run13_flowbmse): LAOF flow decoder, bounded-MSE objective, seed 1000. "
     "Not yet written up in the brain; policy 45418140 evaluated on st-07."),
    ("libero_plus", "dino", "canonical", "lp_dino_run7_44422849",
     "44422849", "44422851", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_dino_dino_44422849_ckpt100000", "standard",
     "run7: DINOv3 tokenizer arm (UniVLA-style)."),
    ("libero_plus", "lapa", "canonical", "lp_lapaft_run12_45344565",
     "45344565", "", 100000, 7228830,
     "probe_analysis/libero_plus_xemb_v1_n2_lapaft_lapaft_45344565_ckpt100000", "standard",
     "run12: LAPA-initialised teacher + CPB. Probe exists although the paper's SR cell is TBD."),
    # ---------------- MimicGen, frontside pool, ckpt070000 unless noted ---------
    ("mimicgen", "ours_single", "canonical", "mg_single2f_armA_43611696",
     "43611696", "43630772", 70000, 3119561,
     "probe_analysis/gate_2f_single_ckpt070000", "standard",
     "arm A: 2-frame ([5]) single-view teacher, seed 1000. The reference every MimicGen "
     "baseline (CLAM, LAOF, DINO, LAPA) was built from; Table I row `1v {0,5}`."),
    ("mimicgen", "ours_single", "alternate", "mg_single4f_42572825",
     "42572825", "42733520", 70000, 0,
     "mimicgen_xemb_probes_20260630/single", "standard",
     "4-frame ([1,5,9]) single-view teacher, seed 1000; Table I row `1v {0,1,5,9}`. Probed "
     "on the grafted default-pool export (~3.08M rows), a different row set."),
    ("mimicgen", "ours_multi", "canonical", "mg_dual2f_armB_43820899",
     "43820899", "43820924", 50000, 3119561,
     "probe_analysis/gate_2f_multi_ckpt050000", "standard",
     "arm B: 2-frame dual-view teacher, seed 1003, trained 50k (ckpt050000); Table I row "
     "`2v {0,5}`. Same 2-frame recipe as arm A and the baselines."),
    ("mimicgen", "ours_multi", "alternate", "mg_dual4f_42742099",
     "42742099", "42813602", 70000, 0,
     "probe_analysis/gate_seed1003_multi_ckpt070000", "standard",
     "4-frame dual-view teacher, seed 1003, ckpt070000; Table I `ours` row (policy 43344870)."),
    ("mimicgen", "clam", "canonical", "mg_clam_run6mg_44697824",
     "44697824", "44712992", 70000, 3119561,
     "probe_analysis/gate_2f_clam_ckpt070000", "standard",
     "run6-mg: arm A + CLAM action-grounding head (7-d mean target, beta=8.75e-4)."),
    ("mimicgen", "laof", "canonical", "mg_flowbmse_run10b_45347502",
     "45347502", "", 70000, 3119561,
     "probe_analysis/gate_2f_fbmse", "standard",
     "run10b: arm A + LAOF flow decoder with the bounded-MSE objective (lambda 7.38e-2), the "
     "same objective as the LIBERO and LIBERO-plus flow arms. Not written up in the brain."),
    ("mimicgen", "laof", "alternate", "mg_flowsl1_run10mg_45069925",
     "45069925", "45101211", 70000, 3119561,
     "probe_analysis/gate_2f_flow", "standard",
     "run10-mg: arm A + LAOF flow decoder, unbounded smooth-L1 objective (lambda 1.51e-3); "
     "the arm the brain's mimicgen-flow-baseline page reports."),
    ("mimicgen", "dino", "canonical", "mg_dino_run7mg_44595589",
     "44595589", "", 70000, 3119561,
     "probe_analysis/gate_2f_dino_ckpt070000", "standard",
     "run7-mg: arm A with frozen DINOv3 tokenizer (UniVLA-style)."),
    ("mimicgen", "lapa", "canonical", "mg_lapaft_run11mg_45041442",
     "45041442", "", 70000, 3119561,
     "probe_analysis/gate_2f_lapaft_ckpt070000", "standard",
     "run11-mg: arm A initialised from LAPA's Open-X checkpoint and finetuned."),
]

# action_dim = raw column index in the probe file; dim_name = the paper's axis
# after DIM_RELABEL, so the two differ for LIBERO's first two dims.
FIELDS = ["benchmark", "method", "role", "arm_key", "teacher_job", "probe_job",
          "ckpt_step", "valid_rows", "probe", "target", "action_dim", "dim_name",
          "r2", "n_train", "n_test", "source_dir", "note"]


def raw_rows(path: Path, reader: str):
    """Yield (probe, target, action_dim, r2, avg_r2, n_train, n_test) from a raw file."""
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if reader == "slice_all" and row["slice"] != "all":
                continue
            if row["feature_set"] != "continuous" or row["split_mode"] != "episode":
                continue
            yield (row["probe_model"], row["target"], int(row["action_dim"]),
                   float(row["r2"]), float(row["avg_r2_for_target"]),
                   int(row["n_train"]), int(row["n_test"]))


def refresh_raw(mirror: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        key, src, reader = arm[3], arm[8], arm[9]
        fname = "slice_probe_r2.csv" if reader == "slice_all" else "action_probe_r2.csv"
        s = mirror / src / fname
        if not s.exists():
            raise SystemExit(f"missing {s}")
        shutil.copyfile(s, RAW_DIR / f"{key}.csv")
        print(f"  copied {s} -> results/probe_perdim_raw/{key}.csv")


def build() -> None:
    out = []
    for (bench, method, role, key, tjob, pjob, step, valid, src, reader, note) in ARMS:
        raw = RAW_DIR / f"{key}.csv"
        if not raw.exists():
            raise SystemExit(f"missing raw copy {raw}; run with --mirror first")
        per = {}
        for probe, target, dim, r2, avg, ntr, nte in raw_rows(raw, reader):
            per.setdefault((probe, target), {"avg": avg, "n": (ntr, nte)})[dim] = r2
        for (probe, target), d in sorted(per.items()):
            dims = sorted(k for k in d if isinstance(k, int))
            assert dims == list(range(7)), (key, probe, target, dims)
            mean = sum(d[k] for k in dims) / 7
            # avg_r2_for_target in the file must equal the mean over the 7 dims
            assert abs(mean - d["avg"]) < 1e-9, (key, probe, target, mean, d["avg"])
            base = dict(benchmark=bench, method=method, role=role, arm_key=key,
                        teacher_job=tjob, probe_job=pjob, ckpt_step=step,
                        valid_rows=valid or "", probe=probe, target=target,
                        n_train=d["n"][0], n_test=d["n"][1], source_dir=src, note=note)
            out.append({**base, "action_dim": -1, "dim_name": "mean", "r2": f"{mean:.6f}"})
            relabel = DIM_RELABEL.get(bench, {})
            for k in dims:
                out.append({**base, "action_dim": k, "dim_name": DIM_NAMES[relabel.get(k, k)],
                            "r2": f"{d[k]:.6f}"})
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"  wrote results/probe_perdim.csv ({len(out)} rows, {len(ARMS)} arms)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mirror", type=Path, default=None,
                   help="local mirror holding probe_analysis/ and "
                        "mimicgen_xemb_probes_20260630/ (see module docstring)")
    a = p.parse_args()
    if a.mirror is not None:
        refresh_raw(a.mirror)
    build()


if __name__ == "__main__":
    main()
