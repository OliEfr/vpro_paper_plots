"""Write results/probe_sr.csv -- probe R^2 next to downstream success rate, per arm.

One row per LAM-teacher arm that has BOTH a latent->action probe and a rollout
evaluation, on each of the three benchmarks. Built 2026-09-17 for the
probe-vs-SR correspondence figures (plot_probe_sr.py). Independent of
plot_probing.py / results/probing_*.csv.

PROBE SIDE. Headline mean R^2 (ridge alpha=1 and MLP [512,256], episode split,
`current_action` and `future_action_mean`) read from each probe directory's
`action_probe_r2_summary.csv` on MN5, mirrored as described in
extract_probe_perdim.py; the file for each arm is copied verbatim into
results/probe_sr_raw/<arm_key>.csv for provenance.

SR SIDE. Typed in below FROM THE EVALUATION OUTPUTS THEMSELVES, not from notes:
  LIBERO       total SR over 40 tasks x 100 episodes, policy ckpt 070000, eval
               seed 1000; mean over per-task `overall.pc_success` in every
               task dir's eval_info.json under
               <account>/runs_lerobot/outputs/eval_split/<rollout group>/
  LIBERO-plus  5-dim sweep, n = 1,942 variants, policy ckpt 060000, from
               rlfv_libero_plus/results/<run>_060000_sweep/summary.json
               (`total.success_pct`, `missing_shards == []` for every run used)
  MimicGen     Table-I 6-task cut (Coffee/Square/Stack x D0/D1, 6 x 50 eps),
               policy ckpt 60k, eval seed 0, computed from
               vpro_mimicgen_eval/results/eval_<policy>_*_60k.json; the
               PANDA10 10-task mean at the same checkpoint is kept as sr_alt.
`sr_paper` is the value printed in the paper's main results table (draft of
2026-09-17) where that row exists, so a mismatch is visible in the dump.
plot_probe_sr.py uses `sr_paper` and the table's rows by default.

Reference arm per benchmark (for paired deltas): the single-view teacher every
baseline on that benchmark was built from -- LIBERO `lam1o5_side`, LIBERO-plus
run3, MimicGen arm A.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "results" / "probe_sr_raw"
OUT_CSV = REPO / "results" / "probe_sr.csv"

# (benchmark, method, arm_key, label, in_paper_table, teacher_job, policy_job,
#  sr, sr_alt, sr_paper, sr_source, probe source dir relative to mirror, note)
ARMS = [
    # ------------------------------------------------------------ LIBERO
    ("libero", "ours_single", "libero_side_45408556", "Ours (single-view)", True,
     "45408556", "45408565", 53.52, None, 53.5,
     "eval_split/2026-09-05_tv_rollout_sqsh_45408565_..._ckpt070000_seed1000",
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_45408556_ckpt070000", "reference arm"),
    ("libero", "ours_single_front", "libero_frontsv_45351533", "Ours (single-view, frontview)", False,
     "45351533", "45351537", 62.65, None, None,
     "eval_split/2026-09-03_rollout_sqsh_45351537_..._ckpt070000",
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_sv_45351533_ckpt070000", "frontview-only teacher; not a paper row"),
    ("libero", "ours_multi", "libero_dual_t71_44161039", "Ours (multi-view)", True,
     "44161039", "44161096", 62.38, None, 62.3,
     "felix eval_split/2026-08-05_rollout_sqsh_44161096_..._ckpt070000",
     "probe_analysis/libero_playdata_slices_lam1o5_lib90t71_44161039_ckpt070000", "slice probe, `all` slice"),
    ("libero", "clam", "libero_clam_45462812", "CLAM-style", True,
     "45462812", "45462816", 57.20, None, 57.2,
     "eval_split/2026-09-06_clam_rollout_sqsh_45462816_..._ckpt070000_seed1000",
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_clam_45462812_ckpt070000", ""),
    ("libero", "laof", "libero_flowbmse_45468967", "LAOF-style", True,
     "45468967", "45468971", 58.98, None, 58.8,
     "eval_split/2026-09-07_tv_rollout_sqsh_45468971_..._ckpt070000_seed1000",
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_flowbmse_45468967_ckpt070000", "paper prints 58.8; eval gives 58.98"),
    ("libero", "dino", "libero_dino_45461745", "UniVLA-style (DINOv3)", True,
     "45461745", "45461749", 59.35, None, 59.3,
     "eval_split/2026-09-06_tv_rollout_sqsh_45461749_..._ckpt070000_seed1000",
     "probe_analysis/rlfv_libero_xemb_v1_lam1o5_side_dino_45461745_ckpt070000", ""),
    ("libero", "lapa", "libero_lapaft_45477499", "Ours (LAPA-pretrained)", True,
     "45477499", "45493220", 59.98, None, 59.9,
     "eval_split/2026-09-07_tvlapaft_rollout_sqsh_45493220_..._ckpt070000_seed1000",
     "probe_analysis/rlfv_libero_xemb_v1_lapaft_side_45477499_ckpt070000", "paper prints 59.9; eval gives 59.98"),
    ("libero", "villax_cont", "libero_villax_prevq", "villa-X (cont. latent)", True,
     "villa-X frozen", "45461495", 42.85, None, 43.8,
     "eval_split/2026-09-06_tvvx_rollout_sqsh_45461495_..._villax_prevq_..._ckpt070000_seed1000",
     "probe_analysis/libero_allemb_fourcam_fullfail_villax_side_2a50e03_labeled__prevq",
     "paper prints 43.8 for cont. and 42.8 for VQ; evals give 42.85 (cont.) and 43.12 (VQ)"),
    ("libero", "villax_vq", "libero_villax_vq", "villa-X (VQ latent)", True,
     "villa-X frozen", "45461515", 43.12, None, 42.8,
     "eval_split/2026-09-06_tvvx_rollout_sqsh_45461515_..._villax_vq_..._ckpt070000_seed1000",
     "probe_analysis/libero_allemb_fourcam_fullfail_villax_side_2a50e03_labeled__vq", "see cont. row"),
    # ------------------------------------------------------------ LIBERO-plus
    ("libero_plus", "ours_single", "lp_single_run3_44001862", "Ours (single-view)", True,
     "44001862", "44014455", 62.10, None, 62.1,
     "rlfv_libero_plus/results/lp_run3_single_44014455_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_single_44001862_ckpt100000", "reference arm"),
    ("libero_plus", "ours_multi", "lp_dualnoaug_run11_44809827", "Ours (multi-view)", True,
     "44809827", "44809830", 66.27, None, 66.7,
     "rlfv_libero_plus/results/lp_run11_dualnoaug_44809830_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_dualnoaug_dualnoaug_44809827_ckpt100000",
     "paper prints 66.7 = the 80k sweep (66.63); 60k sweep is 66.27"),
    ("libero_plus", "ours_multi_aug", "lp_dual_run4_44056649", "Ours (multi-view, aug on)", False,
     "44056649", "44064425", 60.14, None, None,
     "rlfv_libero_plus/results/lp_run4_multi_44064425_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_multi_44056649_ckpt100000", "run4; not a paper row"),
    ("libero_plus", "clam", "lp_clam7d_run10_44815478", "CLAM-style", True,
     "44815478", "44815481", 63.08, None, 63.1,
     "rlfv_libero_plus/results/lp_run10_clam7d_44815481_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_clam7d_44815478_ckpt100000", "run10, 7-d target"),
    ("libero_plus", "clam_35d", "lp_clam35d_run6_44424197", "CLAM-style (35-d target)", False,
     "44424197", "44424200", 51.29, None, None,
     "rlfv_libero_plus/results/lp_run6_clam_44424200_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_clam_44424197_ckpt100000", "run6; not a paper row"),
    ("libero_plus", "laof", "lp_flowbmse_run13_45415659", "LAOF-style", True,
     "45415659", "45418140", 60.45, None, 60.5,
     "rlfv_libero_plus/results/lp_run13_flowbmse_45418140_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_flowbmse_flowbmse_45415659_ckpt100000", ""),
    ("libero_plus", "dino", "lp_dino_run7_44422849", "UniVLA-style (DINOv3)", True,
     "44422849", "44451211", 60.87, None, 60.9,
     "rlfv_libero_plus/results/lp_run7_dino_44451211_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_dino_dino_44422849_ckpt100000", ""),
    ("libero_plus", "lapa", "lp_lapaft_run12_45344565", "Ours (LAPA-pretrained)", True,
     "45344565", "45354862", 67.87, None, None,
     "rlfv_libero_plus/results/lp_run12_lapaft_45354862_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_lapaft_lapaft_45344565_ckpt100000", "paper cell is TBD"),
    ("libero_plus", "scratch_cpb", "lp_scratchcpb_run13_45411217", "Ours (single-view, CPB control)", False,
     "45411217", "45411220", 64.06, None, None,
     "rlfv_libero_plus/results/lp_run13_scratchcpb_45411220_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_scratchcpb_scratchcpb_45411217_ckpt100000", "run13 scratch-CPB control; not a paper row"),
    ("libero_plus", "villax_vq", "lp_villax_vq_run5", "villa-X (VQ latent)", True,
     "villa-X frozen", "44223295", 45.11, None, 45.1,
     "rlfv_libero_plus/results/lp_run5_villax_44223295_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_villax", ""),
    ("libero_plus", "villax_cont", "lp_villax_prevq_run6", "villa-X (cont. latent)", True,
     "villa-X frozen", "44425958", 51.49, None, 51.5,
     "rlfv_libero_plus/results/lp_run6_villax_prevq_44425958_060000_sweep/summary.json",
     "probe_analysis/libero_plus_xemb_v1_n2_villax_prevq", ""),
    # ------------------------------------------------------------ MimicGen (6-task cut @60k; sr_alt = PANDA10 mean @60k)
    ("mimicgen", "ours_single", "mg_single2f_armA_43611696", "Ours (single-view)", True,
     "43611696", "43630801", 31.00, 19.00, 31.0,
     "vpro_mimicgen_eval/results/eval_43630801_mgxemb_run3_2f_fullft_60k.json",
     "probe_analysis/gate_2f_single_ckpt070000", "reference arm (arm A, 2-frame)"),
    ("mimicgen", "ours_single_4f", "mg_single4f_42572825", "Ours (single-view, 4-frame)", False,
     "42572825", "43408372", 30.33, 18.60, None,
     "vpro_mimicgen_eval/results/eval_43408372_mgxemb_run3_single_fullft_60k.json",
     "mimicgen_xemb_probes_20260630/single", "probed on the grafted default-pool export (~3.08M rows), other arms on 3,119,561"),
    ("mimicgen", "ours_multi", "mg_dual4f_42742099", "Ours (multi-view)", True,
     "42742099", "43344870", 38.67, 23.40, 38.7,
     "vpro_mimicgen_eval/results/eval_43344870_mgxemb_run4_fullft_60k.json",
     "probe_analysis/gate_seed1003_multi_ckpt070000", "4-frame dual, the paper's multi-view row"),
    ("mimicgen", "ours_multi_2f", "mg_dual2f_armB_43820899", "Ours (multi-view, 2-frame)", False,
     "43820899", "43836138", 35.00, 21.60, None,
     "vpro_mimicgen_eval/results/eval_43836138_mgxemb_run4_2f_fullft_60k.json",
     "probe_analysis/gate_2f_multi_ckpt050000", "arm B, teacher ckpt 050000; not a paper row"),
    ("mimicgen", "clam", "mg_clam_run6mg_44697824", "CLAM-style", True,
     "44697824", "44712994", 32.33, 19.60, 32.3,
     "vpro_mimicgen_eval/results/eval_44712994_mgxemb_run6_clam_2f_fullft_60k.json",
     "probe_analysis/gate_2f_clam_ckpt070000", ""),
    ("mimicgen", "laof", "mg_flowbmse_run10b_45347502", "LAOF-style", True,
     "45347502", "45357797", 33.33, 20.40, 33.4,
     "vpro_mimicgen_eval/results/eval_45357797_mgxemb_run10_bmse_2f_fullft_60k.json",
     "probe_analysis/gate_2f_fbmse", "bounded-MSE flow (run10b); paper prints 33.4, eval gives 33.33"),
    ("mimicgen", "laof_sl1", "mg_flowsl1_run10mg_45069925", "LAOF-style (smooth-L1)", False,
     "45069925", "45114086", 27.33, 16.40, None,
     "vpro_mimicgen_eval/results/eval_45114086_mgxemb_run10_flow_2f_fullft_60k.json",
     "probe_analysis/gate_2f_flow", "run10-mg, unbounded objective; not a paper row"),
    ("mimicgen", "dino", "mg_dino_run7mg_44595589", "UniVLA-style (DINOv3)", True,
     "44595589", "44628515", 28.67, 17.40, 28.7,
     "vpro_mimicgen_eval/results/eval_44628515_mgxemb_run7_dino_2f_fullft_60k.json",
     "probe_analysis/gate_2f_dino_ckpt070000", ""),
    ("mimicgen", "lapa", "mg_lapaft_run11mg_45041442", "Ours (LAPA-pretrained)", True,
     "45041442", "45055978", 30.33, 18.80, 30.4,
     "vpro_mimicgen_eval/results/eval_45055978_mgxemb_run11_lapaft_2f_policy_fullft_60k.json",
     "probe_analysis/gate_2f_lapaft_ckpt070000", "paper prints 30.4; eval gives 30.33"),
    ("mimicgen", "villax_cont", "mg_villax_prevq_run8mg", "villa-X (cont. latent)", True,
     "villa-X frozen", "44705485", 21.00, 12.60, 22.0,
     "vpro_mimicgen_eval/results/eval_44705485_mgxemb_run8_villax_prevq_2f_fullft_60k.json",
     "probe_analysis/gate_2f_villax_prevq", "paper prints 22.0 = the 80k eval; 60k gives 21.00"),
    ("mimicgen", "villax_vq", "mg_villax_vq_run8mg", "villa-X (VQ latent)", True,
     "villa-X frozen", "", None, None, 18.8,
     "no evaluation found on st-07 or in the brain (the VQ arm was not run on MimicGen); paper value only",
     "probe_analysis/gate_2f_villax_vq", "SR exists only as the paper-table value"),
]

REFERENCE = {"libero": "ours_single", "libero_plus": "ours_single", "mimicgen": "ours_single"}

FIELDS = ["benchmark", "method", "arm_key", "label", "in_paper_table", "is_reference",
          "teacher_job", "policy_job", "mlp_r2", "ridge_r2", "mlp_r2_future", "ridge_r2_future",
          "sr", "sr_alt", "sr_paper", "sr_source", "probe_source_dir", "note"]


def read_probe(path: Path) -> dict:
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["feature_set"] != "continuous" or r["split_mode"] != "episode":
                continue
            if "slice" in r and r["slice"] != "all":
                continue
            out[(r["probe_model"], r["target"])] = float(r["mean_r2"])
    return out


def refresh_raw(mirror: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        key, src = arm[2], arm[11]
        d = mirror / src
        s = d / "action_probe_r2_summary.csv"
        if not s.exists():
            s = d / "slice_probe_r2_summary.csv"
        if not s.exists():
            raise SystemExit(f"missing probe summary under {d}")
        shutil.copyfile(s, RAW_DIR / f"{key}.csv")
        print(f"  copied {s} -> results/probe_sr_raw/{key}.csv")


def build() -> None:
    rows = []
    for (bench, method, key, label, in_tab, tjob, pjob, sr, sr_alt, sr_paper,
         sr_src, psrc, note) in ARMS:
        raw = RAW_DIR / f"{key}.csv"
        if not raw.exists():
            raise SystemExit(f"missing raw copy {raw}; run with --mirror first")
        p = read_probe(raw)
        rows.append(dict(
            benchmark=bench, method=method, arm_key=key, label=label,
            in_paper_table=int(in_tab), is_reference=int(REFERENCE[bench] == method),
            teacher_job=tjob, policy_job=pjob,
            mlp_r2=f"{p[('mlp', 'current_action')]:.4f}",
            ridge_r2=f"{p[('ridge', 'current_action')]:.4f}",
            mlp_r2_future=f"{p[('mlp', 'future_action_mean')]:.4f}",
            ridge_r2_future=f"{p[('ridge', 'future_action_mean')]:.4f}",
            sr="" if sr is None else f"{sr:.2f}", sr_alt="" if sr_alt is None else f"{sr_alt:.2f}",
            sr_paper="" if sr_paper is None else f"{sr_paper:.1f}",
            sr_source=sr_src, probe_source_dir=psrc, note=note))
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote results/probe_sr.csv ({len(rows)} arms)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mirror", type=Path, default=None,
                   help="local mirror holding probe_analysis/ and mimicgen_xemb_probes_20260630/")
    a = p.parse_args()
    if a.mirror is not None:
        refresh_raw(a.mirror)
    build()


if __name__ == "__main__":
    main()
