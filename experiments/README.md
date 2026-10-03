# Experiments

Scripts that **write** `results/`, kept here so a dump is traceable to the code
that produced it. They are deliberately **not** part of `build_figures.sh`: they
need the experiment machines, the raw latent exports, and heavy dependencies
(`umap-learn`, `scikit-learn`, `pyarrow`) that must stay out of the pinned
`vpro-plots` env — adding them there would move `matplotlib`/`freetype` and
rewrite every committed figure.

The split is the one `results/README.md` already describes: an experiment writes
a dump, a plot script renders it. Rebuilding a figure needs only the dump.

## `fit_umap_dumps.py`

Writes `results/umap_teachers.csv`, `results/umap_hardware.csv` and
`results/umap_decodability.csv` — the UMAP coordinates behind
`plot_umap_teachers.py` and `plot_umap_hardware.py`.

Runs on the experiments workstation `tueilsy-st-022`, where the labelled latent
exports live, in the env that produced the reference figures:

    /mnt/data/workspace/.conda/rlfv/bin/python experiments/fit_umap_dumps.py

Reads `/mnt/data/workspace/runs_root/runs_lerobot/latent_labels/` (LIBERO
teachers and the DK1 sweep) and writes CSVs next to itself; ~2 minutes, CPU only,
no GPU. It emits every checkpoint it fits — the committed CSVs are trimmed to
the ones the figures use and rounded to 2 decimals, which is well under a pixel
at print size.

Pinned choices, all shared with the reference analysis: seed 42, UMAP
`n_neighbors=30, min_dist=0.05`, euclidean; a 1 s frame stride (`frame_index %
20`) to kill within-episode autocorrelation; the fixed 1500-episode subset
(`subset300_eps_allemb.json`); and a per-group balanced sample, so a panel never
shows one robot or one source more densely than another.

One UMAP is fitted per model, jointly across that model's checkpoints — never
across models, whose latent spaces are trained separately and share no basis.

## `extract_xemb_realworld_transfer.py`

Writes the `results/xemb_realworld_transfer*` dumps and `results/frames_xemb_realworld/`
behind `plot_xemb_realworld_transfer.py` (human <-> robot latent transfer, DK1). Needs a
GPU, torchcodec/opencv/einops and the LAM package; on st-07 the `mg-latent` env:

    P=~/miniconda3/envs/mg-latent/bin/python
    $P experiments/extract_xemb_realworld_transfer.py search --stage <stage>   # ~3 min
    $P experiments/extract_xemb_realworld_transfer.py swap   --stage <stage>   # ~10 min
    $P experiments/extract_xemb_realworld_transfer.py dump   --stage <stage>   # ~1 min

`<stage>` holds `exports/sharedlam2/` (the full-dataset latent export).
`--dataset-root` (canonical DK1 front+side videos + meta) and `--ckpt` (sharedlam2
`pretrained_model`, copied from MN5) default to the st-07 copies. The LAM code is
`lerobot_policy_lam_plain_dino`, md5-identical to the p19 training snapshot; a
re-extraction of exported latents matches them at cos 0.994-1.000. The pairs in
`SELECTED` were picked by eye from the `search` output.

## `extract_xemb_latent_grid.py`

Writes the `results/xemb_latent_grid*` dumps and `results/frames_xemb_latent_grid/`
behind `plot_xemb_latent_grid.py` (one latent applied to robot and human frames). Same
machine, env and inputs as `extract_xemb_realworld_transfer.py`, whose loaders it imports:

    P=~/miniconda3/envs/mg-latent/bin/python
    $P experiments/extract_xemb_latent_grid.py search   # ~4 min, candidate scores
    $P experiments/extract_xemb_latent_grid.py stats    # ~1 min, the selected latents
    $P experiments/extract_xemb_latent_grid.py render   # ~1 min, the figure cells

`SELECTED` (one latent per direction) and `ROWS` (start frames with the gripper / hand in
view) were picked by eye from the top of the search.

## `make_xemb_videos.py`

Videos of both transfer results, from the same selections as the two figures
(`plot_xemb_realworld_transfer.py`, `plot_xemb_latent_grid.py`); all at half speed:

    ~/miniconda3/envs/mg-latent/bin/python experiments/make_xemb_videos.py   # ~1 min, GPU

- `figures/xemb_transfer_video.mp4` -- the one to show: title cards, the four robot /
  human pairs (real footage, latents filling in), then the latent-action grid.
- `figures/xemb_realworld_transfer.mp4`, `figures/xemb_latent_grid.mp4` -- the two parts.
- `figures/xemb_headline_{away,up_left,left}.mp4` -- one latent from a real demo (left)
  driving a robot frame and a human frame (LAM decoder output); `xemb_headline_pair_banana.mp4`
  -- one real robot / human pair with the nearest latent.
