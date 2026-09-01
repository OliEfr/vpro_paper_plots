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
