# vpro_paper_plots

Figures for the VPRO paper — an IEEE T-RL (Transactions on Robot Learning)
submission, so `\documentclass[10pt,journal]{IEEEtran}` and a two-column body by
default. That is why a figure here is single-column unless it is deliberately
built to span both, and why every width in `style.py` is derived rather than
picked.

Raw experiment numbers go in `results/`, plot scripts turn them into PDFs in
`figures/`, and the PDFs get included in the paper by hand.

```
results/     raw dumps from experiments — append-only input, see results/README.md
figures/     generated output, committed so the paper's figures are versioned
style.py     shared styling; style_science.py is the alternative look
plot_*.py    one script per figure family
tasks.py     display names for the real-world tasks, shared by both of its figures
```

## Usage

```sh
pip install -r requirements.txt
python plot_probing.py         # figures/probing.pdf, from every results/probing_*.csv
python plot_realworld.py       # figures/realworld_scaling.pdf
python plot_realworld_bars.py  # figures/realworld_alltasks.pdf
python plot_libero_radar.py    # figures/libero_radar_{h,nonh}.pdf, one per split
python plot_libero_plus_radar.py  # figures/libero_plus_radar.pdf
python plot_radar_row.py       # figures/radar_row.pdf, all three radars in one figure*
```

`--style science` swaps `style.py` for `style_science.py` — the
[SciencePlots](https://github.com/garrettj403/SciencePlots) look, which needs
`pip install SciencePlots` and falls back to STIX on a box without LaTeX. The
default `paper` style reproduces the document's own body text; that is the one
to ship.

**Always regenerate both styles.** A script writes `<name>.pdf` under `paper`
and `<name>_science.pdf` under `science`, so one run updates only half of what
is committed and the two versions of a figure silently start showing different
numbers. Whenever a `results/*.csv` or a plot script changes, run the script
twice and commit both outputs:

```sh
python plot_probing.py && python plot_probing.py --style science
```

Each script also prints a plain-text table of the numbers it plotted. Read it —
it is the fastest way to catch a stale dump, and it is the accessible view of
the same data.

## Including a figure

**The LaTeX snippet to paste is in each plot script's module docstring** — float
environment, width, and a starting caption, sitting next to the code that fixes
the width so it cannot go stale. `python plot_probing.py --help` prints it.

The float environment is not a free choice: it follows from the width the script
built at. Anything sized to `TEXT_WIDTH` (7.140in) needs `figure*` and
`width=\textwidth`; anything sized to `COL_WIDTH` (3.487in) needs a plain
`figure` and `width=\columnwidth`. `probing.pdf` and `radar_row.pdf` are the
former, the real-world figures and the three single-column radars are the
latter.

**Never rescale.** A `width=0.9\columnwidth` scales the text with it, and 8pt
figure text stops being 8pt on the page. If a figure needs to be smaller, shrink
its `figsize` and rebuild.

## Conventions

These exist so figures look like one set rather than four separate documents.
They live in `style.py`, each derived in a comment there — change them in that
file, not in individual scripts.

**Sizing and type** come from `IEEEtran.cls` for this paper's actual
`\documentclass[10pt,journal]`; the conference and compsoc branches of the class
use different values. Figure text is 8pt — IEEEtran `\footnotesize`, a step below
the body, which reads as subordinate the way figure text should — set in the
Computer Modern the document itself renders in. Fonts embed as TrueType, because
IEEE PDF eXpress rejects the Type 3 that matplotlib defaults to.

The three single-column radars are set at 10pt instead, to match the body text,
via a local `RADAR_PT` override in each of `plot_libero_radar.py` and
`plot_libero_plus_radar.py`. That is a deliberate exception and the paper
carries two figure text sizes because of it. If the other figures should follow,
delete both overrides and set `FIG_PT_TEX = 10` here — the knob exists for
exactly that, and it keeps the set consistent.

`plot_radar_row.py` does **not** override. It packs the same three radars into a
`figure*`, where a panel is only ~2.4in wide and 10pt labels would swamp the
circle they belong to, so it stays at the shared 8pt — matching `probing.pdf`,
the repo's other two-column figure. So the same radar is 10pt on its own and 8pt
in the row, which is correct: both land on the page at the size the width they
were built for implies.

**Never `bbox_inches="tight"`.** Tight cropping shrinks the canvas to fit its
content, so the PDF comes out a fraction off `figsize`, `width=\textwidth`
rescales it, and the point size above stops being true. Scripts reserve their own
margins with `subplots_adjust` instead. The cost is that clipping is now
possible — which is why the last step below is to look at the output.

**Hue never carries identity alone.** `PALETTE` is a fixed-order Okabe-Ito
subset, assigned by position and checked over *all* pairs rather than adjacent
ones — in a grouped plot every series sits next to every other. Its worst pair
sits at the accessible floor under deuteranopia, which is permissible only while
a second channel — `MARKERS`, `HATCHES`, or `LINE_OURS`/`LINE_OTHER` where marks
are joined — separates the same series. Proceedings still get photocopied in
grayscale besides. Let hue carry identity alone and the palette has to be
re-derived first.

The radars are the one deliberate exception. They keep the red/blue/green the
LIBERO radars were already drawn in before this repo generated them, so the
paper does not show two colour schemes for the same three policies — red/green
is a dichromat collision that `PALETTE`'s derivation does not cover. They stay
readable only because the second and third channels are doing the work there:
each polygon has its own dash pattern *and* its own marker shape. Those figures
are the reason the rule is written as "hue never carries identity **alone**"
rather than "always use `PALETTE`"; drop their dashes or their markers and they
have to move to `PALETTE` first.

`plot_libero_radar.py` and `plot_libero_plus_radar.py` are **independent
copies**, not a shared module — different experiments, different axes, different
captions. What they must keep in common is that the palette, the dashes and the
markers are assigned by column position, so an arm sitting in the same column
position looks the same in both figures. Reorder one dump's `_sr` columns and
the same policy changes appearance between two figures in the same paper. A
layout fix in one script has to be made in the other by hand.

`plot_radar_row.py` is a **consumer** of both, not a third copy. It imports
their spoke orders, labels, titles and palette so a rename follows into the row
figure instead of the two versions of a panel disagreeing; only the row layout
lives in that file. It draws one legend for all three panels, which is only
valid while the three dumps carry the same arms in the same order — it checks
that at build time and refuses rather than mislabelling two thirds of the
figure.

**One channel per factor.** Where a figure crosses two factors, each gets its own
channel and its own legend, so a reader can hold one fixed and scan the other.
Folding both into one channel needs *n*×*m* legend entries to say what two short
lists say directly.

**Numbers stay out of the figures.** Figures show shape and ordering; exact
values live in the text table each script prints, and in the paper's tables.

## Adding a figure

1. Dump the numbers to `results/<name>.csv` and document the schema in
   `results/README.md`. The stem names the figure the script writes.
2. Copy the closest `plot_*.py` and import from `style.py`. Do not set rcParams
   or figure sizes locally.
3. Put the LaTeX embed snippet in the module docstring — float environment,
   width, and a starting caption — so whoever pastes it into the paper does not
   have to work out whether it is a `figure` or a `figure*`.
4. Run it, **open the PNG and look at it.** The palette is validated
   automatically; layout is not. Label collisions and overflow only show up by
   eye.
