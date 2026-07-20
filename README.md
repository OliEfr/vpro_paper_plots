# vpro_paper_plots

Figures for the VPRO paper. Raw experiment numbers go in `results/`, plot
scripts turn them into PDFs in `figures/`, and the PDFs get included in the
paper by hand.

```
results/     raw dumps from experiments — append-only input, see results/README.md
figures/     generated output, committed so the paper's figures are versioned
style.py         shared styling: IEEEtran sizing, Computer Modern fonts, palette
style_science.py alternative styling: the SciencePlots look (see below)
plot_*.py        one script per figure family; its LaTeX \includegraphics
                 snippet lives at the top of the file, in the module docstring
```

## Usage

```sh
pip install -r requirements.txt
python plot_probing.py              # rebuilds figures/probing.pdf from every results/probing_*.csv
python plot_probing.py results/probing_mimicgen.csv --name probing_mimicgen
python plot_probing.py --style science   # SciencePlots look -> figures/probing_science.pdf
```

## Two styles

`--style` picks the look; both build at the same `\textwidth`, so either drops
into the paper the same way.

- **`paper`** (default, `style.py`) — reproduces the paper's own body text
  exactly: Computer Modern at 10pt, IEEEtran column widths, recessive grey axes.
  This is the one to ship.
- **`science`** (`style_science.py`) — the SciencePlots
  [garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) `science
  + ieee` aesthetic: STIX serif, thin boxed axes, inward ticks, the SciencePlots
  palette. Needs `pip install SciencePlots`. On a box without LaTeX it uses the
  `no-latex` fallback (STIX instead of true Computer Modern); see the note atop
  `style_science.py`.

Both are drop-in style modules exposing the same names (`PALETTE`, `MARKERS`,
`apply_style`, `save`, …), so a new plot script written against one gets the
other for free.

Each script prints a plain-text table of the numbers it plotted alongside
writing the figure. Read it — it is the fastest way to catch a stale dump, and
it is the accessible view of the same data.

## Including a figure

**The LaTeX snippet to paste is at the top of each plot file** — the float
environment, the width, and a starting caption, in the module docstring. That is
the single source of truth per figure; this README does not copy it, so it
cannot go stale. Read it with either:

```sh
python plot_probing.py --help     # prints the docstring
sed -n '1,40p' plot_probing.py    # or just open the file
```

The float environment is not a free choice: it follows from the width the script
built at. Anything sized to `TEXT_WIDTH` (7.140in) needs `figure*` and
`width=\textwidth`; anything sized to `COL_WIDTH` (3.487in) needs a plain
`figure` and `width=\columnwidth`. `probing.pdf` is the former, so it goes in a
`figure*`.

**Never rescale.** A `width=0.9\columnwidth` scales the text with it, and 10pt
figure text stops being 10pt on the page. If a figure needs to be smaller,
shrink its `figsize` and rebuild.

## Conventions

These exist so figures look like one set rather than four separate documents.
They live in `style.py`; change them there, not in individual scripts.

**Sizing.** Read out of `IEEEtran.cls` for this paper's actual
`\documentclass[10pt,journal]` — `\textwidth` 43pc = 516 TeX pt, `\columnsep`
1pc, so `\columnwidth` = 252 TeX pt. The conference and compsoc branches of the
class use different values; these are the journal ones.

**Font size is 8pt** (`FIG_PT`) — IEEEtran `\footnotesize`, the caption size, a
step below the 10pt body, which reads as subordinate the way figure text should.
Set `FIG_PT_TEX = 10` in `style.py` to match body instead. It is specified in
TeX points (1/72.27in) and converted to the PostScript points (1/72in)
matplotlib measures in — a bare `8` would render 0.37% large. Every piece of
figure text is this one size; if a row *looks* larger, suspect a tall glyph, not
the font. Computer Modern's math braces `\{ \}` render ~40% taller than digits
and made the legend read as oversized, which is why the marker labels use
brackets `[0,5]` and not the paper's set braces.

**Typeface matches too.** `main.tex` loads no font package, so the document
renders in Computer Modern; matplotlib ships CM, so figures use `cmr10` and the
`cm` mathtext set rather than a Times-like approximation. Fonts embed as
TrueType — IEEE PDF eXpress rejects Type 3, which is matplotlib's default.

**Never `bbox_inches="tight"`.** Tight cropping shrinks the canvas to fit its
content, so the saved PDF comes out a fraction off `figsize`, `width=\textwidth`
rescales it, and the point sizes above stop being true. Scripts reserve their
own margins with `subplots_adjust` instead. The cost is that clipping is now
possible — which is why step 5 below says to look at the output.

**Color.** `PALETTE` is a fixed-order Okabe-Ito subset, assigned by position and
never cycled. Checked with a CVD validator over *all* pairs, not just adjacent
ones — in a grouped scatter every series sits next to every other. Worst pair is
ΔE 18.7 under normal vision, ΔE 7.6 under deuteranopia.

**Shape and hatch are not decoration.** That 7.6 is only permissible while a
second, non-color channel also separates the series — and proceedings get
printed and photocopied in grayscale besides. `MARKERS` and `HATCHES` are that
channel. If you ever let hue carry identity alone, re-derive the palette first.
The marker set (`o ^ s X`) is one shape per visual family so the four read apart
at print size; every mark carries a near-black contour for definition, and the
`(ours)` method's mark a red one (`MARKER_EDGE_OURS`) so the hero config is
flagged in every suite colour at once.

**One channel per factor.** `probing.pdf` has two independent factors, so color
carries the eval suite and marker shape carries the LAM configuration, with a
separate legend for each. That lets a reader hold one fixed and scan the other.
Folding both into one channel would need suites × methods legend entries to say
what two short lists say directly.

**Numbers stay out of the figures.** Figures show shape and ordering; exact
values live in the text table each script prints, and in the paper's tables.

## Adding a figure

1. Dump the numbers to `results/<metric>_<suite>.csv`, document the schema in
   `results/README.md`.
2. Copy `plot_probing.py` as a starting point — it already handles suite
   globbing, method-column discovery, the text table view, and saving.
3. Import from `style.py`. Do not set rcParams or figure sizes locally.
4. Put the LaTeX embed snippet in the module docstring — float environment,
   width, and a starting caption — so whoever pastes it into the paper does not
   have to work out whether it is a `figure` or a `figure*`.
5. Run it, **open the PNG and look at it.** The palette is validated
   automatically; layout is not. Label collisions and overflow only show up by
   eye.
