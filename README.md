# vpro_paper_plots

Figures for the VPRO paper. Raw experiment numbers go in `results/`, plot
scripts turn them into PDFs in `figures/`, and the PDFs get included in the
paper by hand.

```
results/     raw dumps from experiments — append-only input, see results/README.md
figures/     generated output, committed so the paper's figures are versioned
style.py     shared IEEEtran figure sizing, fonts, palette
plot_*.py    one script per figure family
```

## Usage

```sh
pip install -r requirements.txt
python plot_probing.py              # rebuilds every results/probing_*.csv
python plot_probing.py results/probing_mimicgen.csv
python plot_probing.py --width text # force \textwidth instead of auto
```

Each script prints a plain-text table of the numbers it plotted alongside
writing the figure. Read it — it is the fastest way to catch a stale dump, and
it is the accessible view of the same data.

## Including a figure

Figures are sized to land in LaTeX at 1:1, so do not rescale them — a
`width=0.9\columnwidth` would shrink the fonts out of spec.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/probing_mimicgen.pdf}
  \caption{...}
  \label{fig:probing-mimicgen}
\end{figure}
```

Use `figure*` with `width=\textwidth` for anything built at `--width text`.

## Conventions

These exist so figures look like one set rather than four separate documents.
They live in `style.py`; change them there, not in individual scripts.

**Sizing.** `COL_WIDTH = 3.5in` (`\columnwidth`), `TEXT_WIDTH = 7.16in`
(`\textwidth`). Building at the exact final width is what keeps 8pt figure text
actually 8pt on the page.

**Fonts.** Serif, to sit next to IEEEtran's Times without looking imported.
Body text is 10pt, so figures use 8pt and 7pt. PDFs embed Type 42 fonts —
IEEE PDF eXpress rejects Type 3, which is matplotlib's default.

**Color.** `PALETTE` is a fixed-order Okabe-Ito subset, assigned by position and
never cycled. It was checked with a CVD validator: worst adjacent pair is
ΔE 11.4 under protanopia. Do not reorder it without re-checking — the ordering
is what makes the adjacent pairs separable.

**Grayscale.** Proceedings get printed and photocopied in black and white, so
every series also carries a hatch pattern from `HATCHES`. Hue is never the only
thing distinguishing two bars.

**Labels.** Two of the four palette steps fall below 3:1 contrast against white,
so bars carry direct value labels; the numbers do the work the fill can't.

## Adding a figure

1. Dump the numbers to `results/<metric>_<suite>.csv`, document the schema in
   `results/README.md`.
2. Copy `plot_probing.py` as a starting point — it already handles suite
   globbing, method-column discovery, the text table view, and saving.
3. Import from `style.py`. Do not set rcParams or figure sizes locally.
4. Run it, **open the PNG and look at it.** The palette is validated
   automatically; layout is not. Label collisions and overflow only show up by
   eye.
