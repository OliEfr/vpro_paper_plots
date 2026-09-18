"""Probe R^2 vs success rate with the METHODS discriminable -- colour = method.

    python plot_probe_sr_methods.py                  # figures/probe_sr_methods.pdf
    python plot_probe_sr_methods.py --axes delta     # figures/probe_sr_methods_delta.pdf
    python plot_probe_sr_methods.py --probe ridge    # single axis, ridge only
    python plot_probe_sr_methods.py --style science
    python plot_probe_sr_methods.py --mean add     # + a mean-over-benchmarks point per method
    python plot_probe_sr_methods.py --mean only    # just those means

Reads results/probe_sr.csv through plot_probe_sr.load(): the paper's main-table
rows with the SR printed in the table (pass --sr verified for the re-verified
numbers). One point per (method, benchmark).

ENCODING. Colour = method, fixed across every figure in this family: the four
teachers of plot_probe_perdim.py keep their PALETTE positions (ours single,
ours multi, CLAM, LAOF) and the remaining table rows take the rest of the
Okabe-Ito set. Marker shape = benchmark (style.MARKERS order: LIBERO circle,
LIBERO-Plus triangle, MimicGen square). Our own teachers carry the red edge
style.py reserves for the hero configuration. Ridge on the left, MLP on the
right, same y axis, so a method's horizontal move between the two panels is
how much of its decodability is nonlinear.

--axes absolute   x = R^2, y = SR (%). Benchmarks sit at different heights
                  (MimicGen ~30, LIBERO ~60); a grey dashed line per
                  benchmark is its least-squares fit, rho per benchmark is in
                  the corner.
--axes delta      x = R^2 - R^2(ours single-view), y = SR - SR(ours
                  single-view), per benchmark, so all three share one origin
                  and the sign quadrants mean the same thing everywhere.
--mean add|only   one extra point per method: its R^2 and SR averaged over
                  the three benchmarks (the paper table's Mean column),
                  drawn as a large X in the method colour. Only methods
                  present on all three benchmarks get one (LAPA has no
                  LIBERO-Plus cell). rho over the means is printed too.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

import plot_probe_sr as base

# Method order = legend order. Colours: first four from style.PALETTE by
# position (shared with plot_probe_perdim.py), then Okabe-Ito vermillion, sky
# blue and two greys for the external villa-X latents.
METHODS = [
    ("ours_single", "Ours (single-view)"),
    ("ours_multi", "Ours (multi-view)"),
    ("clam", "CLAM-style"),
    ("laof", "LAOF-style"),
    ("dino", "UniVLA-style (DINOv3)"),
    ("lapa", "Ours (LAPA-pretrained)"),
    ("villax_cont", "villa-X (cont. latent)"),
    ("villax_vq", "villa-X (VQ latent)"),
]
EXTRA_COLORS = ["#D55E00", "#56B4E9", "#7F7F7F", "#BDBDBD"]
OURS = {"ours_single", "ours_multi", "lapa"}
PROBES = {"ridge": ("ridge_r2", "ridge (linear)"), "mlp": ("mlp_r2", "MLP (nonlinear)")}


def _pt(style) -> float:
    """Figure text size: style.FIG_PT where defined, else the active rcParams size."""
    import matplotlib
    return float(getattr(style, "FIG_PT", matplotlib.rcParams["font.size"]))


def colours(style):
    return dict(zip([k for k, _ in METHODS], list(style.PALETTE[:4]) + EXTRA_COLORS))


MEAN_MARKER = "X"
MEAN_SIZE = 70


def method_means(df, xcol, ycol):
    """Per-method mean over benchmarks, only for methods present on all of them."""
    g = df.groupby("method")
    full = g.benchmark.nunique() == len(base.SUITES)
    m = g[[xcol, ycol]].mean()[full]
    return m


def draw(ax, df, pkey, style, axes_mode, mean_mode, note_loc="lower right", colors=None,
         edge_ours=None):
    col = colors or colours(style)
    red = style.MARKER_EDGE_OURS if edge_ours is None else edge_ours
    xcol = pkey if axes_mode == "absolute" else "d_" + pkey
    ycol = "sr" if axes_mode == "absolute" else "d_sr"
    if axes_mode == "delta":
        ax.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
        ax.axvline(0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
    notes = []
    for si, (skey, slabel) in enumerate(base.SUITES):
        sub = df[df.benchmark == skey]
        if mean_mode == "only":
            notes.append(rf"{slabel} $\rho$={base.spearman(sub[xcol], sub[ycol]):+.2f}")
            continue
        if axes_mode == "absolute" and len(sub) >= 3 and mean_mode == "none":
            b, a = np.polyfit(sub[xcol], sub[ycol], 1)
            xs = np.array([sub[xcol].min(), sub[xcol].max()])
            ax.plot(xs, a + b * xs, color=style.INK_MUTED, linewidth=0.7,
                    linestyle=style.LINE_OTHER, zorder=2)
        notes.append(rf"{slabel} $\rho$={base.spearman(sub[xcol], sub[ycol]):+.2f}")
        for mkey, _ in METHODS:
            r = sub[sub.method == mkey]
            if r.empty:
                continue
            ours = mkey in OURS
            ax.scatter(r[xcol], r[ycol], s=26, color=col[mkey], marker=style.MARKERS[si],
                       edgecolor=red if ours else style.MARKER_EDGE,
                       linewidth=0.7 if ours else 0.4, zorder=4 if ours else 3)
    if mean_mode != "none":
        m = method_means(df, xcol, ycol)
        for mkey, r in m.iterrows():
            ours = mkey in OURS
            ax.scatter(r[xcol], r[ycol], s=MEAN_SIZE, color=col[mkey], marker=MEAN_MARKER,
                       edgecolor=style.MARKER_EDGE_OURS if ours else style.MARKER_EDGE,
                       linewidth=0.8 if ours else 0.5, zorder=6)
        notes.append(rf"mean over benchmarks $\rho$={base.spearman(m[xcol], m[ycol]):+.2f} (n={len(m)})")
    if note_loc == "none":
        pass
    elif note_loc == "upper left":
        ax.text(0.015, 0.97, "\n".join(notes), transform=ax.transAxes, ha="left", va="top",
                color=style.INK, fontsize=_pt(style) - 0.5, linespacing=1.15)
    else:
        ax.text(0.985, 0.03, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                color=style.INK, fontsize=_pt(style) - 0.5, linespacing=1.15)
    ax.grid(True)
    ax.set_axisbelow(True)


def make_figure(df, style, probes, axes_mode, mean_mode, name):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    col = colours(style)
    n = len(probes)
    w = style.TEXT_WIDTH if n == 2 else style.COL_WIDTH
    h = w * (0.42 if n == 2 else 0.80)
    fig, axes = plt.subplots(1, n, figsize=(w, h), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], probes):
        pkey, plabel = PROBES[p]
        draw(ax, df, pkey, style, axes_mode, mean_mode)
        if axes_mode == "absolute":
            ax.set_xlabel(rf"probe $R^2$, {plabel}")
        else:
            ax.set_xlabel(rf"$\Delta$ probe $R^2$ vs. ours single-view, {plabel}")
    axes[0, 0].set_ylabel("success rate (%)" if axes_mode == "absolute"
                          else r"$\Delta$ success rate vs. ours single-view (pp)")

    method_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=col[k],
               markeredgecolor=style.MARKER_EDGE_OURS if k in OURS else style.MARKER_EDGE,
               markeredgewidth=0.7 if k in OURS else 0.4, label=lab) for k, lab in METHODS]
    bench_handles = [
        Line2D([], [], linestyle="none", marker=style.MARKERS[i], markersize=5,
               markerfacecolor="white", markeredgecolor=style.INK, markeredgewidth=0.6, label=lab)
        for i, (_, lab) in enumerate(base.SUITES)] if mean_mode != "only" else []
    if mean_mode != "none":
        bench_handles.append(Line2D([], [], linestyle="none", marker=MEAN_MARKER, markersize=8,
                                    markerfacecolor="white", markeredgecolor=style.INK,
                                    markeredgewidth=0.6, label="mean over benchmarks"))
    ncol_m = 4 if n == 2 else 2
    top = 0.80 if n == 2 else 0.72
    fig.legend(handles=method_handles, loc="upper center", ncol=ncol_m,
               bbox_to_anchor=(0.5, 1.005), frameon=False, handletextpad=0.4)
    fig.legend(handles=bench_handles, loc="upper center", ncol=len(bench_handles),
               bbox_to_anchor=(0.5, top + 0.075 if n == 2 else top + 0.09), frameon=False,
               handletextpad=0.4)
    fig.subplots_adjust(left=0.07 if n == 2 else 0.15, right=0.99, bottom=0.14 if n == 2 else 0.12,
                        top=top, wspace=0.07)
    return style.save(fig, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--axes", choices=["absolute", "delta"], default="absolute")
    p.add_argument("--probe", choices=["both", "ridge", "mlp"], default="both")
    p.add_argument("--sr", choices=["paper", "verified"], default="paper")
    p.add_argument("--mean", choices=["none", "add", "only"], default="none",
                   help="add or show only a mean-over-benchmarks point per method")
    a = p.parse_args()
    style = importlib.import_module("style" if a.style == "paper" else "style_science")
    style.apply_style()
    df = base.load("sr_paper" if a.sr == "paper" else "sr", all_arms=False)
    probes = ["ridge", "mlp"] if a.probe == "both" else [a.probe]
    name = "probe_sr_methods"
    name += "" if a.axes == "absolute" else "_delta"
    name += "" if a.probe == "both" else f"_{a.probe}"
    name += "" if a.mean == "none" else f"_mean{a.mean}"
    name += "" if a.sr == "paper" else "_verified"
    name += "" if a.style == "paper" else "_science"
    make_figure(df, style, probes, a.axes, a.mean, name)


if __name__ == "__main__":
    main()
