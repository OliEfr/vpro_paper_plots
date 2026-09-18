"""Does the latent->action probe predict downstream success? Draft visualisations.

    python plot_probe_sr.py --variant scatter   # figures/probe_sr_scatter.pdf
    python plot_probe_sr.py --variant delta     # figures/probe_sr_delta.pdf
    python plot_probe_sr.py --variant rank      # figures/probe_sr_rank.pdf
    python plot_probe_sr.py --variant rho       # figures/probe_sr_rho.pdf
    python plot_probe_sr.py --variant all
    default: the paper's main-table rows with the SR printed in the table
    --sr verified --all-arms: every probed arm, SR recomputed from eval outputs

Reads results/probe_sr.csv (experiments/build_probe_sr.py): one row per LAM
teacher arm with probe R^2 (ridge = linear, MLP = nonlinear) and its rollout
success rate, both the paper-table value and the re-verified one. These are exploration
drafts for choosing a figure, not the figure -- variants scatter/rank use two
or three panels.

VARIANTS
  scatter  R^2 (x) vs SR (y), one panel per probe, colour = benchmark. SR
           scales differ per benchmark (MimicGen ~30, LIBERO ~60) so the
           legend carries the WITHIN-benchmark Spearman rho for each probe;
           the cloud itself mostly shows the benchmark offsets.
  delta    Paired deltas against each benchmark's single-view reference:
           x = R^2(arm) - R^2(ref), y = SR(arm) - SR(ref) in pp. One axis.
           Filled = ridge, hollow = MLP, same colour per benchmark, and a
           thin tie joins the two points of one arm so the horizontal shift
           from ridge to MLP is visible. Off-diagonal quadrants = the probe
           got the sign wrong; the counts are printed in the corners.
  rank     Bump chart per benchmark: each arm's rank under ridge R^2, MLP R^2
           and SR, joined by a line; crossings are misrankings. Colour = arm.
  rho      Summary bars: within-benchmark Spearman rho(probe, SR) for ridge
           and MLP per benchmark, plus the pooled paired-delta rho.

Spearman rho is computed by hand (average ranks, Pearson on ranks) because the
pinned figure env has no scipy -- deliberately, see experiments/README.md.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent / "results" / "probe_sr.csv"

SUITES = [("libero", "LIBERO"), ("libero_plus", "LIBERO-Plus"), ("mimicgen", "MimicGen")]
PROBES = [("ridge_r2", "ridge (linear)"), ("mlp_r2", "MLP (nonlinear)")]
OURS_METHODS = {"ours_single", "ours_multi", "ours_single_front", "ours_multi_aug",
                "ours_single_4f", "ours_multi_2f", "scratch_cpb"}
# One colour per method family, shared by every variant that colours by arm.
METHOD_COLORS = {
    "ours_single": "#0072B2", "ours_single_front": "#56B4E9", "ours_single_4f": "#56B4E9",
    "scratch_cpb": "#56B4E9",
    "ours_multi": "#009E73", "ours_multi_aug": "#66C2A5", "ours_multi_2f": "#66C2A5",
    "clam": "#E69F00", "clam_35d": "#F0C060",
    "laof": "#CC79A7", "laof_sl1": "#E0A8C8",
    "dino": "#D55E00", "lapa": "#8C564B",
    "villax_cont": "#7F7F7F", "villax_vq": "#BDBDBD",
}


def _pt(style) -> float:
    """Figure text size: style.FIG_PT where defined, else the active rcParams size."""
    import matplotlib
    return float(getattr(style, "FIG_PT", matplotlib.rcParams["font.size"]))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def load(sr_col: str, all_arms: bool) -> pd.DataFrame:
    df = pd.read_csv(CSV)
    if not all_arms:
        df = df[df.in_paper_table == 1]
    df = df[df[sr_col].notna()].copy()
    df["sr"] = df[sr_col]
    ref = df[df.is_reference == 1].set_index("benchmark")
    if len(ref) != len(SUITES):
        raise SystemExit("reference arm missing for a benchmark")
    for col in ["ridge_r2", "mlp_r2", "sr"]:
        df["d_" + col] = df[col] - df.benchmark.map(ref[col])
    return df


def rhos(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for skey, slabel in SUITES:
        sub = df[df.benchmark == skey]
        for pkey, plabel in PROBES:
            rows.append(dict(benchmark=slabel, probe=plabel, n=len(sub),
                             rho=spearman(sub[pkey], sub.sr)))
    pooled = df[df.is_reference == 0]
    for pkey, plabel in PROBES:
        rows.append(dict(benchmark="pooled deltas", probe=plabel, n=len(pooled),
                         rho=spearman(pooled["d_" + pkey], pooled.d_sr)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def fig_scatter(df, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w = style.TEXT_WIDTH
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.40), sharey=True)
    for ax, (pkey, plabel) in zip(axes, PROBES):
        for si, (skey, slabel) in enumerate(SUITES):
            sub = df[df.benchmark == skey]
            rho = spearman(sub[pkey], sub.sr)
            ax.scatter(sub[pkey], sub.sr, s=22, color=style.PALETTE[si], marker=style.MARKERS[si],
                       edgecolor=style.MARKER_EDGE, linewidth=0.4, zorder=3,
                       label=rf"{slabel}  ($\rho$={rho:+.2f}, n={len(sub)})")
            ours = sub[sub.method.isin(OURS_METHODS)]
            ax.scatter(ours[pkey], ours.sr, s=22, facecolor="none", marker=style.MARKERS[si],
                       edgecolor=style.MARKER_EDGE_OURS, linewidth=0.6, zorder=4)
            # per-benchmark least-squares line, only as a visual guide
            if len(sub) >= 3:
                b, a = np.polyfit(sub[pkey], sub.sr, 1)
                xs = np.array([sub[pkey].min(), sub[pkey].max()])
                ax.plot(xs, a + b * xs, color=style.PALETTE[si], linewidth=0.8,
                        linestyle=style.LINE_OTHER, zorder=2)
        ax.set_xlabel(rf"probe $R^2$, {plabel}")
        ax.grid(True)
        ax.legend(loc="upper left", handlelength=1.0)
    axes[0].set_ylabel("success rate (%)")
    fig.text(0.5, 0.96, "red outline = our teachers; dashed = per-benchmark least squares",
             ha="center", va="top", color=style.INK_MUTED)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.90, wspace=0.08)
    return style.save(fig, name)


def fig_delta(df, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = df[df.is_reference == 0]
    w = style.TEXT_WIDTH
    fig, ax = plt.subplots(figsize=(w, w * 0.42))
    ax.axhline(0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
    ax.axvline(0, color=style.INK_MUTED, linewidth=0.6, zorder=1)
    # off-diagonal quadrants (probe sign != SR sign) lightly shaded
    lim_x = max(abs(d.d_ridge_r2).max(), abs(d.d_mlp_r2).max()) * 1.12
    lim_y = abs(d.d_sr).max() * 1.15
    ax.fill_between([-lim_x, 0], 0, lim_y, color=style.GRID, alpha=0.45, zorder=0, linewidth=0)
    ax.fill_between([0, lim_x], -lim_y, 0, color=style.GRID, alpha=0.45, zorder=0, linewidth=0)
    for si, (skey, slabel) in enumerate(SUITES):
        sub = d[d.benchmark == skey]
        for _, r in sub.iterrows():
            ax.plot([r.d_ridge_r2, r.d_mlp_r2], [r.d_sr, r.d_sr], color=style.PALETTE[si],
                    linewidth=0.5, alpha=0.7, zorder=2)
        ax.scatter(sub.d_ridge_r2, sub.d_sr, s=24, color=style.PALETTE[si], marker=style.MARKERS[si],
                   edgecolor=style.MARKER_EDGE, linewidth=0.4, zorder=4, label=slabel)
        ax.scatter(sub.d_mlp_r2, sub.d_sr, s=24, facecolor="white", marker=style.MARKERS[si],
                   edgecolor=style.PALETTE[si], linewidth=0.9, zorder=3)
    # sign-agreement counts per probe, printed in the corners
    for pkey, plabel, y in [("d_ridge_r2", "ridge", 0.97), ("d_mlp_r2", "MLP", 0.90)]:
        agree = int(((d[pkey] > 0) == (d.d_sr > 0)).sum())
        rho = spearman(d[pkey], d.d_sr)
        ax.text(0.015, y, rf"{plabel}: sign agrees {agree}/{len(d)}, $\rho$={rho:+.2f}",
                transform=ax.transAxes, ha="left", va="top")
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_xlabel(r"$\Delta$ probe $R^2$ vs. single-view reference teacher")
    ax.set_ylabel(r"$\Delta$ success rate (pp)")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=style.INK_MUTED,
                       markeredgecolor=style.MARKER_EDGE, markersize=5, label="filled: ridge"),
                Line2D([], [], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor=style.INK_MUTED, markersize=5, label="hollow: MLP")]
    ax.legend(handles=handles, loc="lower right", ncol=1, handlelength=1.0)
    ax.text(0.985, 0.97, "shaded: probe and SR disagree in sign", transform=ax.transAxes,
            ha="right", va="top", color=style.INK_MUTED)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.98)
    return style.save(fig, name)


def fig_rank(df, style, name):
    import matplotlib.pyplot as plt

    w = style.TEXT_WIDTH
    fig, axes = plt.subplots(1, 3, figsize=(w, w * 0.42))
    cols = [("ridge_r2", r"ridge $R^2$"), ("mlp_r2", r"MLP $R^2$"), ("sr", "SR")]
    for ax, (skey, slabel) in zip(axes, SUITES):
        sub = df[df.benchmark == skey].copy()
        # unique ranks so tied arms never share a line; ties broken by SR
        for c, _ in cols:
            order = sub.sort_values([c, "sr"], ascending=False).index
            sub.loc[order, "rk_" + c] = np.arange(1, len(sub) + 1)
        for _, r in sub.iterrows():
            ys = [r["rk_" + c] for c, _ in cols]
            col = METHOD_COLORS.get(r.method, "#333333")
            ours = r.method in OURS_METHODS
            ax.plot(range(3), ys, color=col, linewidth=1.3 if ours else 0.9, zorder=3,
                    marker="o", markersize=3.5, markeredgecolor=style.MARKER_EDGE_OURS if ours else col,
                    markeredgewidth=0.6)
            ax.text(2.10, ys[-1], r.label, va="center", ha="left", fontsize=_pt(style) - 2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([lab for _, lab in cols])
        ax.set_xlim(-0.2, 2.3)
        n = len(sub)
        ax.set_ylim(n + 0.6, 0.4)
        ax.set_yticks(range(1, n + 1))
        ax.set_title(slabel, pad=3)
        ax.grid(True, axis="y")
        ax.tick_params(axis="x", length=0)
        for s in ("left",):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("rank (1 = best)")
    fig.subplots_adjust(left=0.045, right=0.865, bottom=0.10, top=0.90, wspace=1.15)
    return style.save(fig, name)


def fig_rho(df, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r = rhos(df)
    groups = [lab for _, lab in SUITES] + ["pooled deltas"]
    w = style.COL_WIDTH
    fig, ax = plt.subplots(figsize=(w, w * 0.70))
    x = np.arange(len(groups))
    bw = 0.36
    for pi, (pkey, plabel) in enumerate(PROBES):
        vals = [r[(r.benchmark == g) & (r.probe == plabel)].rho.iloc[0] for g in groups]
        ns = [r[(r.benchmark == g) & (r.probe == plabel)].n.iloc[0] for g in groups]
        xb = x + (pi - 0.5) * bw
        ax.bar(xb, vals, width=bw, color=style.PALETTE[pi], hatch=style.HATCHES[pi],
               edgecolor=style.MARKER_EDGE, linewidth=0.35, zorder=3, label=plabel)
        for xi, v, n in zip(xb, vals, ns):
            ax.text(xi, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=_pt(style) - 1)
    ax.axhline(0, color=style.INK_MUTED, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={r[r.benchmark == g].n.iloc[0]})" for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r"Spearman $\rho$(probe $R^2$, SR)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.20, top=0.97)
    return style.save(fig, name)


VARIANTS = {"scatter": fig_scatter, "delta": fig_delta, "rank": fig_rank, "rho": fig_rho}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(VARIANTS) + ["all"], default="all")
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--sr", choices=["paper", "verified"], default="paper",
                   help="paper: SR as printed in the main table (default); "
                        "verified: SR recomputed from the eval outputs")
    p.add_argument("--all-arms", action="store_true",
                   help="include arms that are not rows of the main table")
    a = p.parse_args()
    style = importlib.import_module("style" if a.style == "paper" else "style_science")
    style.apply_style()
    df = load("sr_paper" if a.sr == "paper" else "sr", a.all_arms)
    print(df[["benchmark", "label", "ridge_r2", "mlp_r2", "sr"]].to_string(index=False))
    print(rhos(df).round(3).to_string(index=False))
    suffix = (("_verified" if a.sr == "verified" else "") + ("_allarms" if a.all_arms else "")
              + ("_science" if a.style == "science" else ""))
    for v in (sorted(VARIANTS) if a.variant == "all" else [a.variant]):
        VARIANTS[v](df, style, f"probe_sr_{v}{suffix}")


if __name__ == "__main__":
    main()
