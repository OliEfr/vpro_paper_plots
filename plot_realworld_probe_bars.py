r"""The probe panel of plot_realworld_probe_decoded.py (its original left side) as a stand-alone single-column figure.

Per-dimension MLP(512,256) probe R^2 for decoding the 5-frame end-effector motion state[t+5]-state[t] from the
frozen 8-D latent, single-view (side) vs multi-view (front + side) LAM, held-out robot episodes on hardware
(results/probe_perdim_realworld.csv, target state_delta_h5). The bars are drawn by the original figure's own
probe_panel(), so they are identical; the figure height is 70 % of the original two-column figure's height.
Emits figures/realworld_probe_bars[_science].{pdf,png}.

LaTeX:

    \begin{figure}[t]
      \centering
      \includegraphics[width=0.62\columnwidth]{figures/realworld_probe_bars_science.pdf}
      \caption{Per-dimension action probing for held-out episodes on hardware: the second view raises
        $R^2$ in $\Delta y$ from $0.45$ to $0.88$.}
      \label{fig:realworld_probe_bars}
    \end{figure}

Usage:
    python plot_realworld_probe_bars.py [--style paper|science]
"""
import argparse

import style
import plot_probe_combo as combo
import plot_realworld_probe_decoded as orig
from plot_probe_perdim_realworld import load

ORIG_HEIGHT_IN = 0.236 + 1.08 + 0.24     # BOT_IN + ROWS_IN + TOP_IN of plot_realworld_probe_decoded.py
HEIGHT_IN = 0.7 * 0.9 * ORIG_HEIGHT_IN    # 30 % lower, then another 10 % (user requests)
WIDTH_FRAC = 0.62                         # of \columnwidth: the caption sits to the right of the figure
LEGEND_LABELS = {"ours_single": "single-view", "ours_multi": "multi-view"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    orig.style = style                     # probe_panel() reads the style module of its own file
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    r2 = load("state_delta_h5")
    w, h = WIDTH_FRAC * style.COL_WIDTH, HEIGHT_IN
    fs = plt.rcParams["font.size"]
    fig = plt.figure(figsize=(w, h))
    # full box on all four sides (user request); the legend therefore sits ABOVE the axes, not inside them
    ax = fig.add_axes([0.30 / w, 0.19 / h, 1 - 0.32 / w, 1 - 0.36 / h])
    orig.probe_panel(ax, r2)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_ylabel("$R^2$", labelpad=1)
    ax.tick_params(axis="x", labelsize=fs - 1.5, pad=1.5)
    ax.tick_params(axis="y", labelsize=fs - 1.5, pad=1.5)
    handles = [Patch(facecolor=combo.COLORS[m], edgecolor=style.MARKER_EDGE, linewidth=0.5, label=LEGEND_LABELS[m])
               for m, _ in orig.LEGEND]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=fs - 1.5, handlelength=0.9, handleheight=0.7, columnspacing=0.8, handletextpad=0.3,
              borderaxespad=0.1)
    for (gkey, _) in orig.GROUPS["state_delta_h5"]:
        print(f"{gkey:8s} " + "  ".join(f"{m}={r2[('mlp', m, gkey)]:.3f}" for m, _ in orig.METHODS))
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"realworld_probe_bars{suffix}")


if __name__ == "__main__":
    main()
