"""Shared figure style for the manuscript panels.

White background, minimal text, four charts in a row.  Colour carries
information; no chart is a table, a diagram, or a text box.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------------------------- style

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

#: qualitative palette, colour-blind safe
BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#1b7837"
ORANGE = "#e08214"
PURPLE = "#762a83"
GREY = "#808080"
TEAL = "#01665e"
LIGHT = "#d9d9d9"

SEQ = cm.viridis
DIV = cm.RdBu_r


def panel(n=4, size=(15.5, 3.6), threed=()):
    """A row of ``n`` axes; indices in ``threed`` are 3-D."""
    fig = plt.figure(figsize=size)
    axes = []
    for i in range(n):
        if i in threed:
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            ax.set_facecolor("white")
            ax.xaxis.pane.set_facecolor("white")
            ax.yaxis.pane.set_facecolor("white")
            ax.zaxis.pane.set_facecolor("white")
            ax.xaxis.pane.set_edgecolor(LIGHT)
            ax.yaxis.pane.set_edgecolor(LIGHT)
            ax.zaxis.pane.set_edgecolor(LIGHT)
            ax.grid(True, color=LIGHT, linewidth=0.4)
        else:
            ax = fig.add_subplot(1, n, i + 1)
        axes.append(ax)
    return fig, axes


def tag(ax, letter, threed=False):
    """Panel letter, upper left, no other decoration."""
    if threed:
        ax.text2D(-0.06, 1.04, letter, transform=ax.transAxes,
                  fontsize=11, fontweight="bold", va="top")
    else:
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")


def save(fig, path):
    fig.subplots_adjust(wspace=0.32, left=0.04, right=0.985,
                        top=0.90, bottom=0.14)
    fig.savefig(path)
    plt.close(fig)
    return path
