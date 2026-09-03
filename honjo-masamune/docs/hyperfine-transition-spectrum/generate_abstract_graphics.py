"""
generate_abstract_graphics.py
=============================

Two submission graphics for the four-route convergence paper.

  graphical_abstract.png   MDPI portrait format (within the required
                           560x1100 min / 2800x5500 max, height x width).
                           Labels only, no prose blocks, no bullet lists.
  graphical_summary.png    landscape overview for slides / README.

Neither reproduces a figure from the paper, which MDPI forbids: both are
newly composed from results/ladder_routes.json.

Layout is done with explicit GridSpec rows rather than hand-placed
coordinates, so nothing overflows the canvas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
FIG = BASE / "figures"

D = json.loads((RES / "ladder_routes.json").read_text(encoding="utf-8"))
RIT = D["ritz_additivity"]["measured"]
HF = D["inert_and_hyperfine"]["measured"]
CL = D["closed_molecular"]["measured"]
ROWS = D["four_route"]["measured"]["rows"]

plt.rcParams["font.family"] = "DejaVu Sans"

C_EST = "#a855f7"
C_INS = "#22c55e"
C_LAD = "#f97316"
C_CAT = "#3b82f6"
C_HF = "#14b8a6"
C_BAD = "#ef4444"
C_OK = "#16a34a"
C_INK = "#0f172a"
C_MUTE = "#64748b"

ROUTES = [
    ("ESTABLISHED", "Schrodinger eigenvalues", C_EST),
    ("INSTRUMENT", "partition address resolved physically", C_INS),
    ("CATALOGUE", "minimum cut at rest", C_CAT),
    ("LADDER", "carrier deleted, rungs carry powers", C_LAD),
]

LETTERS = [("$n$", True), (r"$\ell$", True), ("$m$", True), ("$s$", False)]


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_routes(ax, fs_title, fs_sub):
    """The four route cards, stacked, drawn in axes coordinates."""
    _blank(ax)
    n = len(ROUTES)
    h = 1.0 / n
    for i, (nm, sub, cc) in enumerate(ROUTES):
        y = 1.0 - (i + 1) * h
        pad = h * 0.11
        ax.add_patch(FancyBboxPatch(
            (0.0, y + pad), 1.0, h - 2 * pad,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=cc, edgecolor=cc, linewidth=1.4, alpha=0.13,
            transform=ax.transAxes, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (0.0, y + pad), 0.016, h - 2 * pad,
            boxstyle="square,pad=0", facecolor=cc, edgecolor="none",
            transform=ax.transAxes, zorder=3))
        ax.text(0.045, y + h * 0.66, nm, fontsize=fs_title,
                fontweight="bold", color=C_INK, va="center")
        ax.text(0.045, y + h * 0.26, sub, fontsize=fs_sub, color=C_MUTE,
                va="center")


def draw_word(ax, fs_letter, fs_tag):
    """The (n, l, m, s) word with the parity letter highlighted."""
    _blank(ax)
    k = len(LETTERS)
    w = 1.0 / k
    for i, (L, fixed) in enumerate(LETTERS):
        cx = i * w
        fc = "#e2e8f0" if fixed else C_HF
        ec = "#94a3b8" if fixed else C_INK
        ax.add_patch(FancyBboxPatch(
            (cx + w * 0.10, 0.34), w * 0.80, 0.52,
            boxstyle="round,pad=0,rounding_size=0.05",
            facecolor=fc, edgecolor=ec, linewidth=2.2 if not fixed else 1.1,
            transform=ax.transAxes, zorder=2))
        ax.text(cx + w * 0.5, 0.60, L, ha="center", va="center",
                fontsize=fs_letter, color="#94a3b8" if fixed else C_INK)
        ax.text(cx + w * 0.5, 0.16, "fixed" if fixed else "rewritten",
                ha="center", va="center", fontsize=fs_tag,
                color="#94a3b8" if fixed else C_INK,
                fontweight="normal" if fixed else "bold")


def plot_ppm(ax, fs):
    """Residual from measurement, per route, in ppm."""
    x = np.arange(len(ROWS))
    for off, key, cc in [(-1.5, "established", C_EST),
                         (-0.5, "instrument", C_INS),
                         (0.5, "catalogue", C_CAT),
                         (1.5, "ladder", C_LAD)]:
        vals = [1e6 * (r[key + "_cm"] - r["nu_measured_cm"])
                / r["nu_measured_cm"] for r in ROWS]
        ax.bar(x + off * 0.2, vals, 0.2, color=cc, edgecolor="black",
               linewidth=0.4, zorder=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["line"] for r in ROWS], fontsize=fs)
    ax.set_ylabel("ppm from measurement", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.set_ylim(-13.5, 2.2)
    ax.grid(True, axis="y", alpha=0.3)
    _despine(ax)
    ax.text(0.985, 0.90, "ladder bar = 0 by construction",
            transform=ax.transAxes, ha="right", fontsize=fs - 1.5,
            color=C_MUTE, style="italic")


def plot_ritz(ax, fs, legend_loc="lower left", xlabel=True):
    ns = [t["n"] for t in RIT["triples_mixed_convention"]]
    mixed = [t["relative_residual"] for t in RIT["triples_mixed_convention"]]
    vac = [t["relative_residual"] for t in RIT["triples_vacuum"]]
    ax.plot(ns, mixed, "o-", color=C_BAD, ms=8, lw=2.2, label="as tabulated")
    ax.plot(ns, vac, "s-", color=C_OK, ms=8, lw=2.2,
            label="air $\\rightarrow$ vacuum")
    ax.axhline(RIT["tabulation_rounding_budget"], color="k", ls=":", lw=1.6,
               label="rounding budget")
    ax.set_yscale("log")
    ax.set_ylim(1.2e-8, 2e-3)
    ax.set_xticks(ns)
    if xlabel:
        ax.set_xlabel("upper level $n$", fontsize=fs)
    ax.set_ylabel("Ritz residual", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.legend(fontsize=fs - 2, loc=legend_loc, framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")
    _despine(ax)


def plot_power(ax, fs):
    cats = ["forbidden\nrung", "21 cm\nparity rung", "smallest\nallowed"]
    vals = [1e-9, HF["hyperfine_power"], HF["smallest_allowed_power"]]
    labs = ["0 exactly", "4.32 $\\times$ 10$^{-7}$", "0.306"]
    bars = ax.bar(cats, vals, color=["#cbd5e1", C_HF, "#7c3aed"],
                  edgecolor="black", linewidth=0.6, width=0.55, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(3e-10, 30)
    ax.set_ylabel(r"rung power $\pi$", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.grid(True, axis="y", alpha=0.3)
    _despine(ax)
    for b, v, lab in zip(bars, vals, labs):
        ax.text(b.get_x() + b.get_width() / 2, v * 2.6, lab, ha="center",
                fontsize=fs - 1, fontweight="bold")


# =========================================================================
# Graphical abstract -- portrait, MDPI
# =========================================================================

def graphical_abstract():
    fig = plt.figure(figsize=(11.0, 20.0))
    gs = GridSpec(
        13, 1, figure=fig,
        height_ratios=[1.05, 2.55, 0.42, 0.52, 1.55, 0.52, 1.75,
                       0.40, 0.42, 0.62, 1.60, 0.46, 0.30],
        left=0.115, right=0.955, top=0.975, bottom=0.022, hspace=0.30)

    # 0: title
    ax = fig.add_subplot(gs[0]); _blank(ax)
    ax.text(0.5, 0.74, "FOUR ROUTES TO A SPECTRAL LINE", ha="center",
            va="center", fontsize=25.5, fontweight="bold", color=C_INK)
    ax.text(0.5, 0.40, "H, H$_2$, H$_2$O   ·   70 lines   ·   NIST / HITRAN",
            ha="center", va="center", fontsize=15, color=C_MUTE)
    ax.plot([0.03, 0.97], [0.13, 0.13], color=C_INK, lw=1.8)

    # 1: the four routes
    draw_routes(fig.add_subplot(gs[1]), 17.5, 12.5)

    # 2: shared alphabet
    ax = fig.add_subplot(gs[2]); _blank(ax)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.10), 1.0, 0.80,
        boxstyle="round,pad=0,rounding_size=0.03", facecolor="#f1f5f9",
        edgecolor=C_INK, linewidth=1.6, transform=ax.transAxes))
    ax.text(0.5, 0.50, r"shared alphabet   $C(n)=2n^{2}$   —   nothing else",
            ha="center", va="center", fontsize=15.5, color=C_INK)

    # 3: heading 1
    ax = fig.add_subplot(gs[3]); _blank(ax)
    ax.text(0.5, 0.72, "THREE ROUTES COINCIDE EXACTLY", ha="center",
            va="center", fontsize=17.5, fontweight="bold", color=C_INK)
    ax.text(0.5, 0.24, "one closed form derived three ways", ha="center",
            va="center", fontsize=12.5, color=C_MUTE, style="italic")

    # 4: ppm plot
    plot_ppm(fig.add_subplot(gs[4]), 12)

    # 5: heading 2
    ax = fig.add_subplot(gs[5]); _blank(ax)
    ax.text(0.5, 0.72, "THE FOURTH ROUTE CATCHES AN ERROR", ha="center",
            va="center", fontsize=17.5, fontweight="bold", color=C_INK)
    ax.text(0.5, 0.24, "circulation is additive — a constraint linking "
            "three measured lines", ha="center", va="center", fontsize=12.5,
            color=C_MUTE, style="italic")

    # 6: ritz plot
    plot_ritz(fig.add_subplot(gs[6]), 12, xlabel=False)

    # 7: the finding
    ax = fig.add_subplot(gs[7]); _blank(ax)
    ax.text(0.5, 0.72, r"$73\times$ the rounding budget", ha="center",
            va="center", fontsize=15, fontweight="bold", color=C_BAD)
    ax.text(0.5, 0.24, "air / vacuum convention error in the source table",
            ha="center", va="center", fontsize=12.5, color=C_INK)

    # 8: heading 3
    ax = fig.add_subplot(gs[8]); _blank(ax)
    ax.text(0.5, 0.70, "THE 21 cm LINE IS THE PARITY RUNG", ha="center",
            va="center", fontsize=17.5, fontweight="bold", color=C_INK)
    ax.text(0.5, 0.22, "it rewrites $s$ alone, and is dipole-forbidden",
            ha="center", va="center", fontsize=12.5, color=C_MUTE,
            style="italic")

    # 9: the word
    draw_word(fig.add_subplot(gs[9]), 20, 11)

    # 10: power plot
    plot_power(fig.add_subplot(gs[10]), 12)

    # 11: the gap
    ax = fig.add_subplot(gs[11]); _blank(ax)
    ax.text(0.5, 0.34, "5.85 orders below the smallest allowed rung — "
            "near-inert, not inert", ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_INK)

    # 12: footer
    ax = fig.add_subplot(gs[12]); _blank(ax)
    ax.plot([0.03, 0.97], [0.86, 0.86], color=C_INK, lw=1.2)
    ax.text(0.5, 0.36, "powers computed from measured transitions, "
            "not assigned", ha="center", va="center", fontsize=12.5,
            color=C_MUTE, style="italic")

    fig.savefig(FIG / "graphical_abstract.png", dpi=125, facecolor="white")
    plt.close(fig)
    im = plt.imread(FIG / "graphical_abstract.png")
    h, w = im.shape[0], im.shape[1]
    ok = (560 <= h <= 2800) and (1100 <= w <= 5500)
    print("  graphical_abstract.png  %d x %d px (h x w)  MDPI range: %s"
          % (h, w, "OK" if ok else "OUT OF RANGE"))


# =========================================================================
# Graphical summary -- landscape
# =========================================================================

def graphical_summary():
    fig = plt.figure(figsize=(19.0, 10.0))
    gs = GridSpec(
        4, 3, figure=fig,
        height_ratios=[0.62, 0.40, 2.05, 0.85],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.045, right=0.965, top=0.955, bottom=0.045,
        hspace=0.38, wspace=0.26)

    # title band
    ax = fig.add_subplot(gs[0, :]); _blank(ax)
    ax.text(0.5, 0.74, "Four Routes to a Spectral Line", ha="center",
            va="center", fontsize=28, fontweight="bold", color=C_INK)
    ax.text(0.5, 0.34, "the complete spectra of H, H$_2$ and H$_2$O derived "
            "four ways — and what the agreement is actually worth",
            ha="center", va="center", fontsize=14, color=C_MUTE,
            style="italic")
    ax.plot([0.0, 1.0], [0.06, 0.06], color=C_INK, lw=1.5)

    # column headings
    heads = [
        ("THE FOUR ROUTES", "sharing only $C(n)=2n^{2}$"),
        ("A CROSS-LINE CONSTRAINT", "circulation additivity, i.e. Ritz"),
        ("THE 21 cm LINE, PLACED", "a rung on the parity letter alone"),
    ]
    for j, (h1, h2) in enumerate(heads):
        ax = fig.add_subplot(gs[1, j]); _blank(ax)
        ax.text(0.0, 0.70, h1, fontsize=14.5, fontweight="bold", color=C_INK,
                va="center")
        ax.text(0.0, 0.24, h2, fontsize=11.5, color=C_MUTE, va="center")

    # ---- column 1: routes over the ppm plot -------------------------
    inner = gs[2, 0].subgridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.50)
    draw_routes(fig.add_subplot(inner[0]), 13, 10)
    plot_ppm(fig.add_subplot(inner[1]), 10.5)

    # ---- column 2: ritz ---------------------------------------------
    inner = gs[2, 1].subgridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.42)
    plot_ritz(fig.add_subplot(inner[0]), 10.5)
    ax = fig.add_subplot(inner[1]); _blank(ax)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.04), 1.0, 0.92, boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#fffbeb", edgecolor="#d97706", linewidth=1.8,
        transform=ax.transAxes))
    ax.text(0.045, 0.85, "closed ladders, measured profiles", fontsize=11.5,
            fontweight="bold", color="#b45309", va="center")
    ax.text(0.045, 0.62,
            "a molecular mode set is a cycle, so\n"
            "composite power is undefined for it.",
            fontsize=10, color=C_INK, va="center", linespacing=1.45)
    ax.text(0.045, 0.26,
            "H$_2$O from HITRAN:\n"
            "   $\\varrho = %.4f$      $u = %.4f$\n"
            "   margin vs electronic:  $%+.3f$"
            % (CL["h2o_rho"], CL["h2o_uniformity"], CL["separation_margin"]),
            fontsize=10, color=C_INK, va="center", linespacing=1.45)

    # ---- column 3: word over power plot -----------------------------
    inner = gs[2, 2].subgridspec(3, 1, height_ratios=[0.62, 0.30, 1.25],
                                 hspace=0.30)
    draw_word(fig.add_subplot(inner[0]), 17, 10)
    ax = fig.add_subplot(inner[1]); _blank(ax)
    ax.text(0.5, 0.55, "chirality is a topological invariant,\n"
            "so a rung on $s$ alone cannot be E1", ha="center", va="center",
            fontsize=11, color=C_INK, linespacing=1.4)
    plot_power(fig.add_subplot(inner[2]), 10.5)

    # ---- verdict row ------------------------------------------------
    ax = fig.add_subplot(gs[3, 0]); _blank(ax)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.06), 1.0, 0.88, boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#fef2f2", edgecolor=C_BAD, linewidth=1.8,
        transform=ax.transAxes))
    ax.text(0.04, 0.74, "agreement here is structural", fontsize=12,
            fontweight="bold", color=C_BAD, va="center")
    ax.text(0.04, 0.32, "three routes, one closed form. shared residual\n"
            "$1.1\\times10^{-5}$ = reduced mass + QED, not disagreement.",
            fontsize=10.5, color=C_INK, va="center", linespacing=1.5)

    ax = fig.add_subplot(gs[3, 1]); _blank(ax)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.06), 1.0, 0.88, boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#f0fdf4", edgecolor=C_OK, linewidth=1.8,
        transform=ax.transAxes))
    ax.text(0.04, 0.74, "what the fourth route buys", fontsize=12,
            fontweight="bold", color=C_OK, va="center")
    ax.text(0.04, 0.32, "standard-air Balmer beside vacuum Lyman.\n"
            "a per-line check cannot see it — each line\n"
            "agrees with the table it came from.",
            fontsize=10.5, color=C_INK, va="center", linespacing=1.5)

    ax = fig.add_subplot(gs[3, 2]); _blank(ax)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.06), 1.0, 0.88, boxstyle="round,pad=0,rounding_size=0.03",
        facecolor="#f8fafc", edgecolor=C_MUTE, linewidth=1.6,
        transform=ax.transAxes))
    ax.text(0.04, 0.74, "stated limits", fontsize=12, fontweight="bold",
            color=C_INK, va="center")
    ax.text(0.04, 0.32, "the ladder reconstructs rather than predicts;\n"
            "one registered expectation was refuted\n"
            "before it was diagnosed.",
            fontsize=10.5, color=C_INK, va="center", linespacing=1.5)

    fig.savefig(FIG / "graphical_summary.png", dpi=150, facecolor="white")
    plt.close(fig)
    im = plt.imread(FIG / "graphical_summary.png")
    print("  graphical_summary.png   %d x %d px (h x w)"
          % (im.shape[0], im.shape[1]))


if __name__ == "__main__":
    graphical_abstract()
    graphical_summary()
