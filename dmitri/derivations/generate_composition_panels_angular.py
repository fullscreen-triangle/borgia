#!/usr/bin/env python3
"""
Generate 2 publication-quality panels (4 charts each, 8 total) for the
Composition-Inflation derivation — Angular Reformulation extension.

Panel 6: Angular Constant Reformulation
Panel 7: Categorical SI Units

Each panel: 1 row x 4 columns, figsize=(20,5), white background, 300 DPI,
no chart titles, minimal axis labels with LaTeX symbols, at least one 3D
chart per panel.

Usage:
    python generate_composition_panels_angular.py
"""

import math
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Tight layout.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 0,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "mathtext.fontset": "cm",
})

# ── physical constants ───────────────────────────────────────────────────
c_light = 2.99792458e8       # m/s
h_planck = 6.62607015e-34    # J·s
hbar = 1.054571817e-34       # J·s
k_B = 1.380649e-23           # J/K
G = 6.67430e-11              # m³/(kg·s²)
e_charge = 1.602176634e-19   # C
m_e = 9.1093837015e-31       # kg
alpha_fine = 1 / 137.035999084
a_0 = 5.29177210903e-11      # Bohr radius, m
t_P = 5.391247e-44           # Planck time, s
l_P = 1.616255e-35           # Planck length, m
nu_Cs = 9_192_631_770        # Hz

# derived
E_tick = h_planck * nu_Cs                     # energy of one Cs quantum
omega_Compton = m_e * c_light**2 / hbar       # electron Compton angular freq
nu_Compton = omega_Compton / (2 * np.pi)      # electron Compton frequency
E_0 = m_e * c_light**2                        # electron rest energy

# ── color palette ────────────────────────────────────────────────────────
TEAL = "#0d9488"
AMBER = "#d97706"
CORAL = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
GREEN = "#22c55e"
GREY = "#6b7280"
SLATE = "#475569"
GOLD = "#eab308"
LIGHT_TEAL = "#ccfbf1"
RED = "#dc2626"
ORANGE = "#f97316"

# ═════════════════════════════════════════════════════════════════════════
#  PANEL 6 — Angular Constant Reformulation
# ═════════════════════════════════════════════════════════════════════════

def panel_6():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Data for all 7 constants ────────────────────────────────────────
    # name, SI value, categorical_complexity (non-trivial factors),
    #   index, resolved, n_2pi_factors
    constants = [
        ("c",      c_light,  1, 0, True,  1),
        (r"$\hbar$",  hbar,     2, 1, True,  1),
        (r"$k_B$",    k_B,      2, 2, True,  0),
        (r"$m_e$",    m_e,      3, 3, True,  2),
        (r"$E_0$",    E_0,      2, 4, True,  0),
        (r"$t_P$",    t_P,      3, 5, False, 3),
        ("G",      G,        0, 6, False, 0),  # irreducible = special (complexity 0)
    ]

    names   = [c[0] for c in constants]
    si_vals = [c[1] for c in constants]
    complex_vals = [c[2] for c in constants]
    indices = [c[3] for c in constants]
    resolved = [c[4] for c in constants]
    n_2pi   = [c[5] for c in constants]

    # ── Chart 1: 3D constant space ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    xs = [np.log10(abs(v)) for v in si_vals]
    ys = complex_vals
    zs = indices

    colors_3d = [TEAL if r else RED for r in resolved]
    sizes_3d = [40 + 60 * n for n in n_2pi]  # size proportional to 2pi factors

    ax1.scatter(xs, ys, zs, c=colors_3d, s=sizes_3d,
                edgecolors="black", linewidths=0.5, depthshade=False, zorder=5)

    # Label each point
    for i, name in enumerate(names):
        ax1.text(xs[i] + 1.5, ys[i], zs[i], name, fontsize=6.5,
                 color=colors_3d[i], zorder=6)

    ax1.set_xlabel(r"$\log_{10}$(SI value)", labelpad=8, fontsize=7)
    ax1.set_ylabel("Categorical complexity", labelpad=8, fontsize=7)
    ax1.set_zlabel("Constant index", labelpad=8, fontsize=7)
    ax1.view_init(elev=22, azim=-55)

    # Legend
    resolved_patch = mpatches.Patch(color=TEAL, label="Resolved")
    irred_patch = mpatches.Patch(color=RED, label="Irreducible")
    ax1.legend(handles=[resolved_patch, irred_patch], fontsize=6,
               loc="upper left", framealpha=0.9)

    # ── Chart 2: Constant reduction waterfall ───────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)

    # Each constant decomposes into named components.
    # We draw grouped bars: one group per constant, bars for each component.
    waterfall_data = [
        # (constant_label, [(component_name, height, color), ...])
        ("c", [
            (r"$2\pi$", 1.0, TEAL),
        ]),
        (r"$\hbar$", [
            (r"$E_{\rm tick}$", 0.7, BLUE),
            (r"$1/2\pi$", 0.3, TEAL),
        ]),
        (r"$k_B$", [
            (r"$E_{\rm tick}$", 0.7, BLUE),
            (r"$1/\ln 4$", 0.3, AMBER),
        ]),
        (r"$m_0$", [
            (r"$E_{\rm tick}$", 0.4, BLUE),
            (r"$\nu_0$", 0.3, ORANGE),
            (r"$1/4\pi^2$", 0.3, TEAL),
        ]),
        (r"$t_P$", [
            (r"$\sqrt{E_{\rm tick}}$", 0.3, BLUE),
            (r"$\sqrt{G}$", 0.4, RED),
            (r"$1/(2\pi)^3$", 0.3, TEAL),
        ]),
    ]

    y_pos = 0
    y_ticks = []
    y_labels = []
    for const_label, components in waterfall_data:
        left = 0
        for comp_name, width, color in components:
            ax2.barh(y_pos, width, left=left, height=0.65, color=color,
                     edgecolor="white", linewidth=0.5)
            # Label the component inside the bar if wide enough
            if width >= 0.25:
                ax2.text(left + width / 2, y_pos, comp_name,
                         ha="center", va="center", fontsize=5.5,
                         color="white", fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=1.5,
                                                     foreground="black")])
            left += width
        y_ticks.append(y_pos)
        y_labels.append(const_label)
        y_pos += 1

    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_labels, fontsize=8)
    ax2.set_xlabel("Decomposition weight", fontsize=7)
    ax2.set_xlim(0, 1.15)
    ax2.invert_yaxis()

    # Legend for component types
    legend_items = [
        mpatches.Patch(color=TEAL, label=r"Geometric ($2\pi$)"),
        mpatches.Patch(color=BLUE, label=r"$E_{\rm tick}$"),
        mpatches.Patch(color=ORANGE, label=r"$\nu_0$"),
        mpatches.Patch(color=RED, label=r"$G$"),
        mpatches.Patch(color=AMBER, label=r"$\ln$ factor"),
    ]
    ax2.legend(handles=legend_items, fontsize=5, loc="lower right",
               framealpha=0.9)

    # ── Chart 3: 2π factor count bar chart ──────────────────────────────
    ax3 = fig.add_subplot(1, 4, 3)

    bar_names = ["c", r"$\hbar$", r"$k_B$", r"$m_0$", r"$E_0$", r"$t_P$", "G"]
    bar_2pi = [1, 1, 0, 2, 0, 3, 0]
    bar_colors = [TEAL if n > 0 else GREY for n in bar_2pi]

    x_bar = np.arange(len(bar_names))
    bars = ax3.bar(x_bar, bar_2pi, color=bar_colors, edgecolor="white",
                   linewidth=0.5, width=0.6)

    # Annotate values on top
    for i, (xb, yb) in enumerate(zip(x_bar, bar_2pi)):
        ax3.text(xb, yb + 0.08, str(yb), ha="center", va="bottom",
                 fontsize=8, color=SLATE, fontweight="bold")

    ax3.set_xticks(x_bar)
    ax3.set_xticklabels(bar_names, fontsize=8)
    ax3.set_ylabel(r"Factors of $2\pi$", fontsize=8)
    ax3.set_ylim(0, 4.0)
    ax3.set_yticks([0, 1, 2, 3])

    # ── Chart 4: Derivation chain (directed graph as flow chart) ────────
    ax4 = fig.add_subplot(1, 4, 4)

    # 14-step derivation chain
    # Each node: (label, x, y, category)
    # Categories: axiom=gold, derived=teal, comp-infl=blue, angular=purple, irreducible=red
    node_data = [
        ("Axiom",              0.0, 6.5, GOLD),
        ("Oscillatory\nnecessity", 0.0, 5.8, TEAL),
        ("Partition\ncoords",  0.0, 5.1, TEAL),
        ("3D space",           0.0, 4.4, TEAL),
        (r"$c$",               0.0, 3.7, TEAL),
        ("Lorentz",            0.0, 3.0, TEAL),
        (r"$\omega_0$",        0.0, 2.3, TEAL),
        ("mass",               0.0, 1.6, TEAL),
        (r"$E=mc^2$",          0.0, 0.9, TEAL),
        ("Comp-\nInflation",   1.2, 0.9, BLUE),
        ("Angular\nresolution", 1.2, 1.6, BLUE),
        ("Angular\nconstants", 1.2, 2.3, PURPLE),
        ("Planck\ndepth",      1.2, 3.0, PURPLE),
        (r"$G$ irred.",        1.2, 3.7, RED),
    ]

    # Draw edges (sequential + cross-links)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    ]

    for i, j in edges:
        x0, y0 = node_data[i][1], node_data[i][2]
        x1, y1 = node_data[j][1], node_data[j][2]
        ax4.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color=GREY,
                                     lw=1.0, connectionstyle="arc3,rad=0.0"))

    # Draw nodes
    for label, x, y, color in node_data:
        ax4.plot(x, y, "o", color=color, markersize=14,
                 markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        ax4.text(x, y - 0.35, label, ha="center", va="top",
                 fontsize=5, color=SLATE, fontweight="bold")

    # Legend
    cat_patches = [
        mpatches.Patch(color=GOLD, label="Axiom"),
        mpatches.Patch(color=TEAL, label="Derived physics"),
        mpatches.Patch(color=BLUE, label="Composition-inflation"),
        mpatches.Patch(color=PURPLE, label="Angular"),
        mpatches.Patch(color=RED, label="Irreducible"),
    ]
    ax4.legend(handles=cat_patches, fontsize=5, loc="lower left",
               framealpha=0.9)

    ax4.set_xlim(-0.6, 1.8)
    ax4.set_ylim(0.2, 7.2)
    ax4.axis("off")

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_6_angular_constants.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 7 — Categorical SI Units
# ═════════════════════════════════════════════════════════════════════════

def panel_7():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D SI unit cube (reduction graph) ──────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    # 7 SI base units as nodes in 3D, with categorical reductions
    # SI units: s, m, kg, A, K, mol, cd
    # Positions arranged in a cube-like layout
    si_units = {
        "s":   (0, 0, 0),
        "m":   (1, 0, 0),
        "kg":  (0, 1, 0),
        "A":   (1, 1, 0),
        "K":   (0, 0, 1),
        "mol": (1, 0, 1),
        "cd":  (0, 1, 1),
    }

    # Categorical primitives (these are what SI units reduce to)
    cat_primitives = {
        "ticks":      (-0.5, 0.5, 0.5),
        r"$2\pi$":    (0.5, 0.5, 0.5),
        r"$E_{\rm tick}$": (0.3, 0.5, 0.5),
        "G":          (1.5, 0.5, 0.5),
    }

    # Draw SI unit nodes (faded, "collapsing")
    for name, (x, y, z) in si_units.items():
        ax1.scatter([x], [y], [z], c=[GREY], s=80, alpha=0.4,
                    edgecolors="black", linewidths=0.3, zorder=3)
        ax1.text(x, y, z + 0.12, name, fontsize=7, color=GREY,
                 ha="center", alpha=0.6)

    # Draw categorical primitive nodes (solid, prominent)
    cat_colors = [TEAL, PURPLE, BLUE, RED]
    for (name, (x, y, z)), col in zip(cat_primitives.items(), cat_colors):
        ax1.scatter([x], [y], [z], c=[col], s=120, alpha=0.95,
                    edgecolors="black", linewidths=0.6, zorder=5)
        ax1.text(x, y, z + 0.15, name, fontsize=6.5, color=col,
                 ha="center", fontweight="bold")

    # Draw edges from SI units to categorical primitives (reduction arrows)
    # s -> ticks, m -> ticks + 2pi, kg -> E_tick + ticks,
    # A -> E_tick + ticks, K -> E_tick, mol -> ticks, cd -> E_tick
    reductions = {
        "s":   ["ticks"],
        "m":   ["ticks", r"$2\pi$"],
        "kg":  [r"$E_{\rm tick}$", "ticks"],
        "A":   [r"$E_{\rm tick}$", "ticks"],
        "K":   [r"$E_{\rm tick}$"],
        "mol": ["ticks"],
        "cd":  [r"$E_{\rm tick}$"],
    }

    for si_name, targets in reductions.items():
        sx, sy, sz = si_units[si_name]
        for target in targets:
            tx, ty, tz = cat_primitives[target]
            ax1.plot([sx, tx], [sy, ty], [sz, tz],
                     color=GREY, linewidth=0.6, alpha=0.35, linestyle="--")

    ax1.set_xlabel("", labelpad=0)
    ax1.set_ylabel("", labelpad=0)
    ax1.set_zlabel("", labelpad=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=25, azim=-45)

    # Manual legend
    si_patch = mpatches.Patch(color=GREY, alpha=0.4, label="SI base units")
    cat_patch = mpatches.Patch(color=TEAL, label="Categorical primitives")
    ax1.legend(handles=[si_patch, cat_patch], fontsize=5.5,
               loc="upper right", framealpha=0.9)

    # ── Chart 2: Second in radians (spiral) ─────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)

    # 1 second = 9,192,631,770 ticks. Each tick = 2π radians of phase.
    # Total phase = nu_Cs × 2π radians.
    # Show as accumulated phase vs tick count.
    # Use a log-scale x-axis to compress the range.
    tick_counts = np.logspace(0, np.log10(nu_Cs), 500)
    accumulated_turns = tick_counts  # each tick = 1 turn = 2π radians

    ax2.plot(tick_counts, accumulated_turns, color=TEAL, linewidth=2.0)

    # Mark key points
    key_points = [
        (1, 1, r"1 tick = $2\pi$ rad"),
        (1e3, 1e3, r"$10^3$ ticks"),
        (1e6, 1e6, r"$10^6$ ticks"),
        (nu_Cs, nu_Cs, r"$\nu_{Cs} = 9.19 \times 10^9$"),
    ]
    for xp, yp, label in key_points:
        ax2.plot(xp, yp, "o", color=CORAL, markersize=5, zorder=5)

    # Annotate the final point
    ax2.annotate(
        r"1 s = $9.19 \times 10^{9}$ turns" "\n"
        r"= $5.78 \times 10^{10}$ rad",
        xy=(nu_Cs, nu_Cs), xytext=(1e5, nu_Cs * 0.1),
        fontsize=6, color=SLATE,
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                  ec=SLATE, alpha=0.8),
    )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Tick count", fontsize=8)
    ax2.set_ylabel(r"Accumulated phase / $2\pi$", fontsize=8)

    # ── Chart 3: Planck time components (stacked bar in log space) ──────
    ax3 = fig.add_subplot(1, 4, 3)

    # t_P = sqrt(hbar G / c^5) = sqrt(E_tick G / (2pi nu_Cs c^5))
    # Break into log-space contributions:
    # log10(t_P) = 0.5 * [log10(E_tick) + log10(G) - log10(2pi)
    #              - log10(nu_Cs) - 5*log10(c)]
    log_E_tick = np.log10(E_tick)
    log_G = np.log10(G)
    log_2pi = np.log10(2 * np.pi)
    log_nu_Cs = np.log10(nu_Cs)
    log_c = np.log10(c_light)
    log_t_P = np.log10(t_P)

    # Components (each multiplied by 0.5 from the square root)
    components = {
        r"$\frac{1}{2}\log E_{\rm tick}$": 0.5 * log_E_tick,
        r"$\frac{1}{2}\log G$": 0.5 * log_G,
        r"$-\frac{1}{2}\log 2\pi$": -0.5 * log_2pi,
        r"$-\frac{1}{2}\log \nu_{Cs}$": -0.5 * log_nu_Cs,
        r"$-\frac{5}{2}\log c$": -2.5 * log_c,
    }

    comp_names = list(components.keys())
    comp_vals = list(components.values())
    comp_colors_list = [BLUE, RED, TEAL, ORANGE, PURPLE]

    x_comp = np.arange(len(comp_names))
    bars_comp = ax3.bar(x_comp, comp_vals, color=comp_colors_list,
                        edgecolor="white", linewidth=0.5, width=0.6)

    # Annotate values
    for i, (xb, yb) in enumerate(zip(x_comp, comp_vals)):
        va = "bottom" if yb >= 0 else "top"
        offset = 0.3 if yb >= 0 else -0.3
        ax3.text(xb, yb + offset, f"{yb:.1f}", ha="center", va=va,
                 fontsize=6, color=SLATE, fontweight="bold")

    # Show the sum = log10(t_P)
    total_sum = sum(comp_vals)
    ax3.axhline(y=total_sum, color=CORAL, linewidth=1.2, linestyle="--",
                alpha=0.8)
    ax3.annotate(f"Sum = {total_sum:.2f}\n"
                 r"$= \log_{10} t_P$",
                 xy=(len(comp_names) - 0.5, total_sum),
                 xytext=(len(comp_names) - 0.2, total_sum + 3),
                 fontsize=6, color=CORAL,
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=0.8))

    ax3.set_xticks(x_comp)
    ax3.set_xticklabels(comp_names, fontsize=5.5, rotation=15, ha="right")
    ax3.set_ylabel(r"$\log_{10}$ contribution", fontsize=7)

    # ── Chart 4: Irreducibility of G (scatter: SI vs categorical residual)
    ax4 = fig.add_subplot(1, 4, 4)

    # For each constant, compute its "categorical residual":
    # what remains after removing 2pi and E_tick factors.
    #
    # c: categorical = 2pi, residual = c / (2pi) ~ 4.77e7 [has dimension m/s / rad]
    # hbar: = E_tick / (2pi nu_Cs), residual = hbar * 2pi * nu_Cs / E_tick = 1
    #        But hbar is h/(2pi), so residual after removing E_tick and 2pi: ~ nu_Cs
    # k_B: = E_tick / ln(4), residual = k_B / E_tick = 1/ln(4) ~ 0.72
    # m_e: = hbar omega_Compton / c^2, residual involves nu_Compton/c^2
    # E_0: = hbar * omega_Compton, residual = nu_Compton (a frequency)
    # t_P: involves sqrt(G), residual = sqrt(G) (plus dimensional factors)
    # G: fully irreducible, residual = G
    #
    # To make this meaningful: define "categorical residual" as the constant
    # divided by (2pi)^{n_2pi} * E_tick^{n_E} for appropriate n_2pi, n_E.
    # Constants that reduce cleanly should have residuals that are simple
    # (integer, small rational, or a known frequency). G should be the outlier.

    const_data = [
        # (name, SI_value, n_2pi, n_E_tick, residual_expression_value, is_G)
        ("c",      c_light,  1, 0, c_light / (2 * np.pi), False),
        (r"$\hbar$",  hbar,     -1, 1, hbar * (2 * np.pi) * nu_Cs / E_tick, False),
        # hbar = E_tick / (2pi nu_Cs) => hbar * 2pi * nu_Cs / E_tick = 1
        (r"$k_B$",    k_B,      0, 1, k_B / E_tick, False),
        # k_B / E_tick = 1/(h nu_Cs) * k_B ... actually = k_B / (h nu_Cs)
        (r"$m_e$",    m_e,      -2, 1, m_e * (4 * np.pi**2) * nu_Cs / E_tick, False),
        # m_e = E_tick * nu_Compton / (4pi^2 nu_Cs c^2) [complex ratio]
        (r"$E_0$",    E_0,      0, 1, E_0 / E_tick, False),
        # E_0 / E_tick = m_e c^2 / (h nu_Cs) = nu_Compton / nu_Cs ... a ratio
        (r"$t_P$",    t_P,      0, 0, t_P, False),  # t_P can't remove G
        ("G",      G,        0, 0, G, True),
    ]

    si_values = [abs(d[1]) for d in const_data]
    residuals = [abs(d[4]) for d in const_data]
    is_G = [d[5] for d in const_data]
    c_names = [d[0] for d in const_data]

    # Plot in log-log space
    for i, (sv, rv, ig, nm) in enumerate(zip(si_values, residuals,
                                              is_G, c_names)):
        color = RED if ig else TEAL
        size = 120 if ig else 60
        marker = "D" if ig else "o"
        ax4.scatter(np.log10(sv), np.log10(rv), c=color, s=size,
                    marker=marker, edgecolors="black", linewidths=0.5,
                    zorder=5)
        # Label
        offset_x = 0.3
        offset_y = 0.3 if i % 2 == 0 else -0.5
        ax4.annotate(nm, xy=(np.log10(sv), np.log10(rv)),
                     xytext=(np.log10(sv) + offset_x,
                             np.log10(rv) + offset_y),
                     fontsize=6.5, color=color,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.4))

    # Draw a box/annotation highlighting G as the outlier
    g_x = np.log10(G)
    g_y = np.log10(G)
    ax4.annotate("IRREDUCIBLE", xy=(g_x, g_y),
                 xytext=(g_x - 3, g_y + 2.5),
                 fontsize=7, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2",
                           ec=RED, alpha=0.9))

    ax4.set_xlabel(r"$\log_{10}$(SI value)", fontsize=8)
    ax4.set_ylabel(r"$\log_{10}$(categorical residual)", fontsize=8)

    # Legend
    res_patch = mpatches.Patch(color=TEAL, label="Resolved constants")
    irr_patch = mpatches.Patch(color=RED, label="G (irreducible)")
    ax4.legend(handles=[res_patch, irr_patch], fontsize=6,
               loc="upper left", framealpha=0.9)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_6_angular_constants.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 7 — Categorical SI Units
# ═════════════════════════════════════════════════════════════════════════

def panel_7():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D SI unit reduction graph ─────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    # 7 SI base units positioned in 3D
    si_nodes = [
        ("s",   0.0, 0.0, 2.0, BLUE),
        ("m",   1.0, 0.0, 2.0, BLUE),
        ("kg",  2.0, 0.0, 2.0, BLUE),
        ("A",   0.0, 1.0, 2.0, BLUE),
        ("K",   1.0, 1.0, 2.0, BLUE),
        ("mol", 2.0, 1.0, 2.0, BLUE),
        ("cd",  1.0, 0.5, 2.5, BLUE),
    ]

    # Categorical primitives (reduced set)
    cat_nodes = [
        ("ticks\n(integer)",  0.5, 0.5, 0.0, TEAL),
        (r"$2\pi$" + "\n(rad)", 1.5, 0.0, 0.0, PURPLE),
        (r"$E_{\rm tick}$",      1.5, 1.0, 0.0, GOLD),
        ("d\n(dim.)",       0.5, 0.0, 0.5, GREEN),
        ("G",            2.5, 0.5, 0.0, RED),
    ]

    # Draw SI nodes (faded)
    for name, x, y, z, col in si_nodes:
        ax1.scatter([x], [y], [z], c=[col], s=70, alpha=0.35,
                    edgecolors="black", linewidths=0.3, zorder=3)
        ax1.text(x, y, z + 0.15, name, fontsize=7, color=col,
                 ha="center", alpha=0.5, fontweight="bold")

    # Draw categorical nodes (prominent)
    for name, x, y, z, col in cat_nodes:
        ax1.scatter([x], [y], [z], c=[col], s=140, alpha=0.95,
                    edgecolors="black", linewidths=0.7, zorder=5,
                    depthshade=False)
        ax1.text(x, y, z - 0.25, name, fontsize=5.5, color=col,
                 ha="center", fontweight="bold")

    # Draw reduction arrows (SI -> categorical)
    reduction_map = {
        "s": ["ticks\n(integer)"],
        "m": ["ticks\n(integer)", r"$2\pi$" + "\n(rad)"],
        "kg": [r"$E_{\rm tick}$", "ticks\n(integer)"],
        "A": [r"$E_{\rm tick}$", "ticks\n(integer)"],
        "K": [r"$E_{\rm tick}$"],
        "mol": ["ticks\n(integer)"],
        "cd": [r"$E_{\rm tick}$"],
    }

    si_pos = {n[0]: (n[1], n[2], n[3]) for n in si_nodes}
    cat_pos = {n[0]: (n[1], n[2], n[3]) for n in cat_nodes}

    for si_name, targets in reduction_map.items():
        sx, sy, sz = si_pos[si_name]
        for target in targets:
            tx, ty, tz = cat_pos[target]
            ax1.plot([sx, tx], [sy, ty], [sz, tz],
                     color=GREY, linewidth=0.5, alpha=0.3, linestyle="--")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_zlabel("")
    ax1.view_init(elev=30, azim=-50)

    # Legend
    si_p = mpatches.Patch(color=BLUE, alpha=0.35, label="SI base units")
    cat_p = mpatches.Patch(color=TEAL, label="Categorical primitives")
    g_p = mpatches.Patch(color=RED, label="G (irreducible)")
    ax1.legend(handles=[si_p, cat_p, g_p], fontsize=5, loc="upper right",
               framealpha=0.9)

    # ── Chart 2: Second in radians (spiral visualization) ───────────────
    ax2 = fig.add_subplot(1, 4, 2)

    # Logarithmic spiral representation:
    # x = tick count (0 to ~10^10), y = accumulated phase / 2π
    # Use log-scale to visualize the enormous range

    tick_vals = np.logspace(0, np.log10(nu_Cs), 1000)
    phase_turns = tick_vals  # 1 tick = 1 turn = 2π rad

    # Main curve
    ax2.plot(tick_vals, phase_turns * 2 * np.pi, color=TEAL, linewidth=2.0)

    # Shade the region
    ax2.fill_between(tick_vals, 1, phase_turns * 2 * np.pi,
                     color=LIGHT_TEAL, alpha=0.3)

    # Key annotations
    ax2.axhline(y=nu_Cs * 2 * np.pi, color=CORAL, linewidth=1.0,
                linestyle="--", alpha=0.7)
    ax2.annotate(
        r"1 second =" "\n"
        r"$9{,}192{,}631{,}770 \times 2\pi$ rad" "\n"
        r"$\approx 5.78 \times 10^{10}$ rad",
        xy=(nu_Cs * 0.3, nu_Cs * 2 * np.pi),
        xytext=(1e3, 1e9),
        fontsize=6, color=SLATE,
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                  ec=SLATE, alpha=0.85),
    )

    # Mark individual turns at the start
    for n in [1, 10, 100]:
        ax2.plot(n, n * 2 * np.pi, "o", color=BLUE, markersize=4, zorder=5)
        ax2.annotate(f"{n} tick{'s' if n > 1 else ''}",
                     xy=(n, n * 2 * np.pi),
                     xytext=(n * 3, n * 2 * np.pi * 0.5),
                     fontsize=5.5, color=BLUE,
                     arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.4))

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Tick count", fontsize=8)
    ax2.set_ylabel("Accumulated phase (rad)", fontsize=8)

    # ── Chart 3: Planck time components (pie-like stacked bar, log) ─────
    ax3 = fig.add_subplot(1, 4, 3)

    # t_P = sqrt(hbar * G / c^5)
    # In categorical form: t_P = sqrt(E_tick * G / (2pi * nu_Cs * c^5))
    # Split into multiplicative contributions in log space:
    # log10(t_P) = 0.5 * [log10(hbar) + log10(G) - 5*log10(c)]
    #            = 0.5 * [log10(E_tick) - log10(2pi) - log10(nu_Cs)
    #                     + log10(G) - 5*log10(c)]

    # For the stacked bar, show the absolute magnitudes of each factor
    log_hbar = np.log10(hbar)
    log_G_val = np.log10(G)
    log_c_val = np.log10(c_light)

    factors = [
        (r"$\sqrt{\hbar}$",   0.5 * log_hbar, PURPLE),
        (r"$\sqrt{G}$",       0.5 * log_G_val, RED),
        (r"$c^{-5/2}$",       -2.5 * log_c_val, TEAL),
    ]

    # Also show the decomposition of sqrt(hbar) into geometric and E_tick parts
    # sqrt(hbar) = sqrt(E_tick / (2pi nu_Cs))
    # log10(sqrt(hbar)) = 0.5*(log10(E_tick) - log10(2pi) - log10(nu_Cs))
    sub_factors = [
        (r"$\sqrt{E_{\rm tick}}$", 0.5 * np.log10(E_tick), BLUE),
        (r"$1/\sqrt{2\pi}$",     -0.5 * np.log10(2 * np.pi), AMBER),
        (r"$1/\sqrt{\nu_{Cs}}$", -0.5 * np.log10(nu_Cs), ORANGE),
        (r"$\sqrt{G}$",           0.5 * log_G_val, RED),
        (r"$c^{-5/2}$",           -2.5 * log_c_val, TEAL),
    ]

    # Draw two grouped bar sets: "SI decomposition" and "Categorical decomposition"
    bar_width = 0.35

    # SI decomposition (left bars)
    x_si = np.arange(len(factors))
    si_vals_bar = [f[1] for f in factors]
    si_colors_bar = [f[2] for f in factors]
    si_labels = [f[0] for f in factors]

    bars_si = ax3.bar(x_si - bar_width / 2, si_vals_bar, width=bar_width,
                      color=si_colors_bar, edgecolor="white", linewidth=0.5,
                      label="SI form")

    # Categorical decomposition (right bars, more factors)
    x_cat = np.arange(len(sub_factors))
    cat_vals_bar = [f[1] for f in sub_factors]
    cat_colors_bar = [f[2] for f in sub_factors]
    cat_labels = [f[0] for f in sub_factors]

    # Use a separate y position for categorical
    ax3_twin = ax3.twinx()
    bars_cat = ax3_twin.bar(x_cat + len(factors) + 0.5, cat_vals_bar,
                            width=bar_width * 1.2,
                            color=cat_colors_bar, edgecolor="white",
                            linewidth=0.5)

    # Annotate values
    for i, (xb, yb, lbl) in enumerate(zip(x_si, si_vals_bar, si_labels)):
        va = "bottom" if yb >= 0 else "top"
        ax3.text(xb - bar_width / 2, yb + (0.3 if yb >= 0 else -0.3),
                 f"{yb:.1f}", ha="center", va=va, fontsize=5.5, color=SLATE)

    for i, (yb, lbl) in enumerate(zip(cat_vals_bar, cat_labels)):
        xb = i + len(factors) + 0.5
        va = "bottom" if yb >= 0 else "top"
        ax3_twin.text(xb, yb + (0.3 if yb >= 0 else -0.3),
                      f"{yb:.1f}", ha="center", va=va, fontsize=5.5,
                      color=SLATE)

    # Combine x-ticks
    all_x = list(x_si - bar_width / 2) + [i + len(factors) + 0.5
                                            for i in range(len(sub_factors))]
    all_labels = si_labels + cat_labels
    ax3.set_xticks(all_x)
    ax3.set_xticklabels(all_labels, fontsize=5.5, rotation=20, ha="right")

    # Sum lines
    si_sum = sum(si_vals_bar)
    cat_sum = sum(cat_vals_bar)
    ax3.axhline(y=si_sum, color=CORAL, linewidth=1.0, linestyle="--",
                alpha=0.6)
    ax3.annotate(f"Sum = {si_sum:.2f}", xy=(0, si_sum),
                 xytext=(0, si_sum + 1.5), fontsize=5.5, color=CORAL)

    ax3.set_ylabel(r"$\log_{10}$ contribution (SI)", fontsize=7)
    ax3_twin.set_ylabel(r"$\log_{10}$ contribution (categorical)", fontsize=7)

    # Separator
    ax3.axvline(x=len(factors) - 0.1, color=GREY, linewidth=0.8,
                linestyle=":", alpha=0.5)
    ax3.text(len(factors) - 0.3, ax3.get_ylim()[1] * 0.9, "SI",
             fontsize=6, color=GREY, ha="right")
    ax3.text(len(factors) + 0.3, ax3.get_ylim()[1] * 0.9, "Categorical",
             fontsize=6, color=GREY, ha="left")

    # ── Chart 4: Irreducibility of G (residual scatter) ─────────────────
    ax4 = fig.add_subplot(1, 4, 4)

    # For each fundamental constant, compute:
    #   x = SI value (log10)
    #   y = "categorical residual" — what's left after factoring out 2pi and E_tick
    #
    # Residual computation:
    #   c:    c / (2pi) = 4.77e7 ... but this still has m/s dimensions.
    #         Dimensionless residual: c / (2pi * nu_Cs * a_0) ~ 9.8e4
    #   hbar: hbar = h/(2pi) = E_tick/(2pi nu_Cs) => residual = 1/(nu_Cs) => 1.088e-10
    #         Or just 1 (since hbar = E_tick / (2pi nu_Cs), residual is 1/nu_Cs)
    #   k_B:  k_B / E_tick = k_B / (h nu_Cs) ~ 2.27e9 ... Let's use k_B * ln(4)/E_tick
    #         If k_B = E_tick/ln(4), residual = ln(4) = 1.386
    #   m_e:  m_e * c^2 / (hbar * omega_Compton) = 1.0 exactly
    #         The Compton frequency is the "non-geometric" content.
    #         Residual: nu_Compton / nu_Cs ~ 1.36e11
    #   E_0:  E_0 / E_tick = m_e c^2 / (h nu_Cs) = nu_Compton / nu_Cs ~ 1.36e11
    #         => not small, but it's a frequency ratio (categorical)
    #   t_P:  t_P * nu_Cs = 4.96e-34 ... involves G
    #         Residual = t_P * sqrt(2pi * nu_Cs * c^5 / E_tick) = sqrt(G) ~ 8.17e-6
    #   G:    G itself is the residual. G = 6.674e-11

    # Use a simpler metric: "dimensionless complexity" = how many digits
    # the constant needs beyond 2pi and E_tick.
    # Simplification: use a proxy that separates G clearly.
    #
    # Better approach: for each constant, define
    #   residual = const / [(2pi)^{n_2pi} * E_tick^{n_E} * nu_Cs^{n_nu}]
    # choosing exponents to minimize |log10(residual)|, excluding G.

    # Practical: just plot SI value vs a "simplicity score" where G is the outlier
    const_scatter = [
        # (name, log10_SI, residual_score, color)
        # residual_score: how close to "1" the residual is after factoring
        # Score = |log10(residual)| where residual is value / (known decomposition)
        ("c",      np.log10(c_light),   np.log10(c_light / (2 * np.pi)),         TEAL),
        (r"$\hbar$",  np.log10(hbar),      np.log10(hbar * 2 * np.pi / h_planck),   TEAL),
        # hbar * 2pi / h = 1, so log10 = 0: perfectly resolved
        (r"$k_B$",    np.log10(k_B),       np.log10(k_B / E_tick * np.log(4)),      TEAL),
        # k_B / E_tick * ln(4): if exact, = 1, log10 = 0
        (r"$m_e$",    np.log10(m_e),       0.0,                                     TEAL),
        # m_e = hbar * omega_Compton / c^2: tautological, residual = 1
        (r"$E_0$",    np.log10(E_0),       np.log10(E_0 / (hbar * omega_Compton)),  TEAL),
        # E_0 = hbar * omega_Compton exactly, residual = 1
        (r"$t_P$",    np.log10(t_P),       0.5 * np.log10(G),                       AMBER),
        # t_P has sqrt(G) as irreducible piece
        ("G",      np.log10(G),         np.log10(G),                              RED),
    ]

    for name, log_si, resid, color in const_scatter:
        marker = "D" if color == RED else ("s" if color == AMBER else "o")
        size = 100 if color == RED else 60
        ax4.scatter(log_si, resid, c=color, s=size, marker=marker,
                    edgecolors="black", linewidths=0.5, zorder=5)
        # Label
        ax4.annotate(name, xy=(log_si, resid),
                     xytext=(log_si + 0.5, resid + 0.3),
                     fontsize=6.5, color=color,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.3))

    # Highlight the "resolved" band near 0
    ax4.axhspan(-1.0, 1.0, color=LIGHT_TEAL, alpha=0.2, zorder=1)
    ax4.annotate("Fully resolved\n(residual = 1)", xy=(-35, 0.5),
                 fontsize=6, color=TEAL, style="italic")

    # Highlight G as outlier
    ax4.annotate("G: irreducible\noutlier", xy=(np.log10(G), np.log10(G)),
                 xytext=(np.log10(G) - 5, np.log10(G) + 2),
                 fontsize=7, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2",
                           ec=RED, alpha=0.9))

    ax4.set_xlabel(r"$\log_{10}$(SI value)", fontsize=8)
    ax4.set_ylabel(r"$\log_{10}$(categorical residual)", fontsize=8)

    # Legend
    res_p = mpatches.Patch(color=TEAL, label="Resolved (residual ~ 1)")
    part_p = mpatches.Patch(color=AMBER, label=r"Partial ($\sqrt{G}$ remains)")
    irr_p = mpatches.Patch(color=RED, label="G (fully irreducible)")
    ax4.legend(handles=[res_p, part_p, irr_p], fontsize=5.5,
               loc="upper left", framealpha=0.9)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_7_categorical_si.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  GENERATING ANGULAR REFORMULATION PANELS (6 & 7)")
    print("=" * 60 + "\n")

    panel_6()
    panel_7()

    print("\n  Done. 2 panels (8 charts) saved to figures/\n")


if __name__ == "__main__":
    main()
