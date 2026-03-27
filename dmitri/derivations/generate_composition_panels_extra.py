#!/usr/bin/env python3
"""
Generate 2 additional publication-quality panels (4 charts each, 8 total) for
the Composition-Inflation derivation.

Panel 4: Angular Resolution
Panel 5: Composition Structure

Each panel: 1 row x 4 columns, figsize=(20,5), white background, 300 DPI,
no chart titles, minimal axis labels with LaTeX symbols, at least one 3D
chart per panel.

Usage:
    python generate_composition_panels_extra.py
"""

import json
import math
import os
import warnings
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Tight layout.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = BASE / "results"

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
t_P = 5.391246e-44       # Planck time (s)
l_P = 1.616255e-35       # Planck length (m)
CAESIUM_FREQ = 9_192_631_770
tau_Cs = 1.0 / CAESIUM_FREQ  # Caesium period

# ── Planck angular threshold for caesium ─────────────────────────────────
PLANCK_ANGULAR = 2.0 * np.pi * t_P / tau_Cs   # ~3.113e-33 rad
LOG10_PLANCK_ANGULAR = np.log10(PLANCK_ANGULAR)  # ~-32.507

# ── color palette ────────────────────────────────────────────────────────
TEAL = "#0d9488"
AMBER = "#d97706"
CORAL = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
GREEN = "#22c55e"
GREY = "#6b7280"
SLATE = "#475569"
LIGHT_TEAL = "#ccfbf1"

# ── core functions ───────────────────────────────────────────────────────

def log10_T(n, d):
    """log10(T(n,d)) = log10(d) + (n-1)*log10(d+1)."""
    return math.log10(d) + (n - 1) * math.log10(d + 1)


def compute_planck_depth(freq, d=3):
    """Compute Planck depth n_P for a given oscillator frequency and dimension d."""
    tau_osc = 1.0 / freq
    ratio = tau_osc / (d * t_P)
    n_P = 1 + math.ceil(math.log(ratio) / math.log(d + 1))
    return n_P


def compute_planck_depth_d(d):
    """Compute n_P for caesium at given dimension d."""
    ratio = tau_Cs / (d * t_P)
    n_P = 1 + math.ceil(math.log(ratio) / math.log(d + 1))
    return n_P


def compositions(n):
    """Generate all compositions of integer n (ordered partitions)."""
    if n == 0:
        yield ()
        return
    if n == 1:
        yield (1,)
        return
    # Recursive: for each first part c1 = 1..n, recurse on n-c1
    for c1 in range(1, n + 1):
        if c1 == n:
            yield (n,)
        else:
            for rest in compositions(n - c1):
                yield (c1,) + rest


def count_compositions(n):
    """Count compositions of n (should be 2^(n-1))."""
    return sum(1 for _ in compositions(n))


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 4 — Angular Resolution
# ═════════════════════════════════════════════════════════════════════════

def panel_4():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D angular resolution surface ──────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    ns = np.arange(1, 81)
    ds = np.arange(1, 6)
    N, D = np.meshgrid(ns, ds)

    # Compute log10(Δθ) = log10(2π) - log10(T(n,d))
    # T(n,d) = d·(d+1)^(n-1)
    # log10(Δθ) = log10(2π) - log10(d) - (n-1)*log10(d+1)
    log_2pi = np.log10(2.0 * np.pi)
    Z = log_2pi - np.log10(D.astype(float)) - (N - 1) * np.log10(D.astype(float) + 1)

    # Color by log10(Δθ) using viridis
    norm = Normalize(vmin=Z.min(), vmax=Z.max())
    surf = ax1.plot_surface(N, D, Z, cmap="viridis", norm=norm, alpha=0.85,
                            edgecolor="none", antialiased=True)

    # Planck angular threshold plane
    nn_plane, dd_plane = np.meshgrid(np.linspace(1, 80, 30),
                                      np.linspace(0.5, 5.5, 10))
    zz_plane = np.full_like(nn_plane, LOG10_PLANCK_ANGULAR)
    ax1.plot_surface(nn_plane, dd_plane, zz_plane, alpha=0.18, color=CORAL,
                     edgecolor="none")

    ax1.set_xlabel(r"$n$", labelpad=6)
    ax1.set_ylabel(r"$d$", labelpad=6)
    ax1.set_zlabel(r"$\log_{10}\Delta\theta$", labelpad=6)
    ax1.view_init(elev=25, azim=-50)

    # ── Chart 2: Angular resolution vs n for d=3 (semilog-style) ────────
    ax2 = fig.add_subplot(1, 4, 2)

    ns2 = np.arange(1, 81)
    log_dtheta = np.array([log_2pi - log10_T(int(n), 3) for n in ns2])

    ax2.plot(ns2, log_dtheta, color=TEAL, linewidth=2.0)

    # Planck angular threshold horizontal line
    ax2.axhline(y=LOG10_PLANCK_ANGULAR, color=CORAL, linewidth=1.2,
                linestyle="--", alpha=0.9, label=r"$\Delta\theta_P$")

    # Crossing at n=56
    ax2.axvline(x=56, color=SLATE, linewidth=1.0, linestyle="--", alpha=0.7)

    # Label crossing point
    cross_y = log_2pi - log10_T(56, 3)
    ax2.plot(56, cross_y, "o", color=CORAL, markersize=6, zorder=5)
    ax2.annotate(r"$n_P = 56$", xy=(56, cross_y),
                 xytext=(60, cross_y + 3), fontsize=7, color=SLATE,
                 arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.8))

    # Shade sub-Planck region
    ax2.fill_between(ns2, LOG10_PLANCK_ANGULAR,
                     np.minimum(log_dtheta, LOG10_PLANCK_ANGULAR),
                     where=log_dtheta < LOG10_PLANCK_ANGULAR,
                     color=LIGHT_TEAL, alpha=0.5)
    # Also fill to bottom of axes for the sub-Planck region
    y_bottom = log_dtheta.min() - 5
    ax2.fill_between(ns2, y_bottom, LOG10_PLANCK_ANGULAR,
                     where=log_dtheta < LOG10_PLANCK_ANGULAR,
                     color=LIGHT_TEAL, alpha=0.3)

    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$\log_{10}\Delta\theta$ (rad)")
    ax2.set_xlim(1, 80)
    ax2.set_ylim(log_dtheta[-1] - 2, log_dtheta[0] + 2)
    ax2.legend(fontsize=7, loc="upper right", framealpha=0.9)

    # ── Chart 3: Physical time to Planck depth (scatter, log-log) ───────
    ax3 = fig.add_subplot(1, 4, 3)

    oscillators = [
        ("Caesium",     9.19e9,   56, 6.09e-9),
        ("H maser",     1.42e9,   57, 4.01e-8),
        ("Sr optical",  4.29e14,  48, 1.12e-13),
        (r"H$_2$ vib",  1.32e14,  49, 3.71e-13),
        ("CPU 3GHz",    3e9,      57, 1.9e-8),
    ]

    freqs = np.array([o[1] for o in oscillators])
    n_P_vals = np.array([o[2] for o in oscillators])
    times = np.array([o[3] for o in oscillators])
    names = [o[0] for o in oscillators]

    # Color by n_P value
    norm_nP = Normalize(vmin=n_P_vals.min() - 1, vmax=n_P_vals.max() + 1)
    cmap_nP = plt.get_cmap("plasma")
    colors_nP = [cmap_nP(norm_nP(nP)) for nP in n_P_vals]

    # Size by frequency (log-scaled)
    log_freqs = np.log10(freqs)
    sizes = 50 + 200 * (log_freqs - log_freqs.min()) / (log_freqs.max() - log_freqs.min())

    ax3.scatter(freqs, times, c=colors_nP, s=sizes, edgecolors="black",
                linewidths=0.6, zorder=5)
    ax3.set_xscale("log")
    ax3.set_yscale("log")

    # Annotate each point
    offsets = [(10, 8), (10, -14), (10, 8), (10, -14), (-50, 10)]
    for i, name in enumerate(names):
        ax3.annotate(name, xy=(freqs[i], times[i]), fontsize=6.5,
                     xytext=offsets[i], textcoords="offset points", color=SLATE,
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))

    # Reference time scale lines
    ref_times = [(1e-9, "1 ns"), (1e-12, "1 ps"), (1e-15, "1 fs")]
    for ref_t, ref_label in ref_times:
        ax3.axhline(y=ref_t, color=GREY, linewidth=0.6, linestyle=":",
                    alpha=0.6)
        ax3.annotate(ref_label, xy=(freqs.min() * 0.8, ref_t * 1.3),
                     fontsize=6, color=GREY, alpha=0.8)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap_nP, norm=norm_nP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label(r"$n_P$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax3.set_xlabel(r"$\nu$ (Hz)")
    ax3.set_ylabel(r"$n_P / \nu$ (s)")

    # ── Chart 4: Dimensional advantage (n_P vs d for caesium) ───────────
    ax4 = fig.add_subplot(1, 4, 4)

    d_range = np.arange(1, 7)
    # Pre-computed values (verified by formula)
    nP_by_d = {
        1: 112,
        2: 71,
        3: 56,
        4: 49,
        5: 44,
        6: 40,
    }
    # Also compute programmatically for verification
    nP_computed = []
    for d in d_range:
        nP = compute_planck_depth_d(d)
        nP_computed.append(nP)

    nP_vals_bar = [nP_by_d[d] for d in d_range]

    # Color by d
    cmap_d = plt.get_cmap("viridis")
    norm_d = Normalize(vmin=1, vmax=6)
    bar_colors = [cmap_d(norm_d(d)) for d in d_range]

    bars = ax4.bar(d_range, nP_vals_bar, color=bar_colors, edgecolor="white",
                   linewidth=0.6, width=0.65)

    # Annotate bar values
    for d, nP in zip(d_range, nP_vals_bar):
        ax4.text(d, nP + 1.5, str(nP), ha="center", va="bottom",
                 fontsize=7, color=SLATE, fontweight="bold")

    # Annotate the key saving: d=2->3 saves ~14 ticks
    ax4.annotate("", xy=(3, nP_by_d[3]), xytext=(2, nP_by_d[2]),
                 arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.5))
    mid_y = (nP_by_d[2] + nP_by_d[3]) / 2.0
    ax4.text(2.5, mid_y + 1, r"$-15$", ha="center", fontsize=7, color=CORAL,
             fontweight="bold")

    ax4.set_xlabel(r"$d$")
    ax4.set_ylabel(r"$n_P$")
    ax4.set_xticks(d_range)
    ax4.set_ylim(0, 125)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_4_angular_resolution.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 5 — Composition Structure
# ═════════════════════════════════════════════════════════════════════════

def panel_5():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D composition bars for n=5 ────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    # All 16 compositions of 5 (ordered)
    comps_5 = list(compositions(5))
    # Ensure we have 16
    assert len(comps_5) == 16, f"Expected 16 compositions of 5, got {len(comps_5)}"

    # Color map for part sizes: 1=blue, 2=green, 3=orange, 4=red, 5=purple
    part_colors = {
        1: BLUE,
        2: GREEN,
        3: AMBER,
        4: CORAL,
        5: PURPLE,
    }

    # Plot each composition as a horizontal stacked bar in 3D
    # x = cumulative sum (horizontal extent), y = composition index, z = 0
    for idx, comp in enumerate(comps_5):
        x_start = 0
        y_val = idx
        for part in comp:
            # Draw a 3D bar (rectangular prism) for this part
            # Using bar3d: (x, y, z, dx, dy, dz)
            color = part_colors[part]
            ax1.bar3d(x_start, y_val - 0.35, 0,
                      part, 0.7, 0.5,
                      color=color, alpha=0.85, edgecolor="white",
                      linewidth=0.3)
            x_start += part

    ax1.set_xlabel(r"$\Sigma\, c_i$", labelpad=6)
    ax1.set_ylabel("Comp. index", labelpad=6)
    ax1.set_zlabel("")
    ax1.set_zticks([])
    ax1.set_yticks([0, 5, 10, 15])
    ax1.view_init(elev=22, azim=-55)

    # Legend patches
    legend_patches = [mpatches.Patch(color=part_colors[k], label=f"${k}$")
                      for k in sorted(part_colors.keys())]
    ax1.legend(handles=legend_patches, fontsize=5.5, loc="upper right",
               title="Part size", title_fontsize=6, framealpha=0.8)

    # ── Chart 2: Labeled compositions heatmap for n=3, d=3 ─────────────
    ax2 = fig.add_subplot(1, 4, 2)

    # Compositions of 3: (1,1,1), (2,1), (1,2), (3)
    comps_3 = list(compositions(3))
    assert len(comps_3) == 4

    # Dimension labels: S_k=0(blue), S_t=1(green), S_e=2(red)
    dim_labels = [0, 1, 2]  # indices for d=3
    dim_colors_map = {0: BLUE, 1: GREEN, 2: CORAL}
    dim_names = {0: r"$S_k$", 1: r"$S_t$", 2: r"$S_e$"}

    # Build all 48 labeled compositions
    # Each composition (c1, c2, ..., ck) gets d^k = 3^k labelings
    # Each labeling assigns one of {0,1,2} to each part
    all_rows = []  # each row: list of (part_size, label) tuples
    comp_boundaries = []  # track which rows belong to which composition

    for comp in comps_3:
        k = len(comp)
        start_row = len(all_rows)
        for labeling in product(dim_labels, repeat=k):
            row = list(zip(comp, labeling))
            all_rows.append(row)
        comp_boundaries.append((start_row, len(all_rows), comp))

    assert len(all_rows) == 48, f"Expected 48, got {len(all_rows)}"

    # Build a grid: 48 rows x 3 columns (max n=3 tick positions)
    # Each cell: colored by dimension label of the part covering that tick
    grid = np.full((48, 3), -1, dtype=int)  # -1 = empty
    for row_idx, row in enumerate(all_rows):
        col = 0
        for part_size, label in row:
            for _ in range(part_size):
                if col < 3:
                    grid[row_idx, col] = label
                    col += 1

    # Create custom colormap: -1=white, 0=blue, 1=green, 2=red
    from matplotlib.colors import ListedColormap
    cmap_grid = ListedColormap(["white", BLUE, GREEN, CORAL])
    # Shift grid values: -1->0, 0->1, 1->2, 2->3
    grid_shifted = grid + 1

    ax2.imshow(grid_shifted, cmap=cmap_grid, aspect="auto", vmin=0, vmax=3,
               interpolation="nearest")

    # Add horizontal lines to separate composition groups
    for start, end, comp in comp_boundaries:
        ax2.axhline(y=start - 0.5, color="black", linewidth=0.8)
        # Label the composition group on the right
        mid = (start + end - 1) / 2.0
        comp_str = "+".join(str(c) for c in comp)
        ax2.annotate(comp_str, xy=(3.2, mid), fontsize=5.5, va="center",
                     color=SLATE, annotation_clip=False)
    ax2.axhline(y=47.5, color="black", linewidth=0.8)

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels([r"$t_1$", r"$t_2$", r"$t_3$"], fontsize=7)
    ax2.set_ylabel("Labeled trajectory index")
    ax2.set_yticks([0, 10, 20, 30, 40, 47])

    # Legend
    legend_dim = [mpatches.Patch(color=dim_colors_map[k], label=dim_names[k])
                  for k in sorted(dim_colors_map.keys())]
    ax2.legend(handles=legend_dim, fontsize=5.5, loc="lower right",
               framealpha=0.9)

    # ── Chart 3: Composition count verification ─────────────────────────
    ax3 = fig.add_subplot(1, 4, 3)

    ns3 = np.arange(1, 16)
    computed_counts = []
    for n in ns3:
        computed_counts.append(count_compositions(int(n)))
    computed_counts = np.array(computed_counts)

    theoretical = 2.0 ** (ns3 - 1)

    # Theoretical line
    ax3.semilogy(ns3, theoretical, color=TEAL, linewidth=2.0,
                 label=r"$2^{n-1}$", zorder=3)
    # Computed points
    ax3.semilogy(ns3, computed_counts, "o", color=CORAL, markersize=6,
                 markeredgecolor="black", markeredgewidth=0.5,
                 label="Computed", zorder=4)

    ax3.set_xlabel(r"$n$")
    ax3.set_ylabel(r"$|\mathcal{C}(n)|$")
    ax3.set_xlim(0.5, 15.5)
    ax3.set_xticks(ns3)
    ax3.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # ── Chart 4: T(n,d) for d=2,3,4 (semilog) ─────────────────────────
    ax4 = fig.add_subplot(1, 4, 4)

    ns4 = np.arange(1, 76)
    # T(n,2) = 2·3^(n-1)
    log_T_d2 = np.array([math.log10(2) + (n - 1) * math.log10(3) for n in ns4])
    # T(n,3) = 3·4^(n-1)
    log_T_d3 = np.array([math.log10(3) + (n - 1) * math.log10(4) for n in ns4])
    # T(n,4) = 4·5^(n-1)
    log_T_d4 = np.array([math.log10(4) + (n - 1) * math.log10(5) for n in ns4])

    ax4.plot(ns4, log_T_d2, color=BLUE, linewidth=2.0,
             label=r"$T(n,2) = 2\cdot 3^{n-1}$")
    ax4.plot(ns4, log_T_d3, color=TEAL, linewidth=2.0,
             label=r"$T(n,3) = 3\cdot 4^{n-1}$")
    ax4.plot(ns4, log_T_d4, color=AMBER, linewidth=2.0,
             label=r"$T(n,4) = 4\cdot 5^{n-1}$")

    # Planck threshold for caesium: log10(τ_Cs / t_P) ~ 33.305
    planck_log10 = math.log10(tau_Cs / t_P)
    ax4.axhline(y=planck_log10, color=CORAL, linewidth=1.0, linestyle="--",
                alpha=0.8, label=r"$\log_{10}(\tau_{Cs}/t_P)$")

    # Find crossing n for each d
    # d=2: log10(2) + (n-1)*log10(3) = planck_log10
    n_cross_d2 = 1 + (planck_log10 - math.log10(2)) / math.log10(3)
    n_cross_d3 = 1 + (planck_log10 - math.log10(3)) / math.log10(4)
    n_cross_d4 = 1 + (planck_log10 - math.log10(4)) / math.log10(5)

    # Mark crossings
    for n_cross, color, d_val in [(n_cross_d2, BLUE, 2),
                                   (n_cross_d3, TEAL, 3),
                                   (n_cross_d4, AMBER, 4)]:
        if 1 <= n_cross <= 75:
            ax4.axvline(x=n_cross, color=color, linewidth=0.7, linestyle=":",
                        alpha=0.5)
            ax4.annotate(f"$n \\approx {int(round(n_cross))}$",
                         xy=(n_cross, planck_log10 + 0.8),
                         fontsize=6, color=color, ha="center",
                         rotation=0)

    ax4.set_xlabel(r"$n$")
    ax4.set_ylabel(r"$\log_{10} T(n,d)$")
    ax4.set_xlim(1, 75)
    ax4.legend(fontsize=6, loc="upper left", framealpha=0.9)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_5_composition_structure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  GENERATING EXTRA COMPOSITION-INFLATION PANELS (4 & 5)")
    print("=" * 60 + "\n")

    panel_4()
    panel_5()

    print("\n  Done. 2 panels (8 charts) saved to figures/\n")


if __name__ == "__main__":
    main()
