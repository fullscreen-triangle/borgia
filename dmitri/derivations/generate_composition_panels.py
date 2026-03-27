#!/usr/bin/env python3
"""
Generate 3 publication-quality panels (4 charts each, 12 total) for the
Composition-Inflation derivation.

Each panel: 1 row x 4 columns, figsize=(20,5), white background, 300 DPI,
no chart titles, minimal axis labels with LaTeX symbols, at least one 3D
chart per panel.

Usage:
    python generate_composition_panels.py
"""

import math
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
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
t_P = 5.391246e-44
l_P = 1.616255e-35
CAESIUM_FREQ = 9_192_631_770

# ── core functions ───────────────────────────────────────────────────────

def T(n, d):
    """Labeled composition count: T(n,d) = d·(d+1)^(n-1)."""
    return d * (d + 1) ** (n - 1)


def T_float(n, d):
    """Float version for large n."""
    return d * (d + 1) ** (n - 1)


def log10_T(n, d):
    """log10(T(n,d)) = log10(d) + (n-1)*log10(d+1)."""
    return math.log10(d) + (n - 1) * math.log10(d + 1)


def compute_planck_depth(freq, d=3):
    """Compute Planck depth n_P."""
    tau_osc = 1.0 / freq
    ratio = tau_osc / (d * t_P)
    n_P = 1 + math.ceil(math.log(ratio) / math.log(d + 1))
    return n_P, tau_osc


# ── color palette ────────────────────────────────────────────────────────
TEAL = "#0d9488"
AMBER = "#d97706"
CORAL = "#ef4444"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
GREEN = "#22c55e"
GREY = "#6b7280"
SLATE = "#475569"

# Oscillators
OSCILLATORS = {
    "Caesium-133": 9.192631770e9,
    "H maser": 1.420405751e9,
    "Sr optical": 4.29e14,
    r"H$_2$ vib": 1.32e14,
    "CPU 3 GHz": 3.0e9,
}

# Enhancement mechanisms
MECHANISMS = [
    ("Ternary", 3.52),
    ("Multimodal", 5.00),
    ("Harmonic", 3.00),
    (r"Poincar$\acute{e}$", 66.00),
    ("Refinement", 43.43),
]


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 1 — Composition-Inflation Growth
# ═════════════════════════════════════════════════════════════════════════

def panel_1():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D surface T(n,d) ──────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ns = np.arange(1, 31)
    ds = np.arange(1, 6)
    N, D = np.meshgrid(ns, ds)
    # Compute log10(T(n,d)) for the surface
    Z = np.log10(D.astype(float)) + (N - 1) * np.log10(D.astype(float) + 1)

    surf = ax1.plot_surface(N, D, Z, cmap="viridis", alpha=0.85,
                            edgecolor="none", antialiased=True)
    ax1.set_xlabel(r"$n$", labelpad=6)
    ax1.set_ylabel(r"$d$", labelpad=6)
    ax1.set_zlabel(r"$\log_{10} T(n,d)$", labelpad=6)
    ax1.view_init(elev=25, azim=-50)
    ax1.set_box_aspect(None)

    # ── Chart 2: Semilog comparison ─────────────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)
    ns2 = np.arange(1, 61)
    T_vals = np.array([3.0 * 4.0 ** (n - 1) for n in ns2])
    lin_vals = 3.0 * ns2
    quad_vals = ns2 ** 2.0

    ax2.semilogy(ns2, T_vals, color=TEAL, linewidth=2.0, label=r"$T(n,3)$")
    ax2.semilogy(ns2, lin_vals, color=GREY, linewidth=1.2, label=r"$3n$")
    ax2.semilogy(ns2, quad_vals, color=GREY, linewidth=1.2, linestyle="--",
                 label=r"$n^2$")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$T(n,3)$")
    ax2.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax2.set_xlim(1, 60)

    # ── Chart 3: Stacked bar (composition breakdown) ────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    n_max_bar = 8
    cmap_bar = plt.get_cmap("tab10")

    bottoms = np.zeros(n_max_bar)
    bars_n = np.arange(1, n_max_bar + 1)
    d = 3
    max_k = n_max_bar
    for k in range(1, max_k + 1):
        contributions = []
        for n in bars_n:
            if k <= n:
                val = math.comb(n - 1, k - 1) * d ** k
            else:
                val = 0
            contributions.append(val)
        contributions = np.array(contributions, dtype=float)
        ax3.bar(bars_n, contributions, bottom=bottoms, width=0.7,
                color=cmap_bar(k - 1), label=f"$k={k}$", edgecolor="white",
                linewidth=0.3)
        bottoms += contributions

    ax3.set_xlabel(r"$n$")
    ax3.set_ylabel(r"$T(n,3)$")
    ax3.legend(fontsize=5.5, ncol=2, loc="upper left", framealpha=0.9)
    ax3.set_xticks(bars_n)

    # ── Chart 4: Inflation factor T(n,3)/(3n) ──────────────────────────
    ax4 = fig.add_subplot(1, 4, 4)
    ns4 = np.arange(1, 51)
    T_over_3n = np.array([3.0 * 4.0 ** (n - 1) / (3.0 * n) for n in ns4])

    ax4.semilogy(ns4, T_over_3n, color=TEAL, linewidth=2.0)
    ax4.axvline(x=56, color=CORAL, linewidth=1.0, linestyle="--", alpha=0.7)
    ax4.annotate(r"$n_P = 56$", xy=(56, T_over_3n[-1] * 0.01), fontsize=7,
                 color=CORAL, ha="center")
    # n=56 is off chart range (1..50), so annotate at edge
    ax4.set_xlabel(r"$n$")
    ax4.set_ylabel(r"$T(n,3)\,/\,(3n)$")
    ax4.set_xlim(1, 50)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_1_composition_growth.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 2 — Planck Depth and Angular Resolution
# ═════════════════════════════════════════════════════════════════════════

def panel_2():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    tau_cs = 1.0 / CAESIUM_FREQ

    # ── Chart 1: 3D ribbons Δθ vs n for d=2,3,4 ────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ns = np.arange(1, 71)
    d_vals = [2, 3, 4]
    d_colors = [BLUE, TEAL, AMBER]

    # Planck angular threshold (for caesium)
    planck_angular = 2.0 * np.pi * t_P / tau_cs
    log_planck = np.log10(planck_angular)

    for idx, d in enumerate(d_vals):
        log_dtheta = np.array([np.log10(2.0 * np.pi) - log10_T(int(n), d)
                               for n in ns])
        # Ribbon: plot as a filled band with width in d-direction
        d_arr = np.full_like(ns, float(d))
        ax1.plot(ns, d_arr, log_dtheta, color=d_colors[idx], linewidth=2.0,
                 label=f"$d={d}$", zorder=3)

    # Planck plane
    nn_plane, dd_plane = np.meshgrid(np.linspace(1, 70, 20),
                                     np.linspace(1.5, 4.5, 5))
    zz_plane = np.full_like(nn_plane, log_planck)
    ax1.plot_surface(nn_plane, dd_plane, zz_plane, alpha=0.15, color=CORAL,
                     edgecolor="none")

    ax1.set_xlabel(r"$n$", labelpad=6)
    ax1.set_ylabel(r"$d$", labelpad=6)
    ax1.set_zlabel(r"$\log_{10}\Delta\theta$", labelpad=6)
    ax1.view_init(elev=20, azim=-55)
    ax1.legend(fontsize=6, loc="upper right")

    # ── Chart 2: Bar chart of n_P for 5 oscillators ────────────────────
    ax2 = fig.add_subplot(1, 4, 2)
    osc_names = list(OSCILLATORS.keys())
    osc_freqs = list(OSCILLATORS.values())
    osc_nP = []
    for freq in osc_freqs:
        nP, _ = compute_planck_depth(freq)
        osc_nP.append(nP)

    # Color by log frequency
    log_freqs = np.log10(osc_freqs)
    norm = Normalize(vmin=min(log_freqs), vmax=max(log_freqs))
    cmap_freq = plt.get_cmap("plasma")
    bar_colors = [cmap_freq(norm(lf)) for lf in log_freqs]

    bars = ax2.bar(range(len(osc_names)), osc_nP, color=bar_colors,
                   edgecolor="white", linewidth=0.5, width=0.65)
    ax2.set_xticks(range(len(osc_names)))
    ax2.set_xticklabels(osc_names, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel(r"$n_P$")
    ax2.set_ylim(40, 62)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap_freq, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label(r"$\log_{10}\nu$ (Hz)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # ── Chart 3: log Δθ vs n for d=3 with markers ──────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    ns3 = np.arange(1, 101)
    log_delta_theta = np.array([np.log10(2.0 * np.pi) - log10_T(int(n), 3)
                                for n in ns3])
    ax3.plot(ns3, log_delta_theta, color=TEAL, linewidth=2.0)
    ax3.axhline(y=log_planck, color=CORAL, linewidth=1.0, linestyle="--",
                alpha=0.8)
    ax3.axvline(x=56, color=SLATE, linewidth=1.0, linestyle=":", alpha=0.7)
    ax3.annotate(r"$n_P=56$", xy=(57, log_delta_theta[0] * 0.3),
                 fontsize=7, color=SLATE)
    ax3.annotate("Planck", xy=(2, log_planck + 0.8), fontsize=7, color=CORAL)
    ax3.set_xlabel(r"$n$")
    ax3.set_ylabel(r"$\log_{10}\Delta\theta$")
    ax3.set_xlim(1, 100)

    # ── Chart 4: Physical time to Planck depth vs frequency ─────────────
    ax4 = fig.add_subplot(1, 4, 4)
    phys_times = []
    for name, freq in OSCILLATORS.items():
        nP, _ = compute_planck_depth(freq)
        phys_t = nP / freq
        phys_times.append(phys_t)

    ax4.scatter(osc_freqs, phys_times, c=bar_colors, s=80, edgecolors="black",
                linewidths=0.5, zorder=3)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel(r"$\nu$ (Hz)")
    ax4.set_ylabel(r"$n_P / \nu$ (s)")

    # Annotate each point
    for i, name in enumerate(osc_names):
        offset = (8, 5) if i % 2 == 0 else (8, -12)
        ax4.annotate(name, xy=(osc_freqs[i], phys_times[i]), fontsize=6,
                     xytext=offset, textcoords="offset points", color=SLATE)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_2_planck_depth.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  PANEL 3 — Unification and Categorical Content
# ═════════════════════════════════════════════════════════════════════════

def panel_3():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ── Chart 1: 3D enhancement mechanisms + composition-inflation ──────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")

    mech_names = [m[0] for m in MECHANISMS]
    mech_logs = [m[1] for m in MECHANISMS]
    cumulative = np.cumsum(mech_logs)

    # Plot mechanism points
    indices = np.arange(1, len(MECHANISMS) + 1)
    scatter_colors = [BLUE, GREEN, AMBER, PURPLE, CORAL]
    ax1.scatter(indices, mech_logs, cumulative, c=scatter_colors, s=60,
                depthshade=False, edgecolors="black", linewidths=0.5, zorder=5)

    # Connecting lines (vertical stems to cumulative)
    for i, (idx, log_e, cum) in enumerate(zip(indices, mech_logs, cumulative)):
        ax1.plot([idx, idx], [log_e, log_e], [0, cum], color=scatter_colors[i],
                 linewidth=1.0, alpha=0.5)

    # Composition-inflation curve
    ns_curve = np.linspace(1, 250, 300)
    log_T_curve = np.array([math.log10(3) + (n - 1) * math.log10(4)
                            for n in ns_curve])
    # Map onto the 3D space: x=n/50 (scaled to fit), y=log_T, z=log_T
    ax1.plot(ns_curve / 50.0, log_T_curve, log_T_curve, color=TEAL,
             linewidth=1.5, alpha=0.8, label=r"$T(n,3)$")

    # Mark n=201
    log_T_201 = math.log10(3) + 200 * math.log10(4)
    ax1.scatter([201 / 50.0], [log_T_201], [log_T_201], color=TEAL, s=80,
                marker="*", edgecolors="black", linewidths=0.5, zorder=6)

    ax1.set_xlabel("Mechanism / " + r"$n/50$", fontsize=7, labelpad=6)
    ax1.set_ylabel(r"$\log_{10}$", fontsize=7, labelpad=6)
    ax1.set_zlabel("Cumulative", fontsize=7, labelpad=6)
    ax1.view_init(elev=20, azim=-45)

    # ── Chart 2: Horizontal bar chart of mechanisms ─────────────────────
    ax2 = fig.add_subplot(1, 4, 2)

    bar_names = [m[0] for m in MECHANISMS] + [r"Comp-Infl ($n$=201)"]
    bar_vals = [m[1] for m in MECHANISMS] + [log_T_201]
    bar_colors_h = [BLUE, GREEN, AMBER, PURPLE, CORAL, TEAL]

    y_pos = np.arange(len(bar_names))
    ax2.barh(y_pos, bar_vals, color=bar_colors_h, edgecolor="white",
             linewidth=0.5, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(bar_names, fontsize=7)
    ax2.set_xlabel(r"$\log_{10}$ enhancement")

    # Annotate values on bars
    for i, v in enumerate(bar_vals):
        ax2.text(v + 1.0, i, f"{v:.1f}", va="center", fontsize=6, color=SLATE)

    # Vertical line at sum = 120.95
    total = sum(m[1] for m in MECHANISMS)
    ax2.axvline(x=total, color=SLATE, linewidth=1.0, linestyle="--", alpha=0.6)
    ax2.annotate(f"Sum = {total:.2f}", xy=(total, len(bar_names) - 0.5),
                 fontsize=6, color=SLATE, ha="right")

    # ── Chart 3: Categorical content of 1 second ───────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    ns3 = np.arange(1, 101)
    log10_T_vals = np.array([math.log10(3) + (n - 1) * math.log10(4)
                             for n in ns3])
    ax3.plot(ns3, log10_T_vals, color=TEAL, linewidth=2.0)
    ax3.set_xlabel(r"$n$")
    ax3.set_ylabel(r"$\log_{10} T(n,3)$")

    # Annotate what 1 second of caesium means
    N_cs = CAESIUM_FREQ
    log_T_1sec = math.log10(3) + (N_cs - 1) * math.log10(4)
    ax3.annotate(
        f"1 s of Cs-133:\n"
        r"$n = 9.19 \times 10^9$" + "\n"
        r"$\sim 10^{5.53 \times 10^9}$ trajectories",
        xy=(70, 45), fontsize=6.5, color=SLATE,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=SLATE,
                  alpha=0.8),
    )
    ax3.set_xlim(1, 100)

    # ── Chart 4: Distance resolution ΔD = 2πr₀/T(n,3) ────────────────
    ax4 = fig.add_subplot(1, 4, 4)
    r0 = 1e-10  # atomic scale, 1 angstrom
    ns4 = np.arange(1, 101)
    # log10(ΔD) = log10(2π r0) - log10(T(n,3))
    log_2pi_r0 = math.log10(2.0 * math.pi * r0)
    log_delta_D = np.array([log_2pi_r0 - log10_T(int(n), 3) for n in ns4])

    ax4.plot(ns4, log_delta_D, color=TEAL, linewidth=2.0)

    # Horizontal lines
    log_lP = math.log10(l_P)
    log_r0 = math.log10(r0)
    ax4.axhline(y=log_lP, color=CORAL, linewidth=1.0, linestyle="--", alpha=0.8)
    ax4.axhline(y=log_r0, color=BLUE, linewidth=1.0, linestyle="--", alpha=0.6)
    ax4.annotate(r"$\ell_P$", xy=(2, log_lP + 0.8), fontsize=7, color=CORAL)
    ax4.annotate(r"$r_0 = 10^{-10}$ m", xy=(2, log_r0 + 0.8), fontsize=7,
                 color=BLUE)

    # Find crossing below Planck length
    crossing_n = None
    for n in ns4:
        if log_2pi_r0 - log10_T(int(n), 3) < log_lP:
            crossing_n = int(n)
            break
    if crossing_n:
        ax4.axvline(x=crossing_n, color=SLATE, linewidth=0.8, linestyle=":",
                    alpha=0.6)
        ax4.annotate(f"$n={crossing_n}$", xy=(crossing_n + 1, log_delta_D[0] * 0.5),
                     fontsize=7, color=SLATE)

    ax4.set_xlabel(r"$n$")
    ax4.set_ylabel(r"$\log_{10}\,\Delta D$ (m)")
    ax4.set_xlim(1, 100)

    fig.tight_layout(pad=1.5)
    out = FIGDIR / "panel_3_unification.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  GENERATING COMPOSITION-INFLATION PANELS")
    print("=" * 60 + "\n")

    panel_1()
    panel_2()
    panel_3()

    print("\n  Done. 3 panels (12 charts) saved to figures/\n")


if __name__ == "__main__":
    main()
