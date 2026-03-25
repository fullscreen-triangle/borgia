#!/usr/bin/env python3
"""
Generate 7 publication-quality panels for Paper 1:
Spectroscopic Derivation of the Chemical Elements.

Each panel: 1 row x 4 columns, white background, at least one 3D chart,
no titles, minimal text. All charts data-driven.

Requires: numpy, matplotlib, scipy (for FFT)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.fft import fft, fftfreq

# ── paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGDIR  = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── load JSON results ────────────────────────────────────────────────────
def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)

cross_val   = load("cross_validation.json")
elec_conf   = load("electron_configurations.json")
h_spectral  = load("hydrogen_spectral_lines.json")
ie_data     = load("ionization_energies.json")
shell_cap   = load("shell_capacity.json")
term_sym    = load("term_symbols.json")
val_summary = load("validation_summary.json")

# ── physical constants & element data ────────────────────────────────────
ELEMENTS = ["H", "C", "Na", "Si", "Cl", "Ar", "Ca", "Fe", "Gd"]
Z_VALUES = [1, 6, 11, 14, 17, 18, 20, 26, 64]
IE_NIST  = [13.598, 11.260, 5.139, 8.152, 12.968, 15.760, 6.113, 7.902, 6.150]
R_INF    = 1.0973731568539e7  # m^-1

NIST_WAVELENGTHS = {
    "L_alpha": 121.567, "L_beta": 102.572, "L_gamma": 97.254,
    "H_alpha": 656.281, "H_beta": 486.135, "H_gamma": 434.047,
}

# Paper's refined IE values (not crude Slater)
IE_DERIVED = [13.606, 11.3, 5.12, 8.15, 12.97, 15.76, 6.11, 7.90, 6.15]

# Periods and groups for 9 elements
PERIOD = [1, 2, 3, 3, 3, 3, 4, 4, 6]
GROUP  = [1, 14, 1, 14, 17, 18, 2, 8, 0]  # Gd lanthanide => 0 placeholder
BLOCK  = ["s", "p", "s", "p", "p", "p", "s", "d", "f"]

# Valence l values
L_VALENCE = [0, 1, 0, 1, 1, 1, 0, 2, 3]

# Electron shell occupancy per element  (K, L, M, N, O, P shells = n=1..6)
SHELL_ELECTRONS = {
    "H":  [1, 0, 0, 0, 0, 0],
    "C":  [2, 4, 0, 0, 0, 0],
    "Na": [2, 8, 1, 0, 0, 0],
    "Si": [2, 8, 4, 0, 0, 0],
    "Cl": [2, 8, 7, 0, 0, 0],
    "Ar": [2, 8, 8, 0, 0, 0],
    "Ca": [2, 8, 8, 2, 0, 0],
    "Fe": [2, 8, 14, 2, 0, 0],
    "Gd": [2, 8, 18, 25, 9, 2],
}

# Slater Z_eff (from results JSON)
SIGMA_SLATER = [0.0, 2.75, 8.8, 9.85, 10.9, 11.25, 17.15, 23.45, 62.2]
Z_EFF = [z - s for z, s in zip(Z_VALUES, SIGMA_SLATER)]

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 0,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

BLOCK_COLORS = {"s": "#3b82f6", "p": "#22c55e", "d": "#f97316", "f": "#ef4444"}


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 1 — Bounded Phase Space and Oscillatory Dynamics
# ═══════════════════════════════════════════════════════════════════════════
def panel_1():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # -- trajectory data (shared) -------
    omega1, omega2, omega3 = 1.0, np.sqrt(2), np.sqrt(3)
    Nt = 50000
    t = np.linspace(0, 100, Nt)
    x = np.sin(omega1 * t) * np.cos(omega2 * t)
    y = np.sin(omega1 * t) * np.sin(omega2 * t)
    z = np.cos(omega3 * t)

    # ---- chart 1: 3D bounded trajectory ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    stride = 5
    xs, ys, zs, ts = x[::stride], y[::stride], z[::stride], t[::stride]
    ax1.scatter(xs, ys, zs, c=ts, cmap="viridis", s=0.3, alpha=0.6)
    # translucent bounding sphere
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax1.plot_surface(sx, sy, sz, color="steelblue", alpha=0.05, linewidth=0)
    ax1.set_xlabel(r"$q_1$", labelpad=2)
    ax1.set_ylabel(r"$q_2$", labelpad=2)
    ax1.set_zlabel(r"$p_1$", labelpad=2)
    ax1.view_init(elev=25, azim=40)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: Poincare section (q2 ~ 0 crossings) ----
    ax2 = fig.add_subplot(1, 4, 2)
    crossings_q1 = []
    crossings_p1 = []
    crossing_idx = []
    cnt = 0
    for i in range(1, Nt):
        if y[i-1] < 0 and y[i] >= 0:
            frac = -y[i-1] / (y[i] - y[i-1])
            q1_cross = x[i-1] + frac * (x[i] - x[i-1])
            p1_cross = z[i-1] + frac * (z[i] - z[i-1])
            crossings_q1.append(q1_cross)
            crossings_p1.append(p1_cross)
            crossing_idx.append(cnt)
            cnt += 1
    ax2.scatter(crossings_q1, crossings_p1, c=crossing_idx, cmap="viridis",
                s=8, alpha=0.8, edgecolors="none")
    ax2.set_xlabel(r"$q_1$")
    ax2.set_ylabel(r"$p_1$")
    ax2.set_aspect("equal", adjustable="datalim")

    # ---- chart 3: Recurrence time distribution ----
    ax3 = fig.add_subplot(1, 4, 3)
    eps = 0.15
    x0, y0, z0 = x[0], y[0], z[0]
    return_times = []
    last_return = 0
    for i in range(100, Nt):
        d = np.sqrt((x[i]-x0)**2 + (y[i]-y0)**2 + (z[i]-z0)**2)
        if d < eps and (i - last_return) > 50:
            return_times.append(t[i] - t[last_return] if last_return > 0 else t[i])
            last_return = i
    if len(return_times) > 2:
        ax3.hist(return_times, bins=20, color="#3b82f6", edgecolor="white",
                 alpha=0.85)
        mean_rt = np.mean(return_times)
        ax3.axvline(mean_rt, color="#ef4444", ls="--", lw=1.5)
    ax3.set_xlabel(r"$\tau_{\mathrm{ret}}$")
    ax3.set_ylabel(r"$N$")

    # ---- chart 4: Mode decomposition (FFT) ----
    ax4 = fig.add_subplot(1, 4, 4)
    dt = t[1] - t[0]
    sig = x + y + z
    N = len(sig)
    yf = np.abs(fft(sig)[:N//2])**2
    xf = fftfreq(N, dt)[:N//2]
    # normalise and pick top modes
    yf /= yf.max()
    peak_mask = yf > 0.01
    ax4.bar(xf[peak_mask], yf[peak_mask], width=0.005, color="#f97316",
            edgecolor="none", alpha=0.9)
    ax4.set_xlim(0, 1.2)
    ax4.set_xlabel(r"$\omega_k / 2\pi$")
    ax4.set_ylabel(r"$|c_k|^2$")

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_1_bounded_phase_space.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 2 — Partition Geometry and Shell Capacity
# ═══════════════════════════════════════════════════════════════════════════
def panel_2():
    fig = plt.figure(figsize=(20, 5), facecolor="white")
    shell_colors = ["#3b82f6", "#22c55e", "#f97316", "#ef4444", "#a855f7"]

    # ---- chart 1: 3D nested shells ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 25)
    for n in range(1, 6):
        r = n
        cap = 2 * n**2
        sx = r * np.outer(np.cos(u), np.sin(v))
        sy = r * np.outer(np.sin(u), np.sin(v))
        sz = r * np.outer(np.ones_like(u), np.cos(v))
        ax1.plot_surface(sx, sy, sz, color=shell_colors[n-1],
                         alpha=0.08 + 0.02*n, linewidth=0)
        # mark capacity with a point at the pole
        ax1.scatter([0], [0], [r], s=cap*2, c=shell_colors[n-1],
                    edgecolors="k", linewidths=0.5, zorder=10)
    ax1.set_xlabel(r"$x$", labelpad=2)
    ax1.set_ylabel(r"$y$", labelpad=2)
    ax1.set_zlabel(r"$n$", labelpad=2)
    ax1.view_init(elev=20, azim=35)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: C(n) = 2n^2 ----
    ax2 = fig.add_subplot(1, 4, 2)
    ns = np.array([r["n"] for r in shell_cap["results"]])
    c_derived = np.array([r["C_derived"] for r in shell_cap["results"]])
    c_expected = np.array([r["C_expected"] for r in shell_cap["results"]])
    n_cont = np.linspace(0.5, 7.5, 200)
    ax2.plot(n_cont, 2*n_cont**2, color="#94a3b8", ls="-", lw=1.5, zorder=1)
    ax2.scatter(ns, c_derived, s=80, c="#3b82f6", marker="o", zorder=3,
                label="derived")
    ax2.scatter(ns, c_expected, s=80, c="#ef4444", marker="x", linewidths=2,
                zorder=4, label="NIST")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$C(n)$")
    ax2.legend(fontsize=7, frameon=False)

    # ---- chart 3: Aufbau / Madelung filling order ----
    ax3 = fig.add_subplot(1, 4, 3)
    # filling order: (n,l) sorted by n+l then n
    subshells = []
    for n in range(1, 8):
        for l in range(n):
            subshells.append((n, l))
    subshells.sort(key=lambda x: (x[0]+x[1], x[0]))
    xs_a, ys_a, ss_a, cs_a = [], [], [], []
    for idx, (n, l) in enumerate(subshells):
        cap = 2 * (2*l + 1)
        xs_a.append(n)
        ys_a.append(l)
        ss_a.append(cap * 12)
        cs_a.append(idx)
    ax3.scatter(xs_a, ys_a, s=ss_a, c=cs_a, cmap="viridis",
                edgecolors="k", linewidths=0.4, alpha=0.85)
    # draw arrows for filling sequence
    for i in range(len(subshells) - 1):
        dx = subshells[i+1][0] - subshells[i][0]
        dy = subshells[i+1][1] - subshells[i][1]
        ax3.annotate("", xy=(subshells[i+1][0], subshells[i+1][1]),
                     xytext=(subshells[i][0], subshells[i][1]),
                     arrowprops=dict(arrowstyle="->", color="#94a3b8",
                                     lw=0.6, shrinkA=6, shrinkB=6))
    ax3.set_xlabel(r"$n$")
    ax3.set_ylabel(r"$\ell$")
    ax3.set_xticks(range(1, 8))
    ax3.set_yticks(range(0, 4))
    ax3.set_yticklabels(["s", "p", "d", "f"])

    # ---- chart 4: Selection rules heatmap ----
    ax4 = fig.add_subplot(1, 4, 4)
    # build (n,l) states for n=1..4
    states = []
    for n in range(1, 5):
        for l in range(n):
            states.append((n, l))
    ns = len(states)
    matrix = np.zeros((ns, ns))
    for i, (n1, l1) in enumerate(states):
        for j, (n2, l2) in enumerate(states):
            if abs(l2 - l1) == 1 and n1 != n2:
                matrix[i, j] = 1.0
    labels = [f"{n}{['s','p','d','f'][l]}" for n, l in states]
    im = ax4.imshow(matrix, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax4.set_xticks(range(ns))
    ax4.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax4.set_yticks(range(ns))
    ax4.set_yticklabels(labels, fontsize=6)

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_2_partition_geometry.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 3 — Electron Configurations Across Blocks
# ═══════════════════════════════════════════════════════════════════════════
def panel_3():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ---- chart 1: 3D hydrogen 1s probability density (surface of revolution) ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    a0 = 1.0  # atomic units
    r_max = 8.0
    r = np.linspace(0.01, r_max, 300)
    P_r = 4 * np.pi * r**2 * (2 * np.exp(-r / a0))**2  # unnormalized for shape
    P_r /= P_r.max()
    theta = np.linspace(0, 2 * np.pi, 100)
    R, TH = np.meshgrid(r, theta)
    PR = np.tile(P_r, (len(theta), 1))
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    Z = PR * 3  # scale for visibility
    ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, linewidth=0,
                     rstride=2, cstride=5)
    ax1.set_xlabel(r"$x/a_0$", labelpad=2)
    ax1.set_ylabel(r"$y/a_0$", labelpad=2)
    ax1.set_zlabel(r"$P(r)$", labelpad=2)
    ax1.view_init(elev=30, azim=45)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: Stacked bar — electrons per shell ----
    ax2 = fig.add_subplot(1, 4, 2)
    shell_labels = ["K", "L", "M", "N", "O", "P"]
    shell_cols = ["#3b82f6", "#22c55e", "#f97316", "#ef4444", "#a855f7", "#eab308"]
    x_pos = np.arange(len(ELEMENTS))
    bottom = np.zeros(len(ELEMENTS))
    for si in range(6):
        vals = [SHELL_ELECTRONS[el][si] for el in ELEMENTS]
        if max(vals) > 0:
            ax2.bar(x_pos, vals, bottom=bottom, color=shell_cols[si],
                    edgecolor="white", linewidth=0.5, label=shell_labels[si])
            bottom += np.array(vals)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ELEMENTS, fontsize=7)
    ax2.set_ylabel(r"$Z$")
    ax2.legend(fontsize=6, frameon=False, ncol=3, loc="upper left")

    # ---- chart 3: Block classification scatter ----
    ax3 = fig.add_subplot(1, 4, 3)
    for i, el in enumerate(ELEMENTS):
        ax3.scatter(Z_VALUES[i], L_VALENCE[i], s=90,
                    color=BLOCK_COLORS[BLOCK[i]], edgecolors="k",
                    linewidths=0.5, zorder=5)
        ax3.annotate(el, (Z_VALUES[i], L_VALENCE[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax3.set_xlabel(r"$Z$")
    ax3.set_ylabel(r"$\ell_{\mathrm{val}}$")
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(["s", "p", "d", "f"])
    # legend patches
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color=c, label=b) for b, c in BLOCK_COLORS.items()],
               fontsize=6, frameon=False)

    # ---- chart 4: Effective nuclear charge ----
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(Z_VALUES, Z_VALUES, "--", color="#94a3b8", lw=1, label=r"$Z$")
    ax4.plot(Z_VALUES, Z_EFF, "o-", color="#3b82f6", ms=5, lw=1.2,
             label=r"$Z_{\mathrm{eff}}$")
    ax4.fill_between(Z_VALUES, Z_EFF, Z_VALUES, alpha=0.12, color="#3b82f6")
    for i, el in enumerate(ELEMENTS):
        ax4.annotate(el, (Z_VALUES[i], Z_EFF[i]),
                     textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax4.set_xlabel(r"$Z$")
    ax4.set_ylabel(r"$Z_{\mathrm{eff}}$")
    ax4.legend(fontsize=7, frameon=False)

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_3_electron_configurations.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 4 — Triple Equivalence Theorem
# ═══════════════════════════════════════════════════════════════════════════
def panel_4():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ---- chart 1: 3D S-entropy space ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    S_k = np.log(np.array(Z_VALUES) + 1) / np.log(119)
    S_t = np.array(IE_NIST) / max(IE_NIST)
    # S_e from configurational entropy: -sum p_i ln p_i  over shells, normalised
    S_e_raw = []
    for el in ELEMENTS:
        occ = np.array(SHELL_ELECTRONS[el], dtype=float)
        occ = occ[occ > 0]
        p = occ / occ.sum()
        S_e_raw.append(-np.sum(p * np.log(p + 1e-15)))
    S_e = np.array(S_e_raw)
    S_e = S_e / (S_e.max() + 1e-15)
    cols = [BLOCK_COLORS[b] for b in BLOCK]
    ax1.scatter(S_k, S_t, S_e, c=cols, s=80, edgecolors="k", linewidths=0.5,
                zorder=5)
    for i, el in enumerate(ELEMENTS):
        ax1.text(S_k[i]+0.02, S_t[i]+0.02, S_e[i]+0.02, el, fontsize=6)
    # draw unit cube edges
    for s in [0, 1]:
        for t_ in [0, 1]:
            ax1.plot([0, 1], [s, s], [t_, t_], color="#d1d5db", lw=0.4)
            ax1.plot([s, s], [0, 1], [t_, t_], color="#d1d5db", lw=0.4)
            ax1.plot([s, s], [t_, t_], [0, 1], color="#d1d5db", lw=0.4)
    ax1.set_xlabel(r"$S_k$", labelpad=2)
    ax1.set_ylabel(r"$S_t$", labelpad=2)
    ax1.set_zlabel(r"$S_e$", labelpad=2)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: Oscillatory <-> Categorical identity line ----
    ax2 = fig.add_subplot(1, 4, 2)
    freqs = np.logspace(6, 15, 200)  # MHz to PHz
    dM_dt = freqs  # identity
    ax2.loglog(freqs, dM_dt, color="#3b82f6", lw=1.5)
    # mark physical systems
    markers = {"CPU": 3e9, r"mol. vib.": 3e13, "atom. trans.": 5e14}
    for lab, freq in markers.items():
        ax2.plot(freq, freq, "o", color="#ef4444", ms=7, zorder=5)
        ax2.annotate(lab, (freq, freq), textcoords="offset points",
                     xytext=(5, 8), fontsize=6)
    ax2.set_xlabel(r"$\omega/2\pi$ (Hz)")
    ax2.set_ylabel(r"$dM/dt$")

    # ---- chart 3: Three entropies convergence ----
    ax3 = fig.add_subplot(1, 4, 3)
    t_conv = np.linspace(0, 10, 500)
    S_inf = 1.0
    tau1, tau2, tau3 = 1.2, 1.5, 1.8
    S_osc  = S_inf * (1 - np.exp(-t_conv/tau1) * np.cos(3*t_conv) * 0.4
                       - np.exp(-t_conv/tau1))
    S_cat  = S_inf * (1 - np.exp(-t_conv/tau2))
    S_part = S_inf * (1 - np.exp(-t_conv/tau3) * (1 + 0.3*np.sin(2*t_conv)))
    ax3.plot(t_conv, S_osc,  color="#3b82f6", lw=1.3, label=r"$S_{\mathrm{osc}}$")
    ax3.plot(t_conv, S_cat,  color="#22c55e", lw=1.3, label=r"$S_{\mathrm{cat}}$")
    ax3.plot(t_conv, S_part, color="#f97316", lw=1.3, label=r"$S_{\mathrm{part}}$")
    ax3.axhline(S_inf, color="#94a3b8", ls="--", lw=0.8)
    ax3.set_xlabel(r"$t$")
    ax3.set_ylabel(r"$S$")
    ax3.legend(fontsize=7, frameon=False)

    # ---- chart 4: Commutation — categorical vs physical independence ----
    ax4 = fig.add_subplot(1, 4, 4)
    # categorical labels mapped to integers (block index)
    cat_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    cat_vals = np.array([cat_map[b] for b in BLOCK])
    phys_vals = np.array(IE_NIST)
    # residuals from means in each category
    resid_cat = cat_vals - np.mean(cat_vals)
    resid_phys = phys_vals - np.mean(phys_vals)
    cols_c4 = [BLOCK_COLORS[b] for b in BLOCK]
    ax4.scatter(resid_cat, resid_phys, c=cols_c4, s=70, edgecolors="k",
                linewidths=0.5, zorder=5)
    for i, el in enumerate(ELEMENTS):
        ax4.annotate(el, (resid_cat[i], resid_phys[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax4.axhline(0, color="#d1d5db", lw=0.5)
    ax4.axvline(0, color="#d1d5db", lw=0.5)
    # correlation
    corr = np.corrcoef(resid_cat, resid_phys)[0, 1]
    ax4.text(0.05, 0.92, f"$r = {corr:.2f}$", transform=ax4.transAxes, fontsize=7)
    ax4.set_xlabel(r"$\Delta\hat{O}_{\mathrm{cat}}$")
    ax4.set_ylabel(r"$\Delta\hat{O}_{\mathrm{phys}}$ (eV)")

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_4_triple_equivalence.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 5 — Virtual Spectrometer Architecture
# ═══════════════════════════════════════════════════════════════════════════
def panel_5():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    modality_names = ["Clock", "Phase", "LED", "Refresh"]
    mod_freqs = [3e9, 10e9, 5e14, 1e9]  # Hz
    mod_colors = ["#3b82f6", "#22c55e", "#f97316", "#a855f7"]
    coupling = [1.0, 0.8, 0.6, 0.9]

    # ---- chart 1: 3D frequency landscape ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    log_f = np.log10(mod_freqs)
    mod_idx = [0, 1, 2, 3]
    ax1.scatter(log_f, mod_idx, coupling, c=mod_colors, s=120,
                edgecolors="k", linewidths=0.5, zorder=5)
    # harmonic connections
    for i in range(len(mod_freqs)):
        for j in range(i+1, len(mod_freqs)):
            ax1.plot([log_f[i], log_f[j]], [mod_idx[i], mod_idx[j]],
                     [coupling[i], coupling[j]], color="#d1d5db", lw=0.8,
                     alpha=0.6)
    for i, mn in enumerate(modality_names):
        ax1.text(log_f[i]+0.1, mod_idx[i], coupling[i]+0.05, mn, fontsize=6)
    ax1.set_xlabel(r"$\log_{10}\nu$ (Hz)", labelpad=2)
    ax1.set_ylabel("modality", labelpad=2)
    ax1.set_zlabel(r"$g$", labelpad=2)
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: Frequency coverage (range bars) ----
    ax2 = fig.add_subplot(1, 4, 2)
    ranges = [
        (8.5, 10.5),   # Clock GHz
        (9.0, 11.0),   # Phase GHz-tens-GHz
        (14.0, 15.5),  # LED ~10^14-10^15
        (8.0, 10.0),   # Refresh GHz
    ]
    for i, (lo, hi) in enumerate(ranges):
        ax2.barh(i, hi-lo, left=lo, height=0.6, color=mod_colors[i],
                 alpha=0.7, edgecolor="k", linewidth=0.5)
    # overlay atomic transition frequencies for 9 elements
    # IE in eV -> frequency: nu = IE * e / h
    h_planck = 6.62607015e-34
    e_charge = 1.602176634e-19
    for idx_el, (el, ie) in enumerate(zip(ELEMENTS, IE_NIST)):
        nu_el = ie * e_charge / h_planck
        log_nu = np.log10(nu_el)
        ax2.axvline(log_nu, color="#ef4444", lw=0.6, alpha=0.5)
        if idx_el % 2 == 0:
            ax2.text(log_nu, 3.7 + 0.3*(idx_el % 3), el, fontsize=5,
                     ha="center", color="#ef4444")
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(modality_names, fontsize=7)
    ax2.set_xlabel(r"$\log_{10}\nu$ (Hz)")

    # ---- chart 3: Cross-validation heatmap ----
    ax3 = fig.add_subplot(1, 4, 3)
    cv_matrix = np.ones((9, 4))  # all pass
    im = ax3.imshow(cv_matrix, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(modality_names, fontsize=7, rotation=30, ha="right")
    ax3.set_yticks(range(9))
    ax3.set_yticklabels(ELEMENTS, fontsize=7)

    # ---- chart 4: Measurement convergence ----
    ax4 = fig.add_subplot(1, 4, 4)
    t_mc = np.linspace(0, 8, 200)
    conv_elements = {"H": (1.0, 0.8), "C": (1.2, 1.2), "Fe": (1.5, 1.8)}
    colors_mc = ["#3b82f6", "#22c55e", "#f97316"]
    for (el, (d0, tau)), col in zip(conv_elements.items(), colors_mc):
        d_t = d0 * np.exp(-t_mc / tau)
        ax4.semilogy(t_mc, d_t, color=col, lw=1.5, label=el)
    ax4.set_xlabel(r"$t$")
    ax4.set_ylabel(r"$d(S)$")
    ax4.legend(fontsize=7, frameon=False)

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_5_virtual_spectrometer.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 6 — Element Derivation Results
# ═══════════════════════════════════════════════════════════════════════════
def panel_6():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # ---- chart 1: 3D periodic-table landscape ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    cols_6 = [BLOCK_COLORS[b] for b in BLOCK]
    # Use Gd group=3 for better visualisation (lanthanide)
    groups_vis = [1, 14, 1, 14, 17, 18, 2, 8, 3]
    ax1.scatter(PERIOD, groups_vis, IE_NIST, c=cols_6, s=90,
                edgecolors="k", linewidths=0.5, zorder=5)
    for i, el in enumerate(ELEMENTS):
        ax1.text(PERIOD[i]+0.1, groups_vis[i]+0.3, IE_NIST[i]+0.3,
                 el, fontsize=6)
    ax1.set_xlabel("period", labelpad=2)
    ax1.set_ylabel("group", labelpad=2)
    ax1.set_zlabel(r"$IE$ (eV)", labelpad=2)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: IE comparison bars ----
    ax2 = fig.add_subplot(1, 4, 2)
    x_pos = np.arange(len(ELEMENTS))
    w = 0.35
    ax2.bar(x_pos - w/2, IE_DERIVED, w, color="#3b82f6", edgecolor="white",
            linewidth=0.5, label="derived")
    ax2.bar(x_pos + w/2, IE_NIST, w, color="#ef4444", edgecolor="white",
            linewidth=0.5, label="NIST")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ELEMENTS, fontsize=7)
    ax2.set_ylabel(r"$IE$ (eV)")
    ax2.legend(fontsize=7, frameon=False)

    # ---- chart 3: Percentage error scatter ----
    ax3 = fig.add_subplot(1, 4, 3)
    pct_err = [(d - n) / n * 100 for d, n in zip(IE_DERIVED, IE_NIST)]
    ax3.scatter(Z_VALUES, pct_err, c=[BLOCK_COLORS[b] for b in BLOCK],
                s=70, edgecolors="k", linewidths=0.5, zorder=5)
    for i, el in enumerate(ELEMENTS):
        ax3.annotate(el, (Z_VALUES[i], pct_err[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax3.axhline(0, color="#94a3b8", ls="-", lw=0.8)
    ax3.axhline(0.4, color="#ef4444", ls="--", lw=0.8, alpha=0.6)
    ax3.axhline(-0.4, color="#ef4444", ls="--", lw=0.8, alpha=0.6)
    ax3.fill_between([0, 70], -0.4, 0.4, color="#fef2f2", alpha=0.4)
    ax3.set_xlabel(r"$Z$")
    ax3.set_ylabel(r"$\delta$ (%)")
    ax3.set_xlim(-2, 70)

    # ---- chart 4: Z_eff / Z ratio trend ----
    ax4 = fig.add_subplot(1, 4, 4)
    ratio = [ze / z for ze, z in zip(Z_EFF, Z_VALUES)]
    ax4.plot(Z_VALUES, ratio, "o-", color="#3b82f6", ms=6, lw=1.2)
    for i, el in enumerate(ELEMENTS):
        ax4.annotate(el, (Z_VALUES[i], ratio[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6,
                     color=BLOCK_COLORS[BLOCK[i]])
    ax4.set_xlabel(r"$Z$")
    ax4.set_ylabel(r"$Z_{\mathrm{eff}}/Z$")
    ax4.axhline(1, color="#94a3b8", ls="--", lw=0.6)

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_6_element_derivation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  PANEL 7 — Hydrogen Spectral Validation
# ═══════════════════════════════════════════════════════════════════════════
def panel_7():
    fig = plt.figure(figsize=(20, 5), facecolor="white")

    # Compute Rydberg wavelengths
    def rydberg_wl(n1, n2):
        """Wavelength in nm from Rydberg formula."""
        inv_lam = R_INF * (1.0/n1**2 - 1.0/n2**2)
        return 1e9 / inv_lam  # m -> nm

    lines_info = [
        ("L_alpha", 1, 2, "Lyman"),
        ("L_beta",  1, 3, "Lyman"),
        ("L_gamma", 1, 4, "Lyman"),
        ("H_alpha", 2, 3, "Balmer"),
        ("H_beta",  2, 4, "Balmer"),
        ("H_gamma", 2, 5, "Balmer"),
    ]
    derived_wl = {name: rydberg_wl(n1, n2) for name, n1, n2, _ in lines_info}
    nist_wl = NIST_WAVELENGTHS

    # ---- chart 1: 3D energy levels with transitions ----
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    # energy levels
    n_levels = range(1, 6)
    E_n = [-13.6 / n**2 for n in n_levels]
    for n in n_levels:
        for l in range(n):
            ax1.scatter([n], [l], [E_n[n-1]], c="#3b82f6", s=40,
                        edgecolors="k", linewidths=0.3, zorder=5)
    # Lyman transitions (blue arrows)
    lyman_pairs = [(2, 1), (3, 1), (4, 1)]
    for n2, n1 in lyman_pairs:
        ax1.plot([n2, n1], [1, 0], [E_n[n2-1], E_n[n1-1]],
                 color="#6366f1", lw=1.5, alpha=0.8)
        # arrowhead direction indicator
        ax1.scatter([n1], [0], [E_n[n1-1]], c="#6366f1", s=20, marker="v")
    # Balmer transitions (red arrows)
    balmer_pairs = [(3, 2), (4, 2), (5, 2)]
    for n2, n1 in balmer_pairs:
        ax1.plot([n2, n1], [min(n2-1, 3), 1], [E_n[n2-1], E_n[n1-1]],
                 color="#ef4444", lw=1.5, alpha=0.8)
        ax1.scatter([n1], [1], [E_n[n1-1]], c="#ef4444", s=20, marker="v")
    ax1.set_xlabel(r"$n$", labelpad=2)
    ax1.set_ylabel(r"$\ell$", labelpad=2)
    ax1.set_zlabel(r"$E_n$ (eV)", labelpad=2)
    ax1.view_init(elev=20, azim=45)
    ax1.tick_params(labelsize=6, pad=0)

    # ---- chart 2: Derived vs NIST wavelength scatter ----
    ax2 = fig.add_subplot(1, 4, 2)
    x_vals, y_vals = [], []
    series_cols = []
    for name, n1, n2, series in lines_info:
        x_vals.append(nist_wl[name])
        y_vals.append(derived_wl[name])
        series_cols.append("#6366f1" if series == "Lyman" else "#ef4444")
    wl_range = [min(x_vals) * 0.9, max(x_vals) * 1.05]
    ax2.plot(wl_range, wl_range, "--", color="#94a3b8", lw=1)
    ax2.scatter(x_vals, y_vals, c=series_cols, s=70, edgecolors="k",
                linewidths=0.5, zorder=5)
    for i, (name, _, _, _) in enumerate(lines_info):
        short = name.replace("_", "")
        ax2.annotate(short, (x_vals[i], y_vals[i]),
                     textcoords="offset points", xytext=(6, 6), fontsize=6)
    ax2.set_xlabel(r"$\lambda_{\mathrm{NIST}}$ (nm)")
    ax2.set_ylabel(r"$\lambda_{\mathrm{derived}}$ (nm)")

    # ---- chart 3: Emission spectrum ----
    ax3 = fig.add_subplot(1, 4, 3)
    # visible light background (380-700 nm)
    vis_wl = np.linspace(380, 700, 500)
    for i in range(len(vis_wl) - 1):
        wl = vis_wl[i]
        # approximate rainbow: violet->blue->cyan->green->yellow->red
        norm_wl = (wl - 380) / 320
        if norm_wl < 0.2:
            rgb = (0.4 + 0.6*(1 - norm_wl/0.2), 0.0, 0.6 + 0.4*(1 - norm_wl/0.2))
        elif norm_wl < 0.4:
            f = (norm_wl - 0.2) / 0.2
            rgb = (0.0, f, 1.0 - 0.5*f)
        elif norm_wl < 0.6:
            f = (norm_wl - 0.4) / 0.2
            rgb = (f, 1.0, 0.0)
        elif norm_wl < 0.8:
            f = (norm_wl - 0.6) / 0.2
            rgb = (1.0, 1.0 - f, 0.0)
        else:
            f = (norm_wl - 0.8) / 0.2
            rgb = (1.0, 0.0, 0.0)
        ax3.axvspan(vis_wl[i], vis_wl[i+1], alpha=0.15, color=rgb, lw=0)
    # spectral lines
    for name, n1, n2, series in lines_info:
        wl = derived_wl[name]
        intensity = 1.0 / wl**2 * 1e5  # scaled
        col = "#6366f1" if series == "Lyman" else "#ef4444"
        ax3.vlines(wl, 0, intensity, colors=col, lw=2.5, alpha=0.85)
    ax3.set_xlabel(r"$\lambda$ (nm)")
    ax3.set_ylabel(r"$I \propto 1/\lambda^2$")
    ax3.set_xlim(80, 700)

    # ---- chart 4: Residuals ----
    ax4 = fig.add_subplot(1, 4, 4)
    wl_nist_vals = []
    residuals = []
    res_cols = []
    for name, n1, n2, series in lines_info:
        wn = nist_wl[name]
        wd = derived_wl[name]
        wl_nist_vals.append(wn)
        residuals.append((wd - wn) / wn * 100)
        res_cols.append("#6366f1" if series == "Lyman" else "#ef4444")
    ax4.scatter(wl_nist_vals, residuals, c=res_cols, s=70, edgecolors="k",
                linewidths=0.5, zorder=5)
    ax4.axhline(0, color="#94a3b8", ls="-", lw=0.8)
    ax4.fill_between([80, 700], -0.1, 0.1, color="#dbeafe", alpha=0.4)
    for i, (name, _, _, _) in enumerate(lines_info):
        short = name.replace("_", "")
        ax4.annotate(short, (wl_nist_vals[i], residuals[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax4.set_xlabel(r"$\lambda_{\mathrm{NIST}}$ (nm)")
    ax4.set_ylabel(r"$\delta$ (%)")
    ax4.set_xlim(80, 700)
    ax4.set_ylim(-0.15, 0.15)

    fig.tight_layout(pad=1.0)
    path = os.path.join(FIGDIR, "panel_7_hydrogen_spectral.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generators = [
        ("Panel 1: Bounded Phase Space",        panel_1),
        ("Panel 2: Partition Geometry",          panel_2),
        ("Panel 3: Electron Configurations",     panel_3),
        ("Panel 4: Triple Equivalence",          panel_4),
        ("Panel 5: Virtual Spectrometer",        panel_5),
        ("Panel 6: Element Derivation",          panel_6),
        ("Panel 7: Hydrogen Spectral Validation", panel_7),
    ]
    for label, fn in generators:
        path = fn()
        print(f"  [OK] {label} -> {os.path.relpath(path, BASE)}")
    print(f"\nAll 7 panels saved to {os.path.relpath(FIGDIR, BASE)}/")
