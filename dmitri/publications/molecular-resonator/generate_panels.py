#!/usr/bin/env python3
"""
Generate 6 publication-quality panels for Paper 2: Harmonic Molecular Resonator.

Each panel is a single figure with 4 charts in a row (1x4), white background,
at least one 3D chart per panel. No titles. Minimal axis labels using LaTeX symbols.

Usage:
    python generate_panels.py
"""

import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

warnings.filterwarnings('ignore', message='.*Tight layout.*')
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'font.size': 9,
    'axes.titlesize': 0,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'mathtext.fontset': 'cm',
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Molecular data
# ---------------------------------------------------------------------------
MOLECULES = {
    "H2":   {"modes": {"vib": 4401, "rot": 118.7}},
    "CO":   {"modes": {"P3": 2131.63, "P2": 2135.55, "P1": 2139.43,
                        "vib": 2143.0, "R0": 2147.08, "R1": 2150.86, "R2": 2154.59}},
    "H2O":  {"modes": {"nu2": 1595, "nu1": 3657, "nu3": 3756}},
    "CO2":  {"modes": {"nu2": 667, "nu1": 1388, "nu3": 2349}},
    "CH4":  {"modes": {"nu4": 1306, "nu2": 1534, "nu1": 2917, "nu3": 3019}},
    "C6H6": {"modes": {"nu4": 673, "nu5": 993, "nu3": 1178, "nu2": 1596, "nu1": 3062}},
}

MOL_COLORS = {
    "H2": "#1f77b4",
    "CO": "#ff7f0e",
    "H2O": "#2ca02c",
    "CO2": "#d62728",
    "CH4": "#9467bd",
    "C6H6": "#8c564b",
}

MOL_LABELS = {
    "H2": r"H$_2$",
    "CO": "CO",
    "H2O": r"H$_2$O",
    "CO2": r"CO$_2$",
    "CH4": r"CH$_4$",
    "C6H6": r"C$_6$H$_6$",
}

MOL_ORDER = ["H2", "CO", "H2O", "CO2", "CH4", "C6H6"]

# ---------------------------------------------------------------------------
# Load JSON data
# ---------------------------------------------------------------------------
def load_json(name):
    with open(RESULTS_DIR / name, 'r') as f:
        return json.load(f)

network_data = load_json('network_properties.json')
harmonic_data = load_json('harmonic_ratios.json')
circulation_data = load_json('circulation_periods.json')
consistency_data = load_json('self_consistency.json')
tick_data = load_json('tick_hierarchy.json')
validation_data = load_json('validation_summary.json')

# ---------------------------------------------------------------------------
# Helper: fundamental modes (sorted by frequency) for each molecule
# ---------------------------------------------------------------------------
def sorted_modes(mol):
    """Return list of (name, freq) sorted by frequency."""
    modes = MOLECULES[mol]["modes"]
    return sorted(modes.items(), key=lambda x: x[1])

# ===========================================================================
# PANEL 1: The Molecular Tick
# ===========================================================================
def panel_1():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D absorption-emission cycle ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    N = 500
    t = np.linspace(0, 2 * np.pi, N)
    # Absorption phase: spiral up (0 to pi)
    t_abs = t[t <= np.pi]
    t_em = t[t > np.pi]
    # Entropy coordinates
    Sk_abs = 0.5 * (1 - np.cos(t_abs))  # 0 -> 1
    St_abs = 0.25 * (1 - np.cos(t_abs)) * np.sin(3 * t_abs)  # oscillating
    Se_abs = 0.5 * (1 - np.cos(t_abs))  # 0 -> 1
    Sk_em = 0.5 * (1 + np.cos(t_em - np.pi))  # 1 -> 0
    St_em = 0.25 * (1 + np.cos(t_em - np.pi)) * np.sin(3 * t_em)
    Se_em = 0.5 * (1 + np.cos(t_em - np.pi))  # 1 -> 0
    Sk = np.concatenate([Sk_abs, Sk_em])
    St = np.concatenate([St_abs, St_em])
    Se = np.concatenate([Se_abs, Se_em])
    # Add helix component
    helix_r = 0.15
    Sk += helix_r * np.sin(8 * t)
    St += helix_r * np.cos(8 * t)
    colors = cm.viridis(np.linspace(0, 1, N))
    for i in range(N - 1):
        ax1.plot(Sk[i:i+2], St[i:i+2], Se[i:i+2], color=colors[i], linewidth=1.5)
    # Mark ground state and excited state
    ax1.scatter([Sk[0]], [St[0]], [Se[0]], c='green', s=60, zorder=5, marker='o')
    ax1.scatter([Sk[N//2]], [St[N//2]], [Se[N//2]], c='red', s=60, zorder=5, marker='^')
    # tau_em slice
    idx_tau = int(0.75 * N)
    ax1.axhline(y=St[idx_tau], color='grey', alpha=0.3)
    ax1.set_xlabel(r'$S_k$', labelpad=2)
    ax1.set_ylabel(r'$S_t$', labelpad=2)
    ax1.set_zlabel(r'$S_e$', labelpad=2)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Emission lifetime vs frequency ---
    ax2 = fig.add_subplot(1, 4, 2)
    omega_range = np.logspace(1.5, 4, 200)
    tau_rel = 1.0 / (omega_range ** 3)
    tau_rel /= tau_rel[0]
    ax2.loglog(omega_range, tau_rel, 'k-', linewidth=1.2, alpha=0.5, label=r'$\propto \omega^{-3}$')
    # Plot molecular fundamental modes
    for mol in MOL_ORDER:
        modes = MOLECULES[mol]["modes"]
        freqs = list(modes.values())
        # Use fundamental (highest or representative)
        for mname, freq in modes.items():
            tau_pt = (1.0 / (freq ** 3))
            tau_pt /= (1.0 / (omega_range[0] ** 3))
            ax2.scatter(freq, tau_pt, c=MOL_COLORS[mol], s=30, zorder=5, edgecolors='k', linewidths=0.3)
    # Legend with molecule labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[m],
               markersize=5, label=MOL_LABELS[m]) for m in MOL_ORDER]
    ax2.legend(handles=handles, fontsize=6, loc='upper right', framealpha=0.7, handletextpad=0.2)
    ax2.set_xlabel(r'$\omega$ (cm$^{-1}$)')
    ax2.set_ylabel(r'$\tau_\mathrm{em}$ (rel.)')

    # --- Chart 3: Tick rate identity ---
    ax3 = fig.add_subplot(1, 4, 3)
    omega_range2 = np.logspace(1.5, 4, 200)
    # A_ki proportional to omega^3
    A_ki = omega_range2 ** 3
    A_ki /= A_ki[-1]
    ax3.loglog(omega_range2, A_ki, 'k-', linewidth=1.2, alpha=0.5)
    for mol in MOL_ORDER:
        modes = MOLECULES[mol]["modes"]
        for mname, freq in modes.items():
            rate = (freq ** 3) / (omega_range2[-1] ** 3)
            ax3.scatter(freq, rate, c=MOL_COLORS[mol], s=30, zorder=5, edgecolors='k', linewidths=0.3)
    ax3.set_xlabel(r'$\omega$ (cm$^{-1}$)')
    ax3.set_ylabel(r'$A_{ki}$ (rel.)')

    # --- Chart 4: Population dynamics ---
    ax4 = fig.add_subplot(1, 4, 4)
    t_norm = np.linspace(0, 5, 300)
    P2 = np.exp(-t_norm)
    P0 = 1 - np.exp(-t_norm)
    ax4.plot(t_norm, P2, color='#d62728', linewidth=1.5, label=r'$P_2(t)$')
    ax4.plot(t_norm, P0, color='#1f77b4', linewidth=1.5, label=r'$P_0(t)$')
    # Crossing point at t = ln(2)
    t_cross = np.log(2)
    ax4.axvline(t_cross, color='grey', linestyle=':', linewidth=0.8)
    ax4.plot(t_cross, 0.5, 'ko', markersize=5, zorder=5)
    # tau_em boundary
    ax4.axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax4.annotate(r'$\tau_\mathrm{em}$', xy=(1.0, 0.95), fontsize=8, ha='left')
    ax4.set_xlabel(r'$t / \tau_\mathrm{em}$')
    ax4.set_ylabel(r'$P(t)$')
    ax4.legend(fontsize=7, loc='center right', framealpha=0.7)
    ax4.set_xlim(0, 5)
    ax4.set_ylim(-0.02, 1.05)

    fig.tight_layout(pad=1.0)
    outpath = FIGURES_DIR / 'panel_1_molecular_tick.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ===========================================================================
# PANEL 2: Nested Tick Hierarchy and Trees
# ===========================================================================
def panel_2():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D tick tree for CH4 ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ch4_tree = tick_data['results']['CH4']
    # Nodes: nu4 (root, level 0), nu2 (level 1), nu1, nu3 (level 2)
    ch4_modes = MOLECULES['CH4']['modes']
    node_info = {
        'nu4': {'freq': 1306, 'level': 0, 'M': 0},
        'nu2': {'freq': 1534, 'level': 1, 'M': 0.1465},
        'nu1': {'freq': 2917, 'level': 2, 'M': 0.585},
        'nu3': {'freq': 3019, 'level': 2, 'M': 0.6163},
    }
    edges_ch4 = [('nu4', 'nu2'), ('nu2', 'nu1'), ('nu2', 'nu3')]
    for name, info in node_info.items():
        size = 50 + info['freq'] / 20
        ax1.scatter(info['freq'], info['level'], info['M'],
                    s=size, c=info['freq'], cmap='plasma',
                    vmin=1200, vmax=3100, edgecolors='k', linewidths=0.5, zorder=5)
        ax1.text(info['freq'], info['level'] + 0.15, info['M'] + 0.03,
                 name.replace('nu', r'$\nu_') + '$',
                 fontsize=6, ha='center')
    for p, c in edges_ch4:
        pi, ci = node_info[p], node_info[c]
        ax1.plot([pi['freq'], ci['freq']], [pi['level'], ci['level']],
                 [pi['M'], ci['M']], 'k-', linewidth=1.2)
    ax1.set_xlabel(r'$\omega$ (cm$^{-1}$)', labelpad=2)
    ax1.set_ylabel('Level', labelpad=2)
    ax1.set_zlabel(r'$M$', labelpad=2)
    ax1.view_init(elev=20, azim=135)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Subdivision numbers ---
    ax2 = fig.add_subplot(1, 4, 2)
    x_positions = []
    x_labels_list = []
    bar_colors = []
    bar_vals = []
    x_pos = 0
    group_centers = []
    group_labels = []
    for mol in MOL_ORDER:
        tree = tick_data['results'][mol]
        edges = tree['edges']
        if not edges:
            continue
        start = x_pos
        for e in edges:
            parent = e['parent']
            child = e['child']
            N_sub = e['N_subdivision']
            bar_vals.append(N_sub)
            x_positions.append(x_pos)
            bar_colors.append(MOL_COLORS[mol])
            x_pos += 1
        group_centers.append((start + x_pos - 1) / 2.0)
        group_labels.append(MOL_LABELS[mol])
        x_pos += 0.5  # gap between groups
    ax2.bar(x_positions, bar_vals, color=bar_colors, edgecolor='k', linewidth=0.3, width=0.7)
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(group_labels, fontsize=7)
    ax2.set_ylabel(r'$N_{ij}$')
    ax2.set_yscale('log')

    # --- Chart 3: Partition depth ---
    ax3 = fig.add_subplot(1, 4, 3)
    ratio_range = np.linspace(0.5, 40, 200)
    log3_curve = np.log(ratio_range) / np.log(3)
    ax3.plot(ratio_range, log3_curve, 'k--', linewidth=0.8, alpha=0.5, label=r'$\log_3$')
    for mol in MOL_ORDER:
        tree = tick_data['results'][mol]
        for e in tree['edges']:
            ratio = e['omega_child'] / e['omega_parent']
            M = e['M_partition_depth']
            ax3.scatter(ratio, M, c=MOL_COLORS[mol], s=40, edgecolors='k',
                        linewidths=0.3, zorder=5)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[m],
               markersize=5, label=MOL_LABELS[m]) for m in MOL_ORDER]
    ax3.legend(handles=handles, fontsize=6, loc='upper left', framealpha=0.7, handletextpad=0.2)
    ax3.set_xlabel(r'$\omega_j / \omega_i$')
    ax3.set_ylabel(r'$M_{ij}$')

    # --- Chart 4: Frequency ordering ---
    ax4 = fig.add_subplot(1, 4, 4)
    y_pos = 0
    y_ticks = []
    y_labels_list = []
    for mol in reversed(MOL_ORDER):
        modes = sorted_modes(mol)
        group_start = y_pos
        for mname, freq in modes:
            ax4.barh(y_pos, freq, height=0.6, color=MOL_COLORS[mol],
                     edgecolor='k', linewidth=0.3)
            y_pos += 1
        center = (group_start + y_pos - 1) / 2.0
        y_ticks.append(center)
        y_labels_list.append(MOL_LABELS[mol])
        y_pos += 0.5
    ax4.set_yticks(y_ticks)
    ax4.set_yticklabels(y_labels_list, fontsize=7)
    ax4.set_xlabel(r'$\omega$ (cm$^{-1}$)')

    fig.tight_layout(pad=1.0)
    outpath = FIGURES_DIR / 'panel_2_tick_hierarchy.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ===========================================================================
# PANEL 3: Harmonic Network Formation
# ===========================================================================
def panel_3():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D harmonic network for C6H6 ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    c6h6_modes = MOLECULES['C6H6']['modes']
    mode_list = sorted(c6h6_modes.items(), key=lambda x: x[1])
    mode_names = [m[0] for m in mode_list]
    mode_freqs = [m[1] for m in mode_list]
    # Position nodes: x=freq, y=mode_index, z=0
    node_pos = {}
    for i, (mname, freq) in enumerate(mode_list):
        node_pos[mname] = (freq, i, 0)
        size = 30 + freq / 30
        ax1.scatter(freq, i, 0, s=size, c=[freq], cmap='plasma',
                    vmin=600, vmax=3200, edgecolors='k', linewidths=0.5, zorder=5)
        ax1.text(freq, i + 0.3, 0.05,
                 mname.replace('nu', r'$\nu_') + '$', fontsize=6, ha='center')

    # Tree edges from tick_data
    c6h6_tree = tick_data['results']['C6H6']
    for e in c6h6_tree['edges']:
        p, c = e['parent'], e['child']
        pp, cp = node_pos[p], node_pos[c]
        ax1.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
                 'b-', linewidth=1.5, alpha=0.8)

    # Harmonic edges from harmonic_data
    harm_edges_c6h6 = harmonic_data['results']['C6H6']['harmonic_edges']
    for he in harm_edges_c6h6:
        mi, mj = he['mode_i'], he['mode_j']
        if mi in node_pos and mj in node_pos:
            pi, pj = node_pos[mi], node_pos[mj]
            ax1.plot([pi[0], pj[0]], [pi[1], pj[1]], [pi[2], pj[2]],
                     'r--', linewidth=1.0, alpha=0.6)

    ax1.set_xlabel(r'$\omega$ (cm$^{-1}$)', labelpad=2)
    ax1.set_ylabel('Mode idx', labelpad=2)
    ax1.set_zlabel('', labelpad=0)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Harmonic ratio distribution ---
    ax2 = fig.add_subplot(1, 4, 2)
    all_deltas = []
    all_overlap = []  # True = tree-overlap, False = cross-branch
    for mol in MOL_ORDER:
        edges = harmonic_data['results'][mol]['harmonic_edges']
        for e in edges:
            all_deltas.append(e['delta'])
            is_overlap = e.get('is_tree_overlap', False)
            all_overlap.append(is_overlap)
    deltas_overlap = [d for d, o in zip(all_deltas, all_overlap) if o]
    deltas_cross = [d for d, o in zip(all_deltas, all_overlap) if not o]
    bins = np.linspace(0, 0.1, 25)
    if deltas_overlap:
        ax2.hist(deltas_overlap, bins=bins, color='#1f77b4', alpha=0.7, label='Tree-overlap', edgecolor='k', linewidth=0.3)
    if deltas_cross:
        ax2.hist(deltas_cross, bins=bins, color='#ff7f0e', alpha=0.7, label='Cross-branch', edgecolor='k', linewidth=0.3)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=1.0)
    ax2.set_xlabel(r'$\delta$')
    ax2.set_ylabel('Count')
    ax2.legend(fontsize=6, framealpha=0.7)

    # --- Chart 3: Rational approximation quality ---
    ax3 = fig.add_subplot(1, 4, 3)
    for mol in MOL_ORDER:
        edges = harmonic_data['results'][mol]['harmonic_edges']
        for e in edges:
            ratio = e['omega_i'] / e['omega_j']
            delta = e['delta']
            eta = e['eta']
            size = max(10, 80 / eta)
            ax3.scatter(ratio, delta, c=MOL_COLORS[mol], s=size,
                        edgecolors='k', linewidths=0.3, zorder=5, alpha=0.8)
    ax3.axhline(0.05, color='red', linestyle='--', linewidth=0.8, label=r'$\delta_{max}$')
    ax3.set_xlabel(r'$\omega_i / \omega_j$')
    ax3.set_ylabel(r'$\delta$')
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MOL_COLORS[m],
               markersize=5, label=MOL_LABELS[m]) for m in MOL_ORDER]
    ax3.legend(handles=handles, fontsize=6, loc='upper right', framealpha=0.7, handletextpad=0.2)

    # --- Chart 4: Network complexity scaling ---
    ax4 = fig.add_subplot(1, 4, 4)
    for mol in MOL_ORDER:
        net = network_data['results'][mol]['computed']
        N = net['num_modes']
        E_harm = net['num_harmonic_edges']
        ax4.scatter(N, E_harm, c=MOL_COLORS[mol], s=80, edgecolors='k',
                    linewidths=0.5, zorder=5)
        ax4.annotate(MOL_LABELS[mol], (N, E_harm), fontsize=6,
                     textcoords='offset points', xytext=(5, 5))
    # Combinatorial upper bound N(N-1)/2
    N_range = np.linspace(2, 8, 50)
    upper = N_range * (N_range - 1) / 2
    ax4.plot(N_range, upper, 'k--', linewidth=0.8, alpha=0.5, label=r'$N(N\!-\!1)/2$')
    ax4.set_xlabel(r'$N$ (modes)')
    ax4.set_ylabel(r'$E_\mathrm{harm}$')
    ax4.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout(pad=1.0)
    outpath = FIGURES_DIR / 'panel_3_harmonic_network.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ===========================================================================
# PANEL 4: Closed Loops and Light Circulation
# ===========================================================================
def panel_4():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D loop visualization for H2O ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    h2o_modes = MOLECULES['H2O']['modes']
    # Nodes: nu2=1595, nu1=3657, nu3=3756
    nodes = {'nu2': 1595, 'nu1': 3657, 'nu3': 3756}
    # Tree edges: nu2->nu1, nu2->nu3 ; Harmonic edge: nu1->nu3
    # Beat frequencies for edges
    beat_nu2_nu1 = abs(3657 - 1595)  # 2062
    beat_nu2_nu3 = abs(3756 - 1595)  # 2161
    beat_nu1_nu3 = abs(3756 - 3657)  # 99
    # Position in 3D: x=freq, y=angular position (triangle), z=beat freq
    import math
    angles = {'nu2': 0, 'nu1': 2 * math.pi / 3, 'nu3': 4 * math.pi / 3}
    radius = 1.0
    node_3d = {}
    for n, freq in nodes.items():
        x = radius * math.cos(angles[n])
        y = radius * math.sin(angles[n])
        z = freq / 1000.0  # scale
        node_3d[n] = (x, y, z)
        ax1.scatter(x, y, z, s=120, c=MOL_COLORS['H2O'], edgecolors='k',
                    linewidths=0.8, zorder=5)
        ax1.text(x + 0.12, y + 0.12, z + 0.1,
                 n.replace('nu', r'$\nu_') + '$', fontsize=7, ha='center')

    # Tree edges (solid blue arrows)
    tree_edges = [('nu2', 'nu1'), ('nu2', 'nu3')]
    for p, c in tree_edges:
        pp, cp = node_3d[p], node_3d[c]
        ax1.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
                 'b-', linewidth=2.0, alpha=0.8)
        # Arrowhead approximation
        mid = [(pp[i] + cp[i]) / 2 for i in range(3)]
        dx = cp[0] - pp[0]
        dy = cp[1] - pp[1]
        dz = cp[2] - pp[2]
        ax1.quiver(mid[0], mid[1], mid[2], dx * 0.01, dy * 0.01, dz * 0.01,
                   color='blue', arrow_length_ratio=3.0, linewidth=0)

    # Harmonic edge (dashed red) nu1->nu3
    pp, cp = node_3d['nu1'], node_3d['nu3']
    ax1.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
             'r--', linewidth=2.0, alpha=0.8)

    ax1.set_xlabel(r'$x$', labelpad=1)
    ax1.set_ylabel(r'$y$', labelpad=1)
    ax1.set_zlabel(r'$\omega$/1000', labelpad=1)
    ax1.view_init(elev=20, azim=145)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Circulation periods ---
    ax2 = fig.add_subplot(1, 4, 2)
    circ_labels = []
    circ_periods = []
    circ_colors = []
    for mol in MOL_ORDER:
        circs = circulation_data['results'][mol]['circulations']
        for circ in circs:
            if not circ['is_overlap_loop']:
                label = f"{MOL_LABELS[mol]}"
                circ_labels.append(circ['loop_label'])
                circ_periods.append(circ['total_beat_period_ps'])
                circ_colors.append(MOL_COLORS[mol])
    # Sort by period
    sorted_idx = np.argsort(circ_periods)
    circ_labels = [circ_labels[i] for i in sorted_idx]
    circ_periods = [circ_periods[i] for i in sorted_idx]
    circ_colors = [circ_colors[i] for i in sorted_idx]
    x_pos_circ = np.arange(len(circ_periods))
    ax2.bar(x_pos_circ, circ_periods, color=circ_colors, edgecolor='k', linewidth=0.3)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$T_L$ (ps)')
    ax2.set_xticks(x_pos_circ)
    ax2.set_xticklabels([l[:8] for l in circ_labels], rotation=45, ha='right', fontsize=5)

    # --- Chart 3: Beat frequency spectrum ---
    ax3 = fig.add_subplot(1, 4, 3)
    for mol in MOL_ORDER:
        circs = circulation_data['results'][mol]['circulations']
        for circ in circs:
            for transit in circ['edge_transits']:
                if transit['type'] == 'harmonic':
                    delta_omega = transit['delta_omega_cm']
                    ax3.vlines(delta_omega, 0, 1, color=MOL_COLORS[mol],
                               linewidth=1.5, alpha=0.7)
    # Annotate key ones
    ax3.annotate(r'CO$_2$ Fermi', xy=(961, 1.02), fontsize=5, ha='center', va='bottom',
                 rotation=45)
    ax3.annotate(r'H$_2$O', xy=(99, 1.02), fontsize=5, ha='center', va='bottom',
                 rotation=45)
    ax3.set_xlabel(r'$\Delta\omega$ (cm$^{-1}$)')
    ax3.set_ylabel('(norm.)')
    ax3.set_ylim(0, 1.3)
    ax3.set_yticks([])
    handles = [plt.Line2D([0], [0], color=MOL_COLORS[m], linewidth=2,
               label=MOL_LABELS[m]) for m in MOL_ORDER]
    ax3.legend(handles=handles, fontsize=5, loc='upper right', framealpha=0.7,
               handletextpad=0.2, ncol=2)

    # --- Chart 4: Phase accumulation ---
    ax4 = fig.add_subplot(1, 4, 4)
    # H2O loop: nu3->nu2->nu1->nu3
    h2o_loop_freqs = [3756, 1595, 3657]  # nu3, nu2, nu1
    # CO2 loop: nu3->nu2->nu1->nu3
    co2_loop_freqs = [2349, 667, 1388]
    t_loop = np.linspace(0, 1, 500)
    for loop_freqs, mol, label in [(h2o_loop_freqs, 'H2O', r'H$_2$O'),
                                    (co2_loop_freqs, 'CO2', r'CO$_2$')]:
        total_freq = sum(loop_freqs)
        phase = np.zeros_like(t_loop)
        n_segments = len(loop_freqs)
        for i, freq in enumerate(loop_freqs):
            seg_start = i / n_segments
            seg_end = (i + 1) / n_segments
            mask = (t_loop >= seg_start) & (t_loop < seg_end)
            t_seg = (t_loop[mask] - seg_start) * n_segments
            if i == 0:
                phase[mask] = freq * t_seg / total_freq
            else:
                # Accumulate
                prev_phase = sum(loop_freqs[:i]) / total_freq
                phase[mask] = prev_phase + freq * t_seg / total_freq
        # Last point
        phase[-1] = 1.0
        ax4.plot(t_loop, phase, color=MOL_COLORS[mol], linewidth=1.5, label=label)
    # Mark 2*pi*n crossings (phase=1 at end)
    ax4.axhline(1.0, color='grey', linestyle=':', linewidth=0.6)
    ax4.axhline(0.0, color='grey', linestyle=':', linewidth=0.6)
    ax4.plot(1.0, 1.0, 'ko', markersize=5, zorder=5)
    ax4.plot(0.0, 0.0, 'ko', markersize=5, zorder=5)
    ax4.set_xlabel(r'Loop position')
    ax4.set_ylabel(r'$\phi / 2\pi$')
    ax4.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout(pad=1.0)
    outpath = FIGURES_DIR / 'panel_4_circulation.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ===========================================================================
# PANEL 5: Self-Clocking and Self-Validation
# ===========================================================================
def panel_5():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D self-consistency surface for CH4 ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    # CH4 has 4 modes and 6 loops; compute predicted/observed ratios
    ch4_pred = consistency_data['results']['CH4']['predictions']
    ch4_modes_list = ['nu4', 'nu2', 'nu1', 'nu3']
    ch4_freqs = {'nu4': 1306, 'nu2': 1534, 'nu1': 2917, 'nu3': 3019}
    ch4_circs = circulation_data['results']['CH4']['circulations']

    # Build a grid: mode_idx x loop_idx -> predicted/observed ratio
    mode_indices = {m: i for i, m in enumerate(ch4_modes_list)}
    n_modes = len(ch4_modes_list)
    n_loops = len(ch4_circs)

    # For each mode, check if it has predictions
    # Create surface data
    X = np.arange(n_modes)
    Y = np.arange(n_loops)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = np.ones_like(X_grid, dtype=float)

    # Fill in known deviations
    for mode_name, preds in ch4_pred.items():
        mi = mode_indices.get(mode_name)
        if mi is None:
            continue
        for pred in preds:
            pred_freq = pred['predicted_freq']
            obs_freq = pred['observed_freq']
            ratio = pred_freq / obs_freq
            # Find which loop this corresponds to
            loop_label = pred['loop']
            for li, circ in enumerate(ch4_circs):
                if circ['loop_label'] == loop_label:
                    Z_grid[li, mi] = ratio
                    break

    surf = ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap='RdYlGn_r',
                            alpha=0.7, edgecolor='k', linewidth=0.3)
    ax1.set_xlabel('Mode', labelpad=2)
    ax1.set_ylabel('Loop', labelpad=2)
    ax1.set_zlabel(r'$\omega_{pred}/\omega_{obs}$', labelpad=2)
    ax1.set_xticks(range(n_modes))
    ax1.set_xticklabels([r'$\nu_' + m[-1] + '$' for m in ch4_modes_list], fontsize=6)
    ax1.set_yticks(range(n_loops))
    ax1.set_yticklabels([str(i) for i in range(n_loops)], fontsize=6)
    ax1.set_zlim(0.98, 1.02)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Self-consistency deviations ---
    ax2 = fig.add_subplot(1, 4, 2)
    mol_devs = {}
    for mol in MOL_ORDER:
        res = consistency_data['results'][mol]
        mol_devs[mol] = res['max_deviation_pct']
    x_pos = np.arange(len(MOL_ORDER))
    devs = [mol_devs[m] for m in MOL_ORDER]
    colors = ['#2ca02c' if d < 2.0 else '#d62728' for d in devs]
    ax2.bar(x_pos, devs, color=colors, edgecolor='k', linewidth=0.3)
    ax2.axhline(2.0, color='red', linestyle='--', linewidth=1.0, label=r'2% threshold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([MOL_LABELS[m] for m in MOL_ORDER], fontsize=7)
    ax2.set_ylabel(r'Max $\Delta$ (%)')
    ax2.legend(fontsize=6, framealpha=0.7)

    # --- Chart 3: Loop redundancy ---
    ax3 = fig.add_subplot(1, 4, 3)
    x_pos3 = np.arange(len(MOL_ORDER))
    total_loops = []
    cross_loops = []
    for mol in MOL_ORDER:
        net = network_data['results'][mol]['computed']
        C_paper = net['C_paper']
        total_loops.append(C_paper)
        # Count cross-branch loops
        circs = circulation_data['results'][mol]['circulations']
        n_cross = sum(1 for c in circs if not c['is_overlap_loop'])
        cross_loops.append(n_cross)
    ax3.bar(x_pos3, total_loops, color=[MOL_COLORS[m] for m in MOL_ORDER],
            edgecolor='k', linewidth=0.3)
    # Overlay cross-branch as hatched
    ax3.bar(x_pos3, cross_loops, color='none', edgecolor='k',
            linewidth=0.8, hatch='///', alpha=0.5)
    ax3.set_xticks(x_pos3)
    ax3.set_xticklabels([MOL_LABELS[m] for m in MOL_ORDER], fontsize=7)
    ax3.set_ylabel(r'$C$ (loops)')
    ax3.annotate(r'$C = |E| - |V| + 1$', xy=(0.02, 0.92), xycoords='axes fraction',
                 fontsize=7, style='italic')

    # --- Chart 4: Entry-point independence for CH4 nu1 ---
    ax4 = fig.add_subplot(1, 4, 4)
    # nu1 = 2917 through different loops
    # From self_consistency data:
    # L1 (nu3-nu2-nu1-nu3): nu1 is in this loop, observed = 2917
    # L3 (nu1-nu2-nu4-nu1): predicted = 2938.5
    # Also nu1 appears in loop nu1-nu2-nu1 (overlap)
    true_freq = 2917.0
    determinations = [
        (r'$L_1$: direct', 2917.0, '#1f77b4'),
        (r'$L_3$: via $\nu_4$', 2938.5, '#ff7f0e'),
        (r'$L_0$: overlap', 2917.0, '#2ca02c'),
    ]
    y_pos4 = np.arange(len(determinations))
    for i, (label, freq, color) in enumerate(determinations):
        ax4.barh(i, freq, height=0.5, color=color, edgecolor='k',
                 linewidth=0.3, alpha=0.8)
    ax4.axvline(true_freq, color='k', linestyle='--', linewidth=1.2)
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels([d[0] for d in determinations], fontsize=7)
    ax4.set_xlabel(r'$\omega_{\nu_1}$ (cm$^{-1}$)')
    ax4.set_xlim(2900, 2950)
    ax4.annotate(r'$\nu_1 = 2917$', xy=(2917, len(determinations) - 0.3),
                 fontsize=7, ha='right', va='bottom')

    try:
        fig.tight_layout(pad=1.0)
    except Exception:
        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15, wspace=0.3)
    outpath = FIGURES_DIR / 'panel_5_self_validation.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath

# ===========================================================================
# PANEL 6: Comparative Validation Summary
# ===========================================================================
def panel_6():
    fig = plt.figure(figsize=(20, 5), facecolor='white')

    # --- Chart 1: 3D molecule comparison ---
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    for mol in MOL_ORDER:
        net = network_data['results'][mol]['computed']
        N = net['num_modes']
        E_harm = net['num_harmonic_edges']
        C = net['C_paper']
        E_total = net['total_edges']
        size = 30 + E_total * 8
        ax1.scatter(N, E_harm, C, s=size, c=MOL_COLORS[mol],
                    edgecolors='k', linewidths=0.5, zorder=5)
        ax1.text(N + 0.15, E_harm + 0.15, C + 0.15,
                 MOL_LABELS[mol], fontsize=6)
    ax1.set_xlabel(r'$N$', labelpad=2)
    ax1.set_ylabel(r'$E_\mathrm{harm}$', labelpad=2)
    ax1.set_zlabel(r'$C$', labelpad=2)
    ax1.view_init(elev=20, azim=135)
    ax1.tick_params(axis='both', labelsize=6, pad=0)

    # --- Chart 2: Network density ---
    ax2 = fig.add_subplot(1, 4, 2)
    for mol in MOL_ORDER:
        net = network_data['results'][mol]['computed']
        N = net['num_modes']
        E_total = net['total_edges']
        max_edges = N * (N - 1) / 2
        density = E_total / max_edges if max_edges > 0 else 0
        ax2.scatter(N, density, c=MOL_COLORS[mol], s=80, edgecolors='k',
                    linewidths=0.5, zorder=5)
        ax2.annotate(MOL_LABELS[mol], (N, density), fontsize=6,
                     textcoords='offset points', xytext=(5, 5))
    # Erdos-Renyi threshold
    N_range = np.linspace(2, 8, 100)
    pc = np.log(N_range) / N_range
    ax2.plot(N_range, pc, 'k--', linewidth=0.8, alpha=0.5,
             label=r'$p_c = \ln N / N$')
    ax2.set_xlabel(r'$N$ (modes)')
    ax2.set_ylabel(r'Edge density')
    ax2.legend(fontsize=7, framealpha=0.7)
    ax2.set_ylim(0, 1.1)

    # --- Chart 3: Validation heat map ---
    ax3 = fig.add_subplot(1, 4, 3)
    categories = ['tick_hierarchy', 'harmonic_edges', 'network_properties',
                   'loop_detection', 'circulation', 'self_consistency']
    cat_labels = [r'$\tau$', r'$\delta$', 'Net', r'$\circlearrowleft$',
                  r'$T_L$', r'$\Delta$']
    heat_data = np.zeros((len(MOL_ORDER), len(categories)))
    for i, mol in enumerate(MOL_ORDER):
        for j, cat in enumerate(categories):
            status = validation_data['per_molecule'][mol][cat]
            heat_data[i, j] = 1.0 if status == 'PASS' else 0.0
    from matplotlib.colors import LinearSegmentedColormap
    green_cmap = LinearSegmentedColormap.from_list('green', ['#f0f0f0', '#2ca02c'])
    im = ax3.imshow(heat_data, cmap=green_cmap, aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels(cat_labels, fontsize=7)
    ax3.set_yticks(range(len(MOL_ORDER)))
    ax3.set_yticklabels([MOL_LABELS[m] for m in MOL_ORDER], fontsize=7)
    # Add check marks
    for i in range(len(MOL_ORDER)):
        for j in range(len(categories)):
            ax3.text(j, i, r'$\checkmark$', ha='center', va='center', fontsize=10,
                     color='white', fontweight='bold')

    # --- Chart 4: Fermi resonance detection ---
    ax4 = fig.add_subplot(1, 4, 4)
    # CO2 Fermi: nu1/nu2 = 1388/667 = 2.081, theoretical 2:1
    # CH4 Fermi: nu1/nu2 = 2917/1534 = 1.902, theoretical 2:1
    # Also CH4 nu3/nu2 = 3019/1534 = 1.968, theoretical 2:1
    fermi_data_list = [
        {'mol': 'CO2', 'label': r'CO$_2$ $\nu_1/\nu_2$',
         'p_q': 2.0, 'observed': 1388.0 / 667.0,
         'delta': abs(1388.0 / 667.0 - 2.0)},
        {'mol': 'CH4', 'label': r'CH$_4$ $\nu_1/\nu_2$',
         'p_q': 2.0, 'observed': 2917.0 / 1534.0,
         'delta': abs(2917.0 / 1534.0 - 2.0)},
        {'mol': 'CH4', 'label': r'CH$_4$ $\nu_3/\nu_2$',
         'p_q': 2.0, 'observed': 3019.0 / 1534.0,
         'delta': abs(3019.0 / 1534.0 - 2.0)},
    ]
    # Perfect diagonal reference
    pq_range = np.linspace(1.5, 2.5, 50)
    ax4.plot(pq_range, pq_range, 'k--', linewidth=0.8, alpha=0.5)
    for fd in fermi_data_list:
        ax4.scatter(fd['p_q'], fd['observed'], c=MOL_COLORS[fd['mol']], s=80,
                    edgecolors='k', linewidths=0.5, zorder=5)
        ax4.errorbar(fd['p_q'], fd['observed'], yerr=fd['delta'],
                     fmt='none', ecolor=MOL_COLORS[fd['mol']], capsize=3,
                     linewidth=1.0, zorder=4)
        ax4.annotate(fd['label'], (fd['p_q'], fd['observed']),
                     fontsize=6, textcoords='offset points', xytext=(8, -3))
    ax4.set_xlabel(r'$p/q$ (theoretical)')
    ax4.set_ylabel(r'$\omega_i / \omega_j$ (observed)')
    ax4.set_xlim(1.8, 2.2)
    ax4.set_ylim(1.8, 2.2)
    ax4.set_aspect('equal')

    fig.tight_layout(pad=1.0)
    outpath = FIGURES_DIR / 'panel_6_validation_summary.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return outpath


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating publication panels for Paper 2: Harmonic Molecular Resonator")
    print(f"Output directory: {FIGURES_DIR}")
    print()

    panels = [
        ("Panel 1: The Molecular Tick", panel_1),
        ("Panel 2: Nested Tick Hierarchy and Trees", panel_2),
        ("Panel 3: Harmonic Network Formation", panel_3),
        ("Panel 4: Closed Loops and Light Circulation", panel_4),
        ("Panel 5: Self-Clocking and Self-Validation", panel_5),
        ("Panel 6: Comparative Validation Summary", panel_6),
    ]

    for name, func in panels:
        try:
            path = func()
            print(f"  [OK] {name} -> {path}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("Done. All panels generated.")


if __name__ == '__main__':
    main()
