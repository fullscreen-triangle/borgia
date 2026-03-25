#!/usr/bin/env python3
"""
Generate 4 publication-quality panels (16 charts total) for the
Categorical Compound Database.

Usage:
    python generate_panels.py

Requires: numpy, matplotlib
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from itertools import combinations
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "compound_encodings.json")) as f:
    encodings = json.load(f)
with open(os.path.join(RESULTS_DIR, "fuzzy_search_results.json")) as f:
    fuzzy = json.load(f)
with open(os.path.join(RESULTS_DIR, "similarity_matrix.json")) as f:
    sim_matrix = json.load(f)
with open(os.path.join(RESULTS_DIR, "clustering.json")) as f:
    clustering = json.load(f)
with open(os.path.join(RESULTS_DIR, "property_search_results.json")) as f:
    property_search = json.load(f)
with open(os.path.join(RESULTS_DIR, "validation_summary.json")) as f:
    validation = json.load(f)

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "diatomic": "#1f77b4",
    "triatomic": "#2ca02c",
    "tetra": "#ff7f0e",
    "poly": "#d62728",
}
TYPE_ORDER = ["diatomic", "triatomic", "tetra", "poly"]

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 0,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ---------------------------------------------------------------------------
# Prepare structured arrays
# ---------------------------------------------------------------------------
names = sorted(encodings.keys())
formulas = [encodings[n]["formula"] for n in names]
types = [encodings[n]["type"] for n in names]
S_k = np.array([encodings[n]["S_k"] for n in names])
S_t = np.array([encodings[n]["S_t"] for n in names])
S_e = np.array([encodings[n]["S_e"] for n in names])
masses = np.array([encodings[n]["mass"] for n in names], dtype=float)
trit_strings = [encodings[n]["trit_string"] for n in names]
modes_list = [encodings[n]["modes"] for n in names]
n_modes = np.array([len(m) for m in modes_list])
colors = [TYPE_COLORS[t] for t in types]

# Point sizes proportional to mass, rescaled for visibility
size_scale = 15 + 120 * (masses - masses.min()) / (masses.max() - masses.min())


def _wireframe_cube(ax, lo=0.0, hi=1.0, color="grey", alpha=0.25, lw=0.6):
    """Draw the edges of a cube [lo,hi]^3."""
    r = [lo, hi]
    edges = []
    for s, e in combinations(np.array(list(np.ndindex(2, 2, 2))), 2):
        s, e = np.array(s), np.array(e)
        if np.sum(np.abs(s - e)) == 1:
            p0 = lo + (hi - lo) * s
            p1 = lo + (hi - lo) * e
            edges.append([p0, p1])
    lc = Line3DCollection(edges, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(lc)


# ===================================================================
# PANEL 1 — S-Entropy Compound Space
# ===================================================================
def panel_1():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # Remove default 2-D axes for the 3-D subplot
    axes[0].remove()
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")

    # --- Chart 1: 3-D S-entropy scatter ----------------------------------
    for tp in TYPE_ORDER:
        idx = [i for i, t in enumerate(types) if t == tp]
        ax3d.scatter(
            S_k[idx], S_t[idx], S_e[idx],
            s=size_scale[idx], c=TYPE_COLORS[tp], label=tp,
            alpha=0.85, edgecolors="k", linewidths=0.3,
        )
    _wireframe_cube(ax3d)
    ax3d.set_xlabel(r"$S_k$", labelpad=4)
    ax3d.set_ylabel(r"$S_t$", labelpad=4)
    ax3d.set_zlabel(r"$S_e$", labelpad=4)
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.view_init(elev=25, azim=135)
    ax3d.legend(fontsize=6, loc="upper left", framealpha=0.7)

    # --- Chart 2: S_k histogram (stacked) ---------------------------------
    ax = axes[1]
    bins = np.linspace(0, 1, 16)
    data_by_type = []
    color_list = []
    label_list = []
    for tp in TYPE_ORDER:
        vals = S_k[[i for i, t in enumerate(types) if t == tp]]
        data_by_type.append(vals)
        color_list.append(TYPE_COLORS[tp])
        label_list.append(tp)
    ax.hist(data_by_type, bins=bins, stacked=True, color=color_list,
            label=label_list, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$S_k$")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, framealpha=0.7)

    # --- Chart 3: S_t vs S_k scatter -------------------------------------
    ax = axes[2]
    for tp in TYPE_ORDER:
        idx = [i for i, t in enumerate(types) if t == tp]
        ax.scatter(
            S_k[idx], S_t[idx],
            s=size_scale[idx], c=TYPE_COLORS[tp], label=tp,
            alpha=0.8, edgecolors="k", linewidths=0.3,
        )
    ax.set_xlabel(r"$S_k$")
    ax.set_ylabel(r"$S_t$")
    ax.legend(fontsize=6, framealpha=0.7)

    # --- Chart 4: S_e bar chart (sorted) ----------------------------------
    ax = axes[3]
    order = np.argsort(S_e)
    bar_colors = [colors[i] for i in order]
    bar_labels = [formulas[i] for i in order]
    ax.bar(range(len(order)), S_e[order], color=bar_colors, edgecolor="white",
           linewidth=0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(bar_labels, rotation=90, fontsize=5)
    ax.set_ylabel(r"$S_e$")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "panel_1_sentropy_space.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
# PANEL 2 — Ternary Encoding and Trie Structure
# ===================================================================
def panel_2():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Chart 1: 3-D trie visualisation ----------------------------------
    axes[0].remove()
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")

    depth3_clusters = clustering["depth_3"]["clusters"]

    # Build tree coordinates: root -> 3 children -> 9 grandchildren -> 27 depth-3
    # x = horizontal spread, y = depth (top=0), z = branch index
    node_coords = {}  # (level, branch_tuple) -> (x, y, z)
    # Root
    node_coords[(0, ())] = (0, 0, 0)
    # Level 1 (3 nodes)
    for b1 in range(3):
        x = (b1 - 1) * 4
        node_coords[(1, (b1,))] = (x, -1, b1 - 1)
    # Level 2 (9 nodes)
    for b1 in range(3):
        for b2 in range(3):
            px, py, pz = node_coords[(1, (b1,))]
            x = px + (b2 - 1) * 1.2
            z = pz + (b2 - 1) * 0.8
            node_coords[(2, (b1, b2))] = (x, -2, z)
    # Level 3 (27 nodes)
    for b1 in range(3):
        for b2 in range(3):
            for b3 in range(3):
                px, py, pz = node_coords[(2, (b1, b2))]
                x = px + (b3 - 1) * 0.35
                z = pz + (b3 - 1) * 0.25
                node_coords[(3, (b1, b2, b3))] = (x, -3, z)

    # Draw edges
    for key, coord in node_coords.items():
        level, branch = key
        if level == 0:
            continue
        parent_branch = branch[:-1]
        parent_coord = node_coords[(level - 1, parent_branch)]
        ax3d.plot(
            [parent_coord[0], coord[0]],
            [parent_coord[1], coord[1]],
            [parent_coord[2], coord[2]],
            "k-", lw=0.5, alpha=0.4,
        )

    # Draw nodes: depth-3 leaves as coloured spheres sized by compound count
    for b1 in range(3):
        for b2 in range(3):
            for b3 in range(3):
                prefix = f"{b1}{b2}{b3}"
                count = len(depth3_clusters.get(prefix, []))
                coord = node_coords[(3, (b1, b2, b3))]
                if count > 0:
                    ax3d.scatter(
                        *coord, s=40 + count * 50,
                        c=[plt.cm.YlOrRd(count / 11)],
                        edgecolors="k", linewidths=0.4, alpha=0.9,
                        zorder=5,
                    )
                else:
                    ax3d.scatter(
                        *coord, s=12, c="lightgrey",
                        alpha=0.35, zorder=3,
                    )

    # Interior and root nodes (small grey)
    for key, coord in node_coords.items():
        level, _ = key
        if level < 3:
            ax3d.scatter(*coord, s=18, c="grey", alpha=0.5, zorder=4)

    ax3d.set_xlabel("$x$", labelpad=2)
    ax3d.set_ylabel("depth", labelpad=2)
    ax3d.set_zlabel("$z$", labelpad=2)
    ax3d.view_init(elev=20, azim=135)

    # --- Chart 2: Ternary heatmap -----------------------------------------
    ax = axes[1]
    # Sort compounds by trit string
    sorted_idx = sorted(range(len(names)), key=lambda i: trit_strings[i])
    mat = np.array([[int(c) for c in trit_strings[i]] for i in sorted_idx])
    cmap = plt.cm.colors.ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([formulas[i] for i in sorted_idx], fontsize=4.5)
    ax.set_xlabel("Trit position")
    ax.set_xticks(range(18))
    ax.set_xticklabels(range(1, 19), fontsize=5)
    # Colourbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.6)
    cbar.ax.set_yticklabels(["0", "1", "2"], fontsize=6)

    # --- Chart 3: Resolution vs unique compounds -------------------------
    ax = axes[2]
    depths = [0, 3, 6, 9, 12, 15, 18]
    unique_cells = [1]  # depth 0 -> 1
    for d_key in ["depth_3", "depth_6", "depth_9", "depth_12"]:
        unique_cells.append(clustering[d_key]["total_cells"])
    # depth 15 and 18: all unique (39)
    unique_cells.append(39)
    unique_cells.append(39)

    ax.plot(depths, unique_cells, "o-", color="#333333", markersize=5, lw=1.5)
    ax.axhline(39, ls="--", color="grey", lw=0.7, alpha=0.6)
    ax.axvline(12, ls=":", color="#d62728", lw=1, alpha=0.7)
    ax.annotate("all 39 unique", xy=(12, 39), xytext=(13.5, 34),
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("Trit depth")
    ax.set_ylabel("Occupied cells")
    ax.set_xticks(depths)

    # --- Chart 4: Cell occupancy at depth 3 --------------------------------
    ax = axes[3]
    prefixes = sorted(depth3_clusters.keys())
    counts = [len(depth3_clusters[p]) for p in prefixes]
    # Dominant type per cell
    def dominant_type(members):
        type_counts = {}
        for m in members:
            t = encodings[m]["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        return max(type_counts, key=type_counts.get)

    bar_cols = [TYPE_COLORS[dominant_type(depth3_clusters[p])] for p in prefixes]
    ax.bar(range(len(prefixes)), counts, color=bar_cols, edgecolor="white",
           linewidth=0.3)
    ax.set_xticks(range(len(prefixes)))
    ax.set_xticklabels(prefixes, fontsize=6, rotation=45)
    ax.set_xlabel("Cell prefix")
    ax.set_ylabel("Compounds")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "panel_2_ternary_encoding.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
# PANEL 3 — Fuzzy Search and Chemical Similarity
# ===================================================================
def panel_3():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Chart 1: 3-D fuzzy search demo (H2O) ----------------------------
    axes[0].remove()
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")

    h2o_cell = fuzzy["H2O"]["depth_3"]
    cell_members = h2o_cell["matches"]  # compounds in same depth-3 cell
    cell_set = set(cell_members)

    # All non-member compounds as small grey dots
    for i, n in enumerate(names):
        if n not in cell_set:
            ax3d.scatter(S_k[i], S_t[i], S_e[i], s=12, c="lightgrey",
                         alpha=0.4, zorder=2)

    # Member compounds as large coloured markers
    for n in cell_members:
        i = names.index(n)
        ax3d.scatter(S_k[i], S_t[i], S_e[i], s=90,
                     c=TYPE_COLORS[types[i]], edgecolors="k",
                     linewidths=0.5, alpha=0.95, zorder=5)
        ax3d.text(S_k[i] + 0.015, S_t[i] + 0.015, S_e[i] + 0.02,
                  formulas[i], fontsize=5, zorder=6)

    # Translucent cube for depth-3 cell
    # The depth-3 prefix for H2O is "202" -> trits for S_k=2, S_t=0, S_e=2
    # Each trit divides [0,1] into thirds: 0->[0,.333], 1->[.333,.667], 2->[.667,1]
    trit_bounds = {0: (0, 1/3), 1: (1/3, 2/3), 2: (2/3, 1)}
    prefix = h2o_cell["prefix"]
    sk_lo, sk_hi = trit_bounds[int(prefix[0])]
    st_lo, st_hi = trit_bounds[int(prefix[1])]
    se_lo, se_hi = trit_bounds[int(prefix[2])]

    # Draw translucent box faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts_faces = [
        [[sk_lo,st_lo,se_lo],[sk_hi,st_lo,se_lo],[sk_hi,st_hi,se_lo],[sk_lo,st_hi,se_lo]],
        [[sk_lo,st_lo,se_hi],[sk_hi,st_lo,se_hi],[sk_hi,st_hi,se_hi],[sk_lo,st_hi,se_hi]],
        [[sk_lo,st_lo,se_lo],[sk_hi,st_lo,se_lo],[sk_hi,st_lo,se_hi],[sk_lo,st_lo,se_hi]],
        [[sk_lo,st_hi,se_lo],[sk_hi,st_hi,se_lo],[sk_hi,st_hi,se_hi],[sk_lo,st_hi,se_hi]],
        [[sk_lo,st_lo,se_lo],[sk_lo,st_hi,se_lo],[sk_lo,st_hi,se_hi],[sk_lo,st_lo,se_hi]],
        [[sk_hi,st_lo,se_lo],[sk_hi,st_hi,se_lo],[sk_hi,st_hi,se_hi],[sk_hi,st_lo,se_hi]],
    ]
    poly = Poly3DCollection(verts_faces, alpha=0.08, facecolor="#2ca02c",
                            edgecolor="#2ca02c", linewidths=0.5)
    ax3d.add_collection3d(poly)
    _wireframe_cube(ax3d)

    ax3d.set_xlabel(r"$S_k$", labelpad=4)
    ax3d.set_ylabel(r"$S_t$", labelpad=4)
    ax3d.set_zlabel(r"$S_e$", labelpad=4)
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.view_init(elev=25, azim=135)

    # --- Chart 2: Search narrowing cascade (CH4) --------------------------
    ax = axes[1]
    ch4_data = fuzzy["CH4"]
    depth_labels = ["depth 3", "depth 6", "depth 9", "depth 12"]
    depth_keys = ["depth_3", "depth_6", "depth_9", "depth_12"]
    y_positions = [3, 2, 1, 0]

    for yi, dk in zip(y_positions, depth_keys):
        matches = ch4_data[dk]["matches"]
        for xi, m in enumerate(matches):
            c = TYPE_COLORS[encodings[m]["type"]]
            ax.barh(yi, 0.9, left=xi, height=0.6, color=c,
                    edgecolor="white", linewidth=0.5)
            ax.text(xi + 0.45, yi, encodings[m]["formula"],
                    ha="center", va="center", fontsize=5, color="white",
                    fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(depth_labels, fontsize=7)
    ax.set_xlabel("Matched compounds")
    ax.set_xlim(-0.2, 7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Chart 3: 39×39 similarity heatmap --------------------------------
    ax = axes[2]
    # Sort by trit string for diagonal clustering
    sorted_names = sorted(names, key=lambda n: encodings[n]["trit_string"])
    n = len(sorted_names)
    mat = np.zeros((n, n))
    for i, ni in enumerate(sorted_names):
        for j, nj in enumerate(sorted_names):
            mat[i, j] = sim_matrix[ni][nj]
    im = ax.imshow(mat, cmap="viridis", aspect="auto", interpolation="nearest")
    sorted_formulas = [encodings[n_]["formula"] for n_ in sorted_names]
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_formulas, rotation=90, fontsize=3.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_formulas, fontsize=3.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.65)
    cbar.ax.tick_params(labelsize=6)

    # --- Chart 4: Chemical group cohesion ---------------------------------
    ax = axes[3]
    groups_data = validation["chemical_group_validation"]
    group_names = list(groups_data.keys())
    intra_vals = [groups_data[g]["mean_intra_similarity"] for g in group_names]
    inter_vals = [groups_data[g]["mean_inter_similarity"] for g in group_names]
    x_pos = np.arange(len(group_names))
    w = 0.35
    ax.bar(x_pos - w/2, intra_vals, w, color="#1f77b4", label="Intra-group")
    ax.bar(x_pos + w/2, inter_vals, w, color="#aaaaaa", label="Inter-group")
    ax.axhline(1.0, ls="--", color="k", lw=0.7, alpha=0.5)
    ax.set_xticks(x_pos)
    # Shorten labels
    short = [g.replace("Homonuclear d", "Homonuc. d").replace("Small h", "Sm. h")
             .replace("Linear t", "Lin. t").replace("Bent t", "Bent t")
             .replace("Hydrogen h", "H-h") for g in group_names]
    ax.set_xticklabels(short, fontsize=5.5, rotation=30, ha="right")
    ax.set_ylabel("Mean prefix length")
    ax.legend(fontsize=6, framealpha=0.7)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "panel_3_fuzzy_search.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
# PANEL 4 — Structure Prediction and Complexity Scaling
# ===================================================================
def panel_4():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Chart 1: 3-D property search ------------------------------------
    axes[0].remove()
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")

    # Constraint box from property_search_results
    sk_range = property_search["high_S_k_low_S_t"]["S_k_range"]
    st_range = property_search["high_S_k_low_S_t"]["S_t_range"]
    inside_names = set(property_search["high_S_k_low_S_t"]["results"])

    for i, n in enumerate(names):
        if n in inside_names:
            ax3d.scatter(S_k[i], S_t[i], S_e[i], s=80,
                         c=TYPE_COLORS[types[i]], edgecolors="k",
                         linewidths=0.5, alpha=0.95, zorder=5)
            ax3d.text(S_k[i] + 0.01, S_t[i] + 0.01, S_e[i] + 0.02,
                      formulas[i], fontsize=4.5, zorder=6)
        else:
            ax3d.scatter(S_k[i], S_t[i], S_e[i], s=12, c="lightgrey",
                         alpha=0.4, zorder=2)

    # Translucent constraint box
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    skl, skh = sk_range
    stl, sth = st_range
    sel, seh = 0.0, 1.0  # full S_e range
    vf = [
        [[skl,stl,sel],[skh,stl,sel],[skh,sth,sel],[skl,sth,sel]],
        [[skl,stl,seh],[skh,stl,seh],[skh,sth,seh],[skl,sth,seh]],
        [[skl,stl,sel],[skh,stl,sel],[skh,stl,seh],[skl,stl,seh]],
        [[skl,sth,sel],[skh,sth,sel],[skh,sth,seh],[skl,sth,seh]],
        [[skl,stl,sel],[skl,sth,sel],[skl,sth,seh],[skl,stl,seh]],
        [[skh,stl,sel],[skh,sth,sel],[skh,sth,seh],[skh,stl,seh]],
    ]
    poly = Poly3DCollection(vf, alpha=0.07, facecolor="#ff7f0e",
                            edgecolor="#ff7f0e", linewidths=0.5)
    ax3d.add_collection3d(poly)
    _wireframe_cube(ax3d)
    ax3d.set_xlabel(r"$S_k$", labelpad=4)
    ax3d.set_ylabel(r"$S_t$", labelpad=4)
    ax3d.set_zlabel(r"$S_e$", labelpad=4)
    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.view_init(elev=25, azim=135)

    # --- Chart 2: Trie vs brute-force scaling (log-log) -------------------
    ax = axes[1]
    N = np.logspace(1, 6, 200)
    brute_force = N * 1024
    trie_const = np.full_like(N, 18)
    ax.loglog(N, brute_force, "-", color="#d62728", lw=1.8, label=r"Brute-force $O(N \times 1024)$")
    ax.loglog(N, trie_const, "-", color="#1f77b4", lw=1.8, label=r"Trie $O(18)$")
    ax.axvline(39, ls=":", color="grey", lw=0.8, alpha=0.7)
    ax.annotate("$N=39$", xy=(39, 50), fontsize=6, color="grey")
    ax.set_xlabel(r"Database size $N$")
    ax.set_ylabel("Operations per query")
    ax.legend(fontsize=6, framealpha=0.7)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    # --- Chart 3: Ternary distance vs Euclidean distance ------------------
    ax = axes[2]
    # Compute all 741 pairs
    pair_idx = list(combinations(range(len(names)), 2))
    ternary_dist = []
    euclidean_dist = []
    for i, j in pair_idx:
        # Ternary distance = common prefix length (from sim_matrix)
        td = sim_matrix[names[i]][names[j]]
        ternary_dist.append(td)
        ed = np.sqrt((S_k[i]-S_k[j])**2 + (S_t[i]-S_t[j])**2 + (S_e[i]-S_e[j])**2)
        euclidean_dist.append(ed)
    ternary_dist = np.array(ternary_dist)
    euclidean_dist = np.array(euclidean_dist)

    ax.scatter(ternary_dist, euclidean_dist, s=6, c="#333333", alpha=0.35)

    # Fit exponential decay: d_E ~ a * exp(-b * prefix_len) + c
    def decay_func(x, a, b, c):
        return a * np.exp(-b * x) + c
    try:
        popt, _ = curve_fit(decay_func, ternary_dist, euclidean_dist,
                            p0=[1.0, 0.3, 0.1], maxfev=5000)
        x_fit = np.linspace(0, 18, 200)
        ax.plot(x_fit, decay_func(x_fit, *popt), "-", color="#d62728", lw=1.5,
                label=f"$d_E \\approx {popt[0]:.2f}\\,e^{{-{popt[1]:.2f}\\,\\ell}} + {popt[2]:.2f}$")
        ax.legend(fontsize=6, framealpha=0.7)
    except Exception:
        pass  # skip fit if it fails

    ax.set_xlabel("Common prefix length")
    ax.set_ylabel(r"Euclidean distance in $S$-space")

    # --- Chart 4: Modes vs S_e -------------------------------------------
    ax = axes[3]
    for tp in TYPE_ORDER:
        idx = [i for i, t in enumerate(types) if t == tp]
        ax.scatter(n_modes[idx], S_e[idx], s=size_scale[idx],
                   c=TYPE_COLORS[tp], label=tp, alpha=0.8,
                   edgecolors="k", linewidths=0.3)
    # Annotate H2O
    h2o_i = names.index("H2O")
    ax.annotate(encodings["H2O"]["formula"],
                xy=(n_modes[h2o_i], S_e[h2o_i]),
                xytext=(n_modes[h2o_i] + 0.6, S_e[h2o_i] - 0.08),
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("Vibrational modes")
    ax.set_ylabel(r"$S_e$")
    ax.legend(fontsize=6, framealpha=0.7)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "panel_4_structure_prediction.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating panels for Categorical Compound Database...")
    for i, fn in enumerate([panel_1, panel_2, panel_3, panel_4], 1):
        path = fn()
        print(f"  Panel {i}: {path}")
    print("Done. All 4 panels (16 charts) saved to figures/.")
