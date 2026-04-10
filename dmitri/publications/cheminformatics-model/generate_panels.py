#!/usr/bin/env python3
"""
Generate 6 publication panels for the Categorical Cheminformatics Models paper.
Each panel: 4 charts in a row, white background, at least one 3D chart,
minimal text, no tables or conceptual diagrams.

Author: Kundai Farai Sachikonye
"""

import json
import math
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# Load data
with open(RESULTS_DIR / "compound_encodings.json", "r", encoding="utf-8") as f:
    encodings = json.load(f)
with open(RESULTS_DIR / "model_i_identification.json", "r") as f:
    model_i = json.load(f)
with open(RESULTS_DIR / "model_ii_similarity.json", "r") as f:
    model_ii = json.load(f)
with open(RESULTS_DIR / "model_iii_property_prediction.json", "r") as f:
    model_iii = json.load(f)
with open(RESULTS_DIR / "model_iv_reaction_feasibility.json", "r") as f:
    model_iv = json.load(f)
with open(RESULTS_DIR / "model_v_gpu_observation.json", "r") as f:
    model_v = json.load(f)
with open(RESULTS_DIR / "model_vi_compiled_probe.json", "r") as f:
    model_vi = json.load(f)

keys = sorted(encodings.keys())

TYPE_COLORS = {
    "diatomic": "#3b82f6",
    "triatomic": "#22c55e",
    "tetra": "#f97316",
    "poly": "#ef4444",
}

def get_color(key):
    return TYPE_COLORS.get(encodings[key]["type"], "#888888")

def add_panel_label(ax, label, x=-0.08, y=1.05):
    if hasattr(ax, "get_zlim"):
        # 3D axes: use fig text annotation instead
        pos = ax.get_position()
        ax.figure.text(pos.x0 + 0.01, pos.y1 - 0.01, label,
                       fontsize=12, fontweight="bold", va="top", ha="left")
    else:
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="right")


# =========================================================================
# PANEL 1: Model I — Categorical Identification
# =========================================================================
def panel_1():
    fig = plt.figure(figsize=(20, 4.5))

    # (A) 3D scatter of all compounds in S-space, colored by type
    ax1 = fig.add_subplot(141, projection="3d")
    for key in keys:
        e = encodings[key]
        ax1.scatter(e["S_k"], e["S_t"], e["S_e"],
                    c=get_color(key), s=40, alpha=0.8, edgecolors="k", linewidth=0.3)
    ax1.set_xlabel("$S_k$")
    ax1.set_ylabel("$S_t$")
    ax1.set_zlabel("$S_e$")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    ax1.view_init(elev=25, azim=135)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) Resolution vs depth: unique fraction
    ax2 = fig.add_subplot(142)
    depths = [d["depth"] for d in model_i["depth_resolution"]]
    unique_frac = [d["unique_fraction"] for d in model_i["depth_resolution"]]
    ax2.bar(depths, unique_frac, width=2, color="#3b82f6", edgecolor="k", linewidth=0.5, alpha=0.85)
    ax2.set_xlabel("Trit depth $k$")
    ax2.set_ylabel("Unique fraction")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)
    add_panel_label(ax2, "B")

    # (C) Speedup vs N on log scale
    ax3 = fig.add_subplot(143)
    # Validated + projected
    validated = [s for s in model_i["speedup_analysis"] if "scale" not in s]
    projected = [s for s in model_i["speedup_analysis"] if s.get("scale") == "projected"]
    ax3.semilogy([v["depth"] for v in validated], [v["speedup"] for v in validated],
                 "o-", color="#3b82f6", markersize=5, label="$N=39$")
    # Show PubChem-scale at depth 18
    pub_18 = [p for p in projected if p["depth"] == 18]
    if pub_18:
        for p in pub_18:
            ax3.semilogy(18, p["speedup"], "^", color="#ef4444", markersize=8)
    ax3.set_xlabel("Trit depth $k$")
    ax3.set_ylabel("Speedup ($Nd/k$)")
    ax3.axhline(1, color="grey", linestyle=":", linewidth=0.5)
    add_panel_label(ax3, "C")

    # (D) Collision count vs depth
    ax4 = fig.add_subplot(144)
    collision_counts = [d["num_collision_groups"] for d in model_i["depth_resolution"]]
    non_unique = [d["total_compounds"] - d["unique_compounds"] for d in model_i["depth_resolution"]]
    ax4.fill_between(depths, non_unique, alpha=0.3, color="#ef4444")
    ax4.plot(depths, non_unique, "o-", color="#ef4444", markersize=5, linewidth=1.5)
    ax4.set_xlabel("Trit depth $k$")
    ax4.set_ylabel("Unresolved compounds")
    ax4.set_ylim(bottom=0)
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_1_identification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 1 saved")


# =========================================================================
# PANEL 2: Model II — Categorical Similarity
# =========================================================================
def shared_prefix_depth(t1, t2):
    d = 0
    for a, b in zip(t1, t2):
        if a == b:
            d += 1
        else:
            break
    return d

def panel_2():
    fig = plt.figure(figsize=(20, 4.5))

    # (A) 3D: compounds colored by family membership
    ax1 = fig.add_subplot(141, projection="3d")
    family_colors = {
        "Halomethanes": "#e11d48",
        "Hydrogen halides": "#7c3aed",
        "Homonuclear diatomics": "#2563eb",
        "Small hydrocarbons": "#059669",
        "Linear triatomics": "#d97706",
        "Bent triatomics": "#dc2626",
    }
    family_members = {}
    for fname, fdata in model_ii["family_cohesion"].items():
        for m in fdata["members"]:
            family_members[m] = fname

    for key in keys:
        e = encodings[key]
        if key in family_members:
            c = family_colors.get(family_members[key], "#999")
            s = 60
        else:
            c = "#cccccc"
            s = 20
        ax1.scatter(e["S_k"], e["S_t"], e["S_e"],
                    c=c, s=s, alpha=0.85, edgecolors="k", linewidth=0.3)
    ax1.set_xlabel("$S_k$"); ax1.set_ylabel("$S_t$"); ax1.set_zlabel("$S_e$")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    ax1.view_init(elev=20, azim=120)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) Cohesion ratio bar chart
    ax2 = fig.add_subplot(142)
    fnames = list(model_ii["family_cohesion"].keys())
    ratios = [model_ii["family_cohesion"][f]["intra_inter_ratio"] for f in fnames]
    colors_bar = ["#22c55e" if r > 1 else "#f97316" for r in ratios]
    short_names = [f.split()[0] for f in fnames]
    bars = ax2.barh(range(len(fnames)), ratios, color=colors_bar, edgecolor="k", linewidth=0.5)
    ax2.set_yticks(range(len(fnames)))
    ax2.set_yticklabels(short_names, fontsize=7)
    ax2.axvline(1.0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Cohesion ratio $R$")
    add_panel_label(ax2, "B")

    # (C) Similarity matrix heatmap (subset: 15 representative compounds)
    ax3 = fig.add_subplot(143)
    subset = ["H2", "HF", "HCl", "N2", "O2", "H2O", "CO2", "SO2",
              "CH4", "NH3", "C2H2", "C6H6", "CH3F", "CH3Cl", "CCl4"]
    n = len(subset)
    mat = np.zeros((n, n))
    for i, a in enumerate(subset):
        for j, b in enumerate(subset):
            mat[i, j] = shared_prefix_depth(
                encodings[a]["trit_string"], encodings[b]["trit_string"]
            )
    im = ax3.imshow(mat, cmap="YlOrRd", aspect="equal", vmin=0, vmax=18)
    ax3.set_xticks(range(n)); ax3.set_xticklabels(subset, rotation=90, fontsize=5)
    ax3.set_yticks(range(n)); ax3.set_yticklabels(subset, fontsize=5)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="Prefix depth")
    add_panel_label(ax3, "C")

    # (D) Intra vs inter similarity scatter per family
    ax4 = fig.add_subplot(144)
    for fname in fnames:
        fd = model_ii["family_cohesion"][fname]
        ax4.scatter(fd["mean_inter_similarity"], fd["mean_intra_similarity"],
                    s=80, c=family_colors.get(fname, "#999"),
                    edgecolors="k", linewidth=0.5, zorder=3)
    # Diagonal line (R=1)
    lim = max(max(fd["mean_intra_similarity"] for fd in model_ii["family_cohesion"].values()),
              max(fd["mean_inter_similarity"] for fd in model_ii["family_cohesion"].values())) + 0.5
    ax4.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax4.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="green")
    ax4.set_xlabel("Mean inter-family $\\bar{\\sigma}_{\\mathrm{inter}}$")
    ax4.set_ylabel("Mean intra-family $\\bar{\\sigma}_{\\mathrm{intra}}$")
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_2_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 2 saved")


# =========================================================================
# PANEL 3: Model III — Property Prediction (ZPVE)
# =========================================================================
def panel_3():
    fig = plt.figure(figsize=(20, 4.5))
    preds = model_iii["predictions"]

    # (A) 3D: compounds sized by ZPVE, colored by prediction error
    ax1 = fig.add_subplot(141, projection="3d")
    for p in preds:
        key = p["compound"]
        e = encodings[key]
        err = p["percent_error"]
        color = cm.RdYlGn_r(min(err / 50.0, 1.0))
        size = max(10, p["zpve_nist_kj_mol"] / 5)
        ax1.scatter(e["S_k"], e["S_t"], e["S_e"],
                    c=[color], s=size, alpha=0.8, edgecolors="k", linewidth=0.3)
    ax1.set_xlabel("$S_k$"); ax1.set_ylabel("$S_t$"); ax1.set_zlabel("$S_e$")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    ax1.view_init(elev=25, azim=140)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) Predicted vs actual ZPVE scatter
    ax2 = fig.add_subplot(142)
    actual = [p["zpve_nist_kj_mol"] for p in preds]
    predicted = [p["zpve_predicted_kj_mol"] for p in preds]
    ax2.scatter(actual, predicted, c=[get_color(p["compound"]) for p in preds],
                s=30, edgecolors="k", linewidth=0.3, alpha=0.8, zorder=3)
    lim = max(max(actual), max(predicted)) * 1.1
    ax2.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("ZPVE (NIST) [kJ/mol]")
    ax2.set_ylabel("ZPVE (predicted) [kJ/mol]")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    add_panel_label(ax2, "B")

    # (C) Error distribution histogram
    ax3 = fig.add_subplot(143)
    errors = [p["percent_error"] for p in preds]
    bins = np.arange(0, max(errors) + 10, 5)
    ax3.hist(errors, bins=bins, color="#3b82f6", edgecolor="k", linewidth=0.5, alpha=0.8)
    ax3.axvline(np.median(errors), color="red", linestyle="--", linewidth=1, label=f"median={np.median(errors):.1f}%")
    ax3.set_xlabel("Percent error (%)")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=7)
    add_panel_label(ax3, "C")

    # (D) Residual vs S-entropy distance to nearest neighbour
    ax4 = fig.add_subplot(144)
    for p in preds:
        key = p["compound"]
        e = encodings[key]
        s_q = np.array([e["S_k"], e["S_t"], e["S_e"]])
        min_dist = float("inf")
        for k2 in keys:
            if k2 == key:
                continue
            e2 = encodings[k2]
            d = np.linalg.norm(s_q - np.array([e2["S_k"], e2["S_t"], e2["S_e"]]))
            if d < min_dist:
                min_dist = d
        ax4.scatter(min_dist, p["percent_error"], c=get_color(key),
                    s=30, edgecolors="k", linewidth=0.3, alpha=0.8)
    ax4.set_xlabel("Distance to nearest neighbour")
    ax4.set_ylabel("Prediction error (%)")
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_3_property_prediction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 3 saved")


# =========================================================================
# PANEL 4: Model IV — Reaction Feasibility
# =========================================================================
def panel_4():
    fig = plt.figure(figsize=(20, 4.5))
    rxns = model_iv["reactions"]

    # (A) 3D: reactant and product centroids connected by arrows
    ax1 = fig.add_subplot(141, projection="3d")
    for rxn in rxns:
        rc = rxn["reactant_centroid"]
        pc = rxn["product_centroid"]
        if pc is not None:
            color = "#22c55e" if rxn["correct"] else "#ef4444"
            ax1.scatter(*rc, c=color, s=40, marker="o", edgecolors="k", linewidth=0.3, alpha=0.7)
            ax1.scatter(*pc, c=color, s=40, marker="^", edgecolors="k", linewidth=0.3, alpha=0.7)
            ax1.plot([rc[0], pc[0]], [rc[1], pc[1]], [rc[2], pc[2]],
                     color=color, linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("$S_k$"); ax1.set_ylabel("$S_t$"); ax1.set_zlabel("$S_e$")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    ax1.view_init(elev=20, azim=130)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) Trajectory distance distribution
    ax2 = fig.add_subplot(142)
    dists = [r["trajectory_distance"] for r in rxns if r["trajectory_distance"] is not None]
    labels_short = [r["equation"].split("->")[0].strip()[:12] for r in rxns if r["trajectory_distance"] is not None]
    feasible_mask = [r["known_feasibility"] == "feasible" for r in rxns if r["trajectory_distance"] is not None]
    colors_bar = ["#22c55e" if f else "#ef4444" for f in feasible_mask]
    y_pos = range(len(dists))
    ax2.barh(y_pos, dists, color=colors_bar, edgecolor="k", linewidth=0.5, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_short, fontsize=5)
    ax2.set_xlabel("Trajectory distance in $\\mathcal{S}$")
    add_panel_label(ax2, "B")

    # (C) Reactant vs product distance to equilibrium (1,1,1)
    ax3 = fig.add_subplot(143)
    eq = np.array([1.0, 1.0, 1.0])
    for rxn in rxns:
        rc = np.array(rxn["reactant_centroid"])
        pc = rxn["product_centroid"]
        if pc is not None:
            pc = np.array(pc)
            dr = np.linalg.norm(rc - eq)
            dp = np.linalg.norm(pc - eq)
            color = "#22c55e" if rxn["predicted_direction"] == "forward" else "#f97316"
            ax3.scatter(dr, dp, c=color, s=50, edgecolors="k", linewidth=0.5, zorder=3)
    lim = 1.2
    ax3.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax3.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color="green")
    ax3.set_xlabel("$d(\\mathbf{s}_{\\mathrm{react}}, (1,1,1))$")
    ax3.set_ylabel("$d(\\mathbf{s}_{\\mathrm{prod}}, (1,1,1))$")
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim)
    add_panel_label(ax3, "C")

    # (D) Accuracy summary: correct/incorrect stacked
    ax4 = fig.add_subplot(144)
    categories = ["Feasible", "Infeasible"]
    correct_f = sum(1 for r in rxns if r["known_feasibility"] == "feasible" and r["correct"])
    wrong_f = sum(1 for r in rxns if r["known_feasibility"] == "feasible" and not r["correct"])
    correct_i = sum(1 for r in rxns if r["known_feasibility"] == "infeasible" and r["correct"])
    wrong_i = sum(1 for r in rxns if r["known_feasibility"] == "infeasible" and not r["correct"])
    ax4.bar(categories, [correct_f, correct_i], color="#22c55e", edgecolor="k", linewidth=0.5, label="Correct")
    ax4.bar(categories, [wrong_f, wrong_i], bottom=[correct_f, correct_i],
            color="#ef4444", edgecolor="k", linewidth=0.5, label="Incorrect")
    ax4.set_ylabel("Number of reactions")
    ax4.legend(fontsize=7)
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_4_reaction_feasibility.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 4 saved")


# =========================================================================
# PANEL 5: Model V — GPU Partition Observation
# =========================================================================
def partition_observation_function(u, modes, S_k, S_t, S_e, omega_ref=4401.0):
    sigma = 0.02 * (1.0 - 0.5 * S_k)
    intensity = sum(math.exp(-((u - w / omega_ref) / sigma) ** 2) for w in modes)
    bw = 0.1 + 0.4 * S_t
    envelope = math.exp(-((u - 0.5) / bw) ** 2)
    depth_mod = 1.0 + 0.3 * S_e * math.sin(50.0 * S_e * u)
    return max(0.0, min(1.0, intensity * envelope * depth_mod))

def panel_5():
    fig = plt.figure(figsize=(20, 4.5))

    # (A) 3D surface: observation function for H₂O
    ax1 = fig.add_subplot(141, projection="3d")
    e = encodings["H2O"]
    res = 80
    U = np.linspace(0, 1, res)
    V = np.linspace(0, 1, res)
    U_grid, V_grid = np.meshgrid(U, V)
    Z = np.zeros_like(U_grid)
    for iy in range(res):
        phase_mod = 0.8 + 0.2 * math.sin(V[iy] * 2 * math.pi)
        for ix in range(res):
            Z[iy, ix] = partition_observation_function(
                U[ix], e["modes"], e["S_k"], e["S_t"], e["S_e"]
            ) * phase_mod
    ax1.plot_surface(U_grid, V_grid, Z, cmap="viridis", alpha=0.85,
                     linewidth=0, antialiased=True)
    ax1.set_xlabel("$u$ (freq. address)")
    ax1.set_ylabel("$v$ (phase)")
    ax1.set_zlabel("$\\mathcal{A}(u,v)$")
    ax1.view_init(elev=30, azim=225)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) 1D observation profiles for 4 molecules
    ax2 = fig.add_subplot(142)
    test_mols = ["H2O", "CO2", "CH4", "C6H6"]
    colors_line = ["#3b82f6", "#22c55e", "#f97316", "#ef4444"]
    u_arr = np.linspace(0, 1, 500)
    for mol, col in zip(test_mols, colors_line):
        em = encodings[mol]
        profile = [partition_observation_function(u, em["modes"], em["S_k"], em["S_t"], em["S_e"]) for u in u_arr]
        ax2.plot(u_arr, profile, color=col, linewidth=1.2, alpha=0.9, label=mol)
    ax2.set_xlabel("$u$ (normalised frequency)")
    ax2.set_ylabel("$\\mathcal{A}(u)$")
    ax2.legend(fontsize=6, loc="upper right")
    add_panel_label(ax2, "B")

    # (C) Self vs cross interference visibility
    ax3 = fig.add_subplot(143)
    self_vis = [s["visibility"] for s in model_v["self_similarity"]]
    cross_vis = [s["interference_visibility"] for s in model_v["interference_similarity"]]
    ax3.boxplot([self_vis, cross_vis], labels=["Self", "Cross"],
                patch_artist=True,
                boxprops=dict(facecolor="#3b82f6", alpha=0.4),
                medianprops=dict(color="red", linewidth=1.5))
    ax3.set_ylabel("Interference visibility $\\bar{\\mathcal{V}}$")
    add_panel_label(ax3, "C")

    # (D) Interference visibility vs S-entropy distance
    ax4 = fig.add_subplot(144)
    for pair in model_v["interference_similarity"]:
        ax4.scatter(pair["s_entropy_distance"], pair["interference_visibility"],
                    s=50, c="#3b82f6", edgecolors="k", linewidth=0.5, alpha=0.8, zorder=3)
    ax4.set_xlabel("S-entropy distance $d$")
    ax4.set_ylabel("Visibility $\\bar{\\mathcal{V}}$")
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_5_gpu_observation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 5 saved")


# =========================================================================
# PANEL 6: Model VI — Compiled Probe Training
# =========================================================================
def panel_6():
    fig = plt.figure(figsize=(20, 4.5))
    history = model_vi["training_history"]

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]
    sharps = [h["partition_sharpness"] for h in history]
    cohs = [h["phase_coherence"] for h in history]

    # (A) 3D: loss landscape (epoch × sharpness × loss)
    ax1 = fig.add_subplot(141, projection="3d")
    colors_3d = cm.coolwarm(np.array(accs))
    ax1.scatter(epochs, sharps, losses, c=colors_3d, s=30, edgecolors="k", linewidth=0.3)
    ax1.plot(epochs, sharps, losses, color="grey", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sharpness")
    ax1.set_zlabel("Loss")
    ax1.view_init(elev=25, azim=135)
    add_panel_label(ax1, "A", x=-0.02)

    # (B) Training curves: loss and accuracy
    ax2 = fig.add_subplot(142)
    ax2_twin = ax2.twinx()
    ax2.plot(epochs, losses, "o-", color="#ef4444", markersize=2, linewidth=1.2, label="Loss")
    ax2_twin.plot(epochs, accs, "s-", color="#22c55e", markersize=2, linewidth=1.2, label="Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss", color="#ef4444")
    ax2_twin.set_ylabel("Accuracy", color="#22c55e")
    ax2.tick_params(axis="y", labelcolor="#ef4444")
    ax2_twin.tick_params(axis="y", labelcolor="#22c55e")
    add_panel_label(ax2, "B")

    # (C) Physical observables over training
    ax3 = fig.add_subplot(143)
    ax3.plot(epochs, sharps, "o-", color="#3b82f6", markersize=2, linewidth=1.2, label="Sharpness")
    ax3.plot(epochs, cohs, "s-", color="#f97316", markersize=2, linewidth=1.2, label="Coherence")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Observable value")
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, 1)
    add_panel_label(ax3, "C")

    # (D) Parameter count comparison (log scale bar chart)
    ax4 = fig.add_subplot(144)
    systems = ["ECFP4\n+Tanimoto", "GNN\n(MPNN)", "Transformer\n(ChemBERTa)", "Categorical\n(I-IV)", "Compiled\nProbe (VI)"]
    params = [0.001, 5e5, 5e7, 0.001, 6e5]  # 0.001 = placeholder for "0"
    colors_sys = ["#9ca3af", "#9ca3af", "#9ca3af", "#22c55e", "#3b82f6"]
    bars = ax4.bar(systems, params, color=colors_sys, edgecolor="k", linewidth=0.5)
    ax4.set_yscale("log")
    ax4.set_ylabel("Trainable parameters")
    ax4.set_ylim(1e-1, 1e9)
    # Mark zero-parameter systems
    for i, p in enumerate(params):
        if p < 1:
            ax4.text(i, 1, "0", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#22c55e")
    add_panel_label(ax4, "D")

    fig.tight_layout(pad=2.0)
    fig.savefig(FIGURES_DIR / "panel_6_compiled_probe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Panel 6 saved")


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    print("Generating 6 panels for Categorical Cheminformatics Models...")
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    panel_6()
    print(f"\nAll panels saved to {FIGURES_DIR}")
    for p in sorted(FIGURES_DIR.glob("panel_*.png")):
        print(f"  {p.name}")
