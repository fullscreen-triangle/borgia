"""Five panels for the structural-correspondence manuscript.

Every value plotted is read from the results files.  Nothing is
simulated, fitted, or drawn by hand.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from panelstyle import (BLUE, DIV, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED,
                        SEQ, TEAL, panel, save, tag)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "docs", "structural-correspondence",
                   "figures")
os.makedirs(OUT, exist_ok=True)

B = json.load(open(os.path.join(RES, "exp_correspondence.json")))["results"]
P = json.load(open(os.path.join(RES, "panel_data.json")))


# =====================================================================
def panel_01():
    """The radius sweep: tolerance is a resolution parameter."""
    fig, ax = panel(threed=(2,))
    sweep = B["B0_radius_sweep"]["rows"]
    rad = np.array([r["radius"] for r in sweep])
    mc = np.array([r["mean_close"] for r in sweep])
    mf = np.array([r["mean_far"] for r in sweep])
    lo = np.array([r["min_close"] for r in sweep])
    hi = np.array([r["max_far"] for r in sweep])
    marg = np.array([r["margin"] for r in sweep])

    # A: mean overlap by radius, with the close/far band
    a0 = ax[0]
    a0.fill_between(rad, lo, mc, color=GREEN, alpha=0.16)
    a0.fill_between(rad, mf, hi, color=RED, alpha=0.16)
    a0.plot(rad, mc, "o-", color=GREEN, markersize=6, label="close (mean)")
    a0.plot(rad, lo, "o--", color=GREEN, markersize=4, alpha=0.75,
            label="close (min)")
    a0.plot(rad, hi, "s--", color=RED, markersize=4, alpha=0.75,
            label="far (max)")
    a0.plot(rad, mf, "s-", color=RED, markersize=6, label="far (mean)")
    a0.set_xlabel("refinement radius")
    a0.set_ylabel("class overlap")
    a0.set_xticks(rad)
    a0.set_ylim(-0.03, 1.02)
    a0.legend(ncol=2, loc="upper right")
    a0.grid(linestyle=":")
    tag(a0, "A")

    # B: the separation margin collapses
    a1 = ax[1]
    cols = [GREEN if m > 0 else GREY for m in marg]
    a1.bar(rad, marg, color=cols, width=0.55, edgecolor="white",
           linewidth=0.5)
    a1.axhline(0, color="black", linewidth=1.0)
    a1.set_xlabel("refinement radius")
    a1.set_ylabel("separation margin")
    a1.set_xticks(rad)
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: every pair, every radius, in 3-D
    a2 = ax[2]
    mat = P["radius_pair_matrix"]
    for i, row in enumerate(mat):
        col = GREEN if row["relation"] == "close" else RED
        xs = np.arange(len(row["by_radius"]))
        ys = np.full(len(xs), i)
        zs = np.array(row["by_radius"])
        a2.plot(xs, ys, zs, color=col, linewidth=1.4, alpha=0.9)
        a2.scatter(xs, ys, zs, color=col, s=16, depthshade=False)
    a2.set_xlabel("radius", labelpad=-5)
    a2.set_ylabel("pair", labelpad=-5)
    a2.set_zlabel("overlap", labelpad=-5)
    a2.set_xticks(range(4))
    a2.view_init(elev=22, azim=-62)
    tag(a2, "C", threed=True)

    # D: per-pair overlap at radius 0 vs radius 1
    a3 = ax[3]
    r0 = np.array([row["by_radius"][0] for row in mat])
    r1 = np.array([row["by_radius"][1] for row in mat])
    isc = np.array([row["relation"] == "close" for row in mat])
    a3.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.0)
    a3.scatter(r0[isc], r1[isc], s=70, color=GREEN, edgecolors="black",
               linewidths=0.5, label="close", zorder=3)
    a3.scatter(r0[~isc], r1[~isc], s=70, color=RED, marker="s",
               edgecolors="black", linewidths=0.5, label="far", zorder=3)
    a3.set_xlabel("overlap at radius 0")
    a3.set_ylabel("overlap at radius 1")
    a3.set_xlim(-0.04, 1.04)
    a3.set_ylim(-0.04, 1.04)
    a3.legend(loc="upper left")
    a3.grid(linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_01_radius.png"))


# =====================================================================
def panel_02():
    """Separation of the annotated groups at the working radius."""
    fig, ax = panel(threed=(3,))
    rows = B["B1_isostere_separation"]["rows"]
    close = [r for r in rows if r["relation"] == "close"]
    far = [r for r in rows if r["relation"] == "far"]
    cv = np.array([r["class_overlap"] for r in close])
    fv = np.array([r["class_overlap"] for r in far])
    ce = np.array([r["element_overlap"] for r in close])
    fe = np.array([r["element_overlap"] for r in far])

    # A: the two groups as strips, with the gap shaded
    a0 = ax[0]
    lo, hi = B["B1_isostere_separation"]["max_far"], \
             B["B1_isostere_separation"]["min_close"]
    a0.axhspan(lo, hi, color=LIGHT, alpha=0.55)
    rng = np.random.default_rng(1)
    a0.scatter(rng.normal(0, 0.045, len(cv)), cv, s=72, color=GREEN,
               edgecolors="black", linewidths=0.5, zorder=3)
    a0.scatter(1 + rng.normal(0, 0.045, len(fv)), fv, s=72, color=RED,
               marker="s", edgecolors="black", linewidths=0.5, zorder=3)
    a0.hlines([cv.mean()], -0.25, 0.25, color=GREEN, linewidth=2.2)
    a0.hlines([fv.mean()], 0.75, 1.25, color=RED, linewidth=2.2)
    a0.set_xticks([0, 1])
    a0.set_xticklabels(["close", "far"])
    a0.set_xlim(-0.45, 1.45)
    a0.set_ylabel("class overlap")
    a0.set_ylim(-0.04, 1.06)
    a0.grid(axis="y", linestyle=":")
    tag(a0, "A")

    # B: class vs element overlap
    a1 = ax[1]
    a1.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.0)
    a1.scatter(ce, cv, s=72, color=GREEN, edgecolors="black",
               linewidths=0.5, label="close", zorder=3)
    a1.scatter(fe, fv, s=72, color=RED, marker="s", edgecolors="black",
               linewidths=0.5, label="far", zorder=3)
    a1.set_xlabel("element overlap")
    a1.set_ylabel("class overlap")
    a1.set_xlim(-0.04, 1.06)
    a1.set_ylim(-0.04, 1.06)
    a1.legend(loc="upper left")
    a1.grid(linestyle=":")
    tag(a1, "B")

    # C: per-pair, sorted
    a2 = ax[2]
    allr = sorted(rows, key=lambda r: -r["class_overlap"])
    v = [r["class_overlap"] for r in allr]
    c = [GREEN if r["relation"] == "close" else RED for r in allr]
    a2.barh(range(len(v)), v, color=c, height=0.7, edgecolor="white",
            linewidth=0.4)
    a2.axvline(B["B1_isostere_separation"]["max_far"], color="black",
               linestyle="--", linewidth=1.0)
    a2.axvline(B["B1_isostere_separation"]["min_close"], color="black",
               linestyle="--", linewidth=1.0)
    a2.set_xlim(0, 1.05)
    a2.set_ylim(-0.7, len(v) - 0.3)
    a2.set_xlabel("class overlap")
    a2.set_yticks([])
    a2.set_ylabel("pair (sorted)")
    a2.grid(axis="x", linestyle=":")
    tag(a2, "C")

    # D: 3-D class / element / size
    a3 = ax[3]
    sz_c = np.array([max(r["n_heavy_a"], r["n_heavy_b"]) for r in close])
    sz_f = np.array([max(r["n_heavy_a"], r["n_heavy_b"]) for r in far])
    a3.scatter(ce, cv, sz_c, s=62, color=GREEN, edgecolors="black",
               linewidths=0.4, depthshade=False)
    a3.scatter(fe, fv, sz_f, s=62, color=RED, marker="s",
               edgecolors="black", linewidths=0.4, depthshade=False)
    a3.set_xlabel("element", labelpad=-5)
    a3.set_ylabel("class", labelpad=-5)
    a3.set_zlabel("atoms", labelpad=-5)
    a3.view_init(elev=20, azim=-60)
    tag(a3, "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_02_separation.png"))


# =====================================================================
def panel_03():
    """Correspondence on matched pairs, and how radius erodes it."""
    fig, ax = panel(threed=(1,))
    det = P["correspondence_detail"]
    mp = B["B3_matched_pair_coverage"]["rows"]

    # A: coverage against the size bound
    a0 = ax[0]
    n1 = np.array([r["n1"] for r in mp], float)
    n2 = np.array([r["n2"] for r in mp], float)
    cov = np.array([r["coverage"] for r in mp])
    bound = np.minimum(n1, n2) / np.maximum(n1, n2)
    o = np.argsort(bound)
    x = np.arange(len(mp))
    a0.bar(x, bound[o], color=LIGHT, width=0.74, label="size bound")
    a0.bar(x, cov[o], color=TEAL, width=0.44, label="coverage")
    a0.set_xlabel("pair (sorted)")
    a0.set_ylabel("coverage")
    a0.set_ylim(0, 1.06)
    a0.set_xticks([])
    a0.legend(loc="lower right")
    a0.grid(axis="y", linestyle=":")
    tag(a0, "A")

    # B: 3-D coverage decay with radius
    a1 = ax[1]
    for i, row in enumerate(det):
        xs = np.array([b["radius"] for b in row["by_radius"]])
        ys = np.full(len(xs), i)
        zs = np.array([b["coverage"] for b in row["by_radius"]])
        a1.plot(xs, ys, zs, color=SEQ(i / max(1, len(det) - 1)),
                linewidth=1.6)
        a1.scatter(xs, ys, zs, color=SEQ(i / max(1, len(det) - 1)),
                   s=18, depthshade=False)
    a1.set_xlabel("radius", labelpad=-5)
    a1.set_ylabel("pair", labelpad=-5)
    a1.set_zlabel("coverage", labelpad=-5)
    a1.set_xticks(range(4))
    a1.view_init(elev=22, azim=-60)
    tag(a1, "B", threed=True)

    # C: matched atoms against structure size
    a2 = ax[2]
    matched = np.array([r["matched"] for r in mp], float)
    mn = np.minimum(n1, n2)
    a2.plot([0, mn.max() + 1], [0, mn.max() + 1], color=GREY,
            linestyle="--", linewidth=1.0)
    a2.scatter(mn, matched, s=76, c=cov, cmap=SEQ, vmin=0.5, vmax=1.0,
               edgecolors="black", linewidths=0.5, zorder=3)
    a2.set_xlabel("atoms in smaller structure")
    a2.set_ylabel("atoms matched")
    a2.grid(linestyle=":")
    tag(a2, "C")

    # D: classes shared, by radius
    a3 = ax[3]
    for rad in range(4):
        vals = [row["by_radius"][rad]["classes_shared"] for row in det]
        a3.plot(range(len(det)), vals, "o-", markersize=4,
                color=SEQ(rad / 3), label=f"r={rad}")
    a3.set_xlabel("pair")
    a3.set_ylabel("classes shared")
    a3.set_xticks([])
    a3.legend(ncol=2)
    a3.grid(axis="y", linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_03_correspondence.png"))


# =====================================================================
def panel_04():
    """The rewiring control."""
    fig, ax = panel(threed=(2,))
    ctrl = B["B4_negative_control"]
    true_margin = ctrl["true_margin"]
    rew_margin = ctrl["separation_margin_shuffled"]

    # A: margin, true against rewired
    a0 = ax[0]
    a0.bar([0, 1], [true_margin, rew_margin],
           color=[GREEN, RED], width=0.5, edgecolor="white", linewidth=0.5)
    a0.axhline(0, color="black", linewidth=1.0)
    a0.set_xticks([0, 1])
    a0.set_xticklabels(["true", "rewired"])
    a0.set_ylabel("separation margin")
    a0.grid(axis="y", linestyle=":")
    tag(a0, "A")

    # B: group means, true against rewired
    a1 = ax[1]
    b1 = B["B1_isostere_separation"]
    x = np.arange(2)
    w = 0.36
    a1.bar(x - w / 2, [b1["mean_close"], b1["mean_far"]], w,
           color=[GREEN, RED], edgecolor="white", linewidth=0.5,
           label="true")
    a1.bar(x + w / 2, [ctrl["mean_close_shuffled"],
                       ctrl["mean_far_shuffled"]], w,
           color=[GREEN, RED], alpha=0.42, edgecolor="white",
           linewidth=0.5, label="rewired")
    a1.set_xticks(x)
    a1.set_xticklabels(["close", "far"])
    a1.set_ylabel("mean class overlap")
    a1.set_ylim(0, 1.02)
    a1.legend()
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: 3-D, the margin surface over radius and condition
    a2 = ax[2]
    sweep = B["B0_radius_sweep"]["rows"]
    rad = np.array([r["radius"] for r in sweep], float)
    marg = np.array([r["margin"] for r in sweep])
    for j, (vals, col, lbl) in enumerate((
            (marg, GREEN, "true"),
            (np.full(len(rad), rew_margin), RED, "rewired"))):
        a2.bar3d(rad - 0.3, np.full(len(rad), j) - 0.3,
                 np.zeros(len(rad)), 0.6, 0.6, vals,
                 color=col, alpha=0.85, shade=True,
                 edgecolor="white", linewidth=0.3)
    a2.set_xlabel("radius", labelpad=-5)
    a2.set_ylabel("condition", labelpad=-5)
    a2.set_zlabel("margin", labelpad=-5)
    a2.set_xticks(range(4))
    a2.set_yticks([0, 1])
    a2.set_yticklabels(["true", "rewired"])
    a2.view_init(elev=22, azim=-58)
    tag(a2, "C", threed=True)

    # D: cross-element pairings per pair
    a3 = ax[3]
    cr = B["B2_cross_element_correspondence"]["rows"]
    cr = sorted(cr, key=lambda r: -r["cross_element_pairs"])
    same = np.array([r["element_pairs"] for r in cr], float)
    cross = np.array([r["cross_element_pairs"] for r in cr], float)
    x = np.arange(len(cr))
    a3.bar(x, same, color=BLUE, width=0.68, edgecolor="white",
           linewidth=0.4, label="same element")
    a3.bar(x, cross, bottom=same, color=ORANGE, width=0.68,
           edgecolor="white", linewidth=0.4, label="cross element")
    a3.set_xlabel("pair (sorted)")
    a3.set_ylabel("pairings")
    a3.set_xticks([])
    a3.legend()
    a3.grid(axis="y", linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_04_control.png"))


# =====================================================================
def panel_05():
    """The drug-like overlap matrix: class against element."""
    fig, ax = panel(threed=(3,))
    M = P["druglike_overlap_matrix"]
    cls = np.array(M["class_overlap"])
    ele = np.array(M["element_overlap"])
    n = len(cls)

    # A: class-overlap matrix
    a0 = ax[0]
    im0 = a0.imshow(cls, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_xlabel("structure")
    a0.set_ylabel("structure")
    cb0 = fig.colorbar(im0, ax=a0, pad=0.02, shrink=0.86, aspect=16)
    cb0.set_label("class overlap", fontsize=8)
    cb0.ax.tick_params(labelsize=7)
    tag(a0, "A")

    # B: element-overlap matrix
    a1 = ax[1]
    im1 = a1.imshow(ele, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    a1.set_xticks([])
    a1.set_yticks([])
    a1.set_xlabel("structure")
    a1.set_ylabel("structure")
    cb1 = fig.colorbar(im1, ax=a1, pad=0.02, shrink=0.86, aspect=16)
    cb1.set_label("element overlap", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    tag(a1, "B")

    # C: the two distributions
    a2 = ax[2]
    iu = np.triu_indices(n, 1)
    cv, ev = cls[iu], ele[iu]
    bins = np.linspace(0, 1, 26)
    a2.hist(ev, bins=bins, color=BLUE, alpha=0.55, label="element",
            edgecolor="white", linewidth=0.3)
    a2.hist(cv, bins=bins, color=ORANGE, alpha=0.65, label="class",
            edgecolor="white", linewidth=0.3)
    a2.axvline(cv.mean(), color=ORANGE, linestyle="--", linewidth=1.4)
    a2.axvline(ev.mean(), color=BLUE, linestyle="--", linewidth=1.4)
    a2.set_xlabel("overlap")
    a2.set_ylabel("pairs")
    a2.legend()
    a2.grid(axis="y", linestyle=":")
    tag(a2, "C")

    # D: 3-D, class vs element vs size difference
    a3 = ax[3]
    sizes = np.array([sum(1 for a in P["atoms"]
                          if a["molecule"] == nm) for nm in M["names"]],
                     float)
    dsz = np.abs(sizes[iu[0]] - sizes[iu[1]])
    a3.scatter(ev, cv, dsz, c=cv - ev, cmap=DIV, vmin=-1, vmax=1,
               s=13, alpha=0.85, edgecolors="none", depthshade=False)
    a3.set_xlabel("element", labelpad=-5)
    a3.set_ylabel("class", labelpad=-5)
    a3.set_zlabel("size diff.", labelpad=-5)
    a3.view_init(elev=20, azim=-62)
    tag(a3, "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_05_matrix.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04, panel_05):
        print("wrote", os.path.basename(fn()))
