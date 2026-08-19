"""Five panels for the canonical-ranking manuscript.

Every value plotted is read from the results files.  Nothing is
simulated, fitted, or drawn by hand.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from panelstyle import (BLUE, DIV, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED,
                        SEQ, TEAL, panel, save, tag)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "docs", "cannonical-ranking-algorithm",
                   "figures")
os.makedirs(OUT, exist_ok=True)

A = json.load(open(os.path.join(RES, "exp_ranking.json")))["results"]
P = json.load(open(os.path.join(RES, "panel_data.json")))


# =====================================================================
def panel_01():
    """The base cut key: a small, unevenly populated discrete space."""
    fig, ax = panel(threed=(0,))
    atoms = P["atoms"]
    sig = np.array([a["sigma"] for a in atoms])
    dep = np.array([a["depth"] for a in atoms])
    deg = np.array([a["heavy_degree"] for a in atoms])
    zs = np.array([a["z"] for a in atoms])

    # A: 3-D occupancy of the (sigma, depth, degree) cells
    a0 = ax[0]
    cells = Counter(zip(sig, dep, deg))
    mx = max(cells.values())
    for (s_, d_, g_), n in cells.items():
        a0.bar3d(s_ - 0.18, d_ - 0.28, 0, 0.36, 0.56, n,
                 color=SEQ(n / mx), alpha=0.92, shade=True,
                 edgecolor="white", linewidth=0.3)
    a0.set_xlabel(r"$\sigma$", labelpad=-4)
    a0.set_ylabel("burial depth", labelpad=-4)
    a0.set_zlabel("atoms", labelpad=-4)
    a0.set_xticks([5, 6])
    a0.set_yticks([1, 2, 3, 4])
    a0.view_init(elev=24, azim=-58)
    tag(a0, "A", threed=True)

    # B: composition of each sigma level by depth
    a1 = ax[1]
    levels = sorted(set(sig.tolist()))
    depths = sorted(set(dep.tolist()))
    bottom = np.zeros(len(levels))
    for i, d_ in enumerate(depths):
        vals = np.array([np.sum((sig == s_) & (dep == d_)) for s_ in levels],
                        float)
        a1.bar([str(int(s_)) for s_ in levels], vals, 0.62, bottom=bottom,
               color=SEQ(i / max(1, len(depths) - 1)),
               edgecolor="white", linewidth=0.5,
               label=f"depth {d_}")
        bottom += vals
    a1.set_xlabel(r"$\sigma$")
    a1.set_ylabel("atoms")
    a1.legend(ncol=2, loc="upper right")
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: key-cell occupancy, area proportional to count
    a2 = ax[2]
    grid = P["sigma_depth_grid"]
    gs = np.array([g["sigma"] for g in grid])
    gd = np.array([g["depth"] for g in grid])
    gc = np.array([g["count"] for g in grid], float)
    sc2 = a2.scatter(gs, gd, s=gc * 11, c=gc, cmap=SEQ,
                     edgecolors="black", linewidths=0.5)
    a2.set_xlabel(r"$\sigma$")
    a2.set_ylabel("burial depth")
    a2.set_xticks([5, 6])
    a2.set_yticks(sorted(set(gd.tolist())))
    a2.set_xlim(4.6, 6.4)
    a2.set_ylim(0.4, 4.6)
    a2.grid(linestyle=":")
    cb2 = fig.colorbar(sc2, ax=a2, pad=0.02, shrink=0.85, aspect=16)
    cb2.set_label("atoms", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    tag(a2, "C")

    # D: distinct-key ratio per molecule
    a3 = ax[3]
    rows = [r for r in A["A1_base_coarseness"]["rows"] if "base_ratio" in r]
    rows.sort(key=lambda r: r["base_ratio"])
    vals = [r["base_ratio"] for r in rows]
    mean = A["A1_base_coarseness"]["mean_base_ratio"]
    cols = [RED if v < mean else BLUE for v in vals]
    a3.bar(range(len(vals)), vals, color=cols, width=0.82,
           edgecolor="white", linewidth=0.3)
    a3.axhline(mean, color="black", linestyle="--", linewidth=1.1)
    a3.axhline(1.0, color=GREY, linestyle=":", linewidth=1.0)
    a3.set_ylim(0, 1.08)
    a3.set_xlabel("molecule (sorted)")
    a3.set_ylabel("distinct keys / atoms")
    a3.set_xticks([])
    a3.grid(axis="y", linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_01_base_key.png"))


# =====================================================================
def panel_02():
    """Refinement: trajectories, speed, and the gain it buys."""
    fig, ax = panel(threed=(1,))
    curves = P["refinement_curves"]

    # A: class count per round
    a0 = ax[0]
    for c in curves:
        h = c["history"]
        col = RED if c["corpus"] == "symmetric" else BLUE
        a0.plot(range(len(h)), np.array(h) / c["n_heavy"], color=col,
                alpha=0.4, linewidth=1.0)
    for lbl, col in (("symmetric", RED), ("drug-like", BLUE)):
        a0.plot([], [], color=col, linewidth=2, label=lbl)
    a0.set_xlabel("refinement round")
    a0.set_ylabel("classes / atoms")
    a0.set_ylim(0, 1.05)
    a0.legend(loc="lower right")
    a0.grid(linestyle=":")
    tag(a0, "A")

    # B: 3-D trajectory surface
    a1 = ax[1]
    maxr = max(len(c["history"]) for c in curves)
    for i, c in enumerate(sorted(curves, key=lambda x: x["n_heavy"])):
        h = c["history"] + [c["history"][-1]] * (maxr - len(c["history"]))
        xs = np.arange(maxr)
        ys = np.full(maxr, c["n_heavy"])
        zs = np.array(h)
        col = RED if c["corpus"] == "symmetric" else BLUE
        a1.plot(xs, ys, zs, color=col, alpha=0.55, linewidth=1.1)
    a1.set_xlabel("round", labelpad=-5)
    a1.set_ylabel("atoms", labelpad=-5)
    a1.set_zlabel("classes", labelpad=-5)
    a1.view_init(elev=22, azim=-62)
    tag(a1, "B", threed=True)

    # C: rounds to stabilise
    a2 = ax[2]
    conv = A["A4_convergence"]
    cs = Counter(conv["symmetric_rounds"])
    cd = Counter(conv["druglike_rounds"])
    ks = sorted(set(cs) | set(cd))
    w = 0.38
    a2.bar([k - w / 2 for k in ks], [cs.get(k, 0) for k in ks], w,
           color=RED, alpha=0.85, edgecolor="white", linewidth=0.4,
           label="symmetric")
    a2.bar([k + w / 2 for k in ks], [cd.get(k, 0) for k in ks], w,
           color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4,
           label="drug-like")
    a2.set_xticks(ks)
    a2.set_xlabel("rounds to stabilise")
    a2.set_ylabel("molecules")
    a2.legend()
    a2.grid(axis="y", linestyle=":")
    tag(a2, "C")

    # D: base -> stable ratio
    a3 = ax[3]
    g = A["A4b_refinement_gain"]["rows"]
    g = [r for r in g if r["final_ratio"] is not None]
    g.sort(key=lambda r: r["final_ratio"])
    base = np.array([r["base_classes"] / r["n_heavy"] for r in g])
    fin = np.array([r["final_ratio"] for r in g])
    x = np.arange(len(g))
    a3.vlines(x, base, fin, color=LIGHT, linewidth=2.2)
    a3.scatter(x, base, s=20, color=ORANGE, zorder=3, label="base key")
    a3.scatter(x, fin, s=20, color=TEAL, zorder=3, label="stable")
    a3.set_xlabel("molecule (sorted)")
    a3.set_ylabel("classes / atoms")
    a3.set_ylim(0, 1.08)
    a3.set_xticks([])
    a3.legend(loc="lower right")
    a3.grid(axis="y", linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_02_refinement.png"))


# =====================================================================
def panel_03():
    """Orbit agreement on the symmetric corpus."""
    fig, ax = panel(threed=(2,))
    rows = [r for r in A["A2_orbit_agreement"]["rows"] if "cut_classes" in r]
    rows.sort(key=lambda r: (r["reference_orbits"], r["n_heavy"]))
    ref = np.array([r["reference_orbits"] for r in rows])
    got = np.array([r["cut_classes"] for r in rows])
    nh = np.array([r["n_heavy"] for r in rows])
    rounds = np.array([r["rounds"] for r in rows])

    # A: measured vs reference
    a0 = ax[0]
    jit = (np.arange(len(rows)) % 3 - 1) * 0.055
    lim = max(ref.max(), got.max()) + 0.7
    a0.plot([0, lim], [0, lim], color=GREY, linestyle="--", linewidth=1.0)
    sc = a0.scatter(ref + jit, got + jit, c=nh, cmap=SEQ, s=68,
                    edgecolors="black", linewidths=0.5, zorder=3)
    a0.set_xlim(0.3, lim)
    a0.set_ylim(0.3, lim)
    a0.set_xlabel("reference orbits")
    a0.set_ylabel("cut classes")
    a0.set_xticks(range(1, int(lim) + 1))
    a0.set_yticks(range(1, int(lim) + 1))
    cb = fig.colorbar(sc, ax=a0, pad=0.02, shrink=0.85, aspect=16)
    cb.set_label("atoms", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    a0.grid(linestyle=":")
    tag(a0, "A")

    # B: residual, against what the control produces on the same molecules
    a1 = ax[1]
    ctrl = {r["name"]: r["control_classes"]
            for r in A["A5_negative_control"]["rows"]}
    resid_k = got - ref
    resid_c = np.array([ctrl[r["name"]] for r in rows]) - ref
    x = np.arange(len(rows))
    a1.axhline(0, color="black", linewidth=1.0)
    a1.bar(x, resid_c, color=RED, width=0.72, edgecolor="white",
           linewidth=0.3, label="index tie-break")
    a1.bar(x, resid_k, color=GREEN, width=0.72, edgecolor="white",
           linewidth=0.3, label="cut refinement")
    a1.scatter(x, resid_k, s=16, color=GREEN, zorder=4)
    a1.set_xlabel("molecule")
    a1.set_ylabel("classes  $-$  orbits")
    a1.set_xticks([])
    a1.legend(loc="upper left")
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: 3-D atoms / orbits / classes
    a2 = ax[2]
    a2.scatter(nh, ref, got, c=rounds, cmap=SEQ, s=62,
               edgecolors="black", linewidths=0.4, depthshade=False)
    mx = max(nh.max(), got.max())
    a2.plot([0, mx], [0, mx], [0, mx], color=GREY, linestyle="--",
            linewidth=0.9)
    a2.set_xlabel("atoms", labelpad=-5)
    a2.set_ylabel("orbits", labelpad=-5)
    a2.set_zlabel("classes", labelpad=-5)
    a2.view_init(elev=20, azim=-60)
    tag(a2, "C", threed=True)

    # D: symmetry compression
    a3 = ax[3]
    comp = got / nh
    o = np.argsort(comp)
    a3.barh(range(len(rows)), comp[o], color=[SEQ(v) for v in comp[o]],
            height=0.78, edgecolor="white", linewidth=0.3)
    a3.axvline(1.0, color=GREY, linestyle=":", linewidth=1.0)
    a3.set_xlim(0, 1.08)
    a3.set_ylim(-0.7, len(rows) - 0.3)
    a3.set_xlabel("classes / atoms")
    a3.set_yticks([])
    a3.set_ylabel("molecule (sorted)")
    a3.grid(axis="x", linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_03_orbits.png"))


# =====================================================================
def panel_04():
    """The negative control: index tie-breaking destroys the orbits."""
    fig, ax = panel(threed=(3,))
    ctrl = A["A5_negative_control"]["rows"]
    ctrl.sort(key=lambda r: (r["reference_orbits"], r["n_heavy"]))
    ref = np.array([r["reference_orbits"] for r in ctrl])
    con = np.array([r["control_classes"] for r in ctrl])
    nh = np.array([r["n_heavy"] for r in ctrl])
    byname = {r["name"]: r for r in A["A2_orbit_agreement"]["rows"]
              if "cut_classes" in r}
    cut = np.array([byname[r["name"]]["cut_classes"] for r in ctrl])

    # A: both against reference
    a0 = ax[0]
    lim = max(con.max(), ref.max()) + 1.0
    a0.plot([0, lim], [0, lim], color=GREY, linestyle="--", linewidth=1.0)
    a0.scatter(ref, cut, s=58, color=GREEN, edgecolors="black",
               linewidths=0.4, label="cut refinement", zorder=3)
    a0.scatter(ref, con, s=58, color=RED, marker="^",
               edgecolors="black", linewidths=0.4, label="index tie-break",
               zorder=3)
    a0.set_xlabel("reference orbits")
    a0.set_ylabel("classes produced")
    a0.set_xlim(0.3, lim)
    a0.set_ylim(0.3, lim)
    a0.legend(loc="upper left")
    a0.grid(linestyle=":")
    tag(a0, "A")

    # B: over-separation magnitude
    a1 = ax[1]
    over_c = con - ref
    over_k = cut - ref
    x = np.arange(len(ctrl))
    a1.bar(x - 0.2, over_k, 0.4, color=GREEN, edgecolor="white",
           linewidth=0.3, label="cut refinement")
    a1.bar(x + 0.2, over_c, 0.4, color=RED, edgecolor="white",
           linewidth=0.3, label="index tie-break")
    a1.axhline(0, color="black", linewidth=1.0)
    a1.set_xlabel("molecule")
    a1.set_ylabel("excess classes")
    a1.set_xticks([])
    a1.legend()
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: control tracks atom count, not symmetry
    a2 = ax[2]
    a2.plot([0, nh.max() + 1], [0, nh.max() + 1], color=GREY,
            linestyle="--", linewidth=1.0)
    a2.scatter(nh, con, s=58, color=RED, marker="^",
               edgecolors="black", linewidths=0.4, label="index tie-break")
    a2.scatter(nh, cut, s=58, color=GREEN, edgecolors="black",
               linewidths=0.4, label="cut refinement")
    a2.set_xlabel("atoms")
    a2.set_ylabel("classes produced")
    a2.legend(loc="upper left")
    a2.grid(linestyle=":")
    tag(a2, "C")

    # D: 3-D separation of the two procedures
    a3 = ax[3]
    a3.scatter(nh, ref, cut, s=52, color=GREEN, edgecolors="black",
               linewidths=0.35, depthshade=False)
    a3.scatter(nh, ref, con, s=52, color=RED, marker="^",
               edgecolors="black", linewidths=0.35, depthshade=False)
    for i in range(len(ctrl)):
        a3.plot([nh[i], nh[i]], [ref[i], ref[i]], [cut[i], con[i]],
                color=LIGHT, linewidth=1.0)
    a3.set_xlabel("atoms", labelpad=-5)
    a3.set_ylabel("orbits", labelpad=-5)
    a3.set_zlabel("classes", labelpad=-5)
    a3.view_init(elev=18, azim=-64)
    tag(a3, "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_04_control.png"))


# =====================================================================
def panel_05():
    """Class structure: how atoms distribute over classes."""
    fig, ax = panel(threed=(0,))
    atoms = P["atoms"]

    # A: 3-D class occupancy per molecule
    a0 = ax[0]
    bym = defaultdict(list)
    for a in atoms:
        bym[a["molecule"]].append(a)
    names = sorted(bym, key=lambda m: len(bym[m]))
    for i, m in enumerate(names):
        rows = bym[m]
        cnt = Counter(r["class_index"] for r in rows)
        xs = np.array(sorted(cnt))
        ys = np.full(len(xs), i)
        zs = np.array([cnt[k] for k in xs])
        a0.bar3d(xs - 0.35, ys - 0.35, np.zeros(len(xs)),
                 0.7, 0.7, zs, color=SEQ(i / max(1, len(names) - 1)),
                 alpha=0.85, shade=True)
    a0.set_xlabel("class index", labelpad=-5)
    a0.set_ylabel("molecule", labelpad=-5)
    a0.set_zlabel("atoms", labelpad=-5)
    a0.view_init(elev=26, azim=-56)
    tag(a0, "A", threed=True)

    # B: class-size distribution
    a1 = ax[1]
    sizes = Counter()
    for m, rows in bym.items():
        for _c, n in Counter(r["class_index"] for r in rows).items():
            sizes[n] += 1
    ks = sorted(sizes)
    a1.bar(ks, [sizes[k] for k in ks], color=BLUE, width=0.72,
           edgecolor="white", linewidth=0.4)
    a1.set_xlabel("atoms in class")
    a1.set_ylabel("classes")
    a1.set_xticks(ks)
    a1.grid(axis="y", linestyle=":")
    tag(a1, "B")

    # C: class count against atom count
    a2 = ax[2]
    xs = np.array([len(bym[m]) for m in names])
    ys = np.array([bym[m][0]["n_classes"] for m in names])
    a2.plot([0, xs.max() + 1], [0, xs.max() + 1], color=GREY,
            linestyle="--", linewidth=1.0)
    a2.scatter(xs, ys, s=52, c=ys / xs, cmap=SEQ, vmin=0, vmax=1,
               edgecolors="black", linewidths=0.4)
    a2.set_xlabel("atoms")
    a2.set_ylabel("stable classes")
    a2.grid(linestyle=":")
    tag(a2, "C")

    # D: depth against degree, coloured by sigma
    a3 = ax[3]
    dep = np.array([a["depth"] for a in atoms], float)
    deg = np.array([a["heavy_degree"] for a in atoms], float)
    sig = np.array([a["sigma"] for a in atoms])
    rng = np.random.default_rng(0)
    sc = a3.scatter(deg + rng.uniform(-0.16, 0.16, len(deg)),
                    dep + rng.uniform(-0.16, 0.16, len(dep)),
                    c=sig, cmap=SEQ, s=24, alpha=0.85,
                    edgecolors="none")
    a3.set_xlabel("heavy degree")
    a3.set_ylabel("burial depth")
    a3.set_yticks(sorted(set(dep.astype(int).tolist())))
    cb = fig.colorbar(sc, ax=a3, pad=0.02, shrink=0.85, aspect=16)
    cb.set_label(r"$\sigma$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    a3.grid(linestyle=":")
    tag(a3, "D")

    return save(fig, os.path.join(OUT, "panel_05_classes.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04, panel_05):
        print("wrote", os.path.basename(fn()))
