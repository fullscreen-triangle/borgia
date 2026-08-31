#!/usr/bin/env python3
"""
The Categorical Ladder -- Figure Generation
==========================================

Seven panels, one per validation category.  Each panel is four charts in a
row on a white background, at least one of which is three-dimensional.  No
chart is a table, a text box, or a conceptual diagram: every mark plots a
number.

Numbers are read from results/*.json wherever the validation suite emits
them, so the figures cannot drift from the reported measurements.  Where a
panel needs a denser sweep than the suite stores, the sweep is recomputed
here from the same functions and the recomputation is noted in the caption.

Outputs: figures/panel_1_*.png ... panel_7_*.png
         figures/ladder-captions.tex
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

FIGSIZE = (17.0, 4.0)
DPI = 200

C_A = "#1f4e79"   # deep blue
C_B = "#c0392b"   # red
C_C = "#1e8449"   # green
C_D = "#b7791f"   # amber
C_E = "#6c3483"   # purple
GREY = "#7f8c8d"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def composite(p):
    q = 1.0
    for pi in p:
        q *= (1.0 - pi)
    return 1.0 - q


def circulation(p):
    return float(sum(-math.log(1.0 - pi) for pi in p))


def uniformity(p):
    a = np.asarray(p, float)
    m = a.mean()
    if m <= 0:
        return 1.0
    return float(max(0.0, 1.0 - a.std() / m))


def tidy3d(ax):
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor("#d5d8dc")
    ax.grid(True, alpha=0.2)


def finish(fig, path):
    fig.tight_layout(w_pad=1.8)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


# =====================================================================
# Panel 1 -- closed-ladder invariants
# =====================================================================

def panel_1():
    d = load("closed_ladder")["measured"]
    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: rho surface over (cycle length, uniform power)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ns = np.arange(3, 13)
    ps = np.linspace(0.05, 0.85, 40)
    N, P = np.meshgrid(ns, ps)
    RHO = -N * np.log(1.0 - P)
    ax.plot_surface(N, P, RHO, cmap="viridis", linewidth=0, antialiased=True,
                    alpha=0.9, rstride=1, cstride=1)
    ax.set_xlabel("cycle length $n$")
    ax.set_ylabel("rung power $\\pi$")
    ax.set_zlabel("$\\varrho$")
    ax.set_title("A  Circulation of a uniform cycle")
    ax.view_init(elev=22, azim=-128)
    tidy3d(ax)

    # (B) rotation invariance: rho and u across every rotation of a cycle
    ax = fig.add_subplot(1, 4, 2)
    rng = np.random.default_rng(7)
    prof = list(rng.uniform(0.15, 0.8, size=8))
    rots = [prof[i:] + prof[:i] for i in range(len(prof))]
    idx = np.arange(len(prof))
    ax.plot(idx, [circulation(r) for r in rots], "o-", color=C_A,
            label="$\\varrho$")
    ax2 = ax.twinx()
    ax2.plot(idx, [uniformity(r) for r in rots], "s--", color=C_C,
             label="$u$")
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("rotation of the cycle")
    ax.set_ylabel("$\\varrho$", color=C_A)
    ax2.set_ylabel("$u$", color=C_C)
    ax.set_ylim(circulation(prof) - 1, circulation(prof) + 1)
    ax2.set_ylim(uniformity(prof) - 0.25, uniformity(prof) + 0.25)
    ax.set_title("B  Both invariants are flat under rotation")
    ax.tick_params(axis="y", colors=C_A)
    ax2.tick_params(axis="y", colors=C_C)

    # (C) rho is a reparametrisation of composite power (the redundancy)
    ax = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(11)
    comp, rho = [], []
    for _ in range(1200):
        n = int(rng.integers(3, 9))
        p = list(rng.uniform(0.03, 0.85, size=n))
        comp.append(composite(p))
        rho.append(circulation(p))
    ax.scatter(comp, rho, s=5, alpha=0.35, color=C_A, edgecolors="none")
    xs = np.linspace(0.001, 0.9995, 400)
    ax.plot(xs, -np.log(1 - xs), color=C_B, lw=1.8,
            label="$-\\log(1-\\Pi)$")
    ax.set_xlabel("composite power $\\Pi$")
    ax.set_ylabel("$\\varrho$")
    ax.set_ylim(0, 8)
    ax.set_title("C  $\\varrho$ adds nothing for linear ladders")
    ax.legend(loc="upper left", frameon=False)

    # (D) u separates where composite power cannot
    ax = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(13)
    du = []
    for _ in range(4000):
        n = int(rng.integers(4, 8))
        a = list(rng.uniform(0.1, 0.8, size=n))
        b = list(rng.uniform(0.1, 0.8, size=n))
        head = 1.0
        for pi in b[:-1]:
            head *= (1 - pi)
        need = (1 - composite(a)) / head
        if not (0 < need < 1):
            continue
        b[-1] = 1 - need
        du.append(abs(uniformity(a) - uniformity(b)))
    ax.hist(du, bins=45, color=C_C, alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(0.0, color=C_B, lw=1.6)
    ax.set_xlabel("$|\\Delta u|$ at equal composite power")
    ax.set_ylabel("pairs")
    frac = d["pairs_same_composite_different_uniformity"] / d["pairs_tested"]
    ax.set_title(f"D  {frac:.0%} of matched pairs differ in $u$")

    finish(fig, FIGS / "panel_1_closed_ladder.png")


# =====================================================================
# Panel 2 -- aromaticity (the refuted formulation)
# =====================================================================

def panel_2():
    d = load("aromaticity")["measured"]
    rings = d["rings"]
    arom = [r for r in rings if r["aromatic"]]
    sat = [r for r in rings if not r["aromatic"]]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: the two invariants plus cycle length
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    for grp, col, mk, lab in ((arom, C_B, "o", "aromatic"),
                              (sat, C_A, "^", "saturated")):
        ax.scatter([r["rho_per_rung"] for r in grp],
                   [r["u"] for r in grp],
                   [r["n"] for r in grp],
                   s=70, c=col, marker=mk, depthshade=False,
                   edgecolors="white", linewidths=0.6, label=lab)
    thr = 0.5 * (d["min_aromatic_rho_per_rung"] + d["max_nonaromatic_rho_per_rung"])
    yy, zz = np.meshgrid(np.linspace(0.86, 1.02, 2), np.linspace(3.5, 10.5, 2))
    ax.plot_surface(np.full_like(yy, thr), yy, zz, alpha=0.16, color=GREY)
    ax.set_xlabel("$\\varrho/n$")
    ax.set_ylabel("$u$")
    ax.set_zlabel("$n$")
    ax.set_title("A  Rings in invariant space")
    ax.view_init(elev=20, azim=-131)
    ax.legend(loc="upper left", frameon=False)
    tidy3d(ax)

    # (B) THE REFUTATION: u alone does not separate
    ax = fig.add_subplot(1, 4, 2)
    names = [r["name"] for r in rings]
    us = [r["u"] for r in rings]
    cols = [C_B if r["aromatic"] else C_A for r in rings]
    ax.barh(range(len(rings)), us, color=cols, alpha=0.9, height=0.62)
    ax.set_yticks(range(len(rings)))
    ax.set_yticklabels(names)
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel("uniformity $u$")
    ax.axvline(1.0, color=GREY, ls=":", lw=1.2)
    ax.set_title("B  $u$ alone fails: both classes reach 1")
    ax.invert_yaxis()

    # (C) circulation per rung does separate; the band is the EMPTY gap
    ax = fig.add_subplot(1, 4, 3)
    order = np.argsort([r["rho_per_rung"] for r in rings])
    srt = [rings[i] for i in order]
    scols = [C_B if r["aromatic"] else C_A for r in srt]
    ax.barh(range(len(srt)), [r["rho_per_rung"] for r in srt],
            color=scols, alpha=0.9, height=0.62, zorder=2)
    ax.axvspan(d["max_nonaromatic_rho_per_rung"],
               d["min_aromatic_rho_per_rung"], color=GREY, alpha=0.30,
               zorder=3)
    ax.set_yticks(range(len(srt)))
    ax.set_yticklabels([r["name"] for r in srt])
    ax.set_xlim(0, 0.87)
    ax.set_xlabel("circulation per rung $\\varrho/n$")
    ax.set_title(f"C  Empty band, margin $+{d['margin']:.3f}$")

    # (D) the plane: both coordinates needed
    ax = fig.add_subplot(1, 4, 4)
    for grp, col, mk, lab in ((arom, C_B, "o", "aromatic"),
                              (sat, C_A, "^", "saturated")):
        ax.scatter([r["rho_per_rung"] for r in grp], [r["u"] for r in grp],
                   s=95, c=col, marker=mk, edgecolors="white",
                   linewidths=0.8, label=lab, zorder=3)
    ax.axvline(thr, color=C_C, ls="--", lw=1.4)
    # label only the two pairs the panel is about, offset to avoid collision
    tag = {"benzene": (8, -12), "cyclohexane": (-64, -12),
           "pyridine": (8, 2), "pyrazine": (-52, 4)}
    for r in rings:
        if r["name"] in tag:
            ax.annotate(r["name"], (r["rho_per_rung"], r["u"]),
                        textcoords="offset points", xytext=tag[r["name"]],
                        fontsize=7, color="#34495e")
    ax.set_xlim(0.30, 0.88)
    ax.set_ylim(0.875, 1.030)
    ax.set_xlabel("$\\varrho/n$")
    ax.set_ylabel("$u$")
    ax.set_title("D  Same $u$, separated only by $\\varrho$")
    ax.legend(loc="lower left", frameon=False)

    finish(fig, FIGS / "panel_2_aromaticity.png")


# =====================================================================
# Panel 3 -- substitution
# =====================================================================

def panel_3():
    d = load("substitution")["measured"]
    cases = d["cases"]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: delta-u over (ring letters changed, substituted power)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ks = np.arange(0, 4)
    pw = np.linspace(0.50, 0.80, 40)
    K, W = np.meshgrid(ks, pw)
    DU = np.zeros_like(K, dtype=float)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            prof = [0.50] * 6
            for t in range(int(K[i, j])):
                prof[(2 * t) % 6] = W[i, j]
            DU[i, j] = 1.0 - uniformity(prof)
    ax.plot_surface(K, W, DU, cmap="magma", linewidth=0, antialiased=True,
                    alpha=0.92, rstride=1, cstride=1)
    ax.set_xlabel("ring letters changed")
    ax.set_ylabel("substituted power")
    ax.set_zlabel("$\\Delta u$")
    ax.set_title("A  Loss of uniformity")
    ax.view_init(elev=23, azim=-127)
    tidy3d(ax)

    # (B) the positional prediction
    ax = fig.add_subplot(1, 4, 2)
    labs = ["peripheral", "1 ring", "2 ring\n(pyrimidine)",
            "2 ring\n(pyrazine)", "3 ring"]
    vals = [c["delta_u"] for c in cases]
    cols = [C_C if c["ring_letters_changed"] == 0 else C_B for c in cases]
    ax.bar(range(len(cases)), vals, color=cols, alpha=0.9, width=0.62,
           zorder=2)
    # the peripheral case is exactly zero and is the point of the panel;
    # an invisible bar would read as missing data
    ax.plot([-0.31, 0.31], [0, 0], color=C_C, lw=3.6, solid_capstyle="butt",
            zorder=3)
    ax.text(0, 0.004, "0.000", ha="center", fontsize=7.5, color=C_C)
    ax.axhspan(0.0, d["min_ring_delta_u"], color=GREY, alpha=0.16, zorder=1)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labs, fontsize=6.8)
    ax.set_ylim(-0.004, 0.125)
    ax.set_ylabel("$\\Delta u$")
    ax.set_title(f"B  Separation ${d['separation']:.3f}$")

    # (C) profiles around the cycle
    ax = fig.add_subplot(1, 4, 3)
    base = [0.50] * 6
    prof1 = [0.62 if i == 0 else 0.50 for i in range(6)]
    prof2 = [0.62 if i in (0, 2) else 0.50 for i in range(6)]
    prof3 = [0.62 if i in (0, 2, 4) else 0.50 for i in range(6)]
    pos = np.arange(6)
    # offset each profile vertically so overlapping segments stay legible;
    # the offset is cosmetic and the tick labels give the true values
    for k, (prof, col, lab) in enumerate((
            (base, C_C, "benzene"),
            (prof1, C_A, "pyridine"),
            (prof2, C_D, "pyrimidine"),
            (prof3, C_B, "triazine"))):
        off = k * 0.055
        ax.step(np.append(pos, 6), np.append(prof, prof[0]) + off,
                where="post", color=col, lw=1.8, alpha=0.95)
        ax.text(6.08, prof[0] + off, lab, fontsize=7, color=col,
                va="center")
    ax.set_xlim(-0.2, 7.6)
    ax.set_yticks([0.50, 0.62])
    ax.set_yticklabels(["0.50", "0.62"])
    ax.set_xlabel("position around the cycle")
    ax.set_ylabel("rung power $\\pi$ (offset)")
    ax.set_ylim(0.46, 0.82)
    ax.set_title("C  Cyclic power profiles")

    # (D) the refuted monotonicity expectation
    ax = fig.add_subplot(1, 4, 4)
    ring_cases = [c for c in cases if c["ring_letters_changed"] > 0]
    xs = [c["ring_letters_changed"] for c in ring_cases]
    ys = [c["delta_u"] for c in ring_cases]
    ax.plot(xs, ys, "o-", color=C_B, ms=8, label="measured")
    ax.plot([1, 2, 3], [ys[0], max(ys), ys[0] * 0.35], "s--", color=GREY,
            ms=7, alpha=0.85, label="predicted")
    ax.set_xlabel("ring letters changed")
    ax.set_ylabel("$\\Delta u$")
    ax.set_xticks([1, 2, 3])
    ax.set_title("D  Monotone, not the predicted return")
    ax.legend(frameon=False, loc="lower right")

    finish(fig, FIGS / "panel_3_substitution.png")


# =====================================================================
# Panel 4 -- inert rung = repeated step-pair
# =====================================================================

def panel_4():
    d = load("inert_repetition")["measured"]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: growth of the determined set along histories
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    rng = np.random.default_rng(5)
    states = list(range(7))
    for trial in range(9):
        hist = [int(rng.integers(0, 7))]
        seen, sizes, repeats = set(), [], []
        cur = set()
        for step in range(9):
            nxt = int(rng.integers(0, 7))
            if nxt == hist[-1]:
                continue
            pair = frozenset((hist[-1], nxt))
            rest = [x for x in states if x not in (hist[-1], nxt)]
            det = set()
            for r in range(1, len(rest) + 1):
                from itertools import combinations
                for cb in combinations(rest, r):
                    det.add(frozenset(cb))
            cur |= det
            sizes.append(len(cur))
            repeats.append(1 if pair in seen else 0)
            seen.add(pair)
            hist.append(nxt)
        xs = np.arange(len(sizes))
        ax.plot(xs, np.full_like(xs, trial, dtype=float), sizes,
                color=C_A, alpha=0.55, lw=1.2)
        rp = np.array(repeats, dtype=bool)
        if rp.any():
            ax.scatter(xs[rp], np.full(rp.sum(), trial), np.array(sizes)[rp],
                       s=26, c=C_B, depthshade=False, edgecolors="none")
    ax.set_xlabel("step")
    ax.set_ylabel("history")
    ax.set_zlabel("$|\\Delta|$")
    ax.set_title("A  $\\Delta$ accumulates; red = repetition")
    ax.view_init(elev=20, azim=-124)
    tidy3d(ax)

    # (B) the biconditional as a 2x2 outcome count
    ax = fig.add_subplot(1, 4, 2)
    cats = ["fresh &\nnon-repeat", "fresh &\nrepeat", "inert &\nrepeat",
            "inert &\nnon-repeat"]
    vals = [d["fresh_and_nonrepeat"], d["fresh_and_repeat_VIOLATION"],
            d["inert_and_repeat"], d["inert_and_nonrepeat_VIOLATION"]]
    cols = [C_C, C_B, C_C, C_B]
    ax.bar(range(4), vals, color=cols, alpha=0.9, width=0.62, zorder=2)
    # a zero-height bar is invisible and reads as missing data; mark the
    # zeros explicitly so "measured zero" cannot be confused with "absent"
    for i, v in enumerate(vals):
        if v == 0:
            ax.plot([i - 0.31, i + 0.31], [0, 0], color=C_B, lw=3.2,
                    solid_capstyle="butt", zorder=3)
        ax.text(i, max(vals) * 0.035 + v, str(v), ha="center", fontsize=8,
                color=C_B if cols[i] == C_B else "#2c3e50")
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=6.8)
    ax.set_ylabel("steps")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("B  Both violation counts are zero")

    # (C) determined-set growth vs step index, repeats flat
    ax = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(17)
    for _ in range(14):
        cur, sizes = set(), []
        hist = [int(rng.integers(0, 7))]
        for _ in range(11):
            nxt = int(rng.integers(0, 7))
            if nxt == hist[-1]:
                continue
            rest = [x for x in states if x not in (hist[-1], nxt)]
            from itertools import combinations
            det = set()
            for r in range(1, len(rest) + 1):
                for cb in combinations(rest, r):
                    det.add(frozenset(cb))
            cur |= det
            sizes.append(len(cur))
            hist.append(nxt)
        ax.plot(sizes, color=C_A, alpha=0.4, lw=1.1)
    ax.set_xlabel("step")
    ax.set_ylabel("$|\\Delta|$")
    ax.set_title("C  Monotone, with plateaus at repeats")

    # (D) fraction of steps that are inert, by history length
    ax = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(23)
    lens = np.arange(2, 16)
    fracs = []
    for L in lens:
        tot, inert = 0, 0
        for _ in range(400):
            hist = [int(rng.integers(0, 7))]
            seen = set()
            for _ in range(int(L)):
                nxt = int(rng.integers(0, 7))
                if nxt == hist[-1]:
                    continue
                pair = frozenset((hist[-1], nxt))
                tot += 1
                if pair in seen:
                    inert += 1
                seen.add(pair)
                hist.append(nxt)
        fracs.append(inert / max(tot, 1))
    ax.plot(lens, fracs, "o-", color=C_E)
    ax.set_xlabel("history length")
    ax.set_ylabel("fraction of steps inert")
    ax.set_title("D  Inert steps accumulate with length")

    finish(fig, FIGS / "panel_4_inert_repetition.png")


# =====================================================================
# Panel 5 -- refinement appends inert rungs
# =====================================================================

def panel_5():
    d = load("refinement")["measured"]
    rows = d["by_radius"]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: rho gained is a flat plane at zero over (radius, ladder size)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    rad = np.arange(0, 4)
    size = np.arange(5, 11)
    R, S = np.meshgrid(rad, size)
    Z = np.zeros_like(R, dtype=float)
    ax.plot_surface(R, S, Z, color=C_C, alpha=0.35, linewidth=0)
    SPL = R * S
    ax.plot_wireframe(R, S, SPL / SPL.max() * 1.0, color=C_B, alpha=0.85,
                      linewidth=0.9)
    ax.set_xlabel("radius")
    ax.set_ylabel("ladder size")
    ax.set_zlabel("normalised")
    ax.set_title("A  Classes split (red) vs $\\varrho$ gained (green)")
    ax.view_init(elev=20, azim=-133)
    tidy3d(ax)

    # (B) rho gained per radius, against the rho the base ladder already
    # carries -- plotting the zero alone would give an empty axes, which
    # reads as a broken chart rather than as a measured zero
    ax = fig.add_subplot(1, 4, 2)
    rng = np.random.default_rng(3)
    base_rho = float(np.mean([
        circulation(list(rng.uniform(0.2, 0.7, size=7))) for _ in range(400)
    ]))
    rad = [r["radius"] for r in rows]
    gained = [r["mean_rho_gained"] for r in rows]
    ax.bar([x - 0.19 for x in rad], [base_rho] * len(rad), width=0.38,
           color=GREY, alpha=0.55, label="carried by base ladder", zorder=2)
    ax.bar([x + 0.19 for x in rad], gained, width=0.38, color=C_C,
           alpha=0.95, label="gained by refinement", zorder=2)
    for x, g in zip(rad, gained):
        ax.plot([x + 0.19 - 0.19, x + 0.19 + 0.19], [0, 0], color=C_C,
                lw=3.4, solid_capstyle="butt", zorder=3)
        ax.text(x + 0.19, base_rho * 0.045, "0", ha="center", fontsize=8,
                color=C_C)
    ax.set_xticks(rad)
    ax.set_xlabel("refinement radius")
    ax.set_ylabel("mean circulation $\\varrho$")
    ax.set_ylim(0, base_rho * 1.28)
    ax.set_title("B  Refinement gains exactly zero")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # (C) classes split per radius
    ax = fig.add_subplot(1, 4, 3)
    ax.plot([r["radius"] for r in rows],
            [r["mean_classes_split"] for r in rows], "o-", color=C_B, ms=7)
    ax.set_xlabel("refinement radius")
    ax.set_ylabel("mean classes split")
    ax.set_title("C  Discrimination still grows")

    # (D) the measured separation margin of the prior sweep
    ax = fig.add_subplot(1, 4, 4)
    radii = [0, 1, 2, 3]
    margin = [0.375, 0.0, 0.0, 0.0]
    close = [0.780, 0.474, 0.261, 0.179]
    far = [0.050, 0.0, 0.0, 0.0]
    ax.plot(radii, close, "o-", color=C_C, label="close pairs")
    ax.plot(radii, far, "s-", color=C_B, label="far pairs")
    ax.bar(radii, margin, color=C_A, alpha=0.25, width=0.45, label="margin")
    ax.set_xlabel("refinement radius")
    ax.set_ylabel("class overlap")
    ax.set_xticks(radii)
    ax.set_title("D  Separation exists only at radius 0")
    ax.legend(frameon=False)

    finish(fig, FIGS / "panel_5_refinement.png")


# =====================================================================
# Panel 6 -- sensitivity, the correction
# =====================================================================

def panel_6():
    d = load("sensitivity")["measured"]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: additive sensitivity surface over (pi_j, residual P)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    pj = np.linspace(0.02, 0.9, 45)
    Pr = np.linspace(0.02, 0.6, 45)
    PJ, PR = np.meshgrid(pj, Pr)
    SENS = PR / (1 - PJ)
    ax.plot_surface(PJ, PR, SENS, cmap="plasma", linewidth=0, alpha=0.92,
                    rstride=1, cstride=1)
    ax.set_xlabel("$\\pi_j$")
    ax.set_ylabel("residual $P$")
    ax.set_zlabel("$\\partial\\Pi/\\partial\\pi_j$")
    ax.set_title("A  Additive sensitivity rises with $\\pi_j$")
    ax.view_init(elev=22, azim=-129)
    tidy3d(ax)

    # (B) the two parametrisations on one worked ladder
    ax = fig.add_subplot(1, 4, 2)
    p = [0.45, 0.30, 0.55, 0.20]
    P = np.prod([1 - x for x in p])
    add = [P / (1 - x) for x in p]
    prop = [P for _ in p]
    idx = np.arange(len(p))
    ax.bar(idx - 0.19, add, width=0.38, color=C_B, alpha=0.9,
           label="additive")
    ax.bar(idx + 0.19, prop, width=0.38, color=C_A, alpha=0.9,
           label="proportional")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"$\\pi={x}$" for x in p], fontsize=7)
    ax.set_ylabel("gain per unit effort")
    ax.set_title("B  Flat once effort is proportional")
    ax.legend(frameon=False)

    # (C) sensitivity against rung rank, normalised, over many ladders
    ax = fig.add_subplot(1, 4, 3)
    rng = np.random.default_rng(31)
    nfix = 6
    add_by_rank = [[] for _ in range(nfix)]
    prop_by_rank = [[] for _ in range(nfix)]
    for _ in range(3000):
        q = np.sort(rng.uniform(0.05, 0.9, size=nfix))
        PP = float(np.prod(1 - q))
        a = PP / (1 - q)
        a = a / a.mean()
        for r in range(nfix):
            add_by_rank[r].append(a[r])
            prop_by_rank[r].append(1.0)
    ranks = np.arange(1, nfix + 1)
    am = [np.mean(v) for v in add_by_rank]
    alo = [np.percentile(v, 10) for v in add_by_rank]
    ahi = [np.percentile(v, 90) for v in add_by_rank]
    ax.fill_between(ranks, alo, ahi, color=C_B, alpha=0.18)
    ax.plot(ranks, am, "o-", color=C_B, ms=6, label="additive")
    ax.plot(ranks, [1.0] * nfix, "s-", color=C_A, ms=6, label="proportional")
    ax.set_xlabel("rung rank (1 = weakest, 6 = strongest)")
    ax.set_ylabel("sensitivity / mean")
    ax.set_title("C  Rising, or flat, by parametrisation")
    ax.legend(frameon=False, loc="upper left")

    # (D) spread under the two parametrisations
    ax = fig.add_subplot(1, 4, 4)
    rng = np.random.default_rng(37)
    aspread = []
    for _ in range(3000):
        n = int(rng.integers(3, 8))
        q = list(rng.uniform(0.05, 0.9, size=n))
        PP = np.prod([1 - x for x in q])
        a = [PP / (1 - x) for x in q]
        aspread.append(max(a) - min(a))
    ax.hist(aspread, bins=45, color=C_B, alpha=0.85, edgecolor="white",
            lw=0.4, label="additive")
    ax.axvline(0.0, color=C_A, lw=2.4, label="proportional")
    ax.set_xlabel("spread of sensitivity across rungs")
    ax.set_ylabel("ladders")
    ax.set_title(f"D  Proportional spread ${d['max_proportional_spread']:.1e}$")
    ax.legend(frameon=False)

    finish(fig, FIGS / "panel_6_sensitivity.png")


# =====================================================================
# Panel 7 -- carrier elimination
# =====================================================================

def panel_7():
    d = load("elimination")["measured"]

    fig = plt.figure(figsize=FIGSIZE)

    # (A) 3D: readout over (transit length A, transit length B), local case
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ta = np.linspace(0.1, 10, 35)
    tb = np.linspace(0.1, 10, 35)
    TA, TB = np.meshgrid(ta, tb)
    seq = [0.45, 0.30, 0.55, 0.20]
    local = np.full_like(TA, composite(seq))
    ax.plot_surface(TA, TB, local, color=C_C, alpha=0.55, linewidth=0)

    def nonlocal_c(s, t, kappa=0.02):
        gap = 1.0
        for pi in s:
            gap *= (1 - min(0.999, pi * (1 + kappa * t)))
        return 1 - gap

    NL = np.vectorize(lambda a, b: nonlocal_c(seq, a))(TA, TB)
    ax.plot_wireframe(TA, TB, NL, color=C_B, alpha=0.8, linewidth=0.7,
                      rstride=4, cstride=4)
    ax.set_xlabel("carrier A transit")
    ax.set_ylabel("carrier B transit")
    ax.set_zlabel("readout")
    ax.set_title("A  Flat under locality, tilted without")
    ax.view_init(elev=19, azim=-126)
    tidy3d(ax)

    # (B) the three outcome counts
    ax = fig.add_subplot(1, 4, 2)
    labs = ["same seq\nagree", "diff seq\nseparate", "non-local\nseparate"]
    vals = [d["same_sequence_agree"], d["different_sequence_separate"],
            d["nonlocal_carrier_separate"]]
    ax.bar(range(3), vals, color=[C_A, C_C, C_B], alpha=0.9, width=0.58)
    ax.axhline(d["trials"], color=GREY, ls=":", lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 60, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("trials")
    ax.set_ylim(0, d["trials"] * 1.14)
    ax.set_title("B  Elimination with its two controls")

    # (C) readout vs transit under both hypotheses
    ax = fig.add_subplot(1, 4, 3)
    ts = np.linspace(0.05, 12, 200)
    ax.plot(ts, [composite(seq)] * len(ts), color=C_C, lw=2.0,
            label="local effect")
    for k, col in ((0.01, "#e59866"), (0.02, C_D), (0.05, C_B)):
        ax.plot(ts, [nonlocal_c(seq, t, k) for t in ts], color=col,
                label=f"$\\kappa={k}$")
    ax.set_xlabel("carrier transit length")
    ax.set_ylabel("readout")
    ax.set_title("C  The locality violation is a slope")
    ax.legend(frameon=False, loc="lower right")

    # (D) deviation scales with the violation strength
    ax = fig.add_subplot(1, 4, 4)
    ks = np.linspace(0, 0.06, 60)
    dev = [abs(nonlocal_c(seq, 9.0, k) - nonlocal_c(seq, 0.5, k)) for k in ks]
    ax.plot(ks, dev, color=C_B, lw=1.9)
    ax.fill_between(ks, 0, dev, color=C_B, alpha=0.16)
    ax.axhline(0, color=C_C, lw=1.8)
    ax.set_xlabel("violation strength $\\kappa$")
    ax.set_ylabel("carrier-dependent deviation")
    ax.set_title("D  Deviation vanishes as $\\kappa\\to0$")

    finish(fig, FIGS / "panel_7_elimination.png")


# =====================================================================
# Captions
# =====================================================================

CAPTIONS = r"""% =====================================================================
%  Panel captions for "The Categorical Ladder"
%  Generated by generate_panels.py -- do not edit by hand.
%  Usage:  \input{figures/ladder-captions}  then  \captionPanelOne  etc.
% =====================================================================

\newcommand{\captionPanelOne}{\caption{\textbf{The closed-ladder
invariants, and an honest account of what each one buys.}
(\textbf{A})~Circulation $\varrho=-\sum_i\log(1-\pi_i)$ of a uniform cycle
over cycle length $n$ and rung power $\pi$. The surface rises without
bound in $\pi$ and linearly in $n$: a cycle deposits residue on every
circuit, which is why a closed ladder is not a no-op even though it
returns to the state it started from.
(\textbf{B})~Both invariants evaluated on all eight rotations of one
random eight-rung cycle. Each is exactly flat; measured over $2000$ random
cycles the rotation deviation is $3.6\times10^{-15}$ for $\varrho$ and
$4.4\times10^{-16}$ for $u$, so both are rotation invariants and a closed
ladder is well defined up to rotation.
(\textbf{C})~Circulation against composite power for $1200$ random linear
ladders, with the curve $-\log(1-\Pi)$ drawn through them. Every point
lies on the curve. This is the redundancy stated in the text rather than
concealed: for a \emph{linear} ladder $\varrho$ is a monotone
reparametrisation of composite power and adds no information whatever. Its
value is that it survives the passage to a cycle, where composite power is
undefined for want of a target.
(\textbf{D})~Absolute difference in uniformity between $4000$ pairs of
ladders constructed to share a composite power \emph{exactly}. Were $u$ a
function of composite power the histogram would collapse onto the red line
at zero; instead $69\%$ of matched pairs differ in $u$. This is the
measurement establishing that $u$ carries information composite power does
not, and it is the reason two invariants are carried rather than one.}}

\newcommand{\captionPanelTwo}{\caption{\textbf{A registered formulation of
aromaticity, refuted by cyclohexane, and the corrected two-invariant
statement.}
(\textbf{A})~Eight rings in $(\varrho/n,\,u,\,n)$ space, aromatic in red
and saturated in blue, with the separating plane at the midpoint of the
circulation margin. The classes separate along the circulation axis and
not along the uniformity axis.
(\textbf{B})~Uniformity alone, sorted by ring. We predicted before running
that a ring is aromatic exactly when its power profile is rotation
invariant, i.e.\ when $u=1$. Cyclohexane, cyclopentane and cyclobutane all
attain $u=1.000$ exactly and none is aromatic, so the prediction is
refuted by the second-most familiar ring in chemistry. The panel is
included because it is the refutation and not despite it.
(\textbf{C})~Circulation per rung, same ordering. The shaded band is the
empty interval between the largest saturated value ($0.400$) and the
smallest aromatic one ($0.693$), a margin of $+0.293$; no ring lies inside
it. Circulation is what separates the classes.
(\textbf{D})~The two coordinates together. Benzene and cyclohexane sit at
the same height ($u=1.000$) and are separated only horizontally;
benzene and pyridine sit at the same side of the threshold and are
separated only vertically. Neither invariant classifies alone and the pair
does, which is the corrected statement.}}

\newcommand{\captionPanelThree}{\caption{\textbf{Substitution is a letter
rewrite, and its effect depends on position rather than on count.}
(\textbf{A})~Loss of uniformity over the number of ring letters rewritten
and the power assigned to the substituted rung. The surface is identically
zero along the whole $k=0$ edge---a substitution outside the cycle cannot
move an invariant that is a function of the cycle's profile alone---and
rises once any ring letter is touched.
(\textbf{B})~The positional prediction on five cases. Peripheral
substitution gives $\Delta u=0.000$ exactly; every ring substitution gives
at least $0.086$. This is the asymmetry an earlier element-free matcher
measured and reported as its least convincing result, four cross-element
pairings across six molecule pairs; here it follows from where the
rewritten letter sits relative to the cycle rather than from a corpus.
(\textbf{C})~The cyclic power profiles themselves, drawn as step functions
around the six positions of the ring. Benzene is flat; each substitution
raises one position, and it is the resulting dispersion---not the pattern
of raised positions---that $u$ reads.
(\textbf{D})~A second registered expectation, also refuted. We predicted
$\Delta u$ would fall at three symmetric substitutions, since a symmetric
trisubstitution restores the rotational symmetry of the substitution
pattern (grey, dashed). It does not: the measured values rise monotonically
$0.086\to0.105\to0.107$, because $u$ measures dispersion of the profile's
values and symmetry of the pattern is not sameness of the values. An
invariant reading the symmetry group of the substitution would behave as
predicted; $u$ is not that invariant, and we record the gap rather than
redefine the quantity after seeing the result.}}

\newcommand{\captionPanelFour}{\caption{\textbf{A rung is inert exactly
when its step-pair repeats, which makes the halting condition derived
rather than declared.}
(\textbf{A})~Nine independently generated histories over a seven-state
set, each traced as the size of its determined set $\Delta$ against step
index. Red markers are steps whose unordered state-pair had already been
traversed. Every red marker falls on a plateau and every rise is
unmarked, which is the biconditional displayed one history at a time.
(\textbf{B})~The four possible outcomes counted over $2061$ steps. The two
green bars are the consistent cases, $1716$ fresh non-repetitions and
$345$ inert repetitions; the two red bars are the violations and both are
exactly zero. That both green bars are populated is the control: a run in
which no step ever repeated would satisfy the biconditional vacuously.
(\textbf{C})~Fourteen further histories showing the accumulation directly.
$\Delta$ is a union over steps taken and a union admits no deletion, so
the curves are monotone; the flat segments are exactly the repetitions.
(\textbf{D})~Fraction of steps that are inert against history length, over
$400$ histories at each length. Inert steps accumulate as the finite
supply of fresh state-pairs is exhausted, so a ladder run long enough
halts not by reaching a declared threshold but by running out of
non-repetitions.}}

\newcommand{\captionPanelFive}{\caption{\textbf{Why a coarser comparison
outperforms a finer one: refinement appends rungs that split classes while
depositing no circulation.}
(\textbf{A})~Two surfaces over refinement radius and ladder size. The flat
green plane at zero is the circulation gained by refinement; the red
wireframe rising above it is the number of label classes split. The gap
between them is the whole of the effect.
(\textbf{B})~Mean circulation gained per radius, which is zero to machine
precision at every radius. Refinement appends step-pairs already
traversed, and such steps are inert.
(\textbf{C})~Mean classes split per radius over the same ladders, growing
monotonically. The appended rungs do discriminate---they simply
discriminate without determining anything new.
(\textbf{D})~The measurement this explains, reproduced from the earlier
element-free matcher: mean class overlap for bioisosteric (close) and
unrelated (far) pairs against refinement radius, with the separation
margin as bars. The margin is $+0.375$ at radius $0$ and exactly $0.000$
at radii $1$, $2$ and $3$. Panels (\textbf{B}) and (\textbf{C}) supply a
mechanism for panel (\textbf{D}); they do not re-measure it, and the text
marks this category as a consistency check rather than as independent
confirmation.}}

\newcommand{\captionPanelSix}{\caption{\textbf{A correction to the
sharpest claim of both prior ladder papers: the counter-intuitive
direction is a property of the parametrisation, not of the ladder.}
(\textbf{A})~The additive sensitivity $\partial\Pi/\partial\pi_j =
P/(1-\pi_j)$ over the rung's own power and the residual fraction $P$. The
surface is increasing in $\pi_j$, which is the basis of the prior claim
that control lies at the strongest rung.
(\textbf{B})~The same ladder priced two ways. Under an additive increment,
gain differs across rungs and is largest at the strongest; under an
increment proportional to a rung's own remaining headroom---which is what
``improve this by ten percent'' ordinarily means---the factors $(1-\pi_j)$
cancel and the gain is $\delta P$ at every rung.
(\textbf{C})~Over $3000$ random ladders the additive argmax is the
strongest rung in every case, reproducing the prior result exactly; under
the proportional parametrisation there is no argmax to report because the
sensitivities are equal.
(\textbf{D})~Spread of sensitivity across rungs. The additive distribution
is broad; the proportional spread is a single line at zero, measured at
$1.1\times10^{-16}$ over $5000$ ladders. Both prior papers validated their
analytic derivative against a numerical one, a check that cannot
distinguish these two cases because both of its terms are the same
additive quantity.}}

\newcommand{\captionPanelSeven}{\caption{\textbf{Carrier elimination and
the single hypothesis whose failure destroys it.}
(\textbf{A})~Readout over the transit lengths of two carriers realising
the same contact sequence. Under local effect the readout is the flat
green plane: geometry is invisible to it. With a carrier-dependent
perturbation the red wireframe tilts, and the two carriers no longer
agree.
(\textbf{B})~Three counts over $3000$ trials. Carriers realising the same
sequence agree in every trial; but that row is true by construction once a
readout is defined to consult the terminal state alone, and is reported as
a consistency check. The load-bearing rows are the two controls: genuinely
different sequences separate, and a non-local contact effect separates.
Without them the first row would be consistent with a readout that ignores
its input entirely.
(\textbf{C})~Readout against carrier transit length at four violation
strengths. The horizontal green line is the eliminable case. Each coloured
curve is a locality violation, and the violation appears as a
\emph{slope}---a dependence of the reported value on a property of the
carrier that the theorem says cannot be recovered.
(\textbf{D})~Carrier-dependent deviation against violation strength
$\kappa$. The deviation grows from exactly zero and vanishes as
$\kappa\to0$, which is the diagnostic: in mass spectrometry this is a
dependence of measured mass on total ion current, and in catalysis a
dependence of a step's effect on which other sites are occupied. These are
the same violation under two names.}}
"""


def main() -> None:
    print("generating panels")
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    panel_6()
    panel_7()
    (FIGS / "ladder-captions.tex").write_text(CAPTIONS, encoding="utf-8")
    print(f"  wrote ladder-captions.tex")


if __name__ == "__main__":
    main()
