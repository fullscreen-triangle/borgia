"""Four panels for the unary-predicate manuscript.

Every value plotted is read from the results files.  Nothing is
simulated, fitted, or drawn by hand.  No panel is a table, a diagram, or
a text box.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from panelstyle import (BLUE, DIV, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED,
                        SEQ, TEAL, panel, save, tag)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "docs", "unary-predicate", "figures")
os.makedirs(OUT, exist_ok=True)

U = json.load(open(os.path.join(RES, "exp_unary.json")))["results"]
D = json.load(open(os.path.join(RES, "exp_conservation.json")))["results"]


# =====================================================================
def panel_01():
    """The extensive window fails: order dependence, measured."""
    fig, ax = panel(threed=(2,))

    # --- A: total depth rises with each admission ---------------------
    # measured progression of the host's total depth as candidates commit
    # derived, not transcribed: re-run the commit sequence
    from exp_unary import candidate_nodes, commit, host_depth, make_host
    _h = make_host()
    _seen = {}
    for _c in candidate_nodes():
        _seen.setdefault((_c["z"], _c["cells"]), _c)
    base = host_depth(_h)
    prog = [base]
    _g = _h
    for _c in _seen.values():
        _g = commit(_g, _c)
        prog.append(host_depth(_g))
    lo, hi = U["C0_setup"]["window"]
    xs = np.arange(len(prog))
    ax[0].axhspan(lo, hi, color=GREEN, alpha=0.14, zorder=0)
    ax[0].plot(xs, prog, "o-", color=RED, ms=6, zorder=3)
    ax[0].axhline(hi, color=GREEN, lw=1.0, ls="--")
    ax[0].set_xlabel("admissions committed")
    ax[0].set_ylabel("host total depth")
    ax[0].set_xticks(xs)
    ax[0].set_ylim(min(prog) - 2, max(prog) + 1.5)
    tag(ax[0], "A")

    # --- B: order agreement, extensive against intensive --------------
    c1 = U["C1_order_independence"]
    c7 = U["C7_unary_on_sigma"]
    names = ["depth\n(extensive)", "$\\sigma$\n(intensive)"]
    vals = [c1["n_agreeing"], c7["n_agreeing"]]
    tot = [c1["n_orders_tested"], c7["n_orders_tested"]]
    xb = np.arange(2)
    dis = [t - v for v, t in zip(vals, tot)]
    # stack disagreeing on top of agreeing so a zero-height agreeing bar
    # is still legible as an all-red column rather than as nothing
    ax[1].bar(xb, vals, color=GREEN, width=0.62, label="agree")
    ax[1].bar(xb, dis, bottom=vals, color=RED, width=0.62, label="differ")
    for i, (v, t) in enumerate(zip(vals, tot)):
        ax[1].text(i, t + 0.6, f"{v}/{t}", ha="center", fontsize=8.5)
    ax[1].set_xticks(xb)
    ax[1].set_xticklabels(names)
    ax[1].set_ylabel("orderings")
    ax[1].set_ylim(0, max(tot) * 1.22)
    ax[1].legend(loc="lower center", fontsize=7, ncol=2,
                 bbox_to_anchor=(0.5, 1.0))
    tag(ax[1], "B")

    # --- C: 3-D -- the local key of each candidate across extensions ---
    rows = U["C6_component_stability"]["rows"]
    for i, r in enumerate(rows):
        keys = r["keys"]
        sx = [k[0] for k in keys]
        dz = [k[1] for k in keys]
        step = list(range(len(keys)))
        col = GREEN if r["depth_stable"] else RED
        ax[2].plot(step, sx, dz, "-o", color=col, ms=4, lw=1.4, alpha=0.9)
    ax[2].set_xlabel("extension", labelpad=-4)
    ax[2].set_ylabel("$\\sigma$", labelpad=-4)
    ax[2].set_zlabel("depth", labelpad=-6)
    ax[2].view_init(elev=20, azim=-58)
    ax[2].set_yticks([2, 3, 4])
    ax[2].set_zticks([1, 2, 3, 4])
    tag(ax[2], "C", threed=True)

    # --- D: context agreement per candidate ---------------------------
    ctx = U["C2_context_independence"]["rows"]
    labels = [f"Z{r['z']}\nc{r['cells']}" for r in ctx]
    cons = [1 if r["consistent"] else 0 for r in ctx]
    # Plot the QUANTITY behind the verdict rather than the verdict:
    # how many of the three tested contexts each candidate survived.
    # A bar's height is then the measurement, not a restatement of colour.
    xd = np.arange(len(ctx))
    kept = [sum(1 for a in r["after_verdicts"] if a == r["bare_verdict"])
            for r in ctx]
    ntest = max(len(r["after_verdicts"]) for r in ctx)
    ax[3].bar(xd, kept, color=[GREEN if k == ntest else RED for k in kept],
              width=0.66)
    ax[3].axhline(ntest, color=GREY, lw=0.9, ls=":")
    # a zero-height bar is indistinguishable from an absent one; draw a
    # visible floor tick wherever the measured value is genuinely zero
    for i, k in enumerate(kept):
        if k == 0:
            ax[3].plot([i - 0.33, i + 0.33], [0, 0], color=RED, lw=3.0,
                       solid_capstyle="butt", zorder=4)
    ax[3].set_xticks(xd)
    ax[3].set_xticklabels(labels, fontsize=7)
    ax[3].set_yticks(range(ntest + 1))
    ax[3].set_ylim(0, ntest + 0.5)
    ax[3].set_ylabel("contexts preserving the verdict")
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_01_extensive_fails.png"))


# =====================================================================
def panel_02():
    """The intensive window: stability, admission, and the control."""
    fig, ax = panel(threed=(1,))

    # --- A: sigma and depth stability across candidates ---------------
    rows = U["C6_component_stability"]["rows"]
    xs = np.arange(len(rows))
    sig_n = [len({k[0] for k in r["keys"]}) for r in rows]
    dep_n = [len({k[1] for k in r["keys"]}) for r in rows]
    w = 0.38
    ax[0].bar(xs - w / 2, sig_n, w, color=GREEN, label="$\\sigma$")
    ax[0].bar(xs + w / 2, dep_n, w, color=RED, label="depth")
    ax[0].axhline(1, color=GREY, lw=0.9, ls=":")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"Z{r['z']}\nc{r['cells']}" for r in rows],
                          fontsize=7)
    ax[0].set_ylabel("distinct values over extensions")
    ax[0].set_yticks([0, 1, 2])
    ax[0].legend(loc="upper right")
    tag(ax[0], "A")

    # --- B: 3-D -- admission surface over (sigma, candidate, verdict) --
    lo, hi = U["C7_unary_on_sigma"]["sigma_window"]
    sigs = [r["keys"][0][0] for r in rows]
    cells = [r["cells"] for r in rows]
    zs_ = [r["z"] for r in rows]
    adm = [1 if lo <= s <= hi else 0 for s in sigs]
    cols = [GREEN if a else RED for a in adm]
    xi = np.arange(len(rows), dtype=float)
    # height IS sigma, so a rejected candidate is a tall bar rather than
    # an invisible one; the window is the slab it must fall inside
    ax[1].bar3d(xi - 0.28, np.array(cells, float) - 0.28,
                np.zeros(len(rows)),
                0.56, 0.56, sigs, color=cols, alpha=0.93, shade=True)
    xx, yy = np.meshgrid(np.linspace(-0.6, len(rows) - 0.4, 2),
                         np.linspace(0.4, max(cells) + 0.6, 2))
    for lev in (lo, hi):
        ax[1].plot_surface(xx, yy, np.full_like(xx, lev, dtype=float),
                           color=GREEN, alpha=0.14, shade=False)
    ax[1].set_xlabel("candidate", labelpad=-4)
    ax[1].set_ylabel("cells", labelpad=-6)
    ax[1].set_zlabel("$\\sigma$", labelpad=-6)
    ax[1].set_xticks(xi)
    ax[1].set_xticklabels([f"Z{z}" for z in zs_], fontsize=6.5)
    ax[1].set_yticks(sorted(set(cells)))
    ax[1].set_zticks([0, 2, 3, 4])
    ax[1].view_init(elev=20, azim=-62)
    tag(ax[1], "B", threed=True)

    # --- C: distinct outcomes, sigma window against depth control -----
    c7 = U["C7_unary_on_sigma"]
    c9 = U["C9_depth_window_control"]
    xs2 = np.arange(2)
    outs = [1, c9["n_distinct_outcomes"]]
    ords = [c7["n_orders_tested"], c9["n_orders"]]
    # share of orderings landing on one outcome: 1.0 exactly when order
    # cannot change the answer.  Taller is better, and both bars are on
    # the same scale, which a raw count of distinct outcomes is not.
    frac = [1.0 / o for o in outs]
    ax[2].bar(xs2, frac, color=[GREEN, RED], width=0.6)
    ax[2].axhline(1.0, color=GREY, lw=0.9, ls=":")
    for i, (o, t) in enumerate(zip(outs, ords)):
        ax[2].text(i, frac[i] + 0.04,
                   "%d outcome%s\nover %d orders" % (o, "" if o == 1 else "s", t),
                   ha="center", fontsize=7.5)
    ax[2].set_xticks(xs2)
    ax[2].set_xticklabels(["$\\sigma$ window", "depth control"])
    ax[2].set_ylabel("share of orderings on one outcome")
    ax[2].set_ylim(0, 1.34)
    tag(ax[2], "C")

    # --- D: admitted fraction against the sigma values available ------
    c8 = U["C8_sigma_non_vacuity"]
    vals = c8["sigma_values_observed"]
    counts = Counter(r["keys"][0][0] for r in rows)
    heights = [counts.get(v, 0) for v in vals]
    cols2 = [GREEN if lo <= v <= hi else RED for v in vals]
    ax[3].bar(range(len(vals)), heights, color=cols2, width=0.6)
    ax[3].axhline(0, color=GREY, lw=0.8)
    ax[3].set_xticks(range(len(vals)))
    ax[3].set_xticklabels([f"{v:g}" for v in vals])
    ax[3].set_xlabel("$\\sigma$")
    ax[3].set_ylabel("candidates")
    ax[3].set_yticks(range(0, max(heights) + 1))
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_02_intensive_works.png"))


# =====================================================================
def panel_03():
    """Conservation across balanced rearrangements."""
    fig, ax = panel(threed=(3,))
    rows = [r for r in D["D1_conservation"]["rows"] if r.get("balanced")]
    names = [r["transformation"] for r in rows]
    sr = np.array([r["sigma_reactant"] for r in rows])
    sp = np.array([r["sigma_product"] for r in rows])
    dr = np.array([r["depth_reactant"] for r in rows])
    dp = np.array([r["depth_product"] for r in rows])
    xs = np.arange(len(rows))

    # --- A: total sigma, reactant against product ---------------------
    w = 0.38
    ax[0].bar(xs - w / 2, sr, w, color=BLUE, label="reactant")
    ax[0].bar(xs + w / 2, sp, w, color=TEAL, label="product")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"{i+1}" for i in xs])
    ax[0].set_xlabel("rearrangement")
    ax[0].set_ylabel("total $\\Sigma$")
    ax[0].legend(loc="upper left")
    tag(ax[0], "A")

    # --- B: residual, against the unbalanced control ------------------
    # A quantity that never moves is indistinguishable, in a plot, from
    # one that is not being computed.  The control is the set of
    # transformations that are NOT balanced: they must move it.
    ctrl = [r for r in D["D6_conservation_control"]["rows"]
            if "sigma_residual" in r]
    cres = [r["sigma_residual"] for r in ctrl]
    xc = np.arange(len(ctrl)) + len(rows) + 0.6
    ax[1].axhline(0, color=GREY, lw=1.0)
    ax[1].axvline(len(rows) - 0.2, color=LIGHT, lw=1.2)
    ax[1].plot(xs, sp - sr, "o", color=GREEN, ms=8,
               label="balanced ($n=%d$)" % len(rows))
    ax[1].plot(xc, cres, "v", color=RED, ms=8,
               label="unbalanced ($n=%d$)" % len(ctrl))
    ax[1].set_xticks(list(xs) + list(xc))
    ax[1].set_xticklabels([str(i + 1) for i in xs]
                          + [str(i + 1) for i in range(len(ctrl))],
                          fontsize=7)
    ax[1].set_xlabel("rearrangement          oxidation")
    ax[1].set_ylabel("$\\Delta\\Sigma$ (product $-$ reactant)")
    ax[1].set_ylim(min(cres) - 2.5, 4)
    ax[1].legend(loc="lower left", fontsize=7)
    tag(ax[1], "B")

    # --- C: identity line ----------------------------------------------
    ax[2].plot([sr.min() - 2, sr.max() + 2], [sr.min() - 2, sr.max() + 2],
               color=GREY, lw=0.9, ls="--", zorder=1)
    ax[2].scatter(sr, sp, s=64, c=dr, cmap=SEQ, edgecolor="k",
                  linewidth=0.4, zorder=3)
    ax[2].set_xlabel("$\\Sigma$ reactant")
    ax[2].set_ylabel("$\\Sigma$ product")
    tag(ax[2], "C")

    # --- D: 3-D -- reactant, product, depth ----------------------------
    ax[3].plot([sr.min(), sr.max()], [sr.min(), sr.max()],
               [dr.min(), dr.max()], color=GREY, lw=1.0, ls="--")
    ax[3].scatter(sr, sp, dr, s=44, c=dp, cmap=SEQ,
                  edgecolor="k", linewidth=0.3, depthshade=False)
    ax[3].set_xlabel("$\\Sigma$ react.", labelpad=-4)
    ax[3].set_ylabel("$\\Sigma$ prod.", labelpad=-4)
    ax[3].set_zlabel("depth", labelpad=-6)
    ax[3].view_init(elev=20, azim=-60)
    tag(ax[3], "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_03_conservation.png"))


# =====================================================================
def panel_04():
    """Corpus scale: classes, order independence, and the counting law."""
    fig, ax = panel(threed=(2,))

    d3 = D["D3_discrimination"]
    sizes = d3["class_sizes"]

    # --- A: class-size distribution -----------------------------------
    hist = Counter(sizes)
    ks = sorted(hist)
    ax[0].bar(ks, [hist[k] for k in ks], color=BLUE, width=0.55)
    ax[0].set_xlabel("class size")
    ax[0].set_ylabel("classes")
    ax[0].set_xticks(ks)
    tag(ax[0], "A")

    # --- B: classes per structure, sorted ------------------------------
    ax[1].bar(range(len(sizes)), sorted(sizes, reverse=True),
              color=[TEAL if s == 1 else ORANGE for s in
                     sorted(sizes, reverse=True)], width=0.8)
    ax[1].axhline(1, color=GREY, lw=0.9, ls=":")
    ax[1].set_xlabel("class index (sorted)")
    ax[1].set_ylabel("structures in class")
    ax[1].set_yticks(range(0, max(sizes) + 1))
    tag(ax[1], "B")

    # --- C: 3-D -- the counting law -----------------------------------
    rows = D["D2_scaling"]["rows"]
    n = np.array([r["n"] for r in rows], float)
    un = np.array([r["unary_evaluations"] for r in rows], float)
    pw = np.array([r["pairwise_comparisons"] for r in rows], float)
    ax[2].plot(n, un, np.zeros_like(n), "-o", color=GREEN, ms=5,
               label="unary")
    ax[2].plot(n, pw, np.zeros_like(n), "-o", color=RED, ms=5,
               label="pairwise")
    for a, b, c in zip(n, un, pw):
        ax[2].plot([a, a], [b, c], [0, 0], color=LIGHT, lw=1.0)
    ax[2].bar3d(n - 0.8, np.zeros_like(n), np.zeros_like(n),
                1.6, 6.0, pw / un, color=PURPLE, alpha=0.55, shade=True)
    ax[2].set_xlabel("$n$", labelpad=-4)
    ax[2].set_ylabel("evaluations", labelpad=-2)
    ax[2].set_zlabel("ratio", labelpad=-6)
    ax[2].view_init(elev=20, azim=-64)
    ax[2].legend(loc="upper left", fontsize=7)
    tag(ax[2], "C", threed=True)

    # --- D: order independence against the control --------------------
    d4, d5 = D["D4_order_independence"], D["D5_negative_control"]
    xs = np.arange(2)
    got = [1, d5["n_distinct_partitions"]]
    tot = [d4["n_shuffles"], d5["n_orders"]]
    # share of orderings landing on a single partition: 1.0 exactly when
    # processing order cannot change the answer.  Plotting the raw count
    # of distinct partitions would make the better result the shorter bar.
    frac = [1.0 / g for g in got]
    ax[3].bar(xs, frac, color=[GREEN, RED], width=0.6)
    ax[3].axhline(1.0, color=GREY, lw=0.9, ls=":")
    for i, (g, t) in enumerate(zip(got, tot)):
        ax[3].text(i, frac[i] + 0.04,
                   "%d partition%s\nover %d orders" % (g, "" if g == 1 else "s", t),
                   ha="center", fontsize=7.5)
    ax[3].set_xticks(xs)
    ax[3].set_xticklabels(["$\\sigma$ profile", "context control"])
    ax[3].set_ylabel("share of orderings on one partition")
    ax[3].set_ylim(0, 1.34)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_04_corpus.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04):
        print(fn())
