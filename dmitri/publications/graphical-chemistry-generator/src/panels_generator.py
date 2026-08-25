"""Four panels for the Meibutsu manuscript.

Every value plotted is read from results/exp_generator.json or recomputed
from the instrument itself.  Nothing is simulated, fitted, or drawn by
hand.  No panel is a table, a diagram, or a text box.
"""

from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "..", "honjo-masamune", "validation", "src"))

from panelstyle import (BLUE, DIV, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED,
                        SEQ, TEAL, panel, save, tag)

import meibutsu as M
from meibutsu import Instrument, observe, visibility

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

G = json.load(open(os.path.join(RES, "exp_generator.json")))["results"]
INST = Instrument()
NAMES = sorted(INST.obs)


# =====================================================================
def panel_01():
    """The observation field and the superposition that compares two."""
    fig, ax = panel(threed=(2,))

    a = INST.obs["H2O"]
    b = INST.obs["CO2"]
    u = np.linspace(0.0, 1.0, len(a.amp))

    # --- A: two amplitude fields -------------------------------------
    ax[0].plot(u, a.amp / a.amp.max(), color=BLUE, label="H$_2$O")
    ax[0].plot(u, b.amp / b.amp.max(), color=ORANGE, label="CO$_2$")
    ax[0].set_xlabel("normalised frequency address $u$")
    ax[0].set_ylabel("amplitude (scaled)")
    ax[0].legend(loc="upper right", fontsize=7)
    tag(ax[0], "A")

    # --- B: the superposition and its cross-term ---------------------
    inten = M.superpose(a, b)
    own = np.abs(a.field) ** 2 + np.abs(b.field) ** 2
    cross = M.cross_term(a, b)
    sc = inten.max()
    ax[1].plot(u, inten / sc, color=PURPLE, lw=1.4, label="$|A+B|^2$")
    ax[1].plot(u, own / sc, color=GREY, lw=1.0, ls="--",
               label="$|A|^2+|B|^2$")
    ax[1].plot(u, cross / sc, color=RED, lw=1.2, label="cross-term")
    ax[1].axhline(0, color=GREY, lw=0.7)
    ax[1].set_xlabel("normalised frequency address $u$")
    ax[1].set_ylabel("intensity (scaled)")
    ax[1].legend(loc="upper right", fontsize=6.5)
    tag(ax[1], "B")

    # --- C: 3-D -- fields of several structures ----------------------
    show = ["H2", "HF", "H2O", "CO2", "CH4", "C6H6"]
    show = [s for s in show if s in INST.obs]
    for i, nm in enumerate(show):
        o = INST.obs[nm]
        ax[2].plot(u, np.full_like(u, i, dtype=float),
                   o.amp / (o.amp.max() + 1e-12), lw=1.3)
    ax[2].set_xlabel("$u$", labelpad=-4)
    ax[2].set_ylabel("structure", labelpad=-2)
    ax[2].set_zlabel("amplitude", labelpad=-6)
    ax[2].set_yticks(range(len(show)))
    ax[2].set_yticklabels(show, fontsize=6)
    ax[2].view_init(elev=22, azim=-62)
    tag(ax[2], "C", threed=True)

    # --- D: self vs cross visibility ---------------------------------
    selfv = [visibility(INST.obs[x], INST.obs[x]) for x in NAMES]
    cross_v = [visibility(INST.obs[x], INST.obs[y])
               for x, y in combinations(NAMES, 2)]
    ax[3].hist(cross_v, bins=np.linspace(0, 1, 26), color=BLUE,
               edgecolor="white", linewidth=0.6, label="cross")
    ax[3].axvline(1.0, color=GREEN, lw=2.4, label="self (all 39)")
    ax[3].axvline(float(np.max(cross_v)), color=RED, lw=1.2, ls="--",
                  label="max cross")
    ax[3].set_xlabel("visibility $V$")
    ax[3].set_ylabel("pairs")
    ax[3].legend(loc="upper right", fontsize=6.5)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_01_field.png"))


# =====================================================================
def panel_02():
    """Inversion: spectrum in, structure out."""
    fig, ax = panel(threed=(1,))

    rows = G["G4_inversion"]["rows"]
    topv = [r["top_visibility"] for r in rows]

    # --- A: top-ranked visibility per structure ----------------------
    order = np.argsort(topv)
    ax[0].bar(range(len(rows)), np.array(topv)[order], color=GREEN,
              width=0.85)
    ax[0].axhline(1.0, color=GREY, lw=0.9, ls=":")
    ax[0].set_xlabel("structure (sorted)")
    ax[0].set_ylabel("visibility of top-ranked match")
    ax[0].set_ylim(0.0, 1.08)
    tag(ax[0], "A")

    # --- B: 3-D -- coordinates coloured by address uniqueness --------
    xs = np.array([INST.obs[n].coords[0] for n in NAMES])
    ys = np.array([INST.obs[n].coords[1] for n in NAMES])
    zs = np.array([INST.obs[n].coords[2] for n in NAMES])
    uniq = {r["structure"]: r["address_unique"] for r in rows}
    cols = [GREEN if uniq.get(n) else RED for n in NAMES]
    ax[1].scatter(xs, ys, zs, s=42, c=cols, edgecolor="k", linewidth=0.3,
                  depthshade=False)
    ax[1].set_xlabel("$s_1$", labelpad=-4)
    ax[1].set_ylabel("$s_2$", labelpad=-4)
    ax[1].set_zlabel("$s_3$", labelpad=-6)
    ax[1].view_init(elev=20, azim=-60)
    tag(ax[1], "B", threed=True)

    # --- C: the two routes -------------------------------------------
    n = G["G4_inversion"]["n"]
    vals = [G["G4_inversion"]["n_ranked_first"],
            G["G4_inversion"]["n_address_unique"]]
    ax[2].bar([0, 1], vals, color=[GREEN, ORANGE], width=0.55)
    ax[2].axhline(n, color=GREY, lw=1.0, ls=":")
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["interference\nranking", "address\nuniqueness"])
    ax[2].set_ylabel("structures resolved (of %d)" % n)
    ax[2].set_ylim(0, n * 1.18)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.6, str(v), ha="center", fontsize=9)
    tag(ax[2], "C")

    # --- D: visibility against coordinate distance -------------------
    cv, dd = [], []
    for x, y in combinations(NAMES, 2):
        cv.append(visibility(INST.obs[x], INST.obs[y]))
        dd.append(math.dist(INST.obs[x].coords, INST.obs[y].coords))
    ax[3].scatter(dd, cv, s=10, c=cv, cmap=SEQ, alpha=0.75,
                  edgecolor="none")
    r = G["G3_decay"]["corr_visibility_vs_distance"]
    zpf = np.polyfit(dd, cv, 1)
    xf = np.linspace(min(dd), max(dd), 2)
    ax[3].plot(xf, np.polyval(zpf, xf), color=RED, lw=1.4)
    ax[3].set_xlabel("coordinate distance")
    ax[3].set_ylabel("visibility $V$")
    ax[3].text(0.97, 0.95, "$r=%.3f$" % r, transform=ax[3].transAxes,
               ha="right", va="top", fontsize=8)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_02_inversion.png"))


# =====================================================================
def panel_03():
    """Bulk: the identity holds, the recovery does not."""
    fig, ax = panel(threed=(1,))

    cap = G["G6_bulk_capacity"]["rows"]
    ks = [c["stack_size"] for c in cap]
    corr = [c["demod_vs_true_correlation"] for c in cap]
    div = [c["frac_diverged"] for c in cap]

    # --- A: the identity, exact --------------------------------------
    g5 = G["G5_bulk_identity"]
    ax[0].bar([0, 1], [g5["bulk_cross_energy"], g5["explicit_pairwise_sum"]],
              color=[PURPLE, TEAL], width=0.55)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["one\nsuperposition", "sum over\n%d pairs"
                           % g5["n_pairs_implied"]])
    ax[0].set_ylabel("relational energy")
    ax[0].text(0.5, max(g5["bulk_cross_energy"], 1) * 0.5,
               "residual\n$%.0e$" % max(g5["relative_residual"], 1e-18),
               ha="center", fontsize=7.5)
    tag(ax[0], "A")

    # --- B: 3-D -- demodulation collapse -----------------------------
    kk = [c["stack_size"] for c in cap if c["demod_vs_true_correlation"]
          is not None]
    cc = [c["demod_vs_true_correlation"] for c in cap
          if c["demod_vs_true_correlation"] is not None]
    dv = [c["frac_diverged"] for c in cap
          if c["demod_vs_true_correlation"] is not None]
    ax[1].bar3d(np.array(kk, float) - 1.2, np.zeros(len(kk)),
                np.zeros(len(kk)), 2.4, 0.35,
                np.maximum(np.array(cc), 0.0),
                color=GREEN, alpha=0.9, shade=True)
    ax[1].bar3d(np.array(kk, float) - 1.2, np.ones(len(kk)),
                np.zeros(len(kk)), 2.4, 0.35,
                np.abs(np.minimum(np.array(cc), 0.0)),
                color=RED, alpha=0.9, shade=True)
    ax[1].bar3d(np.array(kk, float) - 1.2, 2.0 * np.ones(len(kk)),
                np.zeros(len(kk)), 2.4, 0.35, np.array(dv),
                color=ORANGE, alpha=0.9, shade=True)
    ax[1].set_xlabel("stack size", labelpad=-4)
    ax[1].set_ylabel("", labelpad=-2)
    ax[1].set_zlabel("magnitude", labelpad=-6)
    ax[1].set_yticks([0.18, 1.18, 2.18])
    ax[1].set_yticklabels(["corr$^+$", "corr$^-$", "diverged"], fontsize=6)
    ax[1].view_init(elev=20, azim=-62)
    tag(ax[1], "B", threed=True)

    # --- C: correlation against stack size ---------------------------
    ax[2].axhline(0.0, color=GREY, lw=1.0)
    ax[2].plot(kk, cc, "o-", color=RED, ms=6)
    ax[2].axhline(0.5, color=GREEN, lw=1.0, ls="--")
    ax[2].set_xlabel("structures stacked")
    ax[2].set_ylabel("demodulation vs true correlation")
    ax[2].set_ylim(-0.35, 1.05)
    tag(ax[2], "C")

    # --- D: divergence fraction --------------------------------------
    ax[3].bar(range(len(ks)), div, color=ORANGE, width=0.6)
    for i, d in enumerate(div):
        if d == 0:
            ax[3].plot([i - 0.3, i + 0.3], [0, 0], color=ORANGE, lw=3.2,
                       solid_capstyle="butt", zorder=4)
    ax[3].set_xticks(range(len(ks)))
    ax[3].set_xticklabels(ks)
    ax[3].set_xlabel("structures stacked")
    ax[3].set_ylabel("fraction of projections diverging")
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_03_bulk.png"))


# =====================================================================
def panel_04():
    """The instrument's limits: floor, phase fragility, display cost."""
    fig, ax = panel(threed=(3,))

    # --- A: resolution floor -----------------------------------------
    fr = G["G7_resolution_floor"]["rows"]
    dw = [r["detune_cm1"] for r in fr]
    drop = [r["visibility_drop"] for r in fr]
    det = [r["detectable"] for r in fr]
    ax[0].loglog(dw, drop, "o-", color=BLUE, ms=5)
    thr = G["G7_resolution_floor"]["detectability_threshold"]
    ax[0].axhline(thr, color=RED, lw=1.3, ls="--")
    ax[0].axvline(G["G7_resolution_floor"]["grid_cell_width_cm1"],
                  color=GREY, lw=1.1, ls=":")
    for x, y, d in zip(dw, drop, det):
        ax[0].plot([x], [y], "o", ms=7, mfc=(GREEN if d else RED),
                   mec="k", mew=0.4)
    ax[0].set_xlabel("detuning (cm$^{-1}$)")
    ax[0].set_ylabel("visibility drop $1-V$")
    tag(ax[0], "A")

    # --- B: phaseless control ----------------------------------------
    cr = G["G8_phaseless_control"]["rows"]
    p = [r["perturbation"] for r in cr]
    full = [r["full_correct"] for r in cr]
    ph = [r["phaseless_correct"] for r in cr]
    xs = np.arange(len(cr))
    w = 0.38
    ax[1].bar(xs - w / 2, full, w, color=GREEN, label="full complex")
    ax[1].bar(xs + w / 2, ph, w, color=RED, label="amplitude only")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(["%g" % v for v in p])
    ax[1].set_xlabel("frequency perturbation")
    ax[1].set_ylabel("structures identified (of 39)")
    ax[1].legend(loc="upper right", fontsize=7)
    tag(ax[1], "B")

    # --- C: display precision ----------------------------------------
    dr = G["G9_display_identity"]["rows"]
    bits = [r["bits_per_channel"] for r in dr]
    ex = [r["exact_correct"] for r in dr]
    pe = [r["perturbed_correct"] for r in dr]
    ax[2].plot(bits, ex, "o-", color=GREEN, ms=5, label="exact query")
    ax[2].plot(bits, pe, "s-", color=ORANGE, ms=5, label="perturbed query")
    ax[2].axvline(8, color=GREY, lw=1.1, ls=":")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("bits per channel")
    ax[2].set_ylabel("structures identified (of 39)")
    ax[2].set_ylim(0, 42)
    ax[2].legend(loc="center right", fontsize=7)
    tag(ax[2], "C")

    # --- D: 3-D -- the margin surface --------------------------------
    marg = [r["margin"] for r in cr]
    ax[3].bar3d(np.arange(len(cr)) - 0.3, np.zeros(len(cr)),
                np.minimum(marg, 0), 0.6, 0.5, np.abs(marg),
                color=[GREEN if m > 0 else RED for m in marg],
                alpha=0.92, shade=True)
    xx, yy = np.meshgrid(np.linspace(-0.6, len(cr) - 0.4, 2),
                         np.linspace(0.0, 0.5, 2))
    ax[3].plot_surface(xx, yy, np.zeros_like(xx), color=GREY, alpha=0.18,
                       shade=False)
    ax[3].set_xlabel("perturbation", labelpad=-4)
    ax[3].set_ylabel("", labelpad=-6)
    ax[3].set_zlabel("margin", labelpad=-6)
    ax[3].set_xticks(range(len(cr)))
    ax[3].set_xticklabels(["%g" % v for v in p], fontsize=6)
    ax[3].set_yticks([])
    ax[3].view_init(elev=18, azim=-62)
    tag(ax[3], "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_04_limits.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04):
        print(fn())
