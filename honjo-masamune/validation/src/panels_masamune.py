"""Four panels for the Masamune converter manuscript.

Every value plotted is read from validation/results/exp_masamune.json.
Nothing is simulated, fitted, or drawn by hand.  No panel is a table, a
diagram, or a text box.
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
OUT = os.path.join(HERE, "..", "..", "docs",
                   "masamune-representation-converter", "figures")
os.makedirs(OUT, exist_ok=True)

M = json.load(open(os.path.join(RES, "exp_masamune.json")))["results"]


# =====================================================================
def panel_01():
    """The supplied fraction: how much of a translated graph is convention."""
    fig, ax = panel(threed=(2,))
    rows = [r for r in M["M1_supplied_fraction"]["rows"] if "smiles_phi" in r]
    phi = np.array([r["smiles_phi"] for r in rows])
    phim = np.array([r["smiles_phi_with_medium"] for r in rows])
    nat = np.array([r["n_atoms"] for r in rows])
    nhv = np.array([r["n_heavy"] for r in rows])
    mean = M["M1_supplied_fraction"]["smiles_mean_phi"]

    # --- A: distribution of phi ---------------------------------------
    ax[0].hist(phi, bins=np.arange(0.45, 0.95, 0.05), color=BLUE,
               edgecolor="white", linewidth=0.8)
    ax[0].axvline(mean, color=RED, lw=1.4, ls="--")
    ax[0].axvline(0.5, color=GREY, lw=1.0, ls=":")
    ax[0].set_xlabel("supplied fraction $\\phi$")
    ax[0].set_ylabel("structures")
    tag(ax[0], "A")

    # --- B: phi against molecular size --------------------------------
    order = np.argsort(nhv)
    ax[1].scatter(nhv[order], phi[order], s=42, c=nat[order], cmap=SEQ,
                  edgecolor="k", linewidth=0.35, zorder=3)
    ax[1].axhline(mean, color=RED, lw=1.2, ls="--")
    ax[1].set_xlabel("heavy atoms")
    ax[1].set_ylabel("$\\phi$")
    ax[1].set_ylim(0.4, 1.0)
    tag(ax[1], "B")

    # --- C: 3-D -- corrected against uncorrected phi over size --------
    ax[2].scatter(nhv, phi, nat, s=34, c=GREEN, edgecolor="k",
                  linewidth=0.3, depthshade=False, label="excl. medium")
    ax[2].scatter(nhv, phim, nat, s=34, c=RED, edgecolor="k",
                  linewidth=0.3, depthshade=False, label="incl. medium")
    for a, b, c, d in zip(nhv, phi, phim, nat):
        ax[2].plot([a, a], [b, c], [d, d], color=LIGHT, lw=0.8)
    ax[2].set_xlabel("heavy atoms", labelpad=-4)
    ax[2].set_ylabel("$\\phi$", labelpad=-4)
    ax[2].set_zlabel("all atoms", labelpad=-6)
    ax[2].view_init(elev=20, azim=-62)
    ax[2].legend(loc="upper left", fontsize=6.5)
    tag(ax[2], "C", threed=True)

    # --- D: the medium-edge inflation per structure -------------------
    infl = phim - phi
    o2 = np.argsort(nhv)
    ax[3].bar(range(len(infl)), infl[o2], color=ORANGE, width=0.85)
    ax[3].axhline(M["M5_medium_edge_defect"]["mean_inflation"],
                  color=RED, lw=1.3, ls="--")
    ax[3].set_xlabel("structure (by heavy-atom count)")
    ax[3].set_ylabel("$\\phi$ inflation from medium edges")
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_01_supplied.png"))


# =====================================================================
def panel_02():
    """Capability containment: refusal before any record is read."""
    fig, ax = panel(threed=(1,))
    rows = M["M2_capability_containment"]["rows"]
    fmts = sorted({r["format"] for r in rows})
    reqs = sorted({len(r["required"]) for r in rows})

    # --- A: refusals per format ---------------------------------------
    per = {f: [0, 0] for f in fmts}
    for r in rows:
        per[r["format"]][0 if r["static_refusal"] else 1] += 1
    xs = np.arange(len(fmts))
    ref = [per[f][0] for f in fmts]
    acc = [per[f][1] for f in fmts]
    ax[0].bar(xs, acc, color=GREEN, width=0.6, label="translatable")
    ax[0].bar(xs, ref, bottom=acc, color=RED, width=0.6,
              label="refused statically")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(fmts)
    ax[0].set_ylabel("requests")
    ax[0].legend(loc="lower center", fontsize=7, ncol=1,
                 bbox_to_anchor=(0.5, 1.0))
    tag(ax[0], "A")

    # --- B: 3-D -- declared capability against request size -----------
    fi = {f: i for i, f in enumerate(fmts)}
    xb = np.array([fi[r["format"]] for r in rows], float)
    yb = np.array([r["n_required"] for r in rows], float)
    # height is the number of asked-for features the format DOES declare.
    # Keying it to the missing count instead would make every satisfiable
    # request a zero-height bar, i.e. invisible.
    sat = np.array([r["n_required"] - len(r["missing"]) for r in rows],
                   float)
    cols = [RED if r["static_refusal"] else GREEN for r in rows]
    ax[1].bar3d(xb - 0.26, yb - 0.26, np.zeros(len(rows)),
                0.52, 0.52, np.maximum(sat, 0.08),
                color=cols, alpha=0.92, shade=True)
    # the diagonal plane sat == asked: a bar reaching it is fully served
    yy = np.array(sorted(set(yb)), float)
    for fidx in range(len(fmts)):
        ax[1].plot([fidx] * len(yy), yy, yy, color=GREY, lw=0.9, ls="--")
    ax[1].set_xlabel("format", labelpad=-4)
    ax[1].set_ylabel("features asked", labelpad=-4)
    ax[1].set_zlabel("features declared", labelpad=-6)
    ax[1].set_xticks(range(len(fmts)))
    ax[1].set_xticklabels(fmts, fontsize=6.5)
    ax[1].set_yticks(sorted(set(int(v) for v in yb)))
    ax[1].set_zticks(sorted(set(int(v) for v in yb)))
    ax[1].view_init(elev=20, azim=-60)
    tag(ax[1], "B", threed=True)

    # --- C: declared capability size per format -----------------------
    decl = {}
    for r in rows:
        decl[r["format"]] = r["n_declared"]
    ax[2].bar(range(len(fmts)), [decl[f] for f in fmts],
              color=[TEAL if decl[f] else RED for f in fmts], width=0.6)
    # a declared-nothing format would otherwise be an absent bar rather
    # than a measured zero
    for i, f in enumerate(fmts):
        if decl[f] == 0:
            ax[2].plot([i - 0.3, i + 0.3], [0, 0], color=RED, lw=3.5,
                       solid_capstyle="butt", zorder=4)
    ax[2].set_xticks(range(len(fmts)))
    ax[2].set_xticklabels(fmts)
    ax[2].set_ylabel("features declared")
    tag(ax[2], "C")

    # --- D: static verdict against post-read outcome ------------------
    agree = sum(1 for r in rows if r["consistent"])
    n = len(rows)
    ax[3].bar([0, 1], [agree, n - agree], color=[GREEN, RED], width=0.55)
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["agree", "disagree"])
    ax[3].set_ylabel("format--request pairs")
    ax[3].set_ylim(0, n * 1.18)
    ax[3].text(0, agree + 0.5, str(agree), ha="center", fontsize=9)
    ax[3].text(1, 0.5, str(n - agree), ha="center", fontsize=9)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_02_capability.png"))


# =====================================================================
def panel_03():
    """Cross-representation disagreement, invisible to identity."""
    fig, ax = panel(threed=(3,))
    rows = [r for r in M["M4_format_disagreement"]["rows"]
            if "same_formula" in r]
    names = [r["molecule"] for r in rows]
    da = np.array([r["deloc_edges_aromatic"] for r in rows], float)
    dk = np.array([r["deloc_edges_kekule"] for r in rows], float)
    pa = np.array([r["phi_aromatic"] for r in rows])
    pk = np.array([r["phi_kekule"] for r in rows])
    xs = np.arange(len(rows))

    # --- A: delocalised contacts, two faithful readings ---------------
    w = 0.38
    ax[0].bar(xs - w / 2, da, w, color=PURPLE, label="aromatic")
    ax[0].bar(xs + w / 2, dk, w, color=ORANGE, label="Kekulé")
    # every Kekule value is zero, which is the finding; without a marker
    # the bars are simply absent and read as missing data
    for i in xs:
        if dk[i] == 0:
            ax[0].plot([i + w / 2 - w / 2.4, i + w / 2 + w / 2.4], [0, 0],
                       color=ORANGE, lw=3.2, solid_capstyle="butt", zorder=4)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(range(1, len(rows) + 1), fontsize=7)
    ax[0].set_xlabel("molecule")
    ax[0].set_ylabel("delocalised contacts")
    ax[0].legend(loc="upper right", fontsize=7)
    tag(ax[0], "A")

    # --- B: the supplied fraction differs consistently ----------------
    ax[1].plot(xs, pa, "o-", color=PURPLE, ms=5, label="aromatic")
    ax[1].plot(xs, pk, "s-", color=ORANGE, ms=5, label="Kekulé")
    for i in xs:
        ax[1].plot([i, i], [pk[i], pa[i]], color=LIGHT, lw=1.0, zorder=0)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(range(1, len(rows) + 1), fontsize=7)
    ax[1].set_xlabel("molecule")
    ax[1].set_ylabel("$\\phi$")
    ax[1].legend(loc="lower right", fontsize=7)
    tag(ax[1], "B")

    # --- C: what each comparison detects ------------------------------
    n = len(rows)
    det_formula = sum(1 for r in rows if not r["same_formula"])
    det_cells = sum(1 for r in rows if not r["same_cell_multiset"])
    ax[2].bar([0, 1], [det_formula, det_cells],
              color=[RED, GREEN], width=0.55)
    ax[2].axhline(n, color=GREY, lw=1.0, ls=":")
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["formula\ncomparison", "cell-count\ncomparison"])
    ax[2].set_ylabel("differences detected (of %d)" % n)
    ax[2].set_ylim(0, n * 1.2)
    ax[2].text(0, det_formula + 0.3, str(det_formula), ha="center", fontsize=9)
    ax[2].text(1, det_cells + 0.3, str(det_cells), ha="center", fontsize=9)
    tag(ax[2], "C")

    # --- D: 3-D -- cell multiset totals against delocalisation --------
    ca = np.array([sum(r["cells_aromatic"]) for r in rows], float)
    ck = np.array([sum(r["cells_kekule"]) for r in rows], float)
    ax[3].scatter(ca, da, pa, s=44, c=PURPLE, edgecolor="k",
                  linewidth=0.3, depthshade=False, label="aromatic")
    ax[3].scatter(ck, dk, pk, s=44, c=ORANGE, edgecolor="k",
                  linewidth=0.3, depthshade=False, label="Kekulé")
    for i in range(len(rows)):
        ax[3].plot([ca[i], ck[i]], [da[i], dk[i]], [pa[i], pk[i]],
                   color=LIGHT, lw=0.9)
    ax[3].set_xlabel("total cells", labelpad=-4)
    ax[3].set_ylabel("deloc. contacts", labelpad=-4)
    ax[3].set_zlabel("$\\phi$", labelpad=-6)
    ax[3].view_init(elev=20, azim=-62)
    ax[3].legend(loc="upper left", fontsize=6.5)
    tag(ax[3], "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_03_disagreement.png"))


# =====================================================================
def panel_04():
    """Verdicts: what the label set actually distinguishes."""
    fig, ax = panel(threed=(2,))
    rows = M["M3_verdict_coverage"]["rows"]
    counts = M["M3_verdict_coverage"]["label_counts"]
    labs = sorted(counts, key=lambda k: -counts[k])

    # --- A: labels realised by the probe set --------------------------
    ax[0].bar(range(len(labs)), [counts[k] for k in labs],
              color=[GREEN if k == "translated" else BLUE for k in labs],
              width=0.6)
    ax[0].set_xticks(range(len(labs)))
    ax[0].set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
    ax[0].set_ylabel("probes")
    tag(ax[0], "A")

    # --- B: value-bearing against failure ------------------------------
    vb = sum(1 for r in rows if r["value_bearing_label"])
    fb = len(rows) - vb
    carried = sum(1 for r in rows if r["carries_value"])
    ax[1].bar([0, 1, 2], [vb, fb, carried],
              color=[GREEN, RED, TEAL], width=0.55)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(["value-\nbearing", "failure", "carried\na value"],
                          fontsize=7.5)
    ax[1].set_ylabel("probes")
    ax[1].set_ylim(0, len(rows) * 1.15)
    for i, v in enumerate([vb, fb, carried]):
        ax[1].text(i, v + 0.15, str(v), ha="center", fontsize=9)
    tag(ax[1], "B")

    # --- C: 3-D -- the label space realised ----------------------------
    real = M["M3_verdict_coverage"]["n_labels_realised"]
    defd = M["M3_verdict_coverage"]["n_labels_defined"]
    li = {k: i for i, k in enumerate(labs)}
    xs = np.array([li[r["label"]] for r in rows], float)
    ys = np.array([1 if r["value_bearing_label"] else 0 for r in rows], float)
    zs = np.array([1 if r["carries_value"] else 0 for r in rows], float)
    cols = [GREEN if r["sound"] else RED for r in rows]
    ax[2].bar3d(xs - 0.3, ys - 0.16, np.zeros(len(rows)),
                0.6, 0.32, np.maximum(zs, 0.05),
                color=cols, alpha=0.92, shade=True)
    ax[2].set_xlabel("label", labelpad=-4)
    ax[2].set_ylabel("may carry", labelpad=-4)
    ax[2].set_zlabel("did carry", labelpad=-8)
    ax[2].set_xticks(range(len(labs)))
    ax[2].set_xticklabels(labs, fontsize=6)
    ax[2].set_yticks([0, 1])
    ax[2].set_zticks([0, 1])
    ax[2].view_init(elev=22, azim=-62)
    tag(ax[2], "C", threed=True)

    # --- D: label coverage of the defined set -------------------------
    ax[3].bar([0, 1], [real, defd - real], color=[GREEN, LIGHT], width=0.5)
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["realised\nby probes", "defined,\nnot probed"])
    ax[3].set_ylabel("labels")
    ax[3].set_ylim(0, defd * 1.15)
    ax[3].text(0, real + 0.2, str(real), ha="center", fontsize=9)
    ax[3].text(1, defd - real + 0.2, str(defd - real), ha="center", fontsize=9)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_04_verdicts.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04):
        print(fn())
