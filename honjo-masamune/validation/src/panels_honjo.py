"""Four panels for the honjo DSL manuscript.

Every value plotted is read from validation/results/exp_honjo.json.
Nothing is simulated, fitted, or drawn by hand.  No panel is a table, a
diagram, or a text box.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from panelstyle import (BLUE, DIV, GREEN, GREY, LIGHT, ORANGE, PURPLE, RED,
                        SEQ, TEAL, panel, save, tag)

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "docs", "honjo-dsl", "figures")
os.makedirs(OUT, exist_ok=True)

H = json.load(open(os.path.join(RES, "exp_honjo.json")))["results"]


# =====================================================================
def panel_01():
    """The floor is a resolution scale, not a truncation."""
    fig, ax = panel(threed=(2,))
    rows = H["H1_no_sharp_cut"]["rows"]
    floors = H["H1_no_sharp_cut"]["floors_swept"]

    # --- A: residue against floor, one line per atom ------------------
    for r in rows:
        fl = [x["floor"] for x in r["sweep"]]
        rs = [x["residue"] for x in r["sweep"]]
        ax[0].plot(fl, rs, "o-", ms=5, label=r["atom"])
    lim = [min(floors), max(floors)]
    ax[0].plot(lim, lim, color=GREY, lw=1.0, ls="--")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("declared floor")
    ax[0].set_ylabel("residue")
    ax[0].legend(loc="upper left", fontsize=7, ncol=2)
    tag(ax[0], "A")

    # --- B: the ratio is flat, which is the actual claim --------------
    for r in rows:
        fl = [x["floor"] for x in r["sweep"]]
        ra = [x["ratio"] for x in r["sweep"]]
        ax[1].plot(fl, ra, "o-", ms=5, label=r["atom"])
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("declared floor")
    ax[1].set_ylabel("residue / floor")
    ax[1].set_ylim(0, max(max(x["ratio"] for x in r["sweep"])
                          for r in rows) * 1.25)
    tag(ax[1], "B")

    # --- C: 3-D -- the whole sweep ------------------------------------
    for i, r in enumerate(rows):
        fl = np.array([x["floor"] for x in r["sweep"]], float)
        rs = np.array([x["residue"] for x in r["sweep"]], float)
        ax[2].plot(np.log2(fl), np.full_like(fl, i, dtype=float), rs,
                   "-o", ms=4, lw=1.5)
    ax[2].set_xlabel("$\\log_2$ floor", labelpad=-4)
    ax[2].set_ylabel("atom", labelpad=-4)
    ax[2].set_zlabel("residue", labelpad=-6)
    ax[2].set_yticks(range(len(rows)))
    ax[2].set_yticklabels([r["atom"] for r in rows], fontsize=6.5)
    ax[2].view_init(elev=20, azim=-60)
    tag(ax[2], "C", threed=True)

    # --- D: values below the floor, over the whole sweep --------------
    n = H["H1_no_sharp_cut"]["n_programs"]
    below = H["H1_no_sharp_cut"]["n_values_below_floor"]
    ax[3].bar([0, 1], [n - below, below], color=[GREEN, RED], width=0.55)
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["at or above\nfloor", "below\nfloor"])
    ax[3].set_ylabel("values produced")
    ax[3].set_ylim(0, n * 1.18)
    ax[3].text(0, n - below + 0.4, str(n - below), ha="center", fontsize=9)
    ax[3].text(1, 0.4, str(below), ha="center", fontsize=9)
    if below == 0:
        ax[3].plot([0.7, 1.3], [0, 0], color=RED, lw=3.5,
                   solid_capstyle="butt", zorder=4)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_01_floor.png"))


# =====================================================================
def panel_02():
    """The conflation control: four failures, one wrapped outcome."""
    fig, ax = panel(threed=(1,))
    rows = H["H4_conflation_control"]["rows"]
    names = [r["program"] for r in rows]
    xs = np.arange(len(rows))

    # --- A: distinct outcomes under each interface --------------------
    nd = H["H4_conflation_control"]["n_distinct_outcomes"]
    nw = H["H4_conflation_control"]["n_wrapped_outcomes"]
    ax[0].bar([0, 1], [nd, nw], color=[GREEN, RED], width=0.55)
    ax[0].axhline(len(rows), color=GREY, lw=1.0, ls=":")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["verdict\ninterface", "value-or-nothing\nwrapper"])
    ax[0].set_ylabel("distinct outcomes (of %d programs)" % len(rows))
    ax[0].set_ylim(0, len(rows) * 1.3)
    for i, v in enumerate([nd, nw]):
        ax[0].text(i, v + 0.12, str(v), ha="center", fontsize=9)
    tag(ax[0], "A")

    # --- B: 3-D -- each program's outcome coordinates ------------------
    kinds = sorted({k for r in rows for k in r["verdict_kinds"]} | {""})
    ki = {k: i for i, k in enumerate(kinds)}
    sts = sorted({r["status"] for r in rows})
    si = {v: i for i, v in enumerate(sts)}
    xb = np.array([si[r["status"]] for r in rows], float)
    yb = np.array([ki[r["verdict_kinds"][0]] if r["verdict_kinds"] else ki[""]
                   for r in rows], float)
    zb = np.array([r["n_bindings"] for r in rows], float)
    ax[1].bar3d(xb - 0.22, yb - 0.22, np.zeros(len(rows)),
                0.44, 0.44, np.maximum(zb, 0.08),
                color=RED, alpha=0.9, shade=True)
    ax[1].set_xlabel("status", labelpad=-4)
    ax[1].set_ylabel("verdict", labelpad=-4)
    ax[1].set_zlabel("bindings emitted", labelpad=-6)
    ax[1].set_xticks(range(len(sts)))
    ax[1].set_xticklabels(sts, fontsize=6)
    ax[1].set_yticks(range(len(kinds)))
    ax[1].set_yticklabels([k or "(none)" for k in kinds], fontsize=6)
    ax[1].set_zticks([0, 1])
    ax[1].view_init(elev=20, azim=-58)
    tag(ax[1], "B", threed=True)

    # --- C: provenance monotonicity across imported programs ----------
    # Panel B already shows every failing program carries no value, so
    # repeating that here would be a second view of one number.  This
    # plots a different measured item: the provenance check.
    h2 = H["H2_provenance_monotone"]
    prows = h2["rows"]
    xp = np.arange(len(prows))
    ntags = [len(r["tags_in_order"]) for r in prows]
    viol = [r["violations"] for r in prows]
    w = 0.36
    ax[2].bar(xp - w / 2, ntags, w, color=BLUE, label="values derived")
    ax[2].bar(xp + w / 2, viol, w, color=RED, label="violations")
    for i, v in enumerate(viol):
        if v == 0:
            ax[2].plot([i + w / 2 - w / 2.3, i + w / 2 + w / 2.3], [0, 0],
                       color=RED, lw=3.2, solid_capstyle="butt", zorder=4)
    ax[2].set_xticks(xp)
    ax[2].set_xticklabels(range(1, len(prows) + 1))
    ax[2].set_xlabel("program")
    ax[2].set_ylabel("count")
    ax[2].legend(loc="upper right", fontsize=7)
    tag(ax[2], "C")

    # --- D: verdict realisation across the example programs -----------
    ex = H["H3_verdict_realisation"]["rows"]
    xe = np.arange(len(ex))
    cols = [GREEN if r["status"] == "ok" else RED for r in ex]
    ax[3].bar(xe, [r["cut_count"] for r in ex], color=cols, width=0.6)
    for i, r in enumerate(ex):
        if r["cut_count"] == 0:
            ax[3].plot([i - 0.3, i + 0.3], [0, 0], color=RED, lw=3.5,
                       solid_capstyle="butt", zorder=4)
    ax[3].set_xticks(xe)
    ax[3].set_xticklabels([r["program"].replace(".hj", "") for r in ex],
                          rotation=20, ha="right", fontsize=7)
    ax[3].set_ylabel("cuts committed")
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_02_conflation.png"))


# =====================================================================
def panel_03():
    """Committed cells and delocalisation carry what connectivity cannot."""
    fig, ax = panel(threed=(3,))
    h5 = H["H5_cell_counts_distinguish"]
    h6 = H["H6_deloc_not_a_count"]

    # --- A: the two interfaces, same atoms, different outcome ---------
    two = h5["two_cells"]
    one = h5["one_cell"]
    labels = ["C(O:2, O:2)", "C(O:1, O:1)"]
    built = [1 if h5["two_cells_compound_built"] else 0,
             1 if h5["one_cell_compound_built"] else 0]
    ax[0].bar([0, 1], built, color=[GREEN, RED], width=0.55)
    for i, b in enumerate(built):
        if b == 0:
            ax[0].plot([i - 0.28, i + 0.28], [0, 0], color=RED, lw=3.5,
                       solid_capstyle="butt", zorder=4)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel("compound produced")
    ax[0].set_yticks([0, 1])
    ax[0].set_ylim(0, 1.25)
    tag(ax[0], "A")

    # --- B: residual vacancy left open by the one-cell reading --------
    open_atoms = h5["one_cell_open_atoms"][0] if h5["one_cell_open_atoms"] else []
    if open_atoms:
        syms = [f"{a['symbol']}{i+1}" for i, a in enumerate(open_atoms)]
        vac = [a["vacancy"] for a in open_atoms]
        com = [a["committed"] for a in open_atoms]
        res = [a["residual"] for a in open_atoms]
        xb = np.arange(len(open_atoms))
        w = 0.27
        ax[1].bar(xb - w, vac, w, color=BLUE, label="vacancy")
        ax[1].bar(xb, com, w, color=TEAL, label="committed")
        ax[1].bar(xb + w, res, w, color=RED, label="residual")
        ax[1].set_xticks(xb)
        ax[1].set_xticklabels(syms)
        ax[1].set_ylabel("cells")
        ax[1].legend(loc="upper right", fontsize=7)
    tag(ax[1], "B")

    # --- C: the delocalised system carries a total, not a count -------
    tot = h6["total_cells"][0]
    ncen = h6["n_centres"][0]
    readable = h6["n_readable_per_pair"]
    ax[2].bar([0, 1, 2], [tot, ncen, readable],
              color=[PURPLE, BLUE, RED], width=0.55)
    if readable == 0:
        ax[2].plot([1.72, 2.28], [0, 0], color=RED, lw=3.5,
                   solid_capstyle="butt", zorder=4)
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["total\ncells", "centres", "readable\nper-pair"],
                          fontsize=7.5)
    ax[2].set_ylabel("count")
    for i, v in enumerate([tot, ncen, readable]):
        ax[2].text(i, v + 0.15, str(v), ha="center", fontsize=9)
    tag(ax[2], "C")

    # --- D: 3-D -- the measured sweep over delocalised systems --------
    # Every point is a system the interpreter actually built.  Height is
    # the total the system carries; the orange spine joins two systems
    # with the SAME centre count and DIFFERENT totals, so the total is
    # not a function of the centres and cannot be divided into a
    # per-pair count.  Red caps mark that no per-pair value is readable.
    sw = [r for r in H["H6_deloc_sweep"]["rows"] if r.get("built")]
    cen = np.array([r["n_centres"] for r in sw], float)
    tots = np.array([r["total_cells"] for r in sw], float)
    ratio = tots / cen
    ax[3].bar3d(cen - 0.34, ratio - 0.06, np.zeros(len(sw)),
                0.68, 0.12, tots,
                color=PURPLE, alpha=0.88, shade=True)
    # the centre count carrying two distinct totals
    amb = H["H6_deloc_sweep"]["centre_counts_with_multiple_totals"]
    for k_, vs in amb.items():
        kk = float(k_)
        vv = [float(v) for v in vs]
        ax[3].plot([kk] * len(vv), [v / kk for v in vv], vv,
                   "-o", color=ORANGE, lw=2.6, ms=6, zorder=6)
    ax[3].set_xlabel("centres", labelpad=-4)
    ax[3].set_ylabel("cells per centre", labelpad=-4)
    ax[3].set_zlabel("total cells", labelpad=-6)
    ax[3].set_xticks(sorted(set(int(v) for v in cen)))
    ax[3].view_init(elev=18, azim=-62)
    tag(ax[3], "D", threed=True)

    return save(fig, os.path.join(OUT, "panel_03_cells.png"))


# =====================================================================
def panel_04():
    """The resolution gate and the scope of target equivalence."""
    fig, ax = panel(threed=(2,))
    rows = H["H7_resolution_gate"]["rows"]
    xs = np.arange(len(rows))
    dec = np.array([r["declared_floor"] for r in rows], float)
    eps = np.array([r["target_resolution"] for r in rows], float)

    # --- A: declared floor against the target's resolution ------------
    # Plot the MARGIN above the target resolution, not the raw floor.
    # A raw log10 floor makes the one admitted program (floor 1.0, log 0)
    # an invisible bar while the refused ones become large downward
    # spikes, which reads backwards: the refused programs look like the
    # substantial ones.  Margin > 0 means resolvable.
    margin = np.log10(dec) - np.log10(eps)
    ax[0].bar(xs, margin, color=[RED if r["sub_resolution"] else GREEN
                                 for r in rows], width=0.55)
    ax[0].axhline(0, color=GREY, lw=1.3, ls="--")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(range(1, len(rows) + 1))
    ax[0].set_xlabel("program")
    ax[0].set_ylabel("decades of floor above $\\epsilon_T$")
    tag(ax[0], "A")

    # --- B: refused against admitted ----------------------------------
    ref = sum(1 for r in rows if r["refused"])
    sub = H["H7_resolution_gate"]["n_sub_resolution"]
    ax[1].bar([0, 1], [sub, ref], color=[ORANGE, RED], width=0.55)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["sub-resolution", "refused"])
    ax[1].set_ylabel("programs")
    ax[1].set_ylim(0, len(rows) * 1.2)
    for i, v in enumerate([sub, ref]):
        ax[1].text(i, v + 0.06, str(v), ha="center", fontsize=9)
    tag(ax[1], "B")

    # --- C: 3-D -- the gate in (floor, target, refused) space ---------
    zr = np.array([1 if r["refused"] else 0 for r in rows], float)
    cols = [RED if r["refused"] else GREEN for r in rows]
    ax[2].bar3d(np.log10(dec) - 0.6, np.log10(eps) - 0.12,
                np.zeros(len(rows)),
                1.2, 0.24, np.maximum(zr, 0.05),
                color=cols, alpha=0.92, shade=True)
    # the gate: floor == target resolution
    gl = np.linspace(np.log10(dec).min() - 1, np.log10(dec).max() + 1, 2)
    ax[2].plot(gl, gl, [0, 0], color=GREY, lw=1.2, ls="--")
    ax[2].set_xlabel("$\\log_{10}$ floor", labelpad=-4)
    ax[2].set_ylabel("$\\log_{10}\\epsilon_T$", labelpad=-4)
    ax[2].set_zlabel("refused", labelpad=-8)
    ax[2].set_zticks([0, 1])
    ax[2].view_init(elev=20, azim=-60)
    tag(ax[2], "C", threed=True)

    # --- D: scope of the equivalence claim ----------------------------
    h8 = H["H8_target_equivalence_scope"]
    ax[3].bar([0, 1], [h8["n_in_scope"], h8["n_excluded"]],
              color=[GREEN, LIGHT], width=0.55)
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["in scope", "excluded"])
    ax[3].set_ylabel("programs")
    ax[3].set_ylim(0, h8["n_total"] * 1.2)
    for i, v in enumerate([h8["n_in_scope"], h8["n_excluded"]]):
        ax[3].text(i, v + 0.06, str(v), ha="center", fontsize=9)
    tag(ax[3], "D")

    return save(fig, os.path.join(OUT, "panel_04_resolution.png"))


if __name__ == "__main__":
    for fn in (panel_01, panel_02, panel_03, panel_04):
        print(fn())
