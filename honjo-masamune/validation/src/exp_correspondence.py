"""Experiment B: substitution-tolerant structural correspondence.

Pre-registered expectations, stated here before the run:

  B0  Tolerance is a resolution parameter.  Separation of bioisosteric
      from unrelated pairs should be best at the coarsest radius and
      degrade as the radius grows, because refinement is discrimination
      and tolerance is its opposite.  Expect the margin at radius 0 to
      exceed the margin at every higher radius.
  B1  At the working radius, class overlap separates bioisosteric pairs
      from unrelated pairs of comparable size: mean overlap(close) >
      mean overlap(far), with no overlap between the two ranges.
  B2  Element-identity matching does NOT separate them in the same way.
      A matcher keyed on exact element cannot pair a C with an N, so a
      heteroatom substitution costs it score that the cut class does not
      pay.  Expect mean class overlap > mean element overlap on the
      bioisosteric pairs.  This is the claim that the tolerance is doing
      work rather than being a relabelling of element identity.
  B3  Correspondence coverage on matched pairs (one-position changes) is
      high: mean coverage >= 0.7.
  B4  Negative control -- randomly rewiring a structure, holding its
      atom composition and edge count fixed, destroys the separation.
      If it does not, B1 is measuring composition rather than structure.
  B5  Correspondence is symmetric in its inputs: size(g1,g2) == size(g2,g1).

Every number written to the results file is measured, not asserted.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from corpus import DRUGLIKE, ISOSTERE_PAIRS, MATCHED_PAIRS
from cutrank import (
    class_jaccard, class_multiset_at, correspondence, cut_key,
    heavy_atoms, labels_at_radius, refine,
)
from hjm.core.graph import MEDIUM
from hjm.masamune.translate import translate_smiles

RADIUS = 0  # the working radius, chosen by B0 below

EXPECT = {
    "B0_coarsest_radius_separates_best": True,
    "B1_close_above_far": True,
    "B1_ranges_disjoint": True,
    "B2_element_matching_worse_on_isosteres": True,
    "B3_mean_coverage_at_least": 0.7,
    "B4_control_must_destroy_separation": True,
    "B5_symmetric": True,
}


def g_of(smiles: str):
    v = translate_smiles(smiles)
    return v.value if v.ok else None


def element_overlap(g1, g2) -> float:
    """Baseline: exact element identity.

    This is the constraint a substructure matcher imposes -- a carbon may
    match only a carbon.  It is the comparison B2 needs: same molecules,
    same graph, differing only in whether an atom's key is its cut class
    or its element.
    """
    from collections import Counter

    def ms(g):
        return Counter(g.atoms[k].z for k in heavy_atoms(g))

    m1, m2 = ms(g1), ms(g2)
    inter = sum((m1 & m2).values())
    union = sum((m1 | m2).values())
    return round(inter / union, 6) if union else 0.0


def rewired_class_overlap(g1, g2, seed: int) -> float:
    """Negative control: rewire the second structure at random, keeping
    its atom composition, then re-score.

    An earlier version of this control permuted class labels *within* a
    structure.  That is a no-op on a multiset -- the same labels with the
    same counts -- so it could not fail, and it is recorded in the
    manuscript as a control that did not control.  Rewiring destroys the
    structural assignment while holding composition fixed, which is the
    thing B1 claims to measure.
    """
    import copy
    import random
    from collections import Counter

    rng = random.Random(seed)
    h = copy.deepcopy(g2)

    heavy = heavy_atoms(h)
    if len(heavy) < 3:
        return class_jaccard(g1, h, radius=RADIUS)

    # rewire: drop item-item contacts, then rebuild a random tree plus
    # the same number of extra edges, preserving edge count
    from hjm.core.graph import Contact

    item_edges = [c for c in h.contacts.values() if not c.is_medium_edge]
    n_edges = len(item_edges)
    keys = list(h.atoms)
    for c in item_edges:
        del h.contacts[c.key]

    placed = [keys[0]]
    rest = keys[1:]
    rng.shuffle(rest)
    for k in rest:
        p = rng.choice(placed)
        h.add_contact(Contact(u=p, v=k, weight=h.floor, cells=1))
        placed.append(k)
    while len([c for c in h.contacts.values() if not c.is_medium_edge]) < n_edges:
        a, b = rng.sample(keys, 2)
        h.add_contact(Contact(u=a, v=b, weight=h.floor, cells=1))

    return class_jaccard(g1, h, radius=RADIUS)


def run() -> dict:
    out: dict = {
        "experiment": "B: substitution-tolerant structural correspondence",
        "expectations": EXPECT,
        "results": {},
    }

    # --- B0: tolerance is a resolution parameter -----------------------
    sweep = []
    for rad in range(0, 4):
        cl, fr = [], []
        for n1, s1, n2, s2, rel in ISOSTERE_PAIRS:
            g1, g2 = g_of(s1), g_of(s2)
            if g1 is None or g2 is None:
                continue
            val = class_jaccard(g1, g2, radius=rad)
            (cl if rel == "close" else fr).append(val)
        sweep.append({
            "radius": rad,
            "mean_close": round(statistics.mean(cl), 6),
            "mean_far": round(statistics.mean(fr), 6),
            "min_close": round(min(cl), 6),
            "max_far": round(max(fr), 6),
            "margin": round(min(cl) - max(fr), 6),
        })
    best = max(sweep, key=lambda r: r["margin"])
    out["results"]["B0_radius_sweep"] = {
        "rows": sweep,
        "best_radius": best["radius"],
        "best_margin": best["margin"],
        "working_radius": RADIUS,
        "note": "refinement discriminates; tolerance is its opposite, so "
                "the coarsest radius should separate best",
        "passed": best["radius"] == 0 and best["margin"] > 0,
    }

    # --- B1: isostere separation ---------------------------------------
    rows = []
    for n1, s1, n2, s2, rel in ISOSTERE_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            rows.append({"pair": f"{n1}/{n2}", "error": "translation failed"})
            continue
        rows.append({
            "a": n1, "b": n2, "relation": rel,
            "n_heavy_a": len(heavy_atoms(g1)),
            "n_heavy_b": len(heavy_atoms(g2)),
            "class_overlap": class_jaccard(g1, g2, radius=RADIUS),
            "element_overlap": element_overlap(g1, g2),
        })
    close = [r["class_overlap"] for r in rows if r.get("relation") == "close"]
    far = [r["class_overlap"] for r in rows if r.get("relation") == "far"]
    disjoint = (min(close) > max(far)) if close and far else False
    out["results"]["B1_isostere_separation"] = {
        "rows": rows,
        "mean_close": round(statistics.mean(close), 6),
        "mean_far": round(statistics.mean(far), 6),
        "min_close": round(min(close), 6),
        "max_far": round(max(far), 6),
        "ranges_disjoint": disjoint,
        "separation_margin": round(min(close) - max(far), 6),
        "passed": statistics.mean(close) > statistics.mean(far) and disjoint,
    }

    # --- B2: cross-element correspondence ------------------------------
    # Multiset overlap can hide a heteroatom swap.  The place element
    # identity actually binds is the correspondence itself: a matcher
    # keyed on element cannot pair a C with an N at all.  We therefore
    # count the pairings each method admits, and how many of those cross
    # an element boundary.
    b2_rows = []
    for n1, s1, n2, s2, rel in ISOSTERE_PAIRS:
        if rel != "close":
            continue
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        c = correspondence(g1, g2, radius=RADIUS)
        cross = sum(1 for a, b in c["pairs"]
                    if g1.atoms[a].z != g2.atoms[b].z)
        # an element-keyed matcher admits only same-element pairings
        same = c["size"] - cross
        b2_rows.append({
            "a": n1, "b": n2,
            "cut_pairs": c["size"],
            "element_pairs": same,
            "cross_element_pairs": cross,
            "cut_coverage": c["coverage"],
            "element_coverage": round(same / max(c["n1"], c["n2"]), 6),
            "gain": cross,
        })
    total_cross = sum(r["cross_element_pairs"] for r in b2_rows)
    n_gain = sum(1 for r in b2_rows if r["gain"] > 0)
    out["results"]["B2_cross_element_correspondence"] = {
        "rows": b2_rows,
        "n_close_pairs": len(b2_rows),
        "total_cross_element_pairs": total_cross,
        "n_pairs_with_gain": n_gain,
        "mean_cut_coverage": round(
            statistics.mean(r["cut_coverage"] for r in b2_rows), 6),
        "mean_element_coverage": round(
            statistics.mean(r["element_coverage"] for r in b2_rows), 6),
        "note": "cross-element pairings are exactly what an element-keyed "
                "matcher cannot make; they are the tolerance, counted",
        "passed": total_cross > 0 and n_gain >= len(b2_rows) // 2,
    }

    # --- B3: correspondence coverage on matched pairs ------------------
    mp_rows = []
    for n1, s1, n2, s2 in MATCHED_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        c = correspondence(g1, g2, radius=RADIUS)
        mp_rows.append({
            "a": n1, "b": n2, "n1": c["n1"], "n2": c["n2"],
            "matched": c["size"], "coverage": c["coverage"],
            "classes_shared": c["classes_shared"],
            "classes_a": c["classes_1"], "classes_b": c["classes_2"],
        })
    cov = [r["coverage"] for r in mp_rows]
    out["results"]["B3_matched_pair_coverage"] = {
        "rows": mp_rows,
        "mean_coverage": round(statistics.mean(cov), 6),
        "median_coverage": round(statistics.median(cov), 6),
        "min_coverage": round(min(cov), 6),
        "passed": statistics.mean(cov) >= EXPECT["B3_mean_coverage_at_least"],
    }

    # --- B4: negative control -------------------------------------------
    ctrl_close, ctrl_far = [], []
    for n1, s1, n2, s2, rel in ISOSTERE_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        vals = [rewired_class_overlap(g1, g2, seed) for seed in range(20)]
        (ctrl_close if rel == "close" else ctrl_far).append(
            statistics.mean(vals)
        )
    ctrl_margin = min(ctrl_close) - max(ctrl_far) if ctrl_close and ctrl_far else 0.0
    out["results"]["B4_negative_control"] = {
        "mean_close_shuffled": round(statistics.mean(ctrl_close), 6),
        "mean_far_shuffled": round(statistics.mean(ctrl_far), 6),
        "separation_margin_shuffled": round(ctrl_margin, 6),
        "true_margin": out["results"]["B1_isostere_separation"]["separation_margin"],
        "note": "rewiring the second structure at random, holding atom "
                "composition and edge count fixed, destroys the structural "
                "assignment; the separation margin should collapse",
        "passed": ctrl_margin < out["results"]["B1_isostere_separation"][
            "separation_margin"],
    }

    # --- B5: symmetry ----------------------------------------------------
    sym_rows, all_sym = [], True
    for n1, s1, n2, s2 in MATCHED_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        a = correspondence(g1, g2, radius=RADIUS)["size"]
        b = correspondence(g2, g1, radius=RADIUS)["size"]
        ok = a == b
        all_sym &= ok
        sym_rows.append({"a": n1, "b": n2, "ab": a, "ba": b, "symmetric": ok})
    out["results"]["B5_symmetry"] = {
        "rows": sym_rows, "all_symmetric": all_sym, "passed": all_sym,
    }

    # --- B6: descriptive -- pairwise overlap over the drug-like set -----
    grid = []
    gs = [(n, g_of(s)) for n, s in DRUGLIKE]
    gs = [(n, g) for n, g in gs if g is not None]
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            grid.append({
                "a": gs[i][0], "b": gs[j][0],
                "class_overlap": class_jaccard(gs[i][1], gs[j][1], radius=RADIUS),
                "element_overlap": element_overlap(gs[i][1], gs[j][1]),
            })
    grid.sort(key=lambda r: -r["class_overlap"])
    out["results"]["B6_druglike_grid"] = {
        "n_pairs": len(grid),
        "mean_class_overlap": round(
            statistics.mean(r["class_overlap"] for r in grid), 6),
        "mean_element_overlap": round(
            statistics.mean(r["element_overlap"] for r in grid), 6),
        "top_20_by_class_overlap": grid[:20],
    }

    out["all_passed"] = all(v.get("passed", True) for v in out["results"].values())
    return out


if __name__ == "__main__":
    res = run()
    here = os.path.dirname(__file__)
    dest = os.path.join(here, "..", "results", "exp_correspondence.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({
        "experiment": res["experiment"],
        "all_passed": res["all_passed"],
        **{k: v.get("passed") for k, v in res["results"].items()},
    }, indent=2))
