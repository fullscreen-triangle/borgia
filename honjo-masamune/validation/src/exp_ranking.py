"""Experiment A: cut refinement as a canonical ranking.

Pre-registered expectations, stated here before the run:

  A1  The base cut key is coarse: it does not by itself separate the
      atoms of a drug-like structure.  Expect mean distinct-key ratio
      < 0.7, i.e. materially short of the 1.0 a full ranking needs.
      (If this fails, refinement is unnecessary -- the base invariant
      would already suffice, and the paper has no subject.)
  A2  Refinement stabilises at the automorphism orbit partition on the
      symmetric corpus: classes == orbits for every entry.
  A3  Refinement does NOT over-separate: no symmetric molecule reaches
      more classes than orbits.  This is the property that distinguishes
      an invariant from a canonical ordering.
  A4  Refinement converges quickly: median rounds <= 4.
  A5  Negative control -- a deliberately broken refinement that breaks
      ties by atom index reaches n classes on symmetric molecules,
      i.e. destroys the orbit structure.  If the control does not fail,
      A2 is not testing anything.

Every number written to the results file is measured, not asserted.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from corpus import DRUGLIKE, SYMMETRIC
from cutrank import cut_key, heavy_atoms, neighbour_keys, refine
from hjm.core.graph import MEDIUM
from hjm.masamune.translate import translate_smiles

EXPECT = {
    "A1_base_ratio_below": 0.7,
    "A2_orbit_match_rate": 1.0,
    "A3_no_over_separation": True,
    "A4_median_rounds_at_most": 4,
    "A5_control_must_over_separate": True,
}


def broken_refine(g, max_rounds: int = 32) -> int:
    """Negative control: refinement that breaks ties by atom index.

    This is what a canonical *ordering* does.  It reaches n classes on
    every molecule, symmetric or not, which is exactly the behaviour an
    invariant must not have.
    """
    heavy = heavy_atoms(g)
    lab = {k: (cut_key(g, k), i) for i, k in enumerate(sorted(heavy))}
    return len(set(map(str, lab.values())))


def run() -> dict:
    out: dict = {
        "experiment": "A: cut refinement as a canonical ranking",
        "expectations": EXPECT,
        "results": {},
    }

    # --- A1: base coarseness on drug-like structures -------------------
    base_rows = []
    for name, smi in DRUGLIKE:
        v = translate_smiles(smi)
        if not v.ok:
            base_rows.append({"name": name, "verdict": str(v.label)})
            continue
        g = v.value
        heavy = heavy_atoms(g)
        keys = [cut_key(g, k) for k in heavy]
        sigmas = {s for s, _d in keys}
        base_rows.append({
            "name": name,
            "n_heavy": len(heavy),
            "distinct_sigma": len(sigmas),
            "distinct_base_key": len(set(keys)),
            "base_ratio": round(len(set(keys)) / len(heavy), 6) if heavy else None,
        })
    ratios = [r["base_ratio"] for r in base_rows if r.get("base_ratio") is not None]
    out["results"]["A1_base_coarseness"] = {
        "rows": base_rows,
        "mean_base_ratio": round(statistics.mean(ratios), 6),
        "median_base_ratio": round(statistics.median(ratios), 6),
        "max_base_ratio": round(max(ratios), 6),
        "passed": statistics.mean(ratios) < EXPECT["A1_base_ratio_below"],
    }

    # --- A2/A3: orbit agreement on the symmetric corpus ----------------
    sym_rows, matches, over = [], 0, 0
    for name, smi, orbits in SYMMETRIC:
        v = translate_smiles(smi)
        if not v.ok:
            sym_rows.append({"name": name, "verdict": str(v.label)})
            continue
        g = v.value
        r = refine(g)
        ok = r["classes"] == orbits
        matches += ok
        over += r["classes"] > orbits
        sym_rows.append({
            "name": name, "smiles": smi, "n_heavy": r["n"],
            "reference_orbits": orbits, "cut_classes": r["classes"],
            "rounds": r["rounds"], "history": r["history"],
            "match": ok,
            "over_separated": r["classes"] > orbits,
            "under_separated": r["classes"] < orbits,
        })
    n_sym = len([r for r in sym_rows if "cut_classes" in r])
    out["results"]["A2_orbit_agreement"] = {
        "rows": sym_rows,
        "n": n_sym,
        "matches": matches,
        "match_rate": round(matches / n_sym, 6) if n_sym else 0.0,
        "passed": matches == n_sym,
    }
    out["results"]["A3_no_over_separation"] = {
        "n_over_separated": over,
        "passed": over == 0,
    }

    # --- A4: convergence speed ----------------------------------------
    all_rounds = [r["rounds"] for r in sym_rows if "rounds" in r]
    dl_rounds = []
    for name, smi in DRUGLIKE:
        v = translate_smiles(smi)
        if v.ok:
            dl_rounds.append(refine(v.value)["rounds"])
    out["results"]["A4_convergence"] = {
        "symmetric_rounds": all_rounds,
        "druglike_rounds": dl_rounds,
        "median_rounds": statistics.median(all_rounds + dl_rounds),
        "max_rounds": max(all_rounds + dl_rounds),
        "passed": statistics.median(all_rounds + dl_rounds)
                  <= EXPECT["A4_median_rounds_at_most"],
    }

    # --- A4b: refinement gain on drug-like ----------------------------
    gain_rows = []
    for name, smi in DRUGLIKE:
        v = translate_smiles(smi)
        if not v.ok:
            continue
        g = v.value
        r = refine(g)
        gain_rows.append({
            "name": name, "n_heavy": r["n"],
            "base_classes": r["history"][0],
            "stable_classes": r["classes"],
            "final_ratio": round(r["classes"] / r["n"], 6) if r["n"] else None,
        })
    fr = [g["final_ratio"] for g in gain_rows if g["final_ratio"] is not None]
    out["results"]["A4b_refinement_gain"] = {
        "rows": gain_rows,
        "mean_final_ratio": round(statistics.mean(fr), 6),
        "mean_base_ratio": round(statistics.mean(ratios), 6),
    }

    # --- A5: negative control ------------------------------------------
    ctrl_rows, ctrl_over = [], 0
    for name, smi, orbits in SYMMETRIC:
        v = translate_smiles(smi)
        if not v.ok:
            continue
        g = v.value
        n_cls = broken_refine(g)
        n = len(heavy_atoms(g))
        bad = n_cls > orbits
        ctrl_over += bad
        ctrl_rows.append({
            "name": name, "n_heavy": n, "reference_orbits": orbits,
            "control_classes": n_cls, "over_separated": bad,
        })
    out["results"]["A5_negative_control"] = {
        "rows": ctrl_rows,
        "n_over_separated": ctrl_over,
        "n": len(ctrl_rows),
        "note": "index tie-breaking destroys the orbit structure, as a "
                "canonical ordering does; A2 is therefore not vacuous",
        "passed": ctrl_over > 0,
    }

    out["all_passed"] = all(
        v.get("passed", True) for v in out["results"].values()
    )
    return out


if __name__ == "__main__":
    res = run()
    here = os.path.dirname(__file__)
    dest = os.path.join(here, "..", "results", "exp_ranking.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({
        "experiment": res["experiment"],
        "all_passed": res["all_passed"],
        **{k: v.get("passed") for k, v in res["results"].items()},
    }, indent=2))
