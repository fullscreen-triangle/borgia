"""Experiment M: what a verdict-carrying converter actually measures.

The manuscript claims several things that can be measured directly
against the reference implementation rather than asserted.  Each is
stated here before the run.

  M1  SUPPLIED FRACTION BY FORMAT.  Translate molecules from each
      supported format and measure phi, the fraction of a graph's own
      elements that were SUPPLIED rather than STATED.  Expect the SMILES
      mean to be substantially above zero -- a valence model supplies
      hydrogens the string does not state -- and make no prediction of
      the exact value.
  M2  CAPABILITY CONTAINMENT.  A request is refusable before any record
      is read whenever the required feature set is not contained in the
      format's declared capability.  Expect the static refusal to agree
      with the post-read outcome on every (format, request) pair: a
      static refusal that a read would have satisfied would be unsound.
  M3  VERDICT COVERAGE.  The verdict labels must each be reachable, and
      no failure label may carry a value.  Expect zero value-bearing
      failures.
  M4  FORMAT DISAGREEMENT.  The same molecule expressed two faithful
      ways may yield graphs that differ while an identity-level
      comparison reports agreement.  Measure how often, and by what
      mechanism.  Expect a nonzero count; make no prediction of size.
  M5  MEDIUM-EDGE CONTROL (defect D2).  phi computed WITH medium edges
      in the denominator is dominated by an artefact of the target
      representation.  Measure both and report the difference, which is
      the quantity that made D2 visible.

Every number written to the results file is measured.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from corpus import DRUGLIKE, SYMMETRIC
from hjm.core.chem import Z_OF
from hjm.core.graph import Prov
from hjm.core.verdict import VALUE_BEARING, Label
# NOTE: hjm.masamune re-exports a FUNCTION named `capability`,
# which shadows the submodule of the same name.  Import the
# names directly rather than the module.
from hjm.masamune.capability import (CAPABILITY, capability,
                                     known_format, missing)
from hjm.masamune.translate import translate, translate_smiles, translate_xyz

EXPECT = {
    "M1_smiles_phi_above_zero": True,
    "M2_static_agrees_with_read": True,
    "M3_no_value_bearing_failure": True,
    "M4_disagreement_count_nonzero": True,
    "M5_medium_inflates_phi": True,
}

SYM_OF = {v: k for k, v in Z_OF.items()}

ALL = [(n, s) for n, s in DRUGLIKE] + [(n, s) for n, s, _o in SYMMETRIC]


def phi_with_medium(g) -> float:
    """phi computed the WRONG way: medium edges in the denominator.

    This is defect D2.  No source states a medium edge, so every one is
    SUPPLIED, and including them makes phi a property of the target
    representation rather than of the source record.
    """
    elems = list(g.atoms.values()) + list(g.contacts.values())
    if not elems:
        return 0.0
    return sum(1 for e in elems if e.prov is Prov.SUPPLIED) / len(elems)


def as_xyz(g, name: str) -> str:
    """Emit a coordinate-only record: atoms and positions, no bond block.

    Coordinates are placed on a line; the reader never derives
    connectivity from them, so their values do not affect what is
    measured -- only their presence does.
    """
    ats = list(g.atoms.values())
    out = [str(len(ats)), name]
    for i, a in enumerate(ats):
        out.append(f"{SYM_OF.get(a.z, 'C')} {i * 1.5:.3f} 0.000 0.000")
    return "\n".join(out)


def formula(g):
    return tuple(sorted(Counter(a.z for a in g.atoms.values()).items()))


def deloc_edges(g):
    return sum(1 for c in g.contacts.values()
               if not c.is_medium_edge and c.deloc_id is not None)


def cell_multiset(g):
    return tuple(sorted(c.cells for c in g.contacts.values()
                        if not c.is_medium_edge))


def run() -> dict:
    out: dict = {
        "experiment": "M: verdict-carrying translation, measured",
        "expectations": EXPECT,
        "results": {},
    }

    # --- M1 / M5: supplied fraction ------------------------------------
    rows = []
    for name, smi in ALL:
        v = translate_smiles(smi, required={"element", "connectivity"})
        if not v.ok:
            rows.append({"molecule": name, "smiles_label": str(v.label)})
            continue
        g = v.value
        vx = translate_xyz(as_xyz(g, name), required={"element", "coords3d"})
        rows.append({
            "molecule": name,
            "n_atoms": len(g.atoms),
            "n_heavy": sum(1 for a in g.atoms.values() if a.z != 1),
            "smiles_phi": round(g.supplied_fraction(), 6),
            "smiles_phi_with_medium": round(phi_with_medium(g), 6),
            "smiles_label": str(v.label),
            "xyz_label": str(vx.label),
            "xyz_phi": vx.payload.get("supplied_fraction"),
            "xyz_absent": vx.payload.get("absent_features"),
        })

    ok = [r for r in rows if "smiles_phi" in r]
    sphi = [r["smiles_phi"] for r in ok]
    sphim = [r["smiles_phi_with_medium"] for r in ok]

    out["results"]["M1_supplied_fraction"] = {
        "n_structures": len(ok),
        "smiles_mean_phi": round(statistics.mean(sphi), 6),
        "smiles_median_phi": round(statistics.median(sphi), 6),
        "smiles_min_phi": round(min(sphi), 6),
        "smiles_max_phi": round(max(sphi), 6),
        "smiles_stdev_phi": round(statistics.pstdev(sphi), 6),
        "n_exactly_zero": sum(1 for x in sphi if x == 0.0),
        "rows": rows,
        "note": "phi excludes medium edges from the denominator "
                "(defect D2); M5 reports the uncorrected quantity",
        "passed": statistics.mean(sphi) > 0.0,
    }

    out["results"]["M5_medium_edge_defect"] = {
        "mean_phi_corrected": round(statistics.mean(sphi), 6),
        "mean_phi_with_medium": round(statistics.mean(sphim), 6),
        "mean_inflation": round(statistics.mean(sphim)
                                - statistics.mean(sphi), 6),
        "max_inflation": round(max(b - a for a, b in zip(sphi, sphim)), 6),
        "min_inflation": round(min(b - a for a, b in zip(sphi, sphim)), 6),
        "note": "including medium edges makes phi a property of the "
                "target representation rather than of the source record; "
                "this is the quantity that made D2 visible",
        "passed": statistics.mean(sphim) > statistics.mean(sphi),
    }

    # --- M2: capability containment ------------------------------------
    REQUESTS = [
        {"element"},
        {"element", "connectivity"},
        {"element", "connectivity", "cellcount"},
        {"element", "connectivity", "stereo"},
        {"element", "coords3d"},
        {"element", "connectivity", "coords3d"},
        {"element", "conformer"},
        {"provenance"},
    ]
    probe_smi = ALL[0][1]
    probe_xyz = "1\nprobe\nC 0.000 0.000 0.000"
    cap_rows, agree = [], 0
    for fmt in ("smiles", "xyz", "inchi"):
        for req in REQUESTS:
            static_refuse = bool(missing(fmt, req))
            text = probe_smi if fmt == "smiles" else probe_xyz
            v = translate(fmt, text, required=set(req))
            read_refuse = str(v.label) == "unsupported"
            consistent = static_refuse == read_refuse
            agree += consistent
            cap_rows.append({
                "format": fmt,
                "required": sorted(req),
                "n_required": len(req),
                "declared": sorted(capability(fmt)),
                "n_declared": len(capability(fmt)),
                "missing": sorted(missing(fmt, req)),
                "static_refusal": static_refuse,
                "label_after_read": str(v.label),
                "consistent": consistent,
            })
    out["results"]["M2_capability_containment"] = {
        "rows": cap_rows,
        "n_pairs": len(cap_rows),
        "n_consistent": agree,
        "agreement": round(agree / len(cap_rows), 6),
        "n_refused_statically": sum(1 for r in cap_rows if r["static_refusal"]),
        "note": "a static refusal costs no record access; what is checked "
                "is that it never refuses what a read would have satisfied",
        "passed": agree == len(cap_rows),
    }

    # --- M3: verdict coverage ------------------------------------------
    probes = [
        ("well-formed smiles", lambda: translate_smiles("CCO")),
        ("malformed smiles", lambda: translate_smiles("C(((")),
        ("empty smiles", lambda: translate_smiles("")),
        ("stereo token, no descriptor", lambda: translate_smiles(
            "F/C=C/F", required={"element", "connectivity", "stereo"})),
        ("xyz, no bond block", lambda: translate_xyz(
            "2\nx\nC 0 0 0\nO 1.4 0 0")),
        ("format with no reader", lambda: translate(
            "inchi", "InChI=1S/CH4/h1H4", required={"element"})),
        ("unknown format", lambda: translate(
            "mol2", "x", required={"element"})),
        ("outside organic subset", lambda: translate_smiles("Zz")),
    ]
    seen, vrows, bad = Counter(), [], 0
    for name, fn in probes:
        v = fn()
        lab = str(v.label)
        carries = v.value is not None
        should = v.label in VALUE_BEARING
        if carries and not should:
            bad += 1
        seen[lab] += 1
        vrows.append({"probe": name, "label": lab,
                      "carries_value": carries,
                      "value_bearing_label": should,
                      "sound": not (carries and not should)})
    out["results"]["M3_verdict_coverage"] = {
        "rows": vrows,
        "labels_realised": sorted(seen),
        "label_counts": dict(seen),
        "n_labels_realised": len(seen),
        "n_labels_defined": len(list(Label)),
        "value_bearing_failures": bad,
        "note": "only a value-bearing label may carry a value; a failure "
                "carrying one is the conflation the design forbids",
        "passed": bad == 0,
    }

    # --- M4: cross-format disagreement ---------------------------------
    # An aromatic ring written lower-case becomes a delocalised system;
    # the same ring written in an alternating Kekule assignment becomes
    # per-contact cell counts of 1 and 2.  Both readings are faithful:
    # each states what its source states.  The graphs differ.
    KEKULE = [
        ("benzene",       "c1ccccc1",        "C1=CC=CC=C1"),
        ("toluene",       "Cc1ccccc1",       "CC1=CC=CC=C1"),
        ("pyridine",      "c1ccncc1",        "C1=CC=NC=C1"),
        ("naphthalene",   "c1ccc2ccccc2c1",  "C1=CC2=CC=CC=C2C=C1"),
        ("phenol",        "Oc1ccccc1",       "OC1=CC=CC=C1"),
        ("aniline",       "Nc1ccccc1",       "NC1=CC=CC=C1"),
        ("p-xylene",      "Cc1ccc(C)cc1",    "CC1=CC=C(C)C=C1"),
        ("pyrazine",      "c1cnccn1",        "C1=CN=CC=N1"),
        ("chlorobenzene", "Clc1ccccc1",      "ClC1=CC=CC=C1"),
        ("styrene",       "C=Cc1ccccc1",     "C=CC1=CC=CC=C1"),
    ]
    drows = []
    for name, arom, kek in KEKULE:
        va, vk = translate_smiles(arom), translate_smiles(kek)
        if not (va.ok and vk.ok):
            drows.append({"molecule": name, "error": "translation failed"})
            continue
        ga, gk = va.value, vk.value
        same_formula = formula(ga) == formula(gk)
        same_cells = cell_multiset(ga) == cell_multiset(gk)
        drows.append({
            "molecule": name,
            "same_formula": same_formula,
            "same_cell_multiset": same_cells,
            "graphs_differ_but_formula_agrees": (not same_cells) and same_formula,
            "deloc_edges_aromatic": deloc_edges(ga),
            "deloc_edges_kekule": deloc_edges(gk),
            "cells_aromatic": list(cell_multiset(ga)),
            "cells_kekule": list(cell_multiset(gk)),
            "phi_aromatic": round(ga.supplied_fraction(), 6),
            "phi_kekule": round(gk.supplied_fraction(), 6),
        })
    good = [r for r in drows if "same_formula" in r]
    undetected = sum(1 for r in good if r["graphs_differ_but_formula_agrees"])
    out["results"]["M4_format_disagreement"] = {
        "rows": drows,
        "n_pairs": len(good),
        "n_graphs_differ": sum(1 for r in good if not r["same_cell_multiset"]),
        "n_formula_agrees": sum(1 for r in good if r["same_formula"]),
        "n_undetected_by_identity": undetected,
        "fraction_undetected": round(undetected / len(good), 6) if good else 0.0,
        "note": "both translations are faithful.  A formula-level or "
                "canonical-identity comparison reports agreement, so the "
                "disagreement is invisible to it",
        "passed": undetected > 0,
    }

    out["all_passed"] = all(v.get("passed", True)
                            for v in out["results"].values())
    return out


if __name__ == "__main__":
    res = run()
    dest = os.path.join(os.path.dirname(__file__), "..", "results",
                        "exp_masamune.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({
        "experiment": res["experiment"],
        "all_passed": res["all_passed"],
        **{k: v.get("passed") for k, v in res["results"].items()},
    }, indent=2))
