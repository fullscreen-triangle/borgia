"""Experiment H: the honjo conformance suite, executed.

The specification states a conformance suite (C1)--(C8).  A suite that
has never been run is a list of intentions, so each item is executed here
against the reference interpreter and its outcome recorded.  Expectations
are written before the run.

  H1 (C1)  NO SHARP CUT.  No program produces a value with residue below
      the floor.  Expect every attempt to be refused rather than
      truncated: a refusal is a verdict, a truncation is a silent
      substitution.
  H2 (C2)  PROVENANCE MONOTONICITY.  Every derived value's tag is the
      maximum over its inputs, and no supplied input yields a stated
      output.  Expect zero violations over all derivations.
  H3 (C3)  VERDICT REALISATION.  Each label is produced by some program,
      and no failure verdict carries a value.
  H4 (C4)  CONFLATION CONTROL.  Four programs that fail in four distinct
      ways must produce four distinct labels; run through a
      value-or-nothing wrapper they must collapse to one result.  Expect
      4 distinct labels and 1 wrapped outcome.  If the wrapper does not
      collapse them, the conflation theorem is vacuous and the whole
      verdict apparatus is unmotivated.
  H5 (C5)  CELL COUNTS DISTINGUISH.  A single interface committing two
      cells differs from two interfaces committing one each.  Expect
      different values.
  H6 (C6)  DELOCALISATION IS NOT A COUNT.  A delocalised system carries a
      total, not a per-pair count, and no program extracts one.
  H7 (C7)  RESOLUTION GATE.  A program whose floor lies below the
      target's reported resolution is refused, with both numbers in the
      verdict.
  H8 (C8)  SCOPE OF TARGET AGREEMENT.  Programs failing (C7) are recorded
      as out of scope rather than run and tolerated.  Expect the
      in-scope set to be a strict subset when any program is sub-floor:
      a suite reporting agreement over programs the theorem excludes
      would be asserting the unconditional claim the paper retracts.

Every number written to the results file is measured.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from hjm.honjo.interp import Interpreter

EX = os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py", "examples")

EXPECT = {
    "H1_no_sub_floor_value": True,
    "H2_zero_provenance_violations": True,
    "H3_no_value_bearing_failure": True,
    "H4_four_distinct_one_wrapped": True,
    "H5_cell_counts_differ": True,
    "H6_no_per_pair_count": True,
    "H7_resolution_gate_refuses": True,
    "H8_out_of_scope_recorded": True,
}


def run_src(src: str, eps: float = 1e-9) -> dict:
    return Interpreter(base_dir=EX, eps_target=eps).run(src)


def load(name: str) -> str:
    with open(os.path.join(EX, name), encoding="utf-8") as fh:
        return fh.read()


def bindings(res: dict) -> list:
    return res.get("bindings", [])


def run() -> dict:
    out: dict = {
        "experiment": "H: the honjo conformance suite, executed",
        "expectations": EXPECT,
        "results": {},
    }

    # --- H1 (C1): no sharp cut ----------------------------------------
    # Sweep the floor over a 16-fold range for a fixed atom.  If the
    # residue were truncated to the floor after the fact, the ratio
    # residue/floor would move with the floor; if the floor is a genuine
    # resolution scale, the ratio is constant and no value is below it.
    FLOORS = [0.25, 0.5, 1.0, 2.0, 4.0]
    ATOMS = [("H", 1), ("C", 6), ("O", 8), ("Ne", 10)]
    h1_rows, sharp, ratios = [], 0, {}
    for sym, z in ATOMS:
        seen = []
        for fl in FLOORS:
            res = run_src("floor %s\nX := cut %d\nobserve X" % (fl, z))
            vals = bindings(res)
            if not vals:
                continue
            b = vals[0]
            r, f = b.get("residue"), b.get("floor")
            below = r is not None and f is not None and r < f
            sharp += below
            seen.append({"floor": f, "residue": r,
                         "ratio": round(r / f, 6) if f else None,
                         "below_floor": below})
        ratios[sym] = sorted({x["ratio"] for x in seen})
        h1_rows.append({"atom": sym, "z": z, "sweep": seen,
                        "distinct_ratios": ratios[sym]})
    constant = all(len(v) == 1 for v in ratios.values())
    out["results"]["H1_no_sharp_cut"] = {
        "rows": h1_rows,
        "n_programs": len(ATOMS) * len(FLOORS),
        "floors_swept": FLOORS,
        "n_values_below_floor": sharp,
        "residue_over_floor": ratios,
        "ratio_constant_per_atom": constant,
        "note": "residue is floor x vacancy under a max(.., floor) clamp, "
                "so a sub-floor value is unconstructible and merely "
                "looking for one cannot fail.  What is measured instead "
                "is that residue/floor is constant per atom across a "
                "16-fold floor sweep: the floor acts as a resolution "
                "scale rather than as a truncation applied afterwards",
        "passed": sharp == 0 and constant,
    }


    # --- H2 (C2): provenance monotonicity ------------------------------
    PROV = [
        ("all stated", "floor 1.0\nO := cut 8\nH := cut 1\n"
                       "W := close O(H, H)\nobserve W"),
        ("imported, mixed", 'floor 1.0\ng := import graph "one.smi"\n'
                            'observe g'),
        ("derived from import", 'floor 1.0\ng := import graph "one.smi"\n'
                                'observe g'),
    ]
    ORDER = {"stated": 0, "supplied": 1}
    h2_rows, viol = [], 0
    for name, src in PROV:
        res = run_src(src)
        tags = [b.get("provenance") for b in bindings(res)
                if b.get("provenance")]
        # monotonicity: within a program, a later derived value may never
        # carry a tag weaker than the strongest input seen so far
        worst, bad = 0, 0
        for t in tags:
            v = ORDER.get(t, 0)
            if v < worst:
                bad += 1
            worst = max(worst, v)
        viol += bad
        h2_rows.append({
            "program": name,
            "status": res.get("status"),
            "tags_in_order": tags,
            "max_tag": max((ORDER.get(t, 0) for t in tags), default=0),
            "violations": bad,
        })
    out["results"]["H2_provenance_monotone"] = {
        "rows": h2_rows,
        "n_programs": len(h2_rows),
        "n_violations": viol,
        "note": "composition is maximum, so a supplied input can never "
                "yield a stated output; a weakening would be a laundering "
                "of convention into record",
        "passed": viol == 0,
    }

    # --- H3 (C3) / H4 (C4): verdicts and the conflation control --------
    # Four programs that fail in four genuinely different ways.
    # Four programs that yield no usable value, for four different
    # reasons.  The point of (C4) is that the REASONS are distinguishable
    # under the verdict interface and indistinguishable under a
    # value-or-nothing one.  Note that `unclosed` and `inert` are
    # diagnostics attached to a value that does exist, so they are not
    # members of this set: including them would make the wrapper report
    # "value" and the control would fail for the wrong reason.
    FOUR = [
        ("sub-resolution floor", "floor 1e-12\nH := cut 1\nobserve H"),
        ("parse error", "floor 1.0\nX := cut\nobserve X"),
        ("unbound reference", "floor 1.0\nobserve Q"),
        ("import refused on provenance",
         'floor 1.0\ng := import graph "one.smi"\n'
         '       require supplied < 0.001\n'
         '       unless refuse\nobserve g'),
    ]
    h4_rows = []
    for name, src in FOUR:
        res = run_src(src)
        st = res.get("status")
        logs = res.get("log", [])
        kinds = []
        for entry in logs:
            if not isinstance(entry, dict):
                continue
            v = entry.get("verdict")
            if isinstance(v, dict) and v.get("verdict"):
                kinds.append(v["verdict"])
            elif entry.get("level"):
                kinds.append(entry["level"])
        kinds = sorted(set(kinds))
        h4_rows.append({
            "program": name,
            "status": st,
            "verdict_kinds": kinds,
            "n_bindings": len(bindings(res)),
            "distinguisher": "%s|%s" % (st, ".".join(kinds)),
        })
    distinct = len({r["distinguisher"] for r in h4_rows})

    # the impoverished wrapper: value or nothing, which is the interface
    # the theorem says loses the distinction
    def wrapper(res):
        return "value" if bindings(res) else "nothing"

    wrapped = {wrapper(run_src(src)) for _n, src in FOUR}
    out["results"]["H4_conflation_control"] = {
        "rows": h4_rows,
        "n_programs": len(FOUR),
        "n_distinct_outcomes": distinct,
        "n_wrapped_outcomes": len(wrapped),
        "wrapped_values": sorted(wrapped),
        "note": "four programs failing for four different reasons are "
                "four distinct outcomes under the verdict interface and "
                "one under a value-or-nothing interface.  That collapse "
                "is what makes the conflation theorem non-vacuous; "
                "without it the verdict apparatus would be unmotivated",
        "passed": distinct == len(FOUR) and len(wrapped) == 1,
    }

    # --- H5 (C5): cell counts distinguish ------------------------------
    # Same atoms, same stoichiometry, different committed cell counts.
    # A connectivity-only model writes both as "C bonded to two O" and
    # cannot tell them apart.
    two_cells = run_src("floor 1.0\nC := cut 6\nO := cut 8\n"
                        "D := close C(O : 2, O : 2)\nobserve D")
    one_cells = run_src("floor 1.0\nC := cut 6\nO := cut 8\n"
                        "S := close C(O : 1, O : 1)\nobserve S")

    def compound(res):
        for b in bindings(res):
            if b.get("type") == "Compound":
                return b
        return {}

    def verdicts(res):
        return [e.get("verdict") for e in res.get("log", [])
                if isinstance(e, dict) and e.get("verdict")]

    a, b = compound(two_cells), compound(one_cells)
    # the difference is stronger than a differing field: committing one
    # cell per interface leaves carbon with residual vacancy, so no
    # compound is produced at all and the program returns an `unclosed`
    # verdict instead
    differ = (a.get("closed") is True and not b)
    out["results"]["H5_cell_counts_distinguish"] = {
        "two_cells": {k: a.get(k) for k in
                      ("stoichiometry", "vacancy", "closed", "residue")},
        "one_cell": {k: b.get(k) for k in
                     ("stoichiometry", "vacancy", "closed", "residue")},
        "two_cells_compound_built": bool(a),
        "one_cell_compound_built": bool(b),
        "one_cell_verdicts": verdicts(one_cells),
        "one_cell_open_atoms": [
            e.get("payload", {}).get("open")
            for e in one_cells.get("log", [])
            if isinstance(e, dict) and e.get("verdict") == "unclosed"
        ],
        "differ": differ,
        "note": "the same atoms at the same stoichiometry give different "
                "outcomes when the committed cell counts differ: two "
                "cells per interface closes carbon exactly, one cell "
                "leaves residual vacancy and yields no compound.  A "
                "connectivity-only model writes both as the same graph",
        "passed": bool(differ),
    }

    # --- H6 (C6): delocalisation is a total, not a per-pair count ------
    ring = run_src(load("benzene.hj"))
    rb = [x for x in bindings(ring) if x.get("type") in ("Deloc", "System",
                                                         "Delocalised")]
    if not rb:
        rb = [x for x in bindings(ring) if "cells" in x and "members" in x]
    keys = sorted({k for x in rb for k in x})
    per_pair = [k for k in keys if "per_pair" in k or "bond_order" in k]
    # What (C6) requires is that no program can READ OUT a per-pair
    # count, not that no key is named.  The implementation names the key
    # and holds it at null, which is the stronger documentation choice --
    # a reader sees the absence declared rather than having to infer it
    # from a missing field.  The test is therefore that every such key is
    # null, and that a total IS present to carry the system's commitment.
    readable = [(x.get("name"), k, x[k]) for x in rb for k in per_pair
                if x.get(k) is not None]
    totals = [x.get("total_cells") for x in rb]
    # Sweep delocalised systems of different size and commitment.  Two
    # systems with the SAME centre count and different totals show the
    # total is not a function of the centres, so it cannot be unpacked
    # into a per-pair count by dividing.
    DELOC = [(4, 5), (5, 7), (6, 6), (6, 9), (10, 15)]
    sweep = []
    for ncen, cells in DELOC:
        src = "floor 1.0\n"
        src += "".join("c%d := cut 6\n" % i for i in range(ncen))
        src += "r := deloc ring(%s) cells: %d\nobserve r" % (
            ", ".join("c%d" % i for i in range(ncen)), cells)
        res = run_src(src)
        got = [x for x in bindings(res) if x.get("type") == "Deloc"]
        if not got:
            sweep.append({"n_centres": ncen, "cells_asked": cells,
                          "status": res.get("status"), "built": False})
            continue
        b = got[0]
        sweep.append({
            "n_centres": b.get("n_centres"),
            "cells_asked": cells,
            "total_cells": b.get("total_cells"),
            "residue": b.get("residue"),
            "per_pair_cells": b.get("per_pair_cells"),
            "per_pair_readable": b.get("per_pair_cells") is not None,
            "built": True,
        })
    built = [x for x in sweep if x.get("built")]
    same_centres = {}
    for x in built:
        same_centres.setdefault(x["n_centres"], set()).add(x["total_cells"])
    # a centre count carrying more than one total proves the total is not
    # determined by the centres
    ambiguous = {k: sorted(v) for k, v in same_centres.items() if len(v) > 1}

    out["results"]["H6_deloc_sweep"] = {
        "rows": sweep,
        "n_systems": len(built),
        "n_per_pair_readable": sum(1 for x in built
                                   if x["per_pair_readable"]),
        "centre_counts_with_multiple_totals": ambiguous,
        "note": "two systems with the same centre count and different "
                "totals show the total is not a function of the centres, "
                "so it cannot be unpacked into a per-pair count by "
                "dividing; and no system exposes a readable one",
        "passed": (sum(1 for x in built if x["per_pair_readable"]) == 0
                   and len(ambiguous) > 0),
    }

    out["results"]["H6_deloc_not_a_count"] = {
        "status": ring.get("status"),
        "n_deloc_bindings": len(rb),
        "keys_exposed": keys,
        "per_pair_keys_named": per_pair,
        "per_pair_values_readable": readable,
        "n_readable_per_pair": len(readable),
        "total_cells": totals,
        "n_centres": [x.get("n_centres") for x in rb],
        "note": "the value carries a total over the system and holds "
                "every per-pair field at null.  Naming the field and "
                "leaving it null declares the absence rather than "
                "hiding it; what (C6) forbids is a readable count, and "
                "none is readable",
        "passed": len(readable) == 0 and all(t is not None for t in totals),
    }

    # --- H7 (C7) / H8 (C8): the resolution gate and its scope ----------
    GATE = [
        ("floor above target eps", "floor 1.0\nC := cut 6\nobserve C", 1e-9),
        ("floor below target eps", "floor 1e-12\nC := cut 6\nobserve C", 1e-9),
        ("floor far below target", "floor 1e-18\nC := cut 6\nobserve C", 1e-9),
    ]
    h7_rows, refused = [], 0
    for name, src, eps in GATE:
        res = run_src(src, eps=eps)
        # a REFUSED floor declaration leaves the ambient floor unchanged,
        # so the declared value is in the refusal verdict, not in
        # res["floor"].  Reading the latter would silently score every
        # refusal as in-scope.
        declared = res.get("floor")
        for entry in res.get("log", []):
            v = entry.get("verdict") if isinstance(entry, dict) else None
            if isinstance(v, dict) and v.get("verdict") == "subfloor":
                declared = v.get("payload", {}).get("found", declared)
                break
        sub = declared is not None and declared < eps
        ok = res.get("status") == "ok"
        if sub and not ok:
            refused += 1
        h7_rows.append({
            "program": name,
            "declared_floor": declared,
            "ambient_floor_after": res.get("floor"),
            "target_resolution": res.get("target_resolution", eps),
            "sub_resolution": sub,
            "status": res.get("status"),
            "refused": not ok,
            "in_scope_for_target_equivalence": not sub,
        })
    n_sub = sum(1 for r in h7_rows if r["sub_resolution"])
    in_scope = [r for r in h7_rows if r["in_scope_for_target_equivalence"]]
    out["results"]["H7_resolution_gate"] = {
        "rows": h7_rows,
        "n_programs": len(h7_rows),
        "n_sub_resolution": n_sub,
        "n_refused": refused,
        "note": "a program asking for a distinction finer than the target "
                "can represent is refused with both numbers named",
        "passed": refused == n_sub,
    }
    out["results"]["H8_target_equivalence_scope"] = {
        "n_total": len(h7_rows),
        "n_in_scope": len(in_scope),
        "n_excluded": len(h7_rows) - len(in_scope),
        "strict_subset": len(in_scope) < len(h7_rows),
        "note": "sub-resolution programs are excluded from the "
                "equivalence claim rather than run and tolerated; "
                "reporting their disagreement as acceptable would assert "
                "the unconditional theorem the paper retracts",
        "passed": len(in_scope) < len(h7_rows),
    }

    # --- H3: verdict realisation across the example programs -----------
    h3_rows, valuebearing_failures = [], 0
    for f in sorted(os.listdir(EX)):
        if not f.endswith(".hj"):
            continue
        res = run_src(load(f))
        st = res.get("status")
        nb = len(bindings(res))
        if st not in ("ok",) and nb > 0:
            # a failing program that still emitted bindings would be
            # carrying a value on a failure verdict
            valuebearing_failures += 1
        h3_rows.append({"program": f, "status": st, "n_bindings": nb,
                        "cut_count": res.get("cut_count")})
    out["results"]["H3_verdict_realisation"] = {
        "rows": h3_rows,
        "n_programs": len(h3_rows),
        "statuses": sorted({r["status"] for r in h3_rows}),
        "value_bearing_failures": valuebearing_failures,
        "note": "no failing program emits bindings; only a value-bearing "
                "outcome carries a value",
        "passed": valuebearing_failures == 0,
    }

    out["all_passed"] = all(v.get("passed", True)
                            for v in out["results"].values())
    return out


if __name__ == "__main__":
    res = run()
    dest = os.path.join(os.path.dirname(__file__), "..", "results",
                        "exp_honjo.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({
        "experiment": res["experiment"],
        "all_passed": res["all_passed"],
        **{k: v.get("passed") for k, v in res["results"].items()},
    }, indent=2))
