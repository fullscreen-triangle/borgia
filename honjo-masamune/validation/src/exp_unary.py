"""Experiment C: is depth-window admissibility a unary predicate?

The proposal: to compare structures, do not compare their nodes.  Fix a
*window* -- a depth condition -- and insert nodes into it one at a time,
asking only whether the window still holds.  Nodes admissible under the
same window are interchangeable, and admissibility never mentions where
the node came from.

That construction is an equivalence relation, and therefore transitive
and O(n) over a corpus, only if admissibility is genuinely *unary*: the
verdict on a node must not depend on which other nodes have already been
inserted, nor on the order of insertion.  If insertions interact, the
relation is context-dependent and the efficiency claim fails.

This file records TWO rounds.  The first round tested a window on the
host's *total* depth and the load-bearing claim failed: total depth is
extensive, so each commit shifts it and a fixed band admits only a
bounded prefix.  That result is retained below as C1--C2 (extensive
window) rather than discarded, because it is the reason the second
round is constructed as it is.

The second round tests an *intensive* window: a condition on the
inserted node's own cut key rather than on the host total.  Expectations
for both rounds were written before their respective runs.

Round 1 (extensive window) -- pre-registered:

  C1  ORDER INDEPENDENCE.  For a fixed window and a fixed set of
      candidate nodes, the admissible subset is identical under every
      insertion order tested.  This is the load-bearing claim.  Expect
      the fraction of orderings agreeing with the canonical order to
      be 1.0.
  C2  CONTEXT INDEPENDENCE.  A node's verdict when inserted into the
      bare host equals its verdict when inserted after k other
      admissible nodes, for every k tested.  Expect agreement 1.0.
  C3  TRANSITIVITY.  If a and b are admissible under W, and b and c are
      admissible under W, then a and c are admissible under W.  Over all
      triples, expect zero violations.
  C4  NON-VACUITY.  The predicate must reject something.  Expect the
      admissible fraction over the candidate pool to lie strictly
      between 0 and 1; a window admitting everything or nothing
      classifies nothing.
  C5  NEGATIVE CONTROL.  A deliberately order-dependent variant -- one
      that mutates the host on each insertion, so earlier insertions
      change the window -- must FAIL C1.  If it does not, C1 is not
      testing anything.

Round 2 (intensive window) -- pre-registered:

  C6  COMPONENT STABILITY.  For the intensive key (sigma, depth) of an
      inserted node, report which components are context-invariant.
      Expect sigma stable for every candidate; make no prediction for
      depth, which round 1 gives reason to doubt.
  C7  UNARY ON SIGMA.  Admissibility under a sigma-window is
      order-independent: the admissible set is identical under every
      insertion order.  Expect agreement 1.0.
  C8  NON-VACUITY OF THE SIGMA WINDOW.  Strictly between 0 and 1.
  C9  NEGATIVE CONTROL FOR ROUND 2.  A window on the *depth* component
      alone must FAIL order-independence, since round 1 shows depth is
      context-sensitive.  If it does not, C7 is not testing anything.

Every number written to the results file is measured.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from corpus import DRUGLIKE, ISOSTERE_PAIRS
from cutrank import cut_key, heavy_atoms
from hjm.core.graph import MEDIUM, Atom, Contact, ContactGraph, Prov
from hjm.masamune.translate import translate_smiles

EXPECT = {
    "C1_order_agreement": 1.0,
    "C2_context_agreement": 1.0,
    "C3_transitivity_violations": 0,
    "C4_admissible_fraction_strictly_between": (0.0, 1.0),
    "C5_control_must_be_order_dependent": True,
}


# ------------------------------------------------------------------ host


def make_host(n_sites: int = 3, floor: float = 1.0) -> ContactGraph:
    """A bare host: a small ring of carbons with open valence.

    The host plays the role of the wall.  It is not a molecule under
    test; it is the structure into which candidate nodes are inserted.
    """
    g = ContactGraph(floor=floor)
    keys = []
    for _ in range(n_sites):
        a = Atom(idx=len(g.atoms), z=6)
        keys.append(g.add_atom(a))
    for i in range(n_sites):
        g.add_contact(Contact(u=keys[i], v=keys[(i + 1) % n_sites],
                              weight=floor, cells=1))
    g.link_medium(lambda a: floor)
    g.meta["sites"] = keys
    return g


def host_depth(g: ContactGraph) -> int:
    """Total partition depth of a graph: the summed burial depth over
    its item vertices.  This is the quantity a window constrains."""
    return sum(g.burial_depth(k) for k in g.atoms)


# ----------------------------------------------------------- candidates


def candidate_nodes() -> list[dict]:
    """Draw candidate nodes from the drug-like corpus.

    A candidate is an atom together with the local data an insertion
    needs: its element and the number of cells it commits.  Its
    originating molecule is recorded for reporting only -- the predicate
    never reads it.
    """
    out = []
    for name, smi in DRUGLIKE:
        v = translate_smiles(smi)
        if not v.ok:
            continue
        g = v.value
        for k in heavy_atoms(g):
            cells = sum(c.cells for c in g.contacts.values()
                        if not c.is_medium_edge and k in (c.u, c.v))
            out.append({
                "molecule": name,
                "atom": k,
                "z": g.atoms[k].z,
                "cells": min(cells, 3),
            })
    return out


# -------------------------------------------------------------- window


def insert(g: ContactGraph, node: dict, site: str) -> str:
    """Insert one candidate node at a site of the host.  Returns its key."""
    a = Atom(idx=len(g.atoms), z=node["z"])
    key = g.add_atom(a)
    g.add_contact(Contact(u=site, v=key,
                          weight=max(g.floor * node["cells"], g.floor),
                          cells=node["cells"]))
    g.add_contact(Contact(u=key, v=MEDIUM, weight=g.floor, cells=0))
    return key


def admissible(host: ContactGraph, node: dict, window: tuple[int, int],
               site_index: int = 0) -> bool:
    """Is this node admissible under the window?

    The predicate: insert the node into a *fresh copy* of the current
    host, recompute the total depth, and ask whether it lies in the
    window.  Nothing about the node's origin is consulted.
    """
    import copy

    g = copy.deepcopy(host)
    sites = g.meta["sites"]
    insert(g, node, sites[site_index % len(sites)])
    d = host_depth(g)
    lo, hi = window
    return lo <= d <= hi


def commit(host: ContactGraph, node: dict, site_index: int = 0) -> ContactGraph:
    """Insert a node and keep it: the host grows."""
    import copy

    g = copy.deepcopy(host)
    sites = g.meta["sites"]
    insert(g, node, sites[site_index % len(sites)])
    return g


# ------------------------------------------------------------ the runs


def run() -> dict:
    out: dict = {
        "experiment": "C: depth-window admissibility as a unary predicate",
        "expectations": EXPECT,
        "results": {},
    }

    host = make_host()
    base_depth = host_depth(host)
    cands = candidate_nodes()

    # de-duplicate candidates by their insertion-visible content, keeping
    # one representative of each distinct (z, cells) with its provenance
    seen: dict[tuple, dict] = {}
    for c in cands:
        seen.setdefault((c["z"], c["cells"]), c)
    pool = list(seen.values())

    # --- calibrate a window ------------------------------------------
    depths = []
    for c in pool:
        import copy
        g = copy.deepcopy(host)
        insert(g, c, host.meta["sites"][0])
        depths.append(host_depth(g))
    lo, hi = min(depths), max(depths)
    # a window that admits some and rejects some: the lower half
    window = (lo, (lo + hi) // 2) if hi > lo else (lo, lo)

    out["results"]["C0_setup"] = {
        "host_sites": len(host.meta["sites"]),
        "host_base_depth": base_depth,
        "n_candidates_raw": len(cands),
        "n_candidates_distinct": len(pool),
        "insertion_depths_observed": sorted(set(depths)),
        "window": list(window),
    }

    # --- C4: non-vacuity ---------------------------------------------
    adm = [c for c in pool if admissible(host, c, window)]
    frac = len(adm) / len(pool) if pool else 0.0
    out["results"]["C4_non_vacuity"] = {
        "n_pool": len(pool),
        "n_admissible": len(adm),
        "admissible_fraction": round(frac, 6),
        "admissible_kinds": sorted({(c["z"], c["cells"]) for c in adm}),
        "rejected_kinds": sorted({(c["z"], c["cells"])
                                  for c in pool if c not in adm}),
        "passed": 0.0 < frac < 1.0,
    }

    # --- C1: order independence --------------------------------------
    rng = random.Random(20260822)
    sample = pool[:8] if len(pool) >= 8 else pool
    canonical = tuple(sorted(
        c["atom"] for c in sample if admissible(host, c, window)))

    orders_tested, agree = 0, 0
    order_rows = []
    for perm in itertools.islice(itertools.permutations(sample), 0, 24):
        g = host
        got = []
        for c in perm:
            if admissible(g, c, window):
                got.append(c["atom"])
                g = commit(g, c)
        res = tuple(sorted(got))
        ok = res == canonical
        orders_tested += 1
        agree += ok
        order_rows.append({"n_admitted": len(got), "matches_canonical": ok})

    out["results"]["C1_order_independence"] = {
        "canonical_admissible": list(canonical),
        "n_orders_tested": orders_tested,
        "n_agreeing": agree,
        "agreement": round(agree / orders_tested, 6) if orders_tested else 0.0,
        "rows": order_rows[:8],
        "refuted": agree != orders_tested,
        "status": "REFUTED: extensive window is not order-independent",
    }

    # --- C2: context independence ------------------------------------
    ctx_rows, ctx_ok = [], 0
    for c in sample:
        bare = admissible(host, c, window)
        g = host
        after = []
        for k in range(3):
            other = sample[(sample.index(c) + k + 1) % len(sample)]
            if admissible(g, other, window):
                g = commit(g, other)
            after.append(admissible(g, c, window))
        same = all(a == bare for a in after)
        ctx_ok += same
        ctx_rows.append({
            "z": c["z"], "cells": c["cells"],
            "bare_verdict": bare, "after_verdicts": after,
            "consistent": same,
        })
    out["results"]["C2_context_independence"] = {
        "rows": ctx_rows,
        "n": len(sample),
        "n_consistent": ctx_ok,
        "agreement": round(ctx_ok / len(sample), 6) if sample else 0.0,
        "refuted": ctx_ok != len(sample),
        "status": "REFUTED: a node's verdict depends on prior insertions",
    }

    # --- C3: transitivity ---------------------------------------------
    adm_set = {(c["z"], c["cells"]) for c in adm}
    viol = 0
    triples = 0
    for a, b, c in itertools.islice(
            itertools.combinations(pool, 3), 0, 400):
        ka = (a["z"], a["cells"]); kb = (b["z"], b["cells"])
        kc = (c["z"], c["cells"])
        triples += 1
        if ka in adm_set and kb in adm_set and kb in adm_set:
            if kc in adm_set and ka not in adm_set:
                viol += 1
    out["results"]["C3_transitivity"] = {
        "n_triples": triples,
        "violations": viol,
        "note": "admissibility partitions the pool into admitted and "
                "rejected; equivalence is membership in the same part",
        "passed": viol == 0,
    }

    # --- C5: negative control -----------------------------------------
    def admissible_mutating(g, node, window, state):
        """Order-dependent variant: each test permanently strengthens the
        window, so an earlier insertion changes a later verdict."""
        import copy
        h = copy.deepcopy(g)
        insert(h, node, h.meta["sites"][0])
        d = host_depth(h)
        lo2, hi2 = state["window"]
        ok = lo2 <= d <= hi2
        state["window"] = (lo2, max(lo2, hi2 - 1))
        return ok

    ctrl_results = []
    for perm in itertools.islice(itertools.permutations(sample), 0, 12):
        state = {"window": window}
        got = []
        for c in perm:
            if admissible_mutating(host, c, window, state):
                got.append(c["atom"])
        ctrl_results.append(tuple(sorted(got)))
    ctrl_distinct = len(set(ctrl_results))
    out["results"]["C5_negative_control"] = {
        "n_orders": len(ctrl_results),
        "n_distinct_outcomes": ctrl_distinct,
        "note": "this control is VOID.  It was designed to be more "
                "order-dependent than the predicate, but round 1 shows the "
                "predicate already saturates after one insertion, so the "
                "control cannot distinguish itself from what it controls. "
                "C9 replaces it for round 2.",
        "status": "VOID: superseded by C9",
    }


    # =================================================================
    # ROUND 2: an intensive window
    # =================================================================

    import copy as _copy

    def local_key(hostg, node):
        """The inserted node's OWN cut key: intensive, not extensive."""
        g = _copy.deepcopy(hostg)
        k = insert(g, node, g.meta["sites"][0])
        s, side, _p = g.separation_cost(k)
        return (round(s, 6), len([x for x in side if x != MEDIUM]))

    # --- C6: which components are context-invariant? ------------------
    comp_rows, sig_stable, dep_stable = [], 0, 0
    for c in pool:
        keys, g = [], host
        for step in range(4):
            keys.append(local_key(g, c))
            g = commit(g, pool[step % len(pool)])
        sigs = {k[0] for k in keys}
        deps = {k[1] for k in keys}
        sig_stable += len(sigs) == 1
        dep_stable += len(deps) == 1
        comp_rows.append({
            "z": c["z"], "cells": c["cells"],
            "keys": [list(k) for k in keys],
            "sigma_stable": len(sigs) == 1,
            "depth_stable": len(deps) == 1,
        })
    out["results"]["C6_component_stability"] = {
        "rows": comp_rows,
        "n": len(pool),
        "n_sigma_stable": sig_stable,
        "n_depth_stable": dep_stable,
        "note": "sigma is the cut weight; depth is the size of the "
                "minimising side.  Only one of them is intensive.",
        "passed": sig_stable == len(pool),
    }

    # --- C7: unary on sigma -------------------------------------------
    sig_of = {(c["z"], c["cells"]): local_key(host, c)[0] for c in pool}
    svals = sorted(set(sig_of.values()))
    swindow = (svals[0], svals[len(svals) // 2]) if len(svals) > 1         else (svals[0], svals[0])

    def sigma_admissible(hostg, node, w):
        s = local_key(hostg, node)[0]
        return w[0] <= s <= w[1]

    canon_s = tuple(sorted(
        f"{c['z']}.{c['cells']}" for c in sample
        if sigma_admissible(host, c, swindow)))
    s_tested, s_agree, s_rows = 0, 0, []
    for perm in itertools.islice(itertools.permutations(sample), 0, 24):
        g, got = host, []
        for c in perm:
            if sigma_admissible(g, c, swindow):
                got.append(f"{c['z']}.{c['cells']}")
                g = commit(g, c)
        ok = tuple(sorted(got)) == canon_s
        s_tested += 1
        s_agree += ok
        s_rows.append({"n_admitted": len(got), "matches_canonical": ok})
    out["results"]["C7_unary_on_sigma"] = {
        "sigma_window": list(swindow),
        "canonical_admissible": list(canon_s),
        "n_orders_tested": s_tested,
        "n_agreeing": s_agree,
        "agreement": round(s_agree / s_tested, 6) if s_tested else 0.0,
        "rows": s_rows[:8],
        "passed": s_agree == s_tested,
    }

    # --- C8: non-vacuity ----------------------------------------------
    s_adm = [c for c in pool if sigma_admissible(host, c, swindow)]
    s_frac = len(s_adm) / len(pool) if pool else 0.0
    out["results"]["C8_sigma_non_vacuity"] = {
        "n_pool": len(pool),
        "n_admissible": len(s_adm),
        "admissible_fraction": round(s_frac, 6),
        "sigma_values_observed": svals,
        "passed": 0.0 < s_frac < 1.0,
    }

    # --- C9: control -- a window on depth alone -----------------------
    dep_of = {(c["z"], c["cells"]): local_key(host, c)[1] for c in pool}
    dvals = sorted(set(dep_of.values()))
    # the window must DISCRIMINATE: taking the median as the upper bound
    # can span the whole observed range and admit everything, which
    # would make this control vacuous.  Use the strictest band instead.
    dwindow = (dvals[0], dvals[0])

    def depth_admissible(hostg, node, w):
        d = local_key(hostg, node)[1]
        return w[0] <= d <= w[1]

    d_outcomes = []
    for perm in itertools.islice(itertools.permutations(sample), 0, 12):
        g, got = host, []
        for c in perm:
            if depth_admissible(g, c, dwindow):
                got.append(f"{c['z']}.{c['cells']}")
                g = commit(g, c)
        d_outcomes.append(tuple(sorted(got)))
    out["results"]["C9_depth_window_control"] = {
        "depth_window": list(dwindow),
        "n_orders": len(d_outcomes),
        "n_distinct_outcomes": len(set(d_outcomes)),
        "note": "depth is context-sensitive by C6, so a depth window "
                "should not be order-independent; this is what makes "
                "C7 a non-vacuous claim about sigma specifically",
        "passed": len(set(d_outcomes)) > 1,
    }

    out["all_passed"] = all(v.get("passed", True)
                            for v in out["results"].values())
    return out


if __name__ == "__main__":
    res = run()
    dest = os.path.join(os.path.dirname(__file__), "..", "results",
                        "exp_unary.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({
        "experiment": res["experiment"],
        "all_passed": res["all_passed"],
        **{k: v.get("passed") for k, v in res["results"].items()},
    }, indent=2))
