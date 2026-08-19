"""Auxiliary measurements for the manuscript panels.

The results files carry the experimental outcomes.  The panels need a few
additional per-atom and per-pair arrays that the experiments summarise
rather than emit.  Everything here is measured by the same code paths;
nothing is simulated.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "honjo-py"))

from collections import Counter

from corpus import DRUGLIKE, ISOSTERE_PAIRS, MATCHED_PAIRS, SYMMETRIC
from cutrank import (
    class_jaccard, correspondence, cut_key, heavy_atoms, labels_at_radius,
    neighbour_keys, refine,
)
from hjm.core.graph import MEDIUM
from hjm.masamune.translate import translate_smiles


def g_of(smi):
    v = translate_smiles(smi)
    return v.value if v.ok else None


def atom_table():
    """Per-atom (sigma, depth, degree, Z, class-index) over both corpora."""
    rows = []
    for name, smi in DRUGLIKE:
        g = g_of(smi)
        if g is None:
            continue
        r = refine(g)
        labs = sorted(set(r["labels"].values()), key=str)
        idx = {l: i for i, l in enumerate(labs)}
        for k in heavy_atoms(g):
            s, side, _ = g.separation_cost(k)
            rows.append({
                "molecule": name,
                "atom": k,
                "z": g.atoms[k].z,
                "sigma": round(s, 6),
                "depth": len([x for x in side if x != MEDIUM]),
                "degree": len(neighbour_keys(g, k)),
                "heavy_degree": len([o for o in neighbour_keys(g, k)
                                     if g.atoms[o].z > 1]),
                "class_index": idx[r["labels"][k]],
                "n_classes": r["classes"],
                "n_heavy": r["n"],
            })
    return rows


def refinement_curves():
    """Class count per round for every molecule in both corpora."""
    out = []
    for name, smi, orb in SYMMETRIC:
        g = g_of(smi)
        if g is None:
            continue
        r = refine(g)
        out.append({"molecule": name, "corpus": "symmetric",
                    "n_heavy": r["n"], "orbits": orb,
                    "history": r["history"], "rounds": r["rounds"]})
    for name, smi in DRUGLIKE:
        g = g_of(smi)
        if g is None:
            continue
        r = refine(g)
        out.append({"molecule": name, "corpus": "druglike",
                    "n_heavy": r["n"], "orbits": None,
                    "history": r["history"], "rounds": r["rounds"]})
    return out


def sigma_depth_grid():
    """Occupancy of the (sigma, depth) key plane over the drug-like set."""
    c = Counter()
    for name, smi in DRUGLIKE:
        g = g_of(smi)
        if g is None:
            continue
        for k in heavy_atoms(g):
            c[cut_key(g, k)] += 1
    return [{"sigma": s, "depth": d, "count": n} for (s, d), n in c.items()]


def radius_pair_matrix():
    """Class overlap for every annotated pair at every radius."""
    out = []
    for n1, s1, n2, s2, rel in ISOSTERE_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        row = {"a": n1, "b": n2, "relation": rel, "by_radius": []}
        for rad in range(4):
            row["by_radius"].append(round(class_jaccard(g1, g2, radius=rad), 6))
        out.append(row)
    return out


def druglike_overlap_matrix():
    """Full class-overlap matrix over the drug-like set at radius 0."""
    names, gs = [], []
    for name, smi in DRUGLIKE:
        g = g_of(smi)
        if g is not None:
            names.append(name)
            gs.append(g)
    n = len(gs)
    m = [[0.0] * n for _ in range(n)]
    e = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
        e[i][i] = 1.0
        for j in range(i + 1, n):
            v = class_jaccard(gs[i], gs[j], radius=0)
            m[i][j] = m[j][i] = v
            c1 = Counter(gs[i].atoms[k].z for k in heavy_atoms(gs[i]))
            c2 = Counter(gs[j].atoms[k].z for k in heavy_atoms(gs[j]))
            inter = sum((c1 & c2).values())
            union = sum((c1 | c2).values())
            ev = round(inter / union, 6) if union else 0.0
            e[i][j] = e[j][i] = ev
    return {"names": names, "class_overlap": m, "element_overlap": e}


def correspondence_detail():
    """Per-pair correspondence with cross-element counts, all radii."""
    out = []
    for n1, s1, n2, s2 in MATCHED_PAIRS:
        g1, g2 = g_of(s1), g_of(s2)
        if g1 is None or g2 is None:
            continue
        row = {"a": n1, "b": n2, "n1": len(heavy_atoms(g1)),
               "n2": len(heavy_atoms(g2)), "by_radius": []}
        for rad in range(4):
            c = correspondence(g1, g2, radius=rad)
            cross = sum(1 for a, b in c["pairs"]
                        if g1.atoms[a].z != g2.atoms[b].z)
            row["by_radius"].append({
                "radius": rad, "size": c["size"],
                "coverage": c["coverage"], "cross": cross,
                "classes_shared": c["classes_shared"],
            })
        out.append(row)
    return out


def class_size_distribution():
    """Distribution of label-class sizes by radius over the drug-like set."""
    out = []
    for rad in range(4):
        sizes = Counter()
        for name, smi in DRUGLIKE:
            g = g_of(smi)
            if g is None:
                continue
            labs = Counter(labels_at_radius(g, rad).values())
            for _l, n in labs.items():
                sizes[n] += 1
        out.append({"radius": rad, "sizes": dict(sorted(sizes.items()))})
    return out


if __name__ == "__main__":
    data = {
        "atoms": atom_table(),
        "refinement_curves": refinement_curves(),
        "sigma_depth_grid": sigma_depth_grid(),
        "radius_pair_matrix": radius_pair_matrix(),
        "druglike_overlap_matrix": druglike_overlap_matrix(),
        "correspondence_detail": correspondence_detail(),
        "class_size_distribution": class_size_distribution(),
    }
    dest = os.path.join(os.path.dirname(__file__), "..", "results",
                        "panel_data.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(json.dumps({k: len(v) if isinstance(v, list) else "obj"
                      for k, v in data.items()}, indent=2))
