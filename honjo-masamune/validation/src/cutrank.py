"""Cut-refinement ranking and correspondence.

The two constructions the manuscripts describe, implemented once and
shared by both validation suites.

  * ``cut_key``      -- the base invariant of a vertex: (sigma, |S*|)
  * ``refine``       -- iterated refinement to the stable partition
  * ``class_multiset`` -- the multiset of stable classes of a structure
  * ``correspondence`` -- greedy class-respecting vertex correspondence

Nothing here fits a parameter.  Every quantity is a minimum cut on the
contact graph or a function of one.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from hjm.core.graph import MEDIUM


# ---------------------------------------------------------------- base


def heavy_atoms(g) -> list[str]:
    return [k for k, a in g.atoms.items() if a.z > 1]


def cut_key(g, k: str) -> tuple:
    """Base invariant: minimum-cut weight against the medium, and the
    size of the minimising side (the burial depth)."""
    sigma, side, _prov = g.separation_cost(k)
    depth = len([s for s in side if s != MEDIUM])
    return (round(sigma, 9), depth)


def neighbour_keys(g, k: str) -> list[str]:
    return [o for o, _c in g.neighbours(k) if o != MEDIUM]


# ---------------------------------------------------------- refinement


def refine(g, max_rounds: int = 32) -> dict:
    """Iterated cut refinement.

    Round 0 labels each heavy atom by its cut key.  Each subsequent round
    replaces a label by the pair (own label, sorted multiset of neighbour
    labels), then compresses to ranks.  Iteration stops when the number of
    classes stops growing.

    Returns a dict with the stable labelling, the class count per round,
    and the round at which it stabilised.
    """
    heavy = heavy_atoms(g)
    if not heavy:
        return {"labels": {}, "history": [0], "rounds": 0, "n": 0, "classes": 0}

    lab: dict[str, object] = {k: cut_key(g, k) for k in heavy}
    history = [len(set(map(str, lab.values())))]

    for r in range(1, max_rounds + 1):
        new: dict[str, tuple] = {}
        for k in heavy:
            nb = sorted(str(lab[o]) for o in neighbour_keys(g, k) if o in lab)
            new[k] = (str(lab[k]), tuple(nb))
        order = {v: i for i, v in enumerate(sorted(set(new.values()), key=str))}
        lab = {k: order[v] for k, v in new.items()}
        n_cls = len(set(lab.values()))
        history.append(n_cls)
        if n_cls == history[-2]:
            break

    return {
        "labels": {k: str(v) for k, v in lab.items()},
        "history": history,
        "rounds": len(history) - 1,
        "n": len(heavy),
        "classes": history[-1],
    }


def stable_classes(g) -> dict[str, str]:
    return refine(g)["labels"]



def labels_at_radius(g, radius: int = 0) -> dict[str, str]:
    """Vertex labels at a given refinement radius.

    ``radius=0`` is the base cut key: maximally tolerant, because two
    atoms agree whenever their separation cost and burial depth agree,
    whatever their element.  Each further round conditions on more of the
    neighbourhood and so discriminates more.  Correspondence uses a small
    radius by design; ranking (Experiment A) uses the stable partition.
    """
    heavy = heavy_atoms(g)
    lab = {k: str(cut_key(g, k)) for k in heavy}
    for _ in range(radius):
        new = {}
        for k in heavy:
            nb = sorted(lab[o] for o in neighbour_keys(g, k) if o in lab)
            new[k] = f"({lab[k]}|{','.join(nb)})"
        lab = new
    return lab


def class_multiset_at(g, radius: int = 0) -> Counter:
    """Multiset of labels at the given radius."""
    return Counter(labels_at_radius(g, radius).values())


def class_multiset(g) -> Counter:
    """Canonical multiset of stable class labels.

    Labels are rank integers, which are basis-dependent, so we re-key by a
    structural signature: for each class, the sorted multiset of base cut
    keys of its members together with its size.
    """
    r = refine(g)
    by_class: dict[str, list[str]] = defaultdict(list)
    for atom, lab in r["labels"].items():
        by_class[lab].append(atom)
    out: Counter = Counter()
    for lab, members in by_class.items():
        sig = (
            tuple(sorted(str(cut_key(g, m)) for m in members)),
            len(members),
        )
        out[sig] += 1
    return out


def structure_digest(g) -> str:
    """A canonical digest of the structure's stable class multiset.

    Two structures with equal digests have identical refined cut
    structure.  This is an invariant, not a hash of a chosen ordering.
    """
    items = sorted(f"{k}:{v}" for k, v in class_multiset(g).items())
    return hashlib.sha256("|".join(items).encode()).hexdigest()[:32]


# ----------------------------------------------------- atom-level keys


def atom_signature(g, k: str, radius: int = 2) -> str:
    """A radius-bounded cut signature for one atom.

    The analogue of a circular-fingerprint atom identifier, but built from
    cut keys rather than from element/degree tuples.
    """
    lab = {a: cut_key(g, a) for a in heavy_atoms(g)}
    cur = {a: str(lab[a]) for a in lab}
    for _ in range(radius):
        nxt = {}
        for a in cur:
            nb = sorted(cur[o] for o in neighbour_keys(g, a) if o in cur)
            nxt[a] = f"({cur[a]}|{','.join(nb)})"
        cur = nxt
    return hashlib.sha256(cur[k].encode()).hexdigest()[:16]


def structure_fingerprint(g, radius: int = 2, nbits: int = 1024) -> set[int]:
    """A folded set fingerprint over cut signatures, for comparison with
    a folded circular fingerprint at the same width."""
    bits = set()
    for k in heavy_atoms(g):
        for r in range(radius + 1):
            h = atom_signature(g, k, radius=r)
            bits.add(int(h, 16) % nbits)
    return bits


# ------------------------------------------------------ correspondence


def correspondence(g1, g2, radius: int = 0) -> dict:
    """Class-respecting vertex correspondence between two structures.

    Atoms may correspond when their *stable classes* agree, which is a
    coarser condition than element identity: a carbon and a nitrogen in
    equivalent structural roles fall in the same class and may correspond,
    while a ring carbon and a chain carbon do not.

    Greedy and deterministic: classes are matched largest-first, ties
    broken by the sorted base cut keys.  No search.
    """
    l1 = labels_at_radius(g1, radius)
    l2 = labels_at_radius(g2, radius)

    def by_label(lab):
        out: dict[str, list[str]] = defaultdict(list)
        for atom, v in lab.items():
            out[v].append(atom)
        return out

    c1, c2 = by_label(l1), by_label(l2)
    pairs: list[tuple[str, str]] = []
    keys = sorted(set(c1) & set(c2), key=lambda s: (-len(c1[s]), str(s)))
    for sig in keys:
        a_list = sorted(c1[sig])
        b_list = sorted(c2[sig])
        for a, b in zip(a_list, b_list):
            pairs.append((a, b))

    n1, n2 = len(heavy_atoms(g1)), len(heavy_atoms(g2))
    denom = max(n1, n2) or 1
    return {
        "pairs": pairs,
        "size": len(pairs),
        "n1": n1,
        "n2": n2,
        "coverage": round(len(pairs) / denom, 6),
        "classes_shared": len(keys),
        "classes_1": len(c1),
        "classes_2": len(c2),
        "radius": radius,
    }


def class_jaccard(g1, g2, radius: int = 0) -> float:
    """Jaccard overlap of cut-class multisets at a given radius.

    The radius is the tolerance dial.  At ``radius=0`` two atoms agree
    when their cut keys agree, so a carbon and a nitrogen in equivalent
    structural roles are interchangeable; higher radii progressively
    require the neighbourhoods to agree too.
    """
    m1, m2 = class_multiset_at(g1, radius), class_multiset_at(g2, radius)
    inter = sum((m1 & m2).values())
    union = sum((m1 | m2).values())
    return round(inter / union, 6) if union else 0.0


# ---------------------------------------------------- reference ECFP


def ecfp_like(g, radius: int = 2, nbits: int = 1024) -> set[int]:
    """A Morgan/ECFP-style folded fingerprint over the same graph.

    Implemented here so the comparison is like-for-like: same molecule,
    same radius, same fold width, differing only in the atom identifier.
    The identifier is the conventional one -- element, degree, charge,
    attached hydrogens -- rather than a cut key.
    """
    heavy = heavy_atoms(g)
    hcount: dict[str, int] = {}
    for k in heavy:
        hcount[k] = sum(1 for o in neighbour_keys(g, k) if g.atoms[o].z == 1)

    ident = {}
    for k in heavy:
        a = g.atoms[k]
        deg = len([o for o in neighbour_keys(g, k) if g.atoms[o].z > 1])
        ident[k] = f"{a.z}.{deg}.{a.charge}.{hcount[k]}"

    bits = set()
    cur = dict(ident)
    for k in heavy:
        bits.add(int(hashlib.sha256(cur[k].encode()).hexdigest(), 16) % nbits)
    for _ in range(radius):
        nxt = {}
        for k in heavy:
            nb = sorted(cur[o] for o in neighbour_keys(g, k)
                        if g.atoms[o].z > 1 and o in cur)
            nxt[k] = f"({cur[k]}|{','.join(nb)})"
        cur = nxt
        for k in heavy:
            bits.add(int(hashlib.sha256(cur[k].encode()).hexdigest(), 16) % nbits)
    return bits


def ecfp_atom_ids(g, radius: int = 2) -> dict[str, str]:
    """Unfolded ECFP-style atom identifiers at the given radius."""
    heavy = heavy_atoms(g)
    hcount = {
        k: sum(1 for o in neighbour_keys(g, k) if g.atoms[o].z == 1)
        for k in heavy
    }
    cur = {}
    for k in heavy:
        a = g.atoms[k]
        deg = len([o for o in neighbour_keys(g, k) if g.atoms[o].z > 1])
        cur[k] = f"{a.z}.{deg}.{a.charge}.{hcount[k]}"
    for _ in range(radius):
        nxt = {}
        for k in heavy:
            nb = sorted(cur[o] for o in neighbour_keys(g, k)
                        if g.atoms[o].z > 1 and o in cur)
            nxt[k] = f"({cur[k]}|{','.join(nb)})"
        cur = nxt
    return {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in cur.items()}


def tanimoto(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return round(len(a & b) / len(a | b), 6) if (a | b) else 0.0
