"""Translation: records -> provenance-tagged contact graphs.

Clause order follows the specification and is not arbitrary: each clause
precedes every clause whose evidence it would corrupt.

  (V1) malformed      - does not parse
  (V2) empty          - parses to no structural content
  (V3) unsupported    - requested features exceed the reader's capability
  (V4) underdetermined- an element admits several readings, none selected
  (V5) incomplete     - a requested element is neither stated nor suppliable
  (V6) translated     - a graph, with its provenance map
"""

from __future__ import annotations

from ..core.chem import symbol, valence
from ..core.graph import Atom, Contact, ContactGraph, Prov
from ..core.verdict import Verdict
from .capability import capability as _cap_of
from .capability import known_format, missing as _missing
from . import smiles as smi


class _Cap:
    """Small shim so the clause code reads as in the specification."""

    missing = staticmethod(_missing)
    capability = staticmethod(_cap_of)
    known_format = staticmethod(known_format)


cap = _Cap()

#: weight assigned to an interface committing k cells
def _bond_weight(cells: int, floor: float) -> float:
    return max(floor * cells, floor)


#: an atom's residual boundary against the medium, from unshared regions
def _medium_weight(unshared: int, floor: float) -> float:
    return max(floor * max(unshared, 1), floor)


def translate_smiles(
    text: str,
    required: set[str] | None = None,
    floor: float = 1.0,
    source_name: str = "<smiles>",
) -> Verdict:
    """Translate a SMILES string.  Returns a verdict, never a bare graph."""
    required = set(required or {"element", "connectivity"})

    # (V3) capability first for a *format*-level shortfall that does not
    # depend on the record: it is decidable without reading anything.
    miss = cap.missing("smiles", required)
    if miss:
        return Verdict.unsupported(list(miss), list(cap.capability("smiles")))

    # (V2) an empty record is EMPTY, not MALFORMED.  The parser raises on
    # empty input, so testing it here is what keeps the two labels
    # distinct: a record that says nothing is not a record that says
    # something ill-formed, and conflating them is exactly the loss the
    # verdict discipline exists to prevent.
    if not text.strip():
        return Verdict.empty(source_name)

    # (V1)
    try:
        p = smi.parse(text)
    except smi.SmilesError as e:
        return Verdict.malformed(f"{source_name}:{e.pos}", e.msg)

    # (V2)
    if not p.atoms:
        return Verdict.empty(source_name)

    # (V4) stereo tokens seen but this reader builds no stereo descriptors
    if "stereo" in required and p.saw_stereo_token:
        return Verdict.underdetermined(
            "stereo", ["reader parses stereo tokens but assigns no descriptor"]
        )

    g = ContactGraph(floor=floor)
    g.meta = {"source": source_name, "source_format": "smiles", "input": text}

    heavy_key: dict[int, str] = {}
    for a in p.atoms:
        at = Atom(idx=len(g.atoms), z=a.z, charge=a.charge, isotope=a.isotope)
        at.prov = Prov.STATED
        heavy_key[a.idx] = g.add_atom(at)

    # aromatic systems: one delocalised block, never per-bond counts
    arom_bond = set()
    for sysid, comp in enumerate(p.aromatic_rings):
        members = [heavy_key[i] for i in comp]
        n_bonds = sum(
            1 for b in p.bonds if b.aromatic and b.a in comp and b.b in comp
        )
        g.delocs[sysid] = {
            "members": members,
            "sigma_cells": n_bonds,
            "delocalised_cells": max(0, len(comp) // 2),
            "total_cells": n_bonds + max(0, len(comp) // 2),
            "provenance": str(Prov.SUPPLIED),
            "convention": smi.CONV_AROMATIC,
            "per_bond_cells": None,  # Prop: not a per-bond attribute
        }
        for b in p.bonds:
            if b.aromatic and b.a in comp and b.b in comp:
                arom_bond.add((min(b.a, b.b), max(b.a, b.b)))
                g.add_contact(
                    Contact(
                        u=heavy_key[b.a],
                        v=heavy_key[b.b],
                        weight=_bond_weight(1, floor),
                        cells=1,
                        deloc_id=sysid,
                        prov=Prov.SUPPLIED,
                        convention=smi.CONV_AROMATIC,
                    )
                )

    for b in p.bonds:
        pair = (min(b.a, b.b), max(b.a, b.b))
        if pair in arom_bond:
            continue
        g.add_contact(
            Contact(
                u=heavy_key[b.a],
                v=heavy_key[b.b],
                weight=_bond_weight(b.cells, floor),
                cells=b.cells,
                prov=Prov.STATED if b.stated else Prov.SUPPLIED,
                convention=None if b.stated else smi.CONV_DEFAULT_BOND,
            )
        )

    # implicit and bracket hydrogens
    for a in p.atoms:
        n_impl = p.implicit_h.get(a.idx, 0)
        for _ in range(n_impl):
            h = Atom(idx=len(g.atoms), z=1)
            h.prov = Prov.SUPPLIED
            h.convention = smi.CONV_IMPLICIT_H
            hk = g.add_atom(h)
            g.add_contact(
                Contact(
                    u=heavy_key[a.idx],
                    v=hk,
                    weight=_bond_weight(1, floor),
                    cells=1,
                    prov=Prov.SUPPLIED,
                    convention=smi.CONV_IMPLICIT_H,
                )
            )
        for _ in range(a.explicit_h):
            h = Atom(idx=len(g.atoms), z=1)
            h.prov = Prov.STATED  # bracket H is written in the record
            hk = g.add_atom(h)
            g.add_contact(
                Contact(
                    u=heavy_key[a.idx],
                    v=hk,
                    weight=_bond_weight(1, floor),
                    cells=1,
                    prov=Prov.STATED,
                )
            )

    # medium edges from unshared regions
    committed: dict[str, int] = {k: 0 for k in g.atoms}
    for c in g.contacts.values():
        if not c.is_medium_edge:
            committed[c.u] = committed.get(c.u, 0) + c.cells
            committed[c.v] = committed.get(c.v, 0) + c.cells

    def unshared_for(atom: Atom) -> int:
        capv, _occ, _nu = valence(atom.z)
        used = committed.get(atom.key, 0)
        return max(1, (capv - used) // 2) if capv > 2 else 1

    g.link_medium(lambda a: _medium_weight(unshared_for(a), floor))

    errs = g.validate()
    if errs:
        return Verdict.subfloor(0.0, floor, "; ".join(errs))

    return Verdict.translated(g, g.supplied_fraction())


def translate_xyz(
    text: str, required: set[str] | None = None, floor: float = 1.0,
    source_name: str = "<xyz>",
) -> Verdict:
    """Translate an XYZ record.  No bond block: connectivity is unsupported."""
    required = set(required or {"element", "coords3d"})
    miss = cap.missing("xyz", required)
    if miss:
        return Verdict.unsupported(list(miss), list(cap.capability("xyz")))

    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return Verdict.empty(source_name)
    try:
        n = int(lines[0].split()[0])
    except (ValueError, IndexError):
        return Verdict.malformed(f"{source_name}:1", "first line is not an atom count")
    body = lines[2 : 2 + n] if len(lines) > 2 else []
    if not body:
        return Verdict.empty(source_name)

    from ..core.chem import Z_OF

    g = ContactGraph(floor=floor)
    g.meta = {"source": source_name, "source_format": "xyz"}
    for ln, line in enumerate(body):
        parts = line.split()
        if len(parts) < 4:
            return Verdict.malformed(f"{source_name}:{ln + 3}", "short atom line")
        sym = parts[0].capitalize()
        if sym not in Z_OF:
            return Verdict.malformed(f"{source_name}:{ln + 3}", f"unknown element {sym}")
        a = Atom(idx=len(g.atoms), z=Z_OF[sym])
        a.coords = tuple(float(x) for x in parts[1:4])
        a.prov = Prov.STATED
        g.add_atom(a)

    g.link_medium(lambda a: floor)
    return Verdict.incomplete(["connectivity", "cellcount"], g.supplied_fraction())


TRANSLATORS = {
    "smiles": translate_smiles,
    "xyz": translate_xyz,
}


def translate(fmt: str, text: str, **kw) -> Verdict:
    if not cap.known_format(fmt):
        return Verdict.unsupported(list(kw.get("required", {"element"})), [])
    fn = TRANSLATORS.get(fmt)
    if fn is None:
        return Verdict.unsupported(
            list(kw.get("required", {"element"})), list(cap.capability(fmt))
        )
    return fn(text, **kw)
