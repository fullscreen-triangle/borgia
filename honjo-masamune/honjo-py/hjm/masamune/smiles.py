"""A SMILES reader that records what it supplied.

Self-contained: no toolkit dependency.  Covers the organic subset,
bracket atoms, ring-closure bonds, branches, explicit bond orders and
lower-case aromatic atoms.

The distinguishing feature is that every atom and bond it produces is
tagged ``stated`` or ``supplied``, and every supplied element names the
convention that produced it.  Implicit hydrogens are supplied by the
organic-subset valence convention; aromatic rings are marked as
delocalised systems rather than being assigned Kekule cell counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.chem import ORGANIC_SUBSET_VALENCE, Z_OF
from ..core.graph import Prov

ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
AROMATIC_LOWER = {"b", "c", "n", "o", "p", "s"}

BOND_CELLS = {"-": 1, "=": 2, "#": 3, ":": 1, "/": 1, "\\": 1}

CONV_IMPLICIT_H = "smiles:organic-subset-implicit-hydrogen"
CONV_AROMATIC = "smiles:aromatic-lowercase-delocalised"
CONV_DEFAULT_BOND = "smiles:adjacent-atoms-single-bond"


class SmilesError(Exception):
    def __init__(self, msg: str, pos: int):
        super().__init__(msg)
        self.pos = pos
        self.msg = msg


@dataclass
class PAtom:
    idx: int
    z: int
    symbol: str
    charge: int = 0
    isotope: int | None = None
    aromatic: bool = False
    explicit_h: int = 0
    h_stated: bool = False       # H count came from brackets
    bracket: bool = False


@dataclass
class PBond:
    a: int
    b: int
    cells: int
    aromatic: bool = False
    stated: bool = True          # False when adjacency implied a single bond
    order_symbol: str = "-"


@dataclass
class ParseResult:
    atoms: list[PAtom] = field(default_factory=list)
    bonds: list[PBond] = field(default_factory=list)
    implicit_h: dict[int, int] = field(default_factory=dict)
    aromatic_rings: list[list[int]] = field(default_factory=list)
    saw_stereo_token: bool = False


def _elem_z(sym: str) -> int:
    s = sym.capitalize() if len(sym) > 1 else sym.upper()
    if s not in Z_OF:
        raise SmilesError(f"unknown element {sym!r}", 0)
    return Z_OF[s]


def parse(text: str) -> ParseResult:
    """Parse a SMILES string into atoms, bonds and implicit-H counts."""
    s = text.strip()
    if not s:
        raise SmilesError("empty input", 0)

    res = ParseResult()
    stack: list[int] = []
    prev: int | None = None
    pending_bond: str | None = None
    ring: dict[int, tuple[int, str | None]] = {}
    i = 0

    def add_atom(a: PAtom) -> int:
        res.atoms.append(a)
        return a.idx

    while i < len(s):
        ch = s[i]

        if ch == "(":
            if prev is None:
                raise SmilesError("branch before any atom", i)
            stack.append(prev)
            i += 1
            continue
        if ch == ")":
            if not stack:
                raise SmilesError("unbalanced ')'", i)
            prev = stack.pop()
            i += 1
            continue
        if ch in BOND_CELLS:
            if ch in "/\\":
                res.saw_stereo_token = True
            pending_bond = ch
            i += 1
            continue
        if ch == ".":
            prev = None
            i += 1
            continue

        if ch.isdigit() or ch == "%":
            if ch == "%":
                num = int(s[i + 1 : i + 3])
                i += 3
            else:
                num = int(ch)
                i += 1
            if prev is None:
                raise SmilesError("ring bond before any atom", i)
            if num in ring:
                other, sym = ring.pop(num)
                bsym = pending_bond or sym
                arom = (
                    res.atoms[other].aromatic
                    and res.atoms[prev].aromatic
                    and bsym is None
                )
                res.bonds.append(
                    PBond(
                        a=other,
                        b=prev,
                        cells=BOND_CELLS.get(bsym or "-", 1),
                        aromatic=arom,
                        stated=bsym is not None,
                        order_symbol=bsym or "-",
                    )
                )
            else:
                ring[num] = (prev, pending_bond)
            pending_bond = None
            continue

        # bracket atom
        if ch == "[":
            j = s.find("]", i)
            if j < 0:
                raise SmilesError("unterminated '['", i)
            body = s[i + 1 : j]
            atom = _parse_bracket(body, i)
            atom.idx = len(res.atoms)
            idx = add_atom(atom)
            _connect(res, prev, idx, pending_bond)
            pending_bond = None
            prev = idx
            i = j + 1
            continue

        # organic subset / aromatic lower-case
        two = s[i : i + 2]
        if two in ("Cl", "Br"):
            sym, step = two, 2
        elif ch.isalpha():
            sym, step = ch, 1
        else:
            raise SmilesError(f"unexpected character {ch!r}", i)

        arom = sym in AROMATIC_LOWER
        # normalise to the element's own casing: 'c' -> 'C', 'Cl' -> 'Cl'.
        # Comparing an upper-cased two-letter symbol against the subset
        # would reject 'Cl' and 'Br', which are in it.
        upper = sym.capitalize() if len(sym) > 1 else sym.upper()
        if upper not in ORGANIC_SUBSET:
            raise SmilesError(f"{sym!r} outside organic subset; use brackets", i)

        atom = PAtom(
            idx=len(res.atoms),
            z=_elem_z(sym),
            symbol=upper.capitalize() if len(upper) > 1 else upper,
            aromatic=arom,
        )
        idx = add_atom(atom)
        _connect(res, prev, idx, pending_bond)
        pending_bond = None
        prev = idx
        i += step

    if ring:
        raise SmilesError(f"unclosed ring bond(s) {sorted(ring)}", len(s))
    if stack:
        raise SmilesError("unbalanced '('", len(s))

    _fill_implicit_h(res)
    res.aromatic_rings = _aromatic_systems(res)
    return res


def _connect(res: ParseResult, prev: int | None, idx: int, bsym: str | None) -> None:
    if prev is None:
        return
    arom = res.atoms[prev].aromatic and res.atoms[idx].aromatic and bsym is None
    res.bonds.append(
        PBond(
            a=prev,
            b=idx,
            cells=BOND_CELLS.get(bsym or "-", 1),
            aromatic=arom,
            stated=bsym is not None,
            order_symbol=bsym or "-",
        )
    )


def _parse_bracket(body: str, pos: int) -> PAtom:
    i = 0
    iso = ""
    while i < len(body) and body[i].isdigit():
        iso += body[i]
        i += 1
    sym = ""
    if i < len(body) and body[i].isalpha():
        sym += body[i]
        i += 1
        if i < len(body) and body[i].islower() and (sym + body[i]).capitalize() in Z_OF:
            sym += body[i]
            i += 1
    if not sym:
        raise SmilesError("bracket atom with no element", pos)
    arom = sym[0].islower()

    h = 0
    h_stated = False
    charge = 0
    while i < len(body):
        c = body[i]
        if c == "H":
            h_stated = True
            i += 1
            n = ""
            while i < len(body) and body[i].isdigit():
                n += body[i]
                i += 1
            h = int(n) if n else 1
        elif c in "+-":
            sign = 1 if c == "+" else -1
            i += 1
            n = ""
            while i < len(body) and body[i].isdigit():
                n += body[i]
                i += 1
            if n:
                charge = sign * int(n)
            else:
                run = 1
                while i < len(body) and body[i] == c:
                    run += 1
                    i += 1
                charge = sign * run
        elif c == "@":
            i += 1
            while i < len(body) and body[i] == "@":
                i += 1
        else:
            i += 1

    return PAtom(
        idx=-1,
        z=_elem_z(sym),
        symbol=sym.capitalize() if len(sym) > 1 else sym.upper(),
        charge=charge,
        isotope=int(iso) if iso else None,
        aromatic=arom,
        explicit_h=h,
        h_stated=h_stated,
        bracket=True,
    )


def _fill_implicit_h(res: ParseResult) -> None:
    """Supply hydrogens for organic-subset atoms that did not state them."""
    used = {a.idx: 0 for a in res.atoms}
    for b in res.bonds:
        # an aromatic bond contributes one sigma cell for valence counting
        c = 1 if b.aromatic else b.cells
        used[b.a] += c
        used[b.b] += c

    for a in res.atoms:
        if a.bracket:
            # bracket atoms state their H count (possibly zero)
            res.implicit_h[a.idx] = 0
            continue
        targets = ORGANIC_SUBSET_VALENCE.get(a.z)
        if not targets:
            res.implicit_h[a.idx] = 0
            continue
        # aromatic ring atoms use one extra cell for the delocalised system
        need = used[a.idx] + (1 if a.aromatic else 0)
        target = next((t for t in targets if t >= need), targets[-1])
        res.implicit_h[a.idx] = max(0, target - need)


def _aromatic_systems(res: ParseResult) -> list[list[int]]:
    """Connected components of the aromatic-bond subgraph."""
    adj: dict[int, set[int]] = {}
    for b in res.bonds:
        if b.aromatic:
            adj.setdefault(b.a, set()).add(b.b)
            adj.setdefault(b.b, set()).add(b.a)
    seen: set[int] = set()
    out: list[list[int]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            stack.extend(adj[n] - seen)
        if len(comp) >= 3:
            out.append(sorted(comp))
    return out
