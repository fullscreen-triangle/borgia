"""Contact graphs, provenance and cuts.

The shared semantic object of both languages.  A contact graph is a finite
weighted graph with a distinguished ``medium`` vertex adjacent to every item,
strictly positive weights, and a floor beta > 0 bounding every weight below.

Every element carries a provenance tag.  Tags compose as the maximum under
``stated < supplied``, so a value derived from supplied data can never be
reported as stated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

MEDIUM = "__medium__"


class Prov(IntEnum):
    """Provenance tag.  Ordered so that ``max`` is composition."""

    STATED = 0
    SUPPLIED = 1

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "stated" if self is Prov.STATED else "supplied"

    @staticmethod
    def join(tags: Iterable["Prov"]) -> "Prov":
        """Compose tags.  Empty input is STATED (nothing was supplied)."""
        out = Prov.STATED
        for t in tags:
            if t > out:
                out = t
        return out


class FloorViolation(Exception):
    """Raised when an edge weight or residue falls below the floor."""


@dataclass
class Atom:
    """An item vertex."""

    idx: int
    z: int
    charge: int = 0
    isotope: int | None = None
    # committed cells against neighbours, filled by the builder
    committed: int = 0
    coords: tuple[float, float, float] | None = None
    prov: Prov = Prov.STATED
    # which convention supplied it, when supplied
    convention: str | None = None

    @property
    def key(self) -> str:
        return f"a{self.idx}"


@dataclass
class Contact:
    """An edge between two items, or between an item and the medium."""

    u: str
    v: str
    weight: float
    cells: int = 1
    deloc_id: int | None = None
    prov: Prov = Prov.STATED
    convention: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.u, self.v) if self.u <= self.v else (self.v, self.u)

    @property
    def is_medium_edge(self) -> bool:
        return self.u == MEDIUM or self.v == MEDIUM


@dataclass
class ContactGraph:
    """Finite weighted graph with an explicit medium vertex.

    Invariants enforced by :meth:`validate`:
      * every item is adjacent to the medium
      * every weight is >= floor > 0
    """

    floor: float
    atoms: dict[str, Atom] = field(default_factory=dict)
    contacts: dict[tuple[str, str], Contact] = field(default_factory=dict)
    delocs: dict[int, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- construction ---------------------------------------------------

    def add_atom(self, atom: Atom) -> str:
        self.atoms[atom.key] = atom
        return atom.key

    def add_contact(self, c: Contact) -> None:
        prev = self.contacts.get(c.key)
        if prev is not None:
            # a repeated pair merges cell counts; provenance composes
            prev.cells = max(prev.cells, c.cells)
            prev.weight = max(prev.weight, c.weight)
            prev.prov = Prov.join([prev.prov, c.prov])
            return
        self.contacts[c.key] = c

    def link_medium(self, weight_fn, convention: str = "medium-residual") -> None:
        """Attach every item to the medium.

        No source record states a medium edge, so every such edge is SUPPLIED.
        They are excluded from the supplied-fraction denominator for exactly
        that reason (see :meth:`supplied_fraction`).
        """
        for key, atom in self.atoms.items():
            w = max(weight_fn(atom), self.floor)
            self.add_contact(
                Contact(
                    u=key,
                    v=MEDIUM,
                    weight=w,
                    cells=0,
                    prov=Prov.SUPPLIED,
                    convention=convention,
                )
            )

    # -- invariants -----------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of invariant violations; empty means valid."""
        errs: list[str] = []
        if self.floor <= 0:
            errs.append(f"floor must be positive, got {self.floor}")
        for k, c in self.contacts.items():
            if c.weight < self.floor:
                errs.append(
                    f"edge {k} weight {c.weight:.6g} below floor {self.floor:.6g}"
                )
        linked = {
            a for c in self.contacts.values() if c.is_medium_edge
            for a in (c.u, c.v) if a != MEDIUM
        }
        for key in self.atoms:
            if key not in linked:
                errs.append(f"item {key} not adjacent to medium")
        return errs

    # -- provenance -----------------------------------------------------

    def supplied_fraction(self) -> float:
        """Fraction of the structure's *own* elements that were supplied.

        Medium edges are excluded from the denominator: no record states
        them, so counting them would make the statistic a property of the
        target representation rather than of the source.
        """
        elems = list(self.atoms.values()) + [
            c for c in self.contacts.values() if not c.is_medium_edge
        ]
        if not elems:
            return 0.0
        n_supp = sum(1 for e in elems if e.prov is Prov.SUPPLIED)
        return n_supp / len(elems)

    def provenance(self) -> Prov:
        """Graph-level tag: the max over its own elements."""
        return Prov.join(
            [a.prov for a in self.atoms.values()]
            + [c.prov for c in self.contacts.values() if not c.is_medium_edge]
        )

    def conventions_used(self) -> list[str]:
        names = {
            e.convention
            for e in list(self.atoms.values()) + list(self.contacts.values())
            if e.convention
        }
        return sorted(names)

    # -- cuts -----------------------------------------------------------

    def neighbours(self, key: str) -> list[tuple[str, Contact]]:
        out = []
        for c in self.contacts.values():
            if c.u == key:
                out.append((c.v, c))
            elif c.v == key:
                out.append((c.u, c))
        return out

    def cut_weight(self, side: set[str]) -> float:
        """Weight of the edge boundary of ``side``."""
        total = 0.0
        for c in self.contacts.values():
            inside_u = c.u in side
            inside_v = c.v in side
            if inside_u != inside_v:
                total += c.weight
        return total

    def cut_edges(self, side: set[str]) -> list[Contact]:
        out = []
        for c in self.contacts.values():
            if (c.u in side) != (c.v in side):
                out.append(c)
        return out

    def separation_cost(self, key: str) -> tuple[float, set[str], Prov]:
        """Minimum cut separating ``key`` from the medium.

        Returns ``(sigma, minimising side, provenance of the cut)``.
        Uses networkx max-flow when available and an exact exponential
        search on very small graphs otherwise, so the module has no hard
        dependency.
        """
        try:
            return self._sepcost_flow(key)
        except ImportError:
            return self._sepcost_bruteforce(key)

    def _sepcost_flow(self, key: str):
        import networkx as nx

        g = nx.Graph()
        for k in list(self.atoms) + [MEDIUM]:
            g.add_node(k)
        for c in self.contacts.values():
            g.add_edge(c.u, c.v, capacity=c.weight)
        cut_value, (side_a, _side_b) = nx.minimum_cut(
            g, key, MEDIUM, capacity="capacity", flow_func=None
        )
        side = set(side_a)
        prov = Prov.join([c.prov for c in self.cut_edges(side)])
        return float(cut_value), side, prov

    def _sepcost_bruteforce(self, key: str):
        items = [k for k in self.atoms if k != key]
        best = None
        best_side: set[str] = {key}
        n = len(items)
        if n > 18:
            raise RuntimeError(
                "graph too large for brute-force min-cut; install networkx"
            )
        for mask in range(1 << n):
            side = {key} | {items[i] for i in range(n) if mask >> i & 1}
            w = self.cut_weight(side)
            if best is None or w < best:
                best, best_side = w, side
        prov = Prov.join([c.prov for c in self.cut_edges(best_side)])
        return float(best), best_side, prov

    def burial_depth(self, key: str) -> int:
        """Number of items on the minimising side of ``key``'s cut.

        1 means the item separates alone (exposed); larger means it comes
        away with a neighbourhood (buried).
        """
        _sigma, side, _p = self.separation_cost(key)
        return len([s for s in side if s != MEDIUM])

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "floor": self.floor,
            "atoms": [
                {
                    "key": a.key,
                    "z": a.z,
                    "charge": a.charge,
                    "isotope": a.isotope,
                    "coords": list(a.coords) if a.coords else None,
                    "provenance": str(a.prov),
                    "convention": a.convention,
                }
                for a in self.atoms.values()
            ],
            "contacts": [
                {
                    "u": c.u,
                    "v": c.v,
                    "weight": round(c.weight, 9),
                    "cells": c.cells,
                    "deloc_id": c.deloc_id,
                    "medium_edge": c.is_medium_edge,
                    "provenance": str(c.prov),
                    "convention": c.convention,
                }
                for c in self.contacts.values()
            ],
            "delocalised_systems": [
                {"id": k, **v} for k, v in sorted(self.delocs.items())
            ],
            "provenance": str(self.provenance()),
            "supplied_fraction": round(self.supplied_fraction(), 6),
            "conventions_used": self.conventions_used(),
            "meta": self.meta,
        }
