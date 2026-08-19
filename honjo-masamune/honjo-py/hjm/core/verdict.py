"""Verdicts.

Every operation that can fail returns a labelled verdict, and only one
label carries a value.  A value-or-nothing interface conflates the failure
labels; the point of this module is that they stay distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Label(str, Enum):
    # masamune translation labels
    TRANSLATED = "translated"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    UNDERDETERMINED = "underdetermined"
    EMPTY = "empty"
    # honjo evaluation labels
    CUT = "cut"
    UNCLOSED = "unclosed"
    INERT = "inert"
    NONCONVERGENT = "nonconvergent"
    UNDERPROVENANCED = "underprovenanced"
    SUBFLOOR = "subfloor"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: the only two labels whose payload contains a value
VALUE_BEARING = {Label.TRANSLATED, Label.CUT}


@dataclass
class Verdict:
    """A labelled outcome.  ``value`` is non-None only for value-bearing
    labels; this is checked in ``__post_init__`` so the invariant cannot be
    violated by construction."""

    label: Label
    payload: dict = field(default_factory=dict)
    value: Any = None

    def __post_init__(self) -> None:
        if self.value is not None and self.label not in VALUE_BEARING:
            raise ValueError(
                f"label {self.label} must not carry a value "
                f"(only {sorted(l.value for l in VALUE_BEARING)} may)"
            )

    @property
    def ok(self) -> bool:
        return self.label in VALUE_BEARING

    def to_dict(self) -> dict:
        out: dict = {"verdict": str(self.label), "payload": self.payload}
        if self.value is not None and hasattr(self.value, "to_dict"):
            out["value"] = self.value.to_dict()
        elif self.value is not None:
            out["value"] = self.value
        else:
            out["value"] = None
        return out

    # -- constructors ---------------------------------------------------

    @staticmethod
    def translated(graph, supplied: float) -> "Verdict":
        return Verdict(
            Label.TRANSLATED, {"supplied_fraction": round(supplied, 6)}, graph
        )

    @staticmethod
    def cut(value) -> "Verdict":
        return Verdict(Label.CUT, {}, value)

    @staticmethod
    def unsupported(missing: list[str], have: list[str]) -> "Verdict":
        return Verdict(
            Label.UNSUPPORTED,
            {"missing_features": sorted(missing), "source_capability": sorted(have)},
        )

    @staticmethod
    def malformed(where: str, detail: str) -> "Verdict":
        return Verdict(Label.MALFORMED, {"position": where, "detail": detail})

    @staticmethod
    def empty(where: str) -> "Verdict":
        return Verdict(Label.EMPTY, {"source": where, "certified": True})

    @staticmethod
    def underdetermined(element: str, readings: list[str]) -> "Verdict":
        return Verdict(
            Label.UNDERDETERMINED, {"element": element, "readings": readings}
        )

    @staticmethod
    def incomplete(missing: list[str], supplied: float | None = None) -> "Verdict":
        p: dict = {"absent_features": sorted(missing)}
        if supplied is not None:
            p["supplied_fraction"] = round(supplied, 6)
        return Verdict(Label.INCOMPLETE, p)

    @staticmethod
    def unclosed(open_atoms: list[dict]) -> "Verdict":
        return Verdict(Label.UNCLOSED, {"open": open_atoms})

    @staticmethod
    def inert(participants: list[dict]) -> "Verdict":
        return Verdict(
            Label.INERT, {"closed_shell": participants, "certified_vacancy_zero": True}
        )

    @staticmethod
    def nonconvergent(chain: list, alignment: float) -> "Verdict":
        return Verdict(
            Label.NONCONVERGENT,
            {"chain_length": len(chain), "terminal_alignment": round(alignment, 9)},
        )

    @staticmethod
    def underprovenanced(requirement: str, measured: float) -> "Verdict":
        return Verdict(
            Label.UNDERPROVENANCED,
            {"requirement": requirement, "measured_supplied": round(measured, 6)},
        )

    @staticmethod
    def subfloor(found: float, floor: float, where: str = "") -> "Verdict":
        return Verdict(
            Label.SUBFLOOR,
            {"found": found, "floor": floor, "where": where},
        )
