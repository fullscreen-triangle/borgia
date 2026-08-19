"""Feature alphabet and per-format capability sets.

A capability set is a claim about *this reader*, not about the format
specification.  Under-declaring is safe; over-declaring is unsound and
silent.  These declarations are deliberately narrow.
"""

from __future__ import annotations

FEATURES = {
    "element",
    "connectivity",
    "cellcount",
    "delocalisation",
    "charge",
    "isotope",
    "hcount",
    "stereo",
    "coords3d",
    "conformer",
    "provenance",
}

#: What each reader in this package can faithfully extract.
CAPABILITY: dict[str, set[str]] = {
    "smiles": {
        "element", "connectivity", "cellcount", "delocalisation",
        "charge", "isotope", "hcount",
    },
    # stereo is NOT declared: this reader parses '/' '\' '@' tokens but
    # does not build stereo descriptors, so declaring it would be an
    # over-declaration.
    "molfile": {
        "element", "connectivity", "cellcount", "charge", "isotope",
        "coords3d",
    },
    "sdf": {
        "element", "connectivity", "cellcount", "charge", "isotope",
        "coords3d", "conformer", "provenance",
    },
    "xyz": {"element", "coords3d"},
    "inchi": set(),  # no reader implemented; every request is unsupported
}


def capability(fmt: str) -> set[str]:
    return CAPABILITY.get(fmt, set())


def missing(fmt: str, required: set[str]) -> set[str]:
    return set(required) - capability(fmt)


def known_format(fmt: str) -> bool:
    return fmt in CAPABILITY
