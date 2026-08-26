"""Meibutsu: an inverse instrument for chemical structure.

A conventional instrument maps an analyte to a measured value.  This
module implements the opposite direction: it accepts a measured value --
a vibrational spectrum, a coordinate triple, or an address -- and returns
the structure that would produce it.

The implementation is deliberately small and has no dependencies beyond
numpy.  Every quantity it computes is either read from a public reference
database or derived here in closed form; nothing is fitted.

Three layers, in dependency order:

  1. ADDRESSING.  A spectrum maps to a point in the unit cube and thence
     to a base-3 address.  This layer is taken from the reference
     implementation of the compound database and is not re-derived.

  2. OBSERVATION.  An address is realised as a complex field on a
     discrete grid -- an amplitude and a phase at each grid point.  This
     is the quantity a fragment shader would evaluate per pixel.

  3. COMPARISON.  Two observation fields are compared by superposition.
     The comparison is a normalised correlation of the two fields, which
     is the fringe visibility of their superposition.

The third layer is where the design differs from a conventional
pipeline: no feature is extracted from either field, and no similarity
function is evaluated on a pair of feature vectors.  The fields are
added, and the visibility of the resulting fringes is read off.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field as _field

import numpy as np

_DB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "categorical-compound-database")
sys.path.insert(0, os.path.abspath(_DB))

from categorical_compound_database import (  # noqa: E402
    COMPOUNDS, CategoricalEncoder, coords_to_trits,
)

# --------------------------------------------------------------------
# Reference scales.  These are properties of the reference database, not
# free parameters: OMEGA_REF is the highest fundamental it contains.
# --------------------------------------------------------------------

OMEGA_REF = 4401.0          # cm^-1, H2 stretch
GRID = 256                  # observation grid points per axis
DEPTH = 18                  # address length in trits

_ENC = CategoricalEncoder(COMPOUNDS)


# ====================================================================
#  Layer 1: addressing
# ====================================================================

def coordinates(modes, b_rot=None):
    """Map a spectrum to its coordinate triple in the unit cube."""
    return (_ENC.compute_S_k(list(modes), b_rot),
            _ENC.compute_S_t(list(modes), b_rot),
            _ENC.compute_S_e(list(modes)))


def address(coords, depth=DEPTH):
    """Map a coordinate triple to its base-3 address."""
    sk, st, se = coords
    return "".join(str(t) for t in coords_to_trits(sk, st, se, depth=depth))


def common_prefix(a: str, b: str) -> int:
    """Length of the longest shared address prefix."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ====================================================================
#  Layer 2: observation
# ====================================================================

@dataclass
class Observation:
    """A complex field on a grid: what one evaluation pass produces.

    ``amp`` and ``phase`` are what a shader writes to two channels of a
    texture; ``field`` is their complex combination.  Storing all three
    is redundant by design -- the redundancy is what lets the comparison
    layer work on the field while the diagnostics work on amplitude and
    phase separately.
    """
    name: str
    coords: tuple
    modes: list
    amp: np.ndarray
    phase: np.ndarray

    @property
    def field(self) -> np.ndarray:
        return self.amp * np.exp(1j * self.phase)

    @property
    def energy(self) -> float:
        return float(np.vdot(self.field, self.field).real)


def observe(modes, coords=None, b_rot=None, grid=GRID, name="") -> Observation:
    """Evaluate the observation field for one spectrum.

    Amplitude places one lobe per vibrational mode at its normalised
    frequency address, with a width set by the first coordinate and an
    envelope set by the second.  Phase accumulates linearly with the
    mode frequencies and with the third coordinate.

    The phase is the load-bearing part.  Amplitude alone would make two
    molecules with the same mode positions indistinguishable regardless
    of their coordinates; the phase term is what carries the coordinate
    information into the comparison.
    """
    modes = list(modes)
    if coords is None:
        coords = coordinates(modes, b_rot)
    sk, st, se = coords

    u = np.linspace(0.0, 1.0, grid)
    width = 0.05 * (1.0 - 0.5 * sk) + 1e-6

    amp = np.zeros(grid)
    for om in modes:
        amp += np.exp(-((u - om / OMEGA_REF) ** 2) / (width ** 2))
    amp *= np.exp(-((u - 0.5) ** 2) / ((0.1 + 0.4 * st) ** 2))

    phase = 2.0 * math.pi * se * u * 8.0
    for om in modes:
        phase = phase + 2.0 * math.pi * (om / OMEGA_REF) * u

    return Observation(name=name, coords=tuple(coords), modes=modes,
                       amp=amp, phase=phase)


# ====================================================================
#  Layer 3: comparison by superposition
# ====================================================================

def superpose(a: Observation, b: Observation) -> np.ndarray:
    """The superposed intensity |A + B|^2.

    This is the only place two observations meet.  Note that it is an
    addition, not a comparison: the two fields coexist on one grid and
    the interference is what results.
    """
    s = a.field + b.field
    return np.abs(s) ** 2


def visibility(a: Observation, b: Observation) -> float:
    """Fringe visibility of the superposition of two observations.

    Defined as the normalised field correlation

        V = |<A, B>| / sqrt(<A,A> <B,B>)

    which is exactly 1 when the two fields are identical (Cauchy-Schwarz
    with equality iff the fields are proportional) and 0 when they are
    orthogonal.  No assumption about the phase distribution is required
    for the self-comparison case, which is why this form is used in
    preference to a pointwise expression averaged over the grid.

    The cross-term of the superposition carries the entire relational
    content: |A+B|^2 = |A|^2 + |B|^2 + 2 Re<A,B>.  The first two terms
    are properties of each observation alone.
    """
    fa, fb = a.field, b.field
    den = math.sqrt(a.energy * b.energy)
    if den <= 0.0:
        return 0.0
    return float(np.abs(np.vdot(fa, fb)) / den)


def cross_term(a: Observation, b: Observation) -> np.ndarray:
    """The relational part of the superposition, pointwise.

    Everything that depends on BOTH observations lives here; the
    remaining terms of |A+B|^2 would be present if the other observation
    were absent.
    """
    return 2.0 * np.real(np.conj(a.field) * b.field)


def stack(observations) -> np.ndarray:
    """Superpose an arbitrary number of observations at once.

    Superposition is linear, so this is a single addition regardless of
    how many observations are supplied -- no pairwise loop appears.  The
    resulting intensity contains every pairwise cross-term.
    """
    total = np.zeros_like(observations[0].field)
    for o in observations:
        total = total + o.field
    return np.abs(total) ** 2


def stacked_cross_energy(observations) -> float:
    """Total relational content of a stack.

    |sum A_i|^2 summed over the grid, minus the sum of the individual
    energies, leaves exactly the sum of all pairwise cross-terms.  This
    identity is what makes bulk comparison possible: the quantity is
    obtained from one superposition, not from C(n,2) comparisons.
    """
    tot = float(np.sum(stack(observations)))
    own = sum(o.energy for o in observations)
    return tot - own


# ====================================================================
#  The inverse direction
# ====================================================================

@dataclass
class Resolution:
    """What the instrument returns when given a measured value.

    ``occupants`` may be empty (the address names an unoccupied cell),
    a single structure, or several (the cell is not resolved at this
    depth).  All three are reported rather than collapsed, because the
    distinction between them is the instrument's own statement about
    the resolution it achieved.
    """
    query_coords: tuple
    query_address: str
    depth: int
    occupants: list = _field(default_factory=list)
    fallback_depth: int = 0
    ranked: list = _field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return len(self.occupants) == 1


class Instrument:
    """The inverse instrument over a reference set of structures."""

    def __init__(self, compounds=None, grid=GRID, depth=DEPTH):
        self.db = compounds if compounds is not None else COMPOUNDS
        self.grid = grid
        self.depth = depth
        self.records = {}
        for name in self.db:
            rec = _ENC.encode(name)
            self.records[name] = rec
        # Coordinates are recomputed rather than read back from the
        # stored record: the encoder rounds to six decimals when it
        # stores, and feeding a rounded coordinate into the field
        # perturbs every downstream visibility at the 1e-6 level for no
        # modelling reason.  The rounding is a property of the storage
        # format, not of the construction.
        self.obs = {
            n: observe(r["modes"],
                       coordinates(r["modes"], self.db[n].get("B_rot")),
                       grid=grid, name=n)
            for n, r in self.records.items()
        }
        self.addresses = {n: r["trit_string"] for n, r in self.records.items()}

    # -- inverse: measured value in, structure out --------------------

    def resolve(self, modes, b_rot=None, depth=None, rank=True) -> Resolution:
        """Given a spectrum, return the structures that would produce it."""
        depth = depth or self.depth
        coords = coordinates(modes, b_rot)
        addr = address(coords, depth)

        occupants = [n for n, a in self.addresses.items()
                     if a[:depth] == addr[:depth]]

        # fall back to the deepest occupied prefix
        fb = depth
        while not occupants and fb > 0:
            fb -= 1
            occupants = [n for n, a in self.addresses.items()
                         if a[:fb] == addr[:fb]]

        ranked = []
        if rank:
            q = observe(modes, coords, grid=self.grid, name="<query>")
            ranked = sorted(
                ((n, visibility(q, self.obs[n])) for n in self.db),
                key=lambda kv: -kv[1])
        return Resolution(query_coords=coords, query_address=addr,
                          depth=depth, occupants=sorted(occupants),
                          fallback_depth=fb, ranked=ranked)

    # -- comparison ---------------------------------------------------

    def visibility_matrix(self):
        names = sorted(self.obs)
        n = len(names)
        M = np.zeros((n, n))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                M[i, j] = visibility(self.obs[a], self.obs[b])
        return names, M

    def bulk(self, names=None):
        """One superposition over many structures."""
        names = names or sorted(self.obs)
        return stack([self.obs[n] for n in names])


__all__ = [
    "OMEGA_REF", "GRID", "DEPTH", "coordinates", "address", "common_prefix",
    "Observation", "observe", "superpose", "visibility", "cross_term",
    "stack", "stacked_cross_energy", "Resolution", "Instrument",
]
