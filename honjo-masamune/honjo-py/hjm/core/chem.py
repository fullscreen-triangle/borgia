"""Shell filling, vacancy, and the geometry table.

The only chemistry in the package.  Everything here is the partition
arithmetic: capacity C(n) = 2n^2, filling under exclusion and the Madelung
order, and the vacancy nu = C_v - q_v.
"""

from __future__ import annotations

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
).split()

Z_OF = {s: i + 1 for i, s in enumerate(SYMBOLS)}

#: subshells in Madelung (n + l, then n) order, with capacities
_ORDER = [
    ("1s", 1, 2), ("2s", 2, 2), ("2p", 2, 6), ("3s", 3, 2), ("3p", 3, 6),
    ("4s", 4, 2), ("3d", 3, 10), ("4p", 4, 6), ("5s", 5, 2), ("4d", 4, 10),
    ("5p", 5, 6), ("6s", 6, 2), ("4f", 4, 14), ("5d", 5, 10), ("6p", 6, 6),
]

#: valence capacity by period for main-group filling
_DUET = {1, 2}


def symbol(z: int) -> str:
    return SYMBOLS[z - 1] if 1 <= z <= len(SYMBOLS) else f"Z{z}"


def shell_capacity(n: int) -> int:
    """C(n) = 2 n^2."""
    return 2 * n * n


def configuration(z: int) -> list[tuple[str, int]]:
    """Fill z electrons in Madelung order.  Returns [(subshell, count)]."""
    remaining = z
    out: list[tuple[str, int]] = []
    for name, _n, cap in _ORDER:
        if remaining <= 0:
            break
        k = min(cap, remaining)
        out.append((name, k))
        remaining -= k
    return out


def valence(z: int) -> tuple[int, int, int]:
    """Return ``(capacity, occupancy, vacancy)`` of the valence shell.

    Main-group treatment: the valence shell is the outermost principal
    level; its capacity is 2 for the n=1 duet and 8 for an s+p octet.
    d-block occupancy is not treated (see module note in the paper's
    limitations) and returns capacity 8 with the s+p count only.
    """
    cfg = configuration(z)
    if not cfg:
        return (2, 0, 2)
    # outermost principal level actually occupied by s or p
    top_n = 0
    for name, k in cfg:
        n = int(name[0])
        if name[1] in "sp" and k > 0:
            top_n = max(top_n, n)
    cap = 2 if top_n == 1 else 8
    occ = sum(k for name, k in cfg if int(name[0]) == top_n and name[1] in "sp")
    return (cap, occ, cap - occ)


def vacancy(z: int) -> int:
    return valence(z)[2]


def is_noble(z: int) -> bool:
    return vacancy(z) == 0


def covalent_valence(z: int) -> int:
    """min(nu, C_v - nu): the number of interfaces the atom commits."""
    cap, _occ, nu = valence(z)
    return min(nu, cap - nu)


#: default implicit-hydrogen valences for the SMILES organic subset
ORGANIC_SUBSET_VALENCE = {
    5: (3,), 6: (4,), 7: (3, 5), 8: (2,), 15: (3, 5),
    16: (2, 4, 6), 9: (1,), 17: (1,), 35: (1,), 53: (1,),
}

#: maximal angular separation of k interface regions on the 2-sphere
IDEAL_ANGLE = {2: 180.0, 3: 120.0, 4: 109.47, 5: 90.0, 6: 90.0}


def geometry(regions: int, lone_pairs: int = 0) -> dict:
    """Ideal geometry for a centre with ``regions`` total interface regions.

    Lone pairs occupy regions and compress the bonded angles; the
    compression magnitude is not computed (the framework gives the
    maximal-separation configuration, not the quantitative correction).
    """
    ideal = IDEAL_ANGLE.get(regions)
    names = {2: "linear", 3: "trigonal planar", 4: "tetrahedral",
             5: "trigonal bipyramidal", 6: "octahedral"}
    return {
        "regions": regions,
        "lone_pairs": lone_pairs,
        "bonded": regions - lone_pairs,
        "ideal_angle_deg": ideal,
        "shape": names.get(regions),
        "compressed_by_lone_pairs": lone_pairs > 0,
        "quantitative_correction": None,  # not derived by the framework
    }
