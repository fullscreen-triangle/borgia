"""Test corpora.

Small, hand-checkable, and chosen so that the expected answer is known
independently of the method under test.  Where a reference value is
asserted (automorphism orbit counts) it is derived by inspection of the
molecular symmetry, not from the implementation.
"""

from __future__ import annotations

#: (name, SMILES, heavy-atom automorphism orbit count)
#: Orbit counts are by inspection of the constitutional graph's symmetry.
SYMMETRIC = [
    ("benzene",       "c1ccccc1",              1),
    ("cyclohexane",   "C1CCCCC1",              1),
    ("ethane",        "CC",                    1),
    ("naphthalene",   "c1ccc2ccccc2c1",        3),
    ("anthracene",    "c1ccc2cc3ccccc3cc2c1",  4),
    ("biphenyl",      "c1ccc(cc1)c1ccccc1",    4),
    ("p-xylene",      "Cc1ccc(C)cc1",          3),
    ("propane",       "CCC",                   2),
    ("neopentane",    "CC(C)(C)C",             2),
    ("pyrazine",      "c1cnccn1",              2),
    ("pyridine",      "c1ccncc1",              4),
    ("toluene",       "Cc1ccccc1",             5),
    ("water-like",    "O",                     1),
    ("methane",       "C",                     1),
    ("ethylene",      "C=C",                   1),
    ("acetylene",     "C#C",                   1),
    ("p-benzoquinone","O=C1C=CC(=O)C=C1",      3),
    # 1,2,4,5-tetramethylbenzene: orbits are {4 methyl C},
    # {4 substituted ring C}, {2 unsubstituted ring CH} = 3
    ("durene",        "Cc1cc(C)c(C)cc1C",      3),
]

#: general drug-like set, no reference values asserted
DRUGLIKE = [
    ("ethanol",       "CCO"),
    ("acetic-acid",   "CC(=O)O"),
    ("acetone",       "CC(C)=O"),
    ("phenol",        "Oc1ccccc1"),
    ("aniline",       "Nc1ccccc1"),
    ("anisole",       "COc1ccccc1"),
    ("benzoic-acid",  "OC(=O)c1ccccc1"),
    ("aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol",   "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("styrene",       "C=Cc1ccccc1"),
    ("indole",        "c1ccc2[nH]ccc2c1"),
    ("quinoline",     "c1ccc2ncccc2c1"),
    ("nicotinamide",  "NC(=O)c1cccnc1"),
    ("salicylic",     "OC(=O)c1ccccc1O"),
    ("catechol",      "Oc1ccccc1O"),
    ("resorcinol",    "Oc1cccc(O)c1"),
    ("hydroquinone",  "Oc1ccc(O)cc1"),
    ("cresol",        "Cc1ccccc1O"),
    ("xylene-o",      "Cc1ccccc1C"),
    ("chlorobenzene", "Clc1ccccc1"),
    ("nitro-like",    "Cc1ccc(cc1)C(=O)O"),
    ("phenylalanine", "NC(Cc1ccccc1)C(=O)O"),
    ("tyrosine",      "NC(Cc1ccc(O)cc1)C(=O)O"),
    ("glycine",       "NCC(=O)O"),
    ("alanine",       "CC(N)C(=O)O"),
    ("serine",        "OCC(N)C(=O)O"),
    ("valine",        "CC(C)C(N)C(=O)O"),
    ("propanol",      "CCCO"),
    ("butanol",       "CCCCO"),
]

#: bioisostere and near-miss pairs with an expected relation:
#:   "close"  -- widely treated as interchangeable in medicinal chemistry
#:   "far"    -- same size, different structural class
ISOSTERE_PAIRS = [
    ("benzene",  "c1ccccc1",      "pyridine",     "c1ccncc1",       "close"),
    ("phenol",   "Oc1ccccc1",     "aniline",      "Nc1ccccc1",      "close"),
    ("phenol",   "Oc1ccccc1",     "thiophenol",   "Sc1ccccc1",      "close"),
    ("benzene",  "c1ccccc1",      "pyrazine",     "c1cnccn1",       "close"),
    ("pyridine", "c1ccncc1",      "pyrimidine",   "c1cncnc1",       "close"),
    ("catechol", "Oc1ccccc1O",    "resorcinol",   "Oc1cccc(O)c1",   "close"),
    ("benzene",  "c1ccccc1",      "cyclohexane",  "C1CCCCC1",       "far"),
    ("ethanol",  "CCO",           "benzene",      "c1ccccc1",       "far"),
    ("propane",  "CCC",           "pyridine",     "c1ccncc1",       "far"),
    ("acetone",  "CC(C)=O",       "naphthalene",  "c1ccc2ccccc2c1", "far"),
]

#: matched pairs differing at one position, for the correspondence test
MATCHED_PAIRS = [
    ("benzene",     "c1ccccc1",                  "toluene",      "Cc1ccccc1"),
    ("phenol",      "Oc1ccccc1",                 "cresol",       "Cc1ccccc1O"),
    ("benzoic",     "OC(=O)c1ccccc1",            "salicylic",    "OC(=O)c1ccccc1O"),
    ("aniline",     "Nc1ccccc1",                 "phenol",       "Oc1ccccc1"),
    ("propanol",    "CCCO",                      "butanol",      "CCCCO"),
    ("glycine",     "NCC(=O)O",                  "alanine",      "CC(N)C(=O)O"),
    ("phenylalanine","NC(Cc1ccccc1)C(=O)O",      "tyrosine",     "NC(Cc1ccc(O)cc1)C(=O)O"),
    ("pyridine",    "c1ccncc1",                  "quinoline",    "c1ccc2ncccc2c1"),
]
