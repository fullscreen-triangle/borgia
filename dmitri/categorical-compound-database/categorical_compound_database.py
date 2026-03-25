#!/usr/bin/env python3
"""
Categorical Compound Database
==============================

Proof-of-concept: encode molecular compounds as ternary addresses in
S-entropy space [0,1]^3, build a trie, and search by prefix matching.

Instead of computing molecular fingerprints and doing O(N) pairwise
comparison, the bounded phase space / ternary S-entropy framework
reduces search to O(k) trie traversal, where k = ternary depth.

S-entropy coordinates:
    S_k  (knowledge entropy)  — spectral identity (Shannon entropy of mode distribution)
    S_t  (temporal entropy)   — temporal spread (how many timescales the molecule spans)
    S_e  (evolution entropy)  — structural complexity (harmonic network connectivity)

Ternary encoding interleaves the three dimensions so that prefix
matching automatically provides multi-resolution fuzzy search.

Author : Kundai Farai Sachikonye
Date   : 2026-03-25
Requires: numpy
"""

import io
import json
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

# Ensure stdout can handle Unicode on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# ---------------------------------------------------------------------------
# Molecular Database — NIST CCCBDB vibrational frequencies (cm^-1)
# ---------------------------------------------------------------------------

COMPOUNDS = {
    # Diatomics
    "H2":    {"formula": "H\u2082",   "modes": [4401], "B_rot": 59.3,
              "type": "diatomic", "mass": 2},
    "D2":    {"formula": "D\u2082",   "modes": [2994], "B_rot": 29.9,
              "type": "diatomic", "mass": 4},
    "N2":    {"formula": "N\u2082",   "modes": [2330], "B_rot": 1.99,
              "type": "diatomic", "mass": 28},
    "O2":    {"formula": "O\u2082",   "modes": [1580], "B_rot": 1.44,
              "type": "diatomic", "mass": 32},
    "F2":    {"formula": "F\u2082",   "modes": [892],  "B_rot": 0.89,
              "type": "diatomic", "mass": 38},
    "Cl2":   {"formula": "Cl\u2082", "modes": [560],  "B_rot": 0.24,
              "type": "diatomic", "mass": 71},
    "CO":    {"formula": "CO",   "modes": [2143], "B_rot": 1.93,
              "type": "diatomic", "mass": 28},
    "NO":    {"formula": "NO",   "modes": [1876], "B_rot": 1.67,
              "type": "diatomic", "mass": 30},
    "HF":    {"formula": "HF",   "modes": [3958], "B_rot": 20.6,
              "type": "diatomic", "mass": 20},
    "HCl":   {"formula": "HCl",  "modes": [2886], "B_rot": 10.4,
              "type": "diatomic", "mass": 36},
    "HBr":   {"formula": "HBr",  "modes": [2559], "B_rot": 8.47,
              "type": "diatomic", "mass": 81},
    "HI":    {"formula": "HI",   "modes": [2230], "B_rot": 6.43,
              "type": "diatomic", "mass": 128},

    # Triatomics
    "H2O":   {"formula": "H\u2082O",  "modes": [1595, 3657, 3756],
              "type": "triatomic", "mass": 18},
    "CO2":   {"formula": "CO\u2082",  "modes": [667, 1388, 2349],
              "type": "triatomic", "mass": 44},
    "SO2":   {"formula": "SO\u2082",  "modes": [518, 1151, 1362],
              "type": "triatomic", "mass": 64},
    "NO2":   {"formula": "NO\u2082",  "modes": [750, 1318, 1617],
              "type": "triatomic", "mass": 46},
    "O3":    {"formula": "O\u2083",   "modes": [701, 1042, 1103],
              "type": "triatomic", "mass": 48},
    "H2S":   {"formula": "H\u2082S",  "modes": [1183, 2615, 2626],
              "type": "triatomic", "mass": 34},
    "HCN":   {"formula": "HCN",  "modes": [712, 2097, 3312],
              "type": "triatomic", "mass": 27},
    "N2O":   {"formula": "N\u2082O",  "modes": [589, 1285, 2224],
              "type": "triatomic", "mass": 44},
    "CS2":   {"formula": "CS\u2082",  "modes": [397, 657, 1535],
              "type": "triatomic", "mass": 76},
    "OCS":   {"formula": "OCS",  "modes": [520, 859, 2062],
              "type": "triatomic", "mass": 60},

    # 4-atom
    "NH3":   {"formula": "NH\u2083",  "modes": [950, 1627, 3337, 3444],
              "type": "tetra", "mass": 17},
    "H2CO":  {"formula": "H\u2082CO", "modes": [1167, 1249, 1500, 1746, 2783, 2843],
              "type": "poly", "mass": 30},
    "PH3":   {"formula": "PH\u2083",  "modes": [992, 1118, 2323, 2328],
              "type": "tetra", "mass": 34},

    # 5-atom (tetrahedral)
    "CH4":   {"formula": "CH\u2084",  "modes": [1306, 1534, 2917, 3019],
              "type": "tetra", "mass": 16},
    "CCl4":  {"formula": "CCl\u2084", "modes": [218, 314, 762, 790],
              "type": "tetra", "mass": 154},
    "SiH4":  {"formula": "SiH\u2084", "modes": [800, 914, 2187, 2191],
              "type": "tetra", "mass": 32},
    "CF4":   {"formula": "CF\u2084",  "modes": [435, 632, 909, 1283],
              "type": "tetra", "mass": 88},

    # 6+ atom (polyatomic)
    "C2H2":  {"formula": "C\u2082H\u2082",
              "modes": [612, 729, 1974, 3289, 3374],
              "type": "poly", "mass": 26},
    "C2H4":  {"formula": "C\u2082H\u2084",
              "modes": [826, 943, 949, 1023, 1236, 1342, 1444, 1623,
                        3026, 3083, 3103, 3106],
              "type": "poly", "mass": 28},
    "C2H6":  {"formula": "C\u2082H\u2086",
              "modes": [822, 995, 1190, 1370, 1388, 1468, 2896, 2954,
                        2969, 2985],
              "type": "poly", "mass": 30},
    "CH3OH": {"formula": "CH\u2083OH",
              "modes": [1033, 1060, 1165, 1345, 1455, 1477, 2844, 2960,
                        3000, 3681],
              "type": "poly", "mass": 32},
    "C6H6":  {"formula": "C\u2086H\u2086",
              "modes": [673, 993, 1178, 1596, 3062],
              "type": "poly", "mass": 78},
    "CH3F":  {"formula": "CH\u2083F",
              "modes": [1049, 1182, 1459, 1467, 2930, 3006],
              "type": "poly", "mass": 34},
    "CH3Cl": {"formula": "CH\u2083Cl",
              "modes": [732, 1017, 1355, 1452, 2937, 3039],
              "type": "poly", "mass": 50},
    "CH3Br": {"formula": "CH\u2083Br",
              "modes": [611, 952, 1306, 1443, 2935, 3056],
              "type": "poly", "mass": 95},
    "HCOOH": {"formula": "HCOOH",
              "modes": [625, 1033, 1105, 1229, 1387, 1770, 2943, 3570],
              "type": "poly", "mass": 46},
    "CH3CN": {"formula": "CH\u2083CN",
              "modes": [362, 920, 1041, 1385, 1448, 2267, 2954, 3009],
              "type": "poly", "mass": 41},
}

# ---------------------------------------------------------------------------
# Reference constants for normalization
# ---------------------------------------------------------------------------

OMEGA_REF_MAX = 4401       # H2 stretch — highest vibrational freq in database
OMEGA_REF_MIN = 218        # CCl4 lowest mode
B_ROT_REF_MIN = 0.39       # large-moment-of-inertia reference

# Rational-ratio harmonicity threshold
HARMONIC_DELTA = 0.05

# Maximum ternary depth (6 trits per dimension = 18 interleaved)
MAX_DEPTH = 18


# ===================================================================
# S-Entropy Encoding
# ===================================================================

class CategoricalEncoder:
    """Compute S-entropy coordinates and ternary addresses for molecules."""

    def __init__(self, compounds_db):
        self.db = compounds_db
        self.omega_ref_max = OMEGA_REF_MAX
        self.omega_ref_min = OMEGA_REF_MIN
        self.b_rot_ref_min = B_ROT_REF_MIN
        self.harmonic_delta = HARMONIC_DELTA

    # --- S_k: knowledge entropy (spectral identity) ---

    def compute_S_k(self, modes, B_rot=None):
        """
        Knowledge entropy: Shannon entropy of normalised frequency distribution.

        For polyatomics (N >= 2):
            p_i = omega_i / sum(omega)
            H   = -sum(p_i * log2(p_i))
            S_k = H / log2(N)                     [0, 1]

        For diatomics (N = 1):
            S_k = omega / omega_ref_max            [0, 1]
        """
        N = len(modes)
        if N == 1:
            return modes[0] / self.omega_ref_max

        total = sum(modes)
        probs = [w / total for w in modes]
        H = -sum(p * math.log2(p) for p in probs if p > 0)
        H_max = math.log2(N)
        return min(H / H_max, 1.0)

    # --- S_t: temporal entropy (timescale spread) ---

    def compute_S_t(self, modes, B_rot=None):
        """
        Temporal entropy: how many timescales the molecule spans.

        For polyatomics:
            S_t = log(omega_max / omega_min) / log(omega_ref_max / omega_ref_min)

        For diatomics with known B_rot:
            S_t = log(omega_vib / B_rot) / log(omega_ref_max / B_rot_ref_min)

        For diatomics without B_rot:
            S_t = 0.1 (minimal spread)
        """
        N = len(modes)
        if N == 1:
            if B_rot is not None and B_rot > 0:
                ratio = modes[0] / B_rot
                ref_ratio = self.omega_ref_max / self.b_rot_ref_min
                val = math.log(ratio) / math.log(ref_ratio) if ratio > 1 else 0.0
                return min(max(val, 0.0), 1.0)
            return 0.1

        omega_max = max(modes)
        omega_min = min(modes)
        if omega_min <= 0 or omega_max <= omega_min:
            return 0.0

        ref_ratio = self.omega_ref_max / self.omega_ref_min
        val = math.log(omega_max / omega_min) / math.log(ref_ratio)
        return min(max(val, 0.0), 1.0)

    # --- S_e: evolution entropy (harmonic connectivity) ---

    @staticmethod
    def _is_harmonic_pair(omega_a, omega_b, delta=0.05):
        """Check if omega_a / omega_b is close to a rational p/q with p,q <= 8."""
        if omega_a == 0 or omega_b == 0:
            return False
        ratio = max(omega_a, omega_b) / min(omega_a, omega_b)
        # Check rationals p/q with 1 <= q <= p <= 8
        for p in range(1, 9):
            for q in range(1, p + 1):
                if abs(ratio - p / q) < delta:
                    return True
        return False

    def compute_S_e(self, modes):
        """
        Evolution entropy: harmonic edge density.

        N_pairs    = N*(N-1)/2
        N_harmonic = count of pairs whose frequency ratio is within delta
                     of a simple rational p/q  (p, q <= 8)
        S_e        = N_harmonic / max(N_pairs, 1)
        """
        N = len(modes)
        if N < 2:
            return 0.0
        n_pairs = N * (N - 1) // 2
        n_harmonic = sum(
            1 for a, b in combinations(modes, 2)
            if self._is_harmonic_pair(a, b, self.harmonic_delta)
        )
        return n_harmonic / max(n_pairs, 1)

    # --- Full encoding ---

    def encode(self, compound_name):
        """
        Return dict with S_k, S_t, S_e, ternary string, and metadata.
        """
        data = self.db[compound_name]
        modes = data["modes"]
        B_rot = data.get("B_rot")

        S_k = self.compute_S_k(modes, B_rot)
        S_t = self.compute_S_t(modes, B_rot)
        S_e = self.compute_S_e(modes)

        trits = coords_to_trits(S_k, S_t, S_e, depth=MAX_DEPTH)

        return {
            "name": compound_name,
            "formula": data["formula"],
            "type": data["type"],
            "mass": data["mass"],
            "modes": modes,
            "S_k": round(S_k, 6),
            "S_t": round(S_t, 6),
            "S_e": round(S_e, 6),
            "trits": trits,
            "trit_string": "".join(str(t) for t in trits),
        }

    def encode_all(self):
        """Encode every compound in the database."""
        return {name: self.encode(name) for name in self.db}


# ===================================================================
# Ternary Encoding / Decoding
# ===================================================================

def coords_to_trits(S_k, S_t, S_e, depth=18):
    """
    Generate interleaved ternary string from (S_k, S_t, S_e).

    Position j mod 3 determines which dimension is refined:
        0 -> S_k,  1 -> S_t,  2 -> S_e

    Each trit = one observation = one oscillation-counting operation.
    """
    trits = []
    remainders = [float(S_k), float(S_t), float(S_e)]
    for j in range(depth):
        dim = j % 3
        scaled = 3.0 * remainders[dim]
        trit = int(scaled)
        trit = min(trit, 2)        # clamp to {0, 1, 2}
        trit = max(trit, 0)
        remainders[dim] = scaled - trit
        trits.append(trit)
    return trits


def trits_to_coords(trits):
    """
    Recover approximate (S_k, S_t, S_e) from ternary string.
    Uses midpoint reconstruction: each trit contributes (t + 0.5) / 3^d.
    """
    S = [0.0, 0.0, 0.0]
    depth_count = [0, 0, 0]
    for j, t in enumerate(trits):
        dim = j % 3
        depth_count[dim] += 1
        S[dim] += (t + 0.5) / (3 ** depth_count[dim])
    return S


def trit_string_to_list(s):
    """Convert '012021...' string to list of ints."""
    return [int(c) for c in s]


# ===================================================================
# Ternary Trie
# ===================================================================

class TrieNode:
    """Node in a base-3 trie."""

    __slots__ = ("children", "compounds")

    def __init__(self):
        self.children = [None, None, None]   # trit 0, 1, 2
        self.compounds = []                  # compound names stored at this node


class TernaryTrie:
    """
    Ternary trie over interleaved S-entropy addresses.

    Insert:  O(k)   — k = ternary depth
    Exact:   O(k)
    Prefix:  O(k + m)  where m = subtree size
    """

    def __init__(self):
        self.root = TrieNode()
        self._size = 0

    # --- Insert ---

    def insert(self, trits, compound_name):
        """Insert compound at the address given by trit list."""
        node = self.root
        for t in trits:
            if node.children[t] is None:
                node.children[t] = TrieNode()
            node = node.children[t]
        node.compounds.append(compound_name)
        self._size += 1

    # --- Exact search ---

    def search_exact(self, trits):
        """Return compounds stored at the exact trit address, or []."""
        node = self.root
        for t in trits:
            if node.children[t] is None:
                return []
            node = node.children[t]
        return list(node.compounds)

    # --- Prefix search (fuzzy) ---

    def search_prefix(self, prefix):
        """
        Return ALL compounds whose trit string begins with `prefix`.

        Shorter prefix = coarser resolution = more matches = fuzzier.
        This is the core fuzzy-search mechanism: resolution is adjustable
        by simply truncating the query.
        """
        node = self.root
        for t in prefix:
            if node.children[t] is None:
                return []
            node = node.children[t]
        # Collect everything in subtree
        return self._collect(node)

    def _collect(self, node):
        """DFS collection of all compounds in subtree."""
        results = list(node.compounds)
        for child in node.children:
            if child is not None:
                results.extend(self._collect(child))
        return results

    # --- All groupings at a given depth ---

    def all_at_depth(self, depth):
        """
        Return dict mapping trit-prefix (as string) -> list of compounds.

        This partitions the entire database into cells at the given resolution.
        """
        groups = defaultdict(list)
        self._walk(self.root, [], depth, groups)
        return dict(groups)

    def _walk(self, node, prefix, target_depth, groups):
        """Walk trie to target depth, collecting compounds."""
        if len(prefix) == target_depth:
            compounds = self._collect(node)
            if compounds:
                groups["".join(str(t) for t in prefix)] = compounds
            return
        # Collect compounds at intermediate nodes (shorter trit strings)
        if node.compounds:
            groups["".join(str(t) for t in prefix)] = list(node.compounds)
        for trit in range(3):
            if node.children[trit] is not None:
                self._walk(node.children[trit], prefix + [trit],
                           target_depth, groups)

    @property
    def size(self):
        return self._size


# ===================================================================
# Categorical Compound Database
# ===================================================================

class CategoricalCompoundDatabase:
    """
    Full categorical compound database with encoding, trie storage,
    fuzzy search, similarity computation, and structure prediction.
    """

    def __init__(self, compounds=None):
        self.compounds = compounds or COMPOUNDS
        self.encoder = CategoricalEncoder(self.compounds)
        self.trie = TernaryTrie()
        self.encodings = {}
        self._build()

    def _build(self):
        """Encode all compounds and insert into trie."""
        self.encodings = self.encoder.encode_all()
        for name, enc in self.encodings.items():
            self.trie.insert(enc["trits"], name)

    # --- Fuzzy search ---

    def fuzzy_search(self, query_coords, resolution=6):
        """
        Find all compounds in the same ternary cell at given resolution.

        query_coords: (S_k, S_t, S_e) tuple
        resolution:   number of trits to match (higher = finer)

        Returns list of compound names.
        """
        trits = coords_to_trits(*query_coords, depth=resolution)
        return self.trie.search_prefix(trits)

    def fuzzy_search_by_name(self, compound_name, resolution=6):
        """Fuzzy search using an existing compound as query."""
        enc = self.encodings[compound_name]
        prefix = enc["trits"][:resolution]
        return self.trie.search_prefix(prefix)

    # --- Ternary similarity ---

    def similarity(self, name_a, name_b):
        """
        Ternary distance: depth at which trit strings first diverge.

        Higher = more similar (they share a longer common prefix).
        Maximum = MAX_DEPTH (identical encoding).
        """
        trits_a = self.encodings[name_a]["trits"]
        trits_b = self.encodings[name_b]["trits"]
        for i, (a, b) in enumerate(zip(trits_a, trits_b)):
            if a != b:
                return i
        return min(len(trits_a), len(trits_b))

    def similarity_matrix(self):
        """Compute pairwise ternary similarity for all compounds."""
        names = sorted(self.encodings.keys())
        n = len(names)
        matrix = {}
        for i in range(n):
            row = {}
            for j in range(n):
                row[names[j]] = self.similarity(names[i], names[j])
            matrix[names[i]] = row
        return matrix

    # --- Property-based search ---

    def find_by_properties(self, S_k_range=None, S_t_range=None,
                           S_e_range=None):
        """
        Structure prediction: find compounds matching property constraints.

        Each range is (min, max) or None (no constraint on that axis).
        """
        results = []
        for name, enc in self.encodings.items():
            if S_k_range and not (S_k_range[0] <= enc["S_k"] <= S_k_range[1]):
                continue
            if S_t_range and not (S_t_range[0] <= enc["S_t"] <= S_t_range[1]):
                continue
            if S_e_range and not (S_e_range[0] <= enc["S_e"] <= S_e_range[1]):
                continue
            results.append(name)
        return results

    # --- Clustering at resolution ---

    def cluster_at_resolution(self, depth):
        """
        Group all compounds into cells at given trit depth.

        Returns dict: trit_prefix_string -> [compound_names]
        """
        return self.trie.all_at_depth(depth)

    # --- Nearest neighbours ---

    def nearest_neighbors(self, compound_name, k=5):
        """
        Find k nearest neighbours by ternary similarity (longest shared prefix).
        """
        sims = []
        for name in self.encodings:
            if name == compound_name:
                continue
            sims.append((name, self.similarity(compound_name, name)))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]


# ===================================================================
# Validation: Chemical Similarity Emerges
# ===================================================================

EXPECTED_GROUPS = {
    "Halomethanes":            ["CH3F", "CH3Cl", "CH3Br"],
    "Hydrogen halides":        ["HF", "HCl", "HBr", "HI"],
    "Homonuclear diatomics":   ["H2", "D2", "N2", "O2", "F2", "Cl2"],
    "Small hydrocarbons":      ["CH4", "C2H2", "C2H4", "C2H6"],
    "Linear triatomics":       ["CO2", "CS2", "N2O", "OCS"],
    "Bent triatomics":         ["H2O", "SO2", "NO2", "O3", "H2S"],
}


def validate_clustering(db):
    """
    For each expected chemical group, measure how well ternary
    proximity reproduces the grouping.

    Metric: mean intra-group similarity vs mean inter-group similarity.
    A ratio > 1 indicates that members of the group are closer to each
    other in ternary space than to the rest of the database.
    """
    all_names = sorted(db.encodings.keys())
    results = {}

    for group_name, members in EXPECTED_GROUPS.items():
        # Intra-group similarity
        intra_sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra_sims.append(db.similarity(members[i], members[j]))
        mean_intra = np.mean(intra_sims) if intra_sims else 0.0

        # Inter-group similarity (each member vs non-members)
        non_members = [n for n in all_names if n not in members]
        inter_sims = []
        for m in members:
            for nm in non_members:
                inter_sims.append(db.similarity(m, nm))
        mean_inter = np.mean(inter_sims) if inter_sims else 0.0

        ratio = mean_intra / mean_inter if mean_inter > 0 else float("inf")
        cohesion = "PASS" if ratio > 1.0 else "MARGINAL" if ratio > 0.8 else "FAIL"

        results[group_name] = {
            "members": members,
            "mean_intra_similarity": round(float(mean_intra), 3),
            "mean_inter_similarity": round(float(mean_inter), 3),
            "intra_inter_ratio": round(float(ratio), 3),
            "cohesion": cohesion,
        }

    return results


# ===================================================================
# Oscillation Counting vs Computation
# ===================================================================

def oscillation_analysis():
    """
    Demonstrate that ternary encoding = counting oscillations.

    Each trit = one refinement = one observation of which third of the
    range the value falls in.  k trits = k oscillation-counting
    operations.  Resolution per dimension = 3^(-k/3).

    Compare:
        Traditional fingerprint search: O(N * d) per query
            N = database size, d = fingerprint dimensionality
        Ternary trie lookup: O(k) per query
            k = ternary depth (number of oscillation counts)
    """
    analysis = {
        "concept": (
            "Each trit is one refinement step: observe which third of the "
            "remaining interval the value falls in. This is equivalent to "
            "counting oscillations of a ternary oscillator and recording "
            "the phase. k trits = k observations = k oscillation-counting "
            "operations."
        ),
        "resolution_per_depth": {},
        "complexity_comparison": {},
    }

    N = len(COMPOUNDS)
    fingerprint_dim = 1024   # typical Morgan fingerprint

    for k in [3, 6, 9, 12, 15, 18]:
        trits_per_dim = k // 3
        resolution = 3 ** (-trits_per_dim)
        n_cells = 3 ** k
        analysis["resolution_per_depth"][k] = {
            "trits_per_dimension": trits_per_dim,
            "resolution_per_axis": round(resolution, 8),
            "total_cells": n_cells,
            "observations_needed": k,
        }

    for label, ops in [
        ("Brute-force fingerprint (Tanimoto)", N * fingerprint_dim),
        ("Trie lookup (depth 6)", 6),
        ("Trie lookup (depth 9)", 9),
        ("Trie lookup (depth 12)", 12),
        ("Trie lookup (depth 18)", 18),
    ]:
        analysis["complexity_comparison"][label] = {
            "operations": ops,
            "description": (
                f"{ops} operations per query"
                + (f" (N={N}, d={fingerprint_dim})" if "fingerprint" in label else "")
            ),
        }

    analysis["speedup_at_depth_12"] = {
        "brute_force_ops": N * fingerprint_dim,
        "trie_ops": 12,
        "ratio": round((N * fingerprint_dim) / 12, 1),
        "interpretation": (
            f"Trie search at depth 12 is ~{round((N * fingerprint_dim) / 12)}x "
            f"fewer operations than brute-force fingerprint comparison over "
            f"{N} compounds."
        ),
    }

    return analysis


# ===================================================================
# JSON Output
# ===================================================================

def save_results(db, results_dir):
    """Save all results as JSON to the results directory."""
    os.makedirs(results_dir, exist_ok=True)

    # 1. Compound encodings
    encodings_out = {}
    for name, enc in sorted(db.encodings.items()):
        encodings_out[name] = {
            "formula": enc["formula"],
            "type": enc["type"],
            "mass": enc["mass"],
            "modes": enc["modes"],
            "S_k": enc["S_k"],
            "S_t": enc["S_t"],
            "S_e": enc["S_e"],
            "trit_string": enc["trit_string"],
        }
    with open(os.path.join(results_dir, "compound_encodings.json"), "w",
              encoding="utf-8") as f:
        json.dump(encodings_out, f, indent=2, ensure_ascii=False)

    # 2. Similarity matrix
    sim_matrix = db.similarity_matrix()
    with open(os.path.join(results_dir, "similarity_matrix.json"), "w",
              encoding="utf-8") as f:
        json.dump(sim_matrix, f, indent=2)

    # 3. Fuzzy search results at different resolutions
    test_queries = ["H2O", "CH4", "CO2", "C2H6", "HCl"]
    fuzzy_results = {}
    for query in test_queries:
        fuzzy_results[query] = {}
        for depth in [3, 6, 9, 12]:
            matches = db.fuzzy_search_by_name(query, resolution=depth)
            prefix = "".join(str(t) for t in db.encodings[query]["trits"][:depth])
            fuzzy_results[query][f"depth_{depth}"] = {
                "prefix": prefix,
                "n_matches": len(matches),
                "matches": matches,
            }
    with open(os.path.join(results_dir, "fuzzy_search_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(fuzzy_results, f, indent=2)

    # 4. Clustering at different resolutions
    clustering = {}
    for depth in [3, 6, 9, 12]:
        groups = db.cluster_at_resolution(depth)
        # Only include cells with > 1 compound (interesting clusters)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        clustering[f"depth_{depth}"] = {
            "total_cells": len(groups),
            "multi_occupancy_cells": len(multi),
            "clusters": multi,
        }
    with open(os.path.join(results_dir, "clustering.json"), "w",
              encoding="utf-8") as f:
        json.dump(clustering, f, indent=2)

    # 5. Property search results
    property_searches = {
        "high_S_k_low_S_t": {
            "description": "High spectral complexity, low temporal spread",
            "S_k_range": [0.9, 1.0],
            "S_t_range": [0.0, 0.3],
            "results": db.find_by_properties(
                S_k_range=(0.9, 1.0), S_t_range=(0.0, 0.3)),
        },
        "low_S_k_high_S_t": {
            "description": "Simple spectrum, wide temporal spread",
            "S_k_range": [0.0, 0.4],
            "S_t_range": [0.5, 1.0],
            "results": db.find_by_properties(
                S_k_range=(0.0, 0.4), S_t_range=(0.5, 1.0)),
        },
        "high_harmonicity": {
            "description": "High harmonic connectivity (S_e > 0.5)",
            "S_e_range": [0.5, 1.0],
            "results": db.find_by_properties(S_e_range=(0.5, 1.0)),
        },
        "low_harmonicity": {
            "description": "Low harmonic connectivity (S_e < 0.15)",
            "S_e_range": [0.0, 0.15],
            "results": db.find_by_properties(S_e_range=(0.0, 0.15)),
        },
        "heavy_complex": {
            "description": "Moderate spectral identity, moderate temporal spread",
            "S_k_range": [0.85, 0.98],
            "S_t_range": [0.3, 0.6],
            "results": db.find_by_properties(
                S_k_range=(0.85, 0.98), S_t_range=(0.3, 0.6)),
        },
    }
    with open(os.path.join(results_dir, "property_search_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(property_searches, f, indent=2)

    # 6. Validation summary
    validation = validate_clustering(db)
    oscillation = oscillation_analysis()
    summary = {
        "chemical_group_validation": validation,
        "oscillation_counting_analysis": oscillation,
    }
    with open(os.path.join(results_dir, "validation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ===================================================================
# Console Output
# ===================================================================

def print_report(db):
    """Print a comprehensive summary to stdout."""

    sep = "=" * 80
    thin = "-" * 80

    # --- Header ---
    print()
    print(sep)
    print("  CATEGORICAL COMPOUND DATABASE")
    print("  Ternary S-Entropy Encoding of Molecular Phase Space")
    print(sep)
    print()

    # --- 1. Compound encodings ---
    print("1. COMPOUND ENCODINGS")
    print(thin)
    header = f"{'Name':<8} {'Formula':<9} {'Type':<11} "
    header += f"{'S_k':>7} {'S_t':>7} {'S_e':>7}   {'Trits (first 12)'}"
    print(header)
    print(thin)
    for name in sorted(db.encodings.keys()):
        enc = db.encodings[name]
        trit_str = enc["trit_string"][:12]
        print(f"{name:<8} {enc['formula']:<9} {enc['type']:<11} "
              f"{enc['S_k']:>7.4f} {enc['S_t']:>7.4f} {enc['S_e']:>7.4f}   "
              f"{trit_str}")
    print(f"\nTotal compounds: {len(db.encodings)}")
    print()

    # --- 2. Chemical similarity groups ---
    print("2. CHEMICAL SIMILARITY GROUPS FROM TERNARY CLUSTERING")
    print(thin)

    validation = validate_clustering(db)
    for group_name, info in validation.items():
        status = info["cohesion"]
        marker = "[+]" if status == "PASS" else "[~]" if status == "MARGINAL" else "[-]"
        print(f"\n  {marker} {group_name}: {', '.join(info['members'])}")
        print(f"      Mean intra-group similarity: {info['mean_intra_similarity']:.3f}")
        print(f"      Mean inter-group similarity: {info['mean_inter_similarity']:.3f}")
        print(f"      Ratio (intra/inter):         {info['intra_inter_ratio']:.3f}  "
              f"({status})")

    # Nearest-neighbour sanity check
    print(f"\n{thin}")
    print("  Nearest-neighbour sanity check:")
    for query in ["H2O", "CH4", "HCl", "CO2", "CH3Cl", "C2H6"]:
        nns = db.nearest_neighbors(query, k=3)
        nn_str = ", ".join(f"{n} ({s})" for n, s in nns)
        print(f"    {query:>6} -> {nn_str}")
    print()

    # --- 3. Fuzzy search examples ---
    print("3. FUZZY SEARCH AT DIFFERENT RESOLUTIONS")
    print(thin)
    test_compounds = ["H2O", "CH4", "CO2", "C2H6", "HCl"]
    for compound in test_compounds:
        print(f"\n  Query: {compound}  "
              f"(S_k={db.encodings[compound]['S_k']:.4f}, "
              f"S_t={db.encodings[compound]['S_t']:.4f}, "
              f"S_e={db.encodings[compound]['S_e']:.4f})")
        for depth in [3, 6, 9, 12]:
            matches = db.fuzzy_search_by_name(compound, resolution=depth)
            prefix = "".join(str(t) for t in db.encodings[compound]["trits"][:depth])
            n = len(matches)
            if n <= 8:
                match_str = ", ".join(matches)
            else:
                match_str = ", ".join(matches[:6]) + f" ... (+{n - 6} more)"
            print(f"    depth {depth:2d} (prefix {prefix}): "
                  f"{n:2d} match(es) -> {match_str}")
    print()

    # --- 4. Structure prediction examples ---
    print("4. STRUCTURE PREDICTION (Property-Based Search)")
    print(thin)

    searches = [
        ("High spectral complexity + low temporal spread",
         {"S_k_range": (0.9, 1.0), "S_t_range": (0.0, 0.3)}),
        ("Simple spectrum + wide temporal spread",
         {"S_k_range": (0.0, 0.4), "S_t_range": (0.5, 1.0)}),
        ("High harmonic connectivity (S_e > 0.5)",
         {"S_e_range": (0.5, 1.0)}),
        ("Low harmonic connectivity (S_e < 0.15)",
         {"S_e_range": (0.0, 0.15)}),
        ("Moderate S_k [0.85,0.98] + moderate S_t [0.3,0.6]",
         {"S_k_range": (0.85, 0.98), "S_t_range": (0.3, 0.6)}),
    ]
    for desc, params in searches:
        results = db.find_by_properties(**params)
        print(f"\n  {desc}")
        constraints = []
        for key, (lo, hi) in params.items():
            axis = key.replace("_range", "")
            constraints.append(f"{axis} in [{lo}, {hi}]")
        print(f"    Constraints: {', '.join(constraints)}")
        print(f"    Results ({len(results)}): {', '.join(results) if results else '(none)'}")
    print()

    # --- 5. Complexity comparison ---
    print("5. OSCILLATION COUNTING VS ALGORITHMIC COMPUTATION")
    print(thin)

    N = len(COMPOUNDS)
    fp_dim = 1024
    brute = N * fp_dim

    print(f"\n  Database size:       N = {N} compounds")
    print(f"  Fingerprint length:  d = {fp_dim} bits (typical Morgan)")
    print(f"  Brute-force search:  O(N * d) = {brute:,} operations per query")
    print()
    print(f"  {'Ternary trie search':<30} {'Depth':>6}  {'Ops':>8}  "
          f"{'Speedup':>10}  {'Resolution'}")
    print(f"  {'-' * 72}")
    for k in [3, 6, 9, 12, 15, 18]:
        trits_per_dim = k // 3
        res = 3 ** (-trits_per_dim)
        speedup = brute / k
        print(f"  {'':30} {k:>6}  {k:>8}  {speedup:>10.0f}x  "
              f"3^(-{trits_per_dim}) = {res:.6f}")

    print()
    print("  Key insight: each trit = one oscillation-counting observation.")
    print("  Search is not computation; it is phase-space addressing.")
    print("  The trie encodes the bounded phase space [0,1]^3 directly.")
    print()

    # --- Clustering summary ---
    print("6. CLUSTERING SUMMARY AT DIFFERENT RESOLUTIONS")
    print(thin)
    for depth in [3, 6, 9, 12]:
        groups = db.cluster_at_resolution(depth)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        print(f"\n  Depth {depth}: {len(groups)} occupied cells, "
              f"{len(multi)} with multiple compounds")
        if multi:
            for prefix, members in sorted(multi.items(),
                                           key=lambda x: -len(x[1])):
                print(f"    [{prefix}] ({len(members)}): {', '.join(members)}")
    print()

    print(sep)
    print("  All results saved to results/ directory as JSON.")
    print(sep)
    print()


# ===================================================================
# Main
# ===================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"

    print("Building Categorical Compound Database ...")
    t0 = time.time()

    db = CategoricalCompoundDatabase()

    t_build = time.time() - t0
    print(f"Database built in {t_build:.4f} s  ({len(db.encodings)} compounds)")

    # Timing: trie search vs brute-force
    t1 = time.time()
    for _ in range(1000):
        db.fuzzy_search_by_name("H2O", resolution=12)
    t_trie = (time.time() - t1) / 1000

    t2 = time.time()
    for _ in range(1000):
        # Brute-force: compare H2O encoding to all others
        ref = db.encodings["H2O"]
        for name, enc in db.encodings.items():
            _ = sum((a - b) ** 2 for a, b in
                    zip([ref["S_k"], ref["S_t"], ref["S_e"]],
                        [enc["S_k"], enc["S_t"], enc["S_e"]]))
    t_brute = (time.time() - t2) / 1000

    print(f"Trie search (depth 12):   {t_trie * 1e6:.1f} us/query")
    print(f"Brute-force (Euclidean):  {t_brute * 1e6:.1f} us/query")
    if t_trie > 0:
        print(f"Empirical speedup:        {t_brute / t_trie:.1f}x")

    # Save results
    save_results(db, str(results_dir))

    # Print report
    print_report(db)

    return 0


if __name__ == "__main__":
    sys.exit(main())
