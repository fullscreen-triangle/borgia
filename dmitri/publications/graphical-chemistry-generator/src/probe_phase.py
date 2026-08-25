"""Which cross-term? cos(phi_A - phi_B) or cos(phi_A + phi_B)?

The specification of the interference observable is ambiguous between a
SUM and a DIFFERENCE of phases.  This is decidable, not a matter of
taste: the observable is required to be maximal for a molecule against
itself and to decay with S-entropy distance.  We measure both variants
against that requirement.
"""
from __future__ import annotations
import math, os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "categorical-compound-database"))
from categorical_compound_database import COMPOUNDS, CategoricalEncoder

W = 256
enc = CategoricalEncoder(COMPOUNDS)
E = {n: enc.encode(n) for n in COMPOUNDS}
OMEGA_REF = 4401.0

def observe(rec, w=W):
    """Complex observation field: amplitude and phase at each address u."""
    u = np.linspace(0.0, 1.0, w)
    Sk, St, Se = rec["S_k"], rec["S_t"], rec["S_e"]
    sigma = 0.05 * (1.0 - 0.5 * Sk) + 1e-6
    amp = np.zeros(w)
    for om in rec["modes"]:
        mu = om / OMEGA_REF
        amp += np.exp(-((u - mu) ** 2) / (sigma ** 2))
    env = np.exp(-((u - 0.5) ** 2) / ((0.1 + 0.4 * St) ** 2))
    amp = amp * env
    phase = 2.0 * math.pi * (Se * u * 8.0)          # depth fringes
    for om in rec["modes"]:
        phase = phase + (om / OMEGA_REF) * u * 2.0 * math.pi
    return amp, phase

def visibility(a, b, mode):
    aA, pA = a
    aB, pB = b
    na = aA / (np.max(aA) + 1e-12)
    nb = aB / (np.max(aB) + 1e-12)
    d = (pA - pB) if mode == "diff" else (pA + pB)
    return float(np.mean(0.5 + 0.5 * na * nb * np.cos(d)))

names = sorted(E)
obs = {n: observe(E[n]) for n in names}

out = {}
for mode in ("diff", "sum"):
    self_v = [visibility(obs[n], obs[n], mode) for n in names]
    cross, dist = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross.append(visibility(obs[a], obs[b], mode))
            dist.append(math.dist(
                (E[a]["S_k"], E[a]["S_t"], E[a]["S_e"]),
                (E[b]["S_k"], E[b]["S_t"], E[b]["S_e"])))
    r = float(np.corrcoef(dist, cross)[0, 1])
    out[mode] = {
        "self_mean": round(float(np.mean(self_v)), 6),
        "self_min": round(float(np.min(self_v)), 6),
        "self_max": round(float(np.max(self_v)), 6),
        "self_is_constant_1": bool(np.allclose(self_v, 1.0, atol=1e-9)),
        "cross_mean": round(float(np.mean(cross)), 6),
        "cross_max": round(float(np.max(cross)), 6),
        "n_cross_exceeding_min_self": int(sum(
            1 for c in cross if c > np.min(self_v) + 1e-12)),
        "corr_visibility_vs_distance": round(r, 6),
    }

print(json.dumps(out, indent=2))
