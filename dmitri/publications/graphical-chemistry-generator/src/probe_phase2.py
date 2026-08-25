"""Correct normalisation: V must be the FRINGE VISIBILITY of the
superposed field, not the mean of a pointwise expression.

Physical definition (Michelson):  V = (I_max - I_min) / (I_max + I_min)
with I = |A_1 + A_2|^2.  Equivalently, for fields with amplitudes a_1,a_2
and phase difference d, the normalised correlation

    V = |<a1 a2 e^{i d}>| / sqrt(<a1^2> <a2^2>)

is 1 exactly when the two fields are identical, by Cauchy-Schwarz.
That is the quantity we test.
"""
from __future__ import annotations
import math, os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "categorical-compound-database"))
from categorical_compound_database import COMPOUNDS, CategoricalEncoder
from probe_phase import observe, E, OMEGA_REF

names = sorted(E)
obs = {n: observe(E[n]) for n in names}

def field(o):
    a, p = o
    return a * np.exp(1j * p)

def vis_corr(fa, fb):
    num = np.abs(np.vdot(fa, fb))
    den = math.sqrt(float(np.vdot(fa, fa).real) * float(np.vdot(fb, fb).real))
    return float(num / (den + 1e-300))

F = {n: field(obs[n]) for n in names}
self_v = [vis_corr(F[n], F[n]) for n in names]
cross, dist = [], []
for i, a in enumerate(names):
    for b in names[i+1:]:
        cross.append(vis_corr(F[a], F[b]))
        dist.append(math.dist((E[a]["S_k"],E[a]["S_t"],E[a]["S_e"]),
                              (E[b]["S_k"],E[b]["S_t"],E[b]["S_e"])))
print(json.dumps({
  "self_all_exactly_1": bool(np.allclose(self_v,1.0,atol=1e-12)),
  "self_min": round(float(np.min(self_v)),12),
  "cross_mean": round(float(np.mean(cross)),6),
  "cross_max": round(float(np.max(cross)),6),
  "n_cross_ge_1": int(sum(1 for c in cross if c >= 1.0-1e-12)),
  "corr_vis_vs_distance": round(float(np.corrcoef(dist,cross)[0,1]),6),
}, indent=2))
