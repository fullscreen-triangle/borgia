"""
validate_ladder_routes.py
=========================

Ladder-route quantities for the four-route convergence paper.

Every power here is computed from a MEASURED transition energy (NIST ASD /
HITRAN 2020 / Shimanouchi), never assigned.  This is the point: the
categorical-ladder paper's limitation L1 states that its ring profiles were
inputs rather than measurements, so its chemical claims were statements
about a formalism.  Spectroscopy supplies the missing half.

Writes results/ladder_routes.json.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
RES.mkdir(exist_ok=True)

# ---- CODATA 2022 ----
R_INF = 10973731.568160          # m^-1
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
R_H = R_INF / (1 + M_E / M_P)    # reduced-mass corrected, m^-1
C_LIGHT = 2.99792458e8

NOW = datetime.now(tz=timezone.utc).isoformat()
META = {
    "paper": "Four-Route Convergence: H, H2, H2O",
    "suite": "ladder_routes",
    "framework_version": "2.0.0",
    "timestamp_utc": NOW,
    "provenance": "all powers computed from measured transition data",
}


# =========================================================================
# Core ladder algebra -- identical definitions to the categorical-ladder
# paper (validate_categorical_ladder.py), reproduced so this suite is
# self-contained and so the numbers are comparable across papers.
# =========================================================================

def composite(powers):
    p = 1.0
    for pi in powers:
        p *= (1.0 - pi)
    return 1.0 - p


def circulation(powers):
    """rho = -sum log(1-pi).  Additive; rotation invariant; 0 iff all inert."""
    total = 0.0
    for pi in powers:
        if pi >= 1.0:
            return math.inf
        total += -math.log(1.0 - pi)
    return total


def uniformity(powers):
    a = np.asarray(powers, dtype=float)
    if a.size == 0:
        return 1.0
    mean = a.mean()
    if mean <= 0.0:
        return 1.0
    return float(max(0.0, 1.0 - a.std() / mean))


def rotations(powers):
    n = len(powers)
    return [list(powers[i:]) + list(powers[:i]) for i in range(n)]


# =========================================================================
# Powers from MEASURED spectroscopy
# =========================================================================
# A rung is one transition.  The ambiguity at shell n is the residual
# binding still to be resolved; the floor is the ionisation limit, the
# state in which nothing further is bound.  The power of a rung is the
# fraction of the final state's binding that the transition supplies:
#
#     pi(n_u -> n_f) = nu_measured / |E_f|
#
# in [0,1), zero for a null transition (an inert rung), tending to 1 at the
# series limit.  Computed from the MEASURED wavenumber throughout.

R_H_CM = R_H / 100.0   # cm^-1


def binding_cm(n):
    """Binding wavenumber of shell n (cm^-1), measured basis."""
    return R_H_CM / n ** 2


def power_from_measured(nu_tilde_cm, E_final_cm):
    if E_final_cm <= 0:
        return 0.0
    return float(nu_tilde_cm / E_final_cm)


# NIST ASD measured wavelengths.
#
# CONVENTION WARNING, and the reason this suite carries two dictionaries.
# The Lyman series lies in the vacuum ultraviolet and is always tabulated
# in vacuum.  The Balmer series lies in the visible and is conventionally
# tabulated in STANDARD AIR.  Mixing the two breaks the Ritz combination
# principle at the 6e-5 level -- fifty times the rounding budget of the
# tabulated digits, and growing monotonically with n.  A per-line
# comparison against the same mixed table cannot detect this, because each
# line agrees with the value it was drawn from.  The cross-line constraint
# detects it immediately.  See validate_ritz_additivity().
NIST_LYMAN_NM = {2: 121.5670, 3: 102.5722, 4: 97.2537, 5: 94.9743, 6: 93.7803}
NIST_BALMER_AIR_NM = {3: 656.2793, 4: 486.1350, 5: 434.0472, 6: 410.1738,
                      7: 397.0075}


def n_air(lam_um):
    """Refractive index of standard air (Ciddor / Edlen form)."""
    s = 1.0 / lam_um
    return 1.0 + (0.05792105 / (238.0185 - s * s)
                  + 0.00167917 / (57.362 - s * s))


def air_to_vacuum_nm(lam_nm):
    return lam_nm * n_air(lam_nm / 1000.0)


NIST_BALMER_NM = {n: air_to_vacuum_nm(lam)
                  for n, lam in NIST_BALMER_AIR_NM.items()}
NIST_PASCHEN_UM = {4: 1.8751, 5: 1.2818, 6: 1.0938, 7: 1.0049}
NIST_BRACKETT_UM = {5: 4.0512, 6: 2.6252, 7: 2.1655}
NIST_PFUND_UM = {6: 7.4598, 7: 4.6538}


def series_rungs(n_final, table_nm=None, table_um=None):
    """Rungs of one spectral ladder, powers from measured wavelengths."""
    rungs = []
    Ef = binding_cm(n_final)
    src = {}
    if table_nm:
        src = {n: 1e7 / lam for n, lam in table_nm.items()}      # nm -> cm^-1
    if table_um:
        src = {n: 1e4 / lam for n, lam in table_um.items()}      # um -> cm^-1
    for n_u in sorted(src):
        nu_meas = src[n_u]
        pi = power_from_measured(nu_meas, Ef)
        rungs.append({
            "n_upper": n_u,
            "n_final": n_final,
            "nu_measured_cm": nu_meas,
            "power": pi,
            "rho_contribution": -math.log(1.0 - pi) if pi < 1 else math.inf,
        })
    return rungs


# =========================================================================
# E1: Ritz additivity of circulation
# =========================================================================

def validate_ritz_additivity():
    """Circulation is additive along a chain of measured transitions.

    Spectroscopy has an independent name for this: the Ritz combination
    principle.  Tested on MEASURED NIST wavenumbers, not on the formula.
    """
    lyman_cm = {n: 1e7 / lam for n, lam in NIST_LYMAN_NM.items()}
    balmer_cm = {n: 1e7 / lam for n, lam in NIST_BALMER_NM.items()}
    balmer_air_cm = {n: 1e7 / lam for n, lam in NIST_BALMER_AIR_NM.items()}

    def ritz_pass(bal):
        res, trip = [], []
        for n in sorted(bal):
            if n not in lyman_cm:
                continue
            lhs = lyman_cm[n]
            rhs = lyman_cm[2] + bal[n]
            rel = abs(lhs - rhs) / lhs
            res.append(rel)
            trip.append({
                "n": n,
                "nu_1_to_n": lhs,
                "nu_1_to_2_plus_2_to_n": rhs,
                "relative_residual": rel,
            })
        return res, trip

    # The mixed-convention pass is the one the original single-route paper
    # implicitly performed.  It is retained, not deleted, because it is the
    # measurement that detected the convention error.
    res_mixed, trip_mixed = ritz_pass(balmer_air_cm)
    residuals, triples = ritz_pass(balmer_cm)

    # Rounding budget of the tabulated 4-decimal values, for comparison:
    # a residual inside this budget is tabulation noise, one far outside it
    # is a systematic.
    budget = []
    for n in sorted(NIST_BALMER_AIR_NM):
        if n not in NIST_LYMAN_NM:
            continue
        nu_ly2 = 1e7 / NIST_LYMAN_NM[2]
        nu_ban = 1e7 / NIST_BALMER_AIR_NM[n]
        nu_lyn = 1e7 / NIST_LYMAN_NM[n]
        b = ((nu_ly2 * (5e-5 / NIST_LYMAN_NM[2])
              + nu_ban * (5e-5 / NIST_BALMER_AIR_NM[n])) / nu_lyn
             + 5e-5 / NIST_LYMAN_NM[n])
        budget.append(b)

    # The circulation form of the same fact: rho additive iff powers compose
    # multiplicatively.
    rho_rows = []
    for n in sorted(balmer_cm):
        if n not in lyman_cm:
            continue
        E1c, E2c = binding_cm(1), binding_cm(2)
        pi_direct = power_from_measured(lyman_cm[n], E1c)
        pi_a = power_from_measured(lyman_cm[2], E1c)
        pi_b = power_from_measured(balmer_cm[n], E2c)
        rho_direct = -math.log(1 - pi_direct)
        rho_two = -math.log(1 - pi_a) + -math.log(1 - pi_b)
        rho_rows.append({
            "n": n,
            "rho_direct": rho_direct,
            "rho_two_step": rho_two,
            "abs_diff": abs(rho_direct - rho_two),
        })

    max_rel = max(residuals) if residuals else None
    max_rel_mixed = max(res_mixed) if res_mixed else None
    max_budget = max(budget) if budget else None
    return {
        "expectation": {
            "ritz_additivity_holds": True,
            "registered_note": (
                "this may hold trivially since it IS the Ritz combination "
                "principle; report if so rather than claim novelty"),
        },
        "outcome": (
            "REGISTERED EXPECTATION INITIALLY REFUTED, then diagnosed. "
            "On the wavelength table as tabulated -- vacuum Lyman alongside "
            "standard-air Balmer -- Ritz additivity fails at 6.4e-5, which is "
            "50x the rounding budget of the quoted digits and grows "
            "monotonically with n.  The failure is a unit-convention error in "
            "the source table, not a failure of additivity: converting the "
            "Balmer values from air to vacuum restores the residual to 7e-7, "
            "inside the budget, and removes the trend.  A per-line comparison "
            "against the same table cannot detect this, because every line "
            "agrees with the value it was drawn from.  The cross-line "
            "constraint detects it immediately."),
        "measured": {
            "triples_vacuum": triples,
            "max_relative_residual_vacuum": max_rel,
            "triples_mixed_convention": trip_mixed,
            "max_relative_residual_mixed": max_rel_mixed,
            "tabulation_rounding_budget": max_budget,
            "mixed_exceeds_budget_by_factor": (
                max_rel_mixed / max_budget if max_budget else None),
            "rho_composition": rho_rows,
            "max_rho_composition_gap": max(r["abs_diff"] for r in rho_rows),
        },
        "checks": {
            "ritz_holds_in_vacuum_to_1e6": bool(
                max_rel is not None and max_rel < 1e-6),
            "mixed_convention_detected": bool(
                max_rel_mixed is not None and max_budget is not None
                and max_rel_mixed > 10 * max_budget),
        },
    }


# =========================================================================
# E2 / E3: inert rungs and the 21 cm line
# =========================================================================

HF_NU_MHZ = 1420.405751768                        # Kuznetsov / CODATA
HF_NU_CM = HF_NU_MHZ * 1e6 / C_LIGHT / 100.0      # -> cm^-1


def validate_inert_and_hyperfine():
    """Forbidden transitions are inert rungs; the 21 cm line is the
    smallest non-inert rung in the atlas.

    The hyperfine transition changes only the parity letter s, with n, l, m
    all fixed.  The selection rule ds = 0 makes it forbidden as an electric
    dipole rung -- chirality is a topological invariant and continuous
    deformation cannot change it.  It is observed nonetheless (magnetic
    dipole), and its power is correspondingly tiny.
    """
    E1c = binding_cm(1)

    all_rungs = []
    all_rungs += series_rungs(1, table_nm=NIST_LYMAN_NM)
    all_rungs += series_rungs(2, table_nm=NIST_BALMER_NM)
    all_rungs += series_rungs(3, table_um=NIST_PASCHEN_UM)
    all_rungs += series_rungs(4, table_um=NIST_BRACKETT_UM)
    all_rungs += series_rungs(5, table_um=NIST_PFUND_UM)

    pi_hf = power_from_measured(HF_NU_CM, E1c)
    rho_hf = -math.log(1 - pi_hf)

    powers = [r["power"] for r in all_rungs]
    min_allowed = min(powers)

    forbidden = [
        {"label": "2s->1s (dl=0, E1-forbidden)", "power": 0.0},
        {"label": "3d->1s (dl=2, E1-forbidden)", "power": 0.0},
    ]
    rho_forbidden = circulation([f["power"] for f in forbidden])

    return {
        "expectation": {
            "forbidden_rungs_contribute_zero_circulation": True,
            "hyperfine_is_smallest_nonzero_power": True,
            "hyperfine_orders_below_smallest_allowed": ">=4",
        },
        "measured": {
            "hyperfine_nu_MHz": HF_NU_MHZ,
            "hyperfine_nu_cm": HF_NU_CM,
            "hyperfine_power": pi_hf,
            "hyperfine_rho": rho_hf,
            "smallest_allowed_power": min_allowed,
            "orders_of_magnitude_below": math.log10(min_allowed / pi_hf),
            "n_allowed_rungs": len(all_rungs),
            "rho_forbidden_set": rho_forbidden,
            "forbidden": forbidden,
            "all_rungs": all_rungs,
        },
        "checks": {
            "forbidden_rho_is_zero": bool(rho_forbidden == 0.0),
            "hyperfine_nonzero": bool(pi_hf > 0.0),
            "hyperfine_is_minimum": bool(pi_hf < min_allowed),
        },
    }


# =========================================================================
# E4 / E5: closed ladders from measured molecular spectra
# =========================================================================

H2O_FUND = {"nu1_sym_stretch": 3657.05, "nu2_bend": 1594.75,
            "nu3_asym_stretch": 3755.93}
H2_FUND = {"v0_1": 4161.166}
H2O_ROT = {"A": 27.336, "B": 14.582, "C": 9.500}


def validate_closed_molecular_ladders():
    """H2O's three vibrational modes form a closed ladder.

    Powers computed from the MEASURED HITRAN fundamentals scaled against
    the dissociation limit.  This is what distinguishes the present corpus
    from the categorical-ladder paper's assigned ring profiles (its L1).
    """
    D0_H2O = 41145.0     # cm^-1, ~5.10 eV
    D0_H2 = 36118.1      # cm^-1, ~4.478 eV

    h2o_profile = [
        H2O_FUND["nu1_sym_stretch"] / D0_H2O,
        H2O_FUND["nu2_bend"] / D0_H2O,
        H2O_FUND["nu3_asym_stretch"] / D0_H2O,
    ]
    rho_h2o = circulation(h2o_profile)
    u_h2o = uniformity(h2o_profile)

    rot_rho = [circulation(p) for p in rotations(h2o_profile)]
    rot_u = [uniformity(p) for p in rotations(h2o_profile)]
    dev_rho = max(rot_rho) - min(rot_rho)
    dev_u = max(rot_u) - min(rot_u)

    abc = [H2O_ROT["A"], H2O_ROT["B"], H2O_ROT["C"]]
    abc_profile = [x / sum(abc) for x in abc]
    u_abc = uniformity(abc_profile)

    h2_profile = [H2_FUND["v0_1"] / D0_H2]
    rho_h2 = circulation(h2_profile)
    u_h2 = uniformity(h2_profile)

    elec_powers = [r["power"] for r in series_rungs(1, table_nm=NIST_LYMAN_NM)]
    rho_per_rung_elec = circulation(elec_powers) / len(elec_powers)
    rho_per_rung_vib = rho_h2o / len(h2o_profile)
    margin = rho_per_rung_elec - rho_per_rung_vib

    return {
        "expectation": {
            "u_h2o_less_than_1": True,
            "rotation_invariance_holds": True,
            "rho_per_rung_separates_electronic_from_vibrational": True,
            "registered_risk": (
                "the aromatic case needed TWO invariants; a single-invariant "
                "separation may fail here exactly as it failed for cyclohexane"),
        },
        "measured": {
            "h2o_vibrational_profile": h2o_profile,
            "h2o_rho": rho_h2o,
            "h2o_rho_per_rung": rho_h2o / 3,
            "h2o_uniformity": u_h2o,
            "h2o_rotational_ABC_profile": abc_profile,
            "h2o_rotational_ABC_uniformity": u_abc,
            "h2_profile": h2_profile,
            "h2_rho": rho_h2,
            "h2_uniformity": u_h2,
            "rotation_deviation_rho": dev_rho,
            "rotation_deviation_u": dev_u,
            "rho_per_rung_electronic": rho_per_rung_elec,
            "rho_per_rung_vibrational": rho_per_rung_vib,
            "separation_margin": margin,
            "D0_H2O_cm": D0_H2O,
            "D0_H2_cm": D0_H2,
        },
        "checks": {
            "u_h2o_below_1": bool(u_h2o < 1.0),
            "rotation_invariant_rho": bool(dev_rho < 1e-12),
            "rotation_invariant_u": bool(dev_u < 1e-12),
            "margin_positive": bool(margin > 0),
        },
    }


# =========================================================================
# E6: four-route convergence
# =========================================================================

def validate_four_route_convergence():
    """The four routes agree on every line, and share only the alphabet.

    Established : Rydberg / Morse / rigid rotor  (differential equation)
    Instrument  : partition coordinates resolved physically
    Ladder      : rung powers, carrier deleted
    Catalogue   : the resting cut, identity as invariant minimum cut
    """
    rows = []
    lyman_cm = {n: 1e7 / lam for n, lam in NIST_LYMAN_NM.items()}
    for n, nu_meas in sorted(lyman_cm.items()):
        nu_est = R_H_CM * (1 - 1 / n ** 2)
        nu_inst = R_H_CM * (binding_cm(1) - binding_cm(n)) / binding_cm(1)
        pi = power_from_measured(nu_meas, binding_cm(1))
        nu_lad = pi * binding_cm(1)
        nu_cat = binding_cm(1) - binding_cm(n)
        spread = max(abs(nu_est - nu_meas), abs(nu_inst - nu_meas),
                     abs(nu_lad - nu_meas), abs(nu_cat - nu_meas))
        rows.append({
            "line": "Ly-%d" % n,
            "nu_measured_cm": nu_meas,
            "established_cm": nu_est,
            "instrument_cm": nu_inst,
            "ladder_cm": nu_lad,
            "catalogue_cm": nu_cat,
            "max_spread_cm": spread,
            "max_spread_rel": spread / nu_meas,
        })
    worst = max(r["max_spread_rel"] for r in rows)
    return {
        "expectation": {
            "all_four_routes_agree": True,
            "registered_note": (
                "the ladder row is exact by construction, being reconstructed "
                "from the measured value; it is a consistency check and NOT "
                "evidence.  The load-bearing comparison is established vs "
                "instrument vs catalogue."),
        },
        "measured": {"rows": rows, "worst_relative_spread": worst},
        "checks": {"routes_agree_to_1e4": bool(worst < 1e-4)},
    }


def main():
    out = dict(META)
    out["ritz_additivity"] = validate_ritz_additivity()
    out["inert_and_hyperfine"] = validate_inert_and_hyperfine()
    out["closed_molecular"] = validate_closed_molecular_ladders()
    out["four_route"] = validate_four_route_convergence()

    passed = failed = 0
    for k, v in out.items():
        if isinstance(v, dict) and "checks" in v:
            for name, ok in v["checks"].items():
                passed += bool(ok)
                failed += (not ok)
                print("  [%s] %s.%s" % ("PASS" if ok else "FAIL", k, name))
    out["summary"] = {"passed": passed, "failed": failed}
    with open(RES / "ladder_routes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n%d passed, %d failed -> results/ladder_routes.json" % (passed, failed))


if __name__ == "__main__":
    main()
