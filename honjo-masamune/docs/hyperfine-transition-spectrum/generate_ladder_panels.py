"""
generate_ladder_panels.py
=========================

Figure panels for the ladder and convergence routes.  Style matches
validate_spectra.py: 20 x 4.8 inches, four subpanels, one of them 3D.

  panel_5_four_routes.png    convergence of the four derivation routes
  panel_6_ladder_rungs.png   powers and circulation from measured spectra
  panel_7_hyperfine_rung.png the 21 cm line as the near-inert parity rung
  panel_8_closed_ladders.png closed molecular ladders, rho and u

Reads results/ladder_routes.json (run validate_ladder_routes.py first).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

D = json.loads((RES / "ladder_routes.json").read_text(encoding="utf-8"))

# palette, consistent with the existing atlas panels
C_EST = "#a855f7"     # established
C_INS = "#22c55e"     # instrument
C_LAD = "#f97316"     # ladder
C_CAT = "#3b82f6"     # catalogue
C_HF = "#58E6D9"      # hyperfine
C_BAD = "#ef4444"
C_GOOD = "#22c55e"


def _finish(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/%s" % name)


# =========================================================================
# Panel 5: four-route convergence
# =========================================================================

def panel_four_routes():
    fig = plt.figure(figsize=(20, 4.8))
    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.88,
                        wspace=0.32)
    rows = D["four_route"]["measured"]["rows"]
    labels = [r["line"] for r in rows]
    x = np.arange(len(rows))

    # (A) signed residual from measurement, per route.  Plotting the raw
    # wavenumbers would show four bars of equal height and say nothing; the
    # residual is where the routes are distinguishable, and it shows that
    # three of them are not.
    ax = fig.add_subplot(1, 4, 1)
    w = 0.2
    for off, key, cc, lab in [(-1.5, "established", C_EST, "established"),
                              (-0.5, "instrument", C_INS, "instrument"),
                              (0.5, "ladder", C_LAD, "ladder"),
                              (1.5, "catalogue", C_CAT, "catalogue")]:
        vals = [1e6 * (r[key + "_cm"] - r["nu_measured_cm"]) / r["nu_measured_cm"]
                for r in rows]
        ax.bar(x + off * w, vals, w, label=lab, color=cc, edgecolor="black",
               linewidth=0.4)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("residual from measurement (ppm)", fontsize=9)
    ax.set_title("(A) Three routes coincide exactly;\nthe ladder is "
                 "reconstructed", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2, framealpha=0.95)
    ax.set_ylim(-14.5, 1.5)
    ax.grid(True, alpha=0.3, axis="y")

    # (B) pairwise spread AMONG the three predictive routes, which is the
    # quantity that would reveal a disagreement.  It is identically zero:
    # they are three derivations of one closed form, not three independent
    # predictions.  Reporting this as if it were confirmation would be the
    # error the single-route paper made.
    ax = fig.add_subplot(1, 4, 2)
    inter = []
    for r in rows:
        v = [r["established_cm"], r["instrument_cm"], r["catalogue_cm"]]
        inter.append(max(max(v) - min(v), 1e-17) / r["nu_measured_cm"])
    common = [abs(r["established_cm"] - r["nu_measured_cm"]) / r["nu_measured_cm"]
              for r in rows]
    # The inter-route spread is machine zero.  Drawing it on a log axis
    # would render floating-point dust as though it were a measurement, so
    # it is clamped to a visible floor and labelled as exactly zero.
    FLOOR = 1e-13
    ax.bar(x - 0.2, [FLOOR] * len(inter), 0.4,
           label="spread among the 3 routes (= 0)",
           color="#94a3b8", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.2, common, 0.4, label="common gap to measurement",
           color=C_BAD, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 3, 3e-5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("relative", fontsize=9)
    ax.set_title("(B) Zero spread between routes;\na shared residual to "
                 "experiment", fontsize=10)
    ax.legend(fontsize=6.5, loc="center right")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.text(x[0] - 0.42, FLOOR * 1.6, "exactly 0 (identical closed form)",
            fontsize=6.5, style="italic")

    # (C) the convention finding: mixed vs vacuum Ritz residual
    ax = fig.add_subplot(1, 4, 3)
    m = D["ritz_additivity"]["measured"]
    ns = [t["n"] for t in m["triples_mixed_convention"]]
    mixed = [t["relative_residual"] for t in m["triples_mixed_convention"]]
    vac = [t["relative_residual"] for t in m["triples_vacuum"]]
    ax.plot(ns, mixed, "o-", color=C_BAD, ms=8, label="as tabulated (mixed)")
    ax.plot(ns, vac, "s-", color=C_GOOD, ms=8, label="Balmer -> vacuum")
    ax.axhline(m["tabulation_rounding_budget"], color="k", ls=":", lw=1.2,
               label="rounding budget")
    ax.set_yscale("log")
    ax.set_xlabel(r"upper level $n$", fontsize=9)
    ax.set_ylabel("Ritz relative residual", fontsize=9)
    ax.set_title("(C) Cross-line constraint detects\na convention error",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # (D) 3D: route x line x log residual against measurement
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    routes = ["established", "instrument", "ladder", "catalogue"]
    cols = [C_EST, C_INS, C_LAD, C_CAT]
    for j, (rt, cc) in enumerate(zip(routes, cols)):
        zs = []
        for r in rows:
            resid = abs(r[rt + "_cm"] - r["nu_measured_cm"]) / r["nu_measured_cm"]
            zs.append(math.log10(max(resid, 1e-17)))
        ax.scatter(np.full(len(rows), j), x, zs, s=55, c=cc,
                   edgecolors="black", linewidths=0.4, depthshade=False)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["est", "inst", "lad", "cat"], fontsize=7)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_zlabel(r"$\log_{10}$ rel. residual", fontsize=8, labelpad=2)
    ax.tick_params(axis="z", labelsize=7, pad=1)
    ax.set_title("(D) Residual landscape", fontsize=10)
    ax.view_init(elev=20, azim=-58)

    _finish(fig, "panel_5_four_routes.png")


# =========================================================================
# Panel 6: rung powers and circulation from measured spectra
# =========================================================================

def panel_ladder_rungs():
    fig = plt.figure(figsize=(20, 4.8))
    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.88,
                        wspace=0.32)
    rungs = D["inert_and_hyperfine"]["measured"]["all_rungs"]
    by_series = {}
    for r in rungs:
        by_series.setdefault(r["n_final"], []).append(r)
    names = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund"}
    cols = {1: "#a855f7", 2: "#22c55e", 3: "#f97316", 4: "#3b82f6",
            5: "#ef4444"}

    # (A) power vs upper level, by series
    ax = fig.add_subplot(1, 4, 1)
    for nf in sorted(by_series):
        rs = sorted(by_series[nf], key=lambda r: r["n_upper"])
        ax.plot([r["n_upper"] for r in rs], [r["power"] for r in rs],
                "o-", color=cols[nf], ms=7, label=names[nf])
    ax.set_xlabel(r"upper level $n_u$", fontsize=9)
    ax.set_ylabel(r"rung power $\pi$", fontsize=9)
    ax.set_title("(A) Powers from measured lines", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (B) circulation contribution per rung
    ax = fig.add_subplot(1, 4, 2)
    for nf in sorted(by_series):
        rs = sorted(by_series[nf], key=lambda r: r["n_upper"])
        ax.plot([r["n_upper"] for r in rs],
                [r["rho_contribution"] for r in rs],
                "s-", color=cols[nf], ms=7, label=names[nf])
    ax.set_xlabel(r"upper level $n_u$", fontsize=9)
    ax.set_ylabel(r"$-\log(1-\pi)$", fontsize=9)
    ax.set_title(r"(B) Circulation contribution", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (C) total circulation per series and per rung
    ax = fig.add_subplot(1, 4, 3)
    ser = sorted(by_series)
    tot = [sum(r["rho_contribution"] for r in by_series[nf]) for nf in ser]
    per = [t / len(by_series[nf]) for t, nf in zip(tot, ser)]
    xx = np.arange(len(ser))
    ax.bar(xx - 0.2, tot, 0.4, label=r"$\varrho$", color="#7c3aed",
           edgecolor="black", linewidth=0.4)
    ax.bar(xx + 0.2, per, 0.4, label=r"$\varrho/n$", color="#fbbf24",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(xx)
    ax.set_xticklabels([names[n] for n in ser], fontsize=8, rotation=20)
    ax.set_ylabel("circulation", fontsize=9)
    ax.set_title("(C) Circulation by series", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # (D) 3D: (n_final, n_upper, power)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    for nf in sorted(by_series):
        rs = by_series[nf]
        ax.scatter([nf] * len(rs), [r["n_upper"] for r in rs],
                   [r["power"] for r in rs], s=60, c=cols[nf],
                   edgecolors="black", linewidths=0.4, depthshade=False,
                   label=names[nf])
    ax.set_xlabel(r"$n_f$", fontsize=8)
    ax.set_ylabel(r"$n_u$", fontsize=8)
    ax.set_zlabel(r"$\pi$", fontsize=8)
    ax.set_title("(D) Rung landscape", fontsize=10)
    ax.view_init(elev=22, azim=-60)

    _finish(fig, "panel_6_ladder_rungs.png")


# =========================================================================
# Panel 7: the hyperfine line as the near-inert parity rung
# =========================================================================

def panel_hyperfine_rung():
    fig = plt.figure(figsize=(20, 4.8))
    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.88,
                        wspace=0.32)
    hf = D["inert_and_hyperfine"]["measured"]
    rungs = hf["all_rungs"]
    powers = sorted(r["power"] for r in rungs)

    # (A) power spectrum with the hyperfine rung marked
    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(range(len(powers)), powers, s=55, c="#7c3aed",
               edgecolors="black", linewidths=0.4, label="allowed E1 rungs",
               zorder=3)
    ax.axhline(hf["hyperfine_power"], color=C_HF, lw=2,
               label="21 cm parity rung")
    ax.axhline(0.0, color="k", ls="--", lw=1.2, label="inert (forbidden)")
    ax.set_yscale("log")
    ax.set_xlabel("rung index (sorted)", fontsize=9)
    ax.set_ylabel(r"power $\pi$", fontsize=9)
    ax.set_title("(A) The 21 cm rung against the atlas", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3, which="both")

    # (B) orders of magnitude gap
    ax = fig.add_subplot(1, 4, 2)
    cats = ["forbidden\n(inert)", "21 cm\nparity rung", "smallest\nallowed E1"]
    vals = [1e-12, hf["hyperfine_power"], hf["smallest_allowed_power"]]
    bars = ax.bar(cats, vals, color=["#94a3b8", C_HF, "#7c3aed"],
                  edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"power $\pi$", fontsize=9)
    ax.set_title("(B) %.2f orders below\nthe smallest allowed rung"
                 % hf["orders_of_magnitude_below"], fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6,
                ("0 (exactly)" if v == 1e-12 else "%.2e" % v),
                ha="center", fontsize=7)

    # (C) the word rewrite: which letters change, for the 21 cm rung set
    # against representative allowed rungs.  A rung is a letter rewrite, so
    # the comparison that matters is which positions move.
    ax = fig.add_subplot(1, 4, 3)
    letters = ["$n$", r"$\ell$", "$m$", "$s$"]
    cases = [
        ("Ly-$\\alpha$\n$2p\\to1s$", [1, 1, 0, 0], "#a855f7"),
        ("H-$\\alpha$\n$3\\to2$", [1, 1, 0, 0], "#22c55e"),
        ("$2s\\to1s$\nforbidden", [1, 0, 0, 0], "#94a3b8"),
        ("21 cm\nparity", [0, 0, 0, 1], C_HF),
    ]
    xx = np.arange(len(letters))
    w = 0.2
    for k, (nm, pat, cc) in enumerate(cases):
        ax.bar(xx + (k - 1.5) * w, pat, w, label=nm, color=cc,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(xx)
    ax.set_xticklabels(letters, fontsize=10)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fixed", "rewritten"], fontsize=8)
    ax.set_title("(C) Which letter the rung rewrites", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="upper center", framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    # (D) 3D: log power vs transition energy vs series
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    nus = [r["nu_measured_cm"] for r in rungs]
    pws = [r["power"] for r in rungs]
    nfs = [r["n_final"] for r in rungs]
    ax.scatter([math.log10(v) for v in nus], nfs,
               [math.log10(p) for p in pws], s=55, c="#7c3aed",
               edgecolors="black", linewidths=0.4, depthshade=False)
    ax.scatter([math.log10(hf["hyperfine_nu_cm"])], [1],
               [math.log10(hf["hyperfine_power"])], s=160, c=C_HF,
               marker="*", edgecolors="black", linewidths=0.6,
               depthshade=False)
    ax.set_xlabel(r"$\log_{10}\tilde\nu$", fontsize=8)
    ax.set_ylabel(r"$n_f$", fontsize=8)
    ax.set_zlabel(r"$\log_{10}\pi$", fontsize=8)
    ax.set_title("(D) Ten decades of transition energy", fontsize=10)
    ax.view_init(elev=20, azim=-62)

    _finish(fig, "panel_7_hyperfine_rung.png")


# =========================================================================
# Panel 8: closed molecular ladders
# =========================================================================

def panel_closed_ladders():
    fig = plt.figure(figsize=(20, 4.8))
    plt.subplots_adjust(left=0.045, right=0.99, bottom=0.13, top=0.88,
                        wspace=0.32)
    c = D["closed_molecular"]["measured"]
    prof = c["h2o_vibrational_profile"]
    mode_names = [r"$\nu_1$" "\nsym str", r"$\nu_2$" "\nbend",
                  r"$\nu_3$" "\nasym str"]

    # (A) the closed profile, drawn as a cycle
    ax = fig.add_subplot(1, 4, 1)
    ang = np.linspace(0, 2 * np.pi, len(prof), endpoint=False) + np.pi / 2
    xs, ys = np.cos(ang), np.sin(ang)
    for i in range(len(prof)):
        j = (i + 1) % len(prof)
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "-", color="#94a3b8", lw=1.5,
                zorder=1)
    ax.scatter(xs, ys, s=[3000 * p for p in prof], c=prof,
                    cmap="viridis", edgecolors="black", linewidths=0.8,
                    zorder=3)
    for i, nm in enumerate(mode_names):
        ax.text(xs[i] * 1.45, ys[i] * 1.45, "%s\n%.4f" % (nm, prof[i]),
                ha="center", va="center", fontsize=7)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.9, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) H$_2$O vibrational cycle\n(powers from HITRAN)",
                 fontsize=10)
    ax.text(0, -1.72, r"$\varrho=%.4f$   $u=%.4f$"
            % (c["h2o_rho"], c["h2o_uniformity"]), ha="center", fontsize=8.5)

    # (B) rotation invariance.  Every rotation of the cycle gives the same
    # rho and the same u, so the deviation is machine zero; plotting it on a
    # log axis would dress floating-point dust as data.  Show the invariants
    # under each rotation instead, which is the substantive statement.
    ax = fig.add_subplot(1, 4, 2)
    n = len(prof)
    rots = [list(prof[i:]) + list(prof[:i]) for i in range(n)]
    rho_r = [-sum(math.log(1 - p) for p in r) for r in rots]
    u_r = []
    for r in rots:
        a = np.asarray(r)
        u_r.append(max(0.0, 1 - a.std() / a.mean()))
    xr = np.arange(n)
    ax.plot(xr, rho_r, "o-", color="#7c3aed", ms=9, label=r"$\varrho$")
    ax.plot(xr, u_r, "s-", color="#fbbf24", ms=9, label="$u$")
    ax.set_xticks(xr)
    ax.set_xticklabels(["rot %d" % i for i in range(n)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("invariant value", fontsize=9)
    ax.set_title("(B) Flat under every rotation\n(deviation $<10^{-16}$)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (C) the separation margin
    ax = fig.add_subplot(1, 4, 3)
    vals = [c["rho_per_rung_electronic"], c["rho_per_rung_vibrational"]]
    ax.bar(["electronic\n(Lyman)", "vibrational\n(H$_2$O)"], vals,
           color=["#a855f7", "#0ea5e9"], edgecolor="black", linewidth=0.5)
    ax.annotate("", xy=(0.5, vals[0]), xytext=(0.5, vals[1]),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
    ax.text(0.5, (vals[0] + vals[1]) / 2, " margin %+.3f"
            % c["separation_margin"], fontsize=8.5, ha="center",
            va="center", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec="black", alpha=0.95))
    ax.set_ylim(0, vals[0] * 1.22)
    ax.set_ylabel(r"$\varrho/n$", fontsize=9)
    ax.set_title("(C) Circulation per rung separates\nthe two modalities",
                 fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # (D) 3D: uniformity vs circulation per rung vs cycle length
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    pts = [
        ("H$_2$O vib", c["h2o_rho_per_rung"], c["h2o_uniformity"], 3, "#0ea5e9"),
        ("H$_2$O rot ABC", c["h2o_rho_per_rung"],
         c["h2o_rotational_ABC_uniformity"], 3, "#14b8a6"),
        ("H$_2$ vib", c["h2_rho"], c["h2_uniformity"], 1, "#f59e0b"),
        ("Lyman", c["rho_per_rung_electronic"], 1.0, 5, "#a855f7"),
    ]
    for nm, rr, uu, nn, cc in pts:
        ax.scatter([rr], [uu], [nn], s=110, c=cc, edgecolors="black",
                   linewidths=0.5, depthshade=False, label=nm)
    ax.set_xlabel(r"$\varrho/n$", fontsize=8)
    ax.set_ylabel("$u$", fontsize=8)
    ax.set_zlabel("cycle length", fontsize=8)
    ax.set_title("(D) Closed-ladder coordinates", fontsize=10)
    ax.legend(fontsize=6, loc="upper left")
    ax.view_init(elev=18, azim=-64)

    _finish(fig, "panel_8_closed_ladders.png")


def main():
    panel_four_routes()
    panel_ladder_rungs()
    panel_hyperfine_rung()
    panel_closed_ladders()
    print("\n4 panels written to figures/")


if __name__ == "__main__":
    main()
