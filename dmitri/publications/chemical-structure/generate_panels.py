#!/usr/bin/env python3
"""
Generate 8 publication-quality panels for
"Why There Are Chemical Structures At All".

Each panel: 1 row x 4 columns, white background, >=1 3D chart, minimal
text, no titles. All charts data-driven (no conceptual/text/table charts).
Data come from the validation JSON in results/ and from the paper's models.

Requires: numpy, matplotlib
"""

import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


shell = load("shell_capacity.json")
valence = load("valence.json")
bonding = load("bonding_criterion.json")
stoich = load("stoichiometry.json")
geom = load("bond_geometry.json")
d3 = load("d3_axis_exchange.json")
blen = load("bond_length.json")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 300,
    "font.size": 9, "axes.titlesize": 0, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

BLUE, GREEN, ORANGE, RED, PURPLE = "#3b82f6", "#22c55e", "#f97316", "#ef4444", "#8b5cf6"


def newfig():
    return plt.figure(figsize=(20, 5), facecolor="white")


def save(fig, name):
    fig.tight_layout(pad=1.0)
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# thickness model used in the paper / validation
def thickness(nu, B0=1.0, kappa=1.0):
    return B0 + kappa * nu


# ===========================================================================
# PANEL 1 — Shell capacity, cumulative count, vacancy ladder
# ===========================================================================
def panel_1():
    fig = newfig()
    ns = np.arange(1, 11)
    Cn = 2 * ns**2
    cum = np.array([sum(2 * k**2 for k in range(1, n + 1)) for n in ns])

    # (1) 3D bars: shell capacity 2n^2 over n, height = capacity
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    xs = ns.astype(float)
    ys = np.zeros_like(xs)
    dz = Cn.astype(float)
    cols = plt.cm.plasma(dz / dz.max())
    ax.bar3d(xs, ys, np.zeros_like(dz), 0.6, 0.6, dz, color=cols, shade=True)
    ax.set_xlabel("n"); ax.set_ylabel(""); ax.set_zlabel(r"$C(n)=2n^2$")
    ax.view_init(elev=22, azim=-60)

    # (2) capacity 2n^2 curve + integer points
    ax = fig.add_subplot(1, 4, 2)
    nn = np.linspace(1, 10, 200)
    ax.plot(nn, 2 * nn**2, color=BLUE, lw=2)
    ax.scatter(ns, Cn, c=RED, s=45, zorder=5, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("n"); ax.set_ylabel(r"$C(n)$")

    # (3) cumulative cubic count
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(nn, (nn * (nn + 1) * (2 * nn + 1) / 3), color=GREEN, lw=2)
    ax.scatter(ns, cum, c=PURPLE, s=45, zorder=5, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("n"); ax.set_ylabel(r"$N_{\rm state}(n)$")

    # (4) vacancy per element (bars), noble = 0
    ax = fig.add_subplot(1, 4, 4)
    vr = shell["vacancy"]["results"]
    syms = [r["sym"] for r in vr]
    vac = [r["vacancy"] for r in vr]
    barcols = [RED if v == 0 else BLUE for v in vac]
    ax.bar(range(len(syms)), vac, color=barcols, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, rotation=90)
    ax.set_ylabel(r"vacancy $\nu$")
    return save(fig, "panel_1_capacity.png")


# ===========================================================================
# PANEL 2 — Valence = min(vacancy, co-vacancy)
# ===========================================================================
def panel_2():
    fig = newfig()
    vr = valence["results"]
    syms = [r["sym"] for r in vr]
    nu = np.array([r["vacancy"] for r in vr])
    cov = np.array([r["co_vacancy"] for r in vr])
    vder = np.array([r["valence_derived"] for r in vr])
    vref = np.array([r["valence_ref"] for r in vr])
    x = np.arange(len(syms))

    # (1) 3D scatter: (vacancy, co-vacancy, valence)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(nu, cov, vder, c=vder, cmap="viridis", s=55,
               edgecolors="k", linewidths=0.4)
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"co-$\nu$"); ax.set_zlabel("valence")
    ax.view_init(elev=20, azim=-55)

    # (2) derived vs reference valence (grouped bars)
    ax = fig.add_subplot(1, 4, 2)
    ax.bar(x - 0.2, vder, 0.4, color=BLUE, label="derived", edgecolor="k", linewidth=0.3)
    ax.bar(x + 0.2, vref, 0.4, color=ORANGE, label="reference", edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(syms, rotation=90)
    ax.set_ylabel("valence"); ax.legend(fontsize=7, frameon=False)

    # (3) min(nu, co-nu) curve over a sweep, with element points
    ax = fig.add_subplot(1, 4, 3)
    q = np.linspace(0, 8, 300)
    ax.plot(q, np.minimum(q, 8 - q), color=GREEN, lw=2)
    ax.scatter(cov, vder, c=RED, s=45, zorder=5, edgecolors="k", linewidths=0.4)
    ax.set_xlabel(r"valence electrons $q_v$"); ax.set_ylabel(r"$\min(\nu,\,8-\nu)$")

    # (4) identity scatter derived==reference
    ax = fig.add_subplot(1, 4, 4)
    ax.plot([0, 4], [0, 4], color="#94a3b8", lw=1, ls="--")
    jitter = (np.random.RandomState(1).rand(len(x)) - 0.5) * 0.12
    ax.scatter(vref + jitter, vder + jitter, c=PURPLE, s=55,
               edgecolors="k", linewidths=0.4)
    ax.set_xlabel("reference valence"); ax.set_ylabel("derived valence")
    return save(fig, "panel_2_valence.png")


# ===========================================================================
# PANEL 3 — Bonding criterion: Delta-thickness over all pairs
# ===========================================================================
def panel_3():
    fig = newfig()
    res = bonding["results"]
    syms = sorted({p.split("-")[0] for p in [r["pair"] for r in res]} |
                  {p.split("-")[1] for p in [r["pair"] for r in res]},
                  key=lambda s: [r["nu_a"] for r in res])  # placeholder
    # build vacancy lookup
    nu_of = {}
    for r in res:
        a, b = r["pair"].split("-")
        nu_of[a] = r["nu_a"]; nu_of[b] = r["nu_b"]
    elems = list(nu_of.keys())
    idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)

    # Delta-thickness matrix
    D = np.full((n, n), np.nan)
    for r in res:
        a, b = r["pair"].split("-")
        D[idx[a], idx[b]] = r["delta_thickness"]
        D[idx[b], idx[a]] = r["delta_thickness"]

    # (1) 3D surface of Delta over (nu_a, nu_b)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    g = np.arange(0, 9)
    NA, NB = np.meshgrid(g, g)
    shared = np.minimum(NA, NB)
    Z = (thickness(NA) + thickness(NB)) - (
        thickness(np.maximum(NA - shared, 0)) + thickness(np.maximum(NB - shared, 0)))
    ax.plot_surface(NA, NB, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel(r"$\nu_a$"); ax.set_ylabel(r"$\nu_b$"); ax.set_zlabel(r"$\Delta B$")
    ax.view_init(elev=24, azim=-58)

    # (2) heatmap of Delta-thickness over element pairs
    ax = fig.add_subplot(1, 4, 2)
    im = ax.imshow(D, cmap="magma", origin="lower")
    ax.set_xticks(range(n)); ax.set_xticklabels(elems, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(elems, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (3) Delta histogram, split bond / no-bond
    ax = fig.add_subplot(1, 4, 3)
    bond_d = [r["delta_thickness"] for r in res if r["predicts_bond"]]
    nobond_d = [r["delta_thickness"] for r in res if not r["predicts_bond"]]
    ax.hist(bond_d, bins=12, color=GREEN, alpha=0.8, label="bond", edgecolor="k", linewidth=0.3)
    ax.hist(nobond_d, bins=1, color=RED, alpha=0.8, label="no bond", edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="#94a3b8", lw=1)
    ax.set_xlabel(r"$\Delta B$"); ax.set_ylabel("count"); ax.legend(fontsize=7, frameon=False)

    # (4) noble vs open: max attainable Delta per element
    ax = fig.add_subplot(1, 4, 4)
    maxd = []
    for e in elems:
        row = [r["delta_thickness"] for r in res
               if e in r["pair"].split("-")]
        maxd.append(max(row) if row else 0.0)
    cols = [RED if nu_of[e] == 0 else BLUE for e in elems]
    ax.bar(range(n), maxd, color=cols, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(n)); ax.set_xticklabels(elems, rotation=90, fontsize=6)
    ax.set_ylabel(r"max $\Delta B$ over partners")
    return save(fig, "panel_3_bonding.png")


# ===========================================================================
# PANEL 4 — Stoichiometry from vacancy matching
# ===========================================================================
def panel_4():
    fig = newfig()
    res = stoich["results"]
    names = [r["molecule"] for r in res]
    nu_c = np.array([r["nu_central"] for r in res])
    nu_l = np.array([r["nu_ligand"] for r in res])
    nlig = np.array([r["formula_ref"][-1] for r in res])
    x = np.arange(len(names))

    # (1) 3D scatter (nu_central, nu_ligand, ligand count)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(nu_c, nu_l, nlig, c=nlig, cmap="plasma", s=70,
               edgecolors="k", linewidths=0.5)
    for xi, yi, zi, nm in zip(nu_c, nu_l, nlig, names):
        ax.text(xi, yi, zi, nm, fontsize=6)
    ax.set_xlabel(r"$\nu_{\rm central}$"); ax.set_ylabel(r"$\nu_{\rm ligand}$")
    ax.set_zlabel("ligands"); ax.view_init(elev=22, azim=-60)

    # (2) ligand count bars
    ax = fig.add_subplot(1, 4, 2)
    ax.bar(x, nlig, color=BLUE, edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("ligands per central")

    # (3) central vacancy vs predicted ligand count (should track)
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(nu_c, nlig, c=GREEN, s=60, edgecolors="k", linewidths=0.4)
    ax.plot([0, 4], [0, 4], color="#94a3b8", ls="--", lw=1)
    for xi, zi, nm in zip(nu_c, nlig, names):
        ax.annotate(nm, (xi, zi), textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel(r"$\nu_{\rm central}$"); ax.set_ylabel("ligand count")

    # (4) derived vs reference ligand count
    ax = fig.add_subplot(1, 4, 4)
    der = np.array([r["formula_derived"][-1] for r in res])
    ax.bar(x - 0.2, der, 0.4, color=BLUE, label="derived", edgecolor="k", linewidth=0.3)
    ax.bar(x + 0.2, nlig, 0.4, color=ORANGE, label="reference", edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("ligands"); ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_4_stoichiometry.png")


# ===========================================================================
# PANEL 5 — Bond geometry: maximal angular separation
# ===========================================================================
def panel_5():
    fig = newfig()
    res = geom["results"]

    # geometry vertex sets for k=2,3,4 (maximal separation on the sphere)
    def points_k(k):
        if k == 2:
            return np.array([[0, 0, 1], [0, 0, -1]])
        if k == 3:
            a = np.array([[math.cos(t), math.sin(t), 0]
                          for t in np.linspace(0, 2 * math.pi, 4)[:3]])
            return a
        if k == 4:
            return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
        return np.zeros((0, 3))

    # (1) 3D: tetrahedral 4-point config (CH4) with bonds from origin
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    P = points_k(4)
    for p in P:
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=BLUE, lw=2)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=GREEN, s=60, edgecolors="k", linewidths=0.5)
    ax.scatter([0], [0], [0], c=ORANGE, s=90, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=20, azim=30)

    # (2) ideal symmetric angle vs k
    ax = fig.add_subplot(1, 4, 2)
    ks = [2, 3, 4]
    ideal = [180.0, 120.0, math.degrees(math.acos(-1 / 3))]
    ax.plot(ks, ideal, "-o", color=BLUE, lw=2, markersize=8,
            markeredgecolor="k", markerfacecolor=GREEN)
    ax.set_xlabel("regions k"); ax.set_ylabel("ideal angle (deg)")
    ax.set_xticks(ks)

    # (3) observed vs ideal, lone-pair compression
    ax = fig.add_subplot(1, 4, 3)
    obs = [r["observed_deg"] for r in res]
    idl = [r["ideal_symmetric_deg"] for r in res]
    nm = [r["molecule"] for r in res]
    klone = [r["k_lone"] for r in res]
    sc = ax.scatter(idl, obs, c=klone, cmap="coolwarm", s=80,
                    edgecolors="k", linewidths=0.5)
    ax.plot([100, 185], [100, 185], color="#94a3b8", ls="--", lw=1)
    for a, b, t in zip(idl, obs, nm):
        ax.annotate(t, (a, b), textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel("ideal (deg)"); ax.set_ylabel("observed (deg)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="lone pairs")

    # (4) angle decreasing with lone-pair count (k=4 family)
    ax = fig.add_subplot(1, 4, 4)
    four = sorted([r for r in res if r["regions_k"] == 4], key=lambda r: r["k_lone"])
    lp = [r["k_lone"] for r in four]
    an = [r["observed_deg"] for r in four]
    labels = [r["molecule"] for r in four]
    ax.plot(lp, an, "-o", color=RED, lw=2, markersize=9,
            markeredgecolor="k", markerfacecolor=ORANGE)
    for a, b, t in zip(lp, an, labels):
        ax.annotate(t, (a, b), textcoords="offset points", xytext=(5, 4), fontsize=6)
    ax.set_xlabel("lone-pair regions"); ax.set_ylabel("observed angle (deg)")
    ax.set_xticks(lp)
    return save(fig, "panel_5_geometry.png")


# ===========================================================================
# PANEL 6 — d=3 axis exchange (SO(3) vs reflection)
# ===========================================================================
def panel_6():
    fig = newfig()
    S2 = np.array([[0, 1], [1, 0]], float)
    S3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)

    # (1) 3D: action of the 3D swap on a tetrahedron's vertices (proper rotation)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    Vr = (S3 @ V.T).T
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=BLUE, s=60, edgecolors="k",
               linewidths=0.5, label="before")
    ax.scatter(Vr[:, 0], Vr[:, 1], Vr[:, 2], c=RED, s=60, marker="^",
               edgecolors="k", linewidths=0.5, label="after swap")
    for a, b in zip(V, Vr):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#cbd5e1", lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=18, azim=35); ax.legend(fontsize=7, frameon=False)

    # (2) determinants: 2D swap (-1) vs 3D swap (+1)
    ax = fig.add_subplot(1, 4, 2)
    dets = [np.linalg.det(S2), np.linalg.det(S3)]
    cols = [RED, GREEN]
    ax.bar(["2D swap", "3D swap"], dets, color=cols, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.axhline(1, color=GREEN, ls=":", lw=1); ax.axhline(-1, color=RED, ls=":", lw=1)
    ax.set_ylabel("determinant"); ax.set_ylim(-1.4, 1.4)

    # (3) 2D swap as reflection: a unit-square mapped across y=x
    ax = fig.add_subplot(1, 4, 3)
    sq = np.array([[0, 0], [1, 0], [1, 0.4], [0, 0.4], [0, 0]], float)
    sqr = (S2 @ sq.T).T
    ax.plot(sq[:, 0], sq[:, 1], color=BLUE, lw=2, label="before")
    ax.plot(sqr[:, 0], sqr[:, 1], color=RED, lw=2, label="after (reflected)")
    ax.plot([-0.2, 1.2], [-0.2, 1.2], color="#94a3b8", ls="--", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7, frameon=False)

    # (4) eigenvalue spectrum (real parts) of the two swaps
    ax = fig.add_subplot(1, 4, 4)
    e2 = np.linalg.eigvals(S2); e3 = np.linalg.eigvals(S3)
    ax.scatter(np.real(e2), np.imag(e2), c=RED, s=80, label="2D",
               edgecolors="k", linewidths=0.5)
    ax.scatter(np.real(e3), np.imag(e3), c=GREEN, s=80, marker="^",
               label="3D", edgecolors="k", linewidths=0.5)
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#cbd5e1", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("Re"); ax.set_ylabel("Im")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_6_dimension.png")


# ===========================================================================
# PANEL 7 — Bond length: convex thickness well
# ===========================================================================
def panel_7():
    fig = newfig()
    D_e, d, r0 = 2.0, 1.0, 1.4

    def B(r):
        return D_e * ((1 - np.exp(-d * (r - r0)))**2 - 1)

    r = np.linspace(0.4, 6.0, 600)
    Bv = B(r)
    rstar = blen["r_star"]; Bstar = blen["B_star"]

    # (1) 3D: family of wells for varying D_e (well depth axis)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    Des = np.linspace(1.0, 3.0, 9)
    for De in Des:
        Bf = De * ((1 - np.exp(-d * (r - r0)))**2 - 1)
        ax.plot(r, np.full_like(r, De), Bf, color=plt.cm.viridis((De - 1) / 2), lw=1.5)
    ax.set_xlabel("r"); ax.set_ylabel(r"$D_e$"); ax.set_zlabel("B(r)")
    ax.view_init(elev=22, azim=-62)

    # (2) the well with minimiser marked
    ax = fig.add_subplot(1, 4, 2)
    ax.plot(r, Bv, color=BLUE, lw=2)
    ax.scatter([rstar], [Bstar], c=RED, s=70, zorder=5, edgecolors="k", linewidths=0.5)
    ax.axvline(rstar, color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel("r"); ax.set_ylabel("B(r)")

    # (3) the two competing terms: wall (decreasing share) + repulsion
    ax = fig.add_subplot(1, 4, 3)
    repulsive = D_e * (np.exp(-2 * d * (r - r0)))      # blows up as r->0
    attractive = -2 * D_e * np.exp(-d * (r - r0))      # sharing pull
    ax.plot(r, repulsive, color=RED, lw=2, label="wall (floor)")
    ax.plot(r, attractive, color=GREEN, lw=2, label="sharing")
    ax.plot(r, Bv + D_e, color=BLUE, lw=2, ls="--", label="sum (shifted)")
    ax.set_ylim(-5, 8)
    ax.set_xlabel("r"); ax.set_ylabel("contribution"); ax.legend(fontsize=7, frameon=False)

    # (4) curvature (second derivative) positive at minimum
    ax = fig.add_subplot(1, 4, 4)
    d2 = np.gradient(np.gradient(Bv, r), r)
    ax.plot(r, d2, color=PURPLE, lw=2)
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.scatter([rstar], [np.interp(rstar, r, d2)], c=RED, s=70, zorder=5,
               edgecolors="k", linewidths=0.5)
    ax.set_xlabel("r"); ax.set_ylabel(r"$B''(r)$")
    return save(fig, "panel_7_bondlength.png")


# ===========================================================================
# PANEL 8 — Existence of structure: open vs closed worlds
# ===========================================================================
def panel_8():
    fig = newfig()
    vr = shell["vacancy"]["results"]
    syms = [r["sym"] for r in vr]
    nu = np.array([r["vacancy"] for r in vr])
    noble = np.array([r["is_noble_ref"] for r in vr])

    # (1) 3D: thickness reduction landscape over (nu_a, nu_b) -- shareable content
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    g = np.arange(0, 9)
    A, Bb = np.meshgrid(g, g)
    shared = np.minimum(A, Bb)
    ax.plot_surface(A, Bb, shared.astype(float), cmap="cividis",
                    edgecolor="none", alpha=0.95)
    ax.set_xlabel(r"$\nu_a$"); ax.set_ylabel(r"$\nu_b$")
    ax.set_zlabel("shareable")
    ax.view_init(elev=24, azim=-58)

    # (2) per-element thickness above the closed-shell minimum
    ax = fig.add_subplot(1, 4, 2)
    excess = thickness(nu) - thickness(0)
    cols = [RED if n == 0 else BLUE for n in nu]
    ax.bar(range(len(syms)), excess, color=cols, edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, rotation=90)
    ax.set_ylabel(r"excess thickness $\kappa\phi(\nu)$")

    # (3) fraction reactive vs inert as a "world" composition pie-free bar
    ax = fig.add_subplot(1, 4, 3)
    n_open = int(np.sum(nu > 0)); n_closed = int(np.sum(nu == 0))
    ax.bar(["open\n(reactive)", "closed\n(inert)"], [n_open, n_closed],
           color=[BLUE, RED], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("element count")

    # (4) total world thickness vs fraction of atoms allowed to bond
    ax = fig.add_subplot(1, 4, 4)
    fracs = np.linspace(0, 1, 50)
    # separated total thickness of all open atoms
    sep_total = np.sum(thickness(nu))
    # as fraction f of open atoms pair up (share min vacancy ~1 each), thickness drops
    open_excess = np.sum(thickness(nu[nu > 0]) - thickness(0))
    world_thick = sep_total - fracs * open_excess * 0.5
    ax.plot(fracs, world_thick, color=GREEN, lw=2)
    ax.fill_between(fracs, world_thick, sep_total, color="#dcfce7", alpha=0.6)
    ax.set_xlabel("fraction bonded"); ax.set_ylabel("total world thickness")
    return save(fig, "panel_8_existence.png")


if __name__ == "__main__":
    gens = [
        ("Panel 1: Capacity & vacancy", panel_1),
        ("Panel 2: Valence", panel_2),
        ("Panel 3: Bonding criterion", panel_3),
        ("Panel 4: Stoichiometry", panel_4),
        ("Panel 5: Geometry", panel_5),
        ("Panel 6: Dimensionality (d=3)", panel_6),
        ("Panel 7: Bond length well", panel_7),
        ("Panel 8: Existence of structure", panel_8),
    ]
    for label, fn in gens:
        p = fn()
        print(f"  [OK] {label} -> {os.path.relpath(p, BASE)}")
    print(f"\nAll 8 panels saved to {os.path.relpath(FIGDIR, BASE)}/")
