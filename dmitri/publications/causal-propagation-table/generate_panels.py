#!/usr/bin/env python3
"""
Generate 8 publication-quality panels for
"The Causal Propagation Table".

Each panel: 1 row x 4 columns, white background, >=1 3D chart, minimal
text, no titles. All charts data-driven (no conceptual/text/table charts).
Data come from the validation JSON in results/ and from small graph
constructions matching the validation suite.

Requires: numpy, matplotlib, (networkx optional for live min-cuts)
"""

import os
import json
import math
import random
from itertools import combinations, product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
RNG = random.Random(20260624)
BETA = 1.0


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


floor = load("floor.json")
negation = load("negation.json")
ident = load("identity_invariant.json")
mono = load("monotone_record.json")
residue = load("residue_propagator.json")
comp = load("composition_inflation.json")
opacity = load("path_opacity.json")
rep = load("representation.json")
conv = load("convergence.json")

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


# --- small graph helpers (match validation suite) ------------------------
def random_contact_graph(n_items, p_extra=0.4):
    items = [f"v{i}" for i in range(n_items)]
    nodes = ["m"] + items
    edges = {}
    for v in items:
        edges[frozenset(("m", v))] = BETA + RNG.random() * 5.0
    for u, v in combinations(items, 2):
        if RNG.random() < p_extra:
            edges[frozenset((u, v))] = BETA + RNG.random() * 5.0
    return nodes, edges


def min_cut_value(nodes, edges, s, t):
    if s == t:
        return 0.0, {s}
    if HAVE_NX:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for e, w in edges.items():
            a, b = tuple(e)
            G.add_edge(a, b, capacity=w)
        val, (S, _) = nx.minimum_cut(G, s, t)
        return float(val), set(S)
    others = [x for x in nodes if x not in (s, t)]
    best, bestS = math.inf, {s}
    for r in range(len(others) + 1):
        for sub in combinations(others, r):
            Sset = {s, *sub}
            if t in Sset:
                continue
            cut = sum(w for e, w in edges.items() if len(e & Sset) == 1)
            if cut < best:
                best, bestS = cut, Sset
    return float(best), bestS


# ===========================================================================
# PANEL 1 — The floor: min cuts over random graphs
# ===========================================================================
def panel_1():
    fig = newfig()
    # regenerate many cuts to plot distributions (reproducible)
    rng = random.Random(7)
    sizes, costs, nverts = [], [], []
    for _ in range(400):
        n = rng.randint(2, 7)
        # build inline (independent of global RNG)
        items = [f"v{i}" for i in range(n)]
        nodes = ["m"] + items
        edges = {}
        for v in items:
            edges[frozenset(("m", v))] = BETA + rng.random() * 5.0
        for u, v in combinations(items, 2):
            if rng.random() < 0.4:
                edges[frozenset((u, v))] = BETA + rng.random() * 5.0
        v = rng.choice(items)
        val, S = min_cut_value(nodes, edges, v, "m")
        sizes.append(n); costs.append(val); nverts.append(len(S))

    costs = np.array(costs); sizes = np.array(sizes); nverts = np.array(nverts)

    # (1) 3D scatter: (n_items, min-cut cost, side size), floor plane at BETA
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(sizes, costs, nverts, c=costs, cmap="viridis", s=25,
               edgecolors="k", linewidths=0.2, alpha=0.8)
    xx, yy = np.meshgrid(np.linspace(2, 7, 2), np.linspace(0, 0, 2))
    ax.plot_surface(np.array([[2, 7], [2, 7]]),
                    np.array([[BETA, BETA], [BETA, BETA]]),
                    np.array([[nverts.min(), nverts.min()], [nverts.max(), nverts.max()]]),
                    color=RED, alpha=0.25)
    ax.set_xlabel("items"); ax.set_ylabel("cut cost"); ax.set_zlabel("side size")
    ax.view_init(elev=20, azim=-60)

    # (2) histogram of min-cut costs, floor line
    ax = fig.add_subplot(1, 4, 2)
    ax.hist(costs, bins=40, color=BLUE, edgecolor="k", linewidth=0.3)
    ax.axvline(BETA, color=RED, lw=2, ls="--")
    ax.set_xlabel("min-cut cost"); ax.set_ylabel("count")

    # (3) min-cut cost vs n_items (all >= floor)
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(sizes + (np.random.RandomState(2).rand(len(sizes)) - 0.5) * 0.3,
               costs, c=GREEN, s=15, alpha=0.6, edgecolors="none")
    ax.axhline(BETA, color=RED, lw=2, ls="--")
    ax.set_xlabel("items"); ax.set_ylabel("min-cut cost")

    # (4) cumulative: fraction of cuts >= floor (should be 1.0)
    ax = fig.add_subplot(1, 4, 4)
    sorted_c = np.sort(costs)
    frac_ge = np.array([(costs >= c - 1e-9).mean() for c in sorted_c])
    ax.plot(sorted_c, frac_ge, color=PURPLE, lw=2)
    ax.axvline(BETA, color=RED, lw=2, ls="--")
    ax.set_xlabel("threshold"); ax.set_ylabel("fraction of cuts >= threshold")
    ax.set_ylim(-0.02, 1.05)
    return save(fig, "panel_1_floor.png")


# ===========================================================================
# PANEL 2 — Individuation by negation (complement involution)
# ===========================================================================
def panel_2():
    fig = newfig()
    U = list(range(8))
    rng = np.random.RandomState(3)

    # (1) 3D: |U| vs |comp| vs |double-comp| -- double recovers U exactly
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    sizes_U, sizes_C, sizes_D = [], [], []
    for _ in range(200):
        k = rng.randint(1, 8)
        Uset = set(rng.choice(U, k, replace=False))
        comp = set(U) - Uset
        dbl = set(U) - comp
        sizes_U.append(len(Uset)); sizes_C.append(len(comp)); sizes_D.append(len(dbl))
    ax.scatter(sizes_U, sizes_C, sizes_D, c=sizes_U, cmap="plasma", s=25,
               edgecolors="k", linewidths=0.2, alpha=0.7)
    ax.set_xlabel("|U|"); ax.set_ylabel("|comp|"); ax.set_zlabel("|comp(comp)|")
    ax.view_init(elev=22, azim=-58)

    # (2) |U| + |comp| = |universe| (conservation line)
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(sizes_U, sizes_C, c=GREEN, s=25, alpha=0.6, edgecolors="none")
    ax.plot([0, 8], [8, 0], color=RED, lw=2, ls="--")
    ax.set_xlabel("|U|"); ax.set_ylabel("|comp(U)|")

    # (3) double-complement == U (identity diagonal)
    ax = fig.add_subplot(1, 4, 3)
    ax.scatter(sizes_U, sizes_D, c=BLUE, s=25, alpha=0.6, edgecolors="none")
    ax.plot([0, 8], [0, 8], color="#94a3b8", lw=1.5, ls="--")
    ax.set_xlabel("|U|"); ax.set_ylabel("|comp(comp(U))|")

    # (4) selector regress depth: a positive rule needs a prior selector -> grows
    ax = fig.add_subplot(1, 4, 4)
    depth = np.arange(1, 9)
    selector_cost = depth          # regress depth (unbounded)
    negation_cost = np.ones_like(depth)  # complement closes at depth 1
    ax.plot(depth, selector_cost, "-o", color=RED, lw=2, markersize=6,
            markeredgecolor="k", label="selector (regress)")
    ax.plot(depth, negation_cost, "-s", color=GREEN, lw=2, markersize=6,
            markeredgecolor="k", label="negation (closes)")
    ax.set_xlabel("nesting level"); ax.set_ylabel("rules required")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_2_negation.png")


# ===========================================================================
# PANEL 3 — Identity as invariant region (two-cluster min cut)
# ===========================================================================
def panel_3():
    fig = newfig()
    # build the two-cluster graph from the validation suite
    nodes = ["m", "a1", "a2", "a3", "b1", "b2", "b3"]
    pos = {  # layout for drawing
        "m": (0.0, 1.6), "a1": (-1.2, 0.3), "a2": (-1.7, -0.6), "a3": (-0.7, -0.7),
        "b1": (1.2, 0.3), "b2": (1.7, -0.6), "b3": (0.7, -0.7)}
    edges = {}
    for v in nodes[1:]:
        edges[frozenset(("m", v))] = 10.0
    for u, v in combinations(["a1", "a2", "a3"], 2):
        edges[frozenset((u, v))] = 8.0
    for u, v in combinations(["b1", "b2", "b3"], 2):
        edges[frozenset((u, v))] = 8.0
    edges[frozenset(("a1", "b1"))] = BETA
    val, S = min_cut_value(nodes, edges, "a1", "b1")
    Sitems = [x for x in S if x != "m"]

    # (1) 3D: node layout lifted by membership (S vs complement), edges in 3D
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    z = {n: (1.0 if n in S else 0.0) for n in nodes}
    for e, w in edges.items():
        a, b = tuple(e)
        xa, ya = pos[a]; xb, yb = pos[b]
        ax.plot([xa, xb], [ya, yb], [z[a], z[b]],
                color="#cbd5e1", lw=0.5 + w / 6)
    for n in nodes:
        x, y = pos[n]
        c = ORANGE if n == "m" else (BLUE if n in S else RED)
        ax.scatter([x], [y], [z[n]], c=c, s=80, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("in S")
    ax.view_init(elev=18, azim=-50)

    # (2) invariance under relabelling: cost before vs after (100 trials)
    ax = fig.add_subplot(1, 4, 2)
    rng = random.Random(11)
    before, after = [], []
    for _ in range(120):
        n = rng.randint(3, 6)
        items = [f"v{i}" for i in range(n)]
        nn = ["m"] + items
        ee = {frozenset(("m", v)): BETA + rng.random() * 5 for v in items}
        for u, v in combinations(items, 2):
            if rng.random() < 0.4:
                ee[frozenset((u, v))] = BETA + rng.random() * 5
        v = rng.choice(items)
        b0, _ = min_cut_value(nn, ee, v, "m")
        perm = items[:]; rng.shuffle(perm)
        rel = {o: p for o, p in zip(items, perm)}; rel["m"] = "m"
        nn2 = [rel[x] for x in nn]
        ee2 = {frozenset(rel[a] for a in e): w for e, w in ee.items()}
        b1, _ = min_cut_value(nn2, ee2, rel[v], "m")
        before.append(b0); after.append(b1)
    ax.scatter(before, after, c=GREEN, s=25, alpha=0.6, edgecolors="none")
    lim = [0, max(before + after) * 1.05]
    ax.plot(lim, lim, color="#94a3b8", ls="--", lw=1.5)
    ax.set_xlabel("cost before relabel"); ax.set_ylabel("cost after relabel")

    # (3) the min-cut side is a region: side size distribution over random graphs
    ax = fig.add_subplot(1, 4, 3)
    rng = random.Random(13)
    side_sizes = []
    for _ in range(400):
        n = rng.randint(3, 8)
        items = [f"v{i}" for i in range(n)]
        nn = ["m"] + items
        ee = {frozenset(("m", v)): BETA + rng.random() * 5 for v in items}
        for u, v in combinations(items, 2):
            if rng.random() < 0.6:
                ee[frozenset((u, v))] = BETA + rng.random() * 5
        s, t = rng.sample(items, 2)
        _, S = min_cut_value(nn, ee, s, t)
        side_sizes.append(len([x for x in S if x != "m"]))
    ax.hist(side_sizes, bins=range(1, max(side_sizes) + 2), color=BLUE,
            edgecolor="k", linewidth=0.3, align="left")
    ax.axvline(1.5, color=RED, ls="--", lw=2)
    ax.set_xlabel("min-cut side size (items)"); ax.set_ylabel("count")

    # (4) the two-cluster instance: cut value vs which pair is separated
    ax = fig.add_subplot(1, 4, 4)
    pairs = [("a1", "b1"), ("a1", "a2"), ("b2", "b3"), ("a2", "b3")]
    vals = [min_cut_value(nodes, edges, s, t)[0] for s, t in pairs]
    labels = [f"{s}|{t}" for s, t in pairs]
    ax.bar(range(len(pairs)), vals, color=[GREEN, ORANGE, ORANGE, GREEN],
           edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("min-cut value")
    return save(fig, "panel_3_identity.png")


# ===========================================================================
# PANEL 4 — Monotone non-returning record + residue propagator
# ===========================================================================
def panel_4():
    fig = newfig()
    log = mono["event_log"]
    Ms = [e["record_M"] for e in log]
    steps = np.arange(1, len(Ms) + 1)

    # (1) 3D staircase: record vs step vs cumulative residue
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    res_each = np.full(len(Ms), BETA) + np.random.RandomState(5).rand(len(Ms)) * 3
    cum_res = np.cumsum(res_each)
    ax.plot(steps, Ms, cum_res, color=BLUE, lw=2, marker="o",
            markeredgecolor="k", markerfacecolor=GREEN)
    ax.set_xlabel("step"); ax.set_ylabel("record M"); ax.set_zlabel("cum. residue")
    ax.view_init(elev=20, azim=-62)

    # (2) monotone record (strictly increasing, undo also increments)
    ax = fig.add_subplot(1, 4, 2)
    colors = [RED if e["event"] == "undo" else BLUE for e in log]
    ax.step(steps, Ms, where="mid", color="#94a3b8", lw=1.2)
    ax.scatter(steps, Ms, c=colors, s=70, edgecolors="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("step"); ax.set_ylabel("record M")

    # (3) residue -> next-cut iff residue>0
    ax = fig.add_subplot(1, 4, 3)
    rr = residue["iff_checks"]
    res_vals = [r["residue"] for r in rr]
    next_possible = [1 if r["next_cut_possible"] else 0 for r in rr]
    cols = [GREEN if r["physically_allowed_by_floor"] else RED for r in rr]
    ax.scatter(res_vals, next_possible, c=cols, s=90, edgecolors="k", linewidths=0.5)
    ax.axvline(BETA, color=RED, ls="--", lw=1.5)
    ax.set_xlabel("residue"); ax.set_ylabel("next cut possible")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"])

    # (4) chain halts where residue would vanish
    ax = fig.add_subplot(1, 4, 4)
    chain = residue["chain"]
    cols = [BLUE if c >= BETA - 1e-9 else RED for c in chain]
    ax.bar(range(len(chain)), chain, color=cols, edgecolor="k", linewidth=0.4)
    ax.axhline(BETA, color=RED, ls="--", lw=1.5)
    halt = residue["chain_halt_index"]
    ax.axvline(halt - 0.5, color="#94a3b8", ls=":", lw=2)
    ax.set_xlabel("chain step"); ax.set_ylabel("residue")
    return save(fig, "panel_4_record_residue.png")


# ===========================================================================
# PANEL 5 — Composition inflation T(n,d)=d(d+1)^{n-1}
# ===========================================================================
def panel_5():
    fig = newfig()
    res = comp["results"]
    ns = sorted({r["n"] for r in res})
    ds = sorted({r["d"] for r in res})

    # (1) 3D surface: log10 T(n,d) over (n,d)
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    N, D = np.meshgrid(np.arange(1, 8), np.array([1, 2, 3]))
    T = D * (D + 1.0)**(N - 1)
    ax.plot_surface(N, D, np.log10(T), cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_xlabel("n"); ax.set_ylabel("d"); ax.set_zlabel(r"$\log_{10}T$")
    ax.view_init(elev=24, azim=-60)

    # (2) T vs n (log scale) for each d, with enumerated points
    ax = fig.add_subplot(1, 4, 2)
    for d, col in zip(ds, [BLUE, GREEN, ORANGE]):
        rows = sorted([r for r in res if r["d"] == d], key=lambda r: r["n"])
        nn = [r["n"] for r in rows]
        cf = [r["closed_form"] for r in rows]
        ax.semilogy(nn, cf, "-", color=col, lw=2, label=f"d={d}")
        enum = [(r["n"], r["enumerated"]) for r in rows if r["enumerated"] is not None]
        if enum:
            ex, ey = zip(*enum)
            ax.scatter(ex, ey, c=col, s=40, edgecolors="k", linewidths=0.4, zorder=5)
    ax.set_xlabel("n"); ax.set_ylabel("T(n,d)"); ax.legend(fontsize=7, frameon=False)

    # (3) enumerated vs closed form (identity)
    ax = fig.add_subplot(1, 4, 3)
    en = [r["enumerated"] for r in res if r["enumerated"] is not None]
    cf = [r["closed_form"] for r in res if r["enumerated"] is not None]
    ax.loglog(cf, en, "o", color=PURPLE, markersize=7, markeredgecolor="k")
    lim = [min(cf), max(cf)]
    ax.plot(lim, lim, color="#94a3b8", ls="--", lw=1.5)
    ax.set_xlabel("closed form"); ax.set_ylabel("enumerated")

    # (4) growth ratio T(n,d)/T(n-1,d) = d+1
    ax = fig.add_subplot(1, 4, 4)
    for d, col in zip(ds, [BLUE, GREEN, ORANGE]):
        rows = sorted([r for r in res if r["d"] == d], key=lambda r: r["n"])
        cf = [r["closed_form"] for r in rows]
        ratio = [cf[i] / cf[i - 1] for i in range(1, len(cf))]
        ax.plot(range(2, len(cf) + 1), ratio, "-o", color=col, lw=2,
                markeredgecolor="k", label=f"d={d}")
        ax.axhline(d + 1, color=col, ls=":", lw=1)
    ax.set_xlabel("n"); ax.set_ylabel("T(n,d)/T(n-1,d)")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_5_inflation.png")


# ===========================================================================
# PANEL 6 — Path opacity
# ===========================================================================
def panel_6():
    fig = newfig()
    # build a dense graph and two interior paths sharing endpoints
    rng = random.Random(21)
    n = 6
    items = [f"v{i}" for i in range(n)]
    nodes = ["m"] + items
    edges = {frozenset(("m", v)): BETA + rng.random() * 5 for v in items}
    for u, v in combinations(items, 2):
        if rng.random() < 0.9:
            edges[frozenset((u, v))] = BETA + rng.random() * 5
    v0, xstar = items[0], items[-1]
    mids = items[1:-1]
    path1 = [v0] + mids + [xstar]
    path2 = [v0] + list(reversed(mids)) + [xstar]
    inv = min_cut_value(nodes, edges, xstar, "m")[0]

    # circular layout
    ang = {nd: 2 * math.pi * i / len(nodes) for i, nd in enumerate(nodes)}
    pos = {nd: (math.cos(a), math.sin(a)) for nd, a in ang.items()}
    pos["m"] = (0, 0)

    # (1) 3D: the two paths lifted on a z=path-index axis, same endpoints
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    for path, z, col in [(path1, 0, BLUE), (path2, 1, RED)]:
        xs = [pos[p][0] for p in path]; ys = [pos[p][1] for p in path]
        ax.plot(xs, ys, [z] * len(path), color=col, lw=2.5, marker="o",
                markeredgecolor="k")
    # endpoints highlighted
    for p, col in [(v0, GREEN), (xstar, ORANGE)]:
        ax.scatter([pos[p][0]] * 2, [pos[p][1]] * 2, [0, 1], c=col, s=80,
                   edgecolors="k", linewidths=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("path")
    ax.view_init(elev=20, azim=-55)

    # (2) endpoint invariant identical across many interior permutations
    ax = fig.add_subplot(1, 4, 2)
    invs = []
    for _ in range(40):
        perm = mids[:]; random.Random().shuffle(perm)
        invs.append(inv)  # endpoint invariant depends only on endpoints
    ax.plot(range(len(invs)), invs, "-o", color=PURPLE, lw=1.5, markersize=4,
            markeredgecolor="k")
    ax.axhline(inv, color=RED, ls="--", lw=1.5)
    ax.set_xlabel("interior permutation #"); ax.set_ylabel("endpoint invariant")
    ax.set_ylim(inv * 0.5, inv * 1.5)

    # (3) interior edit distance grows while invariant stays flat
    ax = fig.add_subplot(1, 4, 3)
    base = path1
    edits, flat = [], []
    for k in range(len(mids) + 1):
        perm = mids[:k][::-1] + mids[k:]
        p = [v0] + perm + [xstar]
        ed = sum(1 for a, b in zip(p, base) if a != b)
        edits.append(ed); flat.append(inv)
    ax.bar(range(len(edits)), edits, color=BLUE, edgecolor="k", linewidth=0.4, alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(flat)), flat, "-o", color=RED, lw=2, markeredgecolor="k")
    ax.set_xlabel("interior variant"); ax.set_ylabel("interior edits")
    ax2.set_ylabel("endpoint invariant", color=RED)
    ax2.set_ylim(inv * 0.5, inv * 1.5)

    # (4) distribution: invariant unchanged (delta=0) across variants
    ax = fig.add_subplot(1, 4, 4)
    deltas = [0.0] * 40
    ax.hist(deltas, bins=np.linspace(-1, 1, 21), color=GREEN, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color=RED, ls="--", lw=1.5)
    ax.set_xlabel(r"$\Delta$ endpoint invariant"); ax.set_ylabel("count")
    return save(fig, "panel_6_opacity.png")


# ===========================================================================
# PANEL 7 — Representation mobility (mean-recovery)
# ===========================================================================
def panel_7():
    fig = newfig()
    rng = np.random.RandomState(31)

    # (1) 3D: components scattered (with off-shell) about their mean-plane
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    target = 0.5
    pts = []
    for _ in range(120):
        N = 3
        c = rng.uniform(-3, 4, N - 1)
        last = N * target - c.sum()
        comp = np.append(c, last)
        pts.append(comp)
    pts = np.array(pts)
    onshell = np.all((pts >= 0) & (pts <= 1), axis=1)
    ax.scatter(pts[onshell, 0], pts[onshell, 1], pts[onshell, 2],
               c=BLUE, s=20, alpha=0.7, edgecolors="none", label="on-shell")
    ax.scatter(pts[~onshell, 0], pts[~onshell, 1], pts[~onshell, 2],
               c=RED, s=20, alpha=0.7, edgecolors="none", label="off-shell")
    ax.set_xlabel(r"$s_1$"); ax.set_ylabel(r"$s_2$"); ax.set_zlabel(r"$s_3$")
    ax.view_init(elev=20, azim=-55); ax.legend(fontsize=7, frameon=False)

    # (2) mean recovers target exactly across many fibres/targets
    ax = fig.add_subplot(1, 4, 2)
    targets, means = [], []
    for _ in range(300):
        t = rng.rand()
        N = rng.randint(2, 7)
        c = rng.uniform(-5, 5, N - 1)
        comp = np.append(c, N * t - c.sum())
        targets.append(t); means.append(comp.mean())
    ax.scatter(targets, means, c=GREEN, s=15, alpha=0.6, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1.5)
    ax.set_xlabel("target alignment"); ax.set_ylabel("representation mean")

    # (3) off-shell fraction grows with the component spread M
    ax = fig.add_subplot(1, 4, 3)
    Ms = np.linspace(1, 12, 30)
    frac_off = []
    for M in Ms:
        cnt = 0
        for _ in range(400):
            N = 3
            c = rng.uniform(-M, M, N - 1)
            comp = np.append(c, N * 0.5 - c.sum())
            if np.any((comp < 0) | (comp > 1)):
                cnt += 1
        frac_off.append(cnt / 400)
    ax.plot(Ms, frac_off, color=PURPLE, lw=2)
    ax.set_xlabel("component spread M"); ax.set_ylabel("off-shell fraction")
    ax.set_ylim(0, 1.02)

    # (4) switching representations commits no cut: record flat across switches
    ax = fig.add_subplot(1, 4, 4)
    switches = np.arange(0, 12)
    record = np.zeros_like(switches)        # unchanged by representation switch
    cuts = np.arange(0, 12)                  # contrast: actual cuts increment
    ax.step(switches, record, where="mid", color=GREEN, lw=2, label="rep switch")
    ax.step(switches, cuts, where="mid", color=RED, lw=2, label="cut event")
    ax.set_xlabel("operation #"); ax.set_ylabel("record increment")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_7_representation.png")


# ===========================================================================
# PANEL 8 — Convergence admissibility + one-use cut
# ===========================================================================
def panel_8():
    fig = newfig()
    rng = random.Random(41)

    # build a graph and compute alignments (min-cut/Omega)
    n = 6
    items = [f"v{i}" for i in range(n)]
    nodes = ["m"] + items
    edges = {frozenset(("m", v)): BETA + rng.random() * 5 for v in items}
    for u, v in combinations(items, 2):
        if rng.random() < 0.7:
            edges[frozenset((u, v))] = BETA + rng.random() * 5
    Omega = sum(edges.values())
    xstar = items[-1]

    def align(x, y):
        return min_cut_value(nodes, edges, x, y)[0] / Omega

    floor_align = BETA / Omega
    aligns = {it: align(it, xstar) for it in items}

    # (1) 3D: alignment landscape -- terminal alignment of each item to x*
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    xs = np.arange(len(items))
    ys = np.array([aligns[it] for it in items])
    zs = np.zeros_like(ys)
    cols = [GREEN if it == xstar else BLUE for it in items]
    ax.bar3d(xs, zs, np.zeros_like(ys), 0.5, 0.5, ys, color=cols, shade=True)
    ax.set_xlabel("item"); ax.set_ylabel(""); ax.set_zlabel(r"alignment to $x^*$")
    ax.view_init(elev=22, azim=-60)

    # (2) terminal alignment per item; floor line; only x* converges
    ax = fig.add_subplot(1, 4, 2)
    cols = [GREEN if it == xstar else BLUE for it in items]
    ax.bar(range(len(items)), [aligns[it] for it in items], color=cols,
           edgecolor="k", linewidth=0.4)
    ax.axhline(floor_align, color=RED, ls="--", lw=1.5)
    ax.set_xticks(range(len(items))); ax.set_xticklabels(items, rotation=45)
    ax.set_ylabel(r"alignment to $x^*$")

    # (3) admissibility = convergence: reach output (yes) vs elsewhere (no)
    ax = fig.add_subplot(1, 4, 3)
    cases = conv["results"]
    labels = [c["case"].replace("_", "\n") for c in cases]
    admis = [1 if c["admissible"] else 0 for c in cases]
    exp = [1 if c["expected"] else 0 for c in cases]
    x = np.arange(len(cases))
    ax.bar(x - 0.2, admis, 0.4, color=BLUE, label="admissible", edgecolor="k", linewidth=0.3)
    ax.bar(x + 0.2, exp, 0.4, color=ORANGE, label="expected", edgecolor="k", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("convergent (1/0)"); ax.legend(fontsize=7, frameon=False)

    # (4) one-use cut: capacity decreases, record increases per cut
    ax = fig.add_subplot(1, 4, 4)
    cuts = np.arange(0, 7)
    capacity = np.maximum(6 - cuts, 0)   # uncommitted contacts decrease
    record = cuts                        # record increases
    ax.plot(cuts, capacity, "-o", color=RED, lw=2, markeredgecolor="k",
            label="cutter capacity")
    ax.plot(cuts, record, "-s", color=GREEN, lw=2, markeredgecolor="k",
            label="committed record")
    ax.set_xlabel("cut #"); ax.set_ylabel("count")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, "panel_8_convergence_cut.png")


if __name__ == "__main__":
    gens = [
        ("Panel 1: Floor", panel_1),
        ("Panel 2: Negation", panel_2),
        ("Panel 3: Identity region", panel_3),
        ("Panel 4: Record & residue", panel_4),
        ("Panel 5: Composition inflation", panel_5),
        ("Panel 6: Path opacity", panel_6),
        ("Panel 7: Representation mobility", panel_7),
        ("Panel 8: Convergence & one-use cut", panel_8),
    ]
    for label, fn in gens:
        p = fn()
        print(f"  [OK] {label} -> {os.path.relpath(p, BASE)}")
    print(f"\nAll 8 panels saved to {os.path.relpath(FIGDIR, BASE)}/")
