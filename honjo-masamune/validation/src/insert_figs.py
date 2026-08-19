"""Insert the generated panels into the two manuscripts."""

from __future__ import annotations

import os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "..", "docs")

FIG = r"""
\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{%s}
\caption{%s}
\label{%s}
\end{figure*}
"""

RANK = [
    ("panel_01_base_key.png", "fig:basekey",
     r"""\textbf{The base cut key occupies a small, unevenly populated discrete
space.} (\textbf{A})~Occupancy of the $(\sigma,\text{burial depth},\text{heavy
degree})$ cells over the $237$ heavy atoms of the drug-like corpus, bar height
the number of atoms and colour tracking it; twelve cells are occupied and one
holds $93$ atoms. (\textbf{B})~The same atoms decomposed by burial depth within
each of the two realised $\sigma$ levels: depth separates atoms that $\sigma$
alone does not, which is why the key is a pair. (\textbf{C})~The
$(\sigma,\text{depth})$ plane with marker area and colour proportional to
occupancy, showing the seven distinct base keys and their unequal populations.
(\textbf{D})~Distinct base keys divided by heavy atoms, one bar per molecule
sorted ascending; the dashed line is the corpus mean $0.508$ and the dotted line
the value $1.0$ a full ranking would require. The base key separates about half
the atoms of a typical structure and is not a ranking on its own."""),

    ("panel_02_refinement.png", "fig:refine",
     r"""\textbf{Refinement saturates quickly and lifts the class count without
reaching one class per atom.} (\textbf{A})~Classes divided by atoms against
refinement round, one line per molecule, symmetric corpus in red and drug-like
in blue; every trajectory is non-decreasing and flat after its stabilisation
round, and the symmetric molecules saturate at markedly lower values.
(\textbf{B})~The same trajectories in three dimensions against heavy-atom count:
the saturation level rises with size for the drug-like set while the symmetric
set stays low regardless of size. (\textbf{C})~Rounds to stabilisation for the
two corpora; the median over both is $2$ and the maximum is $4$.
(\textbf{D})~Per molecule, the base-key ratio (orange) and the stable ratio
(teal) joined by a grey segment, sorted by the stable value: the mean rises from
$0.508$ to $0.827$, so refinement does substantial work and still leaves atoms
unseparated."""),

    ("panel_03_orbits.png", "fig:orbits",
     r"""\textbf{Refinement recovers the automorphism orbit partition exactly.}
(\textbf{A})~Cut classes against independently determined reference orbit counts
for the eighteen symmetric molecules, coloured by heavy-atom count and jittered
slightly so coincident points remain visible; all lie on the identity diagonal
($18/18$). (\textbf{B})~Residual classes minus orbits per molecule for the cut
refinement (green, identically zero) against the index tie-breaking control
(red), which exceeds the orbit count by up to ten classes on the same molecules.
(\textbf{C})~Atoms, orbits and classes in three dimensions with the identity line
drawn; points coloured by rounds to stabilisation lie in the orbit plane and well
below the atom-count diagonal. (\textbf{D})~Classes divided by atoms per
molecule, sorted; values well below the dotted line at $1.0$ are molecules whose
own symmetry the refinement declines to break."""),

    ("panel_04_control.png", "fig:control",
     r"""\textbf{The negative control: breaking ties by atom index destroys the
orbit structure.} (\textbf{A})~Classes produced against reference orbits for the
cut refinement (green circles, on the diagonal) and for the index tie-breaking
control (red triangles, above it). (\textbf{B})~Excess classes over the reference
per molecule for both procedures; the control over-separates on $16$ of $18$
molecules and the refinement on none. (\textbf{C})~Classes produced against
heavy-atom count: the control tracks the atom-count diagonal, because an ordering
must assign a distinct rank to every atom, while the refinement does not.
(\textbf{D})~The two procedures in the (atoms, orbits, classes) space with a
vertical segment joining each molecule's two outcomes; segment length is the
control's over-separation."""),

    ("panel_05_classes.png", "fig:classes",
     r"""\textbf{Class structure of the stable partition.} (\textbf{A})~Atoms per
class index for every molecule of the drug-like corpus, one row of bars per
molecule ordered by size; most classes are singletons and the tall bars are the
symmetric positions the refinement correctly declines to split.
(\textbf{B})~Distribution of class sizes over the corpus: singletons dominate,
with a smaller population of size-two classes and a few larger.
(\textbf{C})~Stable classes against heavy-atom count, coloured by their ratio,
with the identity diagonal drawn; molecules below the diagonal retain unbroken
symmetry. (\textbf{D})~Burial depth against heavy degree for all $237$ atoms,
jittered for visibility and coloured by $\sigma$; the two $\sigma$ levels
separate cleanly in this plane, showing what the second component of the key
contributes."""),
]

CORR = [
    ("panel_01_radius.png", "fig:radius",
     r"""\textbf{Tolerance is a resolution parameter: the coarsest radius
separates best.} (\textbf{A})~Mean class overlap against refinement radius for
the \textsf{close} (green) and \textsf{far} (red) groups, with the shaded bands
running to the minimum \textsf{close} and maximum \textsf{far} value; the bands
are disjoint only at radius $0$. (\textbf{B})~The separation margin, minimum
\textsf{close} minus maximum \textsf{far}, by radius: $+0.375$ at radius $0$ and
exactly $0$ at radii $1$, $2$ and $3$. (\textbf{C})~Every annotated pair traced
across all four radii in three dimensions, \textsf{close} in green and
\textsf{far} in red; the \textsf{close} traces descend toward the \textsf{far}
floor as radius grows. (\textbf{D})~Overlap at radius $1$ against overlap at
radius $0$ for each pair, with the identity line; every \textsf{close} pair falls
below the diagonal, and one falls to zero."""),

    ("panel_02_separation.png", "fig:separation",
     r"""\textbf{Separation of the annotated groups at the working radius.}
(\textbf{A})~Class overlap for the two groups with horizontal bars at the group
means; the shaded band is the gap between the highest \textsf{far} value
($0.125$) and the lowest \textsf{close} value ($0.500$), which no pair occupies.
(\textbf{B})~Class overlap against element overlap per pair with the identity
line: the two \textsf{far} points on the floor at element overlap $0.5$ and $1.0$
are pairs an element-keyed comparison cannot distinguish and the cut class
separates completely. (\textbf{C})~All ten pairs sorted by class overlap, with
the two dashed lines marking the boundaries of the empty band.
(\textbf{D})~Element overlap, class overlap and structure size in three
dimensions, showing that the separation is not an artefact of the pairs having
different sizes."""),

    ("panel_03_correspondence.png", "fig:corr",
     r"""\textbf{Correspondence on matched pairs and its decay with radius.}
(\textbf{A})~Coverage (teal) against the size bound $\min(n_1,n_2)/\max(n_1,n_2)$
(grey) per pair, sorted; three of the eight pairs reach their ceiling, so raw
coverage understates the match quality for pairs of unequal size.
(\textbf{B})~Coverage against radius for each matched pair in three dimensions;
every pair loses coverage monotonically as the labelling discriminates more.
(\textbf{C})~Atoms matched against the atom count of the smaller structure, with
the identity line and colour by coverage; points on the line are pairs in which
every atom of the smaller structure found a correspondent. (\textbf{D})~Classes
shared between the two structures of each pair at each radius, showing the same
erosion from a different quantity."""),

    ("panel_04_control.png", "fig:rewire",
     r"""\textbf{The rewiring control and the cross-element gain.}
(\textbf{A})~Separation margin for the true structures ($+0.375$) and after
rewiring the second structure of each pair at random while holding atom
composition and edge count fixed ($-0.069$): the separation does not survive.
(\textbf{B})~Group means before and after rewiring; the \textsf{close} mean
collapses from $0.780$ to near the \textsf{far} value, while the \textsf{far}
mean barely moves. (\textbf{C})~Margin over radius under both conditions in three
dimensions: the true margin exists only at radius $0$ and the rewired margin is
negative everywhere. (\textbf{D})~Pairings admitted per \textsf{close} pair split
into same-element (blue) and cross-element (orange); the cross-element pairings,
which an element-keyed matcher cannot make, total $4$ across the six pairs and
occur in three of them."""),

    ("panel_05_matrix.png", "fig:matrix",
     r"""\textbf{Class and element overlap over the drug-like set.}
(\textbf{A})~Class-overlap matrix for all $\binom{30}{2}$ pairs of the drug-like
corpus at radius $0$, structures in corpus order. (\textbf{B})~Element-overlap
matrix for the same pairs on the same scale; the block structure differs
visibly, so the two quantities are not proxies for one another.
(\textbf{C})~Distributions of the two overlaps over the $406$ pairs with their
means marked: class overlap is shifted toward lower values, because agreeing on
composition is easier than agreeing on structural role. (\textbf{D})~Element
overlap, class overlap and absolute size difference in three dimensions,
coloured by class minus element overlap; points far from the diagonal plane are
pairs on which the two criteria most disagree."""),
]


def insert(path, figs, anchors):
    s = open(path, encoding="utf-8").read()
    if "graphicspath" not in s:
        s = s.replace("\\usepackage{graphicx}",
                      "\\usepackage{graphicx}\n\\graphicspath{{figures/}}")
    for anchor, idx in sorted(anchors, key=lambda a: -s.index(a[0])):
        fn, lbl, cap = figs[idx]
        block = FIG % (fn, cap, lbl)
        pos = s.index(anchor)
        s = s[:pos] + block + "\n" + s[pos:]
    open(path, "w", encoding="utf-8").write(s)
    return s.count("includegraphics")


if __name__ == "__main__":
    n1 = insert(
        os.path.join(DOCS, "cannonical-ranking-algorithm",
                     "canonical-ranking-algorithm.tex"),
        RANK,
        [("\\subsection{The base key is coarse}", 0),
         ("\\subsection{Refinement is necessary and sufficient in the measured range}", 1),
         ("\\subsection{Refinement recovers the orbit partition}", 2),
         ("\\subsection{Negative control}", 3),
         ("\\section{Relation to existing methods}", 4)],
    )
    n2 = insert(
        os.path.join(DOCS, "structural-correspondence",
                     "structural-correspondence.tex"),
        CORR,
        [("\\subsection{The radius sweep}", 0),
         ("\\subsection{Separation of bioisosteric from unrelated pairs}", 1),
         ("\\subsection{Correspondence on matched pairs}", 2),
         ("\\subsection{Negative control}", 3),
         ("\\section{Relation to existing methods}", 4)],
    )
    print("ranking figures:", n1)
    print("correspondence figures:", n2)
