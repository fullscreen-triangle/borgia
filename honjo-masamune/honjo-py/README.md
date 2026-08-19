# hjm — Honjo Masamune for Python

A prototyping implementation of two small languages that share one
semantic core. Everything produces JSON.

* **masamune** — translates chemical structure records (SMILES, XYZ) into
  *contact graphs*, tagging every element `stated` or `supplied`, and
  returning a labelled **verdict** rather than a graph-or-nothing. Also a
  plan language for running that over a corpus.
* **honjo** — computes on contact graphs with a single primitive, the
  **cut**, carrying a floor and a provenance tag on every value.

No hard dependencies. `networkx` is used for exact min-cut when present;
a brute-force fallback handles small graphs without it.

```bash
pip install -e .          # or just put the directory on PYTHONPATH
python -m hjm run examples/water.hj
```

## The three properties

**No sharp cut.** No value may claim zero residue. Every value carries a
floor `β > 0` and a residue `≥ β`; a `Value` with a smaller residue raises
at construction.

**No laundered provenance.** Every element of a translated graph is tagged
`stated` (the record said so) or `supplied` (a convention said so, and it
is named). Tags compose as the maximum under `stated < supplied`, so no
sequence of operations turns supplied data into a stated result.

**No empty-as-failure.** Every operation that can fail returns a labelled
verdict, and only `translated`/`cut` carry a value — enforced in
`Verdict.__post_init__`, so the invariant cannot be violated by
construction.

## honjo in one screen

```
floor 1.0                        -- nothing is free

C  := cut 6                      -- arity-one cut: individuate Z=6
O  := cut 8
H  := cut 1

W   := close O(H, H)             -- closure: 2:1, four regions, two lone pairs
CO2 := close C(O : 2, O : 2)     -- committed cell counts: each C=O is 2

ring := deloc ring(c1,c2,c3,c4,c5,c6) cells: 9   -- one system, not 6 bonds

g := import graph "target.smi"   -- structures arrive from masamune
       require supplied < 0.10
       unless refuse

observe W
assert W.vacancy == 0  emit "water did not close"
```

Run it:

```bash
python -m hjm run examples/water.hj
python -m hjm run -e 'floor 1.0
C := cut 6
observe C'
```

### Why `close Ne(H)` is not an error

It is a verdict:

```json
{"verdict": "inert",
 "payload": {"closed_shell": [{"symbol": "Ne", "vacancy": 0}],
             "certified_vacancy_zero": true}}
```

which is a *different fact* from `close N(H,H)`:

```json
{"verdict": "unclosed",
 "payload": {"open": [{"symbol": "N", "vacancy": 3,
                       "committed": 2, "residual": 1}]}}
```

"Neon forms no compound" and "this ammonia attempt was short one hydrogen"
are different claims. An interface returning nothing for both has said
neither.

## masamune in one screen

```bash
python -m hjm translate "CCO"
python -m hjm translate "c1ccccc1" --require element,connectivity,delocalisation
python -m hjm caps                       # per-format capability sets
```

Measured on the bundled examples:

| record | atoms | supplied fraction |
|---|---|---|
| `O=C=O` | 3 | **0.000** |
| `[CH3][CH2][OH]` | 9 | 0.118 |
| `c1ccccc1` | 12 | 0.750 |
| `CCO` | 9 | 0.824 |

`CCO` and `[CH3][CH2][OH]` are the same molecule. They differ in what the
record *states*, and that difference is what the conventional pipeline
discards.

Medium edges are excluded from the denominator: no record states them, so
counting them would make the statistic a property of the target
representation rather than of the source.

### Plans

```
plan provenance_audit {
  source lib : smiles at "compounds.smi"
  budget 100 records

  let raw  := read lib
  let mols := translate raw
                require element, connectivity, cellcount
                expect supplied < 0.25
                else report

  let core := select mols where supplied == 0.0
  assert core.count > 1  emit "too few fully-stated structures"
  emit core with provenance
}
```

```bash
python -m hjm run examples/audit.ms
```

Declaration order is execution order — no dependency inference. The
capability check runs **before any file is opened**, so a plan asking a
format for something it cannot state is refused with `records_read: 0`.

## Python API

```python
from hjm import run_honjo, run_plan, translate_smiles

out = run_honjo("floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)")
out["cut_count"]                      # 4 — monotone clock
[b["name"] for b in out["bindings"]]  # ['O', 'H', 'W']

v = translate_smiles("CCO")
v.label                               # Label.TRANSLATED
v.value.supplied_fraction()           # 0.8235...
v.value.separation_cost("a0")         # (sigma, minimising side, provenance)
v.value.burial_depth("a0")            # items on the minimising side
```

## Layout

```
hjm/
  core/     graph.py     contact graph, provenance, cuts, min-cut
            verdict.py   the verdict algebra
            chem.py      shell filling, vacancy, geometry table
  masamune/ capability.py per-format feature sets
            smiles.py     a SMILES reader that records what it supplied
            translate.py  clause-ordered translation to verdicts
            plan.py       the plan language
  honjo/    interp.py     lexer, parser, evaluator
  cli.py                  python -m hjm
```

## Tests

```bash
python -m pytest tests/ -q     # 45 tests
```

Several are negative controls whose pass condition is a refusal: the
resolution gate, the capability refusal before reading, the four-way
verdict distinguishability, and the plan assertion failure.

## What this does not do

* **No bond energetics.** `close` decides *whether* an interface lowers
  boundary thickness and *how many* cells it commits, not by how much.
  Residues are separation costs, not energies.
* **No covalent/ionic distinction.** Interfaces are treated through
  vacancy sharing alone.
* **Multiple bonding is counted, not adjudicated.** Nothing decides
  whether a double interface or two singles is the realised structure; a
  program states which it means.
* **No perception.** No ring perception, no aromaticity model beyond
  SMILES' own lower-case marking, no stereo descriptors — `stereo` is
  deliberately absent from the SMILES capability set, and requesting it
  yields `unsupported`.
* **Capability declarations are asserted, not verified.** Under-declaring
  is safe; over-declaring is unsound and silent. Nothing here checks them
  against an independent reader.
* **The `d`-block is not treated.** `valence()` handles main-group s/p
  filling; transition metals return the s+p count and are not meaningful.
