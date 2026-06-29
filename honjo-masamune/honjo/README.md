# honjo — the Honjo Masamune language (TypeScript playground compiler)

A cut-primitive DSL for cheminformatics by individuation. This is the
**TypeScript / playground** implementation — fast, in-browser-ready, and the
exploratory target. The authoritative implementation is the Rust reference
target (`../honjo-rs`). Both compile from one front end through one typed
core IR (Cut-IR); they differ only in how a cut is realised. See the language
specification in `../docs/honjo-dsl/honjo-masamune-dsl.tex`.

## The idea in one line

Every operation is a **cut** — individuating a part of a bounded whole at the
irreducible cost `floor > 0`. Generation is a length-one cut; a bond is a cut
between two items; a compound is a cut to shell closure; a track is a
residue-chained sequence of cuts. One verb, one composition rule.

## Build & run

```bash
npm install
npm run build           # tsc -> dist/
npm test                # 32 checks
node dist/cli.js examples/water.hj
node dist/cli.js examples/track.hj --ir   # show the Cut-IR
```

## The language

```
floor 1.0                      -- the cost of any cut; the currency. nothing is free.

C  := cut 6                    -- individuate Z=6 (a length-one cut) -> carbon
OH := O ~ H when delta > 0     -- a cut between two items (a bond), guarded
W  := close O(H, H)            -- cut to shell closure -> water, bent 104.5 deg
path := track O in W           -- a residue chain of cuts (the causal table)
          with reps mass, charge, time
          until converge
          yield amalgamation
observe W                      -- force the measurement now
assert W.valence == closed emit "did not close"
```

### Core forms

| form | meaning | arity | binds to |
|------|---------|-------|----------|
| `cut N` | individuate an atom of atomic number N | 1 | spectroscopic atom derivation |
| `a ~ b [when c]` | bond (shared boundary), exists iff `delta > 0` | 2 | boundary-thickness criterion |
| `close X(a,...)` | drive every vacancy to zero | closure | vacancy closure + VSEPR geometry |
| `track x in P until <admit> yield r` | propagate x's uncertainty | chain | accountable propagation + S-entropy convergence |
| `floor f` | set the ambient floor (`f > 0`) | — | the contact floor |
| `observe e` | force measurement (commit the cut now) | — | — |
| `assert c emit "msg"` | pre-flight check | — | — |

### Accountability

Every value carries its **floor** and **residue**; there are no bare numbers.
A value of zero or negative floor — the forbidden sharp cut — is rejected at
compile time (`floor 0`, `5.0#0`, `floor -1` all fail the type checker). The
**cut count `M`** is a monotone clock: it strictly increases at every cut event,
so evaluation *is* measurement, never a cached recomputation.

## Pipeline

```
source .hj
  -> lexer.ts     (tokens, §3)
  -> parser.ts    (AST, EBNF §4)
  -> types.ts     (accountability check, §5; no-zero-residue)
  -> lower.ts     (Cut-IR, §8)        <- shared by both targets
  -> interp.ts    (small-step opsem §6, exact back end)
```

The stdlib verbs (`stdlib.ts`) port the framework's validated reference
operations so honjo's results match: shell capacity `C(n)=2n^2`, vacancy/valence,
Δ-thickness bonding, vacancy-matched stoichiometry, VSEPR angles, and
convergence-admissible propagation.
