# Workbench

An editor for the three languages, with results computed on your own
machine.

## Why a local engine

The browser can run the JavaScript build of the honjo front end, and
that is what happens by default. It is a real implementation and it
agrees with the reference compiler on the whole tutorial set. But it is
a *second* implementation, and where two implementations disagree one of
them is wrong.

Connecting a local engine means the Rust reference compiler does the
work. Your source never leaves the machine — the browser talks to a
process you started, over loopback.

## Connect in three steps

**1. Build and start the engine.**

```
cd honjo-rs
cargo install --path .
honjo serve
```

It prints something like:

```
  honjo local engine
  ------------------
  listening   http://127.0.0.1:8731
  token       6e2bfab86a16c322bd0c2c3b75c2e593
```

**2. Copy the token.**

**3. Paste it into the workbench** — click the engine badge in the top
right, paste, press Connect.

The badge turns green and every result is then labelled
`rust reference engine`.

## What the token is and is not

* It is generated fresh each time `honjo serve` starts, and it dies when
  you stop the process. There is nothing to revoke and nothing stored.
* It is held in `sessionStorage`, not `localStorage`: persisting a dead
  secret across browser sessions would serve no purpose.
* It is compared in constant time, so a wrong token leaks nothing about
  the right one through response timing.
* It is **not** an account. There is no registration and no server
  besides the one you started.

The listener binds to `127.0.0.1` only. It is not reachable from the
network, and the token is a second line of defence rather than the
first.

### If the page origin is refused

The engine allows `http://localhost:3000` and `http://127.0.0.1:3000` by
default. Serving the workbench from anywhere else needs that origin
allowed explicitly:

```
honjo serve --origin https://your.host
```

### On Windows

The token is currently derived from process time and address rather than
from the OS entropy source, and `honjo serve` prints a warning saying
so. It is unique enough to prevent accidental collision on a
single-user machine, but it is not cryptographically strong. If that
matters for your setup, run the engine under WSL, where `/dev/urandom`
is used instead.

## Which languages execute

| Language | Extension | Executes | Engine |
|---|---|---|---|
| honjo | `.hnj` | yes | Rust reference, or JS in-browser |
| masamune | `.msm` | no | — |
| meibutsu | `.mbt` | no | — |

The masamune and meibutsu tutorials are illustrative: the workbench
shows measured results for them rather than pretending a run occurred.
Their measurements come from the validation suites, not from the editor.

## Where the displayed numbers come from

Every number in the results panel is generated from a validation run.
No component contains a measured value as a literal.

```
python src/data/generate.py
```

reads

* `honjo-py/hjm/masamune/capability.py` — the declared capability sets
* `validation/results/exp_masamune.json`
* `validation/results/exp_honjo.json`
* `../dmitri/publications/graphical-chemistry-generator/results/exp_generator.json`

and writes `src/data/*.json`. If a result file is missing it fails
rather than emitting a partial file.

## The interference panel

The Interference tab computes an observation field for each of two
structures, superposes them, and draws the result. The canvas is not a
picture of a comparison made elsewhere: the pixel buffer and the
reported visibility come from the same two arrays, in the browser.

That is only defensible if the browser computes what the reference
implementation computes.

```
npm run check:field
```

compares coordinates, energies and all pairwise visibilities against a
dump from the Python reference and fails on any disagreement beyond
1e-9. Current state: 126/126 agree to 1e-13, self-visibility is exactly
1 for every structure, and the cross-term identity residual is 9e-16.

The **display precision** buttons quantise the field to a fixed number
of bits per channel, as a framebuffer would, and recompute the
visibility from the quantised arrays. Eight bits is what an ordinary
framebuffer stores; the measured result is that inversion accuracy at 8
bits equals accuracy at 16 exactly. The array that is displayed can be
the array that is compared.

Reference spectra come from `src/data/spectra.json`, generated from the
same NIST-derived database the validation used. They are not typed by
hand — a first attempt at that got three of twelve wrong.

## Keeping the tutorials honest

```
npm run check:tutorials
```

runs every executable tutorial through the browser engine and fails if
one of them errors. With a local engine reachable it runs them through
the Rust compiler too and reports any divergence in cut count:

```
HONJO_ENDPOINT=http://127.0.0.1:8731 HONJO_TOKEN=<token> npm run check:tutorials
```

Current state: 5/5 tutorials run, 0 divergences.

Run both checks together with `npm run check`.

## The editor markers

Gutter markers are predictions that the compiler will refuse, computed
from the same declared capability sets the compiler uses. A marker on

```
require element, connectivity, coords3d
```

against a SMILES source says that request is refused statically, before
the file is opened — which is what the compiler does.

There is deliberately no style advice. A linter that flags what the
language permits trains people to ignore it.
