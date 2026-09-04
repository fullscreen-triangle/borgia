/**
 * Tutorial programs.
 *
 * The honjo (.hnj) entries are executable and are checked against both
 * engines by scripts/check-tutorials.mjs. If one stops running, that
 * script fails — a tutorial that does not execute is worse than no
 * tutorial, because it teaches syntax the compiler rejects.
 *
 * The masamune (.msm) and meibutsu (.mbt) entries are illustrative:
 * neither language has an executable back end in this build. They are
 * marked `executable: false` and the workbench says so rather than
 * pretending a run occurred.
 *
 * `expect` describes what the program should produce. For executable
 * tutorials it is checked; for the others it is documentation.
 */

export const TUTORIALS = {
  honjo: {
    "01_cut.hnj": {
      executable: true,
      title: "The cut",
      source: `-- Individuation is not free.
-- Every value carries the floor it was resolved against.

floor 1.0                 -- the ambient resolution

C := cut 6                -- individuate Z=6 against the medium
observe C                 -- force the measurement now

-- The atom is not looked up in a table. Its shell structure,
-- term symbol and vacancy follow from the cut itself.`,
      expect: `Atom  [He] 2s2 2p2  term 3P_0  vacancy 4
cut count M = 1`,
    },

    "06_derivation.hnj": {
      executable: true,
      title: "Derivation, not lookup",
      source: `-- The atom is computed from Z by shell arithmetic:
-- Madelung filling, subshell capacity 2(2l+1), Hund's three rules.
-- Nothing here is read out of a table, which is why elements that
-- no table in this project ever listed still resolve.

floor 1.0

Fe := cut 26              -- iron: an open d shell
observe Fe

Cu := cut 29              -- copper: ground state departs from aufbau
observe Cu

U  := cut 92              -- uranium: the f block
observe U

-- Period and group are read off the derived configuration, so the
-- atom reports its own place in the table without being told it.`,
      expect: `Fe  [Ar] 3d6 4s2   5D_4    period=4 group=8
Cu  [Ar] 3d10 4s1  2S_1/2  period=4 group=11  [departs from aufbau]
U   [Rn] 5f4 7s2   5I_4    period=7 group=f
cut count M = 3`,
    },

    "02_closure.hnj": {
      executable: true,
      title: "Closure and geometry",
      source: `-- A compound is not a list of bonds. It is the state in
-- which every vacancy has been driven to zero.

floor 1.0

O := cut 8
H := cut 1

-- A bond is admitted only if it lowers thickness.
OH := O ~ H when delta > 0
observe OH

-- close drives every vacancy to zero; the stoichiometry and
-- the geometry are consequences, not inputs.
W := close O(H, H)
observe W

assert W.valence == closed emit "water did not close"`,
      expect: `OH admitted (delta > 0)
W : Compound  OH2  geometry=bent  angle=104.5  closed=true
Nobody supplied the 104.5 degrees — it follows from region count.`,
    },

    "03_refusal.hnj": {
      executable: true,
      title: "A closed shell refuses",
      source: `-- Not every request can be satisfied, and the language
-- says so rather than returning something plausible.

floor 1.0

Na := cut 11
Cl := cut 17
observe Na
observe Cl

NaCl := close Na(Cl)      -- 1:1, both open-shell
observe NaCl

-- Neon has vacancy 0: there is nothing to bond with.
Ne := cut 10
dead := Na ~ Ne when delta > 0
observe dead              -- exists = false, not an error`,
      expect: `NaCl closes 1:1
dead : exists=false — the refusal is a value, not an exception`,
    },

    "04_floor.hnj": {
      executable: true,
      title: "The floor is a resolution",
      source: `-- The floor is not a tolerance applied after the fact.
-- It is the scale the value is expressed in.
--
-- Run this, then change the floor to 2.0 and run again.
-- The residue scales with it; residue/floor does not.

floor 0.5

H := cut 1
C := cut 6
O := cut 8
Ne := cut 10

observe H                 -- residue/floor = 1.0
observe C                 -- residue/floor = 4.0
observe O                 -- residue/floor = 2.0
observe Ne                -- residue/floor = 1.0`,
      expect: `Measured over a 16-fold floor sweep, residue/floor is constant
per atom: 1.0 (H), 4.0 (C), 2.0 (O), 1.0 (Ne).
A truncation would show a ratio that moved with the floor.`,
    },

    "00_sandbox.hnj": {
      executable: true,
      title: "Sandbox — generator, closure, query",
      source: `-- SANDBOX. Edit anything here and press Run.
--
-- Three things happen, and each one draws a different panel:
--   cut Z     derives an atom from Z alone      -> Structure (shells)
--   close     drives vacancies to zero          -> Structure (3D)
--   track     follows an item to convergence    -> Trajectories, Invariants
--
-- Nothing below is tabulated. Change the 8 to a 7 and the
-- molecule becomes pyramidal, because the geometry is computed
-- from the vacancy count, not looked up.

floor 1.0
import honjo.causal

O := cut 8
H := cut 1
N := cut 7

-- Closure: bonds are admitted only while a vacancy remains.
W := close O(H, H)          -- bent, 104.5 degrees
A := close N(H, H, H)       -- pyramidal, 107 degrees

observe O
observe W
observe A

-- Query: follow oxygen through the compound it participates in.
path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path`,
      expect: `O : Z=8  vacancy=2  valence=6
W : Compound OH2  geometry=bent       angle=104.5  closed=true
A : Compound NH3  geometry=pyramidal  angle=107    closed=true
path : steps=2  converged=true  amalgamation=[O~H#1, O~H#2]

Both geometries are derived: k = bonded domains + lone pairs, and
the angle follows from k and the lone-pair count. Neither molecule
was looked up.`,
    },

    "05_track.hnj": {
      executable: true,
      title: "Tracking through a process",
      source: `-- Tracking an item through a process. The item is followed
-- through the compound it participates in, and the result is
-- the amalgamation of what it passed through.
--
-- Representation switching is allowed: mass, charge and time
-- are different ways of naming the same passage.

floor 1.0
import honjo.causal

O := cut 8
H := cut 1
W := close O(H, H)

path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path              -- the amalgamation IS the result`,
      expect: `path : steps=2  converged=true  amalgamation=[O~H#1, O~H#2]
The propagation converges rather than being cut off at a step
limit, and the two contacts oxygen passed through are named.`,
    },
  },

  masamune: {
    "01_provenance.msm": {
      executable: true,
      title: "What the record did not say",
      source: `-- Masamune translates a record into a contact graph and
-- reports how much of the graph the record actually stated.

plan hello {
  source lib : smiles at "one.smi"

  let raw  := read lib
  let mols := translate raw
                require element, connectivity
                else report

  emit mols with provenance
}`,
      expect: `Verdict: translated
Ethanol is written CCO -- three heavy atoms and two bonds. The graph
that comes out has nine atoms and eight contacts, because six hydrogens
and their bonds were supplied by the valence convention. Nothing in the
record mentions them.`,
    },

    "02_static_refusal.msm": {
      executable: true,
      title: "Refused before reading",
      source: `-- A request for a feature the format cannot state is
-- refused without opening the source. The editor marks the
-- line, and the marker is computed from the same declared
-- capability set the runner uses.

plan needs_geometry {
  source lib : smiles at "compounds.smi"

  let raw  := read lib
  let mols := translate raw
                require element, connectivity, coords3d
                else refuse

  emit mols
}`,
      expect: `status: refused -- missing coords3d.
records_read: 0. No record was opened; the refusal is static.
Measured across the reference set, 19 of 24 format/request pairs are
decided this way, and the static verdict agrees with the post-read
outcome on all 24.`,
    },

    "03_threshold.msm": {
      executable: true,
      title: "A threshold that admits nothing",
      source: `-- expect states a requirement before the run. This one
-- cannot be met by any SMILES source, and the editor says so
-- on the expect line.

plan mostly_stated {
  source lib : smiles at "compounds.smi"

  let raw  := read lib
  let mols := translate raw
                require element, connectivity, cellcount
                expect supplied < 0.25
                else report

  emit mols with provenance
}`,
      expect: `Every structure comes back incomplete rather than translated.
The measured minimum supplied fraction over this corpus is 0.500, so a
threshold of 0.25 admits nothing. The plan surfaces the question instead
of computing over supplied data.`,
    },

    "04_select_and_assert.msm": {
      executable: true,
      title: "Select, then assert",
      source: `-- select narrows a verdict set by a condition on the
-- provenance. assert states an expectation about what survived,
-- and halts the plan if it does not hold.
--
-- The corpus contains one record written with bracket atoms.
-- Bracket hydrogens are stated, so that structure alone reaches
-- a supplied fraction of exactly zero, and the assertion holds.

plan strict {
  source lib : smiles at "mixed.smi"
  budget 8 records

  let raw  := read lib
  let mols := translate raw
                require element, connectivity
                else report

  let core := select mols where supplied == 0.0

  assert core.count > 0  emit "no fully-stated structures in this corpus"
  emit core with provenance
}`,
      expect: `read 8 (the budget truncates 9 records to 8)
translate: 8 translated
select: 1 kept, 7 dropped -- only ammonia-explicit, written [NH3],
  reaches a supplied fraction of exactly 0.0
assert: observed 1, passed true
emit: one record, with its provenance

Change the assertion to > 1 and the plan halts instead, with
status assertion-failed and the emit never running.`,
    },

    "05_bracket_hydrogen.msm": {
      executable: true,
      title: "Stating what convention would supply",
      source: `-- Bracket atoms state their hydrogens explicitly. Compare
-- the supplied fraction of the same molecule written both ways
-- in mixed.smi: 'ethanol' as CCO and 'ethanol-explicit' as
-- [CH3][CH2][OH].
--
-- Same molecule. Different information sets.

plan compare_forms {
  source lib : smiles at "mixed.smi"

  let raw  := read lib
  let mols := translate raw
                require element, connectivity, hcount
                else report

  emit mols with provenance
}`,
      expect: `Every emitted record carries its own supplied fraction.
A bracket hydrogen is written in the record, so it is stated; an
implicit hydrogen is supplied by the organic-subset convention and
names that convention in its provenance.`,
    },
  },

  meibutsu: {
    "01_field.mbt": {
      executable: true,
      title: "The observation field",
      source: `-- A structure is realised as a complex field on a grid:
-- an amplitude and a phase at every point. This is what one
-- evaluation pass writes to a texture.

grid 256

spectrum water [3657, 1595, 3756]

observe water
report coordinates, energy, peak`,
      expect: `Three amplitude lobes, one per mode, at each mode's
normalised frequency address. The coordinates come out near
(0.944, 0.285, 1.0): water's third coordinate is exactly 1
because all three of its pairwise frequency ratios land on
low-order rationals -- the molecule interferes with itself.`,
    },

    "02_superposition.mbt": {
      executable: true,
      title: "Comparison by addition",
      source: `-- Two structures are compared by ADDING their fields.
-- The relational content is the cross-term; no similarity
-- function is evaluated on extracted features.
--
-- Spectra can be named from the reference set instead of
-- typed, which is how the corpus avoids transcription errors.

spectrum a = H2O
spectrum b = CO2

observe a
observe b

superpose a b`,
      expect: `|A+B|^2 = |A|^2 + |B|^2 + 2 Re<A,B>
own_energy is the first two terms -- what would be there if the
other structure were absent. relational is the third.

Water against carbon dioxide gives a small visibility and a
cross-term that is nearly balanced between constructive and
destructive points: they share almost nothing, and the
disagreement averages away rather than being reported as a
small similarity.`,
    },

    "03_self.mbt": {
      executable: true,
      title: "A field against itself",
      source: `-- The same structure on both sides. Cauchy-Schwarz with
-- equality gives visibility exactly 1 -- not approximately,
-- and with no assumption about how the phase is distributed.

spectrum w = H2O

observe w
superpose w w`,
      expect: `visibility: 1
constructive: 256, destructive: 0 -- every grid point in phase
relational equals own_energy exactly, because a field superposed
with itself contributes its whole energy to the cross-term.`,
    },

    "04_inversion.mbt": {
      executable: true,
      title: "Spectrum in, structure out",
      source: `-- The inverse direction. Given a measured value, rank the
-- structures that could have produced it.
--
-- Try replacing C6H6 with a spectrum you type by hand, and
-- watch what it ranks against.

spectrum query = C6H6

invert query`,
      expect: `The generating structure ranks first at visibility 1.
Measured over the 39-structure reference set, interference ranks
the true structure first 39/39. The addressing route -- base-3
prefix descent -- resolves only 27/39 uniquely, so it is a screen
and this is the route that identifies.`,
    },

    "05_refusal.mbt": {
      executable: true,
      title: "What the language refuses",
      source: `-- The reference frequency is a property of the corpus,
-- not a knob. Changing it would silently rescale every
-- address, so the language refuses rather than obliging.
--
-- Uncomment the last line to see the refusal.

grid 256
reference 4401

spectrum ok = HF
observe ok
report coordinates

-- reference 1000`,
      expect: `Declaring the reference at its actual value is accepted.
Declaring any other value is refused, with both numbers named.

The same applies to a mode frequency of zero or a negative
grid: these are refused at the line that wrote them rather
than producing a field that looks plausible.`,
    },
  },
};

export const LANGUAGES = Object.keys(TUTORIALS);
