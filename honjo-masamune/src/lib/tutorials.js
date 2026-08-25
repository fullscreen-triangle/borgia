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
      expect: `Atom  [He] 2s2 2p2  term 3P0  vacancy 4
cut count M = 1`,
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
      executable: false,
      title: "What the record did not say",
      source: `-- Masamune translates a record into a contact graph and
-- reports how much of the graph the record actually stated.

plan hello {
  source input : smiles at "ethanol.smi"

  let records := read input

  let result := translate records
    require element, connectivity
    else report

  emit result with provenance
}`,
      expect: `Verdict: translated
Measured over 48 structures, the mean supplied fraction is 0.734
and the minimum is 0.500. Every hydrogen is supplied: the SMILES
string never writes one.`,
    },

    "02_static_refusal.msm": {
      executable: false,
      title: "Refused before reading",
      source: `-- A request for a feature the format cannot state is
-- refused without opening the file. The editor marks this
-- line: the marker is computed from the same declared
-- capability set the compiler uses.

plan needs_geometry {
  source input : smiles at "benzene.smi"

  let records := read input

  let result := translate records
    require element, connectivity, coords3d
    else refuse
}`,
      expect: `Verdict: unsupported — missing coords3d.
Measured: 19 of 24 format/request pairs are decided statically,
and the static verdict agrees with the post-read outcome on all 24.`,
    },

    "03_threshold.msm": {
      executable: false,
      title: "A threshold that admits nothing",
      source: `-- expect states a requirement in advance. This one cannot
-- be met by any SMILES source, and the editor says so.

plan mostly_stated {
  source lib : smiles at "compounds.smi"

  let raw := read lib

  let mols := translate raw
    require element, connectivity, cellcount
    expect supplied < 0.25
    else report

  emit mols with provenance
}`,
      expect: `Every structure is reported rather than emitted.
The measured minimum φ over the corpus is 0.500, so a threshold
of 0.25 admits nothing. The plan surfaces the question instead of
computing over supplied data.`,
    },
  },

  meibutsu: {
    "01_field.mbt": {
      executable: false,
      title: "The observation field",
      source: `-- A structure is realised as a complex field on a grid:
-- an amplitude and a phase at every point. This is what one
-- evaluation pass writes to a texture.

meibutsu field {
  spectrum [3657, 1595, 3756]     -- water, cm-1
  grid 256
  reference 4401

  compute amplitude, phase, energy
  display field
}`,
      expect: `One amplitude lobe per mode at its normalised frequency
address. The phase carries the coordinate information; the
amplitude alone would not distinguish structures whose modes
coincide.`,
    },

    "02_superposition.mbt": {
      executable: false,
      title: "Comparison by addition",
      source: `-- Two structures are compared by adding their fields.
-- The relational content is the cross-term. No similarity
-- function is evaluated on extracted features.

meibutsu compare {
  spectrum a [3657, 1595, 3756]   -- water
  spectrum b [1333, 667, 2349]    -- carbon dioxide

  superpose
  compute visibility
  display interference
}`,
      expect: `|A+B|^2 = |A|^2 + |B|^2 + 2 Re<A,B>
The first two terms are properties of each structure alone.
Self-comparison is exactly 1 for all 39 reference structures,
by Cauchy-Schwarz — not by assumption about phase.`,
    },

    "03_inversion.mbt": {
      executable: false,
      title: "Spectrum in, structure out",
      source: `-- The inverse direction: given a measured value, return
-- the structure that would have produced it.

meibutsu invert {
  query [3657, 1595, 3756]

  route address           -- base-3 prefix descent
  route interference      -- visibility ranking

  compute agreement
}`,
      expect: `Interference ranks the generating structure first 39/39.
Address uniqueness resolves only 27/39, so the address is a
screen and the interference route does the identification.
The two routes disagree and both are reported.`,
    },
  },
};

export const LANGUAGES = Object.keys(TUTORIALS);
