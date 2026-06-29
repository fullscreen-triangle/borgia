// Honjo Masamune — Standard library (§7)
// Each verb binds to a validated framework operation. The logic here is ported
// from the framework's validated reference implementations so that honjo's
// results match: Individuate (spectroscopic atom), Bond (Delta-thickness),
// Close (vacancy closure + geometry), Propagate (accountable propagation).

// ----------------------------------------------------------------------------
//  Atom value
// ----------------------------------------------------------------------------
export interface AtomVal {
  ty: "Atom";
  Z: number;
  symbol: string;
  config: string;
  term: string;
  qv: number;       // valence-shell occupancy
  capV: number;     // valence-shell capacity (duet 2 / octet 8 for light elements)
  vacancy: number;  // nu = capV - qv (holes to closure)
  valence: number;  // bonding capacity = min(nu, capV - nu)  (chemical-structure def)
  floor: number;
  residue: number;
}

export interface BondVal {
  ty: "Bond";
  a: string; b: string;
  delta: number;     // shared content Delta-thickness
  shared: number;    // min(nu_a, nu_b) cells committed
  exists: boolean;
  floor: number;
  residue: number;
}

export interface CompoundVal {
  ty: "Compound";
  central: string;
  ligand: string;
  formula: [number, number]; // [central count, ligand count]
  ligands: number;
  geometry: string;
  angleDeg: number | null;
  valenceClosed: boolean;
  floor: number;
  residue: number;
}

export interface PathVal {
  ty: "Path";
  item: string;
  steps: number;          // committed cut count along the chain
  converged: boolean;
  amalgamation: string[]; // accounted contacts
  reps: string[];         // representations used
  floor: number;
  residue: number;        // total committed boundary
}

export type CutVal = AtomVal | BondVal | CompoundVal | PathVal;

// ----------------------------------------------------------------------------
//  Element table (light elements; valence shell + common data)
//  qv = valence electrons, capV = 2 (duet, n=1) or 8 (octet, main group)
// ----------------------------------------------------------------------------
interface ElemRec { sym: string; Z: number; qv: number; capV: number; config: string; term: string; }

const ELEMENTS: ElemRec[] = [
  { sym: "H",  Z: 1,  qv: 1, capV: 2, config: "1s1",                 term: "2S1/2" },
  { sym: "He", Z: 2,  qv: 2, capV: 2, config: "1s2",                 term: "1S0" },
  { sym: "Li", Z: 3,  qv: 1, capV: 8, config: "[He] 2s1",            term: "2S1/2" },
  { sym: "Be", Z: 4,  qv: 2, capV: 8, config: "[He] 2s2",            term: "1S0" },
  { sym: "B",  Z: 5,  qv: 3, capV: 8, config: "[He] 2s2 2p1",        term: "2P1/2" },
  { sym: "C",  Z: 6,  qv: 4, capV: 8, config: "[He] 2s2 2p2",        term: "3P0" },
  { sym: "N",  Z: 7,  qv: 5, capV: 8, config: "[He] 2s2 2p3",        term: "4S3/2" },
  { sym: "O",  Z: 8,  qv: 6, capV: 8, config: "[He] 2s2 2p4",        term: "3P2" },
  { sym: "F",  Z: 9,  qv: 7, capV: 8, config: "[He] 2s2 2p5",        term: "2P3/2" },
  { sym: "Ne", Z: 10, qv: 8, capV: 8, config: "[He] 2s2 2p6",        term: "1S0" },
  { sym: "Na", Z: 11, qv: 1, capV: 8, config: "[Ne] 3s1",            term: "2S1/2" },
  { sym: "Mg", Z: 12, qv: 2, capV: 8, config: "[Ne] 3s2",            term: "1S0" },
  { sym: "Al", Z: 13, qv: 3, capV: 8, config: "[Ne] 3s2 3p1",        term: "2P1/2" },
  { sym: "Si", Z: 14, qv: 4, capV: 8, config: "[Ne] 3s2 3p2",        term: "3P0" },
  { sym: "P",  Z: 15, qv: 5, capV: 8, config: "[Ne] 3s2 3p3",        term: "4S3/2" },
  { sym: "S",  Z: 16, qv: 6, capV: 8, config: "[Ne] 3s2 3p4",        term: "3P2" },
  { sym: "Cl", Z: 17, qv: 7, capV: 8, config: "[Ne] 3s2 3p5",        term: "2P3/2" },
  { sym: "Ar", Z: 18, qv: 8, capV: 8, config: "[Ne] 3s2 3p6",        term: "1S0" },
];

const BY_Z = new Map(ELEMENTS.map((e) => [e.Z, e]));

// Shell capacity C(n) = 2 n^2 (the partition-coordinate count) — exposed as a
// pure verb for programs that want it.
export function shellCapacity(n: number): number {
  let c = 0;
  for (let l = 0; l < n; l++) c += 2 * (2 * l + 1);
  return c; // == 2 n^2
}

// ----------------------------------------------------------------------------
//  Thickness model (matches the validated chemical-structure suite):
//    B(nu) = B0 + kappa * phi(nu),  phi(nu) = nu,  B0 = floor (closed-shell min)
// ----------------------------------------------------------------------------
function thickness(nu: number, floor: number, kappa = 1.0): number {
  return floor + kappa * nu;
}

// ----------------------------------------------------------------------------
//  Individuate (cut Z)  — spectroscopic atom individuation
// ----------------------------------------------------------------------------
export function individuate(Z: number, floor: number): AtomVal {
  if (!Number.isInteger(Z) || Z < 1) throw new Error(`cut: atomic number must be a positive integer (got ${Z})`);
  const e = BY_Z.get(Z);
  if (!e) throw new Error(`cut: element Z=${Z} not in the light-element table (1..18 supported)`);
  const vacancy = e.capV - e.qv;
  const valence = Math.min(vacancy, e.capV - vacancy); // bonding capacity (chemical-structure def)
  // residue = the boundary deposited individuating this atom from the medium;
  // bounded below by the floor, increasing with vacancy (the partition malformation).
  const residue = thickness(vacancy, floor);
  return {
    ty: "Atom", Z, symbol: e.sym, config: e.config, term: e.term,
    qv: e.qv, capV: e.capV, vacancy, valence, floor, residue,
  };
}

// ----------------------------------------------------------------------------
//  Bond (a ~ b)  — boundary-thickness bonding criterion
// ----------------------------------------------------------------------------
export function bond(a: AtomVal, b: AtomVal, floor: number, kappa = 1.0): BondVal {
  const shared = Math.min(a.vacancy, b.vacancy);
  const sep = thickness(a.vacancy, floor, kappa) + thickness(b.vacancy, floor, kappa);
  const joined =
    thickness(Math.max(a.vacancy - shared, 0), floor, kappa) +
    thickness(Math.max(b.vacancy - shared, 0), floor, kappa);
  const delta = sep - joined;            // shared content; > 0 iff both open-shell
  const exists = delta > 1e-12;
  return {
    ty: "Bond", a: a.symbol, b: b.symbol, delta, shared, exists,
    floor, residue: Math.max(delta, floor),
  };
}

// ----------------------------------------------------------------------------
//  Close X(ligands)  — compound by vacancy closure + geometry
// ----------------------------------------------------------------------------
const ANGLE_TET = (Math.acos(-1 / 3) * 180) / Math.PI; // 109.4712 deg

export function close(central: AtomVal, ligands: AtomVal[], floor: number): CompoundVal {
  if (ligands.length === 0) throw new Error("close: needs at least one ligand");
  const lig = ligands[0];

  // homonuclear (e.g. H + H) -> diatomic
  if (central.symbol === lig.symbol) {
    return {
      ty: "Compound", central: central.symbol, ligand: lig.symbol,
      formula: [2, 0], ligands: 1, geometry: "linear", angleDeg: 180,
      valenceClosed: true, floor, residue: thickness(central.vacancy, floor),
    };
  }

  // Stoichiometry by valence matching: each central-ligand bond consumes one
  // unit of valence on each side. nLig = central.valence / ligand.valence.
  const vC = Math.max(central.valence, 1);
  const vL = Math.max(lig.valence, 1);
  const nLig = Math.max(Math.round(vC / vL), 1);

  // Electron-domain count for VSEPR geometry: bonded domains + lone pairs on
  // the central atom. Lone pairs = (valence-shell electrons - electrons in
  // sigma bonds) / 2, for the octet (main-group) case.
  const bondedDomains = nLig;
  const lonePairs = central.capV === 8
    ? Math.max(0, Math.floor((central.qv - nLig) / 2))
    : 0;
  const k = bondedDomains + lonePairs;

  let geometry = "point";
  let angleDeg: number | null = null;
  if (k === 1) { geometry = "terminal"; angleDeg = null; }
  else if (k === 2) { geometry = lonePairs === 0 ? "linear" : "bent"; angleDeg = lonePairs === 0 ? 180 : ANGLE_TET; }
  else if (k === 3) { geometry = lonePairs === 0 ? "trigonal" : "bent"; angleDeg = lonePairs === 0 ? 120 : 117.0; }
  else if (k >= 4) {
    geometry = lonePairs === 0 ? "tetrahedral"
             : lonePairs === 1 ? "pyramidal" : "bent";
    // symmetric backbone 109.47, compressed by lone-pair repulsion (observed-style)
    angleDeg = lonePairs === 0 ? round2(ANGLE_TET) : lonePairs === 1 ? 107.0 : 104.5;
  }

  return {
    ty: "Compound", central: central.symbol, ligand: lig.symbol,
    formula: [1, nLig], ligands: nLig, geometry, angleDeg,
    valenceClosed: true, floor, residue: thickness(0, floor) + nLig * floor,
  };
}

function round2(x: number): number { return Math.round(x * 100) / 100; }

// ----------------------------------------------------------------------------
//  Propagate (track x in P) — accountable propagation with convergence
// ----------------------------------------------------------------------------
export function propagate(
  item: AtomVal,
  process: CompoundVal | PathVal,
  reps: string[],
  admit: "converge" | "diverge" | { holds: boolean },
  floor: number,
): PathVal {
  // The amalgamation is the accounted chain of contacts by which the item is
  // tracked through the process. For a compound, the contacts are the bonds
  // the item participates in; each is one committed cut (a residue step).
  const contacts: string[] = [];
  let steps = 0;
  let residue = 0;

  if (process.ty === "Compound") {
    const n = process.ligands;
    for (let i = 0; i < n; i++) {
      contacts.push(`${process.central}~${process.ligand}#${i + 1}`);
      steps += 1;
      residue += floor; // each committed contact deposits >= floor
    }
  } else {
    steps = process.steps;
    residue = process.residue;
    contacts.push(...process.amalgamation);
  }

  // Admissibility = S-entropy convergence to the output. A tracked chain that
  // reaches the output converges; we model convergence as "the chain closed
  // with positive committed residue" (residue >= floor) for `converge`.
  let converged: boolean;
  if (admit === "converge") converged = residue >= floor && steps > 0;
  else if (admit === "diverge") converged = !(residue >= floor && steps > 0);
  else converged = admit.holds;

  return {
    ty: "Path", item: item.symbol, steps, converged,
    amalgamation: converged ? contacts : [],
    reps: reps.length ? reps : ["mass"],
    floor, residue,
  };
}
