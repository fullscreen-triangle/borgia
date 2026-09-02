// Honjo Masamune — Shell arithmetic
//
// The atom is DERIVED, not looked up. Everything below follows from three
// inputs — the Madelung order, the subshell capacity 2(2l+1), and Hund's
// three rules — applied to a single integer Z.
//
// This replaces the eighteen-row element table that previously backed
// `individuate`. That table made the tutorial's central claim ("the atom is
// not looked up in a table") false, and capped the language at Z <= 18. The
// port is from the validated reference
// (dmitri/publications/atomic-derivation/validate_spectroscopic_derivation.py),
// which is checked against NIST; test/shell.test.ts re-checks this port
// against the same committed benchmark in
// atomic-derivation/results/term_symbols.json.
//
// Consequence worth stating plainly: adding an element is no longer an edit.
// Z=26 works because 26 is an integer, not because someone typed a row for it.

/** A filled subshell: principal n, azimuthal l, occupancy. */
export type Subshell = { n: number; l: number; occ: number };

/** Subshell capacity: two spin states per m_l value. */
export function subshellCapacity(l: number): number {
  return 2 * (2 * l + 1);
}

/**
 * Shell capacity C(n) = sum over l<n of 2(2l+1) = 2n^2 — the partition
 * coordinate count. Computed as the sum rather than as 2n^2 so that the
 * identity is a derived fact the tests can check, not an assumption.
 */
export function shellCapacity(n: number): number {
  let c = 0;
  for (let l = 0; l < n; l++) c += subshellCapacity(l);
  return c;
}

const L_LETTER = ["s", "p", "d", "f", "g", "h"];

/** Term-symbol letters for total orbital angular momentum L = 0,1,2,... */
const S_LABELS = "S P D F G H I K L M N O Q R T U V".split(" ");

/**
 * Madelung (n+l, then n) filling order. Listed explicitly rather than
 * generated so the ordering is auditable against the textbook sequence.
 */
export const MADELUNG_ORDER: [number, number][] = [
  [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [3, 2], [4, 1],
  [5, 0], [4, 2], [5, 1], [6, 0], [4, 3], [5, 2], [6, 1], [7, 0],
  [5, 3], [6, 2], [7, 1],
];

/** Noble-gas cores, canonically ordered by (n, l), for abbreviation. */
const CORE_CONFIGS: [string, Subshell[]][] = [
  ["[Rn]", expand([[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[4,3,14],[5,0,2],[5,1,6],[5,2,10],[6,0,2],[6,1,6]])],
  ["[Xe]", expand([[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[5,0,2],[5,1,6]])],
  ["[Kr]", expand([[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6]])],
  ["[Ar]", expand([[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6]])],
  ["[Ne]", expand([[1,0,2],[2,0,2],[2,1,6]])],
  ["[He]", expand([[1,0,2]])],
];

function expand(rows: number[][]): Subshell[] {
  return rows.map(([n, l, occ]) => ({ n, l, occ }));
}

/** Canonical NIST ordering: by (n, l), not by filling order. */
function canonical(cfg: Subshell[]): Subshell[] {
  return [...cfg].sort((a, b) => (a.n - b.n) || (a.l - b.l));
}

/**
 * Ground-state configurations that do not follow strict aufbau filling.
 *
 * These are empirical: a half- or fully-filled d shell lies below the
 * configuration Madelung predicts. Listing them is an admission that the
 * ordering rule is a good approximation rather than a law — the alternative
 * would be to silently report a wrong configuration for chromium. Each entry
 * is the NIST ground state.
 */
const AUFBAU_EXCEPTIONS: Record<number, number[][]> = {
  24: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,5],[4,0,1]],            // Cr [Ar]3d5 4s1
  29: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,1]],           // Cu [Ar]3d10 4s1
  41: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,4],[5,0,1]],   // Nb
  42: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,5],[5,0,1]],   // Mo
  44: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,7],[5,0,1]],   // Ru
  45: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,8],[5,0,1]],   // Rh
  46: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10]],          // Pd (no 5s)
  47: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[5,0,1]],  // Ag
  64: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[4,3,7],[5,0,2],[5,1,6],[5,2,1],[6,0,2]], // Gd
  78: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[4,3,14],[5,0,2],[5,1,6],[5,2,9],[6,0,1]], // Pt
  79: [[1,0,2],[2,0,2],[2,1,6],[3,0,2],[3,1,6],[3,2,10],[4,0,2],[4,1,6],[4,2,10],[4,3,14],[5,0,2],[5,1,6],[5,2,10],[6,0,1]], // Au
};

/** Element symbols, indexed by Z-1. A name is a label, not a derivation. */
const SYMBOLS = (
  "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca " +
  "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr " +
  "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd " +
  "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg " +
  "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm " +
  "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split(" ");

export const MAX_Z = SYMBOLS.length; // 118

export function symbolOf(Z: number): string {
  return SYMBOLS[Z - 1] ?? `Z${Z}`;
}

/** Fill subshells in Madelung order until Z electrons are placed. */
export function aufbauConfig(Z: number): Subshell[] {
  let remaining = Z;
  const cfg: Subshell[] = [];
  for (const [n, l] of MADELUNG_ORDER) {
    if (remaining <= 0) break;
    const occ = Math.min(remaining, subshellCapacity(l));
    cfg.push({ n, l, occ });
    remaining -= occ;
  }
  if (remaining > 0) {
    throw new Error(
      `cut: Z=${Z} exceeds the subshells in the Madelung order (through 7p)`
    );
  }
  return canonical(cfg);
}

/** Ground-state configuration: aufbau, corrected where NIST disagrees. */
export function deriveConfiguration(Z: number): Subshell[] {
  const ex = AUFBAU_EXCEPTIONS[Z];
  if (ex) return canonical(expand(ex));
  return aufbauConfig(Z);
}

/** Whether this Z is one where strict aufbau gives the wrong ground state. */
export function isAufbauException(Z: number): boolean {
  return AUFBAU_EXCEPTIONS[Z] !== undefined;
}

/** Render as "1s2 2s2 2p2", abbreviating with the largest matching core. */
export function configToString(cfg: Subshell[], abbreviate = true): string {
  const full = () => cfg.map((s) => `${s.n}${L_LETTER[s.l]}${s.occ}`).join(" ");
  if (!abbreviate) return full();

  const byNL = new Map(cfg.map((s) => [`${s.n},${s.l}`, s.occ]));
  for (const [label, core] of CORE_CONFIGS) {
    const matches = core.every((c) => byNL.get(`${c.n},${c.l}`) === c.occ);
    if (!matches) continue;
    const coreSet = new Set(core.map((c) => `${c.n},${c.l}`));
    const valence = cfg.filter((s) => !coreSet.has(`${s.n},${s.l}`));
    if (valence.length === 0) return label;
    return `${label} ` + valence.map((s) => `${s.n}${L_LETTER[s.l]}${s.occ}`).join(" ");
  }
  return full();
}

/** Subshells that are neither empty nor full — the ones that carry the term. */
export function openShells(cfg: Subshell[]): Subshell[] {
  return cfg.filter((s) => s.occ > 0 && s.occ < subshellCapacity(s.l));
}

/**
 * Spin and orbital angular momentum of one subshell under Hund's first two
 * rules: maximise S by filling each m_l singly before pairing, then take the
 * resulting M_L.
 */
function couple(s: Subshell): { S: number; ML: number } {
  const nOrb = 2 * s.l + 1;
  const mls: number[] = [];
  for (let m = s.l; m >= -s.l; m--) mls.push(m);

  const occPer = new Array(nOrb).fill(0);
  const spinPer = new Array(nOrb).fill(0);

  let remaining = s.occ;
  for (let i = 0; i < nOrb && remaining > 0; i++) { spinPer[i] += 0.5; occPer[i] += 1; remaining--; }
  for (let i = 0; i < nOrb && remaining > 0; i++) { spinPer[i] -= 0.5; occPer[i] += 1; remaining--; }

  return {
    S: spinPer.reduce((a, b) => a + b, 0),
    ML: mls.reduce((acc, m, i) => acc + m * occPer[i], 0),
  };
}

function formatJ(J: number): string {
  return J === Math.trunc(J) ? String(J) : `${Math.round(2 * J)}/2`;
}

/**
 * Ground-state term symbol {2S+1}L_J from Hund's three rules. Multiple open
 * subshells (Gd is the case that forces this) are coupled high-spin by summing
 * their S and M_L contributions.
 */
export function hundTerm(cfg: Subshell[]): string {
  const open = openShells(cfg);
  if (open.length === 0) return "1S_0";

  let totalS = 0;
  let totalML = 0;
  for (const s of open) {
    const { S, ML } = couple(s);
    totalS += S;
    totalML += ML;
  }

  const L = Math.abs(Math.round(totalML));
  const mult = Math.round(2 * totalS + 1);

  // Hund's third rule: J = |L - S| below half filling, L + S above.
  const occ = open.reduce((a, s) => a + s.occ, 0);
  const cap = open.reduce((a, s) => a + subshellCapacity(s.l), 0);
  const J = occ <= cap / 2 ? Math.abs(L - totalS) : L + totalS;

  const letter = L < S_LABELS.length ? S_LABELS[L] : `[${L}]`;
  return `${mult}${letter}_${formatJ(J)}`;
}

/**
 * Valence shell: the subshells of the highest occupied principal number,
 * together with any open inner subshell (a partly-filled d or f still bonds).
 */
export function valenceShell(cfg: Subshell[]): Subshell[] {
  const nMax = Math.max(...cfg.map((s) => s.n));
  const outer = cfg.filter((s) => s.n === nMax);
  const innerOpen = cfg.filter((s) => s.n < nMax && s.occ < subshellCapacity(s.l) && s.occ > 0);
  return [...innerOpen, ...outer];
}

/**
 * Valence-shell occupancy and capacity.
 *
 * Capacity is the duet for n=1 and the octet for s+p valence shells; where an
 * open d shell participates it contributes its own capacity. This is the
 * closure target that `vacancy` counts down to.
 */
export function valenceCounts(cfg: Subshell[]): { qv: number; capV: number } {
  const nMax = Math.max(...cfg.map((s) => s.n));
  if (nMax === 1) {
    const occ = cfg.find((s) => s.n === 1 && s.l === 0)?.occ ?? 0;
    return { qv: occ, capV: 2 };
  }
  const sp = cfg.filter((s) => s.n === nMax && s.l <= 1);
  const qv = sp.reduce((a, s) => a + s.occ, 0);
  return { qv, capV: 8 };
}

/** The full derived description of one atom, from Z alone. */
export interface DerivedAtom {
  Z: number;
  symbol: string;
  config: Subshell[];
  configStr: string;
  term: string;
  qv: number;
  capV: number;
  vacancy: number;
  valence: number;
  period: number;
  group: number | null;
  exception: boolean;
}

/**
 * Period and group read off the derived configuration rather than from a
 * table: the period is the highest principal number, the group follows from
 * where the valence electrons sit.
 */
function periodGroup(cfg: Subshell[]): { period: number; group: number | null } {
  // The period is the row of the table, which is not always the highest
  // principal number present: palladium's ground state is [Kr]4d10 with an
  // empty 5s, yet it sits in period 5. Taking max(n) alone reports 4 and then
  // reads the d count off the wrong shell. An (n-1)d subshell that is open or
  // just-filled belongs to row n, so the period is at least that.
  const nMax = Math.max(...cfg.map((s) => s.n));
  const dLate = cfg.filter((s) => s.l === 2 && s.occ > 0).map((s) => s.n + 1);
  const period = Math.max(nMax, ...(dLate.length ? dLate : [nMax]));
  const outer = cfg.filter((s) => s.n === period);
  const s = outer.find((x) => x.l === 0)?.occ ?? 0;
  const p = outer.find((x) => x.l === 1)?.occ ?? 0;
  const d = cfg.find((x) => x.n === period - 1 && x.l === 2)?.occ ?? 0;
  const f = cfg.find((x) => x.n === period - 2 && x.l === 3)?.occ ?? 0;

  if (f > 0 && f < 14) return { period, group: null };       // lanthanide / actinide
  // Transition block. The count is s+d whether or not the ground state moved
  // an electron out of s: copper is 3d10 4s1, and s+d = 11 is still group 11.
  // Testing d>0 alone (rather than d<10) is what keeps the filled-d exceptions
  // — Cu, Ag, Au, Pd — in their real groups instead of reporting them as
  // alkali metals on the strength of a lone s electron.
  if (d > 0 && p === 0) return { period, group: Math.min(s + d, 12) };
  // p block spans groups 13..18, so a valence count of s+p (2..8) maps by +10:
  // carbon's 2s2 2p2 gives 4 -> group 14.
  if (p > 0) return { period, group: 10 + s + p };
  // Helium is the one placement the valence count cannot give: its 1s2 is an
  // s-block configuration, but a full duet is a closed shell, so it belongs
  // with the noble gases. The exception is chemical, not arithmetic.
  if (period === 1 && s === 2) return { period, group: 18 };
  return { period, group: s };                                // s block
}

export function deriveAtom(Z: number): DerivedAtom {
  if (!Number.isInteger(Z) || Z < 1) {
    throw new Error(`cut: atomic number must be a positive integer (got ${Z})`);
  }
  if (Z > MAX_Z) {
    throw new Error(`cut: Z=${Z} is beyond the named elements (1..${MAX_Z})`);
  }
  const config = deriveConfiguration(Z);
  const { qv, capV } = valenceCounts(config);
  const vacancy = Math.max(capV - qv, 0);
  // Bonding capacity: an atom shares the smaller of its holes and its
  // available electrons, so valence peaks at half filling.
  const valence = Math.min(vacancy, capV - vacancy);
  const { period, group } = periodGroup(config);

  return {
    Z,
    symbol: symbolOf(Z),
    config,
    configStr: configToString(config),
    term: hundTerm(config),
    qv, capV, vacancy, valence, period, group,
    exception: isAufbauException(Z),
  };
}
