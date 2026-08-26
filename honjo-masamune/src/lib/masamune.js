/**
 * Masamune in the browser: SMILES to contact graph, with provenance.
 *
 * A port of the Python reference (honjo-py/hjm/masamune). It exists so
 * that .msm plans execute client-side, the way .hnj programs already
 * do, rather than requiring a second local process.
 *
 * The port is checked against the reference by
 * scripts/check-masamune.mjs, which compares the supplied fraction,
 * atom and contact counts, and verdict labels for every structure in
 * the validation corpus. If the two drift apart that script fails.
 *
 * What this deliberately does NOT do: repair records, infer geometry,
 * or guess at anything the source did not state. Every element the
 * reader adds is tagged SUPPLIED and names the convention that added
 * it.
 */

/* ------------------------------------------------------------------ */
/*  Provenance                                                        */
/* ------------------------------------------------------------------ */

/**
 * Provenance tags, ordered so that composition is a maximum: a value
 * derived from a supplied input can never be stated.
 */
export const PROV = { STATED: 0, SUPPLIED: 1 };
export const PROV_NAME = ["stated", "supplied"];

export function provJoin(tags) {
  let m = PROV.STATED;
  for (const t of tags) if (t > m) m = t;
  return m;
}

/* ------------------------------------------------------------------ */
/*  Chemistry tables                                                  */
/* ------------------------------------------------------------------ */

const SYMBOLS = (
  "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca " +
  "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
).split(" ");

export const Z_OF = Object.fromEntries(SYMBOLS.map((s, i) => [s, i + 1]));
export const SYMBOL_OF = Object.fromEntries(SYMBOLS.map((s, i) => [i + 1, s]));

const ORGANIC_SUBSET = new Set(["B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]);
const AROMATIC_LOWER = new Set(["b", "c", "n", "o", "p", "s"]);
const BOND_CELLS = { "-": 1, "=": 2, "#": 3, ":": 1, "/": 1, "\\": 1 };

/** Target valences for the organic subset, lowest first. */
const ORGANIC_SUBSET_VALENCE = {
  5: [3], 6: [4], 7: [3, 5], 8: [2], 15: [3, 5],
  16: [2, 4, 6], 9: [1], 17: [1], 35: [1], 53: [1],
};

export const CONV_IMPLICIT_H = "smiles:organic-subset-implicit-hydrogen";
export const CONV_AROMATIC = "smiles:aromatic-lowercase-delocalised";
export const CONV_DEFAULT_BOND = "smiles:adjacent-atoms-single-bond";
export const CONV_MEDIUM = "medium-residual";

/* ------------------------------------------------------------------ */
/*  Capability                                                        */
/* ------------------------------------------------------------------ */

export const FEATURES = [
  "element", "connectivity", "cellcount", "delocalisation", "charge",
  "isotope", "hcount", "stereo", "coords3d", "conformer", "provenance",
];

/**
 * What each reader can faithfully extract. Under-declaring is safe;
 * over-declaring is unsound and silent. `stereo` is declared by no
 * format: the reader parses stereo tokens but builds no descriptors.
 */
export const CAPABILITY = {
  smiles: ["cellcount", "charge", "connectivity", "delocalisation", "element", "hcount", "isotope"],
  molfile: ["cellcount", "charge", "connectivity", "coords3d", "element", "isotope"],
  sdf: ["cellcount", "charge", "conformer", "connectivity", "coords3d", "element", "isotope", "provenance"],
  xyz: ["coords3d", "element"],
  inchi: [],
};

export function capability(fmt) {
  return CAPABILITY[fmt] || [];
}

export function missing(fmt, required) {
  const have = new Set(capability(fmt));
  return [...required].filter((f) => !have.has(f)).sort();
}

export function knownFormat(fmt) {
  return Object.prototype.hasOwnProperty.call(CAPABILITY, fmt);
}

/* ------------------------------------------------------------------ */
/*  Verdicts                                                          */
/* ------------------------------------------------------------------ */

export const LABEL = {
  TRANSLATED: "translated",
  UNSUPPORTED: "unsupported",
  MALFORMED: "malformed",
  EMPTY: "empty",
  UNDERDETERMINED: "underdetermined",
  INCOMPLETE: "incomplete",
};

/** Only these labels may carry a value. */
const VALUE_BEARING = new Set([LABEL.TRANSLATED]);

/**
 * A labelled outcome. The invariant that a failure carries no value is
 * enforced here, at construction, rather than checked afterwards.
 */
export function verdict(label, payload = {}, value = null) {
  if (value !== null && !VALUE_BEARING.has(label)) {
    throw new Error(`label ${label} must not carry a value`);
  }
  return { label, payload, value, ok: VALUE_BEARING.has(label) };
}

/* ------------------------------------------------------------------ */
/*  SMILES                                                            */
/* ------------------------------------------------------------------ */

export class SmilesError extends Error {
  constructor(msg, pos) {
    super(msg);
    this.pos = pos;
    this.msg = msg;
  }
}

function elemZ(sym) {
  const up = sym.length > 1
    ? sym[0].toUpperCase() + sym.slice(1).toLowerCase()
    : sym.toUpperCase();
  const z = Z_OF[up];
  if (!z) throw new SmilesError(`unknown element '${sym}'`, 0);
  return z;
}

/**
 * Parse a SMILES string into atoms, bonds and implicit-H counts.
 *
 * Deliberately narrow: the organic subset, brackets, ring closures,
 * branches and the four bond symbols. Anything else raises rather than
 * being guessed at.
 */
export function parseSmiles(text) {
  const s = text.trim();
  if (!s) throw new SmilesError("empty input", 0);

  const atoms = [];
  const bonds = [];
  const ringOpen = new Map(); // digit -> {idx, bond, pos}
  const stack = [];
  let prev = null;
  let pendingBond = null;
  let sawStereo = false;
  let i = 0;

  const connect = (a, b, bsym) => {
    if (a === null || b === null) return;
    const stated = bsym !== null && bsym !== undefined;
    const sym = stated ? bsym : "-";
    const aromatic = sym === ":" ||
      (!stated && atoms[a].aromatic && atoms[b].aromatic);
    bonds.push({
      a, b,
      cells: BOND_CELLS[sym] ?? 1,
      stated: stated && sym !== ":",
      aromatic,
      symbol: sym,
    });
  };

  while (i < s.length) {
    const c = s[i];

    if (c === "(") { stack.push(prev); i += 1; continue; }
    if (c === ")") {
      if (!stack.length) throw new SmilesError("unbalanced ')'", i);
      prev = stack.pop();
      i += 1;
      continue;
    }
    if (c in BOND_CELLS) {
      if (c === "/" || c === "\\") sawStereo = true;
      pendingBond = c;
      i += 1;
      continue;
    }
    if (c === ".") { prev = null; pendingBond = null; i += 1; continue; }

    // ring closure digit
    if (/[0-9]/.test(c) || c === "%") {
      let tag;
      if (c === "%") {
        tag = s.slice(i + 1, i + 3);
        i += 3;
      } else {
        tag = c;
        i += 1;
      }
      if (prev === null) throw new SmilesError("ring bond with no atom", i);
      if (ringOpen.has(tag)) {
        const o = ringOpen.get(tag);
        ringOpen.delete(tag);
        connect(o.idx, prev, o.bond ?? pendingBond);
      } else {
        ringOpen.set(tag, { idx: prev, bond: pendingBond, pos: i });
      }
      pendingBond = null;
      continue;
    }

    // bracket atom
    if (c === "[") {
      const end = s.indexOf("]", i);
      if (end < 0) throw new SmilesError("unterminated '['", i);
      const a = parseBracket(s.slice(i + 1, end), i);
      a.idx = atoms.length;
      atoms.push(a);
      connect(prev, a.idx, pendingBond);
      prev = a.idx;
      pendingBond = null;
      i = end + 1;
      continue;
    }

    // organic-subset atom
    const two = s.slice(i, i + 2);
    let sym = null;
    if (ORGANIC_SUBSET.has(two)) sym = two;
    else if (ORGANIC_SUBSET.has(c) || AROMATIC_LOWER.has(c)) sym = c;

    if (sym === null) {
      if (c === "@") { sawStereo = true; i += 1; continue; }
      throw new SmilesError(`unexpected character '${c}'`, i);
    }

    const aromatic = AROMATIC_LOWER.has(sym);
    // Normalise to the element's own casing: 'c' -> 'C', 'Cl' -> 'Cl'.
    // Upper-casing a two-letter symbol would reject Cl and Br, which
    // are in the subset.
    const upper = sym.length > 1
      ? sym[0].toUpperCase() + sym.slice(1).toLowerCase()
      : sym.toUpperCase();
    if (!ORGANIC_SUBSET.has(upper)) {
      throw new SmilesError(`${sym} outside organic subset; use brackets`, i);
    }

    const a = {
      idx: atoms.length, z: elemZ(upper), aromatic,
      bracket: false, charge: 0, isotope: null, explicitH: 0,
    };
    atoms.push(a);
    connect(prev, a.idx, pendingBond);
    prev = a.idx;
    pendingBond = null;
    i += sym.length;
  }

  if (ringOpen.size) {
    const [tag, o] = [...ringOpen.entries()][0];
    throw new SmilesError(`unclosed ring bond '${tag}'`, o.pos);
  }
  if (stack.length) throw new SmilesError("unbalanced '('", s.length);

  const res = {
    atoms, bonds, implicitH: {}, sawStereoToken: sawStereo,
    aromaticRings: [],
  };
  fillImplicitH(res);
  res.aromaticRings = aromaticSystems(res);
  return res;
}

function parseBracket(body, pos) {
  let j = 0;
  let iso = "";
  while (j < body.length && /[0-9]/.test(body[j])) { iso += body[j]; j += 1; }

  let sym = "";
  if (j < body.length && /[A-Za-z]/.test(body[j])) {
    sym += body[j];
    j += 1;
    if (j < body.length && /[a-z]/.test(body[j])) {
      const two = sym + body[j];
      if (Z_OF[two[0].toUpperCase() + two.slice(1)]) { sym = two; j += 1; }
    }
  }
  if (!sym) throw new SmilesError("bracket atom with no element", pos);

  const aromatic = /^[a-z]/.test(sym);
  let explicitH = 0;
  let charge = 0;

  while (j < body.length) {
    const ch = body[j];
    if (ch === "@") { j += 1; continue; }        // stereo, not built
    if (ch === "H") {
      j += 1;
      let n = "";
      while (j < body.length && /[0-9]/.test(body[j])) { n += body[j]; j += 1; }
      explicitH = n ? parseInt(n, 10) : 1;
      continue;
    }
    if (ch === "+" || ch === "-") {
      const sign = ch === "+" ? 1 : -1;
      j += 1;
      let n = "";
      while (j < body.length && /[0-9]/.test(body[j])) { n += body[j]; j += 1; }
      if (n) charge = sign * parseInt(n, 10);
      else {
        let run = 1;
        while (j < body.length && body[j] === ch) { run += 1; j += 1; }
        charge = sign * run;
      }
      continue;
    }
    j += 1;
  }

  return {
    idx: -1, z: elemZ(sym), aromatic, bracket: true,
    charge, isotope: iso ? parseInt(iso, 10) : null, explicitH,
  };
}

/** Supply hydrogens for organic-subset atoms that did not state them. */
function fillImplicitH(res) {
  const used = {};
  for (const a of res.atoms) used[a.idx] = 0;
  for (const b of res.bonds) {
    // an aromatic bond contributes one sigma cell for valence counting
    const c = b.aromatic ? 1 : b.cells;
    used[b.a] += c;
    used[b.b] += c;
  }
  for (const a of res.atoms) {
    if (a.bracket) { res.implicitH[a.idx] = 0; continue; }
    const targets = ORGANIC_SUBSET_VALENCE[a.z];
    if (!targets) { res.implicitH[a.idx] = 0; continue; }
    const need = used[a.idx] + (a.aromatic ? 1 : 0);
    const target = targets.find((t) => t >= need) ?? targets[targets.length - 1];
    res.implicitH[a.idx] = Math.max(0, target - need);
  }
}

/** Connected components of the aromatic-bond subgraph, size >= 3. */
function aromaticSystems(res) {
  const adj = new Map();
  for (const b of res.bonds) {
    if (!b.aromatic) continue;
    if (!adj.has(b.a)) adj.set(b.a, new Set());
    if (!adj.has(b.b)) adj.set(b.b, new Set());
    adj.get(b.a).add(b.b);
    adj.get(b.b).add(b.a);
  }
  const seen = new Set();
  const out = [];
  for (const start of [...adj.keys()].sort((x, y) => x - y)) {
    if (seen.has(start)) continue;
    const comp = [];
    const stack = [start];
    while (stack.length) {
      const n = stack.pop();
      if (seen.has(n)) continue;
      seen.add(n);
      comp.push(n);
      for (const m of adj.get(n)) if (!seen.has(m)) stack.push(m);
    }
    if (comp.length >= 3) out.push(comp.sort((x, y) => x - y));
  }
  return out;
}

/* ------------------------------------------------------------------ */
/*  Contact graph                                                     */
/* ------------------------------------------------------------------ */

export const MEDIUM = "__medium__";

/**
 * A weighted graph on the atoms with a distinguished medium vertex
 * adjacent to every atom.
 */
export function makeGraph(floor = 1.0) {
  return { floor, atoms: {}, contacts: {}, delocs: {}, meta: {}, n: 0 };
}

export function addAtom(g, atom) {
  const key = `a${g.n}`;
  g.n += 1;
  g.atoms[key] = { ...atom, key };
  return key;
}

export function addContact(g, c) {
  const key = [c.u, c.v].sort().join("|");
  g.contacts[key] = {
    ...c,
    key,
    isMedium: c.u === MEDIUM || c.v === MEDIUM,
  };
}

/**
 * Fraction of the graph's OWN elements that were supplied.
 *
 * Medium edges are excluded from the denominator: no record states
 * them, so counting them would make the statistic a property of the
 * target representation rather than of the source.
 */
export function suppliedFraction(g) {
  const elems = [
    ...Object.values(g.atoms),
    ...Object.values(g.contacts).filter((c) => !c.isMedium),
  ];
  if (!elems.length) return 0;
  const n = elems.filter((e) => e.prov === PROV.SUPPLIED).length;
  return n / elems.length;
}

/* ------------------------------------------------------------------ */
/*  Translation                                                       */
/* ------------------------------------------------------------------ */

function bondWeight(cells, floor) {
  return Math.max(floor * cells, floor);
}

/**
 * Translate a SMILES record. Returns a verdict, never a bare graph.
 *
 * Clause order matters and is the same as the reference:
 *   (V3) capability, decidable without reading anything
 *   (V2) empty record — checked before parsing, because the parser
 *        raises on empty input and would otherwise report MALFORMED
 *   (V1) malformed
 *   (V4) requested but underdetermined
 */
export function translateSmiles(text, {
  required = ["element", "connectivity"],
  floor = 1.0,
  sourceName = "<smiles>",
} = {}) {
  const req = [...required];

  const miss = missing("smiles", req);
  if (miss.length) {
    return verdict(LABEL.UNSUPPORTED, {
      missing_features: miss,
      have: capability("smiles"),
    });
  }

  if (!text.trim()) {
    return verdict(LABEL.EMPTY, { source: sourceName, certified: true });
  }

  let p;
  try {
    p = parseSmiles(text);
  } catch (err) {
    if (err instanceof SmilesError) {
      return verdict(LABEL.MALFORMED, {
        position: `${sourceName}:${err.pos}`,
        detail: err.msg,
      });
    }
    throw err;
  }

  if (!p.atoms.length) {
    return verdict(LABEL.EMPTY, { source: sourceName, certified: true });
  }

  if (req.includes("stereo") && p.sawStereoToken) {
    return verdict(LABEL.UNDERDETERMINED, {
      element: "stereo",
      readings: ["reader parses stereo tokens but assigns no descriptor"],
    });
  }

  const g = makeGraph(floor);
  g.meta = { source: sourceName, source_format: "smiles", input: text };

  const heavyKey = {};
  for (const a of p.atoms) {
    heavyKey[a.idx] = addAtom(g, {
      z: a.z, charge: a.charge, isotope: a.isotope,
      prov: PROV.STATED, convention: null,
    });
  }

  // Aromatic systems: one delocalised block, never per-bond counts.
  const aromBond = new Set();
  p.aromaticRings.forEach((comp, sysid) => {
    const inComp = new Set(comp);
    const nBonds = p.bonds.filter(
      (b) => b.aromatic && inComp.has(b.a) && inComp.has(b.b)
    ).length;
    g.delocs[sysid] = {
      members: comp.map((i) => heavyKey[i]),
      sigma_cells: nBonds,
      delocalised_cells: Math.max(0, Math.floor(comp.length / 2)),
      total_cells: nBonds + Math.max(0, Math.floor(comp.length / 2)),
      provenance: PROV_NAME[PROV.SUPPLIED],
      convention: CONV_AROMATIC,
      // deliberately absent: a per-bond count would state a pairwise
      // fact the delocalised system does not determine
      per_bond_cells: null,
    };
    for (const b of p.bonds) {
      if (!b.aromatic || !inComp.has(b.a) || !inComp.has(b.b)) continue;
      aromBond.add(`${Math.min(b.a, b.b)}-${Math.max(b.a, b.b)}`);
      addContact(g, {
        u: heavyKey[b.a], v: heavyKey[b.b],
        weight: bondWeight(1, floor), cells: 1,
        deloc_id: sysid, prov: PROV.SUPPLIED, convention: CONV_AROMATIC,
      });
    }
  });

  for (const b of p.bonds) {
    const pair = `${Math.min(b.a, b.b)}-${Math.max(b.a, b.b)}`;
    if (aromBond.has(pair)) continue;
    addContact(g, {
      u: heavyKey[b.a], v: heavyKey[b.b],
      weight: bondWeight(b.cells, floor), cells: b.cells,
      deloc_id: null,
      prov: b.stated ? PROV.STATED : PROV.SUPPLIED,
      convention: b.stated ? null : CONV_DEFAULT_BOND,
    });
  }

  // implicit and bracket hydrogens
  for (const a of p.atoms) {
    const nImpl = p.implicitH[a.idx] || 0;
    for (let k = 0; k < nImpl; k += 1) {
      const hk = addAtom(g, {
        z: 1, charge: 0, isotope: null,
        prov: PROV.SUPPLIED, convention: CONV_IMPLICIT_H,
      });
      addContact(g, {
        u: heavyKey[a.idx], v: hk, weight: bondWeight(1, floor), cells: 1,
        deloc_id: null, prov: PROV.SUPPLIED, convention: CONV_IMPLICIT_H,
      });
    }
    for (let k = 0; k < a.explicitH; k += 1) {
      // a bracket H is written in the record, so it is stated
      const hk = addAtom(g, {
        z: 1, charge: 0, isotope: null, prov: PROV.STATED, convention: null,
      });
      addContact(g, {
        u: heavyKey[a.idx], v: hk, weight: bondWeight(1, floor), cells: 1,
        deloc_id: null, prov: PROV.STATED, convention: null,
      });
    }
  }

  // medium edges from unshared regions
  const committed = {};
  for (const k of Object.keys(g.atoms)) committed[k] = 0;
  for (const c of Object.values(g.contacts)) {
    if (c.isMedium) continue;
    committed[c.u] = (committed[c.u] || 0) + c.cells;
    committed[c.v] = (committed[c.v] || 0) + c.cells;
  }
  for (const [k, at] of Object.entries(g.atoms)) {
    const cap = at.z === 1 ? 1 : 4;
    const unshared = Math.max(0, cap - (committed[k] || 0));
    addContact(g, {
      u: k, v: MEDIUM,
      weight: Math.max(floor * unshared, floor),
      cells: 0, deloc_id: null,
      prov: PROV.SUPPLIED, convention: CONV_MEDIUM,
    });
  }

  return verdict(
    LABEL.TRANSLATED,
    { supplied_fraction: round6(suppliedFraction(g)) },
    g
  );
}

function round6(x) {
  return Math.round(x * 1e6) / 1e6;
}

export function translate(fmt, text, opts = {}) {
  if (!knownFormat(fmt)) {
    return verdict(LABEL.UNSUPPORTED, {
      missing_features: [...(opts.required || ["element"])],
      have: [],
    });
  }
  if (fmt === "smiles") return translateSmiles(text, opts);
  return verdict(LABEL.UNSUPPORTED, {
    missing_features: missing(fmt, opts.required || ["element"]),
    have: capability(fmt),
  });
}
