/**
 * Sources, and the five adapters that resolve against them.
 *
 * CONSTRAINT INHERITED FROM THE REFERENCE, AND BINDING HERE:
 *   "Every adapter resolves against a local fixture or a local engine. No
 *    adapter performs network I/O, and none may be added that does: the
 *    prototype's claims are properties of the compiler, and a live service
 *    can neither confirm nor refute them."
 *
 * In the browser that constraint is sharper, not weaker. This module contains
 * no `fetch`, no `XMLHttpRequest`, and no dynamic import. Everything resolves
 * against the embedded fixture. A page that phoned home for its numbers would
 * be demonstrating someone else's engine, not this compiler.
 *
 * Ported from hfq/adapters.py and hfq/biocat.py.
 */

import { ResultSet, Refusal } from "./model.js";

/* ------------------------------------------------------------------ *
 * Capability requirements per abstract predicate.
 * ------------------------------------------------------------------ */

/**
 * The base table, plus the biocatalysis delta that `hfq/biocat.py` installs at
 * import time. Merged here because a browser module has no import side effects
 * worth relying on, and a table that is silently mutated by whichever module
 * loaded first is a bug waiting for a bundler change.
 */
export const PREDICATE_FEATURES = {};

export function loadPredicateFeatures(table) {
  for (const [k, v] of Object.entries(table)) {
    PREDICATE_FEATURES[k] = new Set(v);
  }
}

/**
 * `bind` is not merely additive here: its presence REMOVES `pattern` for the
 * three predicates that can be answered by scanning a supplied key set rather
 * than by matching a graph shape. That is why this cannot be a table lookup
 * plus a fixed increment.
 */
export function biocatRequiredFeatures(request) {
  const base = PREDICATE_FEATURES[request.predicate];
  if (base === undefined) {
    throw new Refusal(`unknown abstract predicate '${request.predicate}'`);
  }
  const req = new Set(base);
  if (request.bindings && request.bindings.length) {
    req.add("bind");
    if (
      request.predicate === "matching" ||
      request.predicate === "excluding" ||
      request.predicate === "sequence_of"
    ) {
      req.delete("pattern");
    }
  }
  return req;
}

/**
 * Ask the adapter first. A table keyed on the predicate alone cannot express
 * that `matching` needs `pattern` against a graph source and does not against
 * a sequence source.
 */
export function resolveFeatures(adapter, request) {
  if (adapter && typeof adapter.requiredFeatures === "function") {
    return adapter.requiredFeatures(request);
  }
  return biocatRequiredFeatures(request);
}

/* ------------------------------------------------------------------ *
 * Base adapter.
 * ------------------------------------------------------------------ */

export class Adapter {
  constructor({ name, namespace, capabilities, snapshot = null }) {
    this.name = name;
    this.namespace = namespace;
    this.capabilities = new Set(capabilities || []);
    this.snapshot = snapshot;
    this.requestsIssued = 0;
    this.lastLowered = null;
  }

  /** The declared capability set. Under-declaring is safe; over-declaring is
   *  unsound AND INVISIBLE -- nothing in this file verifies it. */
  declares(f) {
    return this.capabilities.has(f);
  }

  lower() {
    throw new Refusal(`${this.name} cannot lower a request`);
  }

  extract() {
    throw new Refusal(`${this.name} cannot extract a result`);
  }

  cost() {
    return 1.0;
  }

  /** Lower, count the request, extract. Order matters: the counter increments
   *  even when extraction refuses, because the request was formed. */
  evaluate(request, inputs) {
    const concrete = this.lower(request, inputs);
    this.lastLowered = concrete;
    this.requestsIssued += 1;
    return this.extract(concrete, request, inputs);
  }
}

/* ------------------------------------------------------------------ *
 * Graph pattern adapter -- the base for RXN and PROV.
 * ------------------------------------------------------------------ */

function isLiteralArg(a) {
  return typeof a === "string" && !a.startsWith("?");
}

export class GraphPatternAdapter extends Adapter {
  constructor(opts) {
    super(opts);
    this.triples = opts.triples || [];
    this.paths = opts.paths || {};
    this.prefixes = opts.prefixes || {};
  }

  seedsFor(request, inputs) {
    const seeds = new Set();
    let any = false;
    for (const a of request.args) {
      if (isLiteralArg(a)) {
        seeds.add(a);
        any = true;
      }
    }
    for (const [, planVar] of request.bindings) {
      any = true;
      for (const m of inputs[planVar].identifiers()) seeds.add(m);
    }
    return any ? seeds : null;
  }

  lower(request, inputs) {
    const path = this.paths[request.predicate] || request.predicate;
    const body = ["SELECT DISTINCT ?o WHERE {"];
    const named = request.args.filter(isLiteralArg);
    if (named.length) {
      body.push("  VALUES ?s { " + named.map((n) => `<${n}>`).join(" ") + " }");
    }
    for (const [v, planVar] of request.bindings) {
      const members = inputs[planVar].identifiers();
      body.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
      body.push("  FILTER(?s = " + v + ")");
    }
    body.push(`  ?s <${path}> ?o .`);
    body.push("}");
    return body.join("\n");
  }

  /**
   * FORWARD traversal. Seeds restrict the SUBJECT; the OBJECT is returned.
   * The literal filter is applied as `s not in literals` ONLY when seeds is
   * null -- it filters on the subject. Reading it as an object filter is the
   * root of the whole inverse-predicate family of defects.
   */
  extract(concrete, request, inputs) {
    const path = this.paths[request.predicate] || request.predicate;
    const seeds = this.seedsFor(request, inputs);
    const pairs = [];
    for (const [s, p, o] of this.triples) {
      if (p !== path) continue;
      if (seeds && !seeds.has(s)) continue;
      pairs.push([o, { _from: s, _via: path }]);
    }
    return ResultSet.of(this.namespace, pairs);
  }

  cost(request, inputs) {
    let n = 0;
    for (const [, planVar] of request.bindings) n += inputs[planVar].size;
    return Math.max(1, n);
  }
}

/* ------------------------------------------------------------------ *
 * Identity: shared by two classes with no common ancestor below
 * GraphPatternAdapter. Pasting the branch into both is how the
 * `measured_on` defect happened, so it lives once, here.
 *
 * Req is `pattern` + `bind` and deliberately NOT `path`: no edge is walked.
 * ------------------------------------------------------------------ */

const Identity = {
  lower(request, inputs) {
    const names = [...new Set(request.args.filter(isLiteralArg))].sort();
    const lines = ["SELECT DISTINCT ?s WHERE {"];
    if (names.length) {
      lines.push("  VALUES ?s { " + names.map((n) => `<${n}>`).join(" ") + " }");
    }
    for (const [v, planVar] of request.bindings) {
      const members = inputs[planVar].identifiers();
      lines.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
      lines.push("  FILTER(?s = " + v + ")");
    }
    lines.push("}");
    return lines.join("\n");
  },

  /**
   * The intersection of what is NAMED with what is HELD.
   *
   * Naming a thing is not evidence the corpus has it. A query language in
   * which putting a URI in a VALUES clause makes it appear in the output has
   * quietly conflated the two, and this is the step that refuses to.
   */
  extract(self, request, inputs) {
    const wanted = new Set(request.args.filter(isLiteralArg));
    const held = new Set();
    for (const [, planVar] of request.bindings) {
      for (const m of inputs[planVar].identifiers()) held.add(m);
    }
    const pairs = [];
    for (const ident of [...wanted].filter((x) => held.has(x)).sort()) {
      const row = request.bindings.length
        ? { ...inputs[request.bindings[0][1]].attrs(ident) }
        : {};
      row._identified = true;
      pairs.push([ident, row]);
    }
    return ResultSet.of(self.namespace, pairs);
  },
};

/* ------------------------------------------------------------------ *
 * FilteringGraphAdapter -- RXN.
 * ------------------------------------------------------------------ */

/**
 * Predicates whose extraction reaches the SUBJECT from the object -- that is,
 * that traverse a stored edge BACKWARDS.
 *
 * Named in one place so that a source cannot implement the lowering for one
 * direction and the extraction for the other. Getting a name wrong here does
 * not fail loudly: the forward extraction applies its literal filter to the
 * subject position, no subject is ever named `CHEBI:catechol`, and the step
 * returns the empty set. An empty result meaning "you walked the edge
 * backwards" is indistinguishable at the emit boundary from one meaning "no
 * reaction consumes this" -- which is cor:onebit arriving from INSIDE the
 * executor, and it is worse there, because the framework's whole claim is
 * that it can tell those two apart.
 *
 * These six are the reference set (hfq/biocat.py). Every one of them is
 * exercised by a plan on this page, and each declares `path` in its Req
 * (see `predicate_features` in the fixture) precisely so that a source which
 * cannot traverse backwards is refused BY NAME before contact rather than
 * answering nothing.
 */
const INVERSE_PREDICATES = new Set([
  "evaluations_of",
  "producers_of",
  "consumers_of",
  "enzymes_from",
  "catalysed_reactions",
  "measured_on",
]);

const COMPARATORS = {
  "==": (a, b) => a === b,
  "!=": (a, b) => a !== b,
  "<": (a, b) => a < b,
  ">": (a, b) => a > b,
  "<=": (a, b) => a <= b,
  ">=": (a, b) => a >= b,
};

export class FilteringGraphAdapter extends GraphPatternAdapter {
  constructor(opts) {
    super(opts);
    this.literalAttr = opts.literal_attr || "_literal";
    // The fixture constructs RXN without these; an absent mapping is empty,
    // never a sentinel masquerading as data.
    this.literals = opts.literals || {};
    this.ranks = opts.ranks || {};
    if (!opts.capabilities) {
      this.capabilities = new Set([
        "pattern", "path", "bind", "filter", "agg", "regex", "neg", "order",
      ]);
    }
  }

  /* --- argument accessors. A missing argument is a Refusal, never a silent
     default: a `matching` with no pattern would otherwise match everything. */

  typeArgs(request) {
    const a = request.args.filter(isLiteralArg);
    if (!a.length) throw new Refusal("typed_as requires a type argument");
    return [a[0], a.length > 1 ? a[1] : null];
  }

  patternArg(request) {
    const a = request.args.filter(isLiteralArg);
    if (!a.length) {
      throw new Refusal(`${request.predicate} requires a pattern argument`);
    }
    return a[0];
  }

  rankArg(request) {
    const a = request.args.filter(isLiteralArg);
    if (!a.length) throw new Refusal("ranked_by requires a key argument");
    return a[0];
  }

  restrictionArgs(request) {
    const a = request.args.filter((x) => isLiteralArg(x) || typeof x === "number");
    if (a.length < 3) {
      throw new Refusal("restricted requires (attribute, operator, value)");
    }
    const op = String(a[1]);
    if (!(op in COMPARATORS)) {
      throw new Refusal(`restricted: unknown operator '${op}'`);
    }
    return [a[0], op, a[2]];
  }

  literalOf(v) {
    return typeof v === "string" ? `"${v}"` : String(v);
  }

  lowerInverse(request, inputs) {
    const path = this.paths[request.predicate] || request.predicate;
    const targets = [...new Set(request.args.filter(isLiteralArg))].sort();
    const lines = ["SELECT DISTINCT ?s WHERE {"];
    if (targets.length) {
      lines.push("  VALUES ?o { " + targets.map((t) => `<${t}>`).join(" ") + " }");
    }
    for (const [v, planVar] of request.bindings) {
      const members = inputs[planVar].identifiers();
      lines.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
      lines.push("  FILTER(?o = " + v + ")");
    }
    lines.push(`  ?s <${path}> ?o .`);
    lines.push("}");
    return lines.join("\n");
  }

  lower(request, inputs) {
    const pred = request.predicate;
    if (INVERSE_PREDICATES.has(pred)) return this.lowerInverse(request, inputs);
    if (pred === "identified_as") return Identity.lower.call(this, request, inputs);

    if (pred === "typed_as") {
      // A two-argument kind test lowers to a MINUS, not a conjunction: a
      // conjunction would demand BOTH kinds, while `extract` computes the
      // difference. The two forms disagree on every row that matters.
      const [want, forbid] = this.typeArgs(request);
      const lines = ["SELECT DISTINCT ?s WHERE {", `  ?s <rdf:type> <${want}> .`];
      if (forbid !== null) {
        lines.push(`  MINUS { ?s <rdf:type> <${forbid}> . }`);
      }
      lines.push("}");
      return lines.join("\n");
    }

    const base = super.lower(request, inputs);
    const body = base.split("\n").slice(0, -1); // drop the closing brace

    if (pred === "matching" || pred === "excluding") {
      const call = `REGEX(STR(?lit), "${this.patternArg(request)}")`;
      body.push(`  ?o <${this.literalAttr}> ?lit .`);
      body.push(pred === "matching" ? `  FILTER(${call})` : `  FILTER(!${call})`);
    } else if (pred === "restricted") {
      const [attr, op, val] = this.restrictionArgs(request);
      body.push(`  ?o <${attr}> ?v .`);
      body.push(`  FILTER(?v ${op} ${this.literalOf(val)})`);
    } else if (pred === "ranked_by") {
      body.push(`  ?o <${this.rankArg(request)}> ?rank .`);
    }
    body.push("}");
    if (pred === "ranked_by") body.push("ORDER BY DESC(?rank)");
    return body.join("\n");
  }

  baseExtract(request, inputs) {
    return GraphPatternAdapter.prototype.extract.call(this, "", request, inputs);
  }

  inverseExtract(request, inputs) {
    const path = this.paths[request.predicate] || request.predicate;
    const targets = new Set(request.args.filter(isLiteralArg));
    for (const [, planVar] of request.bindings) {
      for (const m of inputs[planVar].identifiers()) targets.add(m);
    }
    const pairs = [];
    for (const [s, p, o] of this.triples) {
      if (p === path && targets.has(o)) {
        pairs.push([s, { _reached: o, _via: path, _direction: "inverse" }]);
      }
    }
    return ResultSet.of(this.namespace, pairs);
  }

  extract(concrete, request, inputs) {
    const pred = request.predicate;

    if (pred === "count_of") {
      const inner = this.baseExtract(request, inputs);
      return ResultSet.of(this.namespace, [
        [`${this.namespace}:count`, { count: inner.size, _over: pred }],
      ]);
    }

    if (pred === "matching" || pred === "excluding") {
      const pattern = this.patternArg(request);
      let rx;
      try {
        rx = new RegExp(pattern);
      } catch (e) {
        // A malformed pattern is a refusal, not an empty answer. "No enzyme
        // lacks cysteine" and "your regex did not compile" must not arrive as
        // the same result.
        throw new Refusal(`${pred}: uncompilable pattern '${pattern}': ${e.message}`);
      }
      const keep =
        pred === "matching" ? (s) => rx.test(s) : (s) => !rx.test(s);
      const pairs = [];
      for (const [ident, row] of this.baseExtract(request, inputs).rows) {
        const lit = row[this.literalAttr] ?? this.literals[ident];
        // Dropped, not counted as a pass. A row with no literal was not
        // scanned, and reporting it as having survived the scan would be a
        // claim the corpus does not support.
        if (lit === undefined || lit === null) continue;
        if (keep(String(lit))) {
          pairs.push([ident, { ...row, _scanned: true, _predicate: pred }]);
        }
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "restricted") {
      const [attr, op, val] = this.restrictionArgs(request);
      const cmp = COMPARATORS[op];
      const pairs = [];
      for (const [ident, row] of this.baseExtract(request, inputs).rows) {
        if (!(attr in row)) continue;
        let ok;
        try {
          ok = cmp(row[attr], val);
        } catch {
          continue;
        }
        if (ok) pairs.push([ident, { ...row, _restricted: `${attr}${op}${val}` }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "ranked_by") {
      const key = this.rankArg(request);
      const rows = [...this.baseExtract(request, inputs).rows.entries()];
      rows.sort((a, b) => {
        const av = a[1][key] ?? this.ranks[a[0]] ?? 0.0;
        const bv = b[1][key] ?? this.ranks[b[0]] ?? 0.0;
        return bv - av;
      });
      return ResultSet.of(
        this.namespace,
        rows.map(([i, r], n) => [i, { ...r, _rank: n }])
      );
    }

    if (pred === "typed_as") {
      const [want, forbid] = this.typeArgs(request);
      const seeds = this.seedsFor(request, inputs);
      const pairs = [];
      for (const [s, p, o] of this.triples) {
        if (p !== "rdf:type" || o !== want) continue;
        if (seeds && !seeds.has(s)) continue;
        if (
          forbid !== null &&
          this.triples.some(
            ([s2, p2, o2]) => s2 === s && p2 === "rdf:type" && o2 === forbid
          )
        ) {
          continue;
        }
        pairs.push([s, { _type: want, _excluded: forbid }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "identified_as") return Identity.extract(this, request, inputs);
    if (INVERSE_PREDICATES.has(pred)) return this.inverseExtract(request, inputs);
    return this.baseExtract(request, inputs);
  }

  cost(request, inputs) {
    const base = super.cost(request, inputs);
    if (request.predicate === "matching" || request.predicate === "excluding") {
      return base + Math.max(1, Object.keys(this.literals).length);
    }
    return base;
  }
}

/* ------------------------------------------------------------------ *
 * ProvenanceAdapter -- PROV.
 * ------------------------------------------------------------------ */

/**
 * Kept EXPLICIT rather than inferred by "is the object a literal", because
 * whether a value is a literal is a fact about this fixture's serialisation
 * and not about what the question is asking.
 */
const SETTING_PREDICATES = ["buffer", "pH", "operator", "date", "wavelength"];

export class ProvenanceAdapter extends GraphPatternAdapter {
  constructor(opts) {
    super(opts);
    if (!opts.capabilities) {
      this.capabilities = new Set(["pattern", "bind", "filter", "path"]);
    }
  }

  lower(request, inputs) {
    const pred = request.predicate;

    if (pred === "evaluations_of") {
      const path = this.paths.evaluations_of || "evaluated";
      const targets = [...new Set(request.args.filter(isLiteralArg))].sort();
      const lines = ["SELECT DISTINCT ?a WHERE {"];
      if (targets.length) {
        lines.push("  VALUES ?t { " + targets.map((t) => `<${t}>`).join(" ") + " }");
      }
      for (const [v, planVar] of request.bindings) {
        const members = inputs[planVar].identifiers();
        lines.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
        lines.push("  FILTER(?t = " + v + ")");
      }
      lines.push(`  ?a <${path}> ?t .`);
      lines.push("}");
      return lines.join("\n");
    }

    if (pred === "settings_of") {
      // Every setting is OPTIONAL: an activity that recorded no wavelength is
      // still an activity, and an inner join would delete it.
      const lines = ["SELECT ?a ?p ?v WHERE {"];
      for (const [v, planVar] of request.bindings) {
        const members = inputs[planVar].identifiers();
        lines.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
        lines.push("  FILTER(?a = " + v + ")");
      }
      for (const s of SETTING_PREDICATES) {
        lines.push(`  OPTIONAL { ?a <${s}> ?v_${s} . }`);
      }
      lines.push("}");
      return lines.join("\n");
    }

    if (pred === "identified_as") return Identity.lower.call(this, request, inputs);

    if (pred === "measured_on") {
      const path = this.paths.measured_on || "measured_with";
      const devices = [...new Set(request.args.filter(isLiteralArg))].sort();
      const lines = ["SELECT DISTINCT ?a WHERE {"];
      if (devices.length) {
        lines.push("  VALUES ?d { " + devices.map((d) => `<${d}>`).join(" ") + " }");
      }
      for (const [v, planVar] of request.bindings) {
        const members = inputs[planVar].identifiers();
        lines.push("  VALUES " + v + " { " + members.map((m) => `<${m}>`).join(" ") + " }");
        lines.push("  FILTER(?d = " + v + ")");
      }
      lines.push(`  ?a <${path}> ?d .`);
      lines.push("}");
      return lines.join("\n");
    }

    return super.lower(request, inputs);
  }

  extract(concrete, request, inputs) {
    const pred = request.predicate;

    if (pred === "evaluations_of") {
      const path = this.paths.evaluations_of || "evaluated";
      const targets = new Set(request.args.filter(isLiteralArg));
      for (const [, planVar] of request.bindings) {
        for (const m of inputs[planVar].identifiers()) targets.add(m);
      }
      const pairs = [];
      for (const [s, p, o] of this.triples) {
        if (p === path && targets.has(o)) pairs.push([s, { _evaluated: o }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "settings_of") {
      const seeds = this.seedsFor(request, inputs) || new Set();
      const pairs = [];
      for (const a of [...seeds].sort()) {
        const attrs = {};
        for (const [s, p, o] of this.triples) {
          if (s === a && SETTING_PREDICATES.includes(p)) attrs[p] = o;
        }
        // The activity is RETURNED even with no settings. "This run recorded
        // no buffer" and "there is no such run" are different answers to Q2
        // and must not collapse.
        attrs._recorded = Object.keys(attrs).sort();
        pairs.push([a, attrs]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "identified_as") return Identity.extract(this, request, inputs);

    if (pred === "measured_on") {
      const path = this.paths.measured_on || "measured_with";
      const devices = new Set(request.args.filter(isLiteralArg));
      for (const [, planVar] of request.bindings) {
        for (const m of inputs[planVar].identifiers()) devices.add(m);
      }
      // Nothing named and nothing bound means the whole extent of the edge --
      // not the empty set. An unrestricted question about a relation is a
      // question about all of it.
      const restrict = devices.size > 0;
      const pairs = [];
      for (const [s, p, o] of this.triples) {
        if (p !== path) continue;
        if (restrict && !devices.has(o)) continue;
        const attrs = { _device: o };
        for (const [s2, p2, o2] of this.triples) {
          if (s2 === s && p2 !== path) attrs[p2] = o2;
        }
        pairs.push([s, attrs]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "measured_with") {
      // FORWARD: seeds restrict the SUBJECT and the OBJECT is returned. The
      // mirror of `measured_on`, and the pair is why the identity branch was
      // factored out rather than pasted.
      const path = this.paths.measured_with || "measured_with";
      const seeds = this.seedsFor(request, inputs);
      const pairs = [];
      for (const [s, p, o] of this.triples) {
        if (p !== path) continue;
        if (seeds && !seeds.has(s)) continue;
        const attrs = { _activity: s };
        for (const [s2, p2, o2] of this.triples) {
          if (s2 === s && p2 !== path) attrs[p2] = o2;
        }
        pairs.push([o, attrs]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    return GraphPatternAdapter.prototype.extract.call(this, concrete, request, inputs);
  }
}

/* ------------------------------------------------------------------ *
 * SequenceAdapter -- SEQ.
 * ------------------------------------------------------------------ */

export class SequenceAdapter extends Adapter {
  constructor(opts) {
    super(opts);
    this.sequences = opts.sequences || {};
    this.owner = opts.owner || {};
    if (!opts.capabilities) {
      // Deliberately NOT `pattern`: this source cannot answer a graph
      // question, and declaring `pattern` would let an ill-formed plan through
      // the static check and fail at the adapter instead.
      this.capabilities = new Set(["lookup", "regex", "neg", "bind", "batch"]);
    }
  }

  requiredFeatures(request) {
    return biocatRequiredFeatures(request);
  }

  /**
   * `C` is a residue to `excluding` and an identifier to `sequence_of`.
   * Guessing which by inspecting the fixture would make the request's meaning
   * depend on the fixture's contents, so the predicate decides.
   */
  splitArgs(request, inputs) {
    const keys = [];
    for (const [, planVar] of request.bindings) {
      keys.push(...inputs[planVar].identifiers());
    }
    const literals = request.args.filter(isLiteralArg);
    if (request.predicate === "sequence_of") {
      return [keys.concat(literals.filter((a) => !keys.includes(a))), []];
    }
    if (!literals.length) return [keys, []];
    // First literal is the pattern; any further literals are keys.
    return [
      keys.concat(literals.slice(1).filter((a) => !keys.includes(a))),
      [literals[0]],
    ];
  }

  /** A retrieval, not a query. There is no graph here to shape. */
  lower(request, inputs) {
    const [keys, residues] = this.splitArgs(request, inputs);
    const lines = [`GET ${this.name}/${request.predicate}`];
    for (const k of [...keys].sort()) lines.push(`  key = ${k}`);
    for (const r of residues) lines.push(`  residue = ${r}`);
    return lines.join("\n");
  }

  extract(concrete, request, inputs) {
    const pred = request.predicate;
    const [keys, residues] = this.splitArgs(request, inputs);

    if (pred === "sequence_of") {
      const pairs = [];
      for (const k of keys) {
        const seq = this.sequences[k];
        if (seq !== undefined) {
          pairs.push([k, { sequence: seq, length: seq.length }]);
        }
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "matching" || pred === "excluding") {
      if (!residues.length) {
        throw new Refusal(`${pred} requires a residue or pattern argument`);
      }
      const pattern = residues[0];
      let rx;
      try {
        rx = new RegExp(pattern);
      } catch (e) {
        throw new Refusal(`${pred}: uncompilable pattern '${pattern}': ${e.message}`);
      }
      let covered = 0;
      let uncovered = 0;
      let pairs = [];
      for (const k of keys) {
        const seq = this.sequences[k];
        if (seq === undefined) {
          // Neither included nor excluded. A key with no sequence was not
          // tested, and both verdicts would be inventions.
          uncovered += 1;
          continue;
        }
        covered += 1;
        const hit = rx.test(seq);
        if ((pred === "matching") === hit) {
          pairs.push([
            k,
            { sequence_length: seq.length, residue: pattern, _scanned: true },
          ]);
        }
      }
      // Coverage travels with every row: a caller reading n=2 needs to know
      // whether it was 2 of 3 or 2 of 300.
      pairs = pairs.map(([k, r]) => [k, { ...r, _covered: covered, _uncovered: uncovered }]);
      return ResultSet.of(this.namespace, pairs);
    }

    throw new Refusal(`${this.name} does not implement '${pred}'`);
  }

  cost(request, inputs) {
    let n = 0;
    for (const [, planVar] of request.bindings) n += inputs[planVar].size;
    return this.capabilities.has("batch") ? 1.0 : Math.max(1, n);
  }
}

/* ------------------------------------------------------------------ *
 * OntologyAdapter -- TAX.
 * ------------------------------------------------------------------ */

export class OntologyAdapter extends Adapter {
  constructor(opts) {
    super(opts);
    this.parents = opts.parents || {};
    this.labels = opts.labels || {};
  }

  ancestors(node) {
    const out = new Set();
    const stack = [...(this.parents[node] || [])];
    while (stack.length) {
      const cur = stack.pop();
      if (out.has(cur)) continue;
      out.add(cur);
      stack.push(...(this.parents[cur] || []));
    }
    return out;
  }

  lower(request, inputs) {
    const [keys, targets] = this.splitArgs(request, inputs);
    const lines = [`GET ${this.name}/${request.predicate}`];
    for (const k of [...keys].sort()) lines.push(`  node = ${k}`);
    for (const t of targets) lines.push(`  under = ${t}`);
    return lines.join("\n");
  }

  splitArgs(request, inputs) {
    const keys = [];
    for (const [, planVar] of request.bindings) {
      keys.push(...inputs[planVar].identifiers());
    }
    const literals = request.args.filter(isLiteralArg);
    if (request.predicate === "descends_from") {
      if (!literals.length) {
        throw new Refusal("descends_from requires an ancestor argument");
      }
      return [
        keys.concat(literals.slice(1).filter((a) => !keys.includes(a))),
        [literals[0]],
      ];
    }
    return [keys.concat(literals.filter((a) => !keys.includes(a))), []];
  }

  extract(concrete, request, inputs) {
    const pred = request.predicate;
    const [keys, targets] = this.splitArgs(request, inputs);

    if (pred === "descends_from") {
      const root = targets[0];
      const pairs = [];
      for (const k of keys) {
        const anc = this.ancestors(k);
        if (anc.has(root)) {
          pairs.push([
            k,
            { _under: root, _label: this.labels[k] ?? null, _depth: anc.size },
          ]);
        }
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "ancestors_of") {
      const pairs = [];
      for (const k of keys) {
        for (const a of [...this.ancestors(k)].sort()) {
          pairs.push([a, { _of: k, _label: this.labels[a] ?? null }]);
        }
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "labelled") {
      const pairs = [];
      for (const k of keys) {
        if (k in this.labels) pairs.push([k, { label: this.labels[k] }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    throw new Refusal(`${this.name} does not implement '${pred}'`);
  }

  cost(request, inputs) {
    let n = 0;
    for (const [, planVar] of request.bindings) n += inputs[planVar].size;
    return Math.max(1, n);
  }
}

/* ------------------------------------------------------------------ *
 * LookupAdapter -- INST.
 * ------------------------------------------------------------------ */

export class LookupAdapter extends Adapter {
  constructor(opts) {
    super(opts);
    this.records = opts.records || {};
    this.links = opts.links || {};
  }

  lower(request, inputs) {
    const keys = new Set(request.args.filter(isLiteralArg));
    for (const [, planVar] of request.bindings) {
      for (const m of inputs[planVar].identifiers()) keys.add(m);
    }
    const lines = [`GET ${this.name}/${request.predicate}`];
    for (const k of [...keys].sort()) lines.push(`  id = ${k}`);
    return lines.join("\n");
  }

  extract(concrete, request, inputs) {
    const pred = request.predicate;
    const keys = new Set(request.args.filter(isLiteralArg));
    for (const [, planVar] of request.bindings) {
      for (const m of inputs[planVar].identifiers()) keys.add(m);
    }

    if (pred === "record") {
      const pairs = [];
      for (const k of [...keys].sort()) {
        if (k in this.records) pairs.push([k, { ...this.records[k] }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    if (pred === "link") {
      const rel = request.args.find(isLiteralArg);
      const table = this.links[rel];
      if (table === undefined) {
        throw new Refusal(`${this.name} holds no link table '${rel}'`);
      }
      const pairs = [];
      for (const [k, vs] of Object.entries(table)) {
        if (keys.size && !keys.has(k)) continue;
        for (const v of vs) pairs.push([v, { _linked_from: k, _relation: rel }]);
      }
      return ResultSet.of(this.namespace, pairs);
    }

    // Only `link` and `record` are implemented. Anything else is a refusal
    // rather than a fallthrough to an empty set.
    throw new Refusal(`${this.name} does not implement '${pred}'`);
  }

  cost(request, inputs) {
    let n = 0;
    for (const [, planVar] of request.bindings) n += inputs[planVar].size;
    return this.capabilities.has("batch") ? 1.0 : Math.max(1, n);
  }
}

/* ------------------------------------------------------------------ *
 * MapAdapter and Registry.
 * ------------------------------------------------------------------ */

export class MapAdapter extends Adapter {
  constructor(opts = {}) {
    super({
      name: opts.name || "maps",
      namespace: opts.namespace || "map",
      capabilities: opts.capabilities || [],
    });
    this.maps = opts.maps || {};
  }

  chain(names) {
    return names.map((n) => {
      const m = this.maps[n];
      if (m === undefined) throw new Refusal(`unknown translation map '${n}'`);
      return m;
    });
  }

  applyChain(names, res) {
    const stages = [];
    let current = res;
    for (const mu of this.chain(names)) {
      const s = new Set(current.identifiers());
      const stage = {
        map: mu.name,
        input_size: s.size,
        retention: mu.retention(s),
        amplification: mu.amplification(s),
      };
      current = mu.apply(current);
      stage.output_size = current.size;
      stages.push(stage);
    }
    return [current, stages];
  }

  /**
   * Computed by tracking TRAJECTORIES, not by multiplying retentions. The
   * retention bounds constrain this number but do not determine it -- an
   * element dropped by the second map may be one the first amplified, and the
   * product cannot see that. This is the whole content of the bounds gap.
   */
  survivingFraction(names, res) {
    const s0 = res.identifiers();
    if (!s0.length) return 1.0;
    const chain = this.chain(names);
    let survivors = 0;
    for (const u of s0) {
      let frontier = new Set([u]);
      for (const mu of chain) {
        frontier = mu.image(frontier);
        if (!frontier.size) break;
      }
      if (frontier.size) survivors += 1;
    }
    return survivors / s0.length;
  }

  lower() {
    throw new Refusal("the map adapter answers no requests");
  }

  extract() {
    throw new Refusal("the map adapter answers no requests");
  }

  cost() {
    return 1.0;
  }
}

export class Registry {
  constructor(adapters = {}) {
    this.adapters = adapters;
  }

  register(adapter) {
    this.adapters[adapter.name] = adapter;
    return adapter;
  }

  get(name) {
    const a = this.adapters[name];
    if (a === undefined) throw new Refusal(`unknown source '${name}'`);
    return a;
  }

  totalRequests() {
    return Object.values(this.adapters).reduce((n, a) => n + a.requestsIssued, 0);
  }

  resetCounters() {
    for (const a of Object.values(this.adapters)) {
      a.requestsIssued = 0;
      a.lastLowered = null;
    }
  }
}
