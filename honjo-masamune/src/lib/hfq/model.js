/**
 * The value types: result sets, translation maps, verdicts, blockers.
 *
 * Ported from hfq/model.py. Three things here are load-bearing and are
 * commented at the point of definition rather than in a header, because each
 * is a place where the obvious implementation is wrong:
 *
 *   - `ResultSet.of` MERGES attribute maps on a repeated identifier. It does
 *     not keep the first, and it does not keep both. That collapse is the
 *     semantics of a result set, not a deduplication convenience.
 *
 *   - `blockerOf` is DELIBERATELY PARTIAL. `answer` and `empty` have no
 *     blocker. Returning one for `empty` would assert an obstruction that did
 *     not occur -- the map is partial, and "nothing matched" is an answer
 *     about the world, not a failure of the machinery.
 *
 *   - `amplification` returns null, not 1 or 0, when nothing was retained.
 *     There is no ratio to report when the denominator is empty, and a
 *     placeholder would be indistinguishable from a measured value.
 */

export const Verdict = {
  ANSWER: "answer",
  EMPTY: "empty",
  SURFACE: "surface",
  TIMEOUT: "timeout",
  REFUSED: "refused",
  STARVED: "starved",
};

export const Blocker = {
  MODEL: "model",
  ENGINE: "engine",
  BUDGET: "budget",
  CORPUS: "corpus",
};

/**
 * In the paper's prose `surface` is `unsupported` and `timeout` is
 * `exhausted`; the code's names are kept so the JSON matches the reference
 * byte for byte, and the prose names are carried alongside for display.
 */
export const VERDICT_PROSE = {
  answer: "answer",
  empty: "empty",
  surface: "unsupported",
  timeout: "exhausted",
  refused: "refused",
  starved: "starved",
};

const BLOCKER_MAP = {
  [Verdict.SURFACE]: Blocker.MODEL,
  [Verdict.TIMEOUT]: Blocker.ENGINE,
  [Verdict.REFUSED]: Blocker.BUDGET,
  [Verdict.STARVED]: Blocker.CORPUS,
};

/** Partial on purpose. `answer` and `empty` are not obstructions. */
export function blockerOf(verdict) {
  return BLOCKER_MAP[verdict] ?? null;
}

/** The eleven capability symbols. Frozen: a twelfth would be a new theory. */
export const FEAT = Object.freeze([
  "pattern",
  "path",
  "filter",
  "bind",
  "agg",
  "neg",
  "order",
  "regex",
  "lookup",
  "link",
  "batch",
]);

const FEAT_SET = new Set(FEAT);

export function isFeature(f) {
  return FEAT_SET.has(f);
}

/** A refusal is a value, not an exception in the ordinary sense. */
export class Refusal extends Error {
  constructor(message) {
    super(message);
    this.name = "Refusal";
  }
}

/**
 * A set of identifiers, each carrying an attribute map.
 *
 * Insertion order is preserved by the underlying Map, but no operation may
 * depend on it: two executions that reach the same set by different routes
 * must compare equal, and `identifiers()` therefore sorts.
 */
export class ResultSet {
  constructor(namespace, rows) {
    this.namespace = namespace;
    this.rows = rows instanceof Map ? rows : new Map(Object.entries(rows || {}));
  }

  static empty(namespace) {
    return new ResultSet(namespace, new Map());
  }

  /**
   * Build from (identifier, attributes) pairs.
   *
   * A repeated identifier MERGES its attribute maps. Two routes that reach the
   * same enzyme -- one carrying `_via`, one carrying `_device` -- produce one
   * row carrying both. Keeping only the first would make the result depend on
   * pair order; keeping both would make a set into a multiset.
   */
  static of(namespace, pairs) {
    const rows = new Map();
    for (const [ident, attrs] of pairs) {
      const prev = rows.get(ident);
      if (prev === undefined) rows.set(ident, { ...(attrs || {}) });
      else Object.assign(prev, attrs || {});
    }
    return new ResultSet(namespace, rows);
  }

  identifiers() {
    return [...this.rows.keys()].sort();
  }

  get size() {
    return this.rows.size;
  }

  attrs(ident) {
    return this.rows.get(ident) || {};
  }

  toJSON() {
    const out = {};
    for (const k of this.identifiers()) out[k] = this.rows.get(k);
    return { namespace: this.namespace, n: this.rows.size, rows: out };
  }
}

/**
 * A relation mu subset of n x n'. Partial, non-functional, non-injective --
 * all three at once, which is why retention and amplification are independent
 * numbers rather than two readings of one.
 */
export class TranslationMap {
  constructor({ name, source_ns, target_ns, pairs }) {
    this.name = name;
    this.sourceNs = source_ns;
    this.targetNs = target_ns;
    this.pairs = pairs instanceof Map ? pairs : new Map(Object.entries(pairs || {}));
  }

  /** Keys with a NON-EMPTY image. A key mapping to [] is not in the domain. */
  domain() {
    const d = new Set();
    for (const [k, v] of this.pairs) if (v && v.length) d.add(k);
    return d;
  }

  image(s) {
    const out = new Set();
    for (const u of s) for (const v of this.pairs.get(u) || []) out.add(v);
    return out;
  }

  /** Fraction of the input that the map is defined on. Empty input: 1.0. */
  retention(s) {
    const set = s instanceof Set ? s : new Set(s);
    if (!set.size) return 1.0;
    const dom = this.domain();
    let kept = 0;
    for (const u of set) if (dom.has(u)) kept += 1;
    return kept / set.size;
  }

  /**
   * Images per retained input. NULL when nothing was retained -- there is no
   * ratio, and 0 or 1 would read as a measurement.
   */
  amplification(s) {
    const set = s instanceof Set ? s : new Set(s);
    const dom = this.domain();
    let kept = 0;
    for (const u of set) if (dom.has(u)) kept += 1;
    if (kept === 0) return null;
    return this.image(set).size / kept;
  }

  apply(res) {
    const pairs = [];
    for (const u of res.identifiers()) {
      for (const v of this.pairs.get(u) || []) {
        pairs.push([v, { ...res.attrs(u), _via: this.name, _preimage: u }]);
      }
    }
    return ResultSet.of(this.targetNs, pairs);
  }
}
