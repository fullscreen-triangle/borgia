/**
 * Stage 1 of the pipeline: Parse.
 *
 * A hand-port of the Python reference (hegel/ckg/validation-federated/hfq/parser.py).
 * The grammar is reproduced exactly, including the three lexing decisions that
 * are load-bearing rather than incidental:
 *
 *   (1) `_CONNECTIVE` is tested BEFORE the literal decision in a filter clause.
 *       While that test sat below the quoted-string case in the reference,
 *       `label == "x" and size > 2` reached neither branch and parsed as a
 *       comparison against the string `"x" and size > 2`.
 *
 *   (2) The connective pattern is bounded on both sides by a non-word
 *       assertion, because an unanchored `or` fires on the `or` inside
 *       `chlorine`.
 *
 *   (3) In `ladder`, the expectation clause is removed BEFORE the rungs are
 *       read. Otherwise the `power` inside `expect power E` matches the rung
 *       pattern and the plan composes a ladder it did not write.
 *
 * Tokenisation is line-oriented: a `let`/`emit`/`budget` keyword at the head of
 * a line opens a clause, and subsequent more-indented lines continue it.
 */

export class PlanError extends Error {
  constructor(message, line = null) {
    super(line == null ? message : `line ${line}: ${message}`);
    this.name = "PlanError";
    this.line = line;
    this.raw = message;
  }
}

const CLAUSE_HEAD = /^\s*(let|emit|budget|assert|plan|\})/;
const ARG = /"([^"]*)"|\?([A-Za-z_]\w*)|([-+]?\d+(?:\.\d+)?)|([A-Za-z_][\w:.\-]*)/g;
const CONNECTIVE = /(?<![\w"])(?:and|or|not)(?![\w"])/i;

/** A `#` outside a quoted string starts a comment. */
function stripComment(line) {
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];
    if (c === '"') quoted = !quoted;
    else if (c === "#" && !quoted) return line.slice(0, i);
  }
  return line;
}

/** Group physical lines into logical clauses, keyed by opening line number. */
function clauses(text) {
  const out = [];
  let current = null;
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i += 1) {
    const raw = stripComment(lines[i]);
    if (!raw.trim()) continue;
    if (CLAUSE_HEAD.test(raw)) {
      if (current) out.push(current);
      current = { line: i + 1, parts: [raw.trim()] };
    } else {
      if (!current) {
        throw new PlanError(`continuation with no clause: ${raw.trim()}`, i + 1);
      }
      current.parts.push(raw.trim());
    }
  }
  if (current) out.push(current);
  return out;
}

/** Quoted -> string; ?v -> "?v"; numeric -> Number; bare token -> string. */
function parseArgs(text) {
  const out = [];
  ARG.lastIndex = 0;
  let m;
  while ((m = ARG.exec(text)) !== null) {
    if (m[1] !== undefined) out.push(m[1]);
    else if (m[2] !== undefined) out.push("?" + m[2]);
    else if (m[3] !== undefined) out.push(Number(m[3]));
    else out.push(m[4]);
  }
  return out;
}

function parseFrom(vari, rhs, line) {
  const head = /^from\s+([A-Za-z_][\w-]*)\s*([\s\S]*)$/.exec(rhs);
  if (!head) throw new PlanError("malformed `from` clause", line);
  const source = head[1];
  const rest = head[2];

  const ask = /\bask\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/.exec(rest);
  if (!ask) throw new PlanError("`from` clause has no `ask` predicate", line);

  const bindings = [];
  const withRe = /\bwith\s+\?(\w+)\s+in\s+(\w+)/g;
  let w;
  while ((w = withRe.exec(rest)) !== null) bindings.push(["?" + w[1], w[2]]);

  const within = /\bwithin\s+(\d+(?:\.\d+)?)/.exec(rest);

  return {
    kind: "from",
    var: vari,
    source,
    predicate: ask[1],
    args: parseArgs(ask[2]),
    beta: bindings.map((b) => b[1]),
    bindings,
    budget: within ? Number(within[1]) : Infinity,
    onUnresolved: /\belse\s+fail\s+unresolved\b/.test(rest) ? "fail" : null,
    onStarved: /\bwhen\s+starved\s+emit\s+partial\b/.test(rest)
      ? "emit partial"
      : null,
    operands: [],
    line,
  };
}

function parseMap(vari, rhs, line) {
  const head = /^map\s+(\w+)\s+via\s+([\s\S]*)$/.exec(rhs);
  if (!head) throw new PlanError("malformed `map` clause", line);
  const src = head[1];
  const rest = head[2];
  const first = /^([A-Za-z_]\w*)/.exec(rest.trim());
  if (!first) throw new PlanError("`map ... via` names no map", line);
  const maps = [first[1]];
  const thenRe = /\bthen\s+via\s+([A-Za-z_]\w*)/g;
  let t;
  while ((t = thenRe.exec(rest)) !== null) maps.push(t[1]);
  const exp = /\bexpect\s+partial\s+(\d+(?:\.\d+)?)/.exec(rest);
  return {
    kind: "map",
    var: vari,
    source: null,
    maps,
    beta: [src],
    bindings: [],
    operands: [],
    expectPartial: exp ? Number(exp[1]) : null,
    budget: Infinity,
    onUnresolved: null,
    onStarved: null,
    line,
  };
}

function parseLadder(vari, rhs, line) {
  const head = /^ladder\s+over\s+(\w+)([\s\S]*)$/.exec(rhs);
  if (!head) throw new PlanError("malformed `ladder` clause", line);
  const src = head[1];
  let rest = head[2];

  // The expectation is removed FIRST. Left in place, the `power` inside
  // `expect power E` matches the rung pattern below and the plan composes a
  // ladder it did not write.
  const exp = /\bexpect\s+power\s+(\d+(?:\.\d+)?)/.exec(rest);
  if (exp) rest = rest.replace(exp[0], " ");

  const rungs = [];
  const rungRe = /\bpower\s+(\d+(?:\.\d+)?)/g;
  let r;
  while ((r = rungRe.exec(rest)) !== null) {
    const p = Number(r[1]);
    if (p < 0 || p > 1) {
      throw new PlanError(`rung power ${p} outside [0,1]`, line);
    }
    rungs.push(p);
  }
  if (!rungs.length) throw new PlanError("ladder declares no rungs", line);

  return {
    kind: "ladder",
    var: vari,
    source: null,
    rungs,
    expectPower: exp ? Number(exp[1]) : null,
    beta: [src],
    bindings: [],
    operands: [src],
    budget: Infinity,
    onUnresolved: null,
    onStarved: null,
    line,
  };
}

const OPS = ["==", "!=", "<=", ">=", "<", ">"];

function parseLet(clause, line) {
  const m = /^let\s+(\w+)\s*=\s*([\s\S]*)$/.exec(clause);
  if (!m) throw new PlanError("malformed `let` clause", line);
  const vari = m[1];
  const rhs = m[2].trim();

  if (rhs.startsWith("from ")) return parseFrom(vari, rhs, line);
  if (rhs.startsWith("map ")) return parseMap(vari, rhs, line);
  if (rhs.startsWith("ladder ")) return parseLadder(vari, rhs, line);

  const setop = /^(union|intersect)\s+(\w+)\s+(\w+)\s*$/.exec(rhs);
  if (setop) {
    return {
      kind: setop[1],
      var: vari,
      source: null,
      beta: [],
      bindings: [],
      operands: [setop[2], setop[3]],
      budget: Infinity,
      onUnresolved: null,
      onStarved: null,
      line,
    };
  }

  const jn = /^join\s+(\w+)\s+(\w+)\s+on\s+([\w:.\-]+)\s*$/.exec(rhs);
  if (jn) {
    return {
      kind: "join",
      var: vari,
      source: null,
      beta: [],
      bindings: [],
      operands: [jn[1], jn[2]],
      attr: jn[3],
      budget: Infinity,
      onUnresolved: null,
      onStarved: null,
      line,
    };
  }

  const fl = /^filter\s+(\w+)\s+where\s+([\w:.\-]+)\s*(==|!=|<=|>=|<|>)\s*([\s\S]+)$/.exec(
    rhs
  );
  if (fl) {
    const val = fl[4].trim();
    // Order is half the fix: this test must precede the literal decision.
    if (CONNECTIVE.test(val)) {
      throw new PlanError(
        "filter admits one comparison; chain filter steps instead of " +
          "writing a boolean connective",
        line
      );
    }
    let value;
    if (val.startsWith('"') && val.endsWith('"')) value = val.slice(1, -1);
    else if (/^[-+]?\d+(\.\d+)?$/.test(val)) value = Number(val);
    else value = val;
    return {
      kind: "filter",
      var: vari,
      source: null,
      beta: [],
      bindings: [],
      operands: [fl[1]],
      attr: fl[2],
      op: fl[3],
      value,
      budget: Infinity,
      onUnresolved: null,
      onStarved: null,
      line,
    };
  }

  throw new PlanError(`unrecognised right-hand side: ${rhs}`, line);
}

const GAPS = new Set(["induction", "vocabulary", "conditions"]);

function parseEmit(clause, line) {
  const dv = /^emit\s+divergence\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)(?:\s+as\s+(\w+))?/.exec(
    clause
  );
  if (dv) {
    return {
      target: null,
      divergence: [dv[1], dv[2]],
      alias: dv[3] || null,
      provenance: false,
      intension: null,
      gap: null,
      line,
    };
  }
  const m = /^emit\s+(\w+)([\s\S]*)$/.exec(clause);
  if (!m) throw new PlanError("malformed `emit` clause", line);
  const rest = m[2];
  const ext = /\bas\s+extension\s+of\s+"([^"]*)"/.exec(rest);
  const gap = /\bbecause\s+(induction|vocabulary|conditions)\b/.exec(rest);
  if (gap && !GAPS.has(gap[1])) throw new PlanError("unknown gap kind", line);
  return {
    target: m[1],
    divergence: null,
    alias: null,
    provenance: /\bwith\s+provenance\b/.test(rest),
    intension: ext ? ext[1] : null,
    gap: gap ? gap[1] : null,
    line,
  };
}

/**
 * Condition (ii) -- every bound variable resolved by a strictly earlier step --
 * is what makes the blame chain terminate, so it is checked here rather than
 * discovered at run time.
 */
function checkWellformed(plan) {
  const seen = new Set();
  for (const s of plan.steps) {
    if (seen.has(s.var)) {
      throw new PlanError(`duplicate binding of ${s.var}`, s.line);
    }
    for (const y of [...s.beta, ...s.operands]) {
      if (!seen.has(y)) {
        throw new PlanError(
          `step ${s.var} binds ${y}, which no earlier step resolves`,
          s.line
        );
      }
    }
    seen.add(s.var);
  }
  for (const e of plan.emits) {
    const names = e.divergence || [e.target];
    for (const n of names) {
      if (!seen.has(n)) {
        throw new PlanError(`emit names unresolved ${n}`, e.line);
      }
    }
  }
}

export function parsePlan(text) {
  let name = null;
  let budget = null;
  const steps = [];
  const emits = [];

  for (const c of clauses(text)) {
    const joined = c.parts.join(" ").trim();
    if (joined.startsWith("}")) continue;
    if (joined.startsWith("plan ")) {
      const m = /^plan\s+([\w-]+)/.exec(joined);
      if (!m) throw new PlanError("malformed `plan` header", c.line);
      name = m[1];
      continue;
    }
    if (joined.startsWith("budget")) {
      const m = /^budget\s+(\d+)\s+requests?/.exec(joined);
      if (!m) throw new PlanError("malformed `budget` declaration", c.line);
      budget = Number(m[1]);
      continue;
    }
    // Soundness assertions are declarative; there is nothing to execute.
    if (joined.startsWith("assert")) continue;
    if (joined.startsWith("let ")) {
      steps.push(parseLet(joined, c.line));
      continue;
    }
    if (joined.startsWith("emit")) {
      emits.push(parseEmit(joined, c.line));
      continue;
    }
    throw new PlanError(`unrecognised clause: ${joined}`, c.line);
  }

  if (!name) throw new PlanError("plan has no name");
  if (budget == null) throw new PlanError("plan has no budget declaration");

  const plan = { name, budget, steps, emits };
  checkWellformed(plan);
  return plan;
}
