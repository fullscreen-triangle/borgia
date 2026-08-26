/**
 * Masamune plan execution in the browser.
 *
 * A port of the Python reference (honjo-py/hjm/masamune/plan.py).
 * Steps run in source order and are never reordered: a plan is a
 * statement of what was asked for, in what order, and reordering it
 * would change what the provenance record means.
 *
 * The browser has no file system, so `source ... at "name"` resolves
 * against a supplied record set rather than a disk path. That is the
 * only intentional difference from the reference, and it is confined
 * to `readSource`.
 */

import {
  translate,
  capability,
  missing,
  knownFormat,
  suppliedFraction,
  LABEL,
  PROV,
} from "@/lib/masamune";

/* ------------------------------------------------------------------ */
/*  Lexer                                                             */
/* ------------------------------------------------------------------ */

const PUNCT = new Set(["{", "}", "(", ")", ",", ":", ".", "<", ">", "="]);

function lex(src) {
  const toks = [];
  const lines = src.split("\n");
  lines.forEach((raw, li) => {
    const line = raw.replace(/--.*$/, "");
    let i = 0;
    while (i < line.length) {
      const c = line[i];
      if (/\s/.test(c)) { i += 1; continue; }

      if (c === '"') {
        const end = line.indexOf('"', i + 1);
        if (end < 0) throw new PlanError(`unterminated string`, li + 1);
        toks.push({ kind: "string", text: line.slice(i + 1, end), line: li + 1 });
        i = end + 1;
        continue;
      }
      if (line.startsWith(":=", i)) {
        toks.push({ kind: "op", text: ":=", line: li + 1 });
        i += 2;
        continue;
      }
      if (line.startsWith("==", i) || line.startsWith(">=", i) ||
          line.startsWith("<=", i) || line.startsWith("!=", i)) {
        toks.push({ kind: "op", text: line.slice(i, i + 2), line: li + 1 });
        i += 2;
        continue;
      }
      if (/[0-9]/.test(c)) {
        let j = i;
        while (j < line.length && /[0-9.]/.test(line[j])) j += 1;
        toks.push({ kind: "number", text: line.slice(i, j), line: li + 1 });
        i = j;
        continue;
      }
      if (/[A-Za-z_]/.test(c)) {
        let j = i;
        while (j < line.length && /[A-Za-z0-9_]/.test(line[j])) j += 1;
        toks.push({ kind: "ident", text: line.slice(i, j), line: li + 1 });
        i = j;
        continue;
      }
      if (PUNCT.has(c)) {
        toks.push({ kind: "op", text: c, line: li + 1 });
        i += 1;
        continue;
      }
      throw new PlanError(`unexpected character '${c}'`, li + 1);
    }
  });
  toks.push({ kind: "eof", text: "", line: lines.length });
  return toks;
}

export class PlanError extends Error {
  constructor(msg, line) {
    super(`line ${line}: ${msg}`);
    this.line = line;
  }
}

/* ------------------------------------------------------------------ */
/*  Parser                                                            */
/* ------------------------------------------------------------------ */

class Parser {
  constructor(toks) { this.t = toks; this.i = 0; }
  peek() { return this.t[this.i]; }
  next() { const t = this.t[this.i]; this.i += 1; return t; }
  expect(kind, text) {
    const t = this.next();
    if (t.kind !== kind || (text !== undefined && t.text !== text)) {
      throw new PlanError(
        `expected '${text ?? kind}', got '${t.text || t.kind}'`, t.line
      );
    }
    return t;
  }
  accept(text) {
    if (this.peek().text === text) { this.i += 1; return true; }
    return false;
  }

  parse() {
    this.expect("ident", "plan");
    const name = this.expect("ident").text;
    this.expect("op", "{");
    const plan = { name, sources: {}, budget: null, steps: [] };
    while (this.peek().text !== "}" && this.peek().kind !== "eof") {
      this.declOrStep(plan);
    }
    this.expect("op", "}");
    return plan;
  }

  declOrStep(plan) {
    const tok = this.peek();

    if (tok.text === "source") {
      this.next();
      const bind = this.expect("ident").text;
      this.expect("op", ":");
      const fmt = this.expect("ident").text;
      this.expect("ident", "at");
      const at = this.expect("string").text;
      plan.sources[bind] = { format: fmt, at };
      return;
    }

    if (tok.text === "budget") {
      this.next();
      plan.budget = parseInt(this.expect("number").text, 10);
      this.expect("ident", "records");
      return;
    }

    if (tok.text === "let") {
      this.next();
      const target = this.expect("ident").text;
      this.expect("op", ":=");
      plan.steps.push(this.rhs(target, tok.line));
      return;
    }

    if (tok.text === "assert") {
      this.next();
      const cond = this.cond();
      let message = null;
      if (this.accept("emit")) message = this.expect("string").text;
      plan.steps.push({
        op: "assert", line: tok.line, target: null,
        args: { cond, message },
      });
      return;
    }

    if (tok.text === "emit") {
      this.next();
      const what = this.expect("ident").text;
      let withProv = false;
      if (this.accept("with")) {
        this.expect("ident", "provenance");
        withProv = true;
      }
      plan.steps.push({
        op: "emit", line: tok.line, target: null,
        args: { name: what, provenance: withProv },
      });
      return;
    }

    throw new PlanError(`unexpected '${tok.text}'`, tok.line);
  }

  rhs(target, line) {
    const tok = this.peek();

    if (tok.text === "read") {
      this.next();
      const source = this.expect("ident").text;
      return { op: "read", line, target, args: { source } };
    }

    if (tok.text === "translate") {
      this.next();
      const input = this.expect("ident").text;
      const require = [];
      let expect = null;
      let onFail = "report";
      if (this.accept("require")) {
        for (;;) {
          require.push(this.expect("ident").text);
          if (!this.accept(",")) break;
        }
      }
      if (this.accept("expect")) {
        this.expect("ident", "supplied");
        const op = this.next().text;
        const rhs = parseFloat(this.expect("number").text);
        expect = { lhs: "supplied", op, rhs };
      }
      if (this.accept("else")) onFail = this.expect("ident").text;
      return {
        op: "translate", line, target,
        args: { input, require, expect, on_fail: onFail },
      };
    }

    if (tok.text === "select") {
      this.next();
      const input = this.expect("ident").text;
      this.expect("ident", "where");
      const cond = this.cond();
      return { op: "select", line, target, args: { input, cond } };
    }

    throw new PlanError(`unknown step '${tok.text}'`, tok.line);
  }

  cond() {
    let lhs = this.expect("ident").text;
    if (this.peek().text === ".") {
      this.next();
      lhs += `.${this.expect("ident").text}`;
    }
    const op = this.next().text;
    const t = this.next();
    const rhs = t.kind === "number" ? parseFloat(t.text) : t.text;
    return { lhs, op, rhs };
  }
}

function compare(a, op, b) {
  switch (op) {
    case "==": return a === b;
    case "!=": return a !== b;
    case "<": return a < b;
    case "<=": return a <= b;
    case ">": return a > b;
    case ">=": return a >= b;
    default: return false;
  }
}

/* ------------------------------------------------------------------ */
/*  Runner                                                            */
/* ------------------------------------------------------------------ */

/**
 * Execute a plan.
 *
 * @param {string} src      plan source
 * @param {object} files    name -> array of {name, text} records
 */
export function runPlan(src, files = {}) {
  let plan;
  try {
    plan = new Parser(lex(src)).parse();
  } catch (err) {
    return {
      plan: null,
      status: "parse-error",
      error: err.message,
      steps: [],
    };
  }

  // Static capability check, before any record is read. A request the
  // format cannot state is refused here, and the refusal names what
  // was missing and what the format does declare.
  for (const st of plan.steps) {
    if (st.op !== "translate") continue;
    const srcName = sourceOf(plan, st);
    if (!srcName) continue;
    const fmt = plan.sources[srcName]?.format ?? "?";
    const req = st.args.require;
    const miss = knownFormat(fmt) ? missing(fmt, req) : [...req].sort();
    if (miss.length) {
      return {
        plan: plan.name,
        status: "refused",
        refusal: {
          reason: "capability",
          step_line: st.line,
          source: srcName,
          format: fmt,
          missing_features: miss,
          source_capability: capability(fmt),
        },
        records_read: 0,
        steps: [],
      };
    }
  }

  const env = {};
  const log = [];
  const steps = [];
  let status = "ok";
  let recordsRead = 0;

  for (const st of plan.steps) {
    if (status !== "ok") break;

    if (st.op === "read") {
      const decl = plan.sources[st.args.source];
      if (!decl) {
        status = "error";
        steps.push({
          step: "read", line: st.line,
          error: `undeclared source '${st.args.source}'`,
        });
        break;
      }
      let recs = files[decl.at];
      if (!recs) {
        status = "error";
        steps.push({
          step: "read", line: st.line,
          error: `no records for '${decl.at}'`,
          available: Object.keys(files),
        });
        break;
      }
      if (plan.budget !== null && recs.length > plan.budget) {
        log.push({
          level: "report", step_line: st.line,
          message: `budget ${plan.budget} < ${recs.length} available; truncated`,
        });
        recs = recs.slice(0, plan.budget);
      }
      recordsRead += recs.length;
      env[st.target] = { kind: "records", items: recs, source: st.args.source };
      steps.push({
        step: "read", line: st.line, target: st.target, count: recs.length,
      });
      continue;
    }

    if (st.op === "translate") {
      const rs = env[st.args.input];
      if (!rs || rs.kind !== "records") {
        status = "error";
        steps.push({
          step: "translate", line: st.line,
          error: `${st.args.input} is not a record set`,
        });
        break;
      }
      const fmt = plan.sources[rs.source].format;
      const req = st.args.require.length
        ? st.args.require
        : ["element", "connectivity"];
      const results = rs.items.map((rec) => {
        let v = translate(fmt, rec.text, { required: req, sourceName: rec.name });
        v = applyExpect(v, st.args.expect);
        return { record: rec.name, verdict: v };
      });
      env[st.target] = { kind: "verdicts", items: results };
      const tally = {};
      for (const r of results) {
        tally[r.verdict.label] = (tally[r.verdict.label] || 0) + 1;
      }
      steps.push({
        step: "translate", line: st.line, target: st.target,
        require: [...req].sort(), tally,
      });
      continue;
    }

    if (st.op === "select") {
      const vs = env[st.args.input];
      if (!vs || vs.kind !== "verdicts") {
        status = "error";
        steps.push({
          step: "select", line: st.line,
          error: `${st.args.input} is not a verdict set`,
        });
        break;
      }
      const kept = vs.items.filter((r) => {
        const val = conditionValue(r, st.args.cond.lhs);
        return val !== null && compare(val, st.args.cond.op, st.args.cond.rhs);
      });
      env[st.target] = { kind: "verdicts", items: kept };
      steps.push({
        step: "select", line: st.line, target: st.target,
        kept: kept.length, dropped: vs.items.length - kept.length,
        condition: st.args.cond,
      });
      continue;
    }

    if (st.op === "assert") {
      const { lhs, op, rhs } = st.args.cond;
      const observed = envValue(env, lhs);
      const passed = observed !== null && compare(observed, op, rhs);
      steps.push({
        step: "assert", line: st.line, condition: st.args.cond,
        observed, passed,
      });
      if (!passed) {
        // the reference calls this 'assertion-failed', not 'halted':
        // the status names what happened, not what the runner did
        status = "assertion-failed";
        log.push({
          level: "assert", step_line: st.line,
          message: st.args.message || "assertion failed",
        });
      }
      continue;
    }

    if (st.op === "emit") {
      const v = env[st.args.name];
      if (!v) {
        status = "error";
        steps.push({
          step: "emit", line: st.line,
          error: `nothing named '${st.args.name}'`,
        });
        break;
      }
      const emitted = (v.items || []).map((r) => {
        const out = { record: r.record, verdict: r.verdict.label };
        if (st.args.provenance) {
          out.payload = r.verdict.payload;
          if (r.verdict.value) {
            out.value = {
              floor: r.verdict.value.floor,
              atoms: Object.keys(r.verdict.value.atoms).length,
              supplied_fraction: suppliedFraction(r.verdict.value),
            };
          }
        }
        return out;
      });
      steps.push({
        step: "emit", line: st.line, name: st.args.name,
        emitted, with_provenance: st.args.provenance,
      });
      continue;
    }
  }

  return {
    plan: plan.name, status, steps, log, records_read: recordsRead,
  };
}

function sourceOf(plan, st) {
  // walk back to the read step that produced this translate's input
  for (const s of plan.steps) {
    if (s.op === "read" && s.target === st.args.input) return s.args.source;
  }
  return null;
}

/**
 * Apply an `expect` clause.
 *
 * A translation whose supplied fraction fails the expectation is
 * substituted to INCOMPLETE: the value is withheld rather than carried
 * forward with a note attached, because a consumer that ignores notes
 * would otherwise compute over data the plan said it did not want.
 */
function applyExpect(v, expect) {
  if (!expect || !v.ok) return v;
  const phi = v.payload?.supplied_fraction;
  if (phi === undefined) return v;
  if (compare(phi, expect.op, expect.rhs)) return v;
  return {
    label: LABEL.INCOMPLETE,
    payload: {
      supplied_fraction: phi,
      expectation: `supplied ${expect.op} ${expect.rhs}`,
      reason: "expectation not met",
    },
    value: null,
    ok: false,
  };
}

function conditionValue(row, lhs) {
  if (lhs === "supplied") {
    const p = row.verdict.payload?.supplied_fraction;
    return p === undefined ? null : p;
  }
  if (lhs === "verdict") return row.verdict.label;
  return null;
}

function envValue(env, lhs) {
  const [name, field] = lhs.split(".");
  const v = env[name];
  if (!v) return null;
  if (field === "count") return (v.items || []).length;
  return null;
}
