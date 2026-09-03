/**
 * Stages 5 and 6: Execute and Emit.
 *
 * The verdict rules, applied IN ORDER. The order is the semantics, not an
 * implementation detail:
 *
 *   (R1) some bound input has a non-answer verdict, or an answer whose
 *        retention falls below the declared expectation   -> starved
 *   (R2) the required features are not a subset of the
 *        source's declared capability set                 -> surface
 *   (R3) remaining budget is below the source's cost at
 *        this input cardinality                           -> refused
 *   (R4) the request does not complete within b           -> timeout
 *   (R5) the extracted result set is empty                -> empty
 *   (R6) otherwise                                        -> answer
 *
 * A step that is both starved and over-budget reports STARVATION, because (R1)
 * precedes (R3) and the earlier obstruction is the one that actually stopped
 * it. Raising the budget on such a step changes nothing, and a diagnosis
 * pointing at the budget would send the reader to fix the wrong thing.
 *
 * Ported from hfq/execute.py.
 */

import { Verdict, blockerOf, ResultSet, Refusal, VERDICT_PROSE } from "./model.js";
import { check, refusalDocument } from "./check.js";
import { resolveFeatures } from "./adapters.js";
import { YieldSpec, solve, kktResiduals } from "./allocate.js";

export class StepResult {
  constructor(fields) {
    Object.assign(
      this,
      {
        step: null,
        source: null,
        verdict: null,
        diagnosis: null,
        culprit: null,
        compositePower: null,
        retention: null,
        amplification: null,
        expected: null,
        allocated: 0.0,
        spent: 0.0,
        shadowPrice: 0.0,
        payload: null,
        snapshot: null,
        loweredForm: null,
        stages: [],
      },
      fields
    );
  }

  get blocker() {
    return blockerOf(this.verdict);
  }

  toJSON(includePayload = true) {
    const d = {
      step: this.step,
      source: this.source,
      verdict: this.verdict,
      verdict_prose: VERDICT_PROSE[this.verdict],
      blocker: this.blocker,
      diagnosis: this.diagnosis,
      allocated: this.allocated,
      spent: this.spent,
      shadow_price: this.shadowPrice,
      n: this.payload ? this.payload.size : 0,
      snapshot: this.snapshot,
      lowered_form: this.loweredForm,
    };
    if (this.retention !== null) d.retention = this.retention;
    if (this.amplification !== null) d.amplification = this.amplification;
    if (this.expected !== null) d.expected = this.expected;
    if (this.compositePower !== null) d.composite_power = this.compositePower;
    if (this.stages.length) d.stages = this.stages;
    // Payload only on an ANSWER. A non-answer carrying rows would invite the
    // reader to use them, and there is nothing standing behind them.
    if (includePayload && this.verdict === Verdict.ANSWER && this.payload) {
      d.payload = this.payload.toJSON();
    }
    return d;
  }
}

export class Execution {
  constructor(fields) {
    Object.assign(
      this,
      {
        planName: null,
        steps: [],
        allocation: null,
        checkReport: null,
        emitted: {},
        requestsIssued: 0,
        haltedEarly: false,
        kkt: null,
        replans: [],
      },
      fields
    );
  }

  byVar(v) {
    return this.steps.find((s) => s.step === v) || null;
  }

  verdicts() {
    const out = {};
    for (const s of this.steps) out[s.step] = s.verdict;
    return out;
  }

  /**
   * Walk backwards from a starved step to the obstruction that caused it.
   *
   * This terminates within m hops because every bound variable is resolved by a
   * STRICTLY earlier step -- the well-formedness condition the parser enforces.
   * Without it the chain could cycle, and a diagnosis that can loop forever is
   * not a diagnosis.
   */
  blameChain(v) {
    const chain = [];
    const seen = new Set();
    let cur = v;
    while (cur && !seen.has(cur)) {
      seen.add(cur);
      const r = this.byVar(cur);
      if (!r) break;
      chain.push({
        step: cur,
        verdict: r.verdict,
        blocker: r.blocker,
        diagnosis: r.diagnosis,
      });
      if (r.verdict !== Verdict.STARVED) break;
      cur = r.culprit;
    }
    return chain;
  }

  toJSON(includePayload = true) {
    return {
      plan: this.planName,
      requests_issued: this.requestsIssued,
      halted_early: this.haltedEarly,
      check: this.checkReport ? this.checkReport.toJSON() : null,
      allocation: this.allocation ? this.allocation.toJSON() : null,
      replans: this.replans.map((r) => ({
        after: r.after,
        remaining: r.remaining,
        allocation: r.allocation.toJSON(),
      })),
      kkt: this.kkt,
      steps: this.steps.map((s) => s.toJSON(includePayload)),
      emitted: this.emitted,
    };
  }
}

/**
 * One yield spec per step.
 *
 * A step whose source is a pure retrieval -- it declares `lookup` and not
 * `pattern` -- is all-or-nothing: there is no partial retrieval to buy, so the
 * concave curve does not describe it and the allocator charges it up front.
 */
export function yieldSpecs(plan, registry, weights = {}) {
  const specs = [];
  for (const s of plan.steps) {
    const w = weights[s.var] === undefined ? 1.0 : weights[s.var];
    if (s.kind !== "from") {
      specs.push(new YieldSpec(s.var, w, { allOrNothing: true, fixedCost: 1.0 }));
      continue;
    }
    const cap = registry.get(s.source).capabilities;
    const aon = cap.has("lookup") && !cap.has("pattern");
    specs.push(new YieldSpec(s.var, w, { allOrNothing: aon, fixedCost: 1.0 }));
  }
  return specs;
}

export class Executor {
  constructor(registry, maps = null) {
    this.registry = registry;
    this.maps = maps;
    this._current = [];
  }

  run(plan, budget = null, { weights = {} } = {}) {
    this.registry.resetCounters();
    const total = budget == null ? plan.budget : budget;

    const report = check(plan, this.registry);

    // The refusal happens BEFORE any request is issued. `requestsIssued` stays
    // at zero, and that zero is the observable claim: nobody had to touch an
    // endpoint to discover that this plan could not work.
    if (!report.wellCapability) {
      const steps = report.failures.map(
        (f) =>
          new StepResult({
            step: f.step,
            source: f.source,
            verdict: Verdict.SURFACE,
            diagnosis: f.reason,
            payload: ResultSet.empty("-"),
          })
      );
      return new Execution({
        planName: plan.name,
        steps,
        checkReport: report,
        haltedEarly: true,
        requestsIssued: this.registry.totalRequests(),
        emitted: { refusal: refusalDocument(plan, report) },
      });
    }

    const specs = yieldSpecs(plan, this.registry, weights);

    // The PLANNED allocation: one water-filling solve over every step at the
    // full budget. It is computed once and never reassigned, because it is the
    // object thm:allocation is a theorem about -- the unique optimum and the
    // single shadow price that the KKT residuals below are checked against.
    // Overwriting it with a later re-solve would silently substitute a solution
    // to a DIFFERENT program (fewer steps, smaller budget) for the one whose
    // optimality was certified, and the residuals would then certify nothing.
    const alloc = solve(specs, total);
    const kkt = kktResiduals(specs, alloc);

    // The re-solves are kept as their own artefact rather than folded into the
    // allocation, so the page can show both: what the allocator committed to in
    // advance, and how the price moved once realised cardinalities came in.
    const replans = [];
    let current = alloc;

    const values = {};
    const results = [];
    this._current = results;
    let remaining = total;
    let halted = false;

    for (let i = 0; i < plan.steps.length; i += 1) {
      const step = plan.steps[i];
      const res = this._runStep(step, values, current, remaining);
      results.push(res);

      // Payload ONLY on an answer. Every other verdict binds the empty set, so
      // a downstream step consuming it is starved rather than quietly handed a
      // partial result it cannot tell apart from a complete one.
      values[step.var] =
        res.verdict === Verdict.ANSWER ? res.payload : ResultSet.empty("-");

      // Clamped at zero. A step may overspend its allocation (an all-or-nothing
      // lookup is charged whole), and a negative remainder handed to solve()
      // would be a budget that does not exist.
      remaining = Math.max(0.0, remaining - res.spent);

      if (step.onUnresolved === "fail" && res.verdict !== Verdict.ANSWER) {
        halted = true;
        break;
      }

      // Re-solve over what is left. The allocation is not a schedule fixed in
      // advance: a step that came in under its allocation returns the surplus,
      // and the shadow price on the remainder falls accordingly.
      if (i + 1 < plan.steps.length && remaining > 0) {
        const rest = specs.slice(i + 1);
        if (rest.length) {
          current = solve(rest, remaining);
          replans.push({
            after: step.var,
            remaining,
            allocation: current,
          });
        }
      }
    }

    const ex = new Execution({
      planName: plan.name,
      steps: results,
      allocation: alloc,
      replans,
      checkReport: report,
      kkt,
      haltedEarly: halted,
      requestsIssued: this.registry.totalRequests(),
    });
    ex.emitted = this._emit(plan, ex, values);
    return ex;
  }

  /* ------------------------- per-step dispatch ------------------------- */

  _runStep(step, values, alloc, remaining) {
    const allocated = alloc.get(step.var);
    const shadow = alloc.shadowPrice;

    // (R1) first, for every kind of step. An obstruction upstream is THE
    // obstruction, whatever this step would have run into next.
    const starved = this._checkStarvation(step);
    if (starved) {
      return new StepResult({
        step: step.var,
        source: step.source,
        verdict: Verdict.STARVED,
        diagnosis: starved.reason,
        culprit: starved.culprit,
        allocated,
        shadowPrice: shadow,
        payload: ResultSet.empty("-"),
        expected: starved.expected,
        retention: starved.retention,
      });
    }

    if (step.kind === "map") return this._runMap(step, values, allocated, shadow);
    if (step.kind === "ladder") return this._runLadder(step, values, allocated, shadow);
    if (step.kind !== "from") return this._runSetop(step, values, allocated, shadow);
    return this._runFrom(step, values, allocated, shadow, remaining);
  }

  /**
   * (R1). A bound input that is not an answer starves this step; so does an
   * answer whose measured retention fell below what the plan declared.
   */
  _checkStarvation(step) {
    const inputs = [...step.beta, ...step.operands];
    for (const y of inputs) {
      const r = this._resultFor(y);
      if (!r) continue;
      if (r.verdict !== Verdict.ANSWER) {
        return {
          culprit: y,
          reason: `bound input ${y} returned ${VERDICT_PROSE[r.verdict]}`,
          expected: null,
          retention: null,
        };
      }
      if (step.expectPartial != null && r.retention != null) {
        if (r.retention < step.expectPartial) {
          return {
            culprit: y,
            reason:
              `bound input ${y} retained ${r.retention.toFixed(3)}, below the ` +
              `declared ${step.expectPartial}`,
            expected: step.expectPartial,
            retention: r.retention,
          };
        }
      }
    }
    return null;
  }

  _resultFor(v) {
    return this._current.find((s) => s.step === v) || null;
  }

  _runFrom(step, values, allocated, shadow, remaining) {
    const adapter = this.registry.get(step.source);
    const inputs = {};
    for (const y of step.beta) {
      inputs[y] = values[y] === undefined ? ResultSet.empty("-") : values[y];
    }

    // (R2). Re-tested here even though the static check passed, because the
    // required features can depend on the request's bindings, and a plan
    // rewritten between check and run must not slip through.
    let required;
    try {
      required = resolveFeatures(adapter, step);
    } catch (e) {
      if (e instanceof Refusal) {
        return new StepResult({
          step: step.var,
          source: step.source,
          verdict: Verdict.REFUSED,
          diagnosis: e.message,
          allocated,
          shadowPrice: shadow,
          payload: ResultSet.empty(adapter.namespace),
        });
      }
      throw e;
    }
    const missing = [...required].filter((f) => !adapter.capabilities.has(f));
    if (missing.length) {
      return new StepResult({
        step: step.var,
        source: step.source,
        verdict: Verdict.SURFACE,
        diagnosis: `${step.source} does not declare ${missing.sort().join(", ")}`,
        allocated,
        shadowPrice: shadow,
        payload: ResultSet.empty(adapter.namespace),
      });
    }

    // (R3). The cost is a function of the INPUT cardinality, so it cannot be
    // known statically -- which is why the budget verdict lives here and not
    // in the check.
    let cost;
    try {
      cost = adapter.cost(step, inputs);
    } catch (e) {
      cost = 1.0;
    }
    if (cost > remaining + 1e-9) {
      return new StepResult({
        step: step.var,
        source: step.source,
        verdict: Verdict.REFUSED,
        diagnosis:
          `costs ${cost} requests at this cardinality; ` +
          `${remaining.toFixed(2)} remain`,
        allocated,
        shadowPrice: shadow,
        payload: ResultSet.empty(adapter.namespace),
      });
    }

    // (R4). The per-step ceiling declared by `within N`.
    if (cost > step.budget) {
      return new StepResult({
        step: step.var,
        source: step.source,
        verdict: Verdict.TIMEOUT,
        diagnosis: `costs ${cost}, exceeding the declared ceiling ${step.budget}`,
        allocated,
        spent: step.budget,
        shadowPrice: shadow,
        payload: ResultSet.empty(adapter.namespace),
      });
    }

    let payload;
    try {
      payload = adapter.evaluate(step, inputs);
    } catch (e) {
      if (e instanceof Refusal) {
        return new StepResult({
          step: step.var,
          source: step.source,
          verdict: Verdict.REFUSED,
          diagnosis: e.message,
          allocated,
          spent: cost,
          shadowPrice: shadow,
          loweredForm: adapter.lastLowered,
          payload: ResultSet.empty(adapter.namespace),
        });
      }
      throw e;
    }

    // (R5) then (R6).
    return new StepResult({
      step: step.var,
      source: step.source,
      verdict: payload.size ? Verdict.ANSWER : Verdict.EMPTY,
      // An empty answer carries NO blocker and no diagnosis of obstruction.
      // "Nothing matched" is a fact about the world, not a failure of the
      // machinery, and labelling it as one would be a lie about the corpus.
      diagnosis: payload.size
        ? null
        : "no member of the extent satisfied the request",
      allocated,
      spent: cost,
      shadowPrice: shadow,
      snapshot: adapter.snapshot,
      loweredForm: adapter.lastLowered,
      payload,
    });
  }

  _runMap(step, values, allocated, shadow) {
    const src = values[step.beta[0]] || ResultSet.empty("-");
    if (!this.maps) {
      return new StepResult({
        step: step.var,
        verdict: Verdict.REFUSED,
        diagnosis: "no translation maps are registered",
        allocated,
        shadowPrice: shadow,
        payload: ResultSet.empty("-"),
      });
    }

    let out;
    let stages;
    try {
      [out, stages] = this.maps.applyChain(step.maps, src);
    } catch (e) {
      if (e instanceof Refusal) {
        return new StepResult({
          step: step.var,
          verdict: Verdict.REFUSED,
          diagnosis: e.message,
          allocated,
          shadowPrice: shadow,
          payload: ResultSet.empty("-"),
        });
      }
      throw e;
    }

    const composite = this.maps.survivingFraction(step.maps, src);
    const ret = stages.length ? stages[0].retention : 1.0;
    const amp = stages.length ? stages[0].amplification : null;

    // The expectation is checked HERE, against the MEASURED retention, and a
    // shortfall is starvation rather than an empty answer: the map lost members
    // the plan had declared it needed.
    if (step.expectPartial != null && ret < step.expectPartial) {
      return new StepResult({
        step: step.var,
        verdict: Verdict.STARVED,
        diagnosis: `retained ${ret.toFixed(3)}, below the declared ${step.expectPartial}`,
        culprit: step.beta[0],
        allocated,
        spent: 1.0,
        shadowPrice: shadow,
        retention: ret,
        amplification: amp,
        expected: step.expectPartial,
        compositePower: composite,
        stages,
        payload: ResultSet.empty("-"),
      });
    }

    return new StepResult({
      step: step.var,
      verdict: out.size ? Verdict.ANSWER : Verdict.EMPTY,
      diagnosis: out.size ? null : "the map image is empty",
      allocated,
      spent: 1.0,
      shadowPrice: shadow,
      retention: ret,
      amplification: amp,
      expected: step.expectPartial,
      compositePower: composite,
      stages,
      payload: out,
    });
  }

  _runLadder(step, values, allocated, shadow) {
    const src = values[step.beta[0]] || ResultSet.empty("-");
    // Composite power is the PRODUCT of the rungs. A ladder is a chain of
    // partial correspondences, and the fraction surviving all of them is the
    // product of the fractions surviving each -- which is why a long ladder is
    // a weak one even when every individual rung looks strong.
    const composite = step.rungs.reduce((a, b) => a * b, 1.0);
    if (step.expectPower != null && composite < step.expectPower) {
      return new StepResult({
        step: step.var,
        verdict: Verdict.STARVED,
        diagnosis:
          `composite power ${composite.toFixed(4)} below the declared ` +
          `${step.expectPower}`,
        culprit: step.beta[0],
        allocated,
        // Costs NOTHING. A ladder is arithmetic over a result set that an
        // earlier step already paid a request for; charging it again would
        // bill the reader twice for one retrieval and make the request
        // counter stop being a count of requests.
        spent: 0.0,
        shadowPrice: shadow,
        compositePower: composite,
        expected: step.expectPower,
        payload: ResultSet.empty("-"),
      });
    }
    return new StepResult({
      step: step.var,
      verdict: src.size ? Verdict.ANSWER : Verdict.EMPTY,
      diagnosis: src.size ? null : "the ladder input is empty",
      allocated,
      spent: 0.0,
      shadowPrice: shadow,
      compositePower: composite,
      expected: step.expectPower,
      payload: src,
    });
  }

  _runSetop(step, values, allocated, shadow) {
    const ops = step.operands.map((o) =>
      values[o] === undefined ? ResultSet.empty("-") : values[o]
    );
    const ns = ops.length ? ops[0].namespace : "-";
    const pairs = [];

    if (step.kind === "union") {
      for (const r of ops) {
        for (const i of r.identifiers()) pairs.push([i, r.attrs(i)]);
      }
    } else if (step.kind === "intersect") {
      const [a, b] = ops;
      const bi = new Set(b.identifiers());
      for (const i of a.identifiers()) {
        if (bi.has(i)) pairs.push([i, { ...a.attrs(i), ...b.attrs(i) }]);
      }
    } else if (step.kind === "join") {
      const [a, b] = ops;
      const idx = new Map();
      for (const j of b.identifiers()) {
        const k = b.attrs(j)[step.attr];
        if (k === undefined) continue;
        if (!idx.has(k)) idx.set(k, []);
        idx.get(k).push(j);
      }
      for (const i of a.identifiers()) {
        const k = a.attrs(i)[step.attr];
        if (k === undefined) continue;
        for (const j of idx.get(k) || []) {
          pairs.push([i, { ...a.attrs(i), ["_joined_" + step.attr]: j }]);
        }
      }
    } else if (step.kind === "filter") {
      const a = ops[0];
      const cmp = {
        "==": (x, y) => x === y,
        "!=": (x, y) => x !== y,
        "<": (x, y) => x < y,
        ">": (x, y) => x > y,
        "<=": (x, y) => x <= y,
        ">=": (x, y) => x >= y,
      }[step.op];
      for (const i of a.identifiers()) {
        const row = a.attrs(i);
        // A row LACKING the attribute is dropped, not passed. It was never
        // tested, and treating an absence as a pass would report rows the
        // filter did not examine.
        if (!(step.attr in row)) continue;
        if (cmp(row[step.attr], step.value)) pairs.push([i, row]);
      }
    }

    const out = ResultSet.of(ns, pairs);
    return new StepResult({
      step: step.var,
      verdict: out.size ? Verdict.ANSWER : Verdict.EMPTY,
      diagnosis: out.size ? null : `the ${step.kind} is empty`,
      allocated,
      // Costs NOTHING, for the same reason a ladder does: a union, an
      // intersection, a join or a filter is computation over result sets
      // already in hand. Only `from` (a request) and `map` (a translation
      // lookup) consult anything outside the execution.
      spent: 0.0,
      shadowPrice: shadow,
      payload: out,
    });
  }

  /* --------------------------- Stage 6: Emit --------------------------- */

  _emit(plan, ex, values) {
    const out = {};
    const stepOf = {};
    for (const s of plan.steps) stepOf[s.var] = s;

    for (const e of plan.emits) {
      if (e.divergence) {
        const [ln, rn] = e.divergence;
        const L = new Set((values[ln] || ResultSet.empty("-")).identifiers());
        const R = new Set((values[rn] || ResultSet.empty("-")).identifiers());
        const lo = [...L].filter((x) => !R.has(x)).sort();
        const ro = [...R].filter((x) => !L.has(x)).sort();
        out[e.alias || `divergence_${ln}_${rn}`] = {
          left: ln,
          right: rn,
          left_only: lo,
          right_only: ro,
          symmetric_difference: lo.length + ro.length,
          union_size: new Set([...L, ...R]).size,
          // A COVERAGE statement, not an error count. Neither route is being
          // called wrong; the number is a lower bound on the correspondences
          // that at least one of them fails to resolve.
          interpretation:
            "lower bound on correspondences at least one route fails to " +
            "resolve; neither route contradicts the other",
        };
        continue;
      }

      const r = ex.byVar(e.target);
      const doc = {
        verdict: r ? r.verdict : null,
        verdict_prose: r ? VERDICT_PROSE[r.verdict] : null,
        blocker: r ? r.blocker : null,
        diagnosis: r ? r.diagnosis : null,
      };
      if (r && r.verdict === Verdict.ANSWER) doc.payload = r.payload.toJSON();

      if (e.provenance) doc.coverage = this._coverage(ex, e.target, stepOf);

      if (e.intension) {
        // An extensional answer offered as the extension of an intensional
        // claim needs its gap named. "These are the datasets" and "these are
        // the datasets we could reach" are different assertions, and only the
        // second one is true.
        doc.admissibility = {
          intension: e.intension,
          gap: e.gap,
          admissible: e.gap === null,
          note:
            e.gap === null
              ? "the extension is offered as complete"
              : `the extension is offered subject to a ${e.gap} gap`,
        };
      }
      out[e.target] = doc;
    }
    return out;
  }

  _coverage(ex, target, stepOf) {
    const anc = this._ancestry(ex, target, stepOf);
    const sources = [
      ...new Set(anc.map((v) => (ex.byVar(v) || {}).source).filter(Boolean)),
    ].sort();
    const snaps = {};
    for (const s of sources) {
      const a = this.registry.adapters[s];
      if (a) snaps[s] = a.snapshot;
    }
    return {
      steps: anc,
      sources,
      snapshots: snaps,
      attrition: this._attrition(ex, anc),
    };
  }

  _ancestry(ex, target, stepOf) {
    const out = [];
    const seen = new Set();
    const stack = [target];
    while (stack.length) {
      const v = stack.pop();
      if (seen.has(v)) continue;
      seen.add(v);
      out.push(v);
      const step = stepOf[v];
      if (step) stack.push(...step.beta, ...step.operands);
    }
    return out.reverse();
  }

  _attrition(ex, vars) {
    const rows = [];
    for (const v of vars) {
      const r = ex.byVar(v);
      if (!r) continue;
      rows.push({
        step: v,
        n: r.payload ? r.payload.size : 0,
        retention: r.retention,
        amplification: r.amplification,
      });
    }
    return rows;
  }
}
