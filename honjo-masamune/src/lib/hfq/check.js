/**
 * Stage 3 of the pipeline: Check.
 *
 * The static admissibility test. It runs BEFORE any request is issued, and its
 * refusal is the single most important behaviour on the page: a plan whose
 * demands exceed a source's declared capability is refused with the obstacle
 * named, and the request counter stays at zero.
 *
 * That is the whole difference from a query language. A SPARQL endpoint given
 * a query it cannot serve returns an empty result -- indistinguishable, over
 * the wire, from a query it served perfectly against a corpus that held
 * nothing. There is one bit where there should be six.
 *
 * Ported from hfq/check.py.
 */

import { FEAT, Verdict } from "./model.js";
import { resolveFeatures } from "./adapters.js";

export class CapabilityFailure {
  constructor({ step, source, required, declared, missing, line }) {
    this.step = step;
    this.source = source;
    this.required = [...required].sort();
    this.declared = [...declared].sort();
    this.missing = [...missing].sort();
    this.line = line;
  }

  /**
   * The refusal must name the REAL obstacle. "Query failed" is not a refusal,
   * it is a shrug: it tells the reader nothing about whether to rewrite the
   * plan, raise the budget, or go find a different corpus.
   */
  get reason() {
    return (
      `step ${this.step} asks ${this.source} for ` +
      `${this.missing.join(", ")}, which it does not declare`
    );
  }

  toJSON() {
    return {
      step: this.step,
      source: this.source,
      required: this.required,
      declared: this.declared,
      missing: this.missing,
      line: this.line,
      reason: this.reason,
    };
  }
}

export class CheckReport {
  constructor({ failures, operations, requirements, nSteps }) {
    this.failures = failures;
    this.operations = operations;
    this.requirements = requirements;
    this.nSteps = nSteps;
  }

  /** True when no step demands a capability its source does not declare. */
  get wellCapability() {
    return this.failures.length === 0;
  }

  /**
   * The linear bound. The check is one pass over the steps, and within a step
   * one pass over a fixed eleven-symbol vocabulary -- so the work is bounded by
   * the product, independent of corpus size. A plan against a billion triples
   * is checked in the same number of operations as one against ten.
   */
  get bound() {
    return this.nSteps * FEAT.length;
  }

  toJSON() {
    return {
      well_capability: this.wellCapability,
      failures: this.failures.map((f) => f.toJSON()),
      operations: this.operations,
      n_steps: this.nSteps,
      bound: this.bound,
      requirements: Object.fromEntries(
        Object.entries(this.requirements).map(([k, v]) => [k, [...v].sort()])
      ),
    };
  }
}

export function check(plan, registry) {
  const failures = [];
  const requirements = {};
  let operations = 0;

  // EVERY step, not only the sourced ones. The bound of thm:static(a) is
  // m|Feat| where m is the length of the PLAN -- it is the claim that checking
  // is linear in the plan and independent of the corpus, so it must be stated
  // over the plan the reader wrote. Counting only `from` steps would quietly
  // report a bound for a shorter plan than the one that was checked, and (V2)
  // compares `operations` against exactly this number.
  const nSteps = plan.steps.length;

  for (const step of plan.steps) {
    // Map, ladder, and set steps carry no source capability demand: they
    // consume result sets that earlier steps already paid for. Charging them
    // against a source would refuse plans that are perfectly admissible.
    if (step.kind !== "from") continue;

    const adapter = registry.get(step.source);
    const required = resolveFeatures(adapter, step);
    requirements[step.var] = required;

    const missing = new Set();
    for (const f of required) {
      operations += 1; // one test per required symbol -- the bound is per-FEAT
      if (!adapter.capabilities.has(f)) missing.add(f);
    }

    if (missing.size) {
      failures.push(
        new CapabilityFailure({
          step: step.var,
          source: step.source,
          required,
          declared: adapter.capabilities,
          missing,
          line: step.line,
        })
      );
    }
  }

  return new CheckReport({ failures, operations, requirements, nSteps });
}

/**
 * The document a statically-refused plan emits.
 *
 * This is NOT an empty result, and the difference is the entire content of the
 * one-bit corollary. An empty result says "the corpus holds nothing matching".
 * This says "the plan was never run, here is the step, here is the source, here
 * is the capability it lacks". A reader can act on the second and cannot act on
 * the first.
 */
export function refusalDocument(plan, report) {
  return {
    plan: plan.name,
    outcome: "refused_statically",
    reason: "ill-capability plan; no request was issued",
    failures: report.failures.map((f) => f.toJSON()),
    steps: [],
  };
}

export const REFUSED_VERDICT = Verdict.SURFACE;
