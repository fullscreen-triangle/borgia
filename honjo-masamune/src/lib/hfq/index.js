/**
 * The HFQ runtime, assembled.
 *
 * This module is the browser's whole entry point: it turns the serialised
 * fixture into a live registry and hands back the six-stage pipeline. It is
 * also the file where the port's central constraint is enforced by
 * construction rather than by discipline.
 *
 * NO NETWORK I/O. There is no `fetch` here, no `XMLHttpRequest`, and no
 * dynamic import. The fixture arrives as a static JSON import that the bundler
 * inlines, so a page built on this module resolves every step against data
 * that shipped with it. The reason is not convenience: the prototype's claims
 * are properties of the COMPILER -- that a plan is refused before contact,
 * that a starved step names its culprit, that the request counter reads zero
 * on a static refusal -- and a live third-party service can neither confirm
 * nor refute any of them. Pointing this at a real endpoint would replace a
 * checkable claim with an anecdote.
 *
 * The corresponding ethical constraint is the same one the paper states: we do
 * not probe someone else's public service with requests designed to elicit its
 * internal behaviour.
 */

import fixture from "./fixture.json";

import {
  Verdict,
  Blocker,
  VERDICT_PROSE,
  blockerOf,
  FEAT,
  isFeature,
  Refusal,
  ResultSet,
  TranslationMap,
} from "./model.js";

import { parsePlan, PlanError } from "./parse.js";

import {
  PREDICATE_FEATURES,
  loadPredicateFeatures,
  biocatRequiredFeatures,
  resolveFeatures,
  Adapter,
  GraphPatternAdapter,
  FilteringGraphAdapter,
  ProvenanceAdapter,
  SequenceAdapter,
  OntologyAdapter,
  LookupAdapter,
  MapAdapter,
  Registry,
} from "./adapters.js";

import {
  CapabilityFailure,
  CheckReport,
  check,
  refusalDocument,
} from "./check.js";

import { YieldSpec, Allocation, solve, kktResiduals } from "./allocate.js";

import { StepResult, Execution, Executor, yieldSpecs } from "./execute.js";

/**
 * Adapter classes by the name the Python side reported for `type(ad).__name__`.
 *
 * The fixture carries the class name rather than a hand-written `kind` tag
 * precisely so this table cannot drift from the reference: if the Python side
 * grows a seventh adapter, the fixture names it and this lookup fails loudly
 * instead of silently constructing the base class and losing the behaviour.
 */
const ADAPTER_KINDS = {
  Adapter,
  GraphPatternAdapter,
  FilteringGraphAdapter,
  ProvenanceAdapter,
  SequenceAdapter,
  OntologyAdapter,
  LookupAdapter,
};

/**
 * Build a registry and a map adapter from a serialised fixture.
 *
 * Every constructor here reads snake_case option keys, which is why the
 * fixture rows are passed through almost verbatim: the serialiser wrote the
 * names the constructors read, so there is no rename step in which a field
 * could be dropped. A field the Python side did not populate is simply absent
 * from the JSON, and the constructor's own default applies -- the same default
 * the reference used.
 */
export function buildRegistry(fx = fixture) {
  loadPredicateFeatures(fx.predicate_features || {});

  const registry = new Registry();
  for (const [name, spec] of Object.entries(fx.sources || {})) {
    const Kind = ADAPTER_KINDS[spec.kind];
    if (!Kind) {
      throw new Error(
        `fixture declares source '${name}' of unknown kind '${spec.kind}'`
      );
    }
    registry.register(new Kind({ ...spec }));
  }

  const maps = {};
  for (const [name, spec] of Object.entries(fx.maps || {})) {
    maps[name] = new TranslationMap(spec);
  }
  const mapAdapter = new MapAdapter({ maps });

  return { registry, maps: mapAdapter, plans: fx.plans || {} };
}

/**
 * Parse and run one plan source against a freshly-built registry.
 *
 * A fresh registry per run is deliberate. The request counter is an observable
 * claim -- a statically refused plan must report zero -- and a counter shared
 * across runs would accumulate, turning a claim about THIS plan into a claim
 * about the session.
 */
export function runSource(source, { budget = null, weights = {} } = {}) {
  const { registry, maps } = buildRegistry();
  const plan = parsePlan(source);
  const ex = new Executor(registry, maps).run(plan, budget, { weights });
  return { plan, execution: ex };
}

/** Run one of the twenty-four plans that shipped with the fixture. */
export function runPlan(name, opts = {}) {
  const source = (fixture.plans || {})[name];
  if (source === undefined) throw new Error(`no plan named '${name}'`);
  return runSource(source, opts);
}

export function planNames() {
  return Object.keys(fixture.plans || {}).sort();
}

export function planSource(name) {
  return (fixture.plans || {})[name] ?? null;
}

export function sourceSummary(fx = fixture) {
  return Object.entries(fx.sources || {})
    .map(([name, s]) => ({
      name,
      kind: s.kind,
      namespace: s.namespace,
      snapshot: s.snapshot,
      capabilities: [...(s.capabilities || [])].sort(),
      extent: {
        triples: (s.triples || []).length || undefined,
        records: Object.keys(s.records || {}).length || undefined,
        sequences: Object.keys(s.sequences || {}).length || undefined,
        parents: Object.keys(s.parents || {}).length || undefined,
        links: Object.keys(s.links || {}).length || undefined,
      },
    }))
    .sort((a, b) => (a.name < b.name ? -1 : 1));
}

export {
  fixture,
  // model
  Verdict,
  Blocker,
  VERDICT_PROSE,
  blockerOf,
  FEAT,
  isFeature,
  Refusal,
  ResultSet,
  TranslationMap,
  // parse
  parsePlan,
  PlanError,
  // adapters
  PREDICATE_FEATURES,
  loadPredicateFeatures,
  biocatRequiredFeatures,
  resolveFeatures,
  Adapter,
  GraphPatternAdapter,
  FilteringGraphAdapter,
  ProvenanceAdapter,
  SequenceAdapter,
  OntologyAdapter,
  LookupAdapter,
  MapAdapter,
  Registry,
  // check
  CapabilityFailure,
  CheckReport,
  check,
  refusalDocument,
  // allocate
  YieldSpec,
  Allocation,
  solve,
  kktResiduals,
  // execute
  StepResult,
  Execution,
  Executor,
  yieldSpecs,
};
