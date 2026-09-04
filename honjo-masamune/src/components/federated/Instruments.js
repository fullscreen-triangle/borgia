/**
 * Instruments — the controls the notebook's own prose tells the reader to operate.
 *
 * Cell [9] says "try lowering the budget until a step reports refused" and cell
 * [8] is titled "Budget, shadow price, and a certificate". Neither shipped with
 * a control. This module supplies them, and keeps every perturbation here so
 * that pages/federated.js stays a document rather than a program.
 *
 * Three rules govern everything below.
 *
 *   1. NO NETWORK. src/lib/hfq/index.js states the constraint the paper states:
 *      every adapter resolves against a local fixture, because the claims are
 *      properties of the compiler and a live service can neither confirm nor
 *      refute them. Nothing here fetches; the fixture is a static import.
 *
 *   2. NEVER MUTATE THE FIXTURE SINGLETON. Perturbations structuredClone it.
 *      Note also that buildRegistry() calls loadPredicateFeatures(), which
 *      writes into a module-global PREDICATE_FEATURES that is never cleared
 *      (adapters.js:30). We perturb sources[*].capabilities only, never
 *      predicate_features, so that global stays exactly what every other cell
 *      on the page is reading.
 *
 *   3. A KNOB MUST MOVE THE FIGURE. Every control here was measured against the
 *      real engine before it was built. One that looked promising was rejected:
 *      the `routes` plan's `within 4` reads like an ontology-depth dial, but
 *      swept 1 through 8 every figure is identical -- the fixture's CHEBI
 *      hierarchy is exhausted at depth 1. A slider that moves without changing
 *      anything is worse than no slider, so cell [11] gets a static figure.
 *
 * Re-running is cheap enough to do on a drag frame: one plan is 0.35 ms and all
 * twenty-four are 5.3 ms, both well inside a 16 ms frame. Re-running rather than
 * caching is also the honest choice -- runSource builds a fresh registry per
 * call by design, because a shared request counter "would turn a claim about
 * THIS plan into a claim about the session."
 */

import { useMemo, useState } from "react";
import {
  fixture,
  buildRegistry,
  parsePlan,
  Executor,
  planSource,
  sourceSummary,
  FEAT,
} from "@/lib/hfq";
import { LinePlot, StepBars, fmt as dfmt } from "@/components/workbench/D3Charts";
import {
  T,
  MONO,
  VERDICT_COLOUR,
  VerdictChip,
  Callout,
  Slider,
  Toggle,
  Instrument,
  diagnosisText,
} from "./Cells";

/* ================================================================== */
/*  Perturbation                                                       */
/* ================================================================== */

/**
 * Run a plan against a registry built from a perturbed copy of the fixture.
 *
 * runSource() hardcodes buildRegistry() with no fixture argument, so a
 * capability perturbation has to assemble the three parts itself. This is the
 * same sequence runSource performs, with a cloned fixture substituted.
 *
 * `caps` maps source name -> array of capability strings, replacing whatever
 * the fixture declared. Anything absent from `caps` is left as-is.
 */
export function perturbRun(source, { caps = null, budget = null, weights = {} } = {}) {
  let fx = fixture;
  if (caps && Object.keys(caps).length) {
    fx = structuredClone(fixture);
    for (const [name, list] of Object.entries(caps)) {
      if (fx.sources && fx.sources[name]) fx.sources[name].capabilities = [...list];
    }
  }
  const { registry, maps } = buildRegistry(fx);
  const plan = parsePlan(source);
  const execution = new Executor(registry, maps).run(plan, budget, { weights });
  return { plan, execution };
}

/** A run's verdict sequence, joined — the thing that either changes or does not. */
function signature(ex) {
  return ex.steps.map((s) => s.verdict).join(",");
}

/**
 * Every budget from 0 to `hi`, with the contiguous runs of identical verdict
 * signature collapsed into regimes.
 *
 * Budget is not a dial with hi+1 settings; it has a handful of thresholds, and
 * the regimes are what the reader should see. 41 runs is about 14 ms, so this
 * belongs in a useMemo on mount, not on a drag frame.
 */
export function sweepBudget(source, hi = 40, weights = {}) {
  const points = [];
  for (let b = 0; b <= hi; b += 1) {
    let ex;
    try {
      ex = perturbRun(source, { budget: b, weights }).execution;
    } catch {
      continue;
    }
    points.push({
      budget: b,
      requests: ex.requestsIssued,
      answers: ex.steps.filter((s) => s.verdict === "answer").length,
      steps: ex.steps.length,
      signature: signature(ex),
      price: ex.allocation ? ex.allocation.shadowPrice : null,
    });
  }
  return { points, regimes: regimesOf(points) };
}

/** Contiguous budget intervals sharing a verdict signature. */
export function regimesOf(points) {
  const out = [];
  for (const p of points) {
    const last = out[out.length - 1];
    if (last && last.signature === p.signature) last.to = p.budget;
    else out.push({ from: p.budget, to: p.budget, signature: p.signature, requests: p.requests });
  }
  return out;
}

/* Shared bits of chrome ------------------------------------------------ */

function Note({ children }) {
  return (
    <div style={{ fontSize: 11.5, color: T.dim, lineHeight: 1.65, margin: "8px 0 0" }}>
      {children}
    </div>
  );
}

function Mono({ children, colour }) {
  return (
    <span style={{ fontFamily: MONO, fontSize: 11.5, color: colour ?? T.text }}>{children}</span>
  );
}

function VerdictRow({ steps }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, margin: "10px 0 0" }}>
      {steps.map((s, i) => (
        <VerdictChip key={`${s.step}${i}`} verdict={s.verdict} title={`${s.step} @ ${s.source ?? "—"}`} />
      ))}
    </div>
  );
}

/* ================================================================== */
/*  1. WaterFilling — one price, not a priority queue                   */
/* ================================================================== */

/**
 * Budget plus one objective-weight slider per step, over the live plan.
 *
 * The sliders are labelled "objective weight" and never "priority", because
 * that is what they are: coefficients w_v in the concave objective the
 * allocator maximises. Calling them priorities would suggest a scheduler
 * running steps in an order, and the whole point of the figure is that there
 * is no order -- there is one price, and every step buys at it.
 *
 * Weights key on STEP NAME (execute.js:197), which is why they are generated
 * from the parsed plan rather than hardcoded.
 */
export function WaterFilling({ source, budget, onBudget, hiBudget = 40 }) {
  const [weights, setWeights] = useState({});

  const parsed = useMemo(() => {
    try {
      return parsePlan(source);
    } catch {
      return null;
    }
  }, [source]);

  const vars = parsed ? parsed.steps.map((s) => s.var) : [];

  // Baseline at the same budget with no weights, so "what did MY weight do"
  // is answerable against something other than the reader's memory.
  const base = useMemo(() => {
    try {
      return perturbRun(source, { budget }).execution;
    } catch {
      return null;
    }
  }, [source, budget]);

  const run = useMemo(() => {
    try {
      return perturbRun(source, { budget, weights }).execution;
    } catch (e) {
      return null;
    }
  }, [source, budget, weights]);

  if (!run || !parsed) return null;

  const alloc = run.allocation;
  const rows = run.steps.map((s) => ({
    label: s.step,
    value: alloc ? alloc.get(s.step) : 0,
    colour: VERDICT_COLOUR[s.verdict] ?? T.dim,
  }));

  // Did upweighting a later step degrade an earlier one? This is the
  // observation the instrument exists for, and it is checked against the
  // unweighted run at the same budget so budget is not the confound.
  const flipped =
    base && base.steps.length === run.steps.length
      ? run.steps
          .map((s, i) => ({ v: s.step, was: base.steps[i].verdict, now: s.verdict }))
          .filter((d) => d.was === "answer" && d.now !== "answer")
      : [];

  // A step allocated nothing that answered anyway: all-or-nothing steps are
  // charged whole out of fixed_spent before the concave program sees the
  // budget, so being off the support does not mean being unfunded.
  const freeRiders = alloc
    ? run.steps.filter((s) => s.verdict === "answer" && !(alloc.get(s.step) > 0))
    : [];

  const residual = run.kkt ? run.kkt.max_stationarity_residual : null;

  // Measured: at budget 6, w={enzymes:3} doubles p* from 0.40 to 0.80 and
  // every verdict stays exactly where it was. A reader who moves a slider,
  // watches the price move, and sees no verdict change should be told that is
  // the expected case rather than left assuming the control is broken.
  const touched = Object.entries(weights).filter(([, w]) => w !== 1);
  const priceMoved =
    base && base.allocation && alloc &&
    Math.abs(alloc.shadowPrice - base.allocation.shadowPrice) > 1e-9;
  const verdictsSame = base && signature(run) === signature(base);

  return (
    <div>
      <Instrument title="allocation controls">
        <Slider
          label="budget"
          value={budget}
          min={0}
          max={hiBudget}
          step={1}
          onChange={onBudget}
          hint="total effort the allocator may split across steps"
        />
        {vars.map((v) => (
          <Slider
            key={v}
            label={`objective weight · ${v}`}
            value={weights[v] ?? 1}
            min={0.2}
            max={3}
            step={0.1}
            fmt={(x) => x.toFixed(1)}
            onChange={(x) => setWeights((w) => ({ ...w, [v]: x }))}
            hint="coefficient in the objective, not a queue position"
          />
        ))}
      </Instrument>

      <StepBars
        rows={rows}
        height={176}
        yLabel="effort"
        rule={
          alloc && alloc.shadowPrice > 0
            ? { y: alloc.shadowPrice, label: `p* = ${dfmt(alloc.shadowPrice, 3)}`, colour: T.warn }
            : null
        }
      />
      <VerdictRow steps={run.steps} />

      <Note>
        bars are effort, coloured by the verdict that effort bought. A dashed
        baseline is a step the allocator funded at zero. The horizontal rule is
        the single shadow price p*: on the support, every step&rsquo;s marginal
        yield equals it — that equality is what water-filling means.
      </Note>

      {flipped.length > 0 && (
        <Callout tone="warn" title="you moved a weight and a different step got worse">
          <Mono colour={T.warn}>
            {flipped.map((d) => `${d.v}: ${d.was} → ${d.now}`).join("   ")}
          </Mono>
          <div style={{ marginTop: 6 }}>
            There is no priority queue here. Raising one weight raises the price
            everyone pays, and a step that could just afford its result at the
            old price cannot at the new one. The allocation is a single market,
            not an ordering.
          </div>
        </Callout>
      )}

      {freeRiders.length > 0 && (
        <Callout tone="violet" title="allocated nothing, answered anyway">
          <Mono colour={T.violet}>
            {freeRiders.map((s) => `${s.step}: effort 0, verdict ${s.verdict}`).join("   ")}
          </Mono>
          <div style={{ marginTop: 6 }}>
            An all-or-nothing step is charged whole out of{" "}
            <Mono colour={T.dim}>fixed_spent</Mono> before the concave program
            runs, so it never appears on the support even when it is the step
            that answered. The objective is not concave over these steps —{" "}
            <Mono colour={T.dim}>rem:concavity-fails</Mono>, and cell [12] lists
            it as a limitation.
          </div>
        </Callout>
      )}

      {touched.length > 0 && priceMoved && verdictsSame && flipped.length === 0 && (
        <Callout tone="dim" title="the price moved; the verdicts did not">
          <Mono colour={T.dim}>
            p*: {dfmt(base.allocation.shadowPrice, 3)} → {dfmt(alloc.shadowPrice, 3)}
            {"   "}verdicts unchanged
          </Mono>
          <div style={{ marginTop: 6 }}>
            This is the ordinary case, not a broken control. A weight changes
            what the allocation costs at the margin; it only changes an outcome
            when the new split pushes some step across the threshold where its
            result stops being affordable. Between thresholds you are paying a
            different price for the same answers.
          </div>
        </Callout>
      )}

      {residual !== null && residual > 1e-6 && (
        <Callout tone="err" title="stationarity residual is non-zero">
          <Mono colour={T.err}>max_stationarity_residual = {residual}</Mono>
          <div style={{ marginTop: 6 }}>
            This should never fire. It means a step on the support has marginal
            yield different from p*, so the allocation is not the optimum the
            certificate claims. The check exists because a certificate nobody
            can watch fail is not a certificate.
          </div>
        </Callout>
      )}
    </div>
  );
}

/* ================================================================== */
/*  2. BudgetSweep — where the thresholds are                           */
/* ================================================================== */

/**
 * Requests issued against budget, over the whole range, as a step function.
 *
 * curve="step" is a semantic requirement, not a style choice. Requests are
 * integers and budget moves in whole units; a monotone interpolation between
 * (4, 0) and (6, 1) draws a curve through (5, 0.5), which asserts that a budget
 * of five issues half a request. It does not. The figure would be a literal
 * falsehood drawn in the default curve.
 */
export function BudgetSweep({ source, budget, hi = 40 }) {
  const { points, regimes } = useMemo(() => sweepBudget(source, hi), [source, hi]);

  if (!points.length) return null;

  const here = points.find((p) => p.budget === budget) ?? points[points.length - 1];
  const idx = points.indexOf(here);

  // Non-monotonicity: more budget bought fewer requests. Measured on mark_q1
  // at b = 3 -> 4, where at b=3 the whole budget went to one cheap step that
  // answered outright, and by b=4 water-filling had spread it thin enough that
  // nothing finished and no request was issued at all.
  //
  // The callout stays lit for as long as the reader is inside the regime the
  // drop opened, rather than for one budget value -- a warning that vanishes
  // on the next tick of the slider is a warning nobody reads.
  const drop = [];
  for (let i = 1; i < points.length; i += 1) {
    if (points[i].requests < points[i - 1].requests) {
      const reg = regimes.find((r) => points[i].budget >= r.from && points[i].budget <= r.to);
      drop.push({ from: points[i - 1], to: points[i], regime: reg });
    }
  }
  const activeDrop = drop.find(
    (d) => d.regime && budget >= d.regime.from && budget <= d.regime.to
  );

  const bands = regimes
    .filter((r) => r.to > r.from)
    .map((r, i) => ({
      from: r.from,
      to: r.to + 1,
      colour: i % 2 ? T.panel : T.surface,
      opacity: 0.5,
    }));

  return (
    <div>
      <LinePlot
        height={210}
        xLabel="budget"
        yLabel="requests issued"
        curve="step"
        showPoints={false}
        xDomain={[0, hi]}
        bands={bands}
        marker={budget}
        series={[
          {
            name: "requests",
            colour: T.accent,
            points: points.map((p) => ({ x: p.budget, y: p.requests, label: `b=${p.budget} → ${p.requests} requests` })),
          },
          {
            name: "answers",
            colour: T.ok,
            points: points.map((p) => ({ x: p.budget, y: p.answers, label: `b=${p.budget} → ${p.answers} answers` })),
          },
        ]}
      />

      <Note>
        stepped, because both series are integer counts over integer budgets —
        a smooth curve here would draw half a request at budget 4.5. Alternating
        bands are the {regimes.length} distinct verdict regimes; the vertical
        line is where the slider is. The sweep is computed once on mount, not
        per drag: {points.length} runs at 0.35 ms each. The weight sliders above,
        by contrast, do re-execute the engine on every frame.
      </Note>

      <div style={{ margin: "10px 0 0", fontFamily: MONO, fontSize: 11 }}>
        <span style={{ color: T.dim }}>at b={here.budget}: </span>
        <span style={{ color: T.accent }}>{here.requests} requests</span>
        <span style={{ color: T.dim }}> · </span>
        <span style={{ color: T.ok }}>{here.answers}/{here.steps} answered</span>
        {here.price !== null && (
          <>
            <span style={{ color: T.dim }}> · p* = </span>
            <span style={{ color: T.warn }}>{dfmt(here.price, 3)}</span>
          </>
        )}
      </div>

      {activeDrop && (
        <Callout tone="orange" title="you paid more and got less">
          <Mono colour={T.orange}>
            b={activeDrop.from.budget} → {activeDrop.from.requests} requests
            {"   "}
            b={activeDrop.to.budget} → {activeDrop.to.requests} requests
          </Mono>
          <div style={{ marginTop: 6 }}>
            At the lower budget almost all of it went to one step, cheap enough
            to finish and answer outright. At the higher budget water-filling
            spread the same money across steps that each now get too little to
            complete. The allocation is optimal for expected yield under the
            stated objective — it is not optimal for the number of requests
            issued, and nothing claimed it was.
          </div>
        </Callout>
      )}

      {regimes.length > 1 && regimes.length < points.length / 3 && (
        <Callout tone="accent" title="budget is not a dial">
          <div>
            {points.length} budgets produce only {regimes.length} distinct
            outcomes. Between thresholds the extra money changes nothing at all:
            the plan has {regimes.length} regimes, and everything in between is
            the same plan with slack.
          </div>
        </Callout>
      )}

      {idx >= 0 && here.requests === points[points.length - 1].requests &&
        here.requests === points[Math.max(0, points.length - 6)].requests && (
        <Callout tone="dim" title="saturated">
          <div>
            Past this point more budget buys nothing. Whatever is still unanswered
            is not unanswered for want of money — see the{" "}
            <Mono colour={T.violet}>surface</Mono> verdicts, which no budget in
            this range removes.
          </div>
        </Callout>
      )}
    </div>
  );
}

/* ================================================================== */
/*  3. PriceWalk — the price after each step resolves                   */
/* ================================================================== */

/**
 * The shadow price across the planned allocation and each re-solve.
 *
 * The prior version of this rendered a labelled text table, and its docstring
 * argued that a sparkline would hide the labels. That is right about a static
 * render. Under a slider the reader needs the trajectory's SHAPE — whether the
 * price falls, and where it jumps — so the chart leads and the table stays
 * underneath, with the labels surviving as axis ticks.
 *
 * replans is NOT one entry per step. execute.js:293 only re-solves when there
 * is a next step and budget left, and a step with onUnresolved "fail" breaks
 * the loop outright. A short walk is a real result and is reported as one.
 */
export function PriceWalk({ source, budget, weights = {} }) {
  const run = useMemo(() => {
    try {
      return perturbRun(source, { budget, weights }).execution;
    } catch {
      return null;
    }
  }, [source, budget, weights]);

  if (!run) return null;

  // A statically refused plan never reaches the allocator at all.
  if (!run.allocation) {
    return (
      <Callout tone="warn" title="there is no price walk">
        <div>
          This plan was refused by the capability check before allocation, at{" "}
          <Mono colour={T.warn}>requests_issued = {run.requestsIssued}</Mono>.
          There is no allocation, so there is no shadow price and nothing to
          walk. That absence is the result, not a gap in the figure.
        </div>
      </Callout>
    );
  }

  const walk = [
    { label: "planned", price: run.allocation.shadowPrice, remaining: run.allocation.budget },
    ...run.replans.map((r) => ({
      label: `after ${r.after}`,
      price: r.allocation.shadowPrice,
      remaining: r.remaining,
    })),
  ];

  const rises = walk.slice(1).filter((w, i) => w.price > walk[i].price + 1e-12);
  const truncated = run.replans.length < run.steps.length - 1;

  return (
    <div>
      <LinePlot
        height={186}
        xLabel="re-solve"
        yLabel="shadow price p*"
        curve="linear"
        series={[
          {
            name: "p*",
            colour: T.warn,
            points: walk.map((w, i) => ({ x: i, y: w.price, label: `${w.label}: p* = ${dfmt(w.price, 4)}` })),
          },
        ]}
      />

      <div style={{ overflowX: "auto", marginTop: 10 }}>
        <table style={{ borderCollapse: "collapse", fontFamily: MONO, fontSize: 11.5, minWidth: 320 }}>
          <thead>
            <tr>
              {["point", "remaining", "p*"].map((h) => (
                <th
                  key={h}
                  style={{
                    textAlign: "left",
                    padding: "4px 14px 4px 0",
                    color: T.dim,
                    borderBottom: `1px solid ${T.border}`,
                    fontWeight: 500,
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {walk.map((w, i) => (
              <tr key={w.label}>
                <td style={{ padding: "4px 14px 4px 0", color: T.text }}>{w.label}</td>
                <td style={{ padding: "4px 14px 4px 0", color: T.dim }}>{dfmt(w.remaining, 3)}</td>
                <td style={{ padding: "4px 14px 4px 0", color: i === 0 ? T.accent : T.warn }}>
                  {dfmt(w.price, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {truncated && (
        <Callout tone="accent" title="the walk is shorter than the plan">
          <Mono colour={T.accent}>
            {run.replans.length} re-solves over {run.steps.length} steps
          </Mono>
          <div style={{ marginTop: 6 }}>
            The executor re-solves only when a next step exists and budget
            remains, and a step declared <Mono colour={T.dim}>on unresolved fail</Mono>{" "}
            stops the loop where it stands. The missing points are not missing
            data — they are places the plan never reached.
          </div>
        </Callout>
      )}

      {rises.length > 0 && (
        <Callout tone="orange" title="the price went up">
          <div>
            Fewer steps are competing after a resolve, so demand fell — but the
            remaining budget fell faster. The price is a ratio, and it rises
            whenever the numerator outlasts the denominator.
          </div>
        </Callout>
      )}
    </div>
  );
}

/* ================================================================== */
/*  4. CapabilityGrid — declare less, declare more                      */
/* ================================================================== */

/**
 * The eleven-symbol vocabulary, taken from the engine rather than retyped.
 *
 * The check's linear bound is one pass per step over exactly this set
 * (check.js:117), so a grid built from anything else would be a grid of
 * capabilities the checker never tests.
 */
const ALL_CAPS = FEAT;

/**
 * Withdraw or grant a capability from any source, and watch what the check does.
 *
 * This is the instrument that lets the reader PRODUCE the limitation the paper
 * confesses in sec:limits, rather than read about it.
 *
 * Withdrawing is safe and loud: revoke `neg` from RXN and mark_q1 collapses to
 * a single surface verdict at zero requests, with the failure naming the step,
 * the source, what was required, what was declared, what is missing, and the
 * line number.
 *
 * Granting is unsound and INVISIBLE: give RXN `regex`, which it does not
 * implement, and nothing changes. The check passes. That is the honest finding
 * and it is not dressed up as a failure the figure can show, because it isn't
 * one — the alarm is carried by the prose, and the identical figure IS the
 * evidence.
 */
export function CapabilityGrid({ planName = "mark_q1", budget = null }) {
  const src0 = useMemo(() => planSource(planName), [planName]);

  // Only the sources this plan actually reaches. The fixture declares
  // thirteen; a grid of all of them against eleven capabilities is 143
  // switches, of which two matter. Showing the two the plan depends on is the
  // difference between an instrument and a wall.
  const sources = useMemo(() => {
    const all = sourceSummary();
    let used = null;
    try {
      used = new Set(parsePlan(src0).steps.map((s) => s.source).filter(Boolean));
    } catch {
      used = null;
    }
    return used && used.size ? all.filter((s) => used.has(s.name)) : all;
  }, [src0]);

  const declared = useMemo(
    () => Object.fromEntries(sources.map((s) => [s.name, new Set(s.capabilities)])),
    [sources]
  );

  const [caps, setCaps] = useState(() =>
    Object.fromEntries(sources.map((s) => [s.name, [...s.capabilities]]))
  );

  const source = src0;

  const base = useMemo(() => {
    try {
      return perturbRun(source, { budget }).execution;
    } catch {
      return null;
    }
  }, [source, budget]);

  const run = useMemo(() => {
    try {
      return perturbRun(source, { caps, budget }).execution;
    } catch {
      return null;
    }
  }, [source, caps, budget]);

  if (!base || !run || !source) return null;

  const toggle = (src, cap) =>
    setCaps((c) => {
      const cur = new Set(c[src] ?? []);
      if (cur.has(cap)) cur.delete(cap);
      else cur.add(cap);
      return { ...c, [src]: [...cur].sort() };
    });

  const withdrawn = [];
  const granted = [];
  for (const s of sources) {
    for (const cap of ALL_CAPS) {
      const on = (caps[s.name] ?? []).includes(cap);
      const dec = declared[s.name].has(cap);
      if (dec && !on) withdrawn.push(`${s.name}:${cap}`);
      if (!dec && on) granted.push(`${s.name}:${cap}`);
    }
  }

  const changed = signature(run) !== signature(base) || run.requestsIssued !== base.requestsIssued;
  const failure = run.checkReport && !run.checkReport.wellCapability
    ? run.checkReport.failures[0]
    : null;

  return (
    <div>
      <Instrument title={`capabilities · plan ${planName}`}>
        {sources.map((s) => (
          <div key={s.name} style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            <span style={{ fontFamily: MONO, fontSize: 11.5, color: T.text, minWidth: 58 }}>
              {s.name}
            </span>
            {ALL_CAPS.map((cap) => (
              <Toggle
                key={cap}
                label={cap}
                on={(caps[s.name] ?? []).includes(cap)}
                declared={declared[s.name].has(cap)}
                onChange={() => toggle(s.name, cap)}
                title={
                  declared[s.name].has(cap)
                    ? `${s.name} declares ${cap} — switch off to withdraw it`
                    : `${s.name} does not declare ${cap} — switch on to grant it falsely`
                }
              />
            ))}
          </div>
        ))}
      </Instrument>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "baseline", marginTop: 4 }}>
        <Mono colour={T.dim}>requests_issued =</Mono>
        <Mono colour={run.requestsIssued === 0 ? T.err : T.accent}>{run.requestsIssued}</Mono>
        <Mono colour={T.dim}>(baseline {base.requestsIssued})</Mono>
      </div>
      <VerdictRow steps={run.steps} />

      {failure && (
        <Callout tone="ok" title="under-declaration: safe, and loud">
          <div style={{ fontFamily: MONO, fontSize: 11.5, color: T.text, lineHeight: 1.7 }}>
            step <span style={{ color: T.accent }}>{failure.step}</span> · source{" "}
            <span style={{ color: T.accent }}>{failure.source}</span> · line{" "}
            <span style={{ color: T.accent }}>{failure.line}</span>
            <br />
            required <span style={{ color: T.dim }}>[{(failure.required ?? []).join(", ")}]</span>
            <br />
            declared <span style={{ color: T.dim }}>[{(failure.declared ?? []).join(", ")}]</span>
            <br />
            missing <span style={{ color: T.err }}>[{(failure.missing ?? []).join(", ")}]</span>
          </div>
          <div style={{ marginTop: 8 }}>
            {failure.reason}
          </div>
          <div style={{ marginTop: 8 }}>
            The plan was refused before anything was asked of anyone:{" "}
            <Mono colour={T.ok}>requests_issued must be 0</Mono>, and it is{" "}
            <Mono colour={run.requestsIssued === 0 ? T.ok : T.err}>{run.requestsIssued}</Mono>.
            A source that admits less than it can do costs you answers and tells
            you exactly which ones.
          </div>
        </Callout>
      )}

      {granted.length > 0 && !changed && (
        <Callout tone="warn" title="over-declaration: unsound, and invisible">
          <Mono colour={T.warn}>granted {granted.join(", ")} — figure unchanged</Mono>
          <div style={{ marginTop: 6 }}>
            Nothing changed. You have just told the system that{" "}
            {granted.map((g) => g.split(":")[0]).join(" and ")} can evaluate
            capabilities it does not implement. The check passed, every verdict
            is what it was, and the request count is identical. In a real
            deployment the results would now be wrong with no verdict registering
            it.
            <br />
            <br />
            This page cannot show you the wrongness, because the fixture answers
            honestly whatever it is asked. That gap is the point: the calculus
            is sound only under <Mono colour={T.dim}>rem:honesty-assumption</Mono>,
            and cell [12] lists it first among the things this does not do.
          </div>
        </Callout>
      )}

      {(withdrawn.length > 0 || granted.length > 0) && !changed && granted.length === 0 && (
        <Callout tone="dim" title="no effect on this plan">
          <div>
            <Mono colour={T.dim}>{withdrawn.join(", ")}</Mono> — this plan never
            asks that source for that capability, so withdrawing it changes
            nothing. Try a capability the plan actually depends on; the failure
            message above names them when one bites.
          </div>
        </Callout>
      )}
    </div>
  );
}

/* ================================================================== */
/*  5. PlanDag — blame terminates because every edge points left        */
/* ================================================================== */

/**
 * The plan drawn as its dependency graph, laid out by hand.
 *
 * Laid out by hand on purpose: x = i * gap IS the plan order, and plan order is
 * the whole content of the figure. A d3 force simulation would randomise that
 * axis and jitter on every render, destroying the only thing worth seeing --
 * that every blame edge points backwards, so there was never a cycle for the
 * walk to get stuck in. This is exactly the trap a "use d3" instruction invites,
 * and declining it is why d3 is used for scales and step curves instead.
 *
 * Edges come from diagnosisText, the same route BlameChain already uses. The
 * diagnosis field is a bare string or null; reading it as an object gives a
 * plausible wrong answer instead of an error, which is why nothing here touches
 * it directly.
 */
export function PlanDag({ steps, picked, onPick }) {
  const nodes = steps.map((s, i) => ({ ...s, i }));

  const edges = [];
  for (const s of nodes) {
    const m = /bound input (\w+) returned/.exec(diagnosisText(s.diagnosis) || "");
    if (m) {
      const from = nodes.find((n) => n.step === m[1]);
      if (from) edges.push({ from: from.i, to: s.i });
    }
  }

  const forward = edges.filter((e) => e.from >= e.to);

  const gap = 104;
  const width = Math.max(320, nodes.length * gap);
  const height = 108;
  const cy = 46;
  const r = 15;

  // The blame walk from the picked node, following edges backwards.
  const lit = new Set();
  if (picked !== null && picked !== undefined) {
    let cur = nodes.findIndex((n) => n.step === picked);
    const seen = new Set();
    while (cur >= 0 && !seen.has(cur)) {
      seen.add(cur);
      lit.add(cur);
      const e = edges.find((x) => x.to === cur);
      cur = e ? e.from : -1;
    }
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={width} height={height} style={{ display: "block" }}>
        <defs>
          <marker id="dagarrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6"
                  markerHeight="6" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={T.dim} />
          </marker>
        </defs>
        {edges.map((e, i) => {
          const x1 = 24 + e.from * gap;
          const x2 = 24 + e.to * gap;
          const on = lit.has(e.from) && lit.has(e.to);
          // Arced above so an edge spanning several steps stays readable.
          const mid = (x1 + x2) / 2;
          const lift = cy - 22 - Math.abs(e.to - e.from) * 6;
          return (
            <path
              key={`e${i}`}
              d={`M${x1},${cy - r} Q${mid},${lift} ${x2},${cy - r}`}
              fill="none"
              stroke={on ? T.err : T.border}
              strokeWidth={on ? 1.8 : 1.2}
              markerEnd="url(#dagarrow)"
              style={{ transition: "stroke 180ms ease" }}
            />
          );
        })}
        {nodes.map((n) => {
          const cx = 24 + n.i * gap;
          const c = VERDICT_COLOUR[n.verdict] ?? T.dim;
          const on = lit.has(n.i);
          return (
            <g
              key={n.step}
              transform={`translate(${cx},0)`}
              onClick={() => onPick && onPick(n.step === picked ? null : n.step)}
              style={{ cursor: "pointer" }}
            >
              <circle cy={cy} r={r} fill={on ? c : T.surface} stroke={c}
                      strokeWidth={on ? 2 : 1.4} opacity={on ? 1 : 0.9}
                      style={{ transition: "fill 180ms ease" }} />
              <text y={cy + 4} textAnchor="middle" fontSize="10" fontFamily={MONO}
                    fill={on ? T.bg : c}>
                {n.i + 1}
              </text>
              <text y={cy + r + 15} textAnchor="middle" fontSize="10" fontFamily={MONO}
                    fill={on ? T.text : T.dim}>
                {n.step}
              </text>
              <text y={18} textAnchor="middle" fontSize="9" fontFamily={MONO} fill={c}>
                {n.verdict}
              </text>
            </g>
          );
        })}
      </svg>

      <Note>
        click a node to walk its blame chain. Every edge is drawn from the step
        that produced a variable to the step that consumed it, and every one
        points <em>left</em> — the parser rejects a plan whose step binds an
        input from a later step, so a cycle cannot be written down. That is why
        the walk terminates in at most m hops rather than needing a visited set
        to save it.
      </Note>

      {forward.length > 0 && (
        <Callout tone="err" title="an edge points forward">
          <div>
            This should be unreachable: <Mono colour={T.err}>prop:blame</Mono>{" "}
            depends on every dependency running strictly earlier, and the parser
            is supposed to enforce it. If you are seeing this, the figure is
            right and something upstream is wrong.
          </div>
        </Callout>
      )}
    </div>
  );
}
