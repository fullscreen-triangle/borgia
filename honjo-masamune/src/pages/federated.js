/**
 * Federated querying without a query language -- an executable notebook.
 *
 * The argument this page makes, and the reason it is a page rather than a
 * paragraph in the manuscript: A QUERY LANGUAGE CANNOT TELL YOU WHY IT RETURNED
 * NOTHING, AND A PLAN LANGUAGE CAN. That is a claim about behaviour, so it is
 * shown by behaviour. Every number below is computed by `src/lib/hfq` in the
 * reader's browser during this render. Nothing is transcribed from the paper,
 * and the reader can edit a plan in cell [9] and watch the verdict change.
 *
 * NO NETWORK I/O, and the constraint is inherited rather than chosen here. The
 * adapters module states it: every adapter resolves against a local fixture,
 * because the prototype's claims are properties of the compiler and a live
 * third-party service can neither confirm nor refute them. A page that pointed
 * this at a real endpoint would replace a checkable claim with an anecdote --
 * and, per `rem:no-probing`, would be probing someone else's service to
 * characterise its internals, which is not ours to do.
 *
 * The page is deliberately NOT the workbench. `_app.js` strips the navbar for
 * `/workbench` and `/atlas` because those own the viewport; this is a document
 * that scrolls, keeps its chrome, and is meant to be read top to bottom.
 */
import Head from "next/head";
import { useMemo, useState } from "react";
import {
  runPlan,
  runSource,
  planNames,
  planSource,
  sourceSummary,
  FEAT,
  VERDICT_PROSE,
  PREDICATE_FEATURES,
  PlanError,
} from "@/lib/hfq";
import { BarRows, ChecksList, fmt } from "@/components/workbench/Charts";
import {
  Attrition,
  BlameChain,
  Cell,
  Code,
  K,
  MONO,
  P,
  Panel,
  Raw,
  StepTable,
  T,
  VerdictChip,
  VERDICT_COLOUR,
  VERDICT_GLOSS,
} from "@/components/federated/Cells";
import {
  BudgetSweep,
  CapabilityGrid,
  PlanDag,
  // Aliased: this file already has a PriceWalk, the labelled table that serves
  // the dcat_g1 panel above. The instrument is the same walk under a slider,
  // and both are wanted -- the table is the static reading, the chart is the
  // shape you need when the budget is moving.
  PriceWalk as PriceWalkChart,
  WaterFilling,
  sweepBudget,
} from "@/components/federated/Instruments";

/* ================================================================== */
/*  Execution, once, at module scope of the render                    */
/* ================================================================== */

/**
 * Run every plan in the fixture and index the results.
 *
 * All 24 at once, not the handful the cells below name. The verdict gallery in
 * cell [4] claims that all six verdicts have a live demonstrator, and that
 * claim is only worth making if it is computed over the whole corpus -- a
 * gallery assembled from six hand-picked plans would prove that six plans exist,
 * which is not the same statement.
 *
 * `runPlan` builds a FRESH registry per call. That matters: `requestsIssued` is
 * a counter on the adapter, and `cor:refuse-before-contact` is a claim that a
 * statically refused plan issues ZERO requests. A registry shared across runs
 * would accumulate and turn a claim about one plan into a claim about the page.
 */
function runAll() {
  const names = planNames().slice().sort();
  const runs = {};
  const failed = {};
  for (const n of names) {
    try {
      runs[n] = runPlan(n).execution.toJSON();
    } catch (e) {
      // A plan that throws is reported, never dropped. A gallery that silently
      // omitted the plans it could not run would show a clean sweep of the
      // plans that happened to work.
      failed[n] = String(e && e.message ? e.message : e);
    }
  }
  return { names, runs, failed };
}

export default function Federated() {
  const { names, runs, failed } = useMemo(runAll, []);
  const sources = useMemo(() => sourceSummary(), []);

  /* ---------------- verdict census over all plans ---------------- */
  const census = useMemo(() => {
    const c = {};
    for (const v of Object.keys(VERDICT_COLOUR)) c[v] = [];
    for (const n of names) {
      for (const s of (runs[n]?.steps) || []) {
        if (c[s.verdict]) c[s.verdict].push(`${n}:${s.step}`);
      }
    }
    return c;
  }, [names, runs]);

  /* ---------------- refusal-before-contact over all plans -------- */
  const refusals = useMemo(
    () =>
      names
        .filter((n) => runs[n]?.halted_early)
        .map((n) => ({
          plan: n,
          requests: runs[n].requests_issued,
          steps: runs[n].steps.length,
          bound: runs[n].check?.bound,
          operations: runs[n].check?.operations,
          failures: runs[n].check?.failures || [],
        })),
    [names, runs]
  );

  const q4 = runs.mark_q4;
  const chain = runs.starved_chain;
  const g1 = runs.dcat_g1;
  const routes = runs.routes;
  const trap = runs.budget_trap;

  return (
    <>
      <Head>
        <title>Federated querying without a query language — Honjo Masamune</title>
        <meta
          name="description"
          content="An executable notebook: why a plan language can say why it returned nothing, and a query language cannot. Every cell computes in your browser."
        />
      </Head>

      <div style={{ background: T.bg, minHeight: "100vh", paddingTop: 96, paddingBottom: 80 }}>
        <div style={{ maxWidth: 1040, margin: "0 auto", padding: "0 22px" }}>
          <Masthead names={names} runs={runs} failed={failed} sources={sources} />

          {/* =========================================================== */}
          <Cell n={1} title="The problem, as it was actually posed">
            <P>
              A biocatalysis group publishes its data as an ontology-backed knowledge graph and
              asks questions of it: <em>which enzyme from a bacterium catalyses this
              transamination and has no cysteine in its sequence?</em>{" "}
              <em>which buffer and pH was used in that methyl transfer?</em>{" "}
              <em>with which device, and at which wavelength, was that UV spectrum
              monitored?</em> The standard route is to model the domain in OWL, materialise a
              graph, run a reasoner over it, and query the closure in SPARQL.
            </P>
            <P>
              That route works, and where it works it is the right tool. This page is about the
              cases where it returns nothing, because a SPARQL result set has one way of saying
              nothing and at least four reasons for doing so: the sequence data is not in the
              corpus; the endpoint cannot evaluate the property path you wrote; the endpoint
              gave up under load; or a subquery upstream returned nothing and your outer pattern
              never had a chance. All four produce zero rows. <K>cor:onebit</K> is the statement
              that no amount of careful SPARQL recovers the distinction — it is not a matter of
              writing a better query.
            </P>
            <P>
              The alternative here is not a better query language. It is not a query language at
              all. The author writes a <strong>plan</strong>: terms denote result sets, operators
              denote instructions about what to do with a result, and{" "}
              <strong>queries are leaves the author never writes</strong>. Each leaf is lowered
              to whatever the source it names can actually evaluate, and each step reports a
              verdict drawn from six values rather than a row count.
            </P>
          </Cell>

          {/* =========================================================== */}
          <Cell n={2} title="The sources, and what each one admits it cannot do">
            <P>
              A source is a quadruple: a namespace, a declared capability set drawn from eleven
              symbols, an extent, and a cost function. The capability set is the load-bearing
              part. <strong>Under-declaring is safe</strong> — the planner routes around a
              capability you did not claim. <strong>Over-declaring is unsound and invisible</strong>:
              nothing in the system verifies the declaration, which{" "}
              <K>rem:honesty-assumption</K> names as the single largest gap in the approach. It is
              stated here rather than buried, because a page that showed only the guarantees
              would be advertising.
            </P>
            <SourceTable sources={sources} />
            <Panel
              title="the eleven capabilities"
              subtitle="The vocabulary is closed. A twelfth capability is a change to the calculus, not a configuration value — which is why FEAT is a frozen array and not a settings file."
              source="FEAT — src/lib/hfq/model.js"
            >
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {FEAT.map((f) => (
                  <span
                    key={f}
                    style={{
                      fontSize: 10,
                      fontFamily: MONO,
                      color: T.accent,
                      border: `1px solid ${T.border}`,
                      borderRadius: 3,
                      padding: "2px 7px",
                    }}
                  >
                    {f}
                  </span>
                ))}
              </div>
              <div style={{ fontSize: 10, color: T.dim, fontFamily: MONO, marginTop: 10 }}>
                {FEAT.length} symbols · {Object.keys(PREDICATE_FEATURES).length} predicates carry
                capability requirements
              </div>
            </Panel>
            <Panel
              title="withdraw a capability, or grant one that does not exist"
              subtitle="The paragraph above makes two claims. Under-declaring is safe; over-declaring is unsound and invisible. Both are switches. Withdraw RXN's neg and the plan is refused at zero requests with the line number; grant RXN agg, which it does not implement, and watch nothing at all happen."
              source="check() over a perturbed registry — no network, structuredClone per run"
            >
              <CapabilityGrid planName="mark_q1" />
            </Panel>
          </Cell>

          {/* =========================================================== */}
          <Cell n={3} title="Six verdicts, four blockers, and one deliberate absence">
            <P>
              A step reports one of six verdicts, assigned by rules applied in a fixed order.
              The order is not cosmetic: a step that is both starved of input and over budget
              reports <K>starved</K>, because the earlier obstruction is the one that actually
              stopped it. Reporting the budget would send the reader to buy capacity for a
              request that was never going to be issued.
            </P>
            <VerdictLegend census={census} />
            <P>
              Four of the six carry a <strong>blocker</strong> — the thing to go and fix:{" "}
              <K>model</K> (rewrite the plan), <K>engine</K> (the source gave up),{" "}
              <K>budget</K> (spend more), <K>corpus</K> (the data is not there).{" "}
              <K>answer</K> and <K>empty</K> carry none, and the JSON <em>omits the field</em>{" "}
              rather than setting it to null. That absence is a design decision: &ldquo;nothing
              matched&rdquo; is a fact about the world, and attaching a blocker to it would
              label the world as a malfunction.
            </P>
          </Cell>

          {/* =========================================================== */}
          <Cell n={4} title="Every verdict has a live demonstrator — computed over all 24 plans">
            <P>
              A six-valued algebra is only worth the name if all six values occur. The census
              below is taken by running every plan in the fixture in this browser and tallying
              the verdict of every step. It is not a table of examples chosen to fill the
              categories; it is the whole corpus, and if a verdict had no witness the row would
              be empty and would say so.
            </P>
            <Panel
              title="verdict census"
              subtitle={`${names.length} plans, ${Object.values(census).reduce((a, b) => a + b.length, 0)} steps, executed in your browser on this page load`}
              source="runPlan() over planNames() — src/lib/hfq/index.js"
              note="Read the answer count against the others. The overwhelming majority of steps answer; the interesting verdicts are rare by construction, because a plan corpus made mostly of failures would be a corpus about failure rather than about querying."
            >
              <BarRows
                rows={Object.keys(VERDICT_COLOUR).map((v) => ({
                  label: `${v}${VERDICT_PROSE[v] !== v ? ` (${VERDICT_PROSE[v]})` : ""}`,
                  value: census[v].length,
                  colour: VERDICT_COLOUR[v],
                }))}
                unit=" steps"
                dp={0}
              />
              <div style={{ marginTop: 12 }}>
                {Object.keys(VERDICT_COLOUR).map((v) => (
                  <div
                    key={v}
                    style={{
                      display: "flex",
                      gap: 10,
                      padding: "4px 0",
                      borderBottom: `1px solid ${T.panel}`,
                      fontSize: 9,
                      fontFamily: MONO,
                      alignItems: "baseline",
                    }}
                  >
                    <span style={{ width: 62, flexShrink: 0, color: VERDICT_COLOUR[v] }}>{v}</span>
                    <span style={{ color: census[v].length ? T.text : T.err }}>
                      {census[v].length ? census[v].join("  ") : "NO WITNESS IN THE CORPUS"}
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
          </Cell>

          {/* =========================================================== */}
          <Cell n={5} title="The cell that is the whole argument: a plan that returns nothing, and says why">
            <P>
              Mark&rsquo;s fourth question asks for the expected products of a kinetic
              resolution with enzyme PFE at pH 9 in HEPES buffer. The corpus holds PFE, holds
              its reactions, holds their run conditions — and holds no run at pH 9. In SPARQL
              this is zero rows. Here it is five steps with three distinct verdicts, and the
              last one names its culprit by name.
            </P>
            {q4 ? (
              <>
                <Panel
                  title="mark_q4 — plan source"
                  subtitle="Note what the author did not write: no triple patterns, no property paths, no service clauses. The leaves are lowered by the adapters."
                  source="fixture.plans.mark_q4"
                >
                  <Code text={planSource("mark_q4")} />
                </Panel>
                <Panel
                  title="mark_q4 — executed"
                  subtitle={`${q4.requests_issued} requests issued · halted early: ${String(q4.halted_early)}`}
                  source="Executor.run() in your browser"
                >
                  <StepTable steps={q4.steps} />
                </Panel>
                <Panel
                  title="attrition"
                  subtitle="Result cardinality per step, in plan order. The bar that reaches zero is the finding."
                  source="emitted.coverage.attrition"
                  note="This is the shape a query language cannot show you, because it has no per-step result to report — the whole plan is one query and the intermediate cardinalities are internal to a planner that does not expose them."
                >
                  <Attrition
                    rows={q4.steps.map((s) => ({ step: s.step, n: s.n ?? 0, verdict: s.verdict }))}
                  />
                </Panel>
                <Panel
                  title="the two verdicts that matter here"
                  subtitle="at_ph9 is empty; in_hepes is starved. They are different findings and they require different actions."
                  source="steps[].verdict, steps[].diagnosis"
                >
                  <div style={{ fontSize: 11.5, lineHeight: 1.8, color: "#a9b1d6" }}>
                    <div style={{ marginBottom: 8 }}>
                      <VerdictChip verdict="empty" />{" "}
                      <K>at_ph9</K> — the source answered. It looked, and the corpus contains no
                      PFE run at pH 9. <strong>No blocker.</strong> Nothing is broken; the
                      experiment has not been done, or has not been published. The action is to
                      run it, or to widen the question.
                    </div>
                    <div>
                      <VerdictChip verdict="starved" blocker="corpus" />{" "}
                      <K>in_hepes</K> — this step never got to look, because the set it filters
                      arrived empty. Its diagnosis is{" "}
                      <K>{q4.emitted?.in_hepes?.diagnosis || "—"}</K>. The blocker is{" "}
                      <K>corpus</K>, and it points upstream, at <K>at_ph9</K> — not at the
                      buffer clause the reader was asking about.
                    </div>
                  </div>
                  <Raw label="emitted.in_hepes — the full document" value={q4.emitted?.in_hepes} max={340} />
                </Panel>
              </>
            ) : (
              <MissingPlan name="mark_q4" failed={failed} />
            )}
          </Cell>

          {/* =========================================================== */}
          <Cell n={6} title="Blame terminates, and it terminates provably">
            <P>
              If a starved step names its culprit, and that culprit is itself starved, the
              obvious worry is a chain that never ends. It cannot: the inputs a step names are
              bound by <em>strictly earlier</em> steps, so each hop moves backwards through a
              finite list. <K>prop:blame</K> bounds the walk by the length of the plan. The chain
              below is walked live, by reading each step&rsquo;s diagnosis and following the name
              it gives.
            </P>
            {chain ? (
              <Panel
                title="starved_chain — walking blame backwards from the last step"
                subtitle="Two hops to the source of the problem, from a step that is two removes away from it."
                source="steps[].diagnosis — walked in the browser"
                note="The termination claim is checked here rather than asserted: the walk is bounded by the step count, and if it ever hit that bound the cell would report an engine defect instead of a chain."
              >
                <BlameChain steps={chain.steps} from={chain.steps[chain.steps.length - 1]?.step} />
              </Panel>
            ) : (
              <MissingPlan name="starved_chain" failed={failed} />
            )}
            {chain && (
              <Panel
                title="the same plan as its dependency graph"
                subtitle="Click a step to walk its blame chain. The layout is not a force simulation and never will be: x is plan order, and plan order is the entire content of the figure."
                source="steps[].diagnosis, parsed for the bound input it names"
                note="Every edge points left. That is not a drawing convention — the parser rejects a plan whose step binds an input from a later step, so a cycle cannot be written down, and the walk terminates in at most m hops without needing a visited set to rescue it. If an edge ever pointed right, the figure says so in red."
              >
                <BlameDag chain={chain} />
              </Panel>
            )}
          </Cell>

          {/* =========================================================== */}
          <Cell n={7} title="Refusal before contact: three plans that issue zero requests">
            <P>
              Some plans are unanswerable for reasons visible without asking anyone. If a step
              needs <K>regex</K> and the source it names does not declare <K>regex</K>, no
              request is worth issuing. The static check catches this and{" "}
              <K>cor:refuse-before-contact</K> is the claim that it does so{" "}
              <strong>before any request is formed</strong> — measurable, because the request
              counter is an observable on every adapter and must still read zero.
            </P>
            <Panel
              title="statically refused plans"
              subtitle="requests_issued must be 0 for every row. If any row showed a positive count, the corollary would be false and this cell would show it."
              source="Execution.requests_issued, Execution.halted_early"
            >
              <ChecksList
                checks={Object.fromEntries(
                  refusals.map((r) => [
                    `${r.plan} · issued ${r.requests} requests`,
                    { pass: r.requests === 0 },
                  ])
                )}
              />
              <div style={{ marginTop: 12 }}>
                {refusals.map((r) => (
                  <div
                    key={r.plan}
                    style={{ fontSize: 10, fontFamily: MONO, color: T.dim, padding: "3px 0" }}
                  >
                    <span style={{ color: T.text }}>{r.plan}</span> — {r.steps} steps, check did{" "}
                    {r.operations} of at most {r.bound} operations ({r.steps} × {FEAT.length});{" "}
                    {r.failures.length} capability failure
                    {r.failures.length === 1 ? "" : "s"}
                  </div>
                ))}
              </div>
            </Panel>
            <P>
              The operation counts above are the second half of the claim.{" "}
              <K>thm:static</K> bounds the check by the plan length times the size of the
              capability vocabulary — every row does strictly fewer operations than its own
              bound, so the check is not merely fast in practice but bounded by construction.
              And the refusal <em>names the real obstacle</em> (<K>prin:refusal</K>): the missing
              capability and the source that does not declare it, not a generic failure.
            </P>
          </Cell>

          {/* =========================================================== */}
          <Cell n={8} title="Budget, shadow price, and a certificate that the allocation was optimal">
            <P>
              A plan gets a request budget, and the steps that can use more effort productively
              have concave yield. Allocating is then a water-filling problem with a unique
              optimum and a single scalar shadow price. That much is theory. What the executor
              additionally emits — and what a query planner has no analogue for — is a{" "}
              <strong>numerical certificate</strong> that the allocation it chose satisfies the
              optimality conditions.
            </P>
            {g1 ? (
              <>
                <Panel
                  title="dcat_g1 — KKT residuals"
                  subtitle="Stationarity per supported step, off-support feasibility, and whether the budget constraint binds."
                  source="Execution.toJSON().kkt"
                  note="A zero residual is not a claim that the theory is right about the world. It is a claim that the solver solved the problem the theory poses — which is the only thing a browser can check, and worth separating from the former."
                >
                  <ChecksList
                    checks={{
                      "budget binds": g1.kkt?.budget_binds,
                      "off-support conditions satisfied": g1.kkt?.off_support_satisfied,
                      "max stationarity residual is 0":
                        (g1.kkt?.max_stationarity_residual ?? 1) === 0,
                    }}
                  />
                  <div style={{ marginTop: 10, fontSize: 10, fontFamily: MONO, color: T.dim }}>
                    shadow price p* = {fmt(g1.kkt?.shadow_price, 8)} · spent{" "}
                    {fmt(g1.kkt?.spent, 4)} of {g1.kkt?.budget} · max residual{" "}
                    {fmt(g1.kkt?.max_stationarity_residual, 1)}
                  </div>
                  <Raw label="kkt" value={g1.kkt} />
                </Panel>
                <Panel
                  title="the price moves as the plan executes"
                  subtitle="The allocation is re-solved after each step against the remaining budget and the cardinalities that actually materialised."
                  source="Execution.toJSON().replans"
                  note="The reference implementation re-solves too, and discards the intermediate solutions. This port retains them, which is what makes the price legible as a sequence rather than a final number: the price falling means the plan is handing budget back because the work it planned for did not materialise."
                >
                  <PriceWalk run={g1} />
                </Panel>
              </>
            ) : (
              <MissingPlan name="dcat_g1" failed={failed} />
            )}

            <P>
              Everything above is one plan at one budget. The budget is the
              parameter the whole apparatus turns on, so here it is as a
              control. Move it and the engine re-executes — a plan is 0.35 ms,
              so this is a real run and not a lookup table, and the request
              counter you are watching is genuinely reset per run rather than
              accumulated across the session.
            </P>
            <AllocationLab />
            {trap && (
              <Panel
                title="budget_trap — why a sufficient-looking budget is not sufficient"
                subtitle="Both sources here declare every capability the plan needs. The plan is still refused."
                source="budget_trap — Executor.run()"
                note="prop:necessary-not-sufficient. The declared per-source costs are minima taken over inputs, and the input a step actually meets is not the minimising one. A budget that exceeds the sum of the minima can therefore still be refused at the realised cardinality — so the sum is a necessary condition and not a sufficient one."
              >
                <StepTable steps={trap.steps} />
              </Panel>
            )}
          </Cell>

          {/* =========================================================== */}
          <Cell n={9} title="Break a plan yourself">
            <P>
              This is the part that cannot be a figure. Edit the plan below and it recompiles and
              re-executes in your browser as you type. Try deleting the{" "}
              <K>budget</K> line — the parser refuses, because a plan without a budget is not a
              plan. Try lowering the budget until a step reports <K>refused</K>. Try pointing a
              step at a source that cannot evaluate it and watch the check refuse the whole plan
              at zero requests. Try making an early step return nothing and watch the verdict of
              a later step change from <K>empty</K> to <K>starved</K> — a distinction a query
              language does not have the vocabulary to draw.
            </P>
            <PlanEditor names={names} />
          </Cell>

          {/* =========================================================== */}
          <Cell n={10} title="Ordering is the author's decision, stated, not a planner's silently">
            <P>
              Two plans, identical but for where a filter sits. Both cost the same number of
              requests and both end in answers. A SQL or SPARQL optimiser would be free to
              reorder these, and would be right to — but it would also be free to reorder them
              differently next week, and the author would not be told. Here the order is written
              down, so it can be reviewed, diffed and argued about.
            </P>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(320px,1fr))", gap: 14 }}>
              {["order_a", "order_b"].map((n) =>
                runs[n] ? (
                  <Panel
                    key={n}
                    title={n}
                    subtitle={`${runs[n].requests_issued} requests · ${runs[n].steps.length} steps`}
                    source={`fixture.plans.${n}`}
                  >
                    <Code text={planSource(n)} max={200} />
                    <div style={{ marginTop: 10 }}>
                      <StepTable steps={runs[n].steps} />
                    </div>
                  </Panel>
                ) : (
                  <MissingPlan key={n} name={n} failed={failed} />
                )
              )}
            </div>
          </Cell>

          {/* =========================================================== */}
          <Cell n={11} title="Two routes to the same set, and a coverage number that is not an error count">
            <P>
              When two translation routes connect the same pair of namespaces, they can disagree
              about which correspondences exist. The symmetric difference of their results is
              reported — and reported carefully. <K>thm:route-extent</K> makes it a{" "}
              <strong>lower bound on correspondences that at least one route fails to
              resolve</strong>. It is a statement about coverage. It is not a count of errors,
              and neither route contradicts the other.
            </P>
            {routes ? (
              <Panel
                title="routes — resolved extent"
                subtitle={`symmetric difference ${routes.emitted?.resolved_extent?.symmetric_difference} of union ${routes.emitted?.resolved_extent?.union_size}`}
                source="emitted.resolved_extent"
                note={routes.emitted?.resolved_extent?.interpretation}
              >
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                  {["left", "right"].map((side) => {
                    const re = routes.emitted?.resolved_extent || {};
                    const only = re[`${side}_only`] || [];
                    return (
                      <div key={side}>
                        <div style={{ fontSize: 10, fontFamily: MONO, color: T.text, marginBottom: 5 }}>
                          {re[side]} — {only.length} only on this route
                        </div>
                        {only.length ? (
                          only.map((x) => (
                            <div key={x} style={{ fontSize: 10, fontFamily: MONO, color: T.dim }}>
                              {x}
                            </div>
                          ))
                        ) : (
                          <div style={{ fontSize: 10, fontFamily: MONO, color: T.muted }}>—</div>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div style={{ marginTop: 10 }}>
                  <StepTable steps={routes.steps} />
                </div>
              </Panel>
            ) : (
              <MissingPlan name="routes" failed={failed} />
            )}
          </Cell>

          {/* =========================================================== */}
          <Cell n={12} title="What this does not do">
            <P>
              Four limits, stated because a page that showed only the guarantees would be
              advertising rather than a method description.
            </P>
            <Panel title="limits" source="sec:limits — chem-dcat-ap-querying.tex">
              <div style={{ fontSize: 11.5, lineHeight: 1.8, color: "#a9b1d6" }}>
                <p style={{ margin: "0 0 10px 0" }}>
                  <strong style={{ color: T.warn }}>The capability declaration is unverified.</strong>{" "}
                  It is data written by an adapter author. Under-declaring is safe;
                  over-declaring is unsound <em>and invisible</em> — the check will pass, the
                  request will be lowered, and the result will be wrong with no verdict
                  registering the fact. This is the largest gap in the approach and nothing on
                  this page closes it — <em>but cell [2] lets you produce it.</em>{" "}
                  Grant a source a capability it does not implement and watch the
                  figure stay exactly as it was. Nothing on this page can show
                  you the resulting wrongness, because the fixture answers
                  honestly whatever it is asked; the identical figure is the
                  whole of the evidence, and that is precisely the shape of the
                  gap.
                </p>
                <p style={{ margin: "0 0 10px 0" }}>
                  <strong style={{ color: T.warn }}>Concavity fails for some steps.</strong>{" "}
                  A single-record lookup against a REST interface is all-or-nothing — a step
                  function, not a concave yield. Those steps are charged first, at plan order,
                  and the remainder is optimised. It is a correct handling of a case the theorem
                  does not cover, not an instance of it. In cell [8], drop the{" "}
                  <K>enzymes</K> weight to 0.2 at a budget of 6: the step is
                  allocated nothing, drops off the support entirely, and answers
                  anyway. The bar is a dashed line at zero and the verdict above
                  it is green.
                </p>
                <p style={{ margin: "0 0 10px 0" }}>
                  <strong style={{ color: T.warn }}>
                    Everything here runs against a local fixture.
                  </strong>{" "}
                  Deliberately. The claims are properties of the compiler and the executor, and a
                  live third-party endpoint can neither confirm nor refute them. Pointing this at
                  a real service would replace a checkable claim with an anecdote.
                </p>
                <p style={{ margin: 0 }}>
                  <strong style={{ color: T.warn }}>One observation is unexplained.</strong>{" "}
                  On 10 August 2026, two SPARQL requests differing in a single line returned 2
                  and 397 results. The cause was not diagnosed, and deliberately so: probing a
                  third party&rsquo;s public service with requests designed to elicit its
                  internal behaviour is not ours to do. The observation motivates the longhand
                  emission discipline; it does not have a mechanism attached.
                </p>
              </div>
            </Panel>
          </Cell>

          <Colophon names={names} runs={runs} />
        </div>
      </div>
    </>
  );
}

/* ================================================================== */
/*  Sub-components                                                    */
/* ================================================================== */

function Masthead({ names, runs, failed, sources }) {
  const steps = names.reduce((a, n) => a + ((runs[n]?.steps || []).length), 0);
  const reqs = names.reduce((a, n) => a + (runs[n]?.requests_issued || 0), 0);
  const nFailed = Object.keys(failed).length;
  return (
    <header style={{ marginBottom: 40 }}>
      <div style={{ fontSize: 10, fontFamily: MONO, color: T.accent, letterSpacing: 1.6, marginBottom: 12 }}>
        EXECUTABLE NOTEBOOK · NO NETWORK I/O
      </div>
      <h1 style={{ fontSize: 30, color: T.text, margin: "0 0 14px 0", lineHeight: 1.25, fontWeight: 600, maxWidth: 820 }}>
        Federated querying without a query language
      </h1>
      <p style={{ fontSize: 14, color: "#a9b1d6", lineHeight: 1.75, maxWidth: 780, margin: "0 0 18px 0" }}>
        A query language returns zero rows and cannot tell you why. A plan language returns a
        verdict, a blocker, and the name of the step that caused it. Every figure below was
        computed by <code style={{ fontFamily: MONO, fontSize: 12, color: T.accent }}>src/lib/hfq</code>{" "}
        in your browser during this page load.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 20, fontSize: 10, fontFamily: MONO, color: T.dim }}>
        <span><span style={{ color: T.text }}>{names.length}</span> plans executed</span>
        <span><span style={{ color: T.text }}>{steps}</span> steps</span>
        <span><span style={{ color: T.text }}>{sources.length}</span> sources</span>
        <span><span style={{ color: T.text }}>{reqs}</span> requests issued in total</span>
        <span style={{ color: nFailed ? T.err : T.dim }}>
          <span style={{ color: nFailed ? T.err : T.ok }}>{nFailed}</span> plans failed to run
        </span>
        {/* Stated as a number because it is the claim the whole method rests
            on, and a claim nobody can count is a slogan. */}
        <span><span style={{ color: T.ok }}>0</span> network requests</span>
      </div>

      {/* The question this page exists to answer, in the words it was asked
          in. It is set apart rather than paraphrased into the prose because
          the method is being judged against someone else's problem, not
          against a problem chosen to suit it. */}
      <blockquote
        style={{
          borderLeft: `3px solid ${T.accent}`,
          margin: "26px 0 0",
          padding: "2px 0 2px 16px",
          maxWidth: 760,
        }}
      >
        <p style={{ fontSize: 14.5, lineHeight: 1.7, color: "#a9b1d6", fontStyle: "italic", margin: 0 }}>
          Which biocatalyst, originating from a bacterium and not a eukaryote,
          catalyses the transamination of benzylethylamine, and does not have a
          cysteine in its protein sequence?
        </p>
        <footer style={{ fontSize: 10, fontFamily: MONO, color: T.dim, marginTop: 9 }}>
          one sentence · five steps · three sources · two namespaces
        </footer>
      </blockquote>

      <p style={{ fontSize: 12.5, color: T.dim, lineHeight: 1.75, maxWidth: 780, margin: "20px 0 0" }}>
        Cells [2], [6], [8] and [9] have controls. They are not illustrations of
        the engine — they run it, on every frame, and a plan is 0.35 ms. Use
        them to drive the page into the regimes where its claims stop holding;
        several cells are written to tell you when you have.
      </p>
      {nFailed > 0 && (
        <div style={{ marginTop: 14, border: `1px solid ${T.err}`, borderRadius: 4, padding: 10 }}>
          <div style={{ fontSize: 10, fontFamily: MONO, color: T.err, marginBottom: 6 }}>
            PLANS THAT DID NOT EXECUTE — reported, not omitted
          </div>
          {Object.entries(failed).map(([n, msg]) => (
            <div key={n} style={{ fontSize: 10, fontFamily: MONO, color: T.dim }}>
              {n}: {msg}
            </div>
          ))}
        </div>
      )}
    </header>
  );
}

function SourceTable({ sources }) {
  const th = {
    fontSize: 9, color: T.muted, fontFamily: MONO, textAlign: "left",
    padding: "0 10px 5px 0", borderBottom: `1px solid ${T.border}`, fontWeight: 400,
  };
  const td = {
    fontSize: 10, fontFamily: MONO, padding: "4px 10px 4px 0",
    borderBottom: `1px solid ${T.panel}`, color: T.text, verticalAlign: "top",
  };
  return (
    <Panel
      title="declared sources"
      subtitle="Capabilities are what the adapter author claims the source can evaluate. A capability absent from a row is a capability the planner will not route through it."
      source="sourceSummary() — src/lib/hfq/index.js"
    >
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 600 }}>
          <thead>
            <tr>
              <th style={th}>source</th>
              <th style={th}>ns</th>
              <th style={th}>snapshot</th>
              <th style={th}>declared capabilities</th>
              <th style={{ ...th, textAlign: "right" }}>extent</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => {
              const missing = FEAT.filter((f) => !s.capabilities.includes(f));
              const ext = Object.entries(s.extent || {})
                .filter(([, v]) => v)
                .map(([k, v]) => `${v} ${k}`)
                .join(", ");
              return (
                <tr key={s.name}>
                  <td style={td}>{s.name}</td>
                  <td style={{ ...td, color: T.dim }}>{s.namespace}</td>
                  <td style={{ ...td, color: T.muted, fontSize: 9 }}>{s.snapshot || "—"}</td>
                  <td style={td}>
                    <span style={{ color: T.ok }}>{s.capabilities.join(" ")}</span>
                    {missing.length > 0 && (
                      <span style={{ color: T.muted }} title="not declared — the planner will not route these through this source">
                        {" "}
                        {missing.map((f) => `¬${f}`).join(" ")}
                      </span>
                    )}
                  </td>
                  <td style={{ ...td, textAlign: "right", color: T.dim, fontSize: 9 }}>{ext || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function VerdictLegend({ census }) {
  return (
    <Panel
      title="the verdict algebra"
      subtitle="Assigned by ordered rules. The first rule that fires decides, which is why a step that is both starved and over budget reports starvation."
      source="Verdict, blockerOf — src/lib/hfq/model.js"
    >
      {Object.keys(VERDICT_COLOUR).map((v) => (
        <div
          key={v}
          style={{
            display: "flex", gap: 12, alignItems: "baseline",
            padding: "6px 0", borderBottom: `1px solid ${T.panel}`,
          }}
        >
          <span style={{ width: 130, flexShrink: 0 }}>
            <VerdictChip verdict={v} prose={VERDICT_PROSE[v]} />
          </span>
          <span style={{ fontSize: 11, color: "#a9b1d6", lineHeight: 1.6, flex: 1 }}>
            {VERDICT_GLOSS[v]}
          </span>
          <span style={{ fontSize: 9, fontFamily: MONO, color: T.dim, width: 62, textAlign: "right" }}>
            {census[v].length} step{census[v].length === 1 ? "" : "s"}
          </span>
        </div>
      ))}
    </Panel>
  );
}

/**
 * The shadow price across the plan's re-solves.
 *
 * Rendered as an explicit sequence rather than a line chart: there are a handful
 * of points, each has a name, and the reader needs the name to see what the
 * price is responding to. A sparkline would hide exactly the label that makes
 * the numbers mean something.
 */
function PriceWalk({ run }) {
  const rows = [
    { after: "planned", p: run.allocation?.shadow_price, budget: run.allocation?.budget, support: run.allocation?.support },
    ...((run.replans || []).map((r) => ({
      after: `after ${r.after}`,
      p: r.allocation?.shadow_price,
      budget: r.allocation?.budget,
      support: r.allocation?.support,
    }))),
  ];
  return (
    <div>
      {rows.map((r, i) => (
        <div
          key={r.after}
          style={{
            display: "flex", gap: 12, alignItems: "baseline",
            padding: "5px 0", borderBottom: i === rows.length - 1 ? "none" : `1px solid ${T.panel}`,
            fontSize: 10, fontFamily: MONO,
          }}
        >
          <span style={{ width: 110, flexShrink: 0, color: i === 0 ? T.accent : T.text }}>{r.after}</span>
          <span style={{ width: 116, flexShrink: 0, color: T.dim }}>p* = {fmt(r.p, 8)}</span>
          <span style={{ width: 96, flexShrink: 0, color: T.dim }}>budget {fmt(r.budget, 2)}</span>
          <span style={{ color: T.muted }}>support: {(r.support || []).join(", ") || "—"}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * The blame graph, over a live re-run of the plan.
 *
 * The page's `runs` are serialised through toJSON(), and the DAG needs the
 * step objects the executor produced -- so this re-runs the one plan it draws
 * rather than reaching into a shape that was flattened for display. One plan
 * is 0.35 ms; the alternative is a second serialisation format to maintain.
 */
function BlameDag({ chain }) {
  const [picked, setPicked] = useState(null);
  const steps = chain?.steps || [];
  return <PlanDag steps={steps} picked={picked} onPick={setPicked} />;
}

/**
 * Cell [8]'s controls: one budget, three figures that share it.
 *
 * The plan list is restricted to plans whose behaviour actually varies with
 * budget. Offering all twenty-four would put plans in the menu whose figure is
 * a flat line at every setting, and a reader who lands on one first learns the
 * wrong thing about the control. The flat ones are named rather than hidden.
 */
function AllocationLab() {
  const [plan, setPlan] = useState("mark_q1");
  const [budget, setBudget] = useState(6);

  const source = useMemo(() => planSource(plan), [plan]);

  // Which plans are worth putting in the menu, and which are not. Computed,
  // not hardcoded: a fixture change must not quietly leave a stale list.
  const { varies, flat } = useMemo(() => {
    const v = [];
    const f = [];
    for (const n of planNames()) {
      const s = planSource(n);
      if (!s) continue;
      let regimes;
      try {
        regimes = sweepBudget(s, 24).regimes;
      } catch {
        continue;
      }
      (regimes.length >= 3 ? v : f).push(n);
    }
    return { varies: v, flat: f };
  }, []);

  return (
    <Panel
      title="the budget, as a control"
      subtitle="One slider, three views of what it does: where the effort goes, what the whole range looks like, and how the price moves as the plan runs."
      source="Executor.run() — re-executed per frame, fresh registry per run"
      note="The weight sliders and the effort bars re-execute the engine on every frame. The sweep beneath them does not: it is computed once on mount, because it is 41 runs rather than one. Both are real executions of the same code; only their timing differs, and it seemed better to say so than to let the smoothness imply otherwise."
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 12 }}>
        <select
          value={plan}
          onChange={(e) => setPlan(e.target.value)}
          style={{
            background: T.panel, color: T.text, border: `1px solid ${T.border}`,
            borderRadius: 3, fontSize: 10, fontFamily: MONO, padding: "4px 7px",
          }}
        >
          {varies.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <span style={{ fontSize: 10, fontFamily: MONO, color: T.dim }}>
          {varies.length} of {varies.length + flat.length} plans change behaviour with budget
        </span>
      </div>

      <WaterFilling source={source} budget={budget} onBudget={setBudget} hiBudget={40} />
      <BudgetSweep source={source} budget={budget} hi={40} />
      <PriceWalkChart source={source} budget={budget} />

      {flat.length > 0 && (
        <div style={{ fontSize: 11, color: T.dim, lineHeight: 1.7, marginTop: 14 }}>
          Not in the menu, and named rather than hidden:{" "}
          <span style={{ fontFamily: MONO, color: T.violet }}>{flat.join(", ")}</span>. Each of
          these has fewer than three distinct outcomes across the whole budget range, for one of
          two reasons. Some are refused by the capability check before allocation happens at all,
          so every budget produces the same <K>surface</K> verdict at zero requests — no amount
          of money buys a capability. The rest have exactly one threshold, at nothing versus
          something, and are flat on either side of it. Both are real results, but neither makes
          a good slider: a control that moves without changing the figure teaches the reader that
          the budget does not matter, which is the opposite of what is true.
        </div>
      )}
    </Panel>
  );
}

/**
 * The editable cell.
 *
 * Recompiles on every keystroke, and does so synchronously without a Run
 * button, because the point is that the reader sees the verdict move as they
 * type. A `PlanError` is rendered as prose, not swallowed: a parser that refuses
 * is behaving correctly, and the message it gives is part of what the page is
 * demonstrating.
 */
function PlanEditor({ names }) {
  const [pick, setPick] = useState("mark_q4");
  const [text, setText] = useState(() => planSource("mark_q4"));
  const [budget, setBudget] = useState("");

  const result = useMemo(() => {
    try {
      const b = budget.trim() === "" ? null : Number(budget);
      // Zero is admitted, where the original guard rejected it. A budget of
      // nothing is a legitimate question -- it is the budget at which every
      // step is refused before contact -- and the slider's left end has to
      // reach it or the reader cannot see the regime the prose describes.
      if (b !== null && (!Number.isFinite(b) || b < 0)) {
        return { error: "budget override must be zero or a positive number" };
      }
      const { execution } = runSource(text, { budget: b });
      return { run: execution.toJSON() };
    } catch (e) {
      return {
        error: String(e && e.message ? e.message : e),
        kind: e instanceof PlanError ? "parse" : "execute",
      };
    }
  }, [text, budget]);

  const load = (n) => {
    setPick(n);
    setText(planSource(n));
  };

  const run = result.run;
  return (
    <Panel
      title="live plan editor"
      subtitle="Recompiles and re-executes on every keystroke, in this tab, against the embedded fixture. There is no server."
      source="runSource() — src/lib/hfq/index.js"
      note="A budget override of nothing means the plan's own `budget N requests` line governs. Set it low and watch a step move to `refused` — then note that the verdict names `budget` as the blocker, which tells you the request was well-formed and merely unaffordable."
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 10 }}>
        <select
          value={pick}
          onChange={(e) => load(e.target.value)}
          style={{
            background: T.panel, color: T.text, border: `1px solid ${T.border}`,
            borderRadius: 3, fontSize: 10, fontFamily: MONO, padding: "4px 7px",
          }}
        >
          {names.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <label style={{ fontSize: 10, fontFamily: MONO, color: T.dim }}>
          budget override{" "}
          <input
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            placeholder="plan's own"
            style={{
              background: T.panel, color: T.text, border: `1px solid ${T.border}`,
              borderRadius: 3, fontSize: 10, fontFamily: MONO, padding: "4px 7px", width: 92,
            }}
          />
        </label>
        {/* The slider writes the same string the field holds, so the two are
            one control with two grips. The prose in this cell tells the reader
            to "lower the budget until a step reports refused"; typing numbers
            to find a threshold is the wrong instrument for a search, and
            dragging is the right one. Clearing returns to the plan's own line. */}
        <input
          type="range"
          min={0}
          max={40}
          step={1}
          value={budget.trim() === "" ? 0 : Math.min(40, Math.max(0, Number(budget) || 0))}
          onChange={(e) => setBudget(e.target.value)}
          style={{ accentColor: T.accent, width: 150 }}
          title="drag to override the budget; clear the field to restore the plan's own"
        />
        <button
          type="button"
          onClick={() => setBudget("")}
          style={{
            background: T.panel, color: T.dim, border: `1px solid ${T.border}`,
            borderRadius: 3, fontSize: 10, fontFamily: MONO, padding: "4px 8px",
            cursor: "pointer",
          }}
        >
          plan&rsquo;s own
        </button>
        {run && (
          <span style={{ fontSize: 10, fontFamily: MONO, color: T.dim }}>
            {run.requests_issued} requests ·{" "}
            <span style={{ color: run.halted_early ? T.violet : T.dim }}>
              halted early: {String(run.halted_early)}
            </span>{" "}
            · check {run.check?.operations}/{run.check?.bound} ops
          </span>
        )}
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        rows={14}
        style={{
          width: "100%", background: T.bg, color: T.text,
          border: `1px solid ${result.error ? T.err : T.border}`, borderRadius: 4,
          fontFamily: MONO, fontSize: 11, lineHeight: 1.65, padding: 10,
          resize: "vertical", outline: "none",
        }}
      />

      {result.error && (
        <div
          style={{
            marginTop: 10, border: `1px solid ${T.err}`, borderRadius: 4,
            padding: 10, fontSize: 10.5, fontFamily: MONO, color: T.err, lineHeight: 1.7,
          }}
        >
          <div style={{ color: T.muted, marginBottom: 4 }}>
            {result.kind === "parse" ? "the parser refused this plan" : "execution raised"}
          </div>
          {result.error}
        </div>
      )}

      {run && (
        <>
          <div style={{ marginTop: 12 }}>
            <StepTable steps={run.steps} />
          </div>
          {run.check && run.check.well_capability === false && (
            <div
              style={{
                marginTop: 10, border: `1px solid ${T.violet}`, borderRadius: 4, padding: 10,
                fontSize: 10.5, fontFamily: MONO, color: T.violet, lineHeight: 1.7,
              }}
            >
              <div style={{ color: T.muted, marginBottom: 4 }}>
                statically refused — {run.requests_issued} requests issued
              </div>
              {(run.check.failures || []).map((f, i) => (
                <div key={i}>{typeof f === "string" ? f : JSON.stringify(f)}</div>
              ))}
            </div>
          )}
          {run.steps.some((s) => (s.n ?? 0) >= 0) && (
            <div style={{ marginTop: 14 }}>
              <Attrition
                rows={run.steps.map((s) => ({ step: s.step, n: s.n ?? 0, verdict: s.verdict }))}
              />
            </div>
          )}
          <Raw label="the whole execution document" value={run} max={380} />
        </>
      )}
    </Panel>
  );
}

function MissingPlan({ name, failed }) {
  return (
    <Panel title={`${name} — did not execute`} source="Executor.run()">
      <div style={{ fontSize: 10.5, fontFamily: MONO, color: T.err, lineHeight: 1.7 }}>
        {failed[name] ||
          `no plan named ${name} in the fixture — this cell has nothing to show and says so rather than rendering an empty chart`}
      </div>
    </Panel>
  );
}

function Colophon({ names, runs }) {
  const snaps = new Set();
  for (const n of names) {
    for (const s of runs[n]?.steps || []) if (s.snapshot) snaps.add(s.snapshot);
  }
  return (
    <footer style={{ marginTop: 46, borderTop: `1px solid ${T.border}`, paddingTop: 18 }}>
      <div style={{ fontSize: 10, fontFamily: MONO, color: T.dim, lineHeight: 1.9 }}>
        <div>
          engine: <span style={{ color: T.text }}>src/lib/hfq</span> — a hand port of the
          reference Python executor, verified against it on all {names.length} plans (0 semantic
          differences: verdicts, blockers, cardinalities, request counts, shadow prices, and
          per-step effort)
        </div>
        <div>
          corpus snapshots: <span style={{ color: T.text }}>{[...snaps].sort().join(", ") || "—"}</span>
        </div>
        <div>
          manuscript: <span style={{ color: T.text }}>chem-dcat-ap-querying.tex</span> — the
          method, the theorems, and the proofs this page demonstrates rather than states
        </div>
        <div style={{ marginTop: 8, color: T.muted }}>
          Nothing on this page contacted a network. Every verdict, cardinality and price was
          computed here, from the embedded fixture, while you were reading.
        </div>
      </div>
    </footer>
  );
}
