/**
 * Notebook primitives for the federated-query page.
 *
 * `components/workbench/Charts.js` already carries the generic ones -- `Panel`,
 * `BarRows`, `Scatter`, `ChecksList`, the theme -- and those are imported rather
 * than reimplemented. What lives here is the vocabulary that only this page has:
 * a verdict is a six-valued thing with a blocker attached, a step sequence
 * attrites, and a starved step names its culprit. None of that has a shape in a
 * chart library because none of it has a shape in a query language.
 *
 * The same rule as Charts.js applies and is worth restating: EVERY COMPONENT
 * HERE TAKES DATA. None contains a verdict, a cardinality, a shadow price, or a
 * request count. Every number the reader sees was produced by `src/lib/hfq`
 * running in their browser on this page load. A component that hardcoded a
 * result would make the page a screenshot of a claim rather than the claim.
 */
import { useMemo, useState } from "react";
import {
  CHART_THEME as T,
  CHART_MONO as MONO,
  Panel,
} from "@/components/workbench/Charts";

/* ------------------------------------------------------------------ */
/*  Verdicts                                                          */
/* ------------------------------------------------------------------ */

/**
 * The six verdicts, their blockers, and the colour each is drawn in.
 *
 * This table is the page's one piece of presentation-only knowledge: which hue
 * a verdict gets. It deliberately does NOT restate the blocker mapping, which
 * is computed by `blockerOf` in `lib/hfq/model.js` and read off each executed
 * step -- a second copy here could drift from the engine and the page would
 * keep rendering confidently.
 *
 * `answer` and `empty` are both non-obstructions, and they are drawn in
 * different colours on purpose. A query language returns the same zero rows for
 * both, and telling them apart is a large part of what this page is about, so
 * showing them in one colour would concede the point typographically.
 */
export const VERDICT_COLOUR = {
  answer: T.ok,
  empty: T.accent,
  surface: T.violet,
  timeout: T.orange,
  refused: T.warn,
  starved: T.err,
};

/** One-line gloss per verdict. Prose, not data -- the engine has no such field. */
export const VERDICT_GLOSS = {
  answer: "the source answered and the result is non-empty",
  empty: "the source answered and nothing in its extent matched",
  surface: "the request needs a capability the source does not declare",
  timeout: "the request did not complete inside the effort allocated to it",
  refused: "the remaining budget is below this source's cost at this input size",
  starved: "an input this step depends on did not arrive usably",
};

/**
 * A verdict chip. `blocker` is rendered when present and omitted when absent,
 * because absence is meaningful: `answer` and `empty` carry no blocker, and the
 * JSON omits the field rather than setting it to null.
 */
export function VerdictChip({ verdict, blocker, prose, title }) {
  const c = VERDICT_COLOUR[verdict] ?? T.dim;
  return (
    <span
      title={title || VERDICT_GLOSS[verdict] || verdict}
      style={{
        display: "inline-flex",
        alignItems: "baseline",
        gap: 5,
        fontSize: 9,
        fontFamily: MONO,
        color: c,
        border: `1px solid ${c}`,
        borderRadius: 3,
        padding: "1px 5px",
        whiteSpace: "nowrap",
      }}
    >
      <span>{prose && prose !== verdict ? `${verdict}/${prose}` : verdict}</span>
      {blocker && <span style={{ color: T.dim, fontSize: 8 }}>{blocker}</span>}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Step table                                                        */
/* ------------------------------------------------------------------ */

/**
 * The executed steps of one plan, in plan order, with what each one cost.
 *
 * Plan order is the presentation order and is not sorted. (V10) on this page is
 * an ordering pair -- `order_a` and `order_b` differ only in where a filter
 * sits -- and a table that sorted its rows would destroy the very difference
 * the cell exists to show.
 */
export function StepTable({ steps, onPick, picked }) {
  if (!steps || !steps.length) return null;
  const th = {
    fontSize: 9,
    color: T.muted,
    fontFamily: MONO,
    textAlign: "left",
    padding: "0 8px 5px 0",
    borderBottom: `1px solid ${T.border}`,
    fontWeight: 400,
  };
  const td = {
    fontSize: 10,
    fontFamily: MONO,
    padding: "4px 8px 4px 0",
    borderBottom: `1px solid ${T.panel}`,
    color: T.text,
    verticalAlign: "top",
  };
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 520 }}>
        <thead>
          <tr>
            <th style={th}>step</th>
            <th style={th}>source</th>
            <th style={th}>verdict</th>
            <th style={{ ...th, textAlign: "right" }}>|result|</th>
            <th style={{ ...th, textAlign: "right" }}>spent</th>
            <th style={{ ...th, textAlign: "right" }}>allocated</th>
            <th style={th}>diagnosis</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s) => {
            const on = picked === s.step;
            return (
              <tr
                key={s.step}
                onClick={onPick ? () => onPick(on ? null : s.step) : undefined}
                style={{
                  cursor: onPick ? "pointer" : "default",
                  background: on ? T.panel : "transparent",
                }}
              >
                <td style={{ ...td, color: on ? T.accent : T.text }}>{s.step}</td>
                <td style={{ ...td, color: T.dim }}>{s.source || "—"}</td>
                <td style={td}>
                  <VerdictChip
                    verdict={s.verdict}
                    blocker={s.blocker}
                    prose={s.verdict_prose}
                  />
                </td>
                <td style={{ ...td, textAlign: "right" }}>
                  {s.n === null || s.n === undefined ? "—" : s.n}
                </td>
                <td style={{ ...td, textAlign: "right", color: T.dim }}>
                  {s.spent === null || s.spent === undefined ? "—" : s.spent}
                </td>
                <td style={{ ...td, textAlign: "right", color: T.dim }}>
                  {s.allocated === null || s.allocated === undefined
                    ? "—"
                    : Number(s.allocated).toFixed(2)}
                </td>
                <td style={{ ...td, color: T.dim, fontSize: 9, maxWidth: 260 }}>
                  {diagnosisText(s.diagnosis)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * A diagnosis is a small open map, not a string: the reference builds different
 * keys per rule, and the port follows it. Rendering `reason` when it exists and
 * a compact key list otherwise keeps the table honest about a shape this page
 * does not control -- an unrecognised diagnosis shows its keys rather than
 * silently rendering blank.
 */
function diagnosisText(d) {
  if (!d) return "";
  if (typeof d === "string") return d;
  if (d.reason) return d.reason;
  const ks = Object.keys(d);
  return ks.length ? ks.map((k) => `${k}=${String(d[k])}`).join(" ") : "";
}

/* ------------------------------------------------------------------ */
/*  Attrition                                                         */
/* ------------------------------------------------------------------ */

/**
 * The cardinality of each step's result, in plan order, drawn as a waterfall.
 *
 * A zero-height bar is drawn as a visible hairline rather than nothing. The
 * whole point of the cell is the step where the count reaches zero, and a bar
 * of literally zero pixels would make the most important row of the chart the
 * only invisible one.
 */
export function Attrition({ rows, height = 116 }) {
  if (!rows || !rows.length) return null;
  const hi = Math.max(...rows.map((r) => r.n || 0), 1);
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 6,
          height,
          borderBottom: `1px solid ${T.border}`,
          paddingBottom: 2,
        }}
      >
        {rows.map((r) => {
          const c = VERDICT_COLOUR[r.verdict] ?? T.accent;
          const h = Math.max(((r.n || 0) / hi) * (height - 18), r.n ? 3 : 2);
          return (
            <div
              key={r.step}
              style={{ flex: 1, minWidth: 34, textAlign: "center" }}
              title={`${r.step}: |result| = ${r.n} (${r.verdict})`}
            >
              <div style={{ fontSize: 9, fontFamily: MONO, color: T.text, marginBottom: 3 }}>
                {r.n}
              </div>
              <div
                style={{
                  height: h,
                  background: r.n ? c : "transparent",
                  borderTop: r.n ? "none" : `2px dashed ${c}`,
                  borderRadius: r.n ? "2px 2px 0 0" : 0,
                  transition: "height 200ms ease",
                }}
              />
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
        {rows.map((r) => (
          <div
            key={r.step}
            style={{
              flex: 1,
              minWidth: 34,
              fontSize: 8,
              fontFamily: MONO,
              color: VERDICT_COLOUR[r.verdict] ?? T.dim,
              textAlign: "center",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            title={r.step}
          >
            {r.step}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Blame                                                             */
/* ------------------------------------------------------------------ */

/**
 * The blame chain for one starved step, walked backwards.
 *
 * `prop:blame` says this terminates within m hops, because the inputs a step
 * names are bound by strictly earlier steps and there are finitely many. The
 * walk below is bounded by the step count for exactly that reason, and if the
 * bound were ever reached that would be a defect in the engine rather than a
 * long chain -- so it is rendered as such instead of being silently truncated.
 */
export function BlameChain({ steps, from }) {
  const byName = useMemo(() => {
    const m = {};
    for (const s of steps || []) m[s.step] = s;
    return m;
  }, [steps]);

  const chain = [];
  let cur = from;
  let guard = (steps || []).length + 1;
  const seen = new Set();
  while (cur && byName[cur] && guard-- > 0 && !seen.has(cur)) {
    seen.add(cur);
    const s = byName[cur];
    chain.push(s);
    // The engine writes the culprit into the diagnosis prose as
    // `bound input <name> returned <verdict>`. Reading the name back out of
    // that sentence is the only route available: the engine does not emit a
    // structured `culprit` field, and inventing one here would be a claim
    // about the engine made by the page.
    //
    // The diagnosis is a BARE STRING on `steps[]` and an object elsewhere, so
    // the text is taken through the same reader the table uses. Matching
    // against `d.reason` alone silently yielded `""` on every step and drew a
    // one-hop chain captioned "terminates in 1 hop" -- true of the drawing,
    // false of the execution.
    const m = /bound input (\w+) returned/.exec(diagnosisText(s.diagnosis));
    cur = m ? m[1] : null;
  }
  if (!chain.length) return null;

  const terminated = guard > 0;
  return (
    <div>
      {chain.map((s, i) => (
        <div
          key={s.step}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "5px 0",
            borderBottom: i === chain.length - 1 ? "none" : `1px solid ${T.panel}`,
          }}
        >
          <span style={{ fontSize: 9, color: T.muted, fontFamily: MONO, width: 42 }}>
            hop {i}
          </span>
          <span style={{ fontSize: 10, color: T.text, fontFamily: MONO, width: 96 }}>
            {s.step}
          </span>
          <VerdictChip verdict={s.verdict} blocker={s.blocker} />
          <span style={{ fontSize: 9, color: T.dim, fontFamily: MONO }}>
            {diagnosisText(s.diagnosis)}
          </span>
        </div>
      ))}
      <div style={{ fontSize: 9, color: terminated ? T.ok : T.err, fontFamily: MONO, marginTop: 7 }}>
        {terminated
          ? `chain terminates in ${chain.length} hop${chain.length === 1 ? "" : "s"} ` +
            `(prop:blame bounds it by the ${(steps || []).length} steps of the plan)`
          : "chain did not terminate within the step count — engine defect"}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Code                                                              */
/* ------------------------------------------------------------------ */

/** A read-only plan listing. Whitespace is significant to the eye here. */
export function Code({ text, max = 420 }) {
  return (
    <pre
      style={{
        margin: 0,
        background: T.bg,
        border: `1px solid ${T.border}`,
        borderRadius: 4,
        padding: 10,
        fontSize: 10,
        lineHeight: 1.65,
        fontFamily: MONO,
        color: T.text,
        overflowX: "auto",
        maxHeight: max,
        overflowY: "auto",
        whiteSpace: "pre",
      }}
    >
      {text}
    </pre>
  );
}

/**
 * A cell whose output is collapsed until asked for. Used where the raw JSON is
 * evidence rather than argument: it should be reachable without being in the
 * way, and it should be the real object, not a summary of it.
 */
export function Raw({ label = "raw JSON", value, max = 300 }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 8 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: "transparent",
          border: `1px solid ${T.border}`,
          borderRadius: 3,
          color: T.dim,
          fontSize: 9,
          fontFamily: MONO,
          padding: "2px 7px",
          cursor: "pointer",
        }}
      >
        {open ? "▾" : "▸"} {label}
      </button>
      {open && <div style={{ marginTop: 6 }}><Code text={JSON.stringify(value, null, 1)} max={max} /></div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Notebook chrome                                                   */
/* ------------------------------------------------------------------ */

/**
 * A numbered notebook cell: prose, then the thing the prose is about.
 *
 * The number is passed in rather than counted by a context, so a cell can be
 * moved while reading the source without renumbering silently.
 */
export function Cell({ n, title, children }) {
  return (
    <section style={{ marginBottom: 34 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 9 }}>
        <span
          style={{
            fontSize: 10,
            fontFamily: MONO,
            color: T.muted,
            border: `1px solid ${T.border}`,
            borderRadius: 3,
            padding: "1px 6px",
            flexShrink: 0,
          }}
        >
          [{n}]
        </span>
        <h2 style={{ fontSize: 15, color: T.text, margin: 0, fontWeight: 600, letterSpacing: 0.2 }}>
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}

/** Body prose. Kept narrow: this is a document, and long lines read badly. */
export function P({ children }) {
  return (
    <p style={{ fontSize: 12.5, lineHeight: 1.75, color: "#a9b1d6", margin: "0 0 11px 0", maxWidth: 760 }}>
      {children}
    </p>
  );
}

/** Inline monospace for identifiers and plan fragments. */
export function K({ children }) {
  return (
    <code
      style={{
        fontFamily: MONO,
        fontSize: 11,
        color: T.accent,
        background: T.panel,
        borderRadius: 3,
        padding: "1px 4px",
      }}
    >
      {children}
    </code>
  );
}

export { Panel, T, MONO };
