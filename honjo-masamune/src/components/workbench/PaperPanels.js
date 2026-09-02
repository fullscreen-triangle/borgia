/**
 * The corpus figures, live.
 *
 * The nine papers ship 50 matplotlib PNGs. This renders the same measurements
 * from the same results JSON, with two things a PNG cannot do:
 *
 *   1. RECOMPUTE. Where the quantity is cheap (the ladder invariants are pure
 *      arithmetic), the browser recomputes it from the profile and compares
 *      against the value committed in the paper. The residual is shown. If the
 *      implementation drifts from the published number, that becomes visible
 *      here instead of staying hidden until someone reruns Python.
 *
 *   2. SHOW THE REFUTATIONS. Each validator records what was expected before
 *      measuring. Two of those expectations failed. A figure of the final
 *      numbers silently drops that; the expectation table keeps it.
 */

import { useMemo, useState } from "react";
import PANELS from "@/data/panels.json";
import {
  circulation, uniformity, rotationDeviation, checkAgainst, composite,
} from "@/lib/ladder";
import {
  Panel, BarRows, Scatter, LadderRing, ExpectationTable, ChecksList,
  Agreement, fmt, SERIES, CHART_THEME as T, CHART_MONO as MONO,
} from "./Charts";

const PAPERS = PANELS.papers ?? {};

/** Human titles for the paper directories. */
const PAPER_TITLE = {
  "atomic-derivation": "Spectroscopic derivation of the elements",
  "categorical-ladder": "The categorical ladder",
  "causal-propagation-table": "Causal propagation table",
  "chemical-structure": "Chemical structure",
  "cheminformatics-model": "Cheminformatics models",
  "graphical-chemistry-generator": "Graphical chemistry generator",
  "molecular-resonator": "Molecular resonator",
};

/* ================================================================== */
/*  Ladder — the one that recomputes                                  */
/* ================================================================== */

function LadderPanels() {
  const paper = PAPERS["categorical-ladder"];
  const [selected, setSelected] = useState("benzene");

  // Every hook runs before any early return: the profiles/arom lookups are
  // written to tolerate a missing paper so the hook order stays fixed.
  const profiles = useMemo(() => paper?.profiles?.rings ?? [], [paper]);
  const arom = paper?.results?.aromaticity;
  const committed = useMemo(() => {
    const m = {};
    for (const r of arom?.measured?.rings ?? []) m[r.name] = r;
    return m;
  }, [arom]);

  // Recompute every ring from its power profile and diff against the paper.
  const rows = useMemo(() => profiles.map((r) => {
    const rho = circulation(r.profile);
    const u = uniformity(r.profile);
    const c = committed[r.name];
    return {
      ...r,
      rho, u,
      rhoCheck: checkAgainst(rho, c?.rho),
      uCheck: checkAgainst(u, c?.u),
      rotDev: rotationDeviation(r.profile, circulation),
    };
  }), [profiles, committed]);

  if (!paper) return <Missing name="categorical-ladder" />;

  const worst = rows.reduce((m, r) => Math.max(
    m, r.rhoCheck.residual ?? 0, r.uCheck.residual ?? 0), 0);
  const allAgree = rows.every((r) => r.rhoCheck.ok !== false && r.uCheck.ok !== false);
  const sel = rows.find((r) => r.name === selected) ?? rows[0];

  return (
    <>
      <Panel
        title="Closed-ladder invariants, recomputed in the browser"
        subtitle={
          <>
            ρ = −Σ log(1−pᵢ) and u = 1 − sd/mean, computed here from the power
            profiles and compared against the values committed in the paper.
            Worst residual across all rings: <b style={{ color: allAgree ? T.ok : T.err }}>
            {worst === 0 ? "exactly 0" : worst.toExponential(2)}</b>.
          </>
        }
        source="dmitri/publications/categorical-ladder/results/aromaticity.json"
      >
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 260px", minWidth: 240 }}>
            <div style={{
              fontSize: 9, color: T.dim, textTransform: "uppercase",
              letterSpacing: 0.5, marginBottom: 6, fontFamily: MONO,
            }}>
              circulation ρ — separates aromatic from saturated
            </div>
            <BarRows
              rows={rows.map((r) => ({
                label: r.name,
                value: r.rho,
                colour: r.aromatic ? T.accent : T.dim,
              }))}
              dp={4}
            />
            <div style={{ fontSize: 9, color: T.dim, marginTop: 6, fontFamily: MONO }}>
              <span style={{ color: T.accent }}>■</span> aromatic&nbsp;&nbsp;
              <span style={{ color: T.dim }}>■</span> saturated
            </div>
          </div>

          <div style={{ flex: "1 1 260px", minWidth: 240 }}>
            <div style={{
              fontSize: 9, color: T.dim, textTransform: "uppercase",
              letterSpacing: 0.5, marginBottom: 6, fontFamily: MONO,
            }}>
              uniformity u — does not
            </div>
            <BarRows
              rows={rows.map((r) => ({
                label: r.name,
                value: r.u,
                colour: r.aromatic ? T.accent : T.dim,
              }))}
              max={1}
              dp={5}
            />
          </div>
        </div>

        <div style={{
          marginTop: 12, padding: 10, background: T.panel, borderRadius: 4,
          fontSize: 10, color: T.text, lineHeight: 1.7,
        }}>
          <b style={{ color: T.warn }}>The refuted prediction, kept.</b> Benzene and
          cyclohexane both score u&nbsp;=&nbsp;1.00000: a saturated ring has every
          rung equal, so it is perfectly uniform without being aromatic. u reads
          the <i>dispersion</i> of rung values, not the <i>symmetry</i> of the
          pattern. Only ρ separates them ({fmt(rows.find((r) => r.name === "benzene")?.rho, 3)} vs{" "}
          {fmt(rows.find((r) => r.name === "cyclohexane")?.rho, 3)}).
        </div>
      </Panel>

      <Panel
        title="A closed ladder is a cycle"
        subtitle="Rung thickness tracks power. Rotating the ring changes nothing the invariants can see — the deviation across all rotations is shown below, and it should sit at machine zero."
      >
        <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <LadderRing powers={sel.profile} label={`n=${sel.profile.length}`} />
          </div>
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 10 }}>
              {rows.map((r) => (
                <button
                  key={r.name}
                  onClick={() => setSelected(r.name)}
                  style={{
                    padding: "2px 8px", fontSize: 10, fontFamily: MONO,
                    background: r.name === sel.name ? T.accent : "transparent",
                    color: r.name === sel.name ? T.bg : T.dim,
                    border: `1px solid ${r.name === sel.name ? T.accent : T.border}`,
                    borderRadius: 3, cursor: "pointer",
                  }}
                >{r.name}</button>
              ))}
            </div>
            <Row k="rungs" v={String(sel.profile.length)} />
            <Row k="profile" v={sel.profile.map((p) => p.toFixed(2)).join("  ")} />
            <Row k="ρ (recomputed)" v={fmt(sel.rho, 8)}
                 badge={<Agreement ok={sel.rhoCheck.ok} residual={sel.rhoCheck.residual} />} />
            <Row k="u (recomputed)" v={fmt(sel.u, 8)}
                 badge={<Agreement ok={sel.uCheck.ok} residual={sel.uCheck.residual} />} />
            <Row k="ρ per rung" v={fmt(sel.rho / sel.profile.length, 8)} />
            <Row k="composite power" v={fmt(composite(sel.profile), 8)} />
            <Row k="rotation deviation" v={sel.rotDev === 0 ? "exactly 0" : sel.rotDev.toExponential(2)} />
          </div>
        </div>

        {paper.profiles?._derived === false && (
          <div style={{
            marginTop: 12, padding: 10, borderRadius: 4,
            border: `1px solid ${T.warn}`, fontSize: 10,
            color: T.text, lineHeight: 1.7,
          }}>
            <b style={{ color: T.warn }}>Open gap (L1).</b> These power profiles are{" "}
            <i>inputs</i>. The values 0.50 for an aromatic ring bond and 0.33/0.30
            for a saturated one were chosen by hand; they do not yet follow from
            the shell and vacancy arithmetic that produces valence and geometry
            elsewhere in the corpus. Until they do, everything above is a result
            about the formalism rather than about chemistry.
          </div>
        )}
      </Panel>

      {arom && (
        <Panel
          title="Expectations, checks, and what the run refuted"
          subtitle={
            <>
              The <code>expectation</code> block already carries the{" "}
              <i>corrected</i> prediction — it reads
              &nbsp;<code>u_separates_benzene_cyclohexane: false</code>&nbsp;
              because the naive claim was refuted and the paper was updated.
              The refutation itself is recorded in the checks below and in the
              validator&apos;s note, so the table alone would make a corrected
              prediction look like one that always held.
            </>
          }
          source="categorical-ladder/results/aromaticity.json"
          note={typeof arom.note === "string" ? arom.note : undefined}
        >
          <ExpectationTable expectation={arom.expectation} measured={arom.measured} />
          {arom.checks && (
            <div style={{ marginTop: 12 }}>
              <div style={{
                fontSize: 9, color: T.dim, fontFamily: MONO, marginBottom: 4,
                textTransform: "uppercase", letterSpacing: 0.5,
              }}>
                checks — where the refutation is actually recorded
              </div>
              <ChecksList checks={arom.checks} />
            </div>
          )}
        </Panel>
      )}

      <SubstitutionPanel paper={paper} />
      <SensitivityPanel paper={paper} />
    </>
  );
}

function SubstitutionPanel({ paper }) {
  const sub = paper.results?.substitution;
  if (!sub?.measured?.cases) return null;
  const cases = sub.measured.cases;
  return (
    <Panel
      title="Ring substitution moves u; peripheral substitution does not"
      subtitle="Δu against the number of ring letters rewritten. The relation is monotone — the second refuted prediction, since u reads dispersion rather than pattern."
      source="categorical-ladder/results/substitution.json"
    >
      <Scatter
        series={[{
          name: "Δu",
          points: cases.map((c) => ({
            x: c.ring_letters_changed ?? c.ring_changed ?? 0,
            y: c.delta_u,
            label: `${c.name}: Δu = ${fmt(c.delta_u, 5)}`,
          })),
        }]}
        connect
        xLabel="ring letters changed"
        yLabel="Δu"
        xTicks={4}
        height={190}
      />
      <div style={{ marginTop: 8 }}>
        <ExpectationTable expectation={sub.expectation} measured={sub.measured} />
      </div>
    </Panel>
  );
}

function SensitivityPanel({ paper }) {
  const sens = paper.results?.sensitivity;
  const m = sens?.measured;
  if (!m) return null;
  return (
    <Panel
      title="Control at the strongest rung is an artefact of additive pricing"
      subtitle={
        <>
          Over {m.trials?.toLocaleString?.() ?? m.trials} random ladders, the additive
          increment always peaks at the strongest rung, while proportional
          improvement is flat — spread {m.max_proportional_spread?.toExponential?.(1)}.
          The correction owed to the two earlier ladder papers.
        </>
      }
      source="categorical-ladder/results/sensitivity.json"
    >
      <BarRows
        rows={[
          { label: "additive → strongest", value: m.additive_fraction ?? 0, colour: T.accent },
          { label: "proportional flat", value: m.proportional_fraction_flat ?? 0, colour: T.violet },
        ]}
        max={1}
        dp={4}
      />
      <div style={{ marginTop: 10 }}>
        <ExpectationTable expectation={sens.expectation} measured={m} />
      </div>
    </Panel>
  );
}

/* ================================================================== */
/*  Generic paper view                                                */
/* ================================================================== */

/**
 * For papers whose quantities are not cheap to recompute, the committed
 * results are still rendered as live marks rather than a bitmap — sortable,
 * hoverable, and legible at any zoom. Nothing here claims to be recomputed.
 */
function GenericPaper({ name }) {
  const paper = PAPERS[name];
  if (!paper) return <Missing name={name} />;
  const entries = Object.entries(paper.results ?? {});

  return (
    <>
      <Panel
        title={PAPER_TITLE[name] ?? name}
        subtitle={`${entries.length} result sets · ${paper.figures?.length ?? 0} figures in the paper. Values below are read from the committed results, not recomputed.`}
        source={`dmitri/publications/${name}/results/`}
      />
      {entries.map(([stem, data]) => (
        <ResultCard key={stem} stem={stem} data={data} paper={name} />
      ))}
    </>
  );
}

/**
 * Render one results file. The shape varies across papers, so this picks the
 * best available view rather than assuming one: expectation tables where the
 * validator registered predictions, numeric bars where it produced scalars,
 * a check list where it produced verdicts.
 */
function ResultCard({ stem, data, paper }) {
  const numeric = useMemo(() => {
    const src = data?.measured ?? data;
    if (!src || typeof src !== "object") return [];
    return Object.entries(src)
      .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
      .map(([k, v]) => ({ label: k.replace(/_/g, " "), value: v }));
  }, [data]);

  /**
   * Some validators key their measurements by subject rather than by
   * quantity — the resonator reports {H2: {N, E_tree, ...}, CO: {...}}. A
   * renderer that only looked for top-level scalars drew nothing at all for
   * those papers, which reads as "no data" rather than "different shape".
   */
  const bySubject = useMemo(() => {
    const src = data?.measured ?? data;
    if (!src || typeof src !== "object") return null;
    for (const key of ["paper_expected", "measured", "results", "derived"]) {
      const v = src[key];
      if (!v || typeof v !== "object" || Array.isArray(v)) continue;
      const subjects = Object.entries(v)
        .filter(([, x]) => x && typeof x === "object" && !Array.isArray(x));
      if (subjects.length < 2) continue;
      const metrics = Object.keys(subjects[0][1])
        .filter((k) => typeof subjects[0][1][k] === "number");
      if (metrics.length === 0) continue;
      return { key, subjects, metrics };
    }
    return null;
  }, [data]);

  const series = useMemo(() => {
    const src = data?.measured ?? data;
    for (const key of Object.keys(src ?? {})) {
      const v = src[key];
      if (Array.isArray(v) && v.length > 1 && typeof v[0] === "object") {
        const numKeys = Object.keys(v[0]).filter((k) => typeof v[0][k] === "number");
        if (numKeys.length >= 2) {
          return { key, rows: v, xKey: numKeys[0], yKey: numKeys[1] };
        }
      }
    }
    return null;
  }, [data]);

  const hasContent = numeric.length > 0 || series || bySubject ||
                     data?.expectation || data?.checks;
  if (!hasContent) return null;

  return (
    <Panel
      title={stem.replace(/_/g, " ")}
      source={`${paper}/results/${stem}.json`}
      note={typeof data?.note === "string" ? data.note : undefined}
    >
      {data?.expectation && (
        <div style={{ marginBottom: numeric.length || series ? 12 : 0 }}>
          <ExpectationTable expectation={data.expectation} measured={data.measured} />
        </div>
      )}
      {series && (
        <div style={{ marginBottom: numeric.length ? 12 : 0 }}>
          <Scatter
            series={[{
              name: series.key,
              points: series.rows.map((r) => ({
                x: r[series.xKey], y: r[series.yKey],
                label: `${r.name ?? r.symbol ?? ""} ${series.xKey}=${fmt(r[series.xKey], 3)} ${series.yKey}=${fmt(r[series.yKey], 4)}`,
              })),
            }]}
            connect
            xLabel={series.xKey}
            yLabel={series.yKey}
            height={185}
          />
        </div>
      )}
      {bySubject && (
        <div style={{ marginBottom: numeric.length ? 12 : 0 }}>
          <div style={{
            fontSize: 9, color: T.dim, fontFamily: MONO, marginBottom: 6,
            textTransform: "uppercase", letterSpacing: 0.5,
          }}>
            {bySubject.key.replace(/_/g, " ")} · {bySubject.subjects.length} subjects
          </div>
          <SubjectTable subjects={bySubject.subjects} metrics={bySubject.metrics} />
        </div>
      )}
      {numeric.length > 0 && numeric.length <= 14 && (
        <BarRows rows={numeric} dp={5} />
      )}
      {numeric.length > 14 && (
        <div style={{ fontSize: 10, color: T.dim, fontFamily: MONO }}>
          {numeric.length} scalar measurements — see {paper}/results/{stem}.json
        </div>
      )}
      {data?.checks && (
        <div style={{ marginTop: 10 }}>
          <ChecksList checks={data.checks} />
        </div>
      )}
    </Panel>
  );
}

/** Metrics keyed by subject, e.g. one row per molecule. */
function SubjectTable({ subjects, metrics }) {
  const cols = metrics.slice(0, 8);
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 10, fontFamily: MONO, minWidth: "100%" }}>
        <thead>
          <tr>
            <th style={{
              textAlign: "left", padding: "4px 8px", color: T.dim, fontWeight: 500,
              borderBottom: `1px solid ${T.border}`, fontSize: 9,
            }}>subject</th>
            {cols.map((m) => (
              <th key={m} style={{
                textAlign: "right", padding: "4px 8px", color: T.dim, fontWeight: 500,
                borderBottom: `1px solid ${T.border}`, fontSize: 9,
              }}>{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {subjects.map(([name, row]) => (
            <tr key={name}>
              <td style={{ padding: "3px 8px", color: T.text, borderBottom: `1px solid ${T.panel}` }}>
                {name}
              </td>
              {cols.map((m) => (
                <td key={m} style={{
                  padding: "3px 8px", textAlign: "right", color: T.text,
                  borderBottom: `1px solid ${T.panel}`,
                }}>
                  {typeof row[m] === "number" ? fmt(row[m], Number.isInteger(row[m]) ? 0 : 4) : "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Missing({ name }) {
  return (
    <Panel title={name} subtitle="No results found for this paper.">
      <div style={{ fontSize: 10, color: T.dim, fontFamily: MONO }}>
        Run <code>python src/data/generate.py</code> to regenerate panel data.
      </div>
    </Panel>
  );
}

function Row({ k, v, badge }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      fontSize: 10, fontFamily: MONO, padding: "3px 0",
      borderBottom: `1px solid ${T.panel}`,
    }}>
      <span style={{ color: T.dim }}>{k}</span>
      <span style={{ color: T.text, textAlign: "right" }}>{v}{badge}</span>
    </div>
  );
}

/* ================================================================== */
/*  Entry                                                             */
/* ================================================================== */

export const PAPER_NAMES = Object.keys(PAPERS);

export default function PaperPanels({ paper }) {
  if (paper === "categorical-ladder") return <LadderPanels />;
  return <GenericPaper name={paper} />;
}

/** Count of result files, for the tab label. */
export const RESULT_COUNT = Object.values(PAPERS)
  .reduce((n, p) => n + Object.keys(p.results ?? {}).length, 0);

export const FIGURE_COUNT = Object.values(PAPERS)
  .reduce((n, p) => n + (p.figures?.length ?? 0), 0);
