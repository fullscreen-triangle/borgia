/**
 * RunCharts — charts for what a program actually computed.
 *
 * The workbench is an editor: you write a program, you run it, you see the
 * result.  Until now "the result" was three key/value rows and a log, while
 * every typed value the interpreter produced -- Atom, Bond, Compound, Path,
 * Scalar, each carrying a floor and a residue, Paths carrying whole
 * trajectories -- was flattened into one line of text and discarded.
 *
 * Nothing here invents data.  Every series is read from `res.named`, which
 * the engine already returns and the page already receives.  If a program
 * binds nothing chartable, these panels say so rather than drawing
 * decoration.
 *
 * Four readings, in the order a run tends to want them:
 *
 *   1. Track trajectories  -- Path values, plotted as convergence
 *   2. Ladder invariants   -- rung power profiles, circulation, uniformity
 *   3. Spectra / modes     -- mode sets emitted by a spectrum program
 *   4. Numeric bindings    -- everything else that carries a number
 */

import { useMemo } from "react";
import {
  CHART_THEME as T,
  CHART_MONO as MONO,
  Panel,
  BarRows,
  Scatter,
  LadderRing,
  SERIES,
  fmt,
} from "./Charts";

/* ------------------------------------------------------------------ */
/*  Reading the run                                                    */
/* ------------------------------------------------------------------ */

/**
 * Split `named` into the kinds we know how to draw.  A value the
 * interpreter produced but this file does not understand is counted, not
 * hidden: the panel says how many were skipped.
 */
export function readRun(named) {
  const paths = [];
  const atoms = [];
  const bonds = [];
  const compounds = [];
  const scalars = [];
  const spectra = [];
  let other = 0;

  for (const [name, v] of Object.entries(named || {})) {
    if (!v || typeof v !== "object") {
      if (typeof v === "number") scalars.push({ name, value: v });
      else other += 1;
      continue;
    }
    switch (v.ty) {
      case "Path": paths.push({ name, ...v }); break;
      case "Atom": atoms.push({ name, ...v }); break;
      case "Bond": bonds.push({ name, ...v }); break;
      case "Compound": compounds.push({ name, ...v }); break;
      case "Scalar": scalars.push({ name, value: v.value, floor: v.floor }); break;
      default:
        if (Array.isArray(v.modes)) spectra.push({ name, ...v });
        else other += 1;
    }
  }
  return { paths, atoms, bonds, compounds, scalars, spectra, other };
}

/* ------------------------------------------------------------------ */
/*  Ladder algebra — same definitions as the papers                    */
/* ------------------------------------------------------------------ */

export function circulation(powers) {
  let t = 0;
  for (const p of powers) {
    if (p >= 1) return Infinity;
    t += -Math.log(1 - p);
  }
  return t;
}

export function uniformity(powers) {
  if (!powers.length) return 1;
  const mean = powers.reduce((a, b) => a + b, 0) / powers.length;
  if (mean <= 0) return 1;
  const varr =
    powers.reduce((a, b) => a + (b - mean) * (b - mean), 0) / powers.length;
  return Math.max(0, 1 - Math.sqrt(varr) / mean);
}

export function composite(powers) {
  let p = 1;
  for (const x of powers) p *= 1 - x;
  return 1 - p;
}

/**
 * The rung powers of a Path.
 *
 * A Path records `steps`, an `amalgamation` naming the contacts the item
 * passed through, and `reps` — which holds representation *names*
 * ("mass", "charge", "time"), not numbers.  Only the amalgamation
 * corresponds to rungs: one committed contact each.
 *
 * If the interpreter later emits explicit per-rung powers we use them.
 * Otherwise each committed contact closes an equal share of the standing
 * ambiguity, which is the uniform profile — and we say so in the panel
 * rather than presenting a derived assumption as measured data.
 */
function pathPowers(p) {
  if (Array.isArray(p.powers) && p.powers.length) {
    const nums = p.powers.map(Number).filter((x) => Number.isFinite(x));
    if (nums.length) return nums;
  }
  const n = Array.isArray(p.amalgamation)
    ? p.amalgamation.length
    : Number(p.steps) || 0;
  if (n <= 0) return [];
  // Equal share per committed contact: the residue falls by the same
  // factor at each rung, so n rungs take the standing gap to residue/floor.
  const target = p.floor > 0 && p.residue > 0
    ? Math.min(0.999, 1 - p.floor / (p.residue + p.floor))
    : 0.5;
  const per = 1 - Math.pow(1 - target, 1 / n);
  return Array(n).fill(per);
}

/** Whether a Path's profile is measured or the uniform fallback. */
function powersAreMeasured(p) {
  return Array.isArray(p.powers) &&
    p.powers.map(Number).filter((x) => Number.isFinite(x)).length > 0;
}

/* ------------------------------------------------------------------ */
/*  Panels                                                             */
/* ------------------------------------------------------------------ */

function Empty({ what }) {
  return (
    <div style={{ fontSize: 11.5, color: T.dim, lineHeight: 1.6 }}>
      This run bound no {what}. The charts read the program&apos;s own
      bindings, so a program that computes none draws none.
    </div>
  );
}

/** 1. Track trajectories. */
function Trajectories({ paths }) {
  if (!paths.length) return <Empty what="paths (`track ... yield`)" />;

  return (
    <>
      {paths.map((p, i) => {
        const powers = pathPowers(p);
        // Residue remaining after each step: the ladder's own reading of
        // how far the trajectory still has to go.
        let gap = 1;
        const conv = powers.map((v, k) => {
          gap *= 1 - v;
          return { x: k + 1, y: gap };
        });
        return (
          <div key={p.name} style={{ marginBottom: 18 }}>
            <div style={{
              fontFamily: MONO, fontSize: 11.5, color: T.text,
              marginBottom: 6,
            }}>
              {p.name}
              <span style={{ color: T.dim }}>
                {"  "}item={String(p.item)} · steps={p.steps} ·{" "}
                <span style={{ color: p.converged ? T.ok : T.warn }}>
                  {p.converged ? "converged" : "did not converge"}
                </span>
                {"  "}· residue={fmt(p.residue)}
              </span>
            </div>
            {conv.length > 1 ? (
              <Scatter
                series={[{
                  points: conv,
                  colour: SERIES[i % SERIES.length],
                  name: p.name,
                }]}
                xLabel="step"
                yLabel="ambiguity remaining"
                connect
                height={170}
              />
            ) : (
              <div style={{ fontSize: 11, color: T.dim }}>
                single step — nothing to plot against step index
              </div>
            )}
            {p.amalgamation?.length > 0 && (
              <div style={{
                fontFamily: MONO, fontSize: 10.5, color: T.dim, marginTop: 6,
              }}>
                amalgamation: [{p.amalgamation.join(", ")}]
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

/** 2. Ladder invariants. */
function Invariants({ paths }) {
  const rows = useMemo(
    () =>
      paths
        .map((p) => ({ p, powers: pathPowers(p) }))
        .filter((r) => r.powers.length > 0),
    [paths]
  );
  if (!rows.length) return <Empty what="rung profiles" />;

  return (
    <>
      {rows.map(({ p, powers }, i) => {
        const rho = circulation(powers);
        const u = uniformity(powers);
        const comp = composite(powers);
        return (
          <div key={p.name} style={{
            display: "flex", gap: 16, alignItems: "flex-start",
            marginBottom: 20, flexWrap: "wrap",
          }}>
            {/* A ring needs three or more rungs to be a ring; with two it
                degenerates to a line and reads as a rendering fault. Below
                three, show the profile as bars instead. */}
            {powers.length >= 3 ? (
              <LadderRing
                powers={powers}
                size={150}
                label={p.name}
                highlight={powers.indexOf(Math.min(...powers))}
              />
            ) : (
              // BarRows reserves a 128px label column, which does not fit
              // beside the invariants; draw the profile directly instead.
              <div style={{ width: 150, flexShrink: 0 }}>
                <div style={{ fontSize: 10, color: T.dim, marginBottom: 5 }}>
                  RUNG PROFILE
                </div>
                {powers.map((v, k) => (
                  <div key={k} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    marginBottom: 4,
                  }}>
                    <div style={{
                      fontSize: 10, color: T.dim, width: 16, flexShrink: 0,
                    }}>{k + 1}</div>
                    <div style={{
                      flex: 1, height: 12, background: T.bg,
                      borderRadius: 2, overflow: "hidden",
                    }}>
                      <div style={{
                        width: `${Math.min(100, v * 100)}%`, height: "100%",
                        background: SERIES[k % SERIES.length],
                      }} />
                    </div>
                    <div style={{
                      fontSize: 10, fontFamily: MONO, color: T.text,
                      width: 42, textAlign: "right", flexShrink: 0,
                    }}>{v.toFixed(3)}</div>
                  </div>
                ))}
                <div style={{
                  fontSize: 9.5, color: T.dim, marginTop: 6, lineHeight: 1.5,
                }}>
                  {powers.length} rung{powers.length === 1 ? "" : "s"} — too
                  few to draw as a cycle
                </div>
              </div>
            )}
            <div style={{ flex: 1, minWidth: 190 }}>
              <BarRows
                rows={[
                  { label: "circulation ϱ", value: rho },
                  { label: "ϱ per rung", value: rho / powers.length },
                  { label: "uniformity u", value: u },
                  { label: "composite power", value: comp },
                ]}
                max={Math.max(rho, 1)}
                dp={4}
              />
              <div style={{
                fontSize: 10.5, color: T.dim, marginTop: 8, lineHeight: 1.6,
              }}>
                {powers.length} rung{powers.length === 1 ? "" : "s"}
                {powersAreMeasured(p)
                  ? ". Powers are the run's own."
                  : ", one per committed contact. The interpreter records the contacts but not a power for each, so the profile is the uniform one consistent with this path's residue — u = 1 here is that assumption, not a measurement."}
              </div>
            </div>
          </div>
        );
      })}
    </>
  );
}

/** 3. Spectra and mode sets. */
function Spectra({ spectra }) {
  if (!spectra.length) return <Empty what="spectra" />;
  return (
    <>
      {spectra.map((s, i) => {
        const modes = (s.modes || []).map((m, k) => ({
          label: m.label || m.name || `mode ${k + 1}`,
          value: Number(m.value ?? m.freq ?? m.wavenumber ?? 0),
        }));
        if (!modes.length) return null;
        const max = Math.max(...modes.map((m) => m.value));
        return (
          <div key={s.name} style={{ marginBottom: 16 }}>
            <div style={{
              fontFamily: MONO, fontSize: 11.5, color: T.text, marginBottom: 6,
            }}>
              {s.name}
              <span style={{ color: T.dim }}>
                {"  "}{modes.length} mode{modes.length === 1 ? "" : "s"}
              </span>
            </div>
            <BarRows rows={modes} max={max} unit=" cm⁻¹" dp={2} />
          </div>
        );
      })}
    </>
  );
}

/** 4. Everything numeric the program bound. */
function Numerics({ scalars, atoms, bonds, compounds }) {
  const rows = [];
  for (const s of scalars) rows.push({ label: s.name, value: Number(s.value) });
  for (const a of atoms) {
    rows.push({ label: `${a.name} · Z`, value: Number(a.Z) });
    if (a.vacancy != null)
      rows.push({ label: `${a.name} · vacancy`, value: Number(a.vacancy) });
    if (a.valence != null)
      rows.push({ label: `${a.name} · valence`, value: Number(a.valence) });
  }
  for (const b of bonds) {
    if (b.delta != null)
      rows.push({ label: `${b.name} · Δ thickness`, value: Number(b.delta) });
    if (b.shared != null)
      rows.push({ label: `${b.name} · shared`, value: Number(b.shared) });
  }
  for (const c of compounds) {
    if (c.angleDeg != null)
      rows.push({ label: `${c.name} · angle°`, value: Number(c.angleDeg) });
  }
  const clean = rows.filter((r) => Number.isFinite(r.value));
  if (!clean.length) return <Empty what="numeric bindings" />;
  const max = Math.max(...clean.map((r) => Math.abs(r.value)), 1);
  return <BarRows rows={clean} max={max} dp={4} />;
}

/* ------------------------------------------------------------------ */
/*  The chart pane                                                     */
/* ------------------------------------------------------------------ */

/**
 * The whole chart view for one run.
 *
 * Shown beside the textual result rather than instead of it: the numbers
 * are the answer, the charts are how a run is read at a glance.
 */
export default function RunCharts({ res }) {
  const parsed = useMemo(() => readRun(res?.named), [res]);

  if (!res) {
    return (
      <div style={{ fontSize: 11.5, color: T.dim, lineHeight: 1.6 }}>
        No run yet. Press ▶ Run, or Ctrl+Enter — the charts read whatever the
        program binds.
      </div>
    );
  }
  if (!res.ok) {
    return (
      <div style={{
        fontSize: 11.5, color: T.err, lineHeight: 1.6,
        border: `1px solid ${T.err}`, borderRadius: 4, padding: 10,
        background: T.bg, whiteSpace: "pre-wrap",
      }}>
        The run failed, so there is nothing to chart.
        {res.error ? `\n\n${res.error}` : ""}
      </div>
    );
  }

  const { paths, spectra, scalars, atoms, bonds, compounds, other } = parsed;
  const nothing =
    !paths.length && !spectra.length && !scalars.length &&
    !atoms.length && !bonds.length && !compounds.length;

  if (nothing) {
    return (
      <div style={{ fontSize: 11.5, color: T.dim, lineHeight: 1.6 }}>
        The run completed and bound nothing chartable
        {other > 0 ? ` (${other} value${other === 1 ? "" : "s"} of a kind these charts do not read)` : ""}.
        Bind a value with <span style={{ fontFamily: MONO }}>let</span> — an
        atom, a bond, a compound, a track — and it appears here.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {paths.length > 0 && (
        <Panel
          title="Track trajectories"
          subtitle="ambiguity remaining after each committed step"
          source="res.named — Path values"
        >
          <Trajectories paths={paths} />
        </Panel>
      )}

      {paths.length > 0 && (
        <Panel
          title="Ladder invariants"
          subtitle="circulation and uniformity of each rung profile"
          source="computed from the run's paths"
          note="ϱ is a reparametrisation of composite power for a linear ladder; it earns its place on cycles, where composite power is undefined. Where the interpreter records contacts but not per-rung powers, the profile shown is the uniform one consistent with the path's residue, and each panel says so."
        >
          <Invariants paths={paths} />
        </Panel>
      )}

      {spectra.length > 0 && (
        <Panel
          title="Spectra"
          subtitle="mode sets emitted by this run"
          source="res.named — spectrum values"
        >
          <Spectra spectra={spectra} />
        </Panel>
      )}

      <Panel
        title="Numeric bindings"
        subtitle="every number this program bound"
        source="res.named"
      >
        <Numerics
          scalars={scalars}
          atoms={atoms}
          bonds={bonds}
          compounds={compounds}
        />
      </Panel>

      {other > 0 && (
        <div style={{ fontSize: 10.5, color: T.dim }}>
          {other} bound value{other === 1 ? "" : "s"} of a kind these charts do
          not read — shown in full in the Result tab.
        </div>
      )}
    </div>
  );
}
