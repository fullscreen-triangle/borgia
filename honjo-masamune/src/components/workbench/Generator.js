/**
 * The periodic table as a consequence, not a table.
 *
 * Every cell here is computed from Z at render time by `deriveAtom` — the same
 * function `cut Z` calls in the language. Nothing on this screen is looked up
 * from an 18-row array, which is the entire claim the generator exists to make
 * visible: if the Madelung order and the shell capacities are right, the table
 * FALLS OUT, including the ten aufbau exceptions that a naive filling gets
 * wrong.
 *
 * The layout is deliberately the real table (18 columns, f-block dropped
 * below), because a derivation that reproduces the familiar shape is checkable
 * by eye in a way a list of configurations is not. Clicking a cell shows the
 * arithmetic that placed it there.
 */

import { useState, useMemo } from "react";
import { deriveAtom, MAX_Z, shellCapacity, subshellCapacity } from "@/lib/honjo";
import { CHART_THEME as T, CHART_MONO as MONO, Panel } from "./Charts";

const L_LETTER = ["s", "p", "d", "f", "g"];

/** Category from the derived group/period alone — no element table consulted. */
function category(a) {
  if (a.group === null) return a.period >= 7 ? "actinide" : "lanthanide";
  if (a.group === 18) return "noble";
  if (a.group === 1 && a.Z !== 1) return "alkali";
  if (a.group === 2) return "alkaline";
  if (a.group >= 3 && a.group <= 12) return "transition";
  if (a.Z === 1) return "nonmetal";
  // The metalloid staircase is a chemical convention, not an arithmetic
  // consequence, so it is named explicitly rather than pretending to derive.
  const metalloid = { 5: 1, 14: 1, 32: 1, 33: 1, 51: 1, 52: 1, 84: 1 };
  if (metalloid[a.Z]) return "metalloid";
  if (a.group === 17) return "halogen";
  if (a.group >= 13 && a.period >= a.group - 10) return "post";
  return "nonmetal";
}

const CAT_COLOUR = {
  alkali: "#f7768e", alkaline: "#ff9e64", transition: "#e0af68",
  post: "#9ece6a", metalloid: "#73daca", nonmetal: "#7dcfff",
  halogen: "#bb9af7", noble: "#c0caf5", lanthanide: "#2ac3de",
  actinide: "#b4f9f8",
};

/** Column in the printed table. The f-block is pulled out into its own rows. */
function tableCell(a) {
  if (a.group === null) {
    // f-block: ordered by how far into the f subshell the filling has gone.
    const f = a.config.find((s) => s.l === 3);
    return { row: a.period === 6 ? 9 : 10, col: 2 + (f ? f.occ : 1) };
  }
  return { row: a.period, col: a.group };
}

/** All 118 atoms, derived once. */
function useTable() {
  return useMemo(() => {
    const out = [];
    for (let Z = 1; Z <= MAX_Z; Z++) {
      try {
        const a = deriveAtom(Z);
        out.push({ ...a, cat: category(a), cell: tableCell(a) });
      } catch (e) {
        out.push({ Z, error: String(e.message ?? e) });
      }
    }
    return out;
  }, []);
}

export default function Generator() {
  const atoms = useTable();
  const [sel, setSel] = useState(6);          // carbon: the corpus's worked example
  const [hover, setHover] = useState(null);
  const [tint, setTint] = useState("category");

  const shown = hover ?? sel;
  const atom = atoms.find((a) => a.Z === shown);
  const failed = atoms.filter((a) => a.error);
  const exceptions = atoms.filter((a) => a.exception);

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      <Header failed={failed.length} exceptions={exceptions.length} />

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 640px", minWidth: 520 }}>
          <Panel
            title="cut Z  —  118 atoms, none of them tabulated"
            subtitle="Each cell is deriveAtom(Z) evaluated in your browser now. Colour encodes a derived quantity, not a stored label. Hover to inspect, click to pin."
            source="honjo/src/shell.ts · aufbauConfig → deriveConfiguration → periodGroup"
          >
            <TintPicker tint={tint} setTint={setTint} />
            <Table
              atoms={atoms} tint={tint} sel={sel} hover={hover}
              onHover={setHover} onPick={setSel}
            />
            <Legend tint={tint} />
          </Panel>
        </div>

        <div style={{ flex: "0 1 400px", minWidth: 340 }}>
          {atom && !atom.error ? <AtomCard a={atom} /> : null}
          <ExceptionPanel exceptions={exceptions} onPick={setSel} />
          <CapacityPanel />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Header({ failed, exceptions }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: T.text, fontFamily: MONO }}>
        Atom generator
      </div>
      <div style={{ fontSize: 11.5, color: T.dim, marginTop: 5, lineHeight: 1.6, maxWidth: 900 }}>
        The periodic table is not stored anywhere in this program. It is produced
        from the Madelung filling order and the subshell capacities 2(2ℓ+1), and
        the fact that the result looks like the table is the check.{" "}
        <span style={{ color: failed ? T.err : T.ok }}>
          {failed ? `${failed} of ${MAX_Z} failed to derive` : `all ${MAX_Z} derive`}
        </span>
        , with <span style={{ color: T.warn }}>{exceptions} aufbau exceptions</span> corrected
        against measurement.
      </div>
    </div>
  );
}

const TINTS = [
  ["category", "block"],
  ["vacancy", "vacancy"],
  ["valence", "valence"],
  ["exception", "exceptions"],
];

function TintPicker({ tint, setTint }) {
  return (
    <div style={{ display: "flex", gap: 4, marginBottom: 10 }}>
      {TINTS.map(([id, label]) => (
        <button key={id} onClick={() => setTint(id)} style={{
          padding: "3px 10px", fontSize: 10, fontFamily: MONO,
          background: tint === id ? T.accent : "transparent",
          color: tint === id ? T.bg : T.dim,
          border: `1px solid ${tint === id ? T.accent : T.border}`,
          borderRadius: 3, cursor: "pointer",
        }}>{label}</button>
      ))}
    </div>
  );
}

function cellColour(a, tint) {
  if (a.error) return T.err;
  if (tint === "category") return CAT_COLOUR[a.cat] ?? T.dim;
  if (tint === "exception") return a.exception ? T.warn : T.muted;
  const v = tint === "vacancy" ? a.vacancy : a.valence;
  const hi = tint === "vacancy" ? 14 : 4;
  const f = Math.min(v / hi, 1);
  return `hsl(${210 - f * 150}, 70%, ${38 + f * 22}%)`;
}

function Table({ atoms, tint, sel, hover, onHover, onPick }) {
  const CW = 34, CH = 30, GAP = 2;
  const rows = 11;
  const width = 18 * (CW + GAP);
  const height = rows * (CH + GAP) + 14;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}
           style={{ display: "block", minWidth: 560 }}
           onMouseLeave={() => onHover(null)}>
        {atoms.map((a) => {
          if (a.error) return null;
          const { row, col } = a.cell;
          const x = (col - 1) * (CW + GAP);
          const y = (row - 1) * (CH + GAP) + (row > 8 ? 10 : 0);
          const on = a.Z === (hover ?? sel);
          return (
            <g key={a.Z}
               onMouseEnter={() => onHover(a.Z)}
               onClick={() => onPick(a.Z)}
               style={{ cursor: "pointer" }}>
              <rect x={x} y={y} width={CW} height={CH} rx="3"
                    fill={cellColour(a, tint)}
                    opacity={on ? 1 : 0.68}
                    stroke={on ? T.text : "none"} strokeWidth={on ? 1.4 : 0} />
              <text x={x + CW / 2} y={y + 13} textAnchor="middle"
                    fontSize="10.5" fontWeight="700" fill={T.bg} fontFamily={MONO}>
                {a.symbol}
              </text>
              <text x={x + CW / 2} y={y + 24} textAnchor="middle"
                    fontSize="7.5" fill={T.bg} opacity="0.8" fontFamily={MONO}>
                {a.Z}
              </text>
              {a.exception && tint !== "exception" && (
                <circle cx={x + CW - 4} cy={y + 4} r="2" fill={T.bg} opacity="0.75" />
              )}
              <title>{`${a.symbol} (Z=${a.Z})  ${a.configStr}  ${a.term}`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Legend({ tint }) {
  if (tint === "category") {
    return (
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
        {Object.entries(CAT_COLOUR).map(([k, c]) => (
          <span key={k} style={{ fontSize: 9, color: T.dim, fontFamily: MONO }}>
            <span style={{
              display: "inline-block", width: 8, height: 8, borderRadius: 2,
              background: c, marginRight: 4,
            }} />{k}
          </span>
        ))}
      </div>
    );
  }
  if (tint === "exception") {
    return (
      <div style={{ fontSize: 10, color: T.dim, marginTop: 10, lineHeight: 1.6 }}>
        Highlighted cells are the ten atoms whose measured ground state disagrees
        with strict Madelung filling. They are the honest part of the derivation:
        the rule is a good rule, and these are where it is wrong.
      </div>
    );
  }
  return (
    <div style={{ fontSize: 10, color: T.dim, marginTop: 10, lineHeight: 1.6 }}>
      Colour runs low (blue) to high (warm). {tint === "vacancy"
        ? "Vacancy is capV − qv: the holes in the valence shell."
        : "Valence is min(vacancy, capV − vacancy) — bonding capacity peaks at half filling, which is why it is not simply the hole count."}
    </div>
  );
}

/* ------------------------------------------------------------------ */

function AtomCard({ a }) {
  const shells = useMemo(() => {
    const by = new Map();
    for (const s of a.config) by.set(s.n, (by.get(s.n) ?? 0) + s.occ);
    return [...by.entries()].sort((x, y) => x[0] - y[0]);
  }, [a]);

  const outer = shells[shells.length - 1];

  return (
    <Panel
      title={`${a.symbol}  ·  Z = ${a.Z}`}
      subtitle={a.configStr}
      source="derived now — no value on this card was read from a table"
    >
      <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
        <ShellDiagram shells={shells} />
        <div style={{ flex: 1 }}>
          <Row k="term symbol" v={a.term} />
          <Row k="period / group" v={`${a.period} / ${a.group ?? "f-block"}`} />
          <Row k="valence electrons" v={`${a.qv} of ${a.capV}`} />
          <Row k="vacancy" v={a.vacancy} />
          <Row k="bonding valence" v={a.valence} />
          <Row k="aufbau" v={a.exception ? "EXCEPTION" : "regular"} warn={a.exception} />
        </div>
      </div>

      <div style={{
        marginTop: 10, fontSize: 9.5, color: T.dim, lineHeight: 1.6,
        borderLeft: `2px solid ${T.muted}`, paddingLeft: 8,
      }}>
        Shell {outer[0]} holds {outer[1]} of {shellCapacity(outer[0])} = 2n² places.
        {a.exception &&
          " Strict Madelung order predicts a different ground state here; the measured configuration is used instead."}
      </div>
    </Panel>
  );
}

/** Occupancy per principal shell, drawn against the 2n² capacity. */
function ShellDiagram({ shells }) {
  const size = 116, cx = size / 2, cy = size / 2;
  return (
    <svg width={size} height={size} style={{ flexShrink: 0 }}>
      {shells.map(([n, occ], i) => {
        const cap = shellCapacity(n);
        const r = 12 + i * 13;
        const frac = occ / cap;
        const circ = 2 * Math.PI * r;
        return (
          <circle key={n} cx={cx} cy={cy} r={r} fill="none"
                  stroke={frac >= 1 ? T.ok : T.accent} strokeWidth="2.5"
                  strokeDasharray={`${circ * frac} ${circ}`}
                  transform={`rotate(-90 ${cx} ${cy})`} strokeLinecap="round">
            <title>{`n=${n}: ${occ}/${cap}`}</title>
          </circle>
        );
      })}
      <circle cx={cx} cy={cy} r="5" fill={T.violet} />
    </svg>
  );
}

function ExceptionPanel({ exceptions, onPick }) {
  return (
    <Panel
      title="Where the rule breaks"
      subtitle="Madelung filling is a rule of thumb, and these atoms are measured to disagree with it. They are corrected explicitly rather than smoothed away."
      source="honjo/src/shell.ts · AUFBAU_EXCEPTIONS"
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {exceptions.map((a) => (
          <button key={a.Z} onClick={() => onPick(a.Z)} style={{
            padding: "4px 8px", fontSize: 10, fontFamily: MONO,
            background: "transparent", border: `1px solid ${T.warn}`,
            color: T.warn, borderRadius: 3, cursor: "pointer",
          }} title={a.configStr}>{a.symbol} {a.Z}</button>
        ))}
      </div>
    </Panel>
  );
}

/** C(n) = Σ 2(2ℓ+1) summed, shown against 2n² — a derived identity, not an axiom. */
function CapacityPanel() {
  const rows = useMemo(() => {
    const out = [];
    for (let n = 1; n <= 7; n++) {
      const parts = [];
      for (let l = 0; l < n; l++) parts.push(subshellCapacity(l));
      const summed = parts.reduce((a, b) => a + b, 0);
      out.push({ n, parts, summed, closed: 2 * n * n });
    }
    return out;
  }, []);
  const allAgree = rows.every((r) => r.summed === r.closed);

  return (
    <Panel
      title="C(n) = 2n², summed not assumed"
      subtitle="The capacity is computed by adding subshell capacities 2(2ℓ+1) for ℓ = 0..n−1. That it equals 2n² is then a fact to check, which is what the right column does."
      source="honjo/src/shell.ts · shellCapacity"
      note={allAgree
        ? "Every row agrees. The closed form is a consequence of the sum, so nothing in the generator has to take 2n² on faith."
        : "A row disagrees — the sum and the closed form have diverged, which is a defect."}
    >
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: MONO }}>
        <tbody>
          {rows.map((r) => (
            <tr key={r.n}>
              <td style={{ padding: "3px 4px", color: T.dim, borderBottom: `1px solid ${T.panel}` }}>
                n={r.n}
              </td>
              <td style={{ padding: "3px 4px", color: T.text, borderBottom: `1px solid ${T.panel}` }}>
                {r.parts.map((p, i) => `${p}${L_LETTER[i]}`).join(" + ")}
              </td>
              <td style={{ padding: "3px 4px", textAlign: "right", color: T.text, borderBottom: `1px solid ${T.panel}` }}>
                {r.summed}
              </td>
              <td style={{
                padding: "3px 4px", textAlign: "right",
                color: r.summed === r.closed ? T.ok : T.err,
                borderBottom: `1px solid ${T.panel}`,
              }}>
                {r.summed === r.closed ? `= 2·${r.n}²` : `≠ ${r.closed}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function Row({ k, v, warn }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      padding: "3px 0", borderBottom: `1px solid ${T.panel}`, fontSize: 10,
      fontFamily: MONO,
    }}>
      <span style={{ color: T.dim }}>{k}</span>
      <span style={{ color: warn ? T.warn : T.text, fontWeight: warn ? 700 : 400 }}>{v}</span>
    </div>
  );
}
