/**
 * Chart primitives shared by the panels: Panel chrome, and the marks that
 * carry a provenance state — an Agreement row, an ExpectationTable, a
 * ChecksList. These render *whether a number was checked*, which is why they
 * are hand-built rather than taken from a library.
 *
 * The plotting marks live in D3Charts.js and are built on d3 scales and shape
 * generators. Anything with an axis belongs there.
 */

const T = {
  bg: "#1a1b26", surface: "#1e1f2e", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", muted: "#3b3d57", accent: "#7dcfff",
  ok: "#9ece6a", warn: "#e0af68", err: "#f7768e", violet: "#bb9af7",
  orange: "#ff9e64",
};

const MONO = "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace";

/** Series colours, ordered so adjacent series stay distinguishable. */
export const SERIES = [T.accent, T.violet, T.ok, T.orange, T.warn, T.err];

export function fmt(x, dp = 4) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "−∞";
  if (x !== 0 && Math.abs(x) < 1e-3) return x.toExponential(1);
  return x.toFixed(dp);
}

/* ------------------------------------------------------------------ */
/*  Frame                                                             */
/* ------------------------------------------------------------------ */

export function Panel({ title, subtitle, source, children, note }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 6, padding: 14, marginBottom: 14,
    }}>
      <div style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: T.text, fontFamily: MONO }}>
          {title}
        </div>
        {subtitle && (
          <div style={{ fontSize: 11, color: T.dim, marginTop: 3, lineHeight: 1.5 }}>
            {subtitle}
          </div>
        )}
      </div>
      {children}
      {note && (
        <div style={{
          fontSize: 10, color: T.dim, marginTop: 10, lineHeight: 1.6,
          borderLeft: `2px solid ${T.muted}`, paddingLeft: 8,
        }}>
          {note}
        </div>
      )}
      {source && (
        <div style={{ fontSize: 9, color: T.muted, marginTop: 8, fontFamily: MONO }}>
          {source}
        </div>
      )}
    </div>
  );
}

/**
 * Agreement badge: did the browser's recomputation match the committed value?
 *
 * `ok === null` means there was nothing to check against, which is reported as
 * "not checked" rather than being allowed to look like a pass.
 */
export function Agreement({ ok, residual, label = "vs paper" }) {
  const colour = ok === null ? T.dim : ok ? T.ok : T.err;
  const text = ok === null ? "not checked" : ok ? "agrees" : "DIFFERS";
  return (
    <span style={{
      fontSize: 9, fontFamily: MONO, color: colour,
      border: `1px solid ${colour}`, borderRadius: 3, padding: "1px 5px",
      marginLeft: 6, whiteSpace: "nowrap",
    }}>
      {text} {label}
      {residual !== null && residual !== undefined && residual > 0 &&
        ` (${residual.toExponential(0)})`}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Bars                                                              */
/* ------------------------------------------------------------------ */

/**
 * Horizontal bars. `rows` is [{label, value, colour?, flag?}].
 * A `flag` renders the row muted with a marker — used where a value is real
 * but produced by a model applied outside the range it was validated on.
 */
export function BarRows({ rows, max, unit = "", height = 18, dp = 4 }) {
  const hi = max ?? Math.max(...rows.map((r) => Math.abs(r.value) || 0), 1e-9);
  const labelW = 128;
  return (
    <div>
      {rows.map((r, i) => {
        const frac = hi > 0 ? Math.abs(r.value) / hi : 0;
        const colour = r.colour ?? SERIES[i % SERIES.length];
        return (
          <div key={r.label + i} style={{
            display: "flex", alignItems: "center", height,
            marginBottom: 3, opacity: r.flag ? 0.55 : 1,
          }}>
            <div style={{
              width: labelW, flexShrink: 0, fontSize: 10, color: T.dim,
              fontFamily: MONO, overflow: "hidden", textOverflow: "ellipsis",
              whiteSpace: "nowrap", paddingRight: 6,
            }} title={r.label}>
              {r.label}
            </div>
            <div style={{ flex: 1, background: T.panel, borderRadius: 2, height: height - 6 }}>
              <div style={{
                width: `${Math.max(frac * 100, 0.5)}%`, height: "100%",
                background: colour, borderRadius: 2,
                transition: "width 200ms ease",
              }} />
            </div>
            <div style={{
              width: 84, flexShrink: 0, textAlign: "right", fontSize: 10,
              color: T.text, fontFamily: MONO, paddingLeft: 8,
            }}>
              {fmt(r.value, dp)}{unit}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Scatter / line                                                    */
/* ------------------------------------------------------------------ */

/**
 * Scatter with optional connecting line. `series` is
 * [{name, colour?, points: [{x, y, label?}]}].
 */
export function Scatter({
  series, width = 460, height = 220, xLabel, yLabel,
  connect = false, xTicks = 5, yTicks = 4, logY = false,
}) {
  const pad = { l: 52, r: 12, t: 10, b: 32 };
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return null;

  const tx = (v) => (logY ? v : v);
  const ty = (v) => (logY ? Math.log10(Math.max(v, 1e-18)) : v);

  const xs = all.map((p) => tx(p.x));
  const ys = all.map((p) => ty(p.y));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x1 === x0) { x0 -= 0.5; x1 += 0.5; }
  if (y1 === y0) { y0 -= 0.5; y1 += 0.5; }
  const padY = (y1 - y0) * 0.08;
  y0 -= padY; y1 += padY;

  const px = (x) => pad.l + ((tx(x) - x0) / (x1 - x0)) * (width - pad.l - pad.r);
  const py = (y) => height - pad.b - ((ty(y) - y0) / (y1 - y0)) * (height - pad.t - pad.b);

  const xtv = Array.from({ length: xTicks }, (_, i) => x0 + (i * (x1 - x0)) / (xTicks - 1));
  const ytv = Array.from({ length: yTicks }, (_, i) => y0 + (i * (y1 - y0)) / (yTicks - 1));

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {ytv.map((v, i) => (
        <g key={`y${i}`}>
          <line x1={pad.l} x2={width - pad.r}
                y1={py(logY ? 10 ** v : v)} y2={py(logY ? 10 ** v : v)}
                stroke={T.border} strokeWidth="1" />
          <text x={pad.l - 6} y={py(logY ? 10 ** v : v) + 3} textAnchor="end"
                fontSize="9" fill={T.dim} fontFamily={MONO}>
            {logY ? `1e${Math.round(v)}` : fmt(v, 2)}
          </text>
        </g>
      ))}
      {xtv.map((v, i) => (
        <text key={`x${i}`} x={px(v)} y={height - pad.b + 14} textAnchor="middle"
              fontSize="9" fill={T.dim} fontFamily={MONO}>
          {fmt(v, Math.abs(v) >= 100 ? 0 : 2)}
        </text>
      ))}

      {series.map((s, si) => {
        const colour = s.colour ?? SERIES[si % SERIES.length];
        const pts = [...s.points].sort((a, b) => a.x - b.x);
        return (
          <g key={s.name}>
            {connect && pts.length > 1 && (
              <polyline
                points={pts.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
                fill="none" stroke={colour} strokeWidth="1.5" opacity="0.8" />
            )}
            {s.points.map((p, i) => (
              <circle key={i} cx={px(p.x)} cy={py(p.y)} r="3.2"
                      fill={colour} opacity="0.9">
                <title>{p.label ?? `${fmt(p.x, 3)}, ${fmt(p.y, 4)}`}</title>
              </circle>
            ))}
          </g>
        );
      })}

      {xLabel && (
        <text x={(width + pad.l) / 2} y={height - 4} textAnchor="middle"
              fontSize="9" fill={T.dim} fontFamily={MONO}>{xLabel}</text>
      )}
      {yLabel && (
        <text x={11} y={height / 2} textAnchor="middle" fontSize="9" fill={T.dim}
              fontFamily={MONO} transform={`rotate(-90 11 ${height / 2})`}>{yLabel}</text>
      )}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Closed ladder                                                     */
/* ------------------------------------------------------------------ */

/**
 * A closed ladder drawn as what it is: a cycle.
 *
 * Each rung is an arc segment whose thickness and colour track its power. A
 * bar chart would order the rungs, and ordering is exactly the structure a
 * cycle does not have — the invariants are rotation-invariant, so the picture
 * should be too.
 */
export function LadderRing({ powers, size = 168, label, highlight = null }) {
  const n = powers.length;
  if (n === 0) return null;
  const cx = size / 2, cy = size / 2;
  const r = size * 0.34;
  const maxP = Math.max(...powers, 1e-9);

  const nodes = powers.map((_, i) => {
    const a = (2 * Math.PI * i) / n - Math.PI / 2;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  });

  return (
    <svg width={size} height={size} style={{ display: "block" }}>
      {powers.map((p, i) => {
        const a = nodes[i], b = nodes[(i + 1) % n];
        const frac = p / maxP;
        const on = highlight === null || highlight === i;
        return (
          <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={frac > 0.75 ? T.accent : frac > 0.45 ? T.violet : T.dim}
                strokeWidth={1 + frac * 5}
                opacity={on ? 0.95 : 0.25} strokeLinecap="round">
            <title>{`rung ${i + 1}: power ${fmt(p, 3)}`}</title>
          </line>
        );
      })}
      {nodes.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r="3.4" fill={T.text} opacity="0.85" />
      ))}
      {label && (
        <text x={cx} y={cy + 4} textAnchor="middle" fontSize="10"
              fill={T.text} fontFamily={MONO}>{label}</text>
      )}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Expectation vs measurement                                        */
/* ------------------------------------------------------------------ */

/**
 * The registered-expectation table.
 *
 * The papers record what was predicted BEFORE measuring, and two of those
 * predictions were refuted. Showing expectation beside measurement — and
 * marking a refutation as a refutation rather than dropping the row — is the
 * one thing a static figure of the final numbers cannot do.
 */
export function ExpectationTable({ expectation, measured }) {
  const keys = Object.keys(expectation ?? {});
  if (keys.length === 0) return null;

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10, fontFamily: MONO }}>
      <thead>
        <tr>
          {["registered expectation", "predicted", "measured", ""].map((h) => (
            <th key={h} style={{
              textAlign: h === "registered expectation" ? "left" : "center",
              padding: "4px 6px", color: T.dim, fontWeight: 500,
              borderBottom: `1px solid ${T.border}`, fontSize: 9,
              textTransform: "uppercase", letterSpacing: 0.5,
            }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => {
          const pred = expectation[k];
          const meas = measured?.[k];
          const comparable = typeof pred === "boolean" && typeof meas === "boolean";
          const held = comparable ? pred === meas : null;
          return (
            <tr key={k}>
              <td style={{ padding: "4px 6px", color: T.text, borderBottom: `1px solid ${T.panel}` }}>
                {k.replace(/_/g, " ")}
              </td>
              <td style={{ padding: "4px 6px", textAlign: "center", color: T.dim, borderBottom: `1px solid ${T.panel}` }}>
                {String(pred)}
              </td>
              <td style={{ padding: "4px 6px", textAlign: "center", color: T.text, borderBottom: `1px solid ${T.panel}` }}>
                {meas === undefined ? "—" : String(meas)}
              </td>
              <td style={{ padding: "4px 6px", textAlign: "center", borderBottom: `1px solid ${T.panel}` }}>
                {held === null ? (
                  <span style={{ color: T.muted }}>—</span>
                ) : held ? (
                  <span style={{ color: T.ok }}>held</span>
                ) : (
                  <span style={{ color: T.warn, fontWeight: 700 }}>REFUTED</span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Pass/fail checks as reported by a validator. */
export function ChecksList({ checks }) {
  if (!checks) return null;
  const entries = Array.isArray(checks)
    ? checks.map((c, i) => [String(i + 1), c])
    : Object.entries(checks);
  return (
    <div style={{ fontSize: 10, fontFamily: MONO }}>
      {entries.map(([k, v]) => {
        const passed = typeof v === "boolean" ? v : v?.pass ?? null;
        return (
          <div key={k} style={{
            display: "flex", justifyContent: "space-between",
            padding: "3px 0", borderBottom: `1px solid ${T.panel}`,
          }}>
            <span style={{ color: T.dim }}>{k.replace(/_/g, " ")}</span>
            <span style={{ color: passed === null ? T.text : passed ? T.ok : T.err }}>
              {passed === null ? String(v) : passed ? "pass" : "FAIL"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export { T as CHART_THEME, MONO as CHART_MONO };
