/**
 * D3Charts — chart marks built on d3 scales, axes and shape generators.
 *
 * The primitives in Charts.js compute pixel positions by hand. That is fine
 * for a bar and wrong for everything else: no nice tick values, no log axis
 * landing on decades, no shared band scale, and no hover without writing
 * hit-testing. d3 is already a dependency; these marks use it for what it is
 * good at, which is scales, axes and path generation.
 *
 * The pattern throughout is: d3 computes, React renders. No d3 selection
 * touches a live node and there is no enter/exit, so there is one source of
 * truth for what is on screen and no chance of the two fighting.
 *
 *   LinePlot    one or more series, curved or stepped, optional log y
 *   RingChart   a cycle drawn as a cycle (d3.arc), for closed ladders
 *   ShellChart  an atom's derived configuration as nested radial bands
 *   Histogram   d3.bin over a value set
 *   Sparkline   inline, axis-free, for a table cell
 *   BarChart    horizontal bars on a band scale, signed zero line
 */

import { useMemo, useState } from "react";
import {
  scaleLinear, scaleLog, scaleBand,
  line as d3line, curveMonotoneX, curveLinear, curveStepAfter,
  arc as d3arc, pie as d3pie,
  extent, max as d3max, bin as d3bin,
  format as d3format, schemeTableau10,
} from "d3";

export const T = {
  bg: "#1a1b26", surface: "#1e1f2e", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", muted: "#3b3d57", accent: "#7dcfff",
  ok: "#9ece6a", warn: "#e0af68", err: "#f7768e", violet: "#bb9af7",
  orange: "#ff9e64",
};
export const MONO = "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace";

export const SERIES = [
  T.accent, T.violet, T.ok, T.orange, T.warn, T.err,
  ...schemeTableau10.slice(0, 4),
];

const fmtNum = d3format(".4~g");

export function fmt(x, dp = 4) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "−∞";
  if (x !== 0 && Math.abs(x) < 1e-3) return x.toExponential(1);
  return x.toFixed(dp);
}

/* ------------------------------------------------------------------ */
/*  Shared frame                                                       */
/* ------------------------------------------------------------------ */

/**
 * Axes, gridlines and labels for a cartesian plot.
 *
 * Ticks come from the scale, so a log scale gets decades and a linear scale
 * gets round numbers — neither of which an evenly-spaced tick loop produces.
 */
function Frame({
  x, y, width, height, margin, xLabel, yLabel,
  xTickCount = 6, yTickCount = 5, children, xTickFormat, yTickFormat,
  bands = [],
}) {
  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;

  const xt = x.ticks ? x.ticks(xTickCount) : x.domain();
  const yt = y.ticks ? y.ticks(yTickCount) : y.domain();
  const xf = xTickFormat ?? (x.tickFormat ? x.tickFormat(xTickCount, "~g") : String);
  const yf = yTickFormat ?? (y.tickFormat ? y.tickFormat(yTickCount, "~g") : String);

  return (
    <g transform={`translate(${margin.left},${margin.top})`}>
      {/* Bands first: a shaded region is ground, not a mark, so gridlines and
          series must both read on top of it. */}
      {bands.map((b, i) => {
        const bx = Math.max(0, Math.min(iw, x(b.from)));
        const bw = Math.max(0, Math.min(iw, x(b.to)) - bx);
        if (!(bw > 0)) return null;
        return (
          <g key={`band${i}`}>
            <rect x={bx} y={0} width={bw} height={ih}
                  fill={b.colour ?? T.panel} opacity={b.opacity ?? 0.18} />
            {b.label && bw > 26 && (
              <text x={bx + bw / 2} y={11} textAnchor="middle" fontSize="8.5"
                    fill={b.colour ?? T.dim} fontFamily={MONO} opacity="0.9">
                {b.label}
              </text>
            )}
          </g>
        );
      })}
      {yt.map((v, i) => (
        <g key={`y${i}`} transform={`translate(0,${y(v)})`}>
          <line x2={iw} stroke={T.border} strokeWidth="1" />
          <text x={-8} dy="0.32em" textAnchor="end" fontSize="9.5"
                fill={T.dim} fontFamily={MONO}>{yf(v)}</text>
        </g>
      ))}
      {xt.map((v, i) => (
        <g key={`x${i}`} transform={`translate(${x(v)},${ih})`}>
          <line y2={4} stroke={T.dim} strokeWidth="1" />
          <text y={16} textAnchor="middle" fontSize="9.5"
                fill={T.dim} fontFamily={MONO}>{xf(v)}</text>
        </g>
      ))}
      <line y1={ih} y2={ih} x2={iw} stroke={T.dim} strokeWidth="1" />
      <line y2={ih} stroke={T.dim} strokeWidth="1" />
      {children}
      {xLabel && (
        <text x={iw / 2} y={ih + 30} textAnchor="middle" fontSize="9.5"
              fill={T.dim} fontFamily={MONO}>{xLabel}</text>
      )}
      {yLabel && (
        <text transform="rotate(-90)" x={-ih / 2} y={-margin.left + 12}
              textAnchor="middle" fontSize="9.5" fill={T.dim}
              fontFamily={MONO}>{yLabel}</text>
      )}
    </g>
  );
}

/* ------------------------------------------------------------------ */
/*  Line / scatter                                                     */
/* ------------------------------------------------------------------ */

/**
 * One or more series over a shared frame, with hover readout.
 *
 * logY switches the scale rather than pre-transforming the data, so ticks
 * land on decades and the tooltip reports the real value. A series whose
 * points are all equal still renders: the domain is padded rather than
 * collapsing to zero height.
 */
export function LinePlot({
  series, height = 200, xLabel, yLabel, logY = false,
  curve = "monotone", showPoints = true, yDomain, xDomain,
  bands = [], rules = [], marker = null,
}) {
  const [hover, setHover] = useState(null);
  const width = 480;
  const margin = { top: 12, right: 14, bottom: 38, left: 58 };
  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;

  const built = useMemo(() => {
    const all = (series || []).flatMap((s) => s.points || []);
    if (!all.length) return null;

    let [x0, x1] = extent(all, (p) => p.x);
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }

    let y0, y1;
    if (yDomain) { [y0, y1] = yDomain; }
    else {
      const ys = logY
        ? all.map((p) => p.y).filter((v) => v > 0)
        : all.map((p) => p.y).filter(Number.isFinite);
      [y0, y1] = extent(ys.length ? ys : [0, 1]);
      if (logY) { y0 /= 3; y1 *= 3; }
      else if (y0 === y1) { const d = Math.abs(y0) || 1; y0 -= d * 0.1; y1 += d * 0.1; }
      else { const p = (y1 - y0) * 0.08; y0 -= p; y1 += p; }
    }

    const xs = scaleLinear()
      .domain(xDomain ?? [x0, x1])
      .range([0, iw]);
    if (!xDomain) xs.nice();
    const ys = (logY ? scaleLog() : scaleLinear()).domain([y0, y1]).range([ih, 0]);
    if (!logY) ys.nice();

    const cv = curve === "step" ? curveStepAfter
      : curve === "linear" ? curveLinear : curveMonotoneX;
    const ok = (p) => Number.isFinite(p.y) && (!logY || p.y > 0);
    const gen = d3line().defined(ok).x((p) => xs(p.x)).y((p) => ys(p.y)).curve(cv);

    const paths = series.map((s, i) => ({
      d: gen([...(s.points || [])].sort((a, b) => a.x - b.x)),
      colour: s.colour ?? SERIES[i % SERIES.length],
      name: s.name,
    }));
    const pts = series.flatMap((s, i) =>
      (s.points || []).filter(ok).map((p) => ({
        cx: xs(p.x), cy: ys(p.y),
        colour: s.colour ?? SERIES[i % SERIES.length],
        label: p.label ?? `${s.name ?? ""} x=${fmtNum(p.x)} y=${fmtNum(p.y)}`,
      })));
    return { x: xs, y: ys, paths, pts };
  }, [series, iw, ih, logY, curve, yDomain, xDomain]);

  if (!built) return null;
  const { x, y, paths, pts } = built;

  return (
    <div style={{ position: "relative" }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}
           style={{ display: "block", overflow: "visible" }}>
        <Frame x={x} y={y} width={width} height={height} margin={margin}
               xLabel={xLabel} yLabel={yLabel} bands={bands}
               yTickFormat={logY ? d3format("~e") : undefined}>
          {paths.map((p, i) => (
            <path key={i} d={p.d} fill="none" stroke={p.colour}
                  strokeWidth="1.8" opacity="0.9"
                  strokeLinecap="round" strokeLinejoin="round" />
          ))}
          {/* Labelled horizontal thresholds. A rule is drawn over the series
              because it is the thing the series is being read against. */}
          {rules.map((r, i) => {
            const ry = y(r.y);
            if (!Number.isFinite(ry)) return null;
            const c = r.colour ?? T.warn;
            return (
              <g key={`rule${i}`}>
                <line x1={0} x2={width - margin.left - margin.right} y1={ry} y2={ry}
                      stroke={c} strokeWidth="1.2"
                      strokeDasharray={r.dash ?? "5 3"} opacity="0.85" />
                {r.label && (
                  <text x={width - margin.left - margin.right - 2} y={ry - 4}
                        textAnchor="end" fontSize="9" fill={c} fontFamily={MONO}>
                    {r.label}
                  </text>
                )}
              </g>
            );
          })}
          {/* Where the reader currently is on the x axis. */}
          {marker !== null && Number.isFinite(x(marker)) && (
            <line x1={x(marker)} x2={x(marker)} y1={0}
                  y2={height - margin.top - margin.bottom}
                  stroke={T.text} strokeWidth="1.4" opacity="0.75" />
          )}
          {showPoints && pts.map((p, i) => (
            <circle key={i} cx={p.cx} cy={p.cy}
                    r={hover === i ? 5 : 3.2} fill={p.colour}
                    stroke={T.bg} strokeWidth={hover === i ? 1.5 : 0}
                    opacity="0.95" style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover(i)}
                    onMouseLeave={() => setHover(null)} />
          ))}
        </Frame>
      </svg>
      {hover !== null && pts[hover] && (
        <div style={{
          position: "absolute", left: 0, top: 0, pointerEvents: "none",
          transform: `translate(${margin.left + pts[hover].cx + 10}px,${margin.top + pts[hover].cy - 26}px)`,
          background: T.panel, border: `1px solid ${T.border}`,
          borderRadius: 3, padding: "3px 7px", fontSize: 10,
          fontFamily: MONO, color: T.text, whiteSpace: "nowrap", zIndex: 5,
        }}>{pts[hover].label}</div>
      )}
      {series.length > 1 && (
        <div style={{
          display: "flex", gap: 12, flexWrap: "wrap", marginTop: 4,
          fontSize: 10, fontFamily: MONO, color: T.dim,
        }}>
          {paths.map((p, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 9, height: 2.5, background: p.colour, borderRadius: 1 }} />
              {p.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Ring — a cycle drawn as a cycle                                    */
/* ------------------------------------------------------------------ */

/**
 * A closed ladder as an annular chart, built with d3.arc and d3.pie.
 *
 * A bar chart of the same numbers would impose an order, and a cycle is
 * exactly the structure that has none: the invariants (circulation,
 * uniformity) are rotation invariant, so the mark should be too. Segment
 * angle is the equal share each rung occupies in the cycle; radial thickness
 * is its power.
 */
export function RingChart({ powers, labels, size = 190, title, onHover }) {
  const [hover, setHover] = useState(null);
  const n = (powers || []).length;

  const arcs = useMemo(() => {
    if (!n) return [];
    const pieGen = d3pie().value(() => 1).sort(null).padAngle(0.035);
    const mx = d3max(powers) || 1;
    const rOut = size * 0.44, rInMin = size * 0.20;
    // A floor keeps a near-zero rung visible as a hairline rather than
    // vanishing into the background.
    const r = scaleLinear().domain([0, mx]).range([size * 0.035, rOut - rInMin]);
    return pieGen(powers).map((d, i) => ({
      i,
      path: d3arc().innerRadius(rOut - r(powers[i])).outerRadius(rOut)
        .cornerRadius(2)(d),
      power: powers[i],
      label: labels?.[i] ?? `rung ${i + 1}`,
    }));
  }, [powers, labels, n, size]);

  if (!n) return null;
  const h = hover !== null ? arcs[hover] : null;

  return (
    <div style={{ width: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ display: "block", overflow: "visible" }}>
        <g transform={`translate(${size / 2},${size / 2})`}>
          <circle r={size * 0.20} fill="none" stroke={T.border}
                  strokeWidth="1" strokeDasharray="2 3" />
          {arcs.map((a) => (
            <path key={a.i} d={a.path} fill={SERIES[a.i % SERIES.length]}
                  opacity={hover === null || hover === a.i ? 0.92 : 0.28}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => { setHover(a.i); onHover?.(a.i); }}
                  onMouseLeave={() => { setHover(null); onHover?.(null); }} />
          ))}
          {h ? (
            <>
              <text textAnchor="middle" dy="-0.2em" fontSize="12"
                    fontFamily={MONO} fill={T.text}>{fmt(h.power, 4)}</text>
              <text textAnchor="middle" dy="1.1em" fontSize="8.5"
                    fontFamily={MONO} fill={T.dim}>{h.label}</text>
            </>
          ) : (
            <>
              <text textAnchor="middle" dy="-0.1em" fontSize="11"
                    fontFamily={MONO} fill={T.text}>{n}</text>
              <text textAnchor="middle" dy="1.2em" fontSize="8.5"
                    fontFamily={MONO} fill={T.dim}>
                rung{n === 1 ? "" : "s"}
              </text>
            </>
          )}
        </g>
      </svg>
      {title && (
        <div style={{
          fontSize: 9.5, color: T.dim, textAlign: "center", marginTop: 2,
          fontFamily: MONO,
        }}>{title}</div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Shells — an atom's configuration                                   */
/* ------------------------------------------------------------------ */

/**
 * The derived electron configuration as nested radial bands.
 *
 * cut Z derives the configuration rather than reading a table, and the shape
 * of that derivation — which subshells fill, in what order, how full the
 * valence shell is — is the thing worth seeing. Each ring is a principal
 * shell; each arc within it is a subshell, its angular span the subshell
 * capacity 2(2l+1) and its fill the occupancy.
 */
export function ShellChart({ config, size = 190, symbol }) {
  const [hover, setHover] = useState(null);

  const { rings, nMax } = useMemo(() => {
    const byN = new Map();
    for (const s of config || []) {
      const n = Number(s.n), l = Number(s.l), occ = Number(s.occ ?? s.e ?? 0);
      if (!Number.isFinite(n) || !Number.isFinite(l)) continue;
      if (!byN.has(n)) byN.set(n, []);
      byN.get(n).push({ n, l, occ });
    }
    if (!byN.size) return { rings: [], nMax: 0 };
    const nMax = Math.max(...byN.keys());
    const rScale = scaleLinear().domain([0.5, nMax + 0.5])
      .range([size * 0.11, size * 0.46]);
    const L = ["s", "p", "d", "f", "g"];

    const rings = [];
    for (const [n, subs] of [...byN.entries()].sort((a, b) => a[0] - b[0])) {
      subs.sort((a, b) => a.l - b.l);
      const caps = subs.map((s) => 2 * (2 * s.l + 1));
      const total = caps.reduce((a, b) => a + b, 0);
      let a0 = -Math.PI / 2;
      const rIn = rScale(n) - size * 0.028;
      const rOut = rScale(n) + size * 0.028;
      for (let i = 0; i < subs.length; i++) {
        const s = subs[i];
        const span = (caps[i] / total) * 2 * Math.PI * 0.94;
        const gap = (2 * Math.PI * 0.06) / subs.length;
        const full = Math.max(0, Math.min(1, s.occ / caps[i]));
        rings.push({
          key: `${n}${L[s.l] ?? s.l}`,
          n, l: s.l, occ: s.occ, cap: caps[i],
          bg: d3arc().innerRadius(rIn).outerRadius(rOut)
            .startAngle(a0).endAngle(a0 + span)(),
          fg: d3arc().innerRadius(rIn).outerRadius(rOut).cornerRadius(1)
            .startAngle(a0).endAngle(a0 + span * full)(),
        });
        a0 += span + gap;
      }
    }
    return { rings, nMax };
  }, [config, size]);

  if (!rings.length) return null;
  const h = hover !== null ? rings[hover] : null;

  return (
    <div style={{ width: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ display: "block", overflow: "visible" }}>
        <g transform={`translate(${size / 2},${size / 2})`}>
          {rings.map((r, i) => (
            <g key={r.key}>
              <path d={r.bg} fill={T.muted} opacity="0.45" />
              <path d={r.fg} fill={SERIES[r.l % SERIES.length]}
                    opacity={hover === null || hover === i ? 0.95 : 0.3}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover(i)}
                    onMouseLeave={() => setHover(null)} />
            </g>
          ))}
          {h ? (
            <>
              <text textAnchor="middle" dy="-0.2em" fontSize="12"
                    fontFamily={MONO} fill={T.text}>{h.key}</text>
              <text textAnchor="middle" dy="1.1em" fontSize="9"
                    fontFamily={MONO} fill={T.dim}>{h.occ}/{h.cap}</text>
            </>
          ) : (
            <text textAnchor="middle" dy="0.34em" fontSize="15"
                  fontFamily={MONO} fill={T.text} fontWeight="700">{symbol}</text>
          )}
        </g>
      </svg>
      <div style={{
        fontSize: 9.5, color: T.dim, textAlign: "center", marginTop: 2,
        fontFamily: MONO,
      }}>
        {nMax} shell{nMax === 1 ? "" : "s"} · derived, not tabulated
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Histogram                                                          */
/* ------------------------------------------------------------------ */

/** d3.bin over a value set, for distributions across many bindings. */
export function Histogram({ values, height = 150, xLabel, bins = 12 }) {
  const width = 480;
  const margin = { top: 10, right: 14, bottom: 36, left: 46 };
  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;

  const built = useMemo(() => {
    const v = (values || []).filter(Number.isFinite);
    if (v.length < 2) return null;
    const [lo, hi] = extent(v);
    const xs = scaleLinear().domain(lo === hi ? [lo - 1, hi + 1] : [lo, hi])
      .range([0, iw]).nice();
    const b = d3bin().domain(xs.domain()).thresholds(bins)(v);
    const ys = scaleLinear().domain([0, d3max(b, (d) => d.length) || 1])
      .range([ih, 0]).nice();
    return {
      x: xs, y: ys,
      bars: b.map((d, i) => ({
        x: xs(d.x0), w: Math.max(1, xs(d.x1) - xs(d.x0) - 1),
        y: ys(d.length), h: ih - ys(d.length), n: d.length, i,
      })),
    };
  }, [values, iw, ih, bins]);

  if (!built) return null;
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}
         style={{ display: "block", overflow: "visible" }}>
      <Frame x={built.x} y={built.y} width={width} height={height}
             margin={margin} xLabel={xLabel} yLabel="count" yTickCount={4}>
        {built.bars.map((b) => (
          <rect key={b.i} x={b.x} y={b.y} width={b.w} height={b.h}
                fill={T.accent} opacity="0.8" rx="1">
            <title>{b.n} value{b.n === 1 ? "" : "s"}</title>
          </rect>
        ))}
      </Frame>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Sparkline                                                          */
/* ------------------------------------------------------------------ */

/** Axis-free inline trace, sized for a table cell. */
export function Sparkline({ values, width = 74, height = 18, colour = T.accent }) {
  const d = useMemo(() => {
    const v = (values || []).filter(Number.isFinite);
    if (v.length < 2) return null;
    const [lo, hi] = extent(v);
    const x = scaleLinear().domain([0, v.length - 1]).range([1, width - 1]);
    const y = scaleLinear().domain(lo === hi ? [lo - 1, hi + 1] : [lo, hi])
      .range([height - 2, 2]);
    return d3line().x((_, i) => x(i)).y((t) => y(t)).curve(curveMonotoneX)(v);
  }, [values, width, height]);
  if (!d) return null;
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={d} fill="none" stroke={colour} strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Bars                                                               */
/* ------------------------------------------------------------------ */

/**
 * Horizontal bars on a d3 band scale, with a signed zero line.
 *
 * A band scale rather than fixed row heights, so the mark stays legible from
 * two rows to thirty, and a diverging domain so a negative value reads as
 * negative rather than as a short bar.
 */
export function BarChart({ rows, height, unit = "", dp = 4, colour }) {
  const width = 480;
  const margin = { top: 6, right: 62, bottom: 22, left: 136 };
  const rowH = 20;
  const list = rows || [];
  const h = height ?? margin.top + margin.bottom + Math.max(1, list.length) * rowH;
  const iw = width - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  const { x, y } = useMemo(() => {
    const vals = list.map((r) => r.value).filter(Number.isFinite);
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    return {
      x: scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).range([0, iw]).nice(),
      y: scaleBand().domain(list.map((r, i) => `${i}:${r.label}`))
        .range([0, ih]).padding(0.28),
    };
  }, [list, iw, ih]);

  if (!list.length) return null;
  const zero = x(0);
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${h}`}
         style={{ display: "block", overflow: "visible" }}>
      <g transform={`translate(${margin.left},${margin.top})`}>
        {x.ticks(5).map((v, i) => (
          <line key={i} x1={x(v)} x2={x(v)} y2={ih} stroke={T.border} strokeWidth="1" />
        ))}
        {list.map((r, i) => {
          const v = Number.isFinite(r.value) ? r.value : 0;
          const xa = x(Math.min(0, v)), xb = x(Math.max(0, v));
          const yy = y(`${i}:${r.label}`);
          return (
            <g key={`${i}:${r.label}`} transform={`translate(0,${yy})`}>
              <text x={-8} y={y.bandwidth() / 2} dy="0.32em" textAnchor="end"
                    fontSize="10" fill={T.dim} fontFamily={MONO}>
                {r.label.length > 23 ? r.label.slice(0, 22) + "…" : r.label}
                <title>{r.label}</title>
              </text>
              <rect x={xa} width={Math.max(1, xb - xa)} height={y.bandwidth()}
                    fill={r.colour ?? colour ?? (v < 0 ? T.err : T.accent)}
                    opacity="0.85" rx="1.5" />
              <text x={xb + 6} y={y.bandwidth() / 2} dy="0.32em" fontSize="10"
                    fill={T.text} fontFamily={MONO}>{fmt(v, dp)}{unit}</text>
            </g>
          );
        })}
        <line x1={zero} x2={zero} y2={ih} stroke={T.dim} strokeWidth="1.2" />
      </g>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Vertical bars on a shared band scale                               */
/* ------------------------------------------------------------------ */

/**
 * Effort per step, in plan order, with an optional horizontal rule.
 *
 * BarChart is horizontal and label-first, which suits a ranked list. This one
 * is vertical and order-first, because the sequence is the plan and the reader
 * is comparing each bar against a single scalar drawn across all of them --
 * the shadow price. A horizontal layout cannot draw that line without turning
 * it into a per-row annotation, which is exactly the wrong reading: the price
 * is one number, not one per step.
 *
 * A bar of zero height draws as a dashed baseline rather than nothing, the
 * same convention Attrition uses, because "allocated nothing" is a result and
 * an absent mark reads as missing data.
 */
export function StepBars({
  rows, height = 168, yLabel, rule = null, fmtV, unit = "",
}) {
  const width = 480;
  const margin = { top: 12, right: 14, bottom: 42, left: 52 };
  const iw = width - margin.left - margin.right;
  const ih = height - margin.top - margin.bottom;

  const built = useMemo(() => {
    const rs = rows || [];
    if (!rs.length) return null;
    const top = Math.max(
      d3max(rs, (r) => r.value) ?? 0,
      rule && Number.isFinite(rule.y) ? rule.y : 0,
      1e-9
    );
    const x = scaleBand().domain(rs.map((r) => r.label)).range([0, iw]).padding(0.28);
    const y = scaleLinear().domain([0, top]).range([ih, 0]).nice();
    return { x, y };
  }, [rows, iw, ih, rule]);

  if (!built) return null;
  const { x, y } = built;
  const bw = x.bandwidth();

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}
         style={{ display: "block", overflow: "visible" }}>
      <Frame x={x} y={y} width={width} height={height} margin={margin}
             yLabel={yLabel} xTickFormat={(v) => v}>
        {(rows || []).map((r) => {
          const zero = !(r.value > 0);
          const h = zero ? 0 : Math.max(1, ih - y(r.value));
          const c = r.colour ?? T.accent;
          return (
            <g key={r.label} transform={`translate(${x(r.label)},0)`}>
              <rect
                x={0}
                y={zero ? ih : ih - h}
                width={bw}
                height={zero ? 0 : h}
                fill={c}
                opacity="0.85"
                rx="2"
                style={{ transition: "height 180ms ease, y 180ms ease" }}
              />
              {zero && (
                <line x1={0} x2={bw} y1={ih} y2={ih}
                      stroke={c} strokeWidth="2" strokeDasharray="3 2" />
              )}
              <text x={bw / 2} y={(zero ? ih : ih - h) - 4} textAnchor="middle"
                    fontSize="9" fill={zero ? T.dim : T.text} fontFamily={MONO}>
                {fmtV ? fmtV(r.value) : fmt(r.value, 2)}{unit}
              </text>
            </g>
          );
        })}
        {rule && Number.isFinite(rule.y) && (
          <g>
            <line x1={0} x2={iw} y1={y(rule.y)} y2={y(rule.y)}
                  stroke={rule.colour ?? T.warn} strokeWidth="1.2"
                  strokeDasharray="5 3" opacity="0.9" />
            {rule.label && (
              <text x={iw - 2} y={y(rule.y) - 4} textAnchor="end" fontSize="9"
                    fill={rule.colour ?? T.warn} fontFamily={MONO}>
                {rule.label}
              </text>
            )}
          </g>
        )}
      </Frame>
    </svg>
  );
}
