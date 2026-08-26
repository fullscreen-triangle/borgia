/**
 * Interference panel.
 *
 * The canvas here is not a picture of a comparison that happened
 * elsewhere. The pixel buffer and the visibility are computed from the
 * same two field arrays, in the browser, by the same functions the
 * reference implementation uses (checked to 1e-13 by
 * scripts/check-field.mjs).
 *
 * That is the whole point of the panel: if the displayed array and the
 * compared array are the same object, then rendering is not a stage
 * after comparison. The measurement supporting this is that quantising
 * the field to 8 bits per channel — an ordinary framebuffer — changes
 * the inversion accuracy not at all.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  observe,
  visibility,
  crossTerm,
  superpose,
  energy,
  renderInterference,
  renderField,
  OMEGA_REF,
} from "@/lib/meibutsu";
import SPECTRA_DATA from "@/data/spectra.json";

const T = {
  bg: "#1a1b26", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", muted: "#3b3d57",
  accent: "#7dcfff", ok: "#9ece6a", warn: "#e0af68", err: "#f7768e",
};

/**
 * Reference spectra, generated from the same NIST-derived database the
 * validation used (src/data/spectra.json). They are not transcribed:
 * a first attempt to type them by hand got three of twelve wrong.
 */
const SPECTRA = Object.fromEntries(
  Object.entries(SPECTRA_DATA.spectra).map(([k, v]) => [k, v.modes])
);
const FORMULA = Object.fromEntries(
  Object.entries(SPECTRA_DATA.spectra).map(([k, v]) => [k, v.formula])
);
const B_ROT = Object.fromEntries(
  Object.entries(SPECTRA_DATA.spectra).map(([k, v]) => [k, v.b_rot])
);

export default function InterferencePanel({ width = 380 }) {
  const [a, setA] = useState("H2O");
  const [b, setB] = useState("CO2");
  const [bits, setBits] = useState(0); // 0 = full precision

  const canvasRef = useRef(null);
  const fieldARef = useRef(null);
  const fieldBRef = useRef(null);

  // b_rot matters for diatomics: with a single mode the temporal
  // coordinate is derived from the rotational constant instead of a
  // frequency span, and omitting it silently changes the field.
  const fa = useMemo(
    () => observe(SPECTRA[a], { bRot: B_ROT[a], grid: 256, name: a }),
    [a]
  );
  const fb = useMemo(
    () => observe(SPECTRA[b], { bRot: B_ROT[b], grid: 256, name: b }),
    [b]
  );

  /**
   * Quantise a field to `n` bits per channel, as a framebuffer would.
   * Applied to the same arrays the canvas draws, so the number reported
   * is the number the picture was made from.
   */
  const quantised = useMemo(() => {
    if (!bits) return { fa, fb };
    const q = (f) => {
      const levels = (1 << bits) - 1;
      let peak = 0;
      for (let i = 0; i < f.grid; i += 1) if (f.amp[i] > peak) peak = f.amp[i];
      if (peak <= 0) peak = 1;
      const amp = new Float64Array(f.grid);
      const phase = new Float64Array(f.grid);
      const TAU = 2 * Math.PI;
      for (let i = 0; i < f.grid; i += 1) {
        amp[i] = (Math.round((f.amp[i] / peak) * levels) / levels) * peak;
        const ph = ((f.phase[i] % TAU) + TAU) % TAU;
        phase[i] = (Math.round((ph / TAU) * levels) / levels) * TAU;
      }
      return { ...f, amp, phase };
    };
    return { fa: q(fa), fb: q(fb) };
  }, [fa, fb, bits]);

  const V = useMemo(
    () => visibility(quantised.fa, quantised.fb),
    [quantised]
  );
  const Vself = useMemo(
    () => visibility(quantised.fa, quantised.fa),
    [quantised]
  );

  /** Total relational energy: the part that needs both fields. */
  const relational = useMemo(() => {
    const c = crossTerm(quantised.fa, quantised.fb);
    let s = 0;
    for (let i = 0; i < c.length; i += 1) s += c[i];
    return s;
  }, [quantised]);

  const own = useMemo(
    () => energy(quantised.fa) + energy(quantised.fb),
    [quantised]
  );

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const w = width - 28;
    const h = 150;
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext("2d");
    const buf = renderInterference(quantised.fa, quantised.fb, w, h);
    ctx.putImageData(new ImageData(buf, w, h), 0, 0);
  }, [quantised, width]);

  useEffect(() => {
    [fieldARef, fieldBRef].forEach((ref, i) => {
      const cv = ref.current;
      if (!cv) return;
      const w = width - 28;
      const h = 34;
      cv.width = w;
      cv.height = h;
      const ctx = cv.getContext("2d");
      const f = i === 0 ? quantised.fa : quantised.fb;
      ctx.putImageData(new ImageData(renderField(f, w, h), w, h), 0, 0);
    });
  }, [quantised, width]);

  const same = a === b;

  return (
    <div style={{ padding: 14 }}>
      <div style={{
        fontSize: 10, letterSpacing: 1, textTransform: "uppercase",
        color: T.dim, marginBottom: 4,
      }}>
        Interference
      </div>
      <div style={{ fontSize: 11.5, color: T.text, marginBottom: 12, lineHeight: 1.5 }}>
        Two structures are compared by adding their fields. The canvas below
        is the superposition, not a chart of one.
      </div>

      {/* selectors */}
      <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
        <Picker label="A" value={a} onChange={setA} />
        <Picker label="B" value={b} onChange={setB} />
      </div>

      {/* the two fields */}
      <Label>field A — amplitude as height, phase as hue</Label>
      <canvas ref={fieldARef} style={cvStyle} />
      <Label>field B</Label>
      <canvas ref={fieldBRef} style={cvStyle} />

      {/* the superposition */}
      <Label>
        |A + B|² — horizontal is frequency address, vertical sweeps the
        relative phase through 2π
      </Label>
      <canvas ref={canvasRef} style={{ ...cvStyle, height: 150 }} />
      <div style={{
        display: "flex", gap: 12, fontSize: 9.5, color: T.dim,
        marginTop: 2, marginBottom: 10,
      }}>
        <Swatch c="#e8a83c" label="constructive" />
        <Swatch c="#4a8ce0" label="destructive" />
      </div>

      {/* numbers, computed from the same arrays */}
      <Row k="visibility V(A,B)" v={V.toFixed(6)}
           tone={same ? T.ok : undefined} />
      <Row k="V(A,A)" v={Vself.toFixed(6)}
           tone={Math.abs(Vself - 1) < 1e-12 ? T.ok : T.err} />
      <Row k="own energy |A|²+|B|²" v={own.toFixed(3)} />
      <Row k="relational Σ cross-term" v={relational.toFixed(3)} />

      {same && (
        <div style={{
          fontSize: 10.5, color: T.ok, marginTop: 8, lineHeight: 1.6,
          borderLeft: `2px solid ${T.ok}`, paddingLeft: 8,
        }}>
          The same structure against itself: V = 1 exactly, by
          Cauchy–Schwarz with equality. Not approximately, and with no
          assumption about how the phase is distributed.
        </div>
      )}

      {/* display precision */}
      <div style={{ marginTop: 14, paddingTop: 12, borderTop: `1px solid ${T.border}` }}>
        <Label>display precision</Label>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
          {[0, 2, 4, 8, 16].map((n) => (
            <button
              key={n}
              onClick={() => setBits(n)}
              style={{
                padding: "3px 9px", fontSize: 10.5, fontFamily: "inherit",
                background: bits === n ? T.accent : "transparent",
                color: bits === n ? T.bg : T.dim,
                border: `1px solid ${bits === n ? T.accent : T.border}`,
                borderRadius: 3, cursor: "pointer",
              }}
            >
              {n === 0 ? "full" : `${n}-bit`}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: T.muted, lineHeight: 1.6 }}>
          Quantising to 8 bits per channel is what a framebuffer stores.
          Measured over the reference set, inversion accuracy at 8 bits
          equals accuracy at 16 bits exactly — so the array that is
          displayed can be the array that is compared.
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function Picker({ label, value, onChange }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: 9.5, color: T.dim, marginBottom: 3 }}>{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%", padding: "4px 6px", fontSize: 11,
          background: T.bg, color: T.text, border: `1px solid ${T.border}`,
          borderRadius: 3, fontFamily: "inherit", outline: "none",
        }}
      >
        {Object.keys(SPECTRA).map((k) => (
          <option key={k} value={k}>
            {FORMULA[k] || k} · {SPECTRA[k].length} mode
            {SPECTRA[k].length > 1 ? "s" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}

function Label({ children }) {
  return (
    <div style={{ fontSize: 9.5, color: T.dim, margin: "8px 0 3px", lineHeight: 1.4 }}>
      {children}
    </div>
  );
}

function Row({ k, v, tone }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      padding: "3px 0", borderBottom: `1px solid ${T.border}`, fontSize: 11,
    }}>
      <span style={{ color: T.dim }}>{k}</span>
      <span style={{ color: tone || T.text }}>{v}</span>
    </div>
  );
}

function Swatch({ c, label }) {
  return (
    <span>
      <span style={{
        display: "inline-block", width: 8, height: 8, background: c,
        borderRadius: 2, marginRight: 4, verticalAlign: "middle",
      }} />
      {label}
    </span>
  );
}

const cvStyle = {
  width: "100%", display: "block", borderRadius: 3,
  border: `1px solid ${T.border}`, imageRendering: "pixelated",
};
