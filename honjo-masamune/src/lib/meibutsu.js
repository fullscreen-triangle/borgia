/**
 * Observation fields and comparison by superposition.
 *
 * This is a port of the Python reference implementation
 * (graphical-chemistry-generator/src/meibutsu.py). It is checked
 * against it by scripts/check-field.mjs, which compares coordinates,
 * visibilities and energies to 1e-9. If the two drift apart that
 * script fails.
 *
 * The reason this exists in JavaScript at all: the comparison is a
 * superposition of two fields, and the superposition is what gets
 * drawn. Computing it on a server and shipping a picture of the result
 * would make the displayed thing a report about a comparison rather
 * than the comparison itself.
 */

/** Highest fundamental in the reference set: H2 stretch, cm^-1. */
export const OMEGA_REF = 4401.0;

/** Reference scales for the temporal coordinate. */
const OMEGA_REF_MAX = 4401.0;
const OMEGA_REF_MIN = 218.0;
const B_ROT_REF_MIN = 0.39;

/** Rational-ratio tolerance for the harmonic coordinate. */
const HARMONIC_DELTA = 0.05;

/* ------------------------------------------------------------------ */
/*  Coordinates                                                       */
/* ------------------------------------------------------------------ */

/**
 * Knowledge coordinate: normalised Shannon entropy of the frequency
 * distribution. One dominant mode gives 0; equal modes give 1.
 */
export function knowledge(modes) {
  const n = modes.length;
  if (n === 0) return 0;
  if (n === 1) return modes[0] / OMEGA_REF_MAX;
  const total = modes.reduce((a, b) => a + b, 0);
  let h = 0;
  for (const w of modes) {
    const p = w / total;
    if (p > 0) h -= p * Math.log2(p);
  }
  return Math.min(h / Math.log2(n), 1.0);
}

/**
 * Temporal coordinate: how many decades of timescale the spectrum
 * spans, against the reference span.
 */
export function temporal(modes, bRot) {
  const n = modes.length;
  if (n === 1) {
    if (bRot && bRot > 0) {
      const ratio = modes[0] / bRot;
      if (ratio <= 1) return 0;
      const v = Math.log(ratio) / Math.log(OMEGA_REF_MAX / B_ROT_REF_MIN);
      return Math.min(Math.max(v, 0), 1);
    }
    return 0.1;
  }
  const hi = Math.max(...modes);
  const lo = Math.min(...modes);
  if (lo <= 0 || hi <= lo) return 0;
  const v = Math.log(hi / lo) / Math.log(OMEGA_REF_MAX / OMEGA_REF_MIN);
  return Math.min(Math.max(v, 0), 1);
}

function isHarmonicPair(a, b, delta = HARMONIC_DELTA) {
  if (a === 0 || b === 0) return false;
  const ratio = Math.max(a, b) / Math.min(a, b);
  for (let p = 1; p <= 8; p += 1) {
    for (let q = 1; q <= p; q += 1) {
      if (Math.abs(ratio - p / q) < delta) return true;
    }
  }
  return false;
}

/**
 * Evolution coordinate: the fraction of mode pairs whose frequency
 * ratio lands on a low-order rational. This is the molecule
 * interfering with itself.
 */
export function evolution(modes) {
  const n = modes.length;
  if (n < 2) return 0;
  const pairs = (n * (n - 1)) / 2;
  let harmonic = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (isHarmonicPair(modes[i], modes[j])) harmonic += 1;
    }
  }
  return harmonic / Math.max(pairs, 1);
}

export function coordinates(modes, bRot) {
  return [knowledge(modes), temporal(modes, bRot), evolution(modes)];
}

/* ------------------------------------------------------------------ */
/*  The observation field                                             */
/* ------------------------------------------------------------------ */

/**
 * Evaluate the field for one spectrum.
 *
 * Amplitude: one Gaussian lobe per mode at its normalised frequency
 * address, width set by the knowledge coordinate, under an envelope set
 * by the temporal coordinate.
 *
 * Phase: accumulates linearly with the modes and with the evolution
 * coordinate. This is the load-bearing part — amplitude alone cannot
 * distinguish structures whose mode positions coincide.
 */
export function observe(modes, { coords = null, bRot = null, grid = 256, name = "" } = {}) {
  const c = coords || coordinates(modes, bRot);
  const [sk, st, se] = c;
  const width = 0.05 * (1.0 - 0.5 * sk) + 1e-6;
  const env = 0.1 + 0.4 * st;

  const amp = new Float64Array(grid);
  const phase = new Float64Array(grid);

  for (let i = 0; i < grid; i += 1) {
    const u = grid === 1 ? 0 : i / (grid - 1);
    let a = 0;
    let ph = 2 * Math.PI * se * u * 8.0;
    for (const om of modes) {
      const mu = om / OMEGA_REF;
      const d = u - mu;
      a += Math.exp(-(d * d) / (width * width));
      ph += 2 * Math.PI * mu * u;
    }
    const e = u - 0.5;
    amp[i] = a * Math.exp(-(e * e) / (env * env));
    phase[i] = ph;
  }

  return { name, coords: c, modes: [...modes], amp, phase, grid };
}

/** Squared norm of the field. */
export function energy(f) {
  let s = 0;
  for (let i = 0; i < f.grid; i += 1) s += f.amp[i] * f.amp[i];
  return s;
}

/**
 * Fringe visibility: the normalised field correlation
 *
 *     V = |<A,B>| / sqrt(<A,A> <B,B>)
 *
 * Exactly 1 when the fields coincide, by Cauchy-Schwarz with equality.
 * This is used in preference to a pointwise average of
 * 1/2 + 1/2 cos(dphi), which attains only ~0.53 on self-comparison and
 * requires an unstated assumption about the phase distribution.
 */
export function visibility(a, b) {
  let re = 0;
  let im = 0;
  let ea = 0;
  let eb = 0;
  for (let i = 0; i < a.grid; i += 1) {
    const d = b.phase[i] - a.phase[i];
    const m = a.amp[i] * b.amp[i];
    re += m * Math.cos(d);
    im += m * Math.sin(d);
    ea += a.amp[i] * a.amp[i];
    eb += b.amp[i] * b.amp[i];
  }
  const den = Math.sqrt(ea * eb);
  if (den <= 0) return 0;
  return Math.hypot(re, im) / den;
}

/**
 * The relational part of the superposition, pointwise.
 *
 * |A+B|^2 = |A|^2 + |B|^2 + 2 Re(conj(A) B)
 *
 * Everything depending on both fields is this third term; the other
 * two would be present if the second structure were absent.
 */
export function crossTerm(a, b) {
  const out = new Float64Array(a.grid);
  for (let i = 0; i < a.grid; i += 1) {
    out[i] = 2 * a.amp[i] * b.amp[i] * Math.cos(b.phase[i] - a.phase[i]);
  }
  return out;
}

/** Superposed intensity |A + B|^2. */
export function superpose(a, b) {
  const out = new Float64Array(a.grid);
  for (let i = 0; i < a.grid; i += 1) {
    const d = b.phase[i] - a.phase[i];
    out[i] =
      a.amp[i] * a.amp[i] +
      b.amp[i] * b.amp[i] +
      2 * a.amp[i] * b.amp[i] * Math.cos(d);
  }
  return out;
}

/**
 * Render a field as a 2-D interference pattern.
 *
 * The horizontal axis is the frequency address; the vertical axis is a
 * phase offset applied to the second field. A horizontal slice at
 * offset 0 is the superposition as computed above; the other rows show
 * how the pattern moves as the relative phase is swept, which is what
 * makes the fringes visible as fringes rather than as a single curve.
 *
 * Writes straight into an RGBA buffer: this array is both what is
 * displayed and what the visibility is computed from.
 */
export function renderInterference(a, b, width, height) {
  const buf = new Uint8ClampedArray(width * height * 4);
  const rows = height;

  // Normalise against the largest intensity anywhere in the sweep, so
  // brightness is comparable between molecule pairs.
  let peak = 0;
  for (let r = 0; r < rows; r += 1) {
    const off = (r / Math.max(rows - 1, 1)) * 2 * Math.PI;
    for (let i = 0; i < a.grid; i += 1) {
      const d = b.phase[i] - a.phase[i] + off;
      const v =
        a.amp[i] * a.amp[i] +
        b.amp[i] * b.amp[i] +
        2 * a.amp[i] * b.amp[i] * Math.cos(d);
      if (v > peak) peak = v;
    }
  }
  if (peak <= 0) peak = 1;

  for (let r = 0; r < rows; r += 1) {
    const off = (r / Math.max(rows - 1, 1)) * 2 * Math.PI;
    for (let x = 0; x < width; x += 1) {
      // sample the field grid across the canvas width
      const gi = Math.min(
        a.grid - 1,
        Math.round((x / Math.max(width - 1, 1)) * (a.grid - 1))
      );
      const d = b.phase[gi] - a.phase[gi] + off;
      const own = a.amp[gi] * a.amp[gi] + b.amp[gi] * b.amp[gi];
      const cross = 2 * a.amp[gi] * b.amp[gi] * Math.cos(d);
      const t = Math.max(0, Math.min(1, (own + cross) / peak));

      // Colour carries the sign of the cross-term: constructive
      // interference warm, destructive cool. A single-channel ramp
      // would hide which of the two is happening.
      const k = (r * width + x) * 4;
      const sign = cross >= 0 ? 1 : -1;
      const mag = Math.sqrt(t);
      if (sign > 0) {
        buf[k] = 40 + 215 * mag;
        buf[k + 1] = 40 + 160 * mag;
        buf[k + 2] = 60 + 60 * mag;
      } else {
        buf[k] = 30 + 60 * mag;
        buf[k + 1] = 60 + 140 * mag;
        buf[k + 2] = 80 + 215 * mag;
      }
      buf[k + 3] = 255;
    }
  }
  return buf;
}

/** Draw a single field's amplitude and phase as a strip. */
export function renderField(f, width, height) {
  const buf = new Uint8ClampedArray(width * height * 4);
  let peak = 0;
  for (let i = 0; i < f.grid; i += 1) if (f.amp[i] > peak) peak = f.amp[i];
  if (peak <= 0) peak = 1;

  for (let x = 0; x < width; x += 1) {
    const gi = Math.min(
      f.grid - 1,
      Math.round((x / Math.max(width - 1, 1)) * (f.grid - 1))
    );
    const a = f.amp[gi] / peak;
    const ph = ((f.phase[gi] % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const hue = ph / (2 * Math.PI);
    const [r, g, bl] = hsv(hue, 0.55, 0.25 + 0.75 * a);
    for (let y = 0; y < height; y += 1) {
      // amplitude sets how far up the column is lit
      const lit = y / height >= 1 - a;
      const k = (y * width + x) * 4;
      buf[k] = lit ? r : 26;
      buf[k + 1] = lit ? g : 27;
      buf[k + 2] = lit ? bl : 38;
      buf[k + 3] = 255;
    }
  }
  return buf;
}

function hsv(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r;
  let g;
  let b;
  switch (i % 6) {
    case 0: [r, g, b] = [v, t, p]; break;
    case 1: [r, g, b] = [q, v, p]; break;
    case 2: [r, g, b] = [p, v, t]; break;
    case 3: [r, g, b] = [p, q, v]; break;
    case 4: [r, g, b] = [t, p, v]; break;
    default: [r, g, b] = [v, p, q];
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
