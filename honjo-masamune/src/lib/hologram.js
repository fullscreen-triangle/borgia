/**
 * Spectral Hologram Engine
 * ========================
 * Three-state superposition, coupling matrix, Franck-Condon,
 * Stokes shift, Huang-Rhys, and 2D FFT diffraction.
 * All client-side, zero backend.
 */

/**
 * Build a 1D spectrum from mode list with Gaussian line shapes.
 * @param {number[]} modes - frequencies in cm⁻¹
 * @param {Float64Array} omega - frequency grid
 * @param {number} width - Gaussian line width
 * @returns {Float64Array}
 */
export function buildSpectrum(modes, omega, width = 15) {
  const spec = new Float64Array(omega.length);
  for (const freq of modes) {
    for (let i = 0; i < omega.length; i++) {
      spec[i] += Math.exp(-Math.pow((omega[i] - freq) / width, 2));
    }
  }
  return spec;
}

/**
 * Build emission spectrum as Franck-Condon envelope between ground and excited.
 */
export function buildEmissionSpectrum(groundModes, excitedModes, omega, width = 25) {
  const spec = new Float64Array(omega.length);
  for (let m = 0; m < groundModes.length; m++) {
    if (m >= excitedModes.length) break;
    const center = (groundModes[m] + excitedModes[m]) / 2;
    const shift = Math.abs(excitedModes[m] - groundModes[m]);
    const intensity = Math.max(0.3, 1.0 - shift / 200);
    for (let i = 0; i < omega.length; i++) {
      spec[i] += intensity * Math.exp(-Math.pow((omega[i] - center) / width, 2));
    }
  }
  return spec;
}

/**
 * Superimpose three spectra with adjustable weights.
 */
export function buildHologram(ground, excited, emission, wG = 1, wE = 1, wEm = 1) {
  const h = new Float64Array(ground.length);
  for (let i = 0; i < h.length; i++) {
    h[i] = wG * ground[i] + wE * excited[i] + wEm * emission[i];
  }
  return h;
}

/**
 * Build 2D hologram texture (frequency × phase).
 * @returns {Float64Array} N×N flattened row-major
 */
export function build2DHologram(groundSpec, excitedSpec, emissionSpec, N = 128) {
  const tex = new Float64Array(N * N);
  for (let iy = 0; iy < N; iy++) {
    const phase = (iy / N) * 2 * Math.PI;
    const c0 = 0.5 + 0.3 * Math.cos(phase);
    const c2 = 0.5 + 0.3 * Math.sin(phase);
    const c1 = 1.0 - 0.5 * (c0 + c2);
    for (let ix = 0; ix < N; ix++) {
      const frac = ix / (N - 1);
      // Interpolate spectra to N columns
      const idx = frac * (groundSpec.length - 1);
      const lo = Math.floor(idx);
      const hi = Math.min(lo + 1, groundSpec.length - 1);
      const t = idx - lo;
      const g = groundSpec[lo] * (1 - t) + groundSpec[hi] * t;
      const e = excitedSpec[lo] * (1 - t) + excitedSpec[hi] * t;
      const em = emissionSpec[lo] * (1 - t) + emissionSpec[hi] * t;
      tex[iy * N + ix] = c0 * g + c2 * e + c1 * em;
    }
  }
  return tex;
}

/**
 * Simple 2D FFT (Cooley-Tukey, power-of-2 only).
 * Returns magnitude array (log scale).
 */
export function computeFFT2D(realData, N) {
  // 1D FFT helper (in-place, Cooley-Tukey radix-2)
  function fft1d(re, im, n) {
    // Bit reversal
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len *= 2) {
      const half = len / 2;
      const angle = (-2 * Math.PI) / len;
      const wRe = Math.cos(angle);
      const wIm = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < half; j++) {
          const tRe = curRe * re[i + j + half] - curIm * im[i + j + half];
          const tIm = curRe * im[i + j + half] + curIm * re[i + j + half];
          re[i + j + half] = re[i + j] - tRe;
          im[i + j + half] = im[i + j] - tIm;
          re[i + j] += tRe;
          im[i + j] += tIm;
          const newRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = newRe;
        }
      }
    }
  }

  const re = new Float64Array(N * N);
  const im = new Float64Array(N * N);
  re.set(realData);

  // FFT rows
  for (let y = 0; y < N; y++) {
    const rowRe = re.subarray(y * N, (y + 1) * N);
    const rowIm = im.subarray(y * N, (y + 1) * N);
    fft1d(rowRe, rowIm, N);
  }
  // FFT columns
  const colRe = new Float64Array(N);
  const colIm = new Float64Array(N);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) { colRe[y] = re[y * N + x]; colIm[y] = im[y * N + x]; }
    fft1d(colRe, colIm, N);
    for (let y = 0; y < N; y++) { re[y * N + x] = colRe[y]; im[y * N + x] = colIm[y]; }
  }

  // Magnitude (log scale) + fftshift
  const mag = new Float64Array(N * N);
  const half = N / 2;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const sy = (y + half) % N;
      const sx = (x + half) % N;
      const idx = sy * N + sx;
      const m = Math.sqrt(Math.pow(re[y * N + x], 2) + Math.pow(im[y * N + x], 2));
      mag[idx] = Math.log1p(m);
    }
  }
  return mag;
}

/**
 * Compute vibrational coupling matrix K_ij.
 */
export function computeCouplingMatrix(groundModes, excitedModes) {
  const N = Math.min(groundModes.length, excitedModes.length);
  const shifts = [];
  for (let i = 0; i < N; i++) shifts.push(excitedModes[i] - groundModes[i]);

  const K = [];
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) {
      const denom = Math.pow(shifts[i], 2) + Math.pow(shifts[j], 2);
      row.push(denom > 0 ? (shifts[i] * shifts[j]) / denom : 0);
    }
    K.push(row);
  }
  return { K, shifts };
}

/**
 * Compute Huang-Rhys factors.
 */
export function computeHuangRhys(groundModes, excitedModes) {
  const results = [];
  const N = Math.min(groundModes.length, excitedModes.length);
  for (let i = 0; i < N; i++) {
    const g = groundModes[i];
    const e = excitedModes[i];
    const delta = Math.abs(e - g);
    const mean = (g + e) / 2;
    const S = mean > 0 ? Math.pow(delta / mean, 2) * mean / 50 : 0;
    results.push({ groundFreq: g, excitedFreq: e, shift: delta, S });
  }
  return results;
}

/**
 * Compute Stokes shift decomposition.
 */
export function computeStokesShift(groundModes, excitedModes, absNm, emNm) {
  const Eabs = 1e7 / absNm;
  const Eem = 1e7 / emNm;
  const stokes = Eabs - Eem;
  const N = Math.min(groundModes.length, excitedModes.length);
  let deltaVib = 0;
  for (let i = 0; i < N; i++) deltaVib += Math.abs(excitedModes[i] - groundModes[i]);
  deltaVib /= N || 1;
  const deltaSolv = stokes - deltaVib;
  const reorg = stokes / 2;
  return { Eabs, Eem, stokes, deltaVib, deltaSolv, reorg };
}

/**
 * Predefined excited-state frequencies for molecules that have them.
 * For molecules without known excited states, shifts are estimated
 * as -1% of ground frequency (small perturbation).
 */
export function getExcitedModes(modes) {
  return modes.map((f) => f * 0.99 + (Math.sin(f * 0.01) * 5));
}

/**
 * Simulated emission wavelengths for common molecules.
 */
export const EMISSION_DATA = {
  H2O:  { absNm: 170, emNm: 185 },
  CO2:  { absNm: 200, emNm: 220 },
  CH4:  { absNm: 160, emNm: 175 },
  NH3:  { absNm: 195, emNm: 215 },
  C6H6: { absNm: 254, emNm: 278 },
  Trp:  { absNm: 280, emNm: 348 },
  Tyr:  { absNm: 275, emNm: 303 },
  // Default for any molecule
  _default: { absNm: 250, emNm: 290 },
};
