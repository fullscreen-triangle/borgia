/**
 * S-Entropy Encoding Engine
 * =========================
 * Client-side computation of (S_k, S_t, S_e) from vibrational frequencies.
 * No backend, no API — the measurement IS the computation.
 */

// Reference values from NIST CCCBDB
const OMEGA_REF_MAX = 4401.0; // H₂ stretch cm⁻¹
const OMEGA_REF_MIN = 218.0;  // CCl₄ lowest mode
const B_ROT_REF_MIN = 0.24;   // Cl₂
const DELTA_MAX = 0.05;
const Q_MAX = 5;

/**
 * Compute S-entropy coordinates from vibrational frequencies.
 * @param {number[]} modes - Vibrational frequencies in cm⁻¹
 * @param {number|null} B_rot - Rotational constant for diatomics
 * @returns {{S_k: number, S_t: number, S_e: number}}
 */
export function computeSEntropy(modes, B_rot = null) {
  const N = modes.length;
  if (N === 0) return { S_k: 0, S_t: 0, S_e: 0 };

  // S_k: Knowledge entropy (Shannon of normalised distribution)
  const total = modes.reduce((s, w) => s + w, 0);
  const p = modes.map((w) => w / total);
  let S_k;
  if (N > 1) {
    const entropy = -p.reduce((s, pi) => s + (pi > 0 ? pi * Math.log(pi) : 0), 0);
    S_k = entropy / Math.log(N);
  } else {
    S_k = 1.0;
  }

  // S_t: Temporal entropy (timescale span)
  let S_t;
  if (N >= 2) {
    const ratio = Math.max(...modes) / Math.min(...modes);
    const refRatio = OMEGA_REF_MAX / OMEGA_REF_MIN;
    S_t = Math.max(0, Math.min(1, Math.log(ratio) / Math.log(refRatio)));
  } else if (B_rot && B_rot > 0) {
    const ratio = modes[0] / B_rot;
    const refRatio = OMEGA_REF_MAX / B_ROT_REF_MIN;
    S_t = Math.max(0, Math.min(1, Math.log(ratio) / Math.log(refRatio)));
  } else {
    S_t = 0.5;
  }

  // S_e: Evolution entropy (harmonic edge density)
  let S_e = 0;
  if (N >= 2) {
    let edges = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const r = modes[i] / modes[j];
        let found = false;
        for (let pp = 1; pp <= Q_MAX && !found; pp++) {
          for (let qq = 1; qq <= Q_MAX && !found; qq++) {
            if (Math.abs(r - pp / qq) < DELTA_MAX) {
              edges++;
              found = true;
            }
          }
        }
      }
    }
    const maxEdges = (N * (N - 1)) / 2;
    S_e = maxEdges > 0 ? edges / maxEdges : 0;
  }

  return { S_k, S_t, S_e };
}

/**
 * Generate interleaved ternary string from S-entropy coordinates.
 * @param {number} S_k
 * @param {number} S_t
 * @param {number} S_e
 * @param {number} depth - Number of trits
 * @returns {string}
 */
export function toTernaryString(S_k, S_t, S_e, depth = 18) {
  const vals = [S_k, S_t, S_e];
  const lo = [0, 0, 0];
  const hi = [1, 1, 1];
  const trits = [];

  for (let i = 0; i < depth; i++) {
    const dim = i % 3;
    const third = (hi[dim] - lo[dim]) / 3;
    const v = vals[dim];
    if (v < lo[dim] + third) {
      trits.push(0);
      hi[dim] = lo[dim] + third;
    } else if (v < lo[dim] + 2 * third) {
      trits.push(1);
      lo[dim] += third;
      hi[dim] = lo[dim] + third;
    } else {
      trits.push(2);
      lo[dim] += 2 * third;
    }
  }
  return trits.join("");
}

/**
 * Shared prefix depth between two ternary strings.
 */
export function sharedPrefixDepth(t1, t2) {
  let d = 0;
  const len = Math.min(t1.length, t2.length);
  for (let i = 0; i < len; i++) {
    if (t1[i] === t2[i]) d++;
    else break;
  }
  return d;
}

/**
 * Compute harmonic edges for a set of modes.
 * @returns {{edges: Array<{i: number, j: number, p: number, q: number, deviation: number}>, density: number}}
 */
export function computeHarmonicNetwork(modes) {
  const N = modes.length;
  const edges = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const r = modes[i] / modes[j];
      let bestP = 1, bestQ = 1, bestDev = Infinity;
      for (let p = 1; p <= Q_MAX; p++) {
        for (let q = 1; q <= Q_MAX; q++) {
          const dev = Math.abs(r - p / q);
          if (dev < bestDev) {
            bestDev = dev;
            bestP = p;
            bestQ = q;
          }
        }
      }
      if (bestDev < DELTA_MAX) {
        edges.push({ i, j, p: bestP, q: bestQ, deviation: bestDev });
      }
    }
  }
  const maxEdges = (N * (N - 1)) / 2;
  return { edges, density: maxEdges > 0 ? edges.length / maxEdges : 0 };
}

/**
 * Detect closed loops (triangles) in harmonic network.
 * Each loop = virtual resonant cavity.
 * @returns {Array<{nodes: number[], modes: number[], circFreq: number, meanDev: number, Q: number}>}
 */
export function detectLoops(modes, edges) {
  const N = modes.length;
  const adj = {};
  const edgeMap = {};
  edges.forEach((e) => {
    if (!adj[e.i]) adj[e.i] = [];
    if (!adj[e.j]) adj[e.j] = [];
    adj[e.i].push(e.j);
    adj[e.j].push(e.i);
    const key = `${Math.min(e.i, e.j)}-${Math.max(e.i, e.j)}`;
    edgeMap[key] = e;
  });

  const loops = [];
  // Find triangles
  for (let a = 0; a < N; a++) {
    for (const b of (adj[a] || [])) {
      if (b <= a) continue;
      for (const c of (adj[b] || [])) {
        if (c <= b) continue;
        if ((adj[a] || []).includes(c)) {
          const loopModes = [modes[a], modes[b], modes[c]];
          // Circulation frequency: 1 / sum of "emission lifetimes" (proxy: 1/ω)
          const tauSum = loopModes.reduce((s, w) => s + 1 / (w * 3e10), 0); // cm⁻¹ → Hz → period
          const circFreq = 1 / tauSum;
          // Mean harmonic deviation across loop edges
          const eAB = edgeMap[`${Math.min(a, b)}-${Math.max(a, b)}`];
          const eBC = edgeMap[`${Math.min(b, c)}-${Math.max(b, c)}`];
          const eAC = edgeMap[`${Math.min(a, c)}-${Math.max(a, c)}`];
          const devs = [eAB, eBC, eAC].filter(Boolean).map((e) => e.deviation);
          const meanDev = devs.length > 0 ? devs.reduce((s, d) => s + d, 0) / devs.length : 0;
          // Q-factor: inverse of mean deviation (sharper resonance = higher Q)
          const Q = meanDev > 0 ? 1 / meanDev : 1000;
          loops.push({ nodes: [a, b, c], modes: loopModes, circFreq, meanDev, Q });
        }
      }
    }
  }
  // Also find 4-cycles (squares)
  for (let a = 0; a < N; a++) {
    for (const b of (adj[a] || [])) {
      if (b <= a) continue;
      for (const c of (adj[b] || [])) {
        if (c <= a || c === a) continue;
        for (const d of (adj[c] || [])) {
          if (d <= a || d === b || d === c) continue;
          if ((adj[a] || []).includes(d) && !(adj[a] || []).includes(c)) {
            const loopModes = [modes[a], modes[b], modes[c], modes[d]];
            const tauSum = loopModes.reduce((s, w) => s + 1 / (w * 3e10), 0);
            const circFreq = 1 / tauSum;
            const relevantEdges = [
              edgeMap[`${Math.min(a, b)}-${Math.max(a, b)}`],
              edgeMap[`${Math.min(b, c)}-${Math.max(b, c)}`],
              edgeMap[`${Math.min(c, d)}-${Math.max(c, d)}`],
              edgeMap[`${Math.min(a, d)}-${Math.max(a, d)}`],
            ].filter(Boolean);
            const devs = relevantEdges.map((e) => e.deviation);
            const meanDev = devs.length > 0 ? devs.reduce((s, d) => s + d, 0) / devs.length : 0;
            const Q = meanDev > 0 ? 1 / meanDev : 1000;
            loops.push({ nodes: [a, b, c, d], modes: loopModes, circFreq, meanDev, Q });
          }
        }
      }
    }
  }
  return loops;
}

/**
 * Compute loop (cavity) fingerprint for a molecule.
 * @returns {{nLoops: number, meanQ: number, meanCircFreq: number, meanSize: number, loopFreqs: number[]}}
 */
export function computeLoopFingerprint(modes, edges) {
  const loops = detectLoops(modes, edges);
  if (loops.length === 0) {
    return { nLoops: 0, meanQ: 0, meanCircFreq: 0, meanSize: 0, loopFreqs: [], loops: [] };
  }
  const meanQ = loops.reduce((s, l) => s + l.Q, 0) / loops.length;
  const meanCircFreq = loops.reduce((s, l) => s + l.circFreq, 0) / loops.length;
  const meanSize = loops.reduce((s, l) => s + l.nodes.length, 0) / loops.length;
  const loopFreqs = loops.map((l) => l.circFreq).sort((a, b) => a - b);
  return { nLoops: loops.length, meanQ, meanCircFreq, meanSize, loopFreqs, loops };
}

/**
 * Loop fingerprint distance between two molecules.
 */
export function loopFingerprintDistance(fp1, fp2) {
  if (fp1.nLoops === 0 && fp2.nLoops === 0) return 0;
  if (fp1.nLoops === 0 || fp2.nLoops === 0) return 1;

  const wN = 0.3, wQ = 0.3, wF = 0.2, wS = 0.2;
  const maxLoops = Math.max(fp1.nLoops, fp2.nLoops, 1);
  const maxQ = Math.max(fp1.meanQ, fp2.meanQ, 1);
  const maxF = Math.max(fp1.meanCircFreq, fp2.meanCircFreq, 1);

  const dN = Math.abs(fp1.nLoops - fp2.nLoops) / maxLoops;
  const dQ = Math.abs(fp1.meanQ - fp2.meanQ) / maxQ;
  const dF = Math.abs(fp1.meanCircFreq - fp2.meanCircFreq) / maxF;
  const dS = Math.abs(fp1.meanSize - fp2.meanSize) / Math.max(fp1.meanSize, fp2.meanSize, 1);

  return Math.sqrt(wN * dN * dN + wQ * dQ * dQ + wF * dF * dF + wS * dS * dS);
}

/**
 * Compute shape entropy S_g from mesh normals (Gauss map entropy).
 * @param {Array<{x: number, y: number, z: number}>} normals - vertex normals
 * @returns {number} S_g in [0, 1]
 */
export function computeShapeEntropy(normals) {
  if (!normals || normals.length === 0) return 0;

  // Bin normals on unit sphere using icosahedral bins (20 bins)
  const nBins = 20;
  const bins = new Float32Array(nBins);

  // Simple binning: use theta-phi grid
  for (const n of normals) {
    const len = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len < 1e-8) continue;
    const nx = n.x / len, ny = n.y / len, nz = n.z / len;
    const theta = Math.acos(Math.max(-1, Math.min(1, nz)));
    const phi = Math.atan2(ny, nx) + Math.PI;
    const tBin = Math.min(Math.floor(theta / Math.PI * 4), 3);
    const pBin = Math.min(Math.floor(phi / (2 * Math.PI) * 5), 4);
    bins[tBin * 5 + pBin]++;
  }

  // Shannon entropy
  const total = bins.reduce((s, b) => s + b, 0);
  if (total === 0) return 0;
  let entropy = 0;
  for (let i = 0; i < nBins; i++) {
    const p = bins[i] / total;
    if (p > 0) entropy -= p * Math.log(p);
  }
  return entropy / Math.log(nBins); // normalise to [0,1]
}

/**
 * Compute ZPVE in kJ/mol from modes in cm⁻¹.
 */
export function computeZPVE(modes) {
  const h = 6.62607015e-34;
  const c = 2.99792458e8;
  const Na = 6.02214076e23;
  const cmToJ = h * c * 100;
  const zpveJ = 0.5 * modes.reduce((s, w) => s + cmToJ * w, 0);
  return (zpveJ * Na) / 1000;
}

/**
 * IDW property prediction from K nearest neighbours in S-space.
 */
export function predictProperty(queryCoords, refCompounds, propertyFn, K = 5) {
  const distances = refCompounds.map((ref) => {
    const d = Math.sqrt(
      (queryCoords.S_k - ref.S_k) ** 2 +
      (queryCoords.S_t - ref.S_t) ** 2 +
      (queryCoords.S_e - ref.S_e) ** 2
    );
    return { ...ref, dist: Math.max(d, 1e-12) };
  });
  distances.sort((a, b) => a.dist - b.dist);
  const nearest = distances.slice(0, K);
  const weights = nearest.map((n) => 1 / n.dist ** 2);
  const wTotal = weights.reduce((s, w) => s + w, 0);
  const values = nearest.map((n) => propertyFn(n));
  return weights.reduce((s, w, i) => s + (w * values[i]) / wTotal, 0);
}

/**
 * NIST CCCBDB compound database (39 compounds).
 */
export const COMPOUNDS = {
  H2:    { name: "H₂",    modes: [4401], B_rot: 59.3, type: "diatomic" },
  D2:    { name: "D₂",    modes: [2994], B_rot: 29.9, type: "diatomic" },
  N2:    { name: "N₂",    modes: [2330], B_rot: 1.99, type: "diatomic" },
  O2:    { name: "O₂",    modes: [1580], B_rot: 1.44, type: "diatomic" },
  F2:    { name: "F₂",    modes: [892],  B_rot: 0.89, type: "diatomic" },
  Cl2:   { name: "Cl₂",   modes: [560],  B_rot: 0.24, type: "diatomic" },
  CO:    { name: "CO",    modes: [2143], B_rot: 1.93, type: "diatomic" },
  NO:    { name: "NO",    modes: [1876], B_rot: 1.67, type: "diatomic" },
  HF:    { name: "HF",    modes: [3958], B_rot: 20.6, type: "diatomic" },
  HCl:   { name: "HCl",   modes: [2886], B_rot: 10.4, type: "diatomic" },
  HBr:   { name: "HBr",   modes: [2559], B_rot: 8.47, type: "diatomic" },
  HI:    { name: "HI",    modes: [2230], B_rot: 6.43, type: "diatomic" },
  H2O:   { name: "H₂O",   modes: [1595, 3657, 3756], type: "triatomic" },
  CO2:   { name: "CO₂",   modes: [667, 1388, 2349], type: "triatomic" },
  SO2:   { name: "SO₂",   modes: [518, 1151, 1362], type: "triatomic" },
  NO2:   { name: "NO₂",   modes: [750, 1318, 1617], type: "triatomic" },
  O3:    { name: "O₃",    modes: [701, 1042, 1103], type: "triatomic" },
  H2S:   { name: "H₂S",   modes: [1183, 2615, 2626], type: "triatomic" },
  HCN:   { name: "HCN",   modes: [712, 2097, 3312], type: "triatomic" },
  N2O:   { name: "N₂O",   modes: [589, 1285, 2224], type: "triatomic" },
  CS2:   { name: "CS₂",   modes: [397, 657, 1535], type: "triatomic" },
  OCS:   { name: "OCS",   modes: [520, 859, 2062], type: "triatomic" },
  NH3:   { name: "NH₃",   modes: [950, 1627, 3337, 3444], type: "tetra" },
  H2CO:  { name: "H₂CO",  modes: [1167, 1249, 1500, 1746, 2783, 2843], type: "poly" },
  PH3:   { name: "PH₃",   modes: [992, 1118, 2323, 2328], type: "tetra" },
  CH4:   { name: "CH₄",   modes: [1306, 1534, 2917, 3019], type: "tetra" },
  CCl4:  { name: "CCl₄",  modes: [218, 314, 762, 790], type: "tetra" },
  SiH4:  { name: "SiH₄",  modes: [800, 914, 2187, 2191], type: "tetra" },
  CF4:   { name: "CF₄",   modes: [435, 632, 909, 1283], type: "tetra" },
  C2H2:  { name: "C₂H₂",  modes: [612, 729, 1974, 3289, 3374], type: "poly" },
  C2H4:  { name: "C₂H₄",  modes: [826,943,949,1023,1236,1342,1444,1623,3026,3083,3103,3106], type: "poly" },
  C2H6:  { name: "C₂H₆",  modes: [822,995,1190,1370,1388,1468,2896,2954,2969,2985], type: "poly" },
  CH3OH: { name: "CH₃OH", modes: [1033,1060,1165,1345,1455,1477,2844,2960,3000,3681], type: "poly" },
  C6H6:  { name: "C₆H₆",  modes: [673, 993, 1178, 1596, 3062], type: "poly" },
  CH3F:  { name: "CH₃F",  modes: [1049,1182,1459,1467,2930,2965,3006], type: "poly" },
  CH3Cl: { name: "CH₃Cl", modes: [732,1017,1355,1452,2937,2967,3039], type: "poly" },
  CH3Br: { name: "CH₃Br", modes: [611,955,1306,1443,2935,2972,3056], type: "poly" },
  HCOOH: { name: "HCOOH", modes: [625,1033,1105,1229,1387,1770,2943,3570], type: "poly" },
  N2O4:  { name: "N₂O₄",  modes: [265,431,672,752,812,1261,1382,1712,1758], type: "poly" },

  // Amino acids (characteristic IR frequencies from NIST/literature)
  Arg:   { name: "Arginine",    modes: [1050,1175,1325,1410,1560,1610,1680,2870,2930,3100,3340], type: "amino_acid" },
  Asn:   { name: "Asparagine",  modes: [1015,1115,1260,1400,1500,1620,1680,2850,2960,3180,3370], type: "amino_acid" },
  Gln:   { name: "Glutamine",   modes: [1035,1160,1310,1405,1530,1615,1670,2860,2940,3200,3350], type: "amino_acid" },
  Met:   { name: "Methionine",  modes: [720,1020,1160,1310,1435,1560,1620,2840,2920,2960,3300], type: "amino_acid" },
  Trp:   { name: "Tryptophan",  modes: [740,880,1015,1100,1230,1340,1460,1560,1620,2860,2940,3060,3400], type: "amino_acid" },
  Tyr:   { name: "Tyrosine",    modes: [830,1015,1105,1175,1270,1445,1520,1600,2860,2930,3060,3200], type: "amino_acid" },
};

/**
 * GLB model path mapping.
 * Maps compound keys to actual GLB file paths in public/models/.
 */
export const GLB_PATHS = {
  Arg:   "/models/arginine.glb",
  Asn:   "/models/asparagine.glb",
  Gln:   "/models/glutamine.glb",
  Met:   "/models/methionine.glb",
  Trp:   "/models/tryptophan.glb",
  Tyr:   "/models/tyrosine.glb",
  CO2:   "/models/co2_adsorption_in_graphite.glb",  // CO₂ in graphite context
};

// Pre-encode all compounds
export function encodeAllCompounds() {
  const encoded = {};
  for (const [key, mol] of Object.entries(COMPOUNDS)) {
    const { S_k, S_t, S_e } = computeSEntropy(mol.modes, mol.B_rot || null);
    const trit = toTernaryString(S_k, S_t, S_e);
    const network = computeHarmonicNetwork(mol.modes);
    const zpve = computeZPVE(mol.modes);
    const loopFP = computeLoopFingerprint(mol.modes, network.edges);
    const glbPath = GLB_PATHS[key] || `/models/molecules/${key}.glb`;
    encoded[key] = { ...mol, key, S_k, S_t, S_e, trit, network, zpve, loopFP, glbPath };
  }
  return encoded;
}
