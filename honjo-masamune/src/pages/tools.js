/**
 * /tools — Categorical Cheminformatics Instruments
 * =================================================
 * Client-side molecular analysis workbench.
 * No backend. No API. No database. Just shaders and geometry.
 *
 * Instruments:
 *   I.   Partition Observation — GPU fragment shader observes molecular state
 *   II.  Interference Similarity — interference of two observation textures
 *   III. Harmonic Network — vibrational mode network with closed loops
 *   IV.  Property Prediction — ZPVE via S-entropy interpolation
 *   V.   Loop Fingerprint — cavity fingerprint from closed harmonic loops
 *   VI.  3D Structure — GLB molecular model with loop overlay + shape entropy
 *   VII. Docking — depth buffer + spectral interference docking
 */
import { useState, useMemo, useCallback } from "react";
import Head from "next/head";
import dynamic from "next/dynamic";
import {
  computeSEntropy,
  toTernaryString,
  sharedPrefixDepth,
  computeHarmonicNetwork,
  computeZPVE,
  computeLoopFingerprint,
  loopFingerprintDistance,
  predictProperty,
  encodeAllCompounds,
  COMPOUNDS,
} from "@/lib/sentropy";

// Dynamic imports for client-only components (WebGL + D3)
const PartitionObserver = dynamic(() => import("@/components/PartitionObserver"), { ssr: false });
const HarmonicNetwork = dynamic(() => import("@/components/HarmonicNetwork"), { ssr: false });
const MoleculeViewer3D = dynamic(() => import("@/components/MoleculeViewer3D"), { ssr: false });
const DockingInstrument = dynamic(() => import("@/components/DockingInstrument"), { ssr: false });
const SpectralHologram = dynamic(() => import("@/components/SpectralHologram"), { ssr: false });
const ChromatographyInstrument = dynamic(() => import("@/components/ChromatographyInstrument"), { ssr: false });
const AtomicExplorer = dynamic(() => import("@/components/AtomicExplorer"), { ssr: false });

const TYPE_COLORS = {
  diatomic: "#3b82f6",
  triatomic: "#22c55e",
  tetra: "#f97316",
  poly: "#ef4444",
  amino_acid: "#a855f7",
  custom: "#6b7280",
};

export default function Tools() {
  const db = useMemo(() => encodeAllCompounds(), []);
  const compoundKeys = Object.keys(db);

  const [selA, setSelA] = useState("H2O");
  const [selB, setSelB] = useState("CO2");
  const [customFreqs, setCustomFreqs] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [activeTab, setActiveTab] = useState("observe");

  // Molecule A (from dropdown or custom input)
  const molA = useMemo(() => {
    if (useCustom && customFreqs.trim()) {
      const modes = customFreqs.split(/[,\s]+/).map(Number).filter((n) => n > 0);
      if (modes.length === 0) return db[selA];
      const { S_k, S_t, S_e } = computeSEntropy(modes);
      const trit = toTernaryString(S_k, S_t, S_e);
      const network = computeHarmonicNetwork(modes);
      const zpve = computeZPVE(modes);
      return { name: "Custom", modes, S_k, S_t, S_e, trit, network, zpve, type: "custom", key: "custom" };
    }
    return db[selA];
  }, [selA, customFreqs, useCustom, db]);

  const molB = useMemo(() => db[selB], [selB, db]);

  // Similarity ranking for molA against all compounds
  const rankings = useMemo(() => {
    if (!molA) return [];
    return compoundKeys
      .filter((k) => k !== molA.key)
      .map((k) => ({
        key: k,
        ...db[k],
        spd: sharedPrefixDepth(molA.trit, db[k].trit),
        dist: Math.sqrt(
          (molA.S_k - db[k].S_k) ** 2 +
          (molA.S_t - db[k].S_t) ** 2 +
          (molA.S_e - db[k].S_e) ** 2
        ),
      }))
      .sort((a, b) => b.spd - a.spd || a.dist - b.dist);
  }, [molA, db, compoundKeys]);

  // ZPVE prediction for molA
  const zpvePredicted = useMemo(() => {
    if (!molA) return null;
    const refs = compoundKeys
      .filter((k) => k !== molA.key)
      .map((k) => db[k]);
    return predictProperty(molA, refs, (r) => r.zpve, 5);
  }, [molA, db, compoundKeys]);

  // Shape entropy state (from GLB normals)
  const [shapeEntropyA, setShapeEntropyA] = useState(null);

  // Loop fingerprint rankings
  const loopRankings = useMemo(() => {
    if (!molA?.loopFP) return [];
    return compoundKeys
      .filter((k) => k !== molA.key)
      .map((k) => ({
        key: k,
        ...db[k],
        loopDist: loopFingerprintDistance(molA.loopFP, db[k].loopFP),
      }))
      .sort((a, b) => a.loopDist - b.loopDist);
  }, [molA, db, compoundKeys]);

  const tabs = [
    { id: "observe", label: "Observation" },
    { id: "interfere", label: "Interference" },
    { id: "network", label: "Network" },
    { id: "fingerprint", label: "Fingerprint" },
    { id: "structure", label: "3D Structure" },
    { id: "docking", label: "Docking" },
    { id: "hologram", label: "Hologram" },
    { id: "predict", label: "Properties" },
    { id: "chromatography", label: "Retention" },
    { id: "atomic", label: "Atomic" },
  ];

  return (
    <>
      <Head>
        <title>Tools | Honjo Masamune</title>
        <meta name="description" content="Client-side categorical cheminformatics instruments. No backend, no database, just GPU shaders." />
      </Head>

      <main className="min-h-screen bg-[#0a0a0a] pt-24 pb-16 px-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <p className="text-[#58E6D9] text-sm tracking-[0.3em] uppercase mb-2">Instruments</p>
            <h1 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-3">
              Categorical Cheminformatics
            </h1>
            <p className="text-neutral-400 text-base max-w-2xl">
              Four GPU instruments operating entirely in your browser.
              No server, no API, no stored database. The measurement IS the computation.
            </p>
          </div>

          {/* Input Section */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
            {/* Molecule A selector */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
              <label className="text-neutral-500 text-xs tracking-wider uppercase block mb-2">
                Molecule A
              </label>
              <div className="flex gap-2 mb-3">
                <button
                  className={`text-xs px-3 py-1 rounded ${!useCustom ? "bg-[#58E6D9]/20 text-[#58E6D9]" : "bg-neutral-800 text-neutral-400"}`}
                  onClick={() => setUseCustom(false)}
                >
                  Database
                </button>
                <button
                  className={`text-xs px-3 py-1 rounded ${useCustom ? "bg-[#58E6D9]/20 text-[#58E6D9]" : "bg-neutral-800 text-neutral-400"}`}
                  onClick={() => setUseCustom(true)}
                >
                  Custom
                </button>
              </div>
              {useCustom ? (
                <textarea
                  className="w-full bg-neutral-800 text-white text-sm p-2 rounded font-mono border border-neutral-700 focus:border-[#58E6D9] outline-none"
                  rows={3}
                  placeholder="Enter frequencies in cm⁻¹ (comma or space separated)"
                  value={customFreqs}
                  onChange={(e) => setCustomFreqs(e.target.value)}
                />
              ) : (
                <select
                  className="w-full bg-neutral-800 text-white text-sm p-2 rounded border border-neutral-700 focus:border-[#58E6D9] outline-none"
                  value={selA}
                  onChange={(e) => setSelA(e.target.value)}
                >
                  {compoundKeys.map((k) => (
                    <option key={k} value={k}>{db[k].name} ({k})</option>
                  ))}
                </select>
              )}
            </div>

            {/* Molecule B selector (for interference) */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
              <label className="text-neutral-500 text-xs tracking-wider uppercase block mb-2">
                Molecule B (for interference)
              </label>
              <select
                className="w-full bg-neutral-800 text-white text-sm p-2 rounded border border-neutral-700 focus:border-[#58E6D9] outline-none mt-8"
                value={selB}
                onChange={(e) => setSelB(e.target.value)}
              >
                {compoundKeys.map((k) => (
                  <option key={k} value={k}>{db[k].name} ({k})</option>
                ))}
              </select>
            </div>

            {/* S-Entropy readout */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-4">
              <label className="text-neutral-500 text-xs tracking-wider uppercase block mb-2">
                S-Entropy Coordinates
              </label>
              {molA && (
                <div className="space-y-2 mt-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-400">S<sub>k</sub></span>
                    <span className="text-white font-mono">{molA.S_k.toFixed(4)}</span>
                  </div>
                  <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                    <div className="h-full bg-[#3b82f6] rounded-full" style={{ width: `${molA.S_k * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-400">S<sub>t</sub></span>
                    <span className="text-white font-mono">{molA.S_t.toFixed(4)}</span>
                  </div>
                  <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                    <div className="h-full bg-[#22c55e] rounded-full" style={{ width: `${molA.S_t * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-neutral-400">S<sub>e</sub></span>
                    <span className="text-white font-mono">{molA.S_e.toFixed(4)}</span>
                  </div>
                  <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                    <div className="h-full bg-[#f97316] rounded-full" style={{ width: `${molA.S_e * 100}%` }} />
                  </div>
                  <div className="mt-2 text-xs text-neutral-500 font-mono break-all">
                    {molA.trit}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-1 mb-4 border-b border-neutral-800 pb-0">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`px-4 py-2 text-sm tracking-wider uppercase transition-colors ${
                  activeTab === tab.id
                    ? "text-[#58E6D9] border-b-2 border-[#58E6D9]"
                    : "text-neutral-500 hover:text-neutral-300"
                }`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Instrument Panels */}
          <div className="bg-neutral-900/30 border border-neutral-800 rounded-xl p-6">
            {/* I. Partition Observation */}
            {activeTab === "observe" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument I: Partition Observation
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  GPU fragment shader observes the categorical state at every pixel.
                  Each peak corresponds to a vibrational mode of {molA.name}.
                </p>
                <PartitionObserver molA={molA} width={1024} height={320} />
                <div className="mt-4 flex flex-wrap gap-4 text-sm">
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">Modes: </span>
                    <span className="text-white font-mono">{molA.modes.length}</span>
                  </div>
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">Range: </span>
                    <span className="text-white font-mono">
                      {Math.min(...molA.modes)}–{Math.max(...molA.modes)} cm⁻¹
                    </span>
                  </div>
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">Type: </span>
                    <span className="text-white">{molA.type}</span>
                  </div>
                </div>
              </div>
            )}

            {/* II. Interference Similarity */}
            {activeTab === "interfere" && molA && molB && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument II: Interference Similarity
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Interference between {molA.name} and {molB.name} observation textures.
                  Bright regions = constructive interference (shared structure).
                </p>
                <PartitionObserver molA={molA} molB={molB} showInterference width={1024} height={320} />

                {/* Similarity ranking */}
                <div className="mt-6">
                  <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
                    Top 10 most similar to {molA.name}
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                    {rankings.slice(0, 10).map((r, i) => (
                      <button
                        key={r.key}
                        className={`text-left bg-neutral-800/50 rounded-lg p-3 border transition-colors ${
                          r.key === selB
                            ? "border-[#58E6D9]/50"
                            : "border-neutral-800 hover:border-neutral-700"
                        }`}
                        onClick={() => setSelB(r.key)}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-neutral-600 text-xs">#{i + 1}</span>
                          <span className="text-white text-sm font-medium">{r.name}</span>
                        </div>
                        <div className="text-xs text-neutral-500">
                          prefix depth: <span className="text-[#58E6D9] font-mono">{r.spd}</span>
                        </div>
                        <div className="text-xs text-neutral-500">
                          distance: <span className="font-mono">{r.dist.toFixed(4)}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* III. Harmonic Network */}
            {activeTab === "network" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument III: Harmonic Molecular Network
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Vibrational modes of {molA.name} as nodes. Harmonic edges connect
                  modes with rational frequency ratios. Teal = closed loop (virtual resonant cavity).
                </p>
                {molA.modes.length >= 2 ? (
                  <HarmonicNetwork
                    modes={molA.modes}
                    edges={molA.network.edges}
                    width={800}
                    height={400}
                  />
                ) : (
                  <div className="text-neutral-500 text-sm py-12 text-center">
                    Diatomic molecules have only one mode — select a polyatomic to see the network.
                  </div>
                )}
                <div className="mt-4 flex flex-wrap gap-4 text-sm">
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">Harmonic edges: </span>
                    <span className="text-white font-mono">{molA.network.edges.length}</span>
                  </div>
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">Network density: </span>
                    <span className="text-white font-mono">{molA.network.density.toFixed(3)}</span>
                  </div>
                  <div className="bg-neutral-800/50 rounded px-3 py-2">
                    <span className="text-neutral-500">S<sub>e</sub>: </span>
                    <span className="text-[#f97316] font-mono">{molA.S_e.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* V. Loop Fingerprint */}
            {activeTab === "fingerprint" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument V: Loop (Cavity) Fingerprint
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Closed loops in the harmonic network = virtual resonant cavities.
                  The fingerprint encodes cavity count, Q-factors, and circulation frequencies.
                </p>

                {/* Fingerprint summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                  <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Cavities</div>
                    <div className="text-2xl font-bold text-[#58E6D9] font-mono">{molA.loopFP.nLoops}</div>
                  </div>
                  <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Mean Q-factor</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {molA.loopFP.meanQ > 0 ? molA.loopFP.meanQ.toFixed(1) : "—"}
                    </div>
                  </div>
                  <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Mean Circ. Freq.</div>
                    <div className="text-lg font-bold text-white font-mono">
                      {molA.loopFP.meanCircFreq > 0 ? molA.loopFP.meanCircFreq.toExponential(2) : "—"}
                    </div>
                    <div className="text-neutral-600 text-xs">Hz</div>
                  </div>
                  <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Mean Loop Size</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {molA.loopFP.meanSize > 0 ? molA.loopFP.meanSize.toFixed(1) : "—"}
                    </div>
                    <div className="text-neutral-600 text-xs">modes/loop</div>
                  </div>
                </div>

                {/* Individual loops */}
                {molA.loopFP.loops.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
                      Detected cavities
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                      {molA.loopFP.loops.map((loop, i) => (
                        <div key={i} className="bg-neutral-800/30 rounded-lg p-3 border border-neutral-800">
                          <div className="flex items-center gap-2 mb-1">
                            <div className="w-2 h-2 rounded-full bg-[#58E6D9]" />
                            <span className="text-white text-sm font-medium">Loop {i + 1}</span>
                            <span className="text-neutral-500 text-xs">({loop.nodes.length} modes)</span>
                          </div>
                          <div className="text-xs text-neutral-400 font-mono">
                            modes: {loop.modes.join(", ")} cm⁻¹
                          </div>
                          <div className="text-xs text-neutral-400 font-mono">
                            Q = {loop.Q.toFixed(1)} · f_circ = {loop.circFreq.toExponential(2)} Hz
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Fingerprint-based ranking */}
                <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
                  Most similar by cavity fingerprint
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  {loopRankings.slice(0, 10).map((r, i) => (
                    <div key={r.key} className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-800">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-neutral-600 text-xs">#{i + 1}</span>
                        <span className="text-white text-sm font-medium">{r.name}</span>
                      </div>
                      <div className="text-xs text-neutral-500">
                        loops: <span className="text-[#58E6D9] font-mono">{r.loopFP.nLoops}</span>
                      </div>
                      <div className="text-xs text-neutral-500">
                        dist: <span className="font-mono">{r.loopDist.toFixed(4)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* VI. 3D Structure */}
            {activeTab === "structure" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument VI: 3D Molecular Structure
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  GLB molecular model with harmonic loop overlay.
                  Teal = modes participating in closed loops (virtual cavities).
                  Shape entropy S_g computed from mesh normals.
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <MoleculeViewer3D
                    mol={molA}
                    width={500}
                    height={400}
                    onShapeEntropy={setShapeEntropyA}
                  />
                  <div className="space-y-4">
                    {/* S-entropy + shape entropy (4D coordinates) */}
                    <div className="bg-neutral-800/50 rounded-xl p-5 border border-neutral-700">
                      <h3 className="text-sm font-semibold text-neutral-300 mb-3 uppercase tracking-wider">
                        4D Categorical Coordinates
                      </h3>
                      <div className="space-y-3">
                        {[
                          { label: "S_k (knowledge)", value: molA.S_k, color: "#3b82f6" },
                          { label: "S_t (temporal)", value: molA.S_t, color: "#22c55e" },
                          { label: "S_e (evolution)", value: molA.S_e, color: "#f97316" },
                          { label: "S_g (shape)", value: shapeEntropyA, color: "#a855f7" },
                        ].map(({ label, value, color }) => (
                          <div key={label}>
                            <div className="flex justify-between text-sm mb-1">
                              <span className="text-neutral-400">{label}</span>
                              <span className="text-white font-mono">
                                {value !== null && value !== undefined ? value.toFixed(4) : "—"}
                              </span>
                            </div>
                            <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{
                                  width: `${(value ?? 0) * 100}%`,
                                  backgroundColor: color,
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="text-neutral-600 text-xs mt-3">
                        {shapeEntropyA !== null
                          ? "S_g extracted from GLB mesh normals (Gauss map entropy)"
                          : "Load a GLB model to compute shape entropy S_g"}
                      </p>
                    </div>

                    {/* Molecule info */}
                    <div className="bg-neutral-800/50 rounded-xl p-5 border border-neutral-700">
                      <h3 className="text-sm font-semibold text-neutral-300 mb-3 uppercase tracking-wider">
                        Molecule
                      </h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Formula</span>
                          <span className="text-white">{molA.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Type</span>
                          <span className="text-white">{molA.type}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Modes</span>
                          <span className="text-white font-mono">{molA.modes.length}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">Cavities</span>
                          <span className="text-[#58E6D9] font-mono">{molA.loopFP.nLoops}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-neutral-400">ZPVE</span>
                          <span className="text-white font-mono">{molA.zpve.toFixed(1)} kJ/mol</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-neutral-600 text-xs mt-4">
                  Place GLB models in <code className="text-neutral-500">public/models/molecules/</code> named
                  by compound key (e.g., H2O.glb, C6H6.glb). Procedural representation shown when GLB is unavailable.
                </p>
              </div>
            )}

            {/* VII. Docking */}
            {activeTab === "docking" && molA && molB && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument VII: Spectral Docking
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Spatial proximity + spectral interference between {molA.name} and {molB.name}.
                  Teal lines = harmonically compatible mode pairs. Adjust separation and rotation to scan poses.
                </p>

                <DockingInstrument molA={molA} molB={molB} width={900} height={400} />

                <p className="text-neutral-600 text-xs mt-4">
                  Docking score combines spectral complementarity (harmonic mode compatibility)
                  with S-entropy distance. No force fields, no MD, no training data.
                </p>
              </div>
            )}

            {/* VIII. Spectral Hologram */}
            {activeTab === "hologram" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument VIII: Spectral Hologram
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Three-state superposition of {molA.name}: ground (IR), excited (Raman),
                  and emission spectra combined into a complete phase space hologram.
                  Adjust weights to explore cross-state structure. Diffraction pattern
                  via 2D FFT reveals molecular symmetry.
                </p>
                <SpectralHologram mol={molA} />
              </div>
            )}

            {/* IV. Property Prediction */}
            {activeTab === "predict" && molA && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument IV: Property Prediction
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Zero-parameter prediction via inverse-distance weighting in S-entropy space.
                  No training, no neural network — just coordinate interpolation.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-neutral-800/50 rounded-xl p-5 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">ZPVE (direct)</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {molA.zpve.toFixed(2)}
                    </div>
                    <div className="text-neutral-500 text-xs">kJ/mol</div>
                  </div>
                  <div className="bg-neutral-800/50 rounded-xl p-5 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">ZPVE (predicted, K=5)</div>
                    <div className="text-2xl font-bold text-[#58E6D9] font-mono">
                      {zpvePredicted !== null ? zpvePredicted.toFixed(2) : "—"}
                    </div>
                    <div className="text-neutral-500 text-xs">kJ/mol</div>
                  </div>
                  <div className="bg-neutral-800/50 rounded-xl p-5 border border-neutral-700">
                    <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Prediction error</div>
                    <div className="text-2xl font-bold text-white font-mono">
                      {zpvePredicted !== null && molA.zpve > 0
                        ? `${(Math.abs(zpvePredicted - molA.zpve) / molA.zpve * 100).toFixed(1)}%`
                        : "—"}
                    </div>
                    <div className="text-neutral-500 text-xs">
                      Lipschitz bound: error → 0 as reference density → ∞
                    </div>
                  </div>
                </div>

                {/* Frequency table */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-800">
                  <h3 className="text-sm font-semibold text-neutral-300 mb-3 uppercase tracking-wider">
                    Vibrational modes
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {molA.modes.map((w, i) => (
                      <div
                        key={i}
                        className="bg-neutral-800 rounded px-3 py-1.5 text-sm font-mono text-white border border-neutral-700"
                      >
                        {w} <span className="text-neutral-500 text-xs">cm⁻¹</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 5 nearest neighbours used for prediction */}
                <div className="mt-4">
                  <h3 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
                    K=5 nearest reference compounds
                  </h3>
                  <div className="grid grid-cols-5 gap-2">
                    {rankings.slice(0, 5).map((r) => (
                      <div key={r.key} className="bg-neutral-800/50 rounded-lg p-3 border border-neutral-800">
                        <div className="text-white text-sm font-medium">{r.name}</div>
                        <div className="text-xs text-neutral-500 font-mono">
                          d = {r.dist.toFixed(4)}
                        </div>
                        <div className="text-xs text-neutral-500 font-mono">
                          ZPVE = {r.zpve.toFixed(1)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* IX. Chromatography Retention */}
            {activeTab === "chromatography" && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument IX: Chromatography Retention
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Predicts C18 (reverse-phase) and HILIC (hydrophilic) retention times directly from
                  partition coordinates (n, ℓ, m). The column IS the projection operator —
                  no force field, no training data, no calibration.
                </p>
                <ChromatographyInstrument />
              </div>
            )}

            {/* X. Atomic Explorer */}
            {activeTab === "atomic" && (
              <div>
                <h2 className="text-lg font-bold text-white mb-1">
                  Instrument X: Partition Atomic Explorer
                </h2>
                <p className="text-neutral-500 text-sm mb-4">
                  Given Z, derives the electronic configuration from the partition selection rules
                  (Δℓ = ±1, Δm ∈ {"{0, ±1}"}, Δs = 0) and shell capacity C(n) = 2n². The periodic
                  table falls out — no shell model is postulated.
                </p>
                <AtomicExplorer />
              </div>
            )}
          </div>

          {/* Footer note */}
          <p className="text-neutral-600 text-xs mt-6 text-center">
            All computation runs in your browser. No data is sent to any server.
            39 NIST CCCBDB compounds. Zero adjustable parameters.
          </p>
        </div>
      </main>
    </>
  );
}
