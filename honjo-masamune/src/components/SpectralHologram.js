/**
 * SpectralHologram — Three-state superposition viewer
 * =====================================================
 * Superimposes ground (IR), excited (Raman), and emission spectra.
 * Computes 2D FFT diffraction pattern, coupling matrix, Stokes shift,
 * and Huang-Rhys factors. All client-side.
 */
import { useRef, useEffect, useMemo, useState } from "react";
import * as d3 from "d3";
import {
  buildSpectrum,
  buildEmissionSpectrum,
  buildHologram,
  build2DHologram,
  computeFFT2D,
  computeCouplingMatrix,
  computeHuangRhys,
  computeStokesShift,
  getExcitedModes,
  EMISSION_DATA,
} from "@/lib/hologram";

// ─── 1D Hologram Chart ──────────────────────────────────────────────

function HologramChart({ groundModes, excitedModes, weights, width = 700, height = 250 }) {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current || !groundModes.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 10, right: 20, bottom: 35, left: 45 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const omega = Float64Array.from({ length: 1000 }, (_, i) => 200 + i * 3.8);
    const sG = buildSpectrum(groundModes, omega);
    const sE = buildSpectrum(excitedModes, omega);
    const sEm = buildEmissionSpectrum(groundModes, excitedModes, omega);
    const holo = buildHologram(sG, sE, sEm, weights[0], weights[1], weights[2]);

    const x = d3.scaleLinear().domain([200, 4000]).range([0, w]);
    const yMax = Math.max(...holo, 0.1);
    const y = d3.scaleLinear().domain([0, yMax * 1.1]).range([h, 0]);

    // Axes
    g.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(8))
      .selectAll("text").attr("fill", "#999").attr("font-size", "7px");
    g.append("g").call(d3.axisLeft(y).ticks(4))
      .selectAll("text").attr("fill", "#999").attr("font-size", "7px");
    g.selectAll(".domain, .tick line").attr("stroke", "#444");

    // Individual spectra (faded)
    const line = (data) => d3.line().x((_, i) => x(omega[i])).y((d) => y(d))(data);
    g.append("path").attr("d", line(sG)).attr("fill", "none").attr("stroke", "#3b82f6").attr("stroke-width", 0.8).attr("opacity", 0.4);
    g.append("path").attr("d", line(sE)).attr("fill", "none").attr("stroke", "#ef4444").attr("stroke-width", 0.8).attr("opacity", 0.4);
    g.append("path").attr("d", line(sEm)).attr("fill", "none").attr("stroke", "#22c55e").attr("stroke-width", 0.8).attr("opacity", 0.4);

    // Hologram (bold)
    const area = d3.area().x((_, i) => x(omega[i])).y0(h).y1((d) => y(d));
    g.append("path").attr("d", area(holo)).attr("fill", "#a855f7").attr("opacity", 0.2);
    g.append("path").attr("d", line(holo)).attr("fill", "none").attr("stroke", "#a855f7").attr("stroke-width", 1.5);

  }, [groundModes, excitedModes, weights, width, height]);

  return <svg ref={ref} width={width} height={height} className="bg-[#0d0d0d] rounded-lg border border-neutral-800" />;
}

// ─── Diffraction Pattern (2D FFT) ───────────────────────────────────

function DiffractionCanvas({ groundModes, excitedModes, size = 256 }) {
  const canvasRef = useRef();

  useEffect(() => {
    if (!canvasRef.current || !groundModes.length) return;
    const N = 128; // power of 2 for FFT

    const omega = Float64Array.from({ length: 500 }, (_, i) => 200 + i * 7.6);
    const sG = buildSpectrum(groundModes, omega);
    const sE = buildSpectrum(excitedModes, omega);
    const sEm = buildEmissionSpectrum(groundModes, excitedModes, omega);
    const tex = build2DHologram(sG, sE, sEm, N);
    const mag = computeFFT2D(tex, N);

    // Render to canvas
    const canvas = canvasRef.current;
    canvas.width = N;
    canvas.height = N;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(N, N);
    const maxMag = Math.max(...mag, 1);

    for (let i = 0; i < N * N; i++) {
      const v = Math.min(255, Math.floor((mag[i] / maxMag) * 255));
      // Inferno-like colormap
      const r = Math.min(255, v * 1.2);
      const g = Math.max(0, v * 0.6 - 30);
      const b = Math.max(0, 80 - v * 0.3);
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [groundModes, excitedModes, size]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, imageRendering: "pixelated" }}
      className="rounded-lg border border-neutral-800"
    />
  );
}

// ─── Coupling Matrix Heatmap ────────────────────────────────────────

function CouplingHeatmap({ groundModes, excitedModes, modeNames, size = 250 }) {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current || !groundModes.length) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const { K } = computeCouplingMatrix(groundModes, excitedModes);
    const N = K.length;
    const cellSize = (size - 40) / N;

    const g = svg.append("g").attr("transform", "translate(35, 5)");
    const color = d3.scaleSequential(d3.interpolateRdYlGn).domain([-1, 1]);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        g.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 1)
          .attr("height", cellSize - 1)
          .attr("fill", color(K[i][j]))
          .attr("rx", 2);
      }
    }

    // Labels
    const labels = modeNames || groundModes.map((_, i) => `${i}`);
    for (let i = 0; i < N; i++) {
      g.append("text").attr("x", -4).attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .attr("fill", "#999").attr("font-size", "7px").text(labels[i] || i);
      g.append("text").attr("x", i * cellSize + cellSize / 2).attr("y", N * cellSize + 12)
        .attr("text-anchor", "middle").attr("fill", "#999").attr("font-size", "7px").text(labels[i] || i);
    }
  }, [groundModes, excitedModes, modeNames, size]);

  return <svg ref={ref} width={size} height={size} className="bg-[#0d0d0d] rounded-lg border border-neutral-800" />;
}

// ─── Main Hologram Component ────────────────────────────────────────

export default function SpectralHologram({ mol }) {
  const [wG, setWG] = useState(1.0);
  const [wE, setWE] = useState(1.0);
  const [wEm, setWEm] = useState(1.0);

  const groundModes = useMemo(() => mol?.modes || [], [mol?.modes]);
  const excitedModes = useMemo(() => getExcitedModes(groundModes), [groundModes]);

  const emData = EMISSION_DATA[mol?.key] || EMISSION_DATA._default;

  const coupling = useMemo(() => {
    if (groundModes.length < 2) return null;
    return computeCouplingMatrix(groundModes, excitedModes);
  }, [groundModes, excitedModes]);

  const huangRhys = useMemo(() => {
    if (groundModes.length < 1) return [];
    return computeHuangRhys(groundModes, excitedModes);
  }, [groundModes, excitedModes]);

  const stokes = useMemo(() => {
    return computeStokesShift(groundModes, excitedModes, emData.absNm, emData.emNm);
  }, [groundModes, excitedModes, emData]);

  if (!mol || !groundModes.length) return null;

  return (
    <div className="space-y-6">
      {/* Weight controls */}
      <div className="flex flex-wrap gap-6 items-center">
        {[
          { label: "Ground (IR)", color: "#3b82f6", val: wG, set: setWG },
          { label: "Excited (Raman)", color: "#ef4444", val: wE, set: setWE },
          { label: "Emission", color: "#22c55e", val: wEm, set: setWEm },
        ].map(({ label, color, val, set }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-neutral-500 text-xs">{label}</span>
            <input
              type="range" min="0" max="2" step="0.1" value={val}
              onChange={(e) => set(parseFloat(e.target.value))}
              className="w-24 accent-[#58E6D9]"
            />
            <span className="text-white text-xs font-mono w-6">{val.toFixed(1)}</span>
          </div>
        ))}
      </div>

      {/* Hologram + Diffraction */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <HologramChart
            groundModes={groundModes}
            excitedModes={excitedModes}
            weights={[wG, wE, wEm]}
            width={700}
            height={250}
          />
          <div className="flex gap-2 mt-2">
            <span className="text-neutral-600 text-xs">Purple = superimposed hologram</span>
            <span className="text-neutral-700 text-xs">|</span>
            <span className="text-[#3b82f6] text-xs">Blue = ground</span>
            <span className="text-[#ef4444] text-xs">Red = excited</span>
            <span className="text-[#22c55e] text-xs">Green = emission</span>
          </div>
        </div>
        <div>
          <DiffractionCanvas groundModes={groundModes} excitedModes={excitedModes} size={250} />
          <div className="text-neutral-600 text-xs mt-2">2D FFT diffraction pattern</div>
        </div>
      </div>

      {/* Extracted quantities */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Stokes Shift</div>
          <div className="text-xl font-bold text-[#a855f7] font-mono">{stokes.stokes.toFixed(1)}</div>
          <div className="text-neutral-600 text-xs">cm⁻¹</div>
        </div>
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">Reorganisation λ</div>
          <div className="text-xl font-bold text-[#f97316] font-mono">{stokes.reorg.toFixed(1)}</div>
          <div className="text-neutral-600 text-xs">cm⁻¹</div>
        </div>
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">ΔE vib</div>
          <div className="text-xl font-bold text-[#3b82f6] font-mono">{stokes.deltaVib.toFixed(1)}</div>
          <div className="text-neutral-600 text-xs">cm⁻¹ (vibrational)</div>
        </div>
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <div className="text-neutral-500 text-xs uppercase tracking-wider mb-1">ΔE solv</div>
          <div className="text-xl font-bold text-[#22c55e] font-mono">{stokes.deltaSolv.toFixed(1)}</div>
          <div className="text-neutral-600 text-xs">cm⁻¹ (solvent)</div>
        </div>
      </div>

      {/* Coupling matrix + Huang-Rhys */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Coupling matrix */}
        {coupling && (
          <div>
            <h4 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
              Coupling Matrix K<sub>ij</sub>
            </h4>
            <CouplingHeatmap
              groundModes={groundModes}
              excitedModes={excitedModes}
              modeNames={groundModes.map((f) => `${f}`)}
              size={Math.min(300, groundModes.length * 35 + 50)}
            />
          </div>
        )}

        {/* Huang-Rhys factors */}
        <div>
          <h4 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
            Huang-Rhys Factors
          </h4>
          <div className="space-y-1">
            {huangRhys.map((hr, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-neutral-500 text-xs font-mono w-16">{hr.groundFreq}</span>
                <div className="flex-1 h-3 bg-neutral-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#a855f7] rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, hr.S * 500)}%` }}
                  />
                </div>
                <span className="text-white text-xs font-mono w-12">{hr.S.toFixed(4)}</span>
                <span className="text-neutral-600 text-xs font-mono w-12">Δ{hr.shift}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Frequency shifts table */}
      {coupling && (
        <div>
          <h4 className="text-sm font-semibold text-neutral-300 mb-2 uppercase tracking-wider">
            Frequency Shifts (Ground → Excited)
          </h4>
          <div className="flex flex-wrap gap-2">
            {coupling.shifts.map((s, i) => (
              <div
                key={i}
                className={`rounded px-3 py-1.5 text-sm font-mono border ${
                  Math.abs(s) > 10
                    ? "border-[#ef4444]/30 bg-[#ef4444]/10 text-[#ef4444]"
                    : "border-neutral-700 bg-neutral-800/50 text-neutral-400"
                }`}
              >
                {groundModes[i]} → {excitedModes[i].toFixed(0)}{" "}
                <span className={s > 0 ? "text-[#22c55e]" : "text-[#ef4444]"}>
                  ({s > 0 ? "+" : ""}{s.toFixed(1)})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
