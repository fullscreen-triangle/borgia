import Head from "next/head";
import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";
import { motion } from "framer-motion";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Composition Heatmap (48 labeled compositions of n=3, d=3)
   ═══════════════════════════════════════════════════════════════════════════ */
function CompositionHeatmap() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 900;
      const height = 520;
      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      // Integer compositions of 3
      const compositions = [
        [1, 1, 1],
        [2, 1],
        [1, 2],
        [3],
      ];
      const compLabels = ["1+1+1", "2+1", "1+2", "3"];

      // S-entropy dimensions
      const dims = ["S_k", "S_t", "S_e"];
      const dimColors = ["#4A9EFF", "#34D399", "#F97316"];

      // Generate all 48 labeled trajectories
      // For each composition, each step is labeled by one of 3 dimensions
      // A composition with k parts has 3^k labelings, but our formula gives
      // d * (d+1)^(n-1) total. For the heatmap we show all 48 as grouped rows.
      const groups = [];
      compositions.forEach((comp, ci) => {
        const k = comp.length; // number of parts
        // Generate all d^k = 3^k label assignments
        const labelCount = Math.pow(3, k);
        for (let li = 0; li < labelCount; li++) {
          const labels = [];
          let tmp = li;
          for (let p = 0; p < k; p++) {
            labels.push(tmp % 3);
            tmp = Math.floor(tmp / 3);
          }
          groups.push({
            compIndex: ci,
            compLabel: compLabels[ci],
            parts: comp,
            dimLabels: labels,
          });
        }
      });

      // Layout: group by composition
      const margin = { top: 60, right: 30, bottom: 50, left: 140 };
      const cellW = 14;
      const cellH = 10;
      const groupGap = 12;

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("font-weight", "600")
        .text("All 48 Labeled Compositions of n = 3, d = 3");

      let yOff = margin.top;
      const compGroupStarts = [];

      compositions.forEach((comp, ci) => {
        const rows = groups.filter((g) => g.compIndex === ci);
        compGroupStarts.push(yOff);

        // Group label
        svg
          .append("text")
          .attr("x", margin.left - 12)
          .attr("y", yOff + (rows.length * cellH) / 2 + 4)
          .attr("text-anchor", "end")
          .attr("fill", "#999")
          .attr("font-size", "11px")
          .attr("font-family", "'Courier New', monospace")
          .text(compLabels[ci]);

        // Row count
        svg
          .append("text")
          .attr("x", margin.left - 12)
          .attr("y", yOff + (rows.length * cellH) / 2 + 18)
          .attr("text-anchor", "end")
          .attr("fill", "#666")
          .attr("font-size", "9px")
          .attr("font-family", "Montserrat, sans-serif")
          .text(`(${rows.length} trajectories)`);

        rows.forEach((row, ri) => {
          const y = yOff + ri * cellH;
          // Draw cells for each part
          let xOff = margin.left;
          row.parts.forEach((partLen, pi) => {
            const color = dimColors[row.dimLabels[pi]];
            // Each part spans partLen cells
            for (let c = 0; c < partLen; c++) {
              svg
                .append("rect")
                .attr("x", xOff + c * cellW)
                .attr("y", y)
                .attr("width", cellW - 1)
                .attr("height", cellH - 1)
                .attr("fill", color)
                .attr("opacity", 0.75)
                .attr("rx", 1);
            }
            xOff += partLen * cellW;
          });
        });

        yOff += rows.length * cellH + groupGap;
      });

      // Legend
      const legendX = margin.left + 3 * cellW + 40;
      const legendY = yOff + 10;
      dims.forEach((dim, i) => {
        svg
          .append("rect")
          .attr("x", legendX + i * 100)
          .attr("y", legendY)
          .attr("width", 12)
          .attr("height", 12)
          .attr("fill", dimColors[i])
          .attr("rx", 2);
        svg
          .append("text")
          .attr("x", legendX + i * 100 + 18)
          .attr("y", legendY + 10)
          .attr("fill", "#aaa")
          .attr("font-size", "11px")
          .attr("font-family", "Montserrat, sans-serif")
          .text(dim);
      });

      // Total count
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", legendY + 38)
        .attr("text-anchor", "middle")
        .attr("fill", "#58E6D9")
        .attr("font-size", "13px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Total: 3 × 4² = 48 distinguishable trajectories");
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[900px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Semilog T(n,3) vs n
   ═══════════════════════════════════════════════════════════════════════════ */
function TrajectoryGrowthChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 800;
      const height = 460;
      const margin = { top: 50, right: 40, bottom: 60, left: 80 };
      const iw = width - margin.left - margin.right;
      const ih = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      // Data
      const nValues = d3.range(1, 61);
      const Tn = nValues.map((n) => ({
        n,
        logT: Math.log10(3) + (n - 1) * Math.log10(4),
        logLinear: Math.log10(3 * n),
        logQuad: Math.log10(n * n),
      }));

      const x = d3.scaleLinear().domain([1, 60]).range([0, iw]);
      const y = d3
        .scaleLinear()
        .domain([0, 36])
        .range([ih, 0]);

      // Grid
      g.append("g")
        .selectAll("line")
        .data(d3.range(0, 37, 4))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", iw)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#222")
        .attr("stroke-width", 0.5);

      // Axes
      g.append("g")
        .attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(12))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      g.append("g")
        .call(d3.axisLeft(y).ticks(10))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      // Axis labels
      g.append("text")
        .attr("x", iw / 2)
        .attr("y", ih + 45)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("n (ticks)");

      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -ih / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("log₁₀ T(n, 3)");

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Trajectory Count T(n, 3) = 3 · 4^(n−1)");

      // Line generators
      const lineGen = d3
        .line()
        .x((d) => x(d.n))
        .y((d) => y(d.logT));

      const lineLinear = d3
        .line()
        .x((d) => x(d.n))
        .y((d) => y(d.logLinear));

      const lineQuad = d3
        .line()
        .x((d) => x(d.n))
        .y((d) => y(d.logQuad));

      // Linear (grey)
      g.append("path")
        .datum(Tn)
        .attr("d", lineLinear)
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "none");

      // Quadratic (grey dashed)
      g.append("path")
        .datum(Tn)
        .attr("d", lineQuad)
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "6,4");

      // T(n,3) (teal)
      g.append("path")
        .datum(Tn)
        .attr("d", lineGen)
        .attr("fill", "none")
        .attr("stroke", "#58E6D9")
        .attr("stroke-width", 2.5);

      // Legend
      const legend = g.append("g").attr("transform", `translate(${iw - 200}, 10)`);
      [
        { label: "T(n, 3) = 3·4^(n−1)", color: "#58E6D9", dash: "none" },
        { label: "Linear: 3n", color: "#555", dash: "none" },
        { label: "Quadratic: n²", color: "#555", dash: "6,4" },
      ].forEach((item, i) => {
        legend
          .append("line")
          .attr("x1", 0)
          .attr("x2", 24)
          .attr("y1", i * 20)
          .attr("y2", i * 20)
          .attr("stroke", item.color)
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", item.dash);
        legend
          .append("text")
          .attr("x", 30)
          .attr("y", i * 20 + 4)
          .attr("fill", "#aaa")
          .attr("font-size", "11px")
          .attr("font-family", "Montserrat, sans-serif")
          .text(item.label);
      });
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[800px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Angular Resolution log₁₀(Δθ) vs n with Planck threshold
   ═══════════════════════════════════════════════════════════════════════════ */
function AngularResolutionChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 800;
      const height = 460;
      const margin = { top: 50, right: 40, bottom: 60, left: 80 };
      const iw = width - margin.left - margin.right;
      const ih = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      // Data: log10(Δθ) = log10(2π) - log10(T(n,3))
      //                  = log10(2π) - [log10(3) + (n-1)*log10(4)]
      const nValues = d3.range(1, 61);
      const data = nValues.map((n) => ({
        n,
        logDtheta:
          Math.log10(2 * Math.PI) -
          (Math.log10(3) + (n - 1) * Math.log10(4)),
      }));

      const planckThreshold = Math.log10(3.1e-33); // ≈ -32.51

      const x = d3.scaleLinear().domain([1, 60]).range([0, iw]);
      const y = d3.scaleLinear().domain([1, -37]).range([0, ih]);

      // Sub-Planck shaded region
      g.append("rect")
        .attr("x", 0)
        .attr("y", y(planckThreshold))
        .attr("width", iw)
        .attr("height", ih - y(planckThreshold))
        .attr("fill", "#58E6D9")
        .attr("opacity", 0.06);

      // Grid
      g.append("g")
        .selectAll("line")
        .data(d3.range(-35, 5, 5))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", iw)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#222")
        .attr("stroke-width", 0.5);

      // Planck threshold line
      g.append("line")
        .attr("x1", 0)
        .attr("x2", iw)
        .attr("y1", y(planckThreshold))
        .attr("y2", y(planckThreshold))
        .attr("stroke", "#F97316")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "8,4");

      g.append("text")
        .attr("x", iw - 4)
        .attr("y", y(planckThreshold) - 8)
        .attr("text-anchor", "end")
        .attr("fill", "#F97316")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Planck angular threshold ≈ −32.5");

      // Axes
      g.append("g")
        .attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(12))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      g.append("g")
        .call(d3.axisLeft(y).ticks(10))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      // Axis labels
      g.append("text")
        .attr("x", iw / 2)
        .attr("y", ih + 45)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("n (ticks)");

      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -ih / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("log₁₀(Δθ)  [radians]");

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Angular Resolution vs. Tick Count");

      // Line
      const lineGen = d3
        .line()
        .x((d) => x(d.n))
        .y((d) => y(d.logDtheta));

      g.append("path")
        .datum(data)
        .attr("d", lineGen)
        .attr("fill", "none")
        .attr("stroke", "#58E6D9")
        .attr("stroke-width", 2.5);

      // Crossing point annotation at n=56
      const cross56 = data.find((d) => d.n === 56);
      if (cross56) {
        g.append("circle")
          .attr("cx", x(56))
          .attr("cy", y(cross56.logDtheta))
          .attr("r", 5)
          .attr("fill", "#58E6D9")
          .attr("stroke", "#fff")
          .attr("stroke-width", 1.5);

        g.append("line")
          .attr("x1", x(56))
          .attr("x2", x(56))
          .attr("y1", y(cross56.logDtheta))
          .attr("y2", y(cross56.logDtheta) - 40)
          .attr("stroke", "#58E6D9")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "3,3");

        g.append("text")
          .attr("x", x(56))
          .attr("y", y(cross56.logDtheta) - 46)
          .attr("text-anchor", "middle")
          .attr("fill", "#58E6D9")
          .attr("font-size", "12px")
          .attr("font-weight", "600")
          .attr("font-family", "Montserrat, sans-serif")
          .text("n = 56  (crosses Planck)");
      }
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[800px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Planck Depth for different oscillators
   ═══════════════════════════════════════════════════════════════════════════ */
function PlanckDepthOscillatorsChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 800;
      const height = 360;
      const margin = { top: 50, right: 40, bottom: 30, left: 200 };
      const iw = width - margin.left - margin.right;
      const ih = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const data = [
        { label: "Strontium optical (4.29×10¹⁴ Hz)", nP: 48 },
        { label: "H₂ vibration (1.32×10¹⁴ Hz)", nP: 49 },
        { label: "Caesium-133 (9.19×10⁹ Hz)", nP: 56 },
        { label: "Hydrogen maser (1.42×10⁹ Hz)", nP: 57 },
        { label: "CPU 3 GHz", nP: 57 },
      ];

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Planck Depth n_P by Oscillator");

      const y = d3
        .scaleBand()
        .domain(data.map((d) => d.label))
        .range([0, ih])
        .padding(0.35);

      const x = d3.scaleLinear().domain([0, 65]).range([0, iw]);

      // Bars
      g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", 0)
        .attr("y", (d) => y(d.label))
        .attr("width", (d) => x(d.nP))
        .attr("height", y.bandwidth())
        .attr("fill", "#58E6D9")
        .attr("opacity", 0.8)
        .attr("rx", 3);

      // Bar labels (nP values)
      g.selectAll(".val")
        .data(data)
        .enter()
        .append("text")
        .attr("x", (d) => x(d.nP) + 8)
        .attr("y", (d) => y(d.label) + y.bandwidth() / 2 + 5)
        .attr("fill", "#58E6D9")
        .attr("font-size", "13px")
        .attr("font-weight", "700")
        .attr("font-family", "'Courier New', monospace")
        .text((d) => d.nP);

      // Y axis labels
      g.selectAll(".label")
        .data(data)
        .enter()
        .append("text")
        .attr("x", -8)
        .attr("y", (d) => y(d.label) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", "end")
        .attr("fill", "#bbb")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text((d) => d.label);

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(7))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[800px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Dimensional Advantage n_P vs d
   ═══════════════════════════════════════════════════════════════════════════ */
function DimensionalAdvantageChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 700;
      const height = 380;
      const margin = { top: 50, right: 40, bottom: 60, left: 80 };
      const iw = width - margin.left - margin.right;
      const ih = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const data = [
        { d: 1, label: "d=1 (binary)", nP: 112 },
        { d: 2, label: "d=2", nP: 71 },
        { d: 3, label: "d=3 (S-entropy)", nP: 56 },
        { d: 4, label: "d=4", nP: 49 },
        { d: 5, label: "d=5", nP: 44 },
      ];

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Planck Depth n_P vs. Dimension d (Caesium-133)");

      const x = d3
        .scaleBand()
        .domain(data.map((d) => d.label))
        .range([0, iw])
        .padding(0.4);

      const y = d3.scaleLinear().domain([0, 120]).range([ih, 0]);

      // Grid lines
      g.append("g")
        .selectAll("line")
        .data(d3.range(0, 121, 20))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", iw)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#1a1a1a")
        .attr("stroke-width", 0.5);

      // Bars
      g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d) => x(d.label))
        .attr("y", (d) => y(d.nP))
        .attr("width", x.bandwidth())
        .attr("height", (d) => ih - y(d.nP))
        .attr("fill", (d) => (d.d === 3 ? "#58E6D9" : "#334155"))
        .attr("rx", 3);

      // Value labels
      g.selectAll(".val")
        .data(data)
        .enter()
        .append("text")
        .attr("x", (d) => x(d.label) + x.bandwidth() / 2)
        .attr("y", (d) => y(d.nP) - 8)
        .attr("text-anchor", "middle")
        .attr("fill", (d) => (d.d === 3 ? "#58E6D9" : "#888"))
        .attr("font-size", "14px")
        .attr("font-weight", "700")
        .attr("font-family", "'Courier New', monospace")
        .text((d) => d.nP);

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      // Y axis
      g.append("g")
        .call(d3.axisLeft(y).ticks(6))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      // Y label
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -ih / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("n_P (ticks to Planck depth)");
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[700px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   D3 CHART: Enhancement Unification horizontal bar chart
   ═══════════════════════════════════════════════════════════════════════════ */
function EnhancementUnificationChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 800;
      const height = 420;
      const margin = { top: 50, right: 60, bottom: 40, left: 230 };
      const iw = width - margin.left - margin.right;
      const ih = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const data = [
        { label: "Ternary encoding", val: 3.52, color: "#334155" },
        { label: "Multi-modal synthesis", val: 5.0, color: "#334155" },
        { label: "Harmonic coincidence", val: 3.0, color: "#334155" },
        { label: "Poincare computing", val: 66.0, color: "#334155" },
        { label: "Continuous refinement", val: 43.43, color: "#334155" },
        { label: "Total (five mechanisms)", val: 120.95, color: "#F97316" },
        { label: "Composition-inflation (n=201)", val: 120.4, color: "#58E6D9" },
      ];

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 28)
        .attr("text-anchor", "middle")
        .attr("fill", "#fff")
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Enhancement Unification: Five Mechanisms → One Formula");

      const y = d3
        .scaleBand()
        .domain(data.map((d) => d.label))
        .range([0, ih])
        .padding(0.3);

      const x = d3.scaleLinear().domain([0, 130]).range([0, iw]);

      // Bars
      g.selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", 0)
        .attr("y", (d) => y(d.label))
        .attr("width", (d) => x(d.val))
        .attr("height", y.bandwidth())
        .attr("fill", (d) => d.color)
        .attr("rx", 3)
        .attr("opacity", 0.9);

      // Value labels
      g.selectAll(".val")
        .data(data)
        .enter()
        .append("text")
        .attr("x", (d) => x(d.val) + 8)
        .attr("y", (d) => y(d.label) + y.bandwidth() / 2 + 5)
        .attr("fill", (d) => d.color === "#334155" ? "#888" : d.color)
        .attr("font-size", "12px")
        .attr("font-weight", "700")
        .attr("font-family", "'Courier New', monospace")
        .text((d) => d.val.toFixed(2));

      // Y axis labels
      g.selectAll(".lab")
        .data(data)
        .enter()
        .append("text")
        .attr("x", -8)
        .attr("y", (d) => y(d.label) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", "end")
        .attr("fill", (d) => {
          if (d.color === "#58E6D9") return "#58E6D9";
          if (d.color === "#F97316") return "#F97316";
          return "#bbb";
        })
        .attr("font-size", "11px")
        .attr("font-weight", (d) =>
          d.color === "#334155" ? "400" : "700"
        )
        .attr("font-family", "Montserrat, sans-serif")
        .text((d) => d.label);

      // Bottom axis
      g.append("g")
        .attr("transform", `translate(0,${ih})`)
        .call(d3.axisBottom(x).ticks(7))
        .call((sel) => sel.select(".domain").attr("stroke", "#444"))
        .call((sel) => sel.selectAll("text").attr("fill", "#888").attr("font-size", "11px"))
        .call((sel) => sel.selectAll("line").attr("stroke", "#444"));

      g.append("text")
        .attr("x", iw / 2)
        .attr("y", ih + 35)
        .attr("text-anchor", "middle")
        .attr("fill", "#aaa")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("log₁₀ enhancement");
    });
  }, []);

  return (
    <div className="w-full overflow-x-auto">
      <svg ref={svgRef} className="w-full max-w-[800px] mx-auto" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   DERIVATION CHAIN COMPONENT (Section 9)
   ═══════════════════════════════════════════════════════════════════════════ */
const derivationSteps = [
  { id: 1, text: "Bounded Phase Space Law", sub: "axiom", cat: "axiom" },
  { id: 2, text: "Oscillatory necessity", sub: "Poincare recurrence", cat: "derived" },
  { id: 3, text: "Partition coordinates (n,l,m,s), C(n) = 2n²", sub: "", cat: "derived" },
  { id: 4, text: "Three-dimensional space, SO(3)", sub: "", cat: "derived" },
  { id: 5, text: "Partition propagation bound: c < ∞", sub: "", cat: "derived" },
  { id: 6, text: "Lorentz invariance", sub: "no light postulate", cat: "derived" },
  { id: 7, text: "Localisation rest frequency: ω₀ > 0", sub: "", cat: "derived" },
  { id: 8, text: "Inertial mass: m₀ = ℏω₀/c²", sub: "", cat: "derived" },
  { id: 9, text: "E = mc²", sub: "theorem, not postulate", cat: "derived" },
  { id: 10, text: "Composition-inflation: T(n,d) = d·(d+1)^(n−1)", sub: "", cat: "composition" },
  { id: 11, text: "Angular resolution: Δθ = 2π/T", sub: "Planck-unconstrained", cat: "composition" },
  { id: 12, text: "Angular constants: c = 2π, ℏ = E_tick/(2π)", sub: "", cat: "angular" },
  { id: 13, text: "Planck depth: n_P = 56", sub: "", cat: "angular" },
  { id: 14, text: "G is irreducible", sub: "", cat: "irreducible" },
];

const catColors = {
  axiom: { border: "#F59E0B", bg: "rgba(245,158,11,0.08)", text: "#F59E0B" },
  derived: { border: "#58E6D9", bg: "rgba(88,230,217,0.06)", text: "#58E6D9" },
  composition: { border: "#4A9EFF", bg: "rgba(74,158,255,0.06)", text: "#4A9EFF" },
  angular: { border: "#A78BFA", bg: "rgba(167,139,250,0.06)", text: "#A78BFA" },
  irreducible: { border: "#EF4444", bg: "rgba(239,68,68,0.08)", text: "#EF4444" },
};

function DerivationChain() {
  return (
    <div className="flex flex-col items-center gap-0">
      {derivationSteps.map((step, i) => {
        const colors = catColors[step.cat];
        return (
          <div key={step.id} className="flex flex-col items-center">
            {/* Connector line */}
            {i > 0 && (
              <div className="w-px h-8" style={{ background: "linear-gradient(to bottom, #333, #555)" }} />
            )}
            {/* Node */}
            <div
              className="relative px-6 py-4 rounded-lg border max-w-lg w-full text-center"
              style={{
                borderColor: colors.border,
                backgroundColor: colors.bg,
              }}
            >
              <span
                className="absolute -top-3 left-4 text-xs font-mono px-2 py-0.5 rounded"
                style={{
                  color: colors.text,
                  backgroundColor: "#0a0a0a",
                  border: `1px solid ${colors.border}`,
                }}
              >
                {step.id}
              </span>
              <p className="text-white text-sm font-semibold leading-snug">
                {step.text}
              </p>
              {step.sub && (
                <p className="text-xs mt-1" style={{ color: colors.text, opacity: 0.8 }}>
                  {step.sub}
                </p>
              )}
            </div>
          </div>
        );
      })}

      {/* Legend */}
      <div className="flex flex-wrap gap-4 mt-10 justify-center">
        {[
          { cat: "axiom", label: "Axiom" },
          { cat: "derived", label: "Derived Physics" },
          { cat: "composition", label: "Composition-Inflation" },
          { cat: "angular", label: "Angular Constants" },
          { cat: "irreducible", label: "Irreducible" },
        ].map((item) => (
          <div key={item.cat} className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: catColors[item.cat].border }}
            />
            <span className="text-neutral-500 text-xs tracking-wide">
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS TABLE (Section 8)
   ═══════════════════════════════════════════════════════════════════════════ */
const constantsData = [
  {
    constant: "c",
    name: "speed of light",
    si: "299,792,458 m/s",
    angular: "2π rad/tick",
    note: "exact",
  },
  {
    constant: "ℏ",
    name: "reduced Planck",
    si: "1.055 × 10⁻³⁴ J·s",
    angular: "E_tick / (2π)",
    note: "exact",
  },
  {
    constant: "k_B",
    name: "Boltzmann",
    si: "1.381 × 10⁻²³ J/K",
    angular: "E_tick / ln(4)",
    note: "exact",
  },
  {
    constant: "m₀",
    name: "rest mass",
    si: "ℏω₀/c²",
    angular: "E_tick · ν₀ / (4π²)",
    note: "exact",
  },
  {
    constant: "E₀",
    name: "rest energy",
    si: "m₀c²",
    angular: "E_tick · ν₀",
    note: "exact",
  },
  {
    constant: "t_P",
    name: "Planck time",
    si: "√(ℏG/c⁵)",
    angular: "√(E_tick · G) / (2π)³",
    note: "contains G",
  },
  {
    constant: "G",
    name: "gravitational",
    si: "6.674 × 10⁻¹¹ m³/(kg·s²)",
    angular: "irreducible",
    note: "irreducible",
  },
];

function ConstantsTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-neutral-700">
            <th className="text-left text-neutral-500 text-xs tracking-widest uppercase py-3 px-4">
              Constant
            </th>
            <th className="text-left text-neutral-500 text-xs tracking-widest uppercase py-3 px-4">
              SI Value
            </th>
            <th className="text-left text-neutral-500 text-xs tracking-widest uppercase py-3 px-4">
              Categorical Angular Expression
            </th>
          </tr>
        </thead>
        <tbody>
          {constantsData.map((row, i) => (
            <tr
              key={row.constant}
              className={`border-b border-neutral-800/60 ${
                row.note === "irreducible"
                  ? "bg-red-950/20"
                  : i % 2 === 0
                  ? "bg-neutral-900/30"
                  : "bg-neutral-900/10"
              }`}
            >
              <td className="py-3 px-4">
                <span className="text-white font-semibold text-sm">
                  {row.constant}
                </span>
                <span className="text-neutral-600 text-xs ml-2">
                  ({row.name})
                </span>
              </td>
              <td className="py-3 px-4 font-mono text-neutral-400 text-sm">
                {row.si}
              </td>
              <td className="py-3 px-4">
                {row.note === "irreducible" ? (
                  <span className="font-mono text-red-400 font-bold text-sm">
                    {row.angular}
                  </span>
                ) : (
                  <span className="font-mono text-[#58E6D9] text-sm">
                    {row.angular}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   GROWTH TABLE (Section 3)
   ═══════════════════════════════════════════════════════════════════════════ */
function GrowthTable() {
  const rows = [
    { n: 1, T: "3", inflation: "1x" },
    { n: 10, T: "786,432", inflation: "26,214x" },
    { n: 20, T: "~3.3 × 10¹¹", inflation: "~5.5 × 10⁹x" },
    { n: 56, T: "3.8 × 10³³", inflation: "~1.3 × 10³²x" },
    { n: 100, T: "~10⁶⁰", inflation: "~10⁵⁸x" },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse max-w-2xl mx-auto">
        <thead>
          <tr className="border-b border-neutral-700">
            <th className="text-center text-neutral-500 text-xs tracking-widest uppercase py-3 px-6">
              n (ticks)
            </th>
            <th className="text-center text-neutral-500 text-xs tracking-widest uppercase py-3 px-6">
              T(n, 3) trajectories
            </th>
            <th className="text-center text-neutral-500 text-xs tracking-widest uppercase py-3 px-6">
              Inflation over linear
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.n}
              className={`border-b border-neutral-800/60 ${
                row.n === 56
                  ? "bg-[#58E6D9]/5"
                  : i % 2 === 0
                  ? "bg-neutral-900/30"
                  : "bg-neutral-900/10"
              }`}
            >
              <td
                className={`text-center py-3 px-6 font-mono text-sm ${
                  row.n === 56 ? "text-[#58E6D9] font-bold" : "text-white"
                }`}
              >
                {row.n}
              </td>
              <td className="text-center py-3 px-6 font-mono text-neutral-300 text-sm">
                {row.T}
              </td>
              <td className="text-center py-3 px-6 font-mono text-neutral-500 text-sm">
                {row.inflation}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION DIVIDER
   ═══════════════════════════════════════════════════════════════════════════ */
function Divider() {
  return (
    <div className="max-w-5xl mx-auto px-6">
      <div className="h-px bg-neutral-800" />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN PAGE COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */
export default function Resolution() {
  useEffect(() => {
    const sections = document.querySelectorAll(".section-reveal");
    sections.forEach((section) => {
      gsap.fromTo(
        section,
        { opacity: 0, y: 60 },
        {
          opacity: 1,
          y: 0,
          duration: 1,
          scrollTrigger: {
            trigger: section,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    });
    return () => ScrollTrigger.getAll().forEach((t) => t.kill());
  }, []);

  return (
    <>
      <Head>
        <title>Resolution: 56 Ticks | Honjo Masamune</title>
        <meta
          name="description"
          content="How composition-inflation achieves Planck-time angular resolution in 56 caesium ticks. Fundamental constants dissolve into geometry."
        />
      </Head>

      <main className="min-h-screen bg-[#0a0a0a]">
        {/* ──────────────────────── SECTION 1: HERO ──────────────────────── */}
        <section className="pt-32 pb-24 px-6">
          <div className="max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <p className="text-[#58E6D9] text-sm tracking-[0.3em] uppercase mb-4">
                Resolution
              </p>
              <h1 className="text-6xl md:text-8xl font-bold text-white tracking-tight mb-4">
                56 Ticks
              </h1>
              <p className="text-xl md:text-2xl text-neutral-300 font-light mb-10 max-w-3xl leading-relaxed">
                How composition-inflation achieves Planck-time angular resolution
                without faster clocks
              </p>
              <div className="h-px bg-neutral-800 mb-10" />
              <p className="text-neutral-400 text-lg leading-relaxed max-w-3xl">
                The Planck time (5.39 &times; 10<sup>&minus;44</sup> s) is conventionally
                considered the ultimate limit of temporal measurement. We show
                this is an artifact of measuring in seconds. When resolution is
                measured in <em className="text-white not-italic font-medium">categorical trajectory count</em> &mdash;
                a dimensionless quantity &mdash; 56 oscillation cycles of
                caesium-133 suffice.
              </p>
            </motion.div>
          </div>
        </section>

        <Divider />

        {/* ──────── SECTION 2: COMPOSITION-INFLATION MECHANISM ──────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              The Composition-Inflation Mechanism
            </h2>
            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              Consider a pendulum with period <span className="font-mono text-white">3</span> ticks.
              The naive count gives 3 categorical states. But the{" "}
              <span className="text-[#58E6D9] font-semibold">integer compositions</span> of 3
              are:
            </p>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 mb-8 max-w-xl">
              <p className="font-mono text-white text-lg leading-loose tracking-wide">
                1+1+1 &nbsp;&nbsp; 2+1 &nbsp;&nbsp; 1+2 &nbsp;&nbsp; 3
              </p>
              <p className="text-neutral-500 text-sm mt-2">
                Four different temporal structures.
              </p>
            </div>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              Each step is labeled by one of three S-entropy dimensions (
              <span className="text-[#4A9EFF] font-mono">S<sub>k</sub></span>,{" "}
              <span className="text-[#34D399] font-mono">S<sub>t</sub></span>,{" "}
              <span className="text-[#F97316] font-mono">S<sub>e</sub></span>).
              The total:
            </p>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 mb-10 max-w-xl">
              <p className="font-mono text-[#58E6D9] text-2xl font-bold">
                3 &times; 4&sup2; = 48
              </p>
              <p className="text-neutral-500 text-sm mt-2">
                Not 3. Forty-eight distinguishable trajectories.
              </p>
            </div>

            <CompositionHeatmap />
          </div>
        </section>

        <Divider />

        {/* ────────────── SECTION 3: THE FORMULA ────────────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              The Formula
            </h2>

            {/* Formula display */}
            <div className="bg-neutral-900/60 border border-[#58E6D9]/20 rounded-xl px-8 py-8 mb-10 max-w-2xl mx-auto text-center">
              <p className="font-mono text-[#58E6D9] text-3xl md:text-4xl font-bold tracking-wide">
                T(n, d) = d &middot; (d+1)<sup>n&minus;1</sup>
              </p>
            </div>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              For <span className="font-mono text-white">d = 3</span> (three-dimensional S-entropy space):
            </p>

            <div className="bg-neutral-900/40 border border-neutral-800 rounded-lg px-6 py-4 mb-6 max-w-xl">
              <p className="font-mono text-white text-xl">
                T(n) = 3 &middot; 4<sup>n&minus;1</sup>
              </p>
            </div>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              This grows <span className="text-white font-semibold">exponentially</span> &mdash;
              not linearly, not quadratically. Each additional tick multiplies the
              trajectory count by 4.
            </p>

            <div className="mb-12">
              <TrajectoryGrowthChart />
            </div>

            <h3 className="text-2xl font-bold text-white tracking-tight mb-6">
              Growth Table
            </h3>
            <GrowthTable />
          </div>
        </section>

        <Divider />

        {/* ────────────── SECTION 4: ANGULAR RESOLUTION ────────────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              Angular Resolution
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              The resolution achieved is{" "}
              <span className="text-white font-semibold">angular</span>, not
              temporal.
            </p>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 mb-8 max-w-xl">
              <p className="font-mono text-[#58E6D9] text-xl font-bold">
                &Delta;&theta; = 2&pi; / T(n, d)
              </p>
              <p className="text-neutral-500 text-sm mt-2">
                Measured in <span className="text-white">radians</span> &mdash; a
                dimensionless quantity. Dimensionless quantities have no Planck
                limit.
              </p>
            </div>

            <p className="text-neutral-400 text-lg leading-relaxed mb-10 max-w-3xl">
              The Planck time constrains temporal durations measured in seconds. It
              does <span className="text-white font-semibold">not</span> constrain
              the number of distinguishable categorical trajectories, which is a
              pure count.
            </p>

            <AngularResolutionChart />

            <p className="text-neutral-500 text-sm text-center mt-4 italic">
              The curve crosses the Planck angular threshold at n = 56. The
              teal-shaded region is sub-Planck angular resolution.
            </p>
          </div>
        </section>

        <Divider />

        {/* ────────────── SECTION 5: PLANCK DEPTH ────────────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              The Planck Depth
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              The <span className="text-[#58E6D9] font-semibold">Planck depth</span>{" "}
              n<sub>P</sub> is the number of oscillation cycles needed to reach
              Planck angular resolution.
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-10">
              <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 text-center">
                <p className="text-neutral-500 text-xs tracking-widest uppercase mb-2">
                  Oscillator
                </p>
                <p className="text-white font-semibold text-lg">Caesium-133</p>
                <p className="text-neutral-600 text-sm font-mono">
                  &nu; = 9.19 &times; 10&sup9; Hz
                </p>
              </div>
              <div className="bg-neutral-900/60 border border-[#58E6D9]/30 rounded-lg px-6 py-5 text-center">
                <p className="text-neutral-500 text-xs tracking-widest uppercase mb-2">
                  Planck Depth
                </p>
                <p className="text-[#58E6D9] font-bold text-4xl">56</p>
                <p className="text-neutral-600 text-sm">ticks</p>
              </div>
              <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 text-center">
                <p className="text-neutral-500 text-xs tracking-widest uppercase mb-2">
                  Wall-clock time
                </p>
                <p className="text-white font-semibold text-lg">6.1 ns</p>
                <p className="text-neutral-600 text-sm">nanoseconds</p>
              </div>
            </div>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              Remarkably,{" "}
              <span className="text-white font-semibold">all oscillators</span>{" "}
              reach Planck depth in 48&ndash;57 ticks, regardless of frequency.
            </p>

            <div className="mb-10">
              <PlanckDepthOscillatorsChart />
            </div>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 max-w-3xl">
              <p className="text-neutral-400 text-base leading-relaxed">
                This universality follows from the logarithmic dependence{" "}
                <span className="font-mono text-white">
                  n<sub>P</sub> &prop; log<sub>4</sub>(&tau;/t<sub>P</sub>)
                </span>
                . Forty-four orders of magnitude in &tau;/t<sub>P</sub> compress into{" "}
                <span className="text-[#58E6D9] font-semibold">~9 ticks</span>{" "}
                of variation.
              </p>
            </div>
          </div>
        </section>

        <Divider />

        {/* ────────── SECTION 6: DIMENSIONAL ADVANTAGE ────────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              The Dimensional Advantage
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              The <span className="text-white font-semibold">three dimensions</span>{" "}
              of S-entropy space (
              <span className="text-[#4A9EFF] font-mono">S<sub>k</sub></span>,{" "}
              <span className="text-[#34D399] font-mono">S<sub>t</sub></span>,{" "}
              <span className="text-[#F97316] font-mono">S<sub>e</sub></span>) are
              not arbitrary &mdash; they determine the base of the exponential.
            </p>

            <div className="mb-10">
              <DimensionalAdvantageChart />
            </div>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 max-w-3xl">
              <p className="text-neutral-400 text-base leading-relaxed mb-3">
                Going from binary (<span className="font-mono text-white">d = 2</span>) to ternary (
                <span className="font-mono text-[#58E6D9]">d = 3</span>) saves{" "}
                <span className="text-white font-semibold">15 ticks</span>.
              </p>
              <p className="text-neutral-400 text-base leading-relaxed">
                The third dimension &mdash; harmonic network density (
                <span className="text-[#F97316] font-mono">S<sub>e</sub></span>)
                &mdash; provides this reduction.
              </p>
            </div>
          </div>
        </section>

        <Divider />

        {/* ──────── SECTION 7: ENHANCEMENT UNIFICATION ──────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              Enhancement Unification
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              Previously, five separate enhancement mechanisms were identified for
              trans-Planckian resolution, totaling{" "}
              <span className="font-mono text-white">
                10<sup>120.95</sup>
              </span>
              .
            </p>

            <p className="text-neutral-400 text-lg leading-relaxed mb-10 max-w-3xl">
              Composition-inflation{" "}
              <span className="text-[#58E6D9] font-semibold">unifies all five</span>{" "}
              into a single formula:
            </p>

            <div className="bg-neutral-900/60 border border-[#58E6D9]/20 rounded-xl px-8 py-6 mb-10 max-w-2xl mx-auto text-center">
              <p className="font-mono text-[#58E6D9] text-xl font-bold">
                T(201, 3) = 3 &middot; 4<sup>200</sup> &asymp; 10<sup>120.4</sup>
              </p>
            </div>

            <div className="mb-10">
              <EnhancementUnificationChart />
            </div>

            <div className="bg-neutral-900/60 border border-neutral-800 rounded-lg px-6 py-5 max-w-3xl">
              <p className="text-neutral-400 text-base leading-relaxed">
                One formula replaces five. The five mechanisms are{" "}
                <span className="text-white font-semibold">projections</span> of
                a single combinatorial process.
              </p>
            </div>
          </div>
        </section>

        <Divider />

        {/* ────── SECTION 8: ANGULAR REFORMULATION OF CONSTANTS ────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              Angular Reformulation of Fundamental Constants
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-6 max-w-3xl">
              When physics is expressed in{" "}
              <span className="text-white font-semibold">
                categorical angular units
              </span>{" "}
              &mdash; ticks for time, radians for angles, tick-energy for energy
              &mdash; six of seven fundamental constants dissolve into geometry.
            </p>

            <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-2 md:p-4 mb-10">
              <ConstantsTable />
            </div>

            <div className="space-y-6 max-w-3xl">
              <div className="bg-neutral-900/60 border-l-2 border-[#58E6D9] pl-6 py-4">
                <p className="text-neutral-300 text-base leading-relaxed">
                  The speed of light is{" "}
                  <span className="font-mono text-[#58E6D9] font-bold">
                    2&pi; radians per tick
                  </span>{" "}
                  &mdash; exact, not approximate. One tick IS one complete phase
                  cycle. The speed of light IS the angular velocity of phase.
                </p>
              </div>

              <div className="bg-neutral-900/60 border-l-2 border-[#58E6D9] pl-6 py-4">
                <p className="text-neutral-300 text-base leading-relaxed">
                  Planck&apos;s constant is energy per radian. Boltzmann&apos;s
                  constant is energy per trajectory refinement. Mass is tick energy
                  times rest frequency divided by geometry.
                </p>
              </div>

              <div className="bg-neutral-900/60 border-l-2 border-[#EF4444] pl-6 py-4">
                <p className="text-neutral-300 text-base leading-relaxed">
                  Only <span className="text-red-400 font-bold">G</span> resists.
                  The gravitational constant is the sole input from gravity into
                  the partition framework &mdash; the coupling between categorical
                  depth (mass = accumulated partition residue) and spatial
                  curvature.
                </p>
              </div>
            </div>
          </div>
        </section>

        <Divider />

        {/* ──────── SECTION 9: THE FIRST-PRINCIPLES CHAIN ──────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              The First-Principles Chain
            </h2>

            <p className="text-neutral-400 text-lg leading-relaxed mb-12 max-w-3xl">
              Fourteen steps from a single axiom to Planck depth and the
              irreducibility of G. Every step is a theorem, not a postulate.
            </p>

            <DerivationChain />
          </div>
        </section>

        <Divider />

        {/* ──────── SECTION 10: WHAT THIS DOES NOT CLAIM ──────── */}
        <section className="section-reveal py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold text-white tracking-tight mb-6">
              What This Does NOT Claim
            </h2>

            <div className="space-y-4 mb-10 max-w-3xl">
              <div className="flex items-start gap-4">
                <span className="text-red-500 font-bold text-xl mt-0.5">&times;</span>
                <p className="text-neutral-300 text-lg leading-relaxed">
                  We do <span className="font-semibold text-white">not</span> claim
                  to measure time intervals shorter than 5.39 &times;
                  10<sup>&minus;44</sup> seconds.
                </p>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-red-500 font-bold text-xl mt-0.5">&times;</span>
                <p className="text-neutral-300 text-lg leading-relaxed">
                  We do <span className="font-semibold text-white">not</span> claim
                  to violate quantum gravity.
                </p>
              </div>
              <div className="flex items-start gap-4">
                <span className="text-red-500 font-bold text-xl mt-0.5">&times;</span>
                <p className="text-neutral-300 text-lg leading-relaxed">
                  We do <span className="font-semibold text-white">not</span> claim
                  the Planck scale is wrong.
                </p>
              </div>
            </div>

            <div className="bg-neutral-900/60 border border-[#58E6D9]/20 rounded-xl px-8 py-8 max-w-3xl">
              <p className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase mb-4 font-semibold">
                We claim
              </p>
              <p className="text-neutral-300 text-lg leading-relaxed mb-4">
                The Planck time constrains temporal intervals in SI units, not
                categorical trajectory counts.
              </p>
              <p className="text-neutral-400 text-base leading-relaxed">
                The distinction is between{" "}
                <span className="text-white font-medium">
                  &ldquo;how many seconds between events A and B?&rdquo;
                </span>{" "}
                (has a Planck limit) and{" "}
                <span className="text-[#58E6D9] font-medium">
                  &ldquo;how many distinguishable trajectories separate events A
                  and B?&rdquo;
                </span>{" "}
                (has no finite limit).
              </p>
            </div>

            {/* Final visual anchor */}
            <div className="mt-24 text-center">
              <p className="text-neutral-700 text-sm tracking-[0.4em] uppercase mb-4">
                Composition-inflation depth for caesium-133
              </p>
              <p className="text-[#58E6D9] text-8xl md:text-9xl font-bold tracking-tight">
                56
              </p>
              <p className="text-neutral-600 text-lg mt-2">
                ticks &middot; 6.1 nanoseconds &middot; 3.8 &times; 10<sup>33</sup>{" "}
                trajectories
              </p>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
