import Head from "next/head";
import { useEffect, useRef, useCallback } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/dist/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// ─── D3 Chart: Bounded Trajectory (Lissajous inside a circle) ───────────────
function LissajousChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    const d3Import = import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 500;
      const height = 500;
      const cx = width / 2;
      const cy = height / 2;
      const R = 180;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      // Boundary circle
      svg
        .append("circle")
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("r", R + 10)
        .attr("fill", "none")
        .attr("stroke", "#333")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4");

      // Label: phase space boundary
      svg
        .append("text")
        .attr("x", cx)
        .attr("y", cy - R - 24)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("PHASE SPACE BOUNDARY");

      // Generate Lissajous curve points
      const points = [];
      const steps = 600;
      const a = 3, b = 2, delta = Math.PI / 4;
      for (let i = 0; i <= steps; i++) {
        const t = (i / steps) * 2 * Math.PI;
        const x = cx + R * 0.85 * Math.sin(a * t + delta);
        const y = cy + R * 0.85 * Math.sin(b * t);
        points.push([x, y]);
      }

      const line = d3
        .line()
        .x((d) => d[0])
        .y((d) => d[1])
        .curve(d3.curveBasis);

      const path = svg
        .append("path")
        .datum(points)
        .attr("d", line)
        .attr("fill", "none")
        .attr("stroke", "#58E6D9")
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.8);

      // Animate draw-on
      const totalLength = path.node().getTotalLength();
      path
        .attr("stroke-dasharray", totalLength)
        .attr("stroke-dashoffset", totalLength);

      // Moving dot
      const dot = svg
        .append("circle")
        .attr("r", 4)
        .attr("fill", "#58E6D9")
        .attr("filter", "url(#glow)")
        .attr("cx", points[0][0])
        .attr("cy", points[0][1]);

      // Glow filter
      const defs = svg.append("defs");
      const filter = defs.append("filter").attr("id", "glow");
      filter
        .append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "coloredBlur");
      const feMerge = filter.append("feMerge");
      feMerge.append("feMergeNode").attr("in", "coloredBlur");
      feMerge.append("feMergeNode").attr("in", "SourceGraphic");

      // GSAP animation for the path draw and dot movement
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: svgRef.current,
          start: "top 80%",
          once: true,
        },
      });

      tl.to(path.node(), {
        strokeDashoffset: 0,
        duration: 3,
        ease: "power2.inOut",
      });

      // Continuous dot animation
      let animFrame;
      let startTime = null;
      function animateDot(timestamp) {
        if (!startTime) startTime = timestamp;
        const elapsed = (timestamp - startTime) / 1000;
        const t = (elapsed * 0.3) % (2 * Math.PI);
        const x = cx + R * 0.85 * Math.sin(a * t + delta);
        const y = cy + R * 0.85 * Math.sin(b * t);
        dot.attr("cx", x).attr("cy", y);
        animFrame = requestAnimationFrame(animateDot);
      }

      // Start dot after path draws
      tl.call(() => {
        animFrame = requestAnimationFrame(animateDot);
      });

      return () => {
        if (animFrame) cancelAnimationFrame(animFrame);
      };
    });

    return () => {
      d3Import.then((cleanup) => {
        if (typeof cleanup === "function") cleanup();
      }).catch(() => {});
    };
  }, []);

  return (
    <div className="w-full max-w-md mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Four Alternatives Grid ────────────────────────────────────────
function AlternativesChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 700;
      const height = 320;
      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const alternatives = [
        { label: "Static", eliminated: true, icon: "flat" },
        { label: "Monotonic", eliminated: true, icon: "rising" },
        { label: "Chaotic", eliminated: true, icon: "chaotic" },
        { label: "Oscillatory", eliminated: false, icon: "wave" },
      ];

      const boxW = 140;
      const boxH = 180;
      const gap = 20;
      const totalW = alternatives.length * boxW + (alternatives.length - 1) * gap;
      const startX = (width - totalW) / 2;
      const startY = (height - boxH) / 2;

      alternatives.forEach((alt, i) => {
        const x = startX + i * (boxW + gap);
        const y = startY;
        const g = svg.append("g").attr("class", `alt-box-${i}`).attr("opacity", 0);

        // Box
        g.append("rect")
          .attr("x", x)
          .attr("y", y)
          .attr("width", boxW)
          .attr("height", boxH)
          .attr("rx", 6)
          .attr("fill", alt.eliminated ? "#111" : "none")
          .attr("stroke", alt.eliminated ? "#333" : "#58E6D9")
          .attr("stroke-width", alt.eliminated ? 1 : 2);

        // Mini chart area
        const chartX = x + 20;
        const chartY = y + 20;
        const chartW = boxW - 40;
        const chartH = 80;

        if (alt.icon === "flat") {
          g.append("line")
            .attr("x1", chartX)
            .attr("y1", chartY + chartH / 2)
            .attr("x2", chartX + chartW)
            .attr("y2", chartY + chartH / 2)
            .attr("stroke", alt.eliminated ? "#555" : "#58E6D9")
            .attr("stroke-width", 2);
        } else if (alt.icon === "rising") {
          g.append("line")
            .attr("x1", chartX)
            .attr("y1", chartY + chartH)
            .attr("x2", chartX + chartW)
            .attr("y2", chartY)
            .attr("stroke", alt.eliminated ? "#555" : "#58E6D9")
            .attr("stroke-width", 2);
        } else if (alt.icon === "chaotic") {
          const chaosLine = d3.line().curve(d3.curveBasis);
          const pts = [];
          for (let j = 0; j <= 20; j++) {
            pts.push([
              chartX + (j / 20) * chartW,
              chartY + chartH / 2 + (Math.random() - 0.5) * chartH * 0.9,
            ]);
          }
          g.append("path")
            .attr("d", chaosLine(pts))
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-width", 2);
        } else if (alt.icon === "wave") {
          const waveLine = d3.line().curve(d3.curveBasis);
          const pts = [];
          for (let j = 0; j <= 40; j++) {
            pts.push([
              chartX + (j / 40) * chartW,
              chartY + chartH / 2 + Math.sin((j / 40) * 4 * Math.PI) * chartH * 0.4,
            ]);
          }
          g.append("path")
            .attr("d", waveLine(pts))
            .attr("fill", "none")
            .attr("stroke", "#58E6D9")
            .attr("stroke-width", 2);
        }

        // Label
        g.append("text")
          .attr("x", x + boxW / 2)
          .attr("y", y + boxH - 30)
          .attr("text-anchor", "middle")
          .attr("fill", alt.eliminated ? "#555" : "#58E6D9")
          .attr("font-size", "13px")
          .attr("font-family", "Montserrat, sans-serif")
          .attr("font-weight", alt.eliminated ? "400" : "600")
          .attr("letter-spacing", "0.1em")
          .text(alt.label.toUpperCase());

        // Strike-through for eliminated
        if (alt.eliminated) {
          g.append("line")
            .attr("x1", x + 10)
            .attr("y1", y + 10)
            .attr("x2", x + boxW - 10)
            .attr("y2", y + boxH - 10)
            .attr("stroke", "#ef4444")
            .attr("stroke-width", 2)
            .attr("stroke-opacity", 0.6);

          g.append("line")
            .attr("x1", x + boxW - 10)
            .attr("y1", y + 10)
            .attr("x2", x + 10)
            .attr("y2", y + boxH - 10)
            .attr("stroke", "#ef4444")
            .attr("stroke-width", 2)
            .attr("stroke-opacity", 0.6);
        }

        // GSAP reveal
        gsap.to(`.alt-box-${i}`, {
          opacity: 1,
          duration: 0.6,
          delay: i * 0.15,
          scrollTrigger: {
            trigger: svgRef.current,
            start: "top 80%",
            once: true,
          },
        });
      });
    });
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Shell Capacities Bar Chart ────────────────────────────────────
function ShellCapacityChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const margin = { top: 40, right: 30, bottom: 50, left: 60 };
      const width = 600;
      const height = 350;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const data = [
        { n: 1, C: 2 },
        { n: 2, C: 8 },
        { n: 3, C: 18 },
        { n: 4, C: 32 },
        { n: 5, C: 50 },
        { n: 6, C: 72 },
        { n: 7, C: 98 },
      ];

      const x = d3
        .scaleBand()
        .domain(data.map((d) => d.n))
        .range([0, innerW])
        .padding(0.3);

      const y = d3
        .scaleLinear()
        .domain([0, 105])
        .range([innerH, 0]);

      // Grid lines
      g.append("g")
        .attr("class", "grid")
        .selectAll("line")
        .data(y.ticks(5))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", innerW)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#1a1a1a")
        .attr("stroke-width", 1);

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x).tickFormat((d) => `n=${d}`))
        .attr("class", "axis")
        .selectAll("text")
        .attr("fill", "#666")
        .attr("font-size", "11px");

      g.selectAll(".axis path, .axis line").attr("stroke", "#333");

      // Y axis
      g.append("g")
        .call(d3.axisLeft(y).ticks(5))
        .attr("class", "axis")
        .selectAll("text")
        .attr("fill", "#666")
        .attr("font-size", "11px");

      g.selectAll(".axis path, .axis line").attr("stroke", "#333");

      // Y label
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerH / 2)
        .attr("y", -45)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("C(n) = 2n\u00B2");

      // Bars
      const bars = g
        .selectAll(".bar")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", (d) => x(d.n))
        .attr("width", x.bandwidth())
        .attr("y", innerH)
        .attr("height", 0)
        .attr("fill", "#58E6D9")
        .attr("fill-opacity", 0.8)
        .attr("rx", 2);

      // Value labels
      const labels = g
        .selectAll(".val-label")
        .data(data)
        .enter()
        .append("text")
        .attr("x", (d) => x(d.n) + x.bandwidth() / 2)
        .attr("y", innerH)
        .attr("text-anchor", "middle")
        .attr("fill", "#58E6D9")
        .attr("font-size", "12px")
        .attr("font-weight", "600")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("opacity", 0)
        .text((d) => d.C);

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("SHELL CAPACITIES C(n) = 2n\u00B2");

      // Animate bars on scroll
      ScrollTrigger.create({
        trigger: svgRef.current,
        start: "top 80%",
        once: true,
        onEnter: () => {
          bars
            .transition()
            .duration(800)
            .delay((d, i) => i * 100)
            .attr("y", (d) => y(d.C))
            .attr("height", (d) => innerH - y(d.C));

          labels
            .transition()
            .duration(800)
            .delay((d, i) => i * 100 + 400)
            .attr("y", (d) => y(d.C) - 8)
            .attr("opacity", 1);
        },
      });
    });
  }, []);

  return (
    <div className="w-full max-w-xl mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Triple Equivalence Converging Curves ──────────────────────────
function TripleEquivalenceChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const margin = { top: 50, right: 100, bottom: 50, left: 60 };
      const width = 650;
      const height = 350;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      // Generate three curves that converge
      const steps = 50;
      const curves = [
        {
          name: "S_osc",
          label: "S\u2092\u209B\u209C",
          color: "#58E6D9",
          offset: 0.35,
          decay: 0.08,
        },
        {
          name: "S_cat",
          label: "S\u209C\u2090\u209C",
          color: "#a78bfa",
          offset: -0.3,
          decay: 0.07,
        },
        {
          name: "S_part",
          label: "S\u209A\u2090\u2099\u209C",
          color: "#f59e0b",
          offset: 0.2,
          decay: 0.09,
        },
      ];

      const convergenceValue = 1.0;

      const allData = curves.map((c) => {
        const pts = [];
        for (let i = 0; i <= steps; i++) {
          const t = i / steps;
          const deviation =
            c.offset * Math.exp(-c.decay * i) * Math.cos(t * Math.PI * 3);
          pts.push({ t, v: convergenceValue + deviation });
        }
        return { ...c, points: pts };
      });

      const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const y = d3.scaleLinear().domain([0.4, 1.6]).range([innerH, 0]);

      // Grid
      g.append("g")
        .selectAll("line")
        .data(y.ticks(4))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", innerW)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#1a1a1a");

      // Convergence line
      g.append("line")
        .attr("x1", 0)
        .attr("x2", innerW)
        .attr("y1", y(convergenceValue))
        .attr("y2", y(convergenceValue))
        .attr("stroke", "#333")
        .attr("stroke-dasharray", "6,4")
        .attr("stroke-width", 1);

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${innerH})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(5)
            .tickFormat((d) => (d === 0 ? "" : d === 1 ? "n\u2192\u221E" : ""))
        )
        .attr("class", "axis");

      g.selectAll(".axis path, .axis line").attr("stroke", "#333");
      g.selectAll(".axis text").attr("fill", "#666").attr("font-size", "11px");

      // X label
      g.append("text")
        .attr("x", innerW / 2)
        .attr("y", innerH + 40)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("increasing partition depth");

      const line = d3
        .line()
        .x((d) => x(d.t))
        .y((d) => y(d.v))
        .curve(d3.curveBasis);

      // Draw curves
      allData.forEach((curve) => {
        const path = g
          .append("path")
          .datum(curve.points)
          .attr("d", line)
          .attr("fill", "none")
          .attr("stroke", curve.color)
          .attr("stroke-width", 2)
          .attr("stroke-opacity", 0.85);

        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength)
          .attr("stroke-dashoffset", totalLength);

        // Legend label at end of curve
        g.append("text")
          .attr("x", innerW + 8)
          .attr("y", y(curve.points[steps].v))
          .attr("fill", curve.color)
          .attr("font-size", "12px")
          .attr("font-family", "Montserrat, sans-serif")
          .attr("font-weight", "600")
          .attr("dominant-baseline", "middle")
          .text(curve.label);

        // GSAP draw animation
        gsap.to(path.node(), {
          strokeDashoffset: 0,
          duration: 2,
          ease: "power2.inOut",
          scrollTrigger: {
            trigger: svgRef.current,
            start: "top 80%",
            once: true,
          },
        });
      });

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 25)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("THREE DESCRIPTIONS CONVERGE");
    });
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Virtual Spectrometer Frequency Ranges ─────────────────────────
function SpectrometerChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const margin = { top: 40, right: 30, bottom: 60, left: 150 };
      const width = 700;
      const height = 300;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const data = [
        { modality: "LED Emission", freqLow: 4.3e14, freqHigh: 7.5e14, logLow: 14.63, logHigh: 14.88, color: "#58E6D9" },
        { modality: "CPU Clock", freqLow: 1e9, freqHigh: 5e9, logLow: 9.0, logHigh: 9.7, color: "#a78bfa" },
        { modality: "DRAM Refresh", freqLow: 1e6, freqHigh: 4e9, logLow: 6.0, logHigh: 9.6, color: "#f59e0b" },
        { modality: "Bus I/O", freqLow: 1e7, freqHigh: 1.6e10, logLow: 7.0, logHigh: 10.2, color: "#ef4444" },
      ];

      const y = d3
        .scaleBand()
        .domain(data.map((d) => d.modality))
        .range([0, innerH])
        .padding(0.35);

      const x = d3.scaleLinear().domain([5, 16]).range([0, innerW]);

      // Grid
      g.append("g")
        .selectAll("line")
        .data(x.ticks(6))
        .enter()
        .append("line")
        .attr("x1", (d) => x(d))
        .attr("x2", (d) => x(d))
        .attr("y1", 0)
        .attr("y2", innerH)
        .attr("stroke", "#1a1a1a");

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${innerH})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(6)
            .tickFormat((d) => `10^${d}`)
        )
        .attr("class", "axis");

      g.selectAll(".axis path, .axis line").attr("stroke", "#333");
      g.selectAll(".axis text").attr("fill", "#666").attr("font-size", "11px");

      // X label
      g.append("text")
        .attr("x", innerW / 2)
        .attr("y", innerH + 45)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Frequency (Hz, log scale)");

      // Y axis labels
      g.append("g")
        .call(d3.axisLeft(y).tickSize(0))
        .attr("class", "axis")
        .selectAll("text")
        .attr("fill", "#999")
        .attr("font-size", "12px")
        .attr("font-family", "Montserrat, sans-serif");

      g.selectAll(".axis .domain").attr("stroke", "none");

      // Bars (frequency ranges)
      data.forEach((d, i) => {
        const bar = g
          .append("rect")
          .attr("x", x(d.logLow))
          .attr("y", y(d.modality))
          .attr("height", y.bandwidth())
          .attr("width", 0)
          .attr("fill", d.color)
          .attr("fill-opacity", 0.7)
          .attr("rx", 3);

        gsap.to(bar.node(), {
          attr: { width: x(d.logHigh) - x(d.logLow) },
          duration: 1,
          delay: i * 0.15,
          ease: "power2.out",
          scrollTrigger: {
            trigger: svgRef.current,
            start: "top 80%",
            once: true,
          },
        });
      });

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("COMPUTATIONAL OSCILLATOR MODALITIES");
    });
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Element Derivation Grouped Bar Chart ──────────────────────────
function ElementDerivationChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const margin = { top: 50, right: 30, bottom: 60, left: 60 };
      const width = 750;
      const height = 400;
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const elements = ["H", "C", "Na", "Si", "Cl", "Ar", "Ca", "Fe", "Gd"];
      const derived = [13.606, 11.3, 5.12, 8.15, 12.97, 15.76, 6.11, 7.9, 6.15];
      const nist = [13.598, 11.26, 5.139, 8.152, 12.968, 15.76, 6.113, 7.902, 6.15];

      const data = elements.map((el, i) => ({
        element: el,
        derived: derived[i],
        nist: nist[i],
      }));

      const x0 = d3
        .scaleBand()
        .domain(elements)
        .range([0, innerW])
        .padding(0.25);

      const x1 = d3
        .scaleBand()
        .domain(["derived", "nist"])
        .range([0, x0.bandwidth()])
        .padding(0.1);

      const y = d3.scaleLinear().domain([0, 18]).range([innerH, 0]);

      // Grid
      g.append("g")
        .selectAll("line")
        .data(y.ticks(6))
        .enter()
        .append("line")
        .attr("x1", 0)
        .attr("x2", innerW)
        .attr("y1", (d) => y(d))
        .attr("y2", (d) => y(d))
        .attr("stroke", "#1a1a1a");

      // X axis
      g.append("g")
        .attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x0))
        .attr("class", "axis");

      g.selectAll(".axis path, .axis line").attr("stroke", "#333");
      g.selectAll(".axis text")
        .attr("fill", "#999")
        .attr("font-size", "12px")
        .attr("font-weight", "600");

      // Y axis
      const yAxis = g
        .append("g")
        .call(d3.axisLeft(y).ticks(6))
        .attr("class", "axis");

      yAxis.selectAll("path, line").attr("stroke", "#333");
      yAxis.selectAll("text").attr("fill", "#666").attr("font-size", "11px");

      // Y label
      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -innerH / 2)
        .attr("y", -45)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .text("Ionization Energy (eV)");

      // Bars
      const groups = g
        .selectAll(".element-group")
        .data(data)
        .enter()
        .append("g")
        .attr("transform", (d) => `translate(${x0(d.element)},0)`);

      // Derived bars
      const derivedBars = groups
        .append("rect")
        .attr("x", x1("derived"))
        .attr("width", x1.bandwidth())
        .attr("y", innerH)
        .attr("height", 0)
        .attr("fill", "#58E6D9")
        .attr("fill-opacity", 0.85)
        .attr("rx", 2);

      // NIST bars
      const nistBars = groups
        .append("rect")
        .attr("x", x1("nist"))
        .attr("width", x1.bandwidth())
        .attr("y", innerH)
        .attr("height", 0)
        .attr("fill", "#a78bfa")
        .attr("fill-opacity", 0.85)
        .attr("rx", 2);

      // Legend
      const legend = svg
        .append("g")
        .attr("transform", `translate(${width - 180}, ${margin.top - 10})`);

      [
        { label: "Derived (BMD)", color: "#58E6D9" },
        { label: "NIST Reference", color: "#a78bfa" },
      ].forEach((item, i) => {
        legend
          .append("rect")
          .attr("x", 0)
          .attr("y", i * 20)
          .attr("width", 12)
          .attr("height", 12)
          .attr("rx", 2)
          .attr("fill", item.color)
          .attr("fill-opacity", 0.85);

        legend
          .append("text")
          .attr("x", 18)
          .attr("y", i * 20 + 10)
          .attr("fill", "#999")
          .attr("font-size", "11px")
          .attr("font-family", "Montserrat, sans-serif")
          .text(item.label);
      });

      // Title
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 25)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("FIRST-PRINCIPLES vs NIST IONIZATION ENERGIES");

      // Animate on scroll
      ScrollTrigger.create({
        trigger: svgRef.current,
        start: "top 80%",
        once: true,
        onEnter: () => {
          derivedBars
            .transition()
            .duration(800)
            .delay((d, i) => i * 80)
            .attr("y", (d) => y(d.derived))
            .attr("height", (d) => innerH - y(d.derived));

          nistBars
            .transition()
            .duration(800)
            .delay((d, i) => i * 80 + 100)
            .attr("y", (d) => y(d.nist))
            .attr("height", (d) => innerH - y(d.nist));
        },
      });
    });
  }, []);

  return (
    <div className="w-full max-w-3xl mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── D3 Chart: Molecular Network (H2O) ──────────────────────────────────────
function MolecularNetworkChart() {
  const svgRef = useRef(null);

  useEffect(() => {
    import("d3").then((d3) => {
      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = 500;
      const height = 400;
      const cx = width / 2;
      const cy = height / 2;

      svg
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet");

      // Glow filter
      const defs = svg.append("defs");
      const filter = defs.append("filter").attr("id", "nodeGlow");
      filter
        .append("feGaussianBlur")
        .attr("stdDeviation", "6")
        .attr("result", "coloredBlur");
      const feMerge = filter.append("feMerge");
      feMerge.append("feMergeNode").attr("in", "coloredBlur");
      feMerge.append("feMergeNode").attr("in", "SourceGraphic");

      // Arrow marker
      defs
        .append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 0 10 6")
        .attr("refX", 8)
        .attr("refY", 3)
        .attr("markerWidth", 8)
        .attr("markerHeight", 5)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,0 L10,3 L0,6 Z")
        .attr("fill", "#58E6D9")
        .attr("fill-opacity", 0.5);

      const nodes = [
        { id: "\u03BD\u2081", label: "\u03BD\u2081 (sym. stretch)", x: cx, y: cy - 120, freq: "3657 cm\u207B\u00B9" },
        { id: "\u03BD\u2082", label: "\u03BD\u2082 (bend)", x: cx - 140, y: cy + 80, freq: "1595 cm\u207B\u00B9" },
        { id: "\u03BD\u2083", label: "\u03BD\u2083 (asym. stretch)", x: cx + 140, y: cy + 80, freq: "3756 cm\u207B\u00B9" },
      ];

      const edges = [
        { source: 0, target: 1 },
        { source: 1, target: 2 },
        { source: 2, target: 0 },
      ];

      const g = svg.append("g");

      // Title
      svg
        .append("text")
        .attr("x", cx)
        .attr("y", 30)
        .attr("text-anchor", "middle")
        .attr("fill", "#555")
        .attr("font-size", "11px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.15em")
        .text("H\u2082O HARMONIC NETWORK");

      // Draw edges as curved paths
      edges.forEach((e, i) => {
        const s = nodes[e.source];
        const t = nodes[e.target];
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        // Offset midpoint toward center for curve
        const cmx = mx + (cx - mx) * 0.3;
        const cmy = my + (cy - my) * 0.3;

        const path = g
          .append("path")
          .attr("d", `M${s.x},${s.y} Q${cmx},${cmy} ${t.x},${t.y}`)
          .attr("fill", "none")
          .attr("stroke", "#58E6D9")
          .attr("stroke-width", 1.5)
          .attr("stroke-opacity", 0.4)
          .attr("marker-end", "url(#arrowhead)");

        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength)
          .attr("stroke-dashoffset", totalLength);

        gsap.to(path.node(), {
          strokeDashoffset: 0,
          duration: 1.2,
          delay: 0.5 + i * 0.3,
          ease: "power2.inOut",
          scrollTrigger: {
            trigger: svgRef.current,
            start: "top 80%",
            once: true,
          },
        });
      });

      // Draw nodes
      nodes.forEach((n, i) => {
        const nodeG = g
          .append("g")
          .attr("transform", `translate(${n.x},${n.y})`)
          .attr("opacity", 0);

        // Outer glow
        nodeG
          .append("circle")
          .attr("r", 28)
          .attr("fill", "#58E6D9")
          .attr("fill-opacity", 0.08)
          .attr("filter", "url(#nodeGlow)");

        // Inner circle
        nodeG
          .append("circle")
          .attr("r", 22)
          .attr("fill", "#111")
          .attr("stroke", "#58E6D9")
          .attr("stroke-width", 2);

        // Mode label
        nodeG
          .append("text")
          .attr("y", 5)
          .attr("text-anchor", "middle")
          .attr("fill", "#58E6D9")
          .attr("font-size", "14px")
          .attr("font-weight", "700")
          .attr("font-family", "Montserrat, sans-serif")
          .text(n.id);

        // Frequency label below
        nodeG
          .append("text")
          .attr("y", 45)
          .attr("text-anchor", "middle")
          .attr("fill", "#666")
          .attr("font-size", "10px")
          .attr("font-family", "Montserrat, sans-serif")
          .text(n.freq);

        gsap.to(nodeG.node(), {
          opacity: 1,
          duration: 0.6,
          delay: i * 0.2,
          scrollTrigger: {
            trigger: svgRef.current,
            start: "top 80%",
            once: true,
          },
        });
      });

      // Circulating light indicator
      const circLabel = svg
        .append("text")
        .attr("x", cx)
        .attr("y", height - 20)
        .attr("text-anchor", "middle")
        .attr("fill", "#444")
        .attr("font-size", "10px")
        .attr("font-family", "Montserrat, sans-serif")
        .attr("letter-spacing", "0.1em")
        .text("CIRCULATING LIGHT \u2192 CLOSED ABSORPTION-EMISSION LOOP");
    });
  }, []);

  return (
    <div className="w-full max-w-md mx-auto">
      <svg ref={svgRef} className="w-full h-auto d3-chart" />
    </div>
  );
}

// ─── Section Wrapper with GSAP scroll animation ─────────────────────────────
function Section({ children, className = "" }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    gsap.fromTo(
      el,
      { opacity: 0, y: 60 },
      {
        opacity: 1,
        y: 0,
        duration: 1,
        ease: "power3.out",
        scrollTrigger: {
          trigger: el,
          start: "top 85%",
          once: true,
        },
      }
    );
  }, []);

  return (
    <section ref={ref} className={`opacity-0 ${className}`}>
      {children}
    </section>
  );
}

// ─── Stat Card for Key Results ───────────────────────────────────────────────
function StatCard({ value, label, sublabel }) {
  return (
    <div className="border border-neutral-800 rounded-lg p-6 text-center bg-[#111]/50">
      <div className="text-3xl font-bold text-[#58E6D9] tracking-wide mb-2">
        {value}
      </div>
      <div className="text-neutral-300 text-sm font-medium tracking-wider uppercase mb-1">
        {label}
      </div>
      {sublabel && (
        <div className="text-neutral-600 text-xs tracking-wide">{sublabel}</div>
      )}
    </div>
  );
}

// ─── Main Framework Page ─────────────────────────────────────────────────────
export default function Framework() {
  useEffect(() => {
    // Refresh ScrollTrigger after mount to catch all elements
    ScrollTrigger.refresh();
    return () => {
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, []);

  return (
    <>
      <Head>
        <title>Framework | Honjo Masamune</title>
        <meta
          name="description"
          content="From axiom to validation: the complete logical chain deriving atomic structure, spectroscopy, and molecular topology from a single postulate about bounded phase space."
        />
      </Head>

      <main className="min-h-screen bg-[#0a0a0a]">
        {/* ── Hero / Epigraph ── */}
        <div className="pt-40 pb-24 px-6">
          <div className="max-w-5xl mx-auto text-center">
            <h1 className="text-4xl md:text-3xl sm:text-2xl font-bold text-white tracking-[0.2em] uppercase mb-8">
              The Framework
            </h1>
            <div className="w-16 h-[1px] bg-[#58E6D9] mx-auto mb-8" />
            <p className="text-neutral-500 text-lg md:text-base max-w-2xl mx-auto leading-relaxed">
              A single axiom about bounded dynamical systems yields, without
              adjustable parameters, the shell structure of atoms, a
              first-principles spectrometer, and exact molecular topology.
            </p>
            <p className="text-neutral-600 text-sm mt-6 tracking-wider">
              AXIOM &rarr; OSCILLATION &rarr; PARTITION &rarr; EQUIVALENCE
              &rarr; INSTRUMENT &rarr; ELEMENTS &rarr; MOLECULES
            </p>
          </div>
        </div>

        {/* ── Section 1: The Axiom ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                I &mdash; Foundation
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              The Axiom
            </h2>
            <blockquote className="border-l-2 border-[#58E6D9] pl-6 mb-10">
              <p className="text-neutral-300 text-lg md:text-base italic leading-relaxed">
                &ldquo;Every persistent dynamical system occupies a bounded
                region of phase space admitting partition and nesting.&rdquo;
              </p>
            </blockquote>
            <div className="grid grid-cols-2 md:grid-cols-1 gap-12 items-center">
              <div>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  This is the sole postulate. A system that persists &mdash;
                  that does not fly apart or collapse to a point &mdash; must
                  inhabit a bounded region of its phase space. The region is not
                  featureless: it admits partition into nested subregions,
                  because any bounded set in a metric space can be subdivided
                  while preserving containment.
                </p>
                <p className="text-neutral-400 leading-relaxed">
                  The axiom says nothing about quantum mechanics, nothing about
                  Coulomb forces, nothing about wavefunctions. It is a statement
                  about dynamical persistence in phase space. Everything that
                  follows &mdash; shell structure, the periodic table, molecular
                  topology &mdash; is a theorem of this single sentence.
                </p>
              </div>
              <LissajousChart />
            </div>
          </div>
        </Section>

        {/* ── Section 2: Oscillatory Necessity ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                II &mdash; Elimination
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              Oscillatory Necessity
            </h2>
            <div className="mb-12">
              <p className="text-neutral-400 leading-relaxed mb-6">
                Given boundedness, what kinds of dynamics are possible? Four
                exhaustive alternatives present themselves: the trajectory can be
                static, monotonic, chaotic, or oscillatory.
              </p>
              <p className="text-neutral-400 leading-relaxed mb-6">
                Static trajectories carry no information and cannot sustain
                persistent structure. Monotonic trajectories eventually violate
                boundedness &mdash; a state variable that only increases must
                eventually exit any finite region. Chaotic trajectories, while
                bounded, admit no stable partition: neighbouring initial
                conditions diverge exponentially, destroying the nested
                sub-structure the axiom demands.
              </p>
              <p className="text-neutral-400 leading-relaxed">
                Only oscillatory dynamics is consistent with both boundedness
                and partition. By Poincar&eacute; recurrence, a bounded
                measure-preserving system must return arbitrarily close to its
                initial state &mdash; this is oscillation. The partition
                coordinate system then follows as a natural indexing of the
                nested oscillatory sub-regions.
              </p>
            </div>
            <AlternativesChart />
          </div>
        </Section>

        {/* ── Section 3: The Partition Coordinate System ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                III &mdash; Structure
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              The Partition Coordinate System
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-1 gap-12 items-start">
              <div>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  The bounded oscillatory region admits a canonical partition
                  indexed by four coordinates: a principal depth{" "}
                  <span className="text-white font-mono">n</span>, an angular
                  partition number{" "}
                  <span className="text-white font-mono">l</span>, a magnetic
                  orientation{" "}
                  <span className="text-white font-mono">m</span>, and a binary
                  parity{" "}
                  <span className="text-white font-mono">s</span>. These are
                  not postulated quantum numbers &mdash; they emerge from
                  counting the independent ways a bounded oscillatory region
                  partitions.
                </p>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  The capacity of the{" "}
                  <span className="text-white font-mono">n</span>-th shell
                  follows by combinatorial enumeration:
                </p>
                <div className="bg-[#111] border border-neutral-800 rounded-lg p-4 mb-6 text-center">
                  <span className="text-[#58E6D9] font-mono text-lg">
                    C(n) = 2n&sup2;
                  </span>
                </div>
                <p className="text-neutral-400 leading-relaxed">
                  This is exactly the Aufbau capacity of atomic shells.
                  The numbers 2, 8, 18, 32 are not inputs &mdash; they
                  are outputs of the partition counting. No physics beyond the
                  axiom has been invoked.
                </p>
              </div>
              <ShellCapacityChart />
            </div>
          </div>
        </Section>

        {/* ── Section 4: The Triple Equivalence ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                IV &mdash; Unification
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              The Triple Equivalence
            </h2>
            <div className="mb-12">
              <p className="text-neutral-400 leading-relaxed mb-6">
                The partition coordinate system yields three distinct but
                equivalent descriptions of the same bounded oscillatory
                object:
              </p>
              <div className="grid grid-cols-3 md:grid-cols-1 gap-6 mb-8">
                <div className="bg-[#111] border border-neutral-800 rounded-lg p-5">
                  <div className="text-[#58E6D9] font-mono text-sm mb-2 tracking-wider">
                    S_osc
                  </div>
                  <p className="text-neutral-500 text-sm leading-relaxed">
                    The oscillatory description: a superposition of
                    characteristic frequencies &omega;&#8342; with bounded
                    amplitudes.
                  </p>
                </div>
                <div className="bg-[#111] border border-neutral-800 rounded-lg p-5">
                  <div className="text-[#a78bfa] font-mono text-sm mb-2 tracking-wider">
                    S_cat
                  </div>
                  <p className="text-neutral-500 text-sm leading-relaxed">
                    The categorical description: a diagram in a category whose
                    objects are shells and morphisms are inter-shell
                    transitions.
                  </p>
                </div>
                <div className="bg-[#111] border border-neutral-800 rounded-lg p-5">
                  <div className="text-[#f59e0b] font-mono text-sm mb-2 tracking-wider">
                    S_part
                  </div>
                  <p className="text-neutral-500 text-sm leading-relaxed">
                    The partition description: a combinatorial structure
                    counting occupied cells in the nested hierarchy.
                  </p>
                </div>
              </div>
              <p className="text-neutral-400 leading-relaxed">
                The Triple Equivalence Theorem states that these three
                descriptions are isomorphic:{" "}
                <span className="text-white font-mono">
                  S&#8338;&#8347;&#8356; &cong; S&#8356;&#8336;&#8348; &cong;
                  S&#8346;&#8336;&#8348;&#8348;
                </span>
                . Any property computable in one description is computable in
                the others. This is not a claim of approximation &mdash; it is
                exact structural identity. Classical and quantum descriptions
                are not alternatives; they are the same object viewed through
                different funnels of the equivalence.
              </p>
            </div>
            <TripleEquivalenceChart />
          </div>
        </Section>

        {/* ── Section 5: The Virtual Spectrometer ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                V &mdash; Instrument
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              The Virtual Spectrometer
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-1 gap-12 items-start">
              <div>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  The Triple Equivalence implies that any physical system
                  satisfying the axiom &mdash; any bounded oscillatory system
                  with partition structure &mdash; is, in the categorical
                  sense, a spectrometer. The key insight: a digital computer is
                  such a system.
                </p>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  A CPU clock oscillates at GHz frequencies. DRAM refresh
                  cycles constitute MHz&ndash;GHz oscillations. LED indicator
                  emission occurs at ~10&sup1;&sup4; Hz. Bus I/O signals
                  oscillate across a broad band. Each of these is a physical
                  oscillator inhabiting a bounded region of its phase space.
                </p>
                <p className="text-neutral-400 leading-relaxed">
                  The computer is not simulating a spectrometer. It{" "}
                  <em className="text-neutral-300">is</em> a spectrometer,
                  by virtue of its physical oscillatory subsystems. The ESDVS
                  (Electronic Spectral Determination via Virtual
                  Spectrometry) framework makes this correspondence
                  mathematically rigorous: computational oscillators sample
                  the same categorical structure as laboratory instruments.
                </p>
              </div>
              <SpectrometerChart />
            </div>
          </div>
        </Section>

        {/* ── Section 6: Element Derivation ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                VI &mdash; Validation
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              Element Derivation
            </h2>
            <div className="mb-12">
              <p className="text-neutral-400 leading-relaxed mb-6">
                Nine elements spanning the periodic table &mdash; from hydrogen
                to gadolinium &mdash; are derived from first principles. For
                each, the framework predicts the full electron configuration
                and the first ionization energy. No empirical constants are
                fitted: the derivation uses only the axiom, the partition
                coordinate system, and the resulting shell capacity formula.
              </p>
              <p className="text-neutral-400 leading-relaxed mb-6">
                The refined ionization energies, computed via the BMD
                (Bounded Molecular Dynamics) shielding model, match NIST
                reference values to within experimental uncertainty in most
                cases. The chart below shows the comparison. The derived
                values (teal) and NIST reference values (violet) are visually
                indistinguishable at this scale &mdash; because they
                numerically agree.
              </p>
            </div>
            <ElementDerivationChart />
            <div className="mt-8 text-center">
              <p className="text-neutral-600 text-xs tracking-wider">
                9 / 9 ELECTRON CONFIGURATIONS EXACTLY REPRODUCED &middot; ZERO
                ADJUSTABLE PARAMETERS
              </p>
            </div>
          </div>
        </Section>

        {/* ── Section 7: Molecular Networks ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                VII &mdash; Topology
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-8">
              Molecular Networks
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-1 gap-12 items-center">
              <div>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  Molecules are not static ball-and-stick structures. In the
                  BMD framework, a molecule is a{" "}
                  <em className="text-neutral-300">harmonic network</em>:
                  a graph whose nodes are vibrational normal modes and whose
                  edges represent resonant energy transfer between modes.
                </p>
                <p className="text-neutral-400 leading-relaxed mb-6">
                  Light circulates through this network via an
                  absorption-emission tick: mode &nu;&#7522; absorbs a photon,
                  redistributes energy to a harmonically proximate mode
                  &nu;&#11388;, which re-emits. If the graph is closed &mdash;
                  if the sequence of ticks returns to the starting mode &mdash;
                  the molecule has a well-defined circulation period &tau;.
                </p>
                <p className="text-neutral-400 leading-relaxed">
                  For H&#8322;O, the three normal modes (&nu;&#8321; symmetric
                  stretch, &nu;&#8322; bend, &nu;&#8323; asymmetric stretch)
                  form a triangular closed loop. The predicted circulation
                  period matches the measured infrared absorption profile
                  exactly. Six molecules have been validated this way, with 11
                  out of 11 circulation periods reproduced.
                </p>
              </div>
              <MolecularNetworkChart />
            </div>
          </div>
        </Section>

        {/* ── Section 8: Key Results ── */}
        <Section className="py-32 px-6">
          <div className="max-w-5xl mx-auto">
            <div className="mb-6">
              <span className="text-[#58E6D9] text-xs tracking-[0.3em] uppercase font-medium">
                VIII &mdash; Summary
              </span>
            </div>
            <h2 className="text-3xl md:text-2xl font-bold text-white tracking-wider mb-12">
              Key Results
            </h2>
            <div className="grid grid-cols-4 md:grid-cols-2 sm:grid-cols-1 gap-6 mb-16">
              <StatCard
                value="9"
                label="Elements Derived"
                sublabel="9/9 exact configurations"
              />
              <StatCard
                value="6"
                label="Molecules Validated"
                sublabel="11/11 circulation periods"
              />
              <StatCard
                value="39"
                label="Compounds"
                sublabel="Categorical database"
              />
              <StatCard
                value="0"
                label="Adjustable Parameters"
                sublabel="Pure first principles"
              />
            </div>
            <div className="border-t border-neutral-800 pt-12">
              <p className="text-neutral-400 leading-relaxed max-w-3xl">
                From a single axiom about bounded phase space, the framework
                derives the periodic table&rsquo;s shell structure, constructs
                a physically grounded virtual spectrometer, reproduces
                ionization energies to experimental accuracy, and predicts
                molecular vibrational topology &mdash; all without adjustable
                parameters. The logical chain is unbroken: axiom to
                oscillation, oscillation to partition, partition to
                equivalence, equivalence to instrument, instrument to element,
                element to molecule.
              </p>
              <div className="mt-8 flex items-center gap-4">
                <div className="w-12 h-[1px] bg-[#58E6D9]" />
                <span className="text-neutral-600 text-xs tracking-[0.2em] uppercase">
                  One postulate. No free parameters. Complete derivation.
                </span>
              </div>
            </div>
          </div>
        </Section>

        {/* Bottom spacer for footer */}
        <div className="h-24" />
      </main>
    </>
  );
}
