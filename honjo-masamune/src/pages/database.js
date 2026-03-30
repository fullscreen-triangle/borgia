import Head from "next/head";
import Layout from "@/components/Layout";
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

/* ─── DATA ─────────────────────────────────────────────────────────── */

const compounds = [
  { name: "H\u2082", Sk: 1.0, St: 0.462, Se: 0.0, type: "diatomic" },
  { name: "N\u2082", Sk: 0.529, St: 0.757, Se: 0.0, type: "diatomic" },
  { name: "O\u2082", Sk: 0.359, St: 0.75, Se: 0.0, type: "diatomic" },
  { name: "CO", Sk: 0.487, St: 0.752, Se: 0.0, type: "diatomic" },
  { name: "HF", Sk: 0.899, St: 0.564, Se: 0.0, type: "diatomic" },
  { name: "HCl", Sk: 0.656, St: 0.603, Se: 0.0, type: "diatomic" },
  { name: "HBr", Sk: 0.582, St: 0.612, Se: 0.0, type: "diatomic" },
  { name: "H\u2082O", Sk: 0.944, St: 0.285, Se: 1.0, type: "triatomic" },
  { name: "CO\u2082", Sk: 0.897, St: 0.419, Se: 0.667, type: "triatomic" },
  { name: "SO\u2082", Sk: 0.937, St: 0.322, Se: 0.667, type: "triatomic" },
  { name: "NH\u2083", Sk: 0.918, St: 0.429, Se: 0.5, type: "tetra" },
  { name: "CH\u2084", Sk: 0.953, St: 0.279, Se: 0.667, type: "tetra" },
  { name: "C\u2086H\u2086", Sk: 0.913, St: 0.504, Se: 0.6, type: "poly" },
  { name: "CH\u2083OH", Sk: 0.953, St: 0.423, Se: 0.622, type: "poly" },
  { name: "C\u2082H\u2086", Sk: 0.954, St: 0.429, Se: 0.667, type: "poly" },
];

const typeColors = {
  diatomic: "#3b82f6",
  triatomic: "#22c55e",
  tetra: "#f97316",
  poly: "#ef4444",
};

const trieCells = [
  { prefix: "020", count: 2, compounds: "F\u2082, Cl\u2082", type: "diatomic" },
  { prefix: "110", count: 3, compounds: "HI, HBr, HCl", type: "diatomic" },
  { prefix: "120", count: 4, compounds: "O\u2082, NO, CO, N\u2082", type: "diatomic" },
  { prefix: "201", count: 2, compounds: "H\u2082S, H\u2082CO", type: "mixed" },
  { prefix: "202", count: 6, compounds: "O\u2083, NO\u2082, PH\u2083, CH\u2084, SO\u2082, H\u2082O", type: "mixed" },
  { prefix: "210", count: 3, compounds: "D\u2082, H\u2082, HF", type: "diatomic" },
  { prefix: "211", count: 11, compounds: "N\u2082O, HCN, C\u2082H\u2082, ...", type: "poly" },
  { prefix: "212", count: 7, compounds: "CS\u2082, OCS, CO\u2082, ...", type: "mixed" },
];

const families = [
  { name: "Hydrogen halides", ratio: 4.38, status: "pass" },
  { name: "Bent triatomics", ratio: 3.64, status: "pass" },
  { name: "Linear triatomics", ratio: 2.44, status: "pass" },
  { name: "Halomethanes", ratio: 1.77, status: "pass" },
  { name: "Homonuclear diatomics", ratio: 1.09, status: "pass" },
  { name: "Small hydrocarbons", ratio: 0.98, status: "marginal" },
];

const searchCascadeHCl = [
  { depth: 3, matches: 3, label: "HI, HBr, HCl" },
  { depth: 6, matches: 2, label: "HBr, HCl" },
  { depth: 9, matches: 1, label: "HCl" },
];

const searchCascadeCH4 = [
  { depth: 3, matches: 6, label: "O\u2083, NO\u2082, PH\u2083, CH\u2084, SO\u2082, H\u2082O" },
  { depth: 6, matches: 4, label: "PH\u2083, CH\u2084, SO\u2082, H\u2082O" },
  { depth: 9, matches: 2, label: "PH\u2083, CH\u2084" },
  { depth: 12, matches: 1, label: "CH\u2084" },
];

/* ─── CHART COMPONENTS ─────────────────────────────────────────────── */

function ScalingChart() {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    d3.select(container).selectAll("*").remove();

    const margin = { top: 30, right: 30, bottom: 60, left: 70 };
    const width = Math.min(container.clientWidth, 700) - margin.left - margin.right;
    const height = 360 - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("class", "d3-chart")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xData = d3.range(1, 9).map((e) => Math.pow(10, e));
    const bruteForce = xData.map((n) => n * 1024);
    const trieSearch = xData.map(() => 18);

    const x = d3.scaleLog().domain([10, 1e8]).range([0, width]);
    const y = d3.scaleLog().domain([10, 1e12]).range([height, 0]);

    // Grid
    svg
      .append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickSize(-height).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#1a1a1a");

    svg
      .append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-width).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#1a1a1a");

    // Axes
    svg
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${height})`)
      .call(
        d3
          .axisBottom(x)
          .ticks(8, "~s")
          .tickFormat((d) => {
            const exp = Math.round(Math.log10(d));
            return `10${exp < 10 ? "\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079"[exp] : exp}`;
          })
      )
      .selectAll("text")
      .attr("fill", "#888");

    svg
      .append("g")
      .attr("class", "axis")
      .call(
        d3
          .axisLeft(y)
          .ticks(12, "~s")
          .tickFormat((d) => {
            const exp = Math.round(Math.log10(d));
            return `10${"\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079"[exp] || exp}`;
          })
      )
      .selectAll("text")
      .attr("fill", "#888");

    // Axis labels
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height + 48)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "12px")
      .text("Database size (compounds)");

    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -55)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "12px")
      .text("Operations per query");

    // Brute force line
    const bfLine = d3
      .line()
      .x((d, i) => x(xData[i]))
      .y((d) => y(d));

    svg
      .append("path")
      .datum(bruteForce)
      .attr("fill", "none")
      .attr("stroke", "#ef4444")
      .attr("stroke-width", 2.5)
      .attr("d", bfLine);

    // Trie line
    const trieLine = d3
      .line()
      .x((d, i) => x(xData[i]))
      .y((d) => y(d));

    svg
      .append("path")
      .datum(trieSearch)
      .attr("fill", "none")
      .attr("stroke", "#58E6D9")
      .attr("stroke-width", 2.5)
      .attr("d", trieLine);

    // Labels
    svg
      .append("text")
      .attr("x", x(1e7) + 5)
      .attr("y", y(1e7 * 1024) - 10)
      .attr("fill", "#ef4444")
      .attr("font-size", "11px")
      .attr("font-weight", "600")
      .text("Brute-force O(N\u00D7d)");

    svg
      .append("text")
      .attr("x", x(1e7) + 5)
      .attr("y", y(18) - 10)
      .attr("fill", "#58E6D9")
      .attr("font-size", "11px")
      .attr("font-weight", "600")
      .text("Trie O(k) = 18");

    // Remove axis domain lines
    svg.selectAll(".axis .domain").attr("stroke", "#333");
  }, []);

  return <div ref={ref} className="w-full flex justify-center" />;
}

function ScatterPlot3D() {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    d3.select(container).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const width = Math.min(container.clientWidth, 700) - margin.left - margin.right;
    const height = 420 - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("class", "d3-chart")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // 3D isometric projection: project (Sk, St, Se) onto 2D
    const angle = Math.PI / 6;
    const project = (sk, st, se) => {
      const px = sk * Math.cos(angle) - st * Math.cos(angle);
      const py = -se + sk * Math.sin(angle) + st * Math.sin(angle);
      return [px, py];
    };

    // Compute projected coordinates
    const projected = compounds.map((c) => {
      const [px, py] = project(c.Sk, c.St, c.Se);
      return { ...c, px, py };
    });

    const xExtent = d3.extent(projected, (d) => d.px);
    const yExtent = d3.extent(projected, (d) => d.py);
    const pad = 0.1;

    const x = d3
      .scaleLinear()
      .domain([xExtent[0] - pad, xExtent[1] + pad])
      .range([0, width]);

    const y = d3
      .scaleLinear()
      .domain([yExtent[0] - pad, yExtent[1] + pad])
      .range([height, 0]);

    // Draw projected axis lines from origin
    const origin = project(0, 0, 0);
    const axisEnds = [
      { label: "S_k", point: project(1, 0, 0), color: "#666" },
      { label: "S_t", point: project(0, 1, 0), color: "#666" },
      { label: "S_e", point: project(0, 0, 1), color: "#666" },
    ];

    axisEnds.forEach(({ label, point, color }) => {
      svg
        .append("line")
        .attr("x1", x(origin[0]))
        .attr("y1", y(origin[1]))
        .attr("x2", x(point[0]))
        .attr("y2", y(point[1]))
        .attr("stroke", color)
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4");

      svg
        .append("text")
        .attr("x", x(point[0]) + (point[0] > origin[0] ? 8 : -8))
        .attr("y", y(point[1]) + (point[1] > origin[1] ? -8 : 14))
        .attr("fill", "#aaa")
        .attr("font-size", "11px")
        .attr("font-weight", "600")
        .attr("text-anchor", point[0] > origin[0] ? "start" : "end")
        .text(label);
    });

    // Tooltip
    const tooltip = d3
      .select(container)
      .append("div")
      .style("position", "absolute")
      .style("background", "rgba(20, 20, 20, 0.95)")
      .style("border", "1px solid #333")
      .style("border-radius", "6px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("color", "#e5e5e5")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10);

    // Draw points
    svg
      .selectAll("circle")
      .data(projected)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.px))
      .attr("cy", (d) => y(d.py))
      .attr("r", 6)
      .attr("fill", (d) => typeColors[d.type])
      .attr("fill-opacity", 0.85)
      .attr("stroke", (d) => typeColors[d.type])
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.4)
      .style("cursor", "pointer")
      .on("mouseover", function (event, d) {
        d3.select(this).attr("r", 9).attr("fill-opacity", 1);
        tooltip
          .html(
            `<strong>${d.name}</strong><br/>S<sub>k</sub>=${d.Sk.toFixed(3)} &nbsp; S<sub>t</sub>=${d.St.toFixed(3)} &nbsp; S<sub>e</sub>=${d.Se.toFixed(3)}`
          )
          .style("left", event.offsetX + 14 + "px")
          .style("top", event.offsetY - 10 + "px")
          .transition()
          .duration(150)
          .style("opacity", 1);
      })
      .on("mouseout", function () {
        d3.select(this).attr("r", 6).attr("fill-opacity", 0.85);
        tooltip.transition().duration(200).style("opacity", 0);
      });

    // Labels for non-overlapping points
    svg
      .selectAll(".label")
      .data(projected)
      .enter()
      .append("text")
      .attr("class", "label")
      .attr("x", (d) => x(d.px) + 10)
      .attr("y", (d) => y(d.py) + 4)
      .attr("fill", "#999")
      .attr("font-size", "10px")
      .text((d) => d.name);

    // Legend
    const legendData = [
      { type: "diatomic", label: "Diatomic" },
      { type: "triatomic", label: "Triatomic" },
      { type: "tetra", label: "Tetratomic" },
      { type: "poly", label: "Polyatomic" },
    ];

    const legend = svg
      .append("g")
      .attr("transform", `translate(${width - 110}, 0)`);

    legendData.forEach((d, i) => {
      legend
        .append("circle")
        .attr("cx", 0)
        .attr("cy", i * 20)
        .attr("r", 5)
        .attr("fill", typeColors[d.type]);
      legend
        .append("text")
        .attr("x", 12)
        .attr("y", i * 20 + 4)
        .attr("fill", "#999")
        .attr("font-size", "11px")
        .text(d.label);
    });
  }, []);

  return <div ref={ref} className="w-full flex justify-center relative" />;
}

function TrieVisualization() {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    d3.select(container).selectAll("*").remove();

    const margin = { top: 30, right: 30, bottom: 20, left: 30 };
    const width = Math.min(container.clientWidth, 800) - margin.left - margin.right;
    const height = 480 - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("class", "d3-chart")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Build tree structure from trie cells
    // Root -> depth-1 nodes (first digit) -> depth-2 nodes -> depth-3 leaf nodes
    const root = { name: "root", children: [] };
    const firstDigits = new Set(trieCells.map((c) => c.prefix[0]));

    firstDigits.forEach((d1) => {
      const node1 = { name: d1, children: [] };
      const matchingCells = trieCells.filter((c) => c.prefix[0] === d1);
      const secondDigits = new Set(matchingCells.map((c) => c.prefix[1]));

      secondDigits.forEach((d2) => {
        const node2 = { name: d1 + d2, children: [] };
        const leafCells = matchingCells.filter((c) => c.prefix[1] === d2);

        leafCells.forEach((cell) => {
          node2.children.push({
            name: cell.prefix,
            count: cell.count,
            compounds: cell.compounds,
            type: cell.type,
          });
        });

        node1.children.push(node2);
      });

      root.children.push(node1);
    });

    const treeLayout = d3.tree().size([width, height - 80]);
    const hierarchy = d3.hierarchy(root);
    treeLayout(hierarchy);

    // Links
    svg
      .selectAll(".link")
      .data(hierarchy.links())
      .enter()
      .append("path")
      .attr("class", "link")
      .attr("fill", "none")
      .attr("stroke", "#333")
      .attr("stroke-width", 1.5)
      .attr(
        "d",
        d3
          .linkVertical()
          .x((d) => d.x)
          .y((d) => d.y + 20)
      );

    // Tooltip
    const tooltip = d3
      .select(container)
      .append("div")
      .style("position", "absolute")
      .style("background", "rgba(20, 20, 20, 0.95)")
      .style("border", "1px solid #333")
      .style("border-radius", "6px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("color", "#e5e5e5")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10)
      .style("max-width", "260px");

    // Nodes
    const cellColors = {
      diatomic: "#3b82f6",
      mixed: "#a855f7",
      poly: "#ef4444",
    };

    const nodes = svg
      .selectAll(".node")
      .data(hierarchy.descendants())
      .enter()
      .append("g")
      .attr("class", "node")
      .attr("transform", (d) => `translate(${d.x},${d.y + 20})`);

    nodes
      .append("circle")
      .attr("r", (d) => {
        if (d.depth === 0) return 14;
        if (d.depth < 3) return 10;
        return 6 + (d.data.count || 1) * 1.5;
      })
      .attr("fill", (d) => {
        if (d.depth === 0) return "#1a1a1a";
        if (d.depth < 3) return "#1a1a1a";
        return cellColors[d.data.type] || "#666";
      })
      .attr("stroke", (d) => {
        if (d.depth === 0) return "#58E6D9";
        if (d.depth < 3) return "#555";
        return cellColors[d.data.type] || "#666";
      })
      .attr("stroke-width", (d) => (d.depth === 0 ? 2 : 1.5))
      .attr("fill-opacity", (d) => (d.depth === 3 ? 0.3 : 1))
      .style("cursor", (d) => (d.depth === 3 ? "pointer" : "default"))
      .on("mouseover", function (event, d) {
        if (d.depth === 3 && d.data.compounds) {
          d3.select(this).attr("fill-opacity", 0.6);
          tooltip
            .html(
              `<strong>Prefix: ${d.data.name}</strong><br/>Count: ${d.data.count}<br/>${d.data.compounds}`
            )
            .style("left", event.offsetX + 14 + "px")
            .style("top", event.offsetY - 10 + "px")
            .transition()
            .duration(150)
            .style("opacity", 1);
        }
      })
      .on("mouseout", function (event, d) {
        if (d.depth === 3) {
          d3.select(this).attr("fill-opacity", 0.3);
          tooltip.transition().duration(200).style("opacity", 0);
        }
      });

    // Node labels
    nodes
      .append("text")
      .attr("dy", (d) => (d.depth === 3 ? -16 : 4))
      .attr("text-anchor", "middle")
      .attr("fill", (d) => (d.depth === 0 ? "#58E6D9" : "#ccc"))
      .attr("font-size", (d) => {
        if (d.depth === 0) return "10px";
        if (d.depth < 3) return "10px";
        return "9px";
      })
      .attr("font-weight", (d) => (d.depth === 0 ? "700" : "400"))
      .text((d) => {
        if (d.depth === 0) return "root";
        return d.data.name;
      });

    // Count labels on leaves
    nodes
      .filter((d) => d.depth === 3 && d.data.count)
      .append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("font-size", "10px")
      .attr("font-weight", "700")
      .text((d) => d.data.count);

    // Depth labels
    const depths = [
      { y: 20, label: "Root" },
      { y: hierarchy.descendants().find((d) => d.depth === 1).y + 20, label: "Depth 1 (S_k)" },
      { y: hierarchy.descendants().find((d) => d.depth === 2).y + 20, label: "Depth 2 (S_t)" },
      { y: hierarchy.descendants().find((d) => d.depth === 3).y + 20, label: "Depth 3 (S_e)" },
    ];

    depths.forEach(({ y: yPos, label }) => {
      svg
        .append("text")
        .attr("x", -10)
        .attr("y", yPos)
        .attr("text-anchor", "end")
        .attr("fill", "#555")
        .attr("font-size", "9px")
        .text(label);
    });
  }, []);

  return <div ref={ref} className="w-full flex justify-center relative" />;
}

function SearchCascadeChart() {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    d3.select(container).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const width = Math.min(container.clientWidth, 700) - margin.left - margin.right;
    const height = 340 - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("class", "d3-chart")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, 14]).range([0, width]);
    const y = d3.scaleLinear().domain([0, 7]).range([height, 0]);

    // Grid
    svg
      .append("g")
      .attr("class", "grid")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).tickSize(-height).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#1a1a1a");

    svg
      .append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-width).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#1a1a1a");

    // Axes
    svg
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(7).tickFormat((d) => (d === 0 ? "" : `${d}`)))
      .selectAll("text")
      .attr("fill", "#888");

    svg
      .append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(y).ticks(7))
      .selectAll("text")
      .attr("fill", "#888");

    svg.selectAll(".axis .domain").attr("stroke", "#333");

    // Axis labels
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height + 48)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "12px")
      .text("Trie depth (digits)");

    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -50)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "12px")
      .text("Candidate matches");

    // HCl line
    const hclLine = d3
      .line()
      .x((d) => x(d.depth))
      .y((d) => y(d.matches))
      .curve(d3.curveMonotoneX);

    svg
      .append("path")
      .datum(searchCascadeHCl)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2.5)
      .attr("d", hclLine);

    svg
      .selectAll(".hcl-dot")
      .data(searchCascadeHCl)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.depth))
      .attr("cy", (d) => y(d.matches))
      .attr("r", 5)
      .attr("fill", "#3b82f6")
      .attr("stroke", "#0a0a0a")
      .attr("stroke-width", 2);

    // HCl labels
    svg
      .selectAll(".hcl-label")
      .data(searchCascadeHCl)
      .enter()
      .append("text")
      .attr("x", (d) => x(d.depth) + 10)
      .attr("y", (d) => y(d.matches) - 10)
      .attr("fill", "#6b9eff")
      .attr("font-size", "10px")
      .text((d) => d.label);

    // CH4 line
    const ch4Line = d3
      .line()
      .x((d) => x(d.depth))
      .y((d) => y(d.matches))
      .curve(d3.curveMonotoneX);

    svg
      .append("path")
      .datum(searchCascadeCH4)
      .attr("fill", "none")
      .attr("stroke", "#f97316")
      .attr("stroke-width", 2.5)
      .attr("d", ch4Line);

    svg
      .selectAll(".ch4-dot")
      .data(searchCascadeCH4)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d.depth))
      .attr("cy", (d) => y(d.matches))
      .attr("r", 5)
      .attr("fill", "#f97316")
      .attr("stroke", "#0a0a0a")
      .attr("stroke-width", 2);

    // CH4 labels
    svg
      .selectAll(".ch4-label")
      .data(searchCascadeCH4)
      .enter()
      .append("text")
      .attr("x", (d, i) => x(d.depth) + 10)
      .attr("y", (d, i) => y(d.matches) + (i === 0 ? 18 : -10))
      .attr("fill", "#ffb366")
      .attr("font-size", "10px")
      .text((d) => {
        if (d.label.length > 30) return d.matches + " compounds";
        return d.label;
      });

    // Legend
    const legend = svg.append("g").attr("transform", `translate(${width - 140}, 0)`);

    [
      { label: "Query: HCl", color: "#3b82f6" },
      { label: "Query: CH\u2084", color: "#f97316" },
    ].forEach((d, i) => {
      legend
        .append("line")
        .attr("x1", 0)
        .attr("x2", 20)
        .attr("y1", i * 20)
        .attr("y2", i * 20)
        .attr("stroke", d.color)
        .attr("stroke-width", 2.5);
      legend
        .append("text")
        .attr("x", 28)
        .attr("y", i * 20 + 4)
        .attr("fill", "#999")
        .attr("font-size", "11px")
        .text(d.label);
    });

    // Unique threshold line
    svg
      .append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", y(1))
      .attr("y2", y(1))
      .attr("stroke", "#58E6D9")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "6,4")
      .attr("opacity", 0.5);

    svg
      .append("text")
      .attr("x", width - 5)
      .attr("y", y(1) - 6)
      .attr("text-anchor", "end")
      .attr("fill", "#58E6D9")
      .attr("font-size", "10px")
      .attr("opacity", 0.7)
      .text("Unique identification");
  }, []);

  return <div ref={ref} className="w-full flex justify-center" />;
}

function CohesionChart() {
  const ref = useRef();

  useEffect(() => {
    if (!ref.current) return;
    const container = ref.current;
    d3.select(container).selectAll("*").remove();

    const margin = { top: 30, right: 30, bottom: 80, left: 70 };
    const width = Math.min(container.clientWidth, 700) - margin.left - margin.right;
    const height = 360 - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("class", "d3-chart")
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(families.map((d) => d.name))
      .range([0, width])
      .padding(0.3);

    const y = d3.scaleLinear().domain([0, 5]).range([height, 0]);

    // Grid
    svg
      .append("g")
      .attr("class", "grid")
      .call(d3.axisLeft(y).tickSize(-width).tickFormat(""))
      .selectAll("line")
      .attr("stroke", "#1a1a1a");

    // Threshold line at 1.0
    svg
      .append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", y(1))
      .attr("y2", y(1))
      .attr("stroke", "#58E6D9")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "6,4")
      .attr("opacity", 0.5);

    svg
      .append("text")
      .attr("x", width + 5)
      .attr("y", y(1) + 4)
      .attr("fill", "#58E6D9")
      .attr("font-size", "9px")
      .attr("opacity", 0.7)
      .text("1.0");

    // Bars
    svg
      .selectAll(".bar")
      .data(families)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(d.name))
      .attr("y", (d) => y(d.ratio))
      .attr("width", x.bandwidth())
      .attr("height", (d) => height - y(d.ratio))
      .attr("rx", 3)
      .attr("fill", (d) => (d.status === "pass" ? "#22c55e" : "#eab308"))
      .attr("fill-opacity", 0.7);

    // Value labels on bars
    svg
      .selectAll(".val-label")
      .data(families)
      .enter()
      .append("text")
      .attr("x", (d) => x(d.name) + x.bandwidth() / 2)
      .attr("y", (d) => y(d.ratio) - 8)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => (d.status === "pass" ? "#22c55e" : "#eab308"))
      .attr("font-size", "12px")
      .attr("font-weight", "700")
      .text((d) => d.ratio.toFixed(2) + "\u00D7");

    // X axis
    svg
      .append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("fill", "#888")
      .attr("font-size", "10px")
      .attr("transform", "rotate(-30)")
      .style("text-anchor", "end");

    // Y axis
    svg
      .append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(y).ticks(5))
      .selectAll("text")
      .attr("fill", "#888");

    svg.selectAll(".axis .domain").attr("stroke", "#333");

    // Y axis label
    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -50)
      .attr("text-anchor", "middle")
      .attr("fill", "#888")
      .attr("font-size", "12px")
      .text("Cohesion ratio (intra / inter)");
  }, []);

  return <div ref={ref} className="w-full flex justify-center" />;
}

/* ─── API CODE BLOCK ───────────────────────────────────────────────── */

function CodeBlock({ title, children }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 overflow-hidden">
      {title && (
        <div className="px-4 py-2 border-b border-neutral-800 text-xs text-neutral-500 font-mono">
          {title}
        </div>
      )}
      <pre className="p-4 overflow-x-auto text-sm leading-relaxed font-mono text-neutral-300">
        <code>{children}</code>
      </pre>
    </div>
  );
}

function ApiEndpoint({ method, path, description, request, response }) {
  return (
    <div className="border border-neutral-800 rounded-xl bg-neutral-900/50 overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-neutral-800 flex items-center gap-3">
        <span className="px-2.5 py-1 rounded text-xs font-bold font-mono bg-emerald-500/20 text-emerald-400 tracking-wider">
          {method}
        </span>
        <span className="font-mono text-sm text-white">{path}</span>
      </div>
      <div className="px-6 py-4 space-y-4">
        <p className="text-neutral-400 text-sm">{description}</p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-1">
          <div>
            <p className="text-xs uppercase tracking-wider text-neutral-500 mb-2 font-semibold">
              Request
            </p>
            <CodeBlock title="application/json">{request}</CodeBlock>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-neutral-500 mb-2 font-semibold">
              Response
            </p>
            <CodeBlock title="200 OK">{response}</CodeBlock>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── SECTION WRAPPER ──────────────────────────────────────────────── */

function Section({ children, className = "" }) {
  const ref = useRef();
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    let ctx;

    const init = async () => {
      try {
        const gsapModule = await import("gsap");
        const gsapInstance = gsapModule.default || gsapModule.gsap;
        const { ScrollTrigger } = await import("gsap/dist/ScrollTrigger");
        gsapInstance.registerPlugin(ScrollTrigger);

        if (ref.current) {
          setVisible(false);
          ctx = gsapInstance.context(() => {
            gsapInstance.fromTo(
              ref.current,
              { y: 50, opacity: 0 },
              {
                y: 0,
                opacity: 1,
                duration: 1,
                ease: "power3.out",
                scrollTrigger: {
                  trigger: ref.current,
                  start: "top 85%",
                  toggleActions: "play none none none",
                },
              }
            );
          });
        }
      } catch (e) {
        setVisible(true);
      }
    };

    init();

    return () => {
      if (ctx) ctx.revert();
    };
  }, []);

  return (
    <section ref={ref} className={className} style={{ opacity: visible ? 1 : 0 }}>
      {children}
    </section>
  );
}

/* ─── MAIN PAGE ────────────────────────────────────────────────────── */

export default function Database() {
  const [activeTab, setActiveTab] = useState(0);

  const apiEndpoints = [
    {
      method: "POST",
      path: "/encode",
      description:
        "Encode a set of vibrational frequencies into S-entropy coordinates and a ternary trie address.",
      request: `{
  "frequencies": [1595, 3657, 3756],
  "unit": "cm-1"
}`,
      response: `{
  "compound": {
    "S_k": 0.9442,
    "S_t": 0.2850,
    "S_e": 1.0000,
    "trit_address": "202222112122",
    "depth": 12
  }
}`,
    },
    {
      method: "POST",
      path: "/search",
      description:
        "Search the database at a given resolution. Lower resolution returns broader matches, higher resolution narrows to exact identification.",
      request: `{
  "frequencies": [1595, 3657, 3756],
  "resolution": 6
}`,
      response: `{
  "query_address": "202222",
  "resolution": 6,
  "matches": [
    {
      "name": "H\u2082O",
      "address": "202222112122",
      "similarity": 18
    }
  ]
}`,
    },
    {
      method: "POST",
      path: "/identify",
      description:
        "Full identification pipeline: encode, search at maximum depth, return the closest match with nearest neighbors.",
      request: `{
  "frequencies": [1595, 3657, 3756]
}`,
      response: `{
  "identification": {
    "name": "H\u2082O",
    "formula": "H\u2082O",
    "confidence": 0.998,
    "S_k": 0.9442,
    "S_t": 0.2850,
    "S_e": 1.0000,
    "nearest_neighbors": [
      { "name": "SO\u2082", "similarity": 5 },
      { "name": "NO\u2082", "similarity": 5 }
    ]
  }
}`,
    },
    {
      method: "POST",
      path: "/predict",
      description:
        "Property-constrained search: find all compounds whose S-entropy coordinates fall within the specified bounds.",
      request: `{
  "constraints": {
    "S_k": [0.9, 1.0],
    "S_t": [0.0, 0.3],
    "S_e": [0.5, 1.0]
  }
}`,
      response: `{
  "matches": [
    "H\u2082O", "NO\u2082", "O\u2083",
    "PH\u2083", "CH\u2084", "SO\u2082", "H\u2082S"
  ],
  "count": 7,
  "resolution": "property-constrained"
}`,
    },
  ];

  return (
    <>
      <Head>
        <title>Database | Honjo Masamune</title>
        <meta
          name="description"
          content="Categorical Compound Database: molecular identification through ternary phase space addressing."
        />
      </Head>

      <main className="pt-24 min-h-screen bg-[#0a0a0a]">
        <Layout>
          <div className="max-w-4xl mx-auto space-y-32 pb-24">
            {/* ─── Section 1: Hero ───────────────────────────────────── */}
            <Section>
              <div className="space-y-6">
                <h1 className="text-5xl font-bold text-white tracking-tight md:text-3xl sm:text-2xl">
                  Categorical Compound Database
                </h1>
                <p className="text-xl text-neutral-400 font-light md:text-lg sm:text-base">
                  Molecular identification through ternary phase space addressing
                </p>
                <div className="h-px w-24 bg-gradient-to-r from-[#58E6D9] to-transparent" />
                <p className="text-neutral-400 leading-relaxed max-w-3xl">
                  Instead of computing fingerprints and comparing them O(N), we address molecules
                  in their natural coordinate space. Every compound maps to a unique point in
                  three-dimensional S-entropy space, and search becomes trie traversal &mdash;{" "}
                  <span className="text-white font-medium">
                    O(k) independent of database size
                  </span>
                  .
                </p>
              </div>
            </Section>

            {/* ─── Section 2: The Problem ────────────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">The Problem</h2>
                <p className="text-neutral-400 leading-relaxed">
                  The traditional cheminformatics pipeline is fundamentally brute-force:
                </p>

                {/* Pipeline flow */}
                <div className="flex items-center gap-3 flex-wrap font-mono text-sm">
                  {["SMILES", "Fingerprint", "Tanimoto", "O(N\u00D7d)"].map((step, i) => (
                    <span key={step} className="flex items-center gap-3">
                      <span className="px-3 py-1.5 rounded border border-neutral-700 bg-neutral-900 text-neutral-300">
                        {step}
                      </span>
                      {i < 3 && <span className="text-neutral-600">&rarr;</span>}
                    </span>
                  ))}
                </div>

                <p className="text-neutral-400 leading-relaxed">
                  At PubChem scale (10<sup>8</sup> compounds), each query costs ~10<sup>11</sup>{" "}
                  operations. The search cost grows linearly with the database. The ternary trie
                  collapses this to a fixed-depth traversal.
                </p>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:p-4">
                  <p className="text-xs uppercase tracking-wider text-neutral-500 mb-4 font-semibold">
                    Operations per query vs. database size (log-log)
                  </p>
                  <ScalingChart />
                </div>
              </div>
            </Section>

            {/* ─── Section 3: S-Entropy Encoding ─────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">S-Entropy Encoding</h2>
                <p className="text-neutral-400 leading-relaxed">
                  Three coordinates extracted from vibrational spectra locate every molecule in
                  phase space:
                </p>

                <div className="grid grid-cols-3 gap-4 md:grid-cols-1">
                  {[
                    {
                      symbol: "S_k",
                      label: "Knowledge",
                      desc: "Shannon entropy of the frequency distribution",
                      question: "WHAT the molecule is",
                      color: "#3b82f6",
                    },
                    {
                      symbol: "S_t",
                      label: "Temporal",
                      desc: "Log-ratio of frequency range",
                      question: "HOW MANY timescales it spans",
                      color: "#22c55e",
                    },
                    {
                      symbol: "S_e",
                      label: "Evolution",
                      desc: "Harmonic edge density",
                      question: "HOW interconnected its modes are",
                      color: "#f97316",
                    },
                  ].map((item) => (
                    <div
                      key={item.symbol}
                      className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-5 space-y-3"
                    >
                      <div className="flex items-baseline gap-2">
                        <span
                          className="font-mono text-lg font-bold"
                          style={{ color: item.color }}
                        >
                          {item.symbol}
                        </span>
                        <span className="text-neutral-500 text-sm">{item.label}</span>
                      </div>
                      <p className="text-neutral-400 text-sm leading-relaxed">{item.desc}</p>
                      <p className="text-xs uppercase tracking-wider" style={{ color: item.color }}>
                        {item.question}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:p-4">
                  <p className="text-xs uppercase tracking-wider text-neutral-500 mb-4 font-semibold">
                    39 compounds projected from S-entropy space (isometric)
                  </p>
                  <ScatterPlot3D />
                </div>
              </div>
            </Section>

            {/* ─── Section 4: Ternary Addressing ─────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">Ternary Addressing</h2>
                <p className="text-neutral-400 leading-relaxed">
                  Each S-entropy coordinate is quantized into ternary digits (0, 1, 2) at
                  increasing resolution. The interleaving scheme assigns position{" "}
                  <span className="font-mono text-white">j mod 3</span> to determine which
                  dimension is refined: <span className="font-mono text-[#3b82f6]">S_k</span> at
                  positions 0, 3, 6, ...; <span className="font-mono text-[#22c55e]">S_t</span>{" "}
                  at 1, 4, 7, ...; <span className="font-mono text-[#f97316]">S_e</span> at
                  2, 5, 8, ...
                </p>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-5 font-mono text-sm space-y-2">
                  <p className="text-neutral-500 text-xs uppercase tracking-wider mb-3 font-sans font-semibold">
                    Address interleaving example: H&#x2082;O
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {"202222112122".split("").map((digit, i) => {
                      const colors = ["#3b82f6", "#22c55e", "#f97316"];
                      const labels = ["k", "t", "e"];
                      return (
                        <div key={i} className="flex flex-col items-center gap-1">
                          <span className="text-neutral-600 text-[10px]">
                            {labels[i % 3]}
                          </span>
                          <span
                            className="w-7 h-7 flex items-center justify-center rounded border text-white font-bold"
                            style={{
                              borderColor: colors[i % 3],
                              color: colors[i % 3],
                              backgroundColor: colors[i % 3] + "15",
                            }}
                          >
                            {digit}
                          </span>
                          <span className="text-neutral-600 text-[10px]">{i}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:p-4">
                  <p className="text-xs uppercase tracking-wider text-neutral-500 mb-4 font-semibold">
                    Ternary trie at depth 3 &mdash; 8 occupied cells
                  </p>
                  <TrieVisualization />
                </div>

                {/* Trie cell table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-neutral-800">
                        <th className="text-left py-3 px-4 text-neutral-500 font-semibold text-xs uppercase tracking-wider">
                          Prefix
                        </th>
                        <th className="text-left py-3 px-4 text-neutral-500 font-semibold text-xs uppercase tracking-wider">
                          Count
                        </th>
                        <th className="text-left py-3 px-4 text-neutral-500 font-semibold text-xs uppercase tracking-wider">
                          Compounds
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {trieCells.map((cell) => (
                        <tr
                          key={cell.prefix}
                          className="border-b border-neutral-800/50 hover:bg-neutral-900/80 transition-colors"
                        >
                          <td className="py-3 px-4 font-mono text-[#58E6D9]">{cell.prefix}</td>
                          <td className="py-3 px-4 text-white font-semibold">{cell.count}</td>
                          <td className="py-3 px-4 text-neutral-400">{cell.compounds}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Section>

            {/* ─── Section 5: Fuzzy Search ───────────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">Fuzzy Search</h2>
                <p className="text-neutral-400 leading-relaxed">
                  Resolution is adjustable: a shorter prefix yields broader matches, a longer
                  prefix narrows toward exact identification. The search cascade shows how
                  candidate sets shrink monotonically with depth.
                </p>

                <div className="grid grid-cols-2 gap-6 md:grid-cols-1">
                  {/* HCl cascade */}
                  <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-5 space-y-4">
                    <p className="font-mono text-sm text-[#3b82f6] font-semibold">
                      Query: HCl
                    </p>
                    {searchCascadeHCl.map((step) => (
                      <div key={step.depth} className="flex items-center gap-3">
                        <span className="text-neutral-500 font-mono text-xs w-16 shrink-0">
                          depth {step.depth}
                        </span>
                        <div
                          className="h-6 rounded flex items-center px-2 text-xs font-mono text-white"
                          style={{
                            width: `${(step.matches / 6) * 100}%`,
                            minWidth: "fit-content",
                            backgroundColor: "#3b82f6",
                            opacity: 0.3 + (1 - step.matches / 6) * 0.7,
                          }}
                        >
                          {step.label}
                        </div>
                        <span className="text-neutral-500 text-xs shrink-0">
                          {step.matches} match{step.matches !== 1 ? "es" : ""}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* CH4 cascade */}
                  <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-5 space-y-4">
                    <p className="font-mono text-sm text-[#f97316] font-semibold">
                      Query: CH&#x2084;
                    </p>
                    {searchCascadeCH4.map((step) => (
                      <div key={step.depth} className="flex items-center gap-3">
                        <span className="text-neutral-500 font-mono text-xs w-16 shrink-0">
                          depth {step.depth}
                        </span>
                        <div
                          className="h-6 rounded flex items-center px-2 text-xs font-mono text-white"
                          style={{
                            width: `${(step.matches / 6) * 100}%`,
                            minWidth: "fit-content",
                            backgroundColor: "#f97316",
                            opacity: 0.3 + (1 - step.matches / 6) * 0.7,
                          }}
                        >
                          {step.matches <= 4 ? step.label : `${step.matches} compounds`}
                        </div>
                        <span className="text-neutral-500 text-xs shrink-0">
                          {step.matches} match{step.matches !== 1 ? "es" : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:p-4">
                  <p className="text-xs uppercase tracking-wider text-neutral-500 mb-4 font-semibold">
                    Search narrowing cascade
                  </p>
                  <SearchCascadeChart />
                </div>
              </div>
            </Section>

            {/* ─── Section 6: Chemical Similarity ────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">Chemical Similarity</h2>
                <p className="text-neutral-400 leading-relaxed">
                  Chemical families emerge from ternary proximity without being encoded
                  explicitly. The cohesion ratio measures intra-family distance versus
                  inter-family distance: values above 1.0 confirm that the addressing scheme
                  recovers genuine chemical similarity.
                </p>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 md:p-4">
                  <p className="text-xs uppercase tracking-wider text-neutral-500 mb-4 font-semibold">
                    Cohesion ratio by chemical family (threshold: 1.0)
                  </p>
                  <CohesionChart />
                </div>

                <div className="grid grid-cols-3 gap-3 md:grid-cols-2 sm:grid-cols-1">
                  {families.map((f) => (
                    <div
                      key={f.name}
                      className="bg-neutral-900/50 border border-neutral-800 rounded-lg p-4 flex items-center justify-between"
                    >
                      <span className="text-neutral-300 text-sm">{f.name}</span>
                      <span
                        className={`font-mono text-sm font-bold ${
                          f.status === "pass" ? "text-emerald-400" : "text-yellow-400"
                        }`}
                      >
                        {f.ratio.toFixed(2)}&times;
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Section>

            {/* ─── Section 7: API Documentation ──────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">API Reference</h2>
                <p className="text-neutral-400 leading-relaxed">
                  RESTful API for encoding, searching, identifying, and predicting molecular
                  compounds through the Categorical Compound Database.
                </p>

                <div className="flex items-center gap-2">
                  <span className="text-neutral-500 text-sm">Base URL:</span>
                  <code className="px-3 py-1 rounded bg-neutral-900 border border-neutral-800 font-mono text-sm text-[#58E6D9]">
                    https://api.honjo-masamune.io/v1
                  </code>
                </div>

                {/* Endpoint tabs */}
                <div className="flex gap-1 border-b border-neutral-800 overflow-x-auto">
                  {apiEndpoints.map((ep, i) => (
                    <button
                      key={ep.path}
                      onClick={() => setActiveTab(i)}
                      className={`px-4 py-2.5 text-sm font-mono whitespace-nowrap transition-colors ${
                        activeTab === i
                          ? "text-[#58E6D9] border-b-2 border-[#58E6D9]"
                          : "text-neutral-500 hover:text-neutral-300"
                      }`}
                    >
                      {ep.path}
                    </button>
                  ))}
                </div>

                <ApiEndpoint {...apiEndpoints[activeTab]} />

                {/* All endpoints listed below for reference */}
                <div className="space-y-6 mt-8">
                  <h3 className="text-xl font-semibold text-white">All Endpoints</h3>
                  <div className="space-y-4">
                    {apiEndpoints.map((ep) => (
                      <ApiEndpoint key={ep.path} {...ep} />
                    ))}
                  </div>
                </div>
              </div>
            </Section>

            {/* ─── Section 8: Performance ────────────────────────────── */}
            <Section>
              <div className="space-y-8">
                <h2 className="text-3xl font-bold text-white sm:text-2xl">Performance</h2>
                <p className="text-neutral-400 leading-relaxed">
                  The ternary trie converts molecular identification from a comparison problem
                  into an addressing problem. The practical consequences:
                </p>

                <div className="grid grid-cols-2 gap-4 md:grid-cols-1">
                  {[
                    {
                      value: "3,328\u00D7",
                      label: "Speedup over brute-force",
                      detail: "N=39, depth 12",
                    },
                    {
                      value: "O(k)",
                      label: "Search complexity",
                      detail: "Independent of database size N",
                    },
                    {
                      value: "12",
                      label: "Observations to identify",
                      detail: "Oscillation-counting measurements",
                    },
                    {
                      value: ">10\u2079\u00D7",
                      label: "Projected speedup",
                      detail: "At PubChem scale (10\u2078 compounds)",
                    },
                  ].map((stat) => (
                    <div
                      key={stat.label}
                      className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6 space-y-2"
                    >
                      <p className="text-3xl font-bold text-[#58E6D9] font-mono sm:text-2xl">
                        {stat.value}
                      </p>
                      <p className="text-white text-sm font-semibold">{stat.label}</p>
                      <p className="text-neutral-500 text-xs">{stat.detail}</p>
                    </div>
                  ))}
                </div>

                <div className="bg-neutral-900/50 border border-neutral-800 rounded-xl p-6">
                  <p className="text-neutral-400 text-sm leading-relaxed">
                    The database grows by insertion, not recomputation. Each new compound is
                    encoded once into its S-entropy coordinates and placed at its ternary
                    address. No existing entries are disturbed. At PubChem scale, the brute-force
                    pipeline requires ~10<sup>11</sup> operations per query; the trie requires
                    exactly <span className="text-white font-mono">k = 18</span> digit
                    comparisons, regardless of whether the database contains 39 or 10<sup>8</sup>{" "}
                    compounds.
                  </p>
                </div>
              </div>
            </Section>
          </div>
        </Layout>
      </main>
    </>
  );
}
