/**
 * HarmonicNetwork — D3 force-directed graph of vibrational mode network
 * Shows modes as nodes, harmonic edges as links, closed loops highlighted.
 */
import { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function HarmonicNetwork({ modes = [], edges = [], width = 400, height = 300 }) {
  const svgRef = useRef();

  useEffect(() => {
    if (!svgRef.current || modes.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const nodes = modes.map((freq, i) => ({
      id: i,
      freq,
      label: `${freq}`,
      r: 6 + Math.log(freq) * 2,
    }));

    const links = edges.map((e) => ({
      source: e.i,
      target: e.j,
      ratio: `${e.p}/${e.q}`,
      deviation: e.deviation,
    }));

    // Detect loops (simple cycle detection for small graphs)
    const loops = [];
    const adj = {};
    edges.forEach((e) => {
      if (!adj[e.i]) adj[e.i] = [];
      if (!adj[e.j]) adj[e.j] = [];
      adj[e.i].push(e.j);
      adj[e.j].push(e.i);
    });
    // Find triangles
    for (let a = 0; a < modes.length; a++) {
      for (const b of (adj[a] || [])) {
        if (b <= a) continue;
        for (const c of (adj[b] || [])) {
          if (c <= b) continue;
          if ((adj[a] || []).includes(c)) {
            loops.push([a, b, c]);
          }
        }
      }
    }

    const loopNodes = new Set(loops.flat());
    const loopEdges = new Set();
    loops.forEach(([a, b, c]) => {
      loopEdges.add(`${Math.min(a,b)}-${Math.max(a,b)}`);
      loopEdges.add(`${Math.min(b,c)}-${Math.max(b,c)}`);
      loopEdges.add(`${Math.min(a,c)}-${Math.max(a,c)}`);
    });

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => d.r + 4));

    // Links
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => {
        const key = `${Math.min(d.source.id ?? d.source, d.target.id ?? d.target)}-${Math.max(d.source.id ?? d.source, d.target.id ?? d.target)}`;
        return loopEdges.has(key) ? "#58E6D9" : "#555";
      })
      .attr("stroke-width", (d) => {
        const key = `${Math.min(d.source.id ?? d.source, d.target.id ?? d.target)}-${Math.max(d.source.id ?? d.source, d.target.id ?? d.target)}`;
        return loopEdges.has(key) ? 2.5 : 1;
      })
      .attr("stroke-opacity", 0.7);

    // Nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => d.r)
      .attr("fill", (d) => loopNodes.has(d.id) ? "#58E6D9" : "#3b82f6")
      .attr("stroke", "#000")
      .attr("stroke-width", 0.5)
      .attr("cursor", "pointer");

    // Labels
    const label = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text((d) => d.freq)
      .attr("font-size", "8px")
      .attr("fill", "#ccc")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => -d.r - 4);

    // Ratio labels on edges
    const edgeLabel = svg.append("g")
      .selectAll("text")
      .data(links)
      .join("text")
      .text((d) => d.ratio)
      .attr("font-size", "7px")
      .attr("fill", "#888")
      .attr("text-anchor", "middle");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y);
      label
        .attr("x", (d) => d.x)
        .attr("y", (d) => d.y);
      edgeLabel
        .attr("x", (d) => (d.source.x + d.target.x) / 2)
        .attr("y", (d) => (d.source.y + d.target.y) / 2);
    });

    // Drag
    node.call(
      d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        })
    );

    return () => simulation.stop();
  }, [modes, edges, width, height]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="w-full rounded-lg border border-neutral-800 bg-[#0d0d0d]"
      viewBox={`0 0 ${width} ${height}`}
    />
  );
}
