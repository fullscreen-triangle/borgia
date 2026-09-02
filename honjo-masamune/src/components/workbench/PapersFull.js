/**
 * The corpus panels at full width.
 *
 * The same `PaperPanels` the results sidebar renders, but given the room the
 * charts were drawn for. This exists because the sidebar version is 400px wide
 * and the panels are laid out at 460 — which meant the figures were technically
 * present and practically unreadable. A chart that has to be scrolled sideways
 * inside a column has not been shown to anyone.
 */

import { useState } from "react";
import PaperPanels, { PAPER_NAMES, RESULT_COUNT, FIGURE_COUNT } from "./PaperPanels";
import { CHART_THEME as T, CHART_MONO as MONO } from "./Charts";

export default function PapersFull() {
  const [paper, setPaper] = useState(
    PAPER_NAMES.includes("categorical-ladder") ? "categorical-ladder" : PAPER_NAMES[0]
  );

  if (!PAPER_NAMES.length) {
    return (
      <div style={{ padding: 24, color: T.dim, fontSize: 12, fontFamily: MONO }}>
        No panel data. Run <span style={{ color: T.text }}>python src/data/generate.py</span>.
      </div>
    );
  }

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: T.text, fontFamily: MONO }}>
          Corpus panels
        </div>
        <div style={{ fontSize: 11.5, color: T.dim, marginTop: 5, lineHeight: 1.6, maxWidth: 900 }}>
          {RESULT_COUNT} result sets across {PAPER_NAMES.length} papers, drawn from the
          same JSON the papers plot — {FIGURE_COUNT} figures. The ladder panels
          recompute their invariants here in the browser and print the residual
          against the committed value, so a drift between the implementation and
          the paper shows up on the panel rather than waiting for someone to rerun
          Python.
        </div>
      </div>

      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 14 }}>
        {PAPER_NAMES.map((n) => (
          <button key={n} onClick={() => setPaper(n)} style={{
            padding: "4px 10px", fontSize: 10.5, fontFamily: MONO,
            background: n === paper ? T.accent : "transparent",
            color: n === paper ? T.bg : T.dim,
            border: `1px solid ${n === paper ? T.accent : T.border}`,
            borderRadius: 3, cursor: "pointer",
          }}>{n.replace(/-/g, " ")}</button>
        ))}
      </div>

      <div style={{
        display: "grid", gap: 14,
        gridTemplateColumns: "repeat(auto-fill, minmax(440px, 1fr))",
        alignItems: "start",
      }}>
        <PaperPanels paper={paper} />
      </div>
    </div>
  );
}
