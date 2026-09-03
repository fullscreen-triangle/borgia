/**
 * Atlas — the derived periodic table and the corpus panels.
 *
 * These two views used to sit on the workbench, where the generator became
 * the page's landing screen and pushed the editor out of the way.  That was
 * the wrong call: the workbench is a tool for writing and running programs,
 * and a periodic table is not that tool.  They live here instead, where
 * they can have the whole width they need without displacing anything.
 */

import { useState } from "react";
import Head from "next/head";
import Link from "next/link";

import Generator from "@/components/workbench/Generator";
import PapersFull from "@/components/workbench/PapersFull";
import { RESULT_COUNT } from "@/components/workbench/PaperPanels";

const T = {
  bg: "#1a1b26", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", accent: "#7dcfff", surface: "#1e1f2e",
};
const MONO = "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace";

const VIEWS = [
  ["table", "periodic table", "All 118 atoms derived from Z at render time"],
  ["panels", `corpus panels (${RESULT_COUNT})`, "Paper figures, recomputed in-browser"],
];

export default function Atlas() {
  const [view, setView] = useState("table");

  return (
    <>
      <Head><title>Atlas — Honjo</title></Head>
      <div style={{
        height: "100vh", display: "flex", flexDirection: "column",
        background: T.bg, color: T.text, fontFamily: MONO, fontSize: 12,
      }}>
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "6px 12px", background: T.panel,
          borderBottom: `1px solid ${T.border}`, flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontWeight: 700, color: T.accent, letterSpacing: 0.5 }}>
              ATLAS
            </span>
            <div style={{ display: "flex", gap: 2, marginLeft: 4 }}>
              {VIEWS.map(([id, label, title]) => (
                <button key={id} onClick={() => setView(id)} title={title}
                  style={{
                    padding: "4px 12px", fontSize: 11, fontFamily: MONO,
                    background: view === id ? T.surface : "transparent",
                    color: view === id ? T.accent : T.dim,
                    border: `1px solid ${view === id ? T.border : "transparent"}`,
                    borderRadius: 4, cursor: "pointer",
                    fontWeight: view === id ? 700 : 400,
                  }}>{label}</button>
              ))}
            </div>
          </div>
          <Link href="/workbench" title="Write and run programs"
             style={{
               fontSize: 11, color: T.dim, textDecoration: "none",
               padding: "3px 8px", border: `1px solid ${T.border}`,
               borderRadius: 4,
             }}>&larr; workbench</Link>
        </div>

        <div style={{ flex: 1, overflow: "hidden" }}>
          {view === "table" ? <Generator /> : <PapersFull />}
        </div>
      </div>
    </>
  );
}
