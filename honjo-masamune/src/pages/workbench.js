/**
 * Workbench — an editor for the three languages, wired to a local engine.
 *
 * Design commitments, each of which is visible in the code below:
 *
 *   1. Every number displayed comes from src/data/*.json, which is
 *      generated from validation results by src/data/generate.py.
 *      No component contains a measured value as a literal.
 *
 *   2. Every result is labelled with the engine that produced it. The
 *      Rust reference compiler and the in-browser build are different
 *      implementations and are never presented as interchangeable.
 *
 *   3. The linter checks capability containment against the real
 *      declared sets, so a refusal shown in the editor is the same
 *      refusal the compiler would issue.
 */

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import Head from "next/head";
import Link from "next/link";

import { run as engineRun, STATUS, ENGINE, loadConnection } from "@/lib/engine";
import ConnectionPanel, { ConnectionBadge } from "@/components/workbench/ConnectionPanel";
import InterferencePanel from "@/components/workbench/InterferencePanel";
import AskPanel from "@/components/workbench/AskPanel";
import MASAMUNE from "@/data/masamune.json";
import HONJO from "@/data/honjo.json";
import MEIBUTSU from "@/data/meibutsu.json";
import { TUTORIALS } from "@/lib/tutorials";
import { lint } from "@/lib/lint";
import { runPlan } from "@/lib/plan";
import { runMbt } from "@/lib/mbt";
import RECORDS from "@/data/records.json";
import PaperPanels, { PAPER_NAMES, RESULT_COUNT } from "@/components/workbench/PaperPanels";
import RunCharts from "@/components/workbench/RunCharts";

const T = {
  bg: "#1a1b26", surface: "#1e1f2e", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", muted: "#3b3d57", accent: "#7dcfff",
  keyword: "#bb9af7", string: "#9ece6a", number: "#ff9e64", comment: "#565f89",
  err: "#f7768e", warn: "#e0af68", ok: "#9ece6a",
  stated: "#9ece6a", supplied: "#7aa2f7",
  hover: "#292b40", active: "#33354a",
};

const MONO = "'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace";

/* ---------------------------------------------------------------- */
/*  Syntax highlighting                                             */
/* ---------------------------------------------------------------- */

const KEYWORDS = [
  "plan", "source", "at", "let", "read", "translate", "require", "expect",
  "else", "report", "refuse", "emit", "with", "provenance", "select",
  "where", "assert", "budget", "join", "map", "via",
  "floor", "cut", "close", "deloc", "ring", "track", "until", "yield",
  "when", "do", "observe", "in", "as", "medium", "converge", "diverge",
  "by", "import", "module", "export", "reps",
  "meibutsu", "field", "spectrum", "grid", "superpose", "compare",
  "compute", "display", "invert", "query", "route", "reference",
];

const TYPES = [
  "smiles", "molfile", "sdf", "pdb", "inchi", "xyz", "graph",
  "element", "connectivity", "cellcount", "delocalisation", "charge",
  "isotope", "hcount", "stereo", "coords3d", "conformer",
  "stated", "supplied", "absent", "verdict",
  "amplitude", "phase", "energy", "visibility", "address",
];

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Tokenise then colour, rather than running regexes over the whole
 * string. Sequential replacement corrupts any keyword that happens to
 * appear inside a string or a comment; scanning left to right does not.
 */
function highlight(src) {
  if (!src) return "";
  const kw = new Set(KEYWORDS);
  const ty = new Set(TYPES);
  let out = "";
  let i = 0;
  const n = src.length;

  while (i < n) {
    const rest = src.slice(i);

    // comment to end of line
    const cm = rest.match(/^--[^\n]*/);
    if (cm) {
      out += `<span style="color:${T.comment};font-style:italic">${escapeHtml(cm[0])}</span>`;
      i += cm[0].length;
      continue;
    }
    // string
    const st = rest.match(/^"(?:[^"\\]|\\.)*"?/);
    if (st && rest[0] === '"') {
      out += `<span style="color:${T.string}">${escapeHtml(st[0])}</span>`;
      i += st[0].length;
      continue;
    }
    // number
    const nu = rest.match(/^\d+(?:\.\d+)?(?:[eE][-+]?\d+)?/);
    if (nu) {
      out += `<span style="color:${T.number}">${nu[0]}</span>`;
      i += nu[0].length;
      continue;
    }
    // identifier / keyword
    const id = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (id) {
      const w = id[0];
      if (kw.has(w)) {
        out += `<span style="color:${T.keyword};font-weight:600">${w}</span>`;
      } else if (ty.has(w)) {
        out += `<span style="color:${T.accent}">${w}</span>`;
      } else {
        out += escapeHtml(w);
      }
      i += w.length;
      continue;
    }
    out += escapeHtml(src[i]);
    i += 1;
  }
  return out;
}

/* ---------------------------------------------------------------- */
/*  Page                                                            */
/* ---------------------------------------------------------------- */

export default function Workbench() {
  const [leftW, setLeftW] = useState(230);
  const [rightW, setRightW] = useState(400);
  const [termH, setTermH] = useState(160);

  const [expanded, setExpanded] = useState(
    new Set(["honjo", "masamune", "meibutsu"])
  );
  /**
   * The workbench opens on a real program rather than an empty editor.
   * With no file open the Run button is disabled, which on a dark
   * surface reads as chrome rather than as a control — the page looked
   * like it had no way to execute anything. Landing on a runnable file
   * makes the first Run the obvious next action.
   */
  const FIRST_FILE = "honjo/01_cut.hnj";

  const [tabs, setTabs] = useState([FIRST_FILE]);
  const [active, setActive] = useState(FIRST_FILE);
  const [edits, setEdits] = useState({});
  const [dirty, setDirty] = useState(new Set());

  const [connection, setConnection] = useState({
    endpoint: "", token: "", status: STATUS.DISCONNECTED,
  });
  const [showConn, setShowConn] = useState(false);
  const [showAsk, setShowAsk] = useState(false);
  const [busy, setBusy] = useState(false);

  const [outTab, setOutTab] = useState("run");
  const [lastRun, setLastRun] = useState(null);
  const [lastPlan, setLastPlan] = useState(null);
  const [lastMbt, setLastMbt] = useState(null);
  const [term, setTerm] = useState([
    { k: "dim", t: "workbench ready — running in-browser engine" },
    { k: "ok", t: "press ▶ Run (or Ctrl+Enter) to execute the open program" },
    { k: "dim", t: "connect a local engine for the reference compiler" },
  ]);

  const termRef = useRef(null);
  const resize = useRef(null);

  useEffect(() => {
    const c = loadConnection();
    setConnection((p) => ({ ...p, endpoint: c.endpoint, token: c.token,
      status: c.token ? STATUS.CONNECTED : STATUS.DISCONNECTED }));
  }, []);

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [term]);

  /* ---- file access ---- */
  const contentOf = useCallback((path) => {
    if (edits[path] !== undefined) return edits[path];
    const [lang, file] = path.split("/");
    return TUTORIALS[lang]?.[file]?.source ?? "";
  }, [edits]);

  const openFile = useCallback((path) => {
    setTabs((p) => (p.includes(path) ? p : [...p, path]));
    setActive(path);
  }, []);

  const closeTab = useCallback((path, e) => {
    e.stopPropagation();
    setTabs((p) => {
      const next = p.filter((x) => x !== path);
      setActive((a) => (a === path ? next[next.length - 1] ?? null : a));
      return next;
    });
  }, []);

  /* ---- run ---- */
  const doRun = useCallback(async () => {
    if (!active || busy) return;
    const src = contentOf(active);
    const name = active.split("/").pop();
    const ext = name.slice(name.lastIndexOf("."));
    setBusy(true);

    const where = connection.token ? "local engine" : "browser engine";
    setTerm((p) => [...p, { k: "cmd", t: `run ${name}  (${where})` }]);

    // .msm runs through the plan runner, in the browser. There is no
    // Rust Masamune, so a local engine does not change this path and
    // the terminal says which engine actually ran.
    if (ext === ".msm") {
      let res;
      try {
        res = runPlan(src, RECORDS.files);
      } catch (err) {
        setTerm((p) => [...p, { k: "err", t: String(err.message || err) }]);
        setBusy(false);
        return;
      }
      setLastPlan(res);
      const lines = [];
      if (res.status === "parse-error") {
        lines.push({ k: "err", t: res.error });
      } else if (res.status === "refused") {
        const r = res.refusal;
        lines.push({ k: "warn", t: `refused at line ${r.step_line}: ${r.reason}` });
        lines.push({ k: "out", t: `  ${r.format} cannot state: ${r.missing_features.join(", ")}` });
        lines.push({ k: "out", t: `  it declares: ${r.source_capability.join(", ") || "nothing"}` });
        lines.push({ k: "dim", t: "  records read: 0 — no source was opened" });
      } else {
        for (const st of res.steps) {
          if (st.error) { lines.push({ k: "err", t: `${st.step}: ${st.error}` }); continue; }
          if (st.step === "read") lines.push({ k: "out", t: `read ${st.count} records into ${st.target}` });
          if (st.step === "translate") {
            const t = Object.entries(st.tally)
              .map(([k, v]) => `${v} ${k}`).join(", ");
            lines.push({ k: "out", t: `translate -> ${st.target}: ${t}` });
          }
          if (st.step === "select") lines.push({ k: "out", t: `select -> ${st.target}: ${st.kept} kept, ${st.dropped} dropped` });
          if (st.step === "assert") {
            lines.push({
              k: st.passed ? "ok" : "err",
              t: `assert ${st.condition.lhs} ${st.condition.op} ${st.condition.rhs}: observed ${st.observed}, ${st.passed ? "passed" : "FAILED"}`,
            });
          }
          if (st.step === "emit") lines.push({ k: "ok", t: `emit ${st.name}: ${st.emitted.length} records` });
        }
        for (const l of res.log || []) lines.push({ k: "warn", t: `${l.level}: ${l.message}` });
        lines.push({ k: "dim", t: `status: ${res.status} · engine: js-browser (no Rust Masamune)` });
      }
      setTerm((p) => [...p, ...lines]);
      setOutTab("plan");
      setBusy(false);
      return;
    }

    // .mbt runs through the field language. Every operation calls
    // into the field code that is checked against the reference, so a
    // program that runs here is running verified numerics.
    if (ext === ".mbt") {
      let res;
      try {
        res = runMbt(src);
      } catch (err) {
        setTerm((p) => [...p, { k: "err", t: String(err.message || err) }]);
        setBusy(false);
        return;
      }
      setLastMbt(res);
      const lines = [];
      if (res.status !== "ok") {
        lines.push({ k: "err", t: res.error });
      }
      for (const st of res.steps) {
        if (st.step === "grid") lines.push({ k: "dim", t: `grid ${st.value}` });
        if (st.step === "reference") lines.push({ k: "dim", t: `reference ${st.value} cm-1` });
        if (st.step === "spectrum") {
          lines.push({
            k: "out",
            t: `spectrum ${st.name}: ${st.modes.length} mode${st.modes.length === 1 ? "" : "s"}` +
               (st.from ? ` (from ${st.from})` : ""),
          });
        }
        if (st.step === "observe") {
          const c = st.coords.map((x) => x.toFixed(4)).join(", ");
          lines.push({ k: "out", t: `observe ${st.name}: coords (${c}) energy ${st.energy.toFixed(4)}` });
        }
        if (st.step === "superpose") {
          lines.push({
            k: st.visibility === 1 ? "ok" : "out",
            t: `superpose ${st.a} ${st.b}: V = ${st.visibility.toFixed(6)}`,
          });
          lines.push({
            k: "dim",
            t: `  own ${st.own_energy.toFixed(3)} · relational ${st.relational.toFixed(3)} · ` +
               `${st.constructive} constructive / ${st.destructive} destructive`,
          });
        }
        if (st.step === "invert") {
          lines.push({ k: "out", t: `invert ${st.query} against ${st.n_reference} references:` });
          for (const r of st.ranked) {
            lines.push({ k: "dim", t: `  ${r.name.padEnd(8)} V = ${r.visibility.toFixed(6)}` });
          }
        }
        if (st.step === "report") {
          for (const row of st.rows) {
            const bits = [];
            if (row.coordinates) bits.push(`(${row.coordinates.map((x) => x.toFixed(4)).join(", ")})`);
            if (row.energy !== undefined) bits.push(`E ${row.energy.toFixed(4)}`);
            if (row.peak) bits.push(`peak ${row.peak.amplitude.toFixed(3)} at u=${row.peak.at.toFixed(3)}`);
            lines.push({ k: "out", t: `  ${row.name}: ${bits.join("  ")}` });
          }
        }
      }
      lines.push({ k: "dim", t: `status: ${res.status} · engine: js-browser` });
      setTerm((p) => [...p, ...lines]);
      setOutTab("interference");
      setBusy(false);
      return;
    }

    const res = await engineRun({
      source: src,
      endpoint: connection.endpoint,
      token: connection.token,
    });
    setLastRun(res);

    if (!res.ok) {
      setTerm((p) => [...p,
        { k: "err", t: res.error || "failed" },
        { k: "dim", t: `engine: ${res.engine}${res.stage ? ` · stage: ${res.stage}` : ""}` },
      ]);
    } else {
      setTerm((p) => [...p,
        ...res.log.map((l) => ({ k: "out", t: l })),
        { k: "ok", t: `cut count M = ${res.cutCount} · floor = ${res.floor}` },
        { k: "dim", t: `engine: ${res.engine}` },
      ]);
    }
    setOutTab("run");
    setBusy(false);
  }, [active, busy, contentOf, connection]);

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); doRun(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doRun]);

  /* ---- resizing ---- */
  const startResize = useCallback((which, e) => {
    e.preventDefault();
    resize.current = { which, x: e.clientX, y: e.clientY, leftW, rightW, termH };
    const move = (ev) => {
      const r = resize.current;
      if (!r) return;
      if (r.which === "left") setLeftW(clamp(r.leftW + ev.clientX - r.x, 170, 380));
      if (r.which === "right") setRightW(clamp(r.rightW - (ev.clientX - r.x), 280, 640));
      if (r.which === "term") setTermH(clamp(r.termH - (ev.clientY - r.y), 90, 420));
    };
    const up = () => {
      resize.current = null;
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }, [leftW, rightW, termH]);

  const src = active ? contentOf(active) : "";
  const markers = useMemo(
    () => (active ? lint(src, active.split(".").pop(), MASAMUNE.capability) : []),
    [src, active]
  );
  const markerByLine = useMemo(() => {
    const m = {};
    markers.forEach((k) => { (m[k.line] ||= []).push(k); });
    return m;
  }, [markers]);

  const tut = active ? TUTORIALS[active.split("/")[0]]?.[active.split("/")[1]] : null;

  return (
    <>
      <Head><title>Workbench — Honjo Masamune</title></Head>

      <ConnectionPanel
        open={showConn}
        onClose={() => setShowConn(false)}
        connection={connection}
        onChange={setConnection}
      />

      <div style={{
        height: "100vh", display: "flex", flexDirection: "column",
        position: "relative",
        background: T.bg, color: T.text, fontFamily: MONO, fontSize: 13,
        overflow: "hidden",
      }}>
        {/* top bar */}
        <div style={{
          height: 38, display: "flex", alignItems: "center",
          justifyContent: "space-between", padding: "0 12px",
          background: T.panel, borderBottom: `1px solid ${T.border}`,
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontWeight: 700, color: T.accent, letterSpacing: 0.5 }}>
              HONJO
            </span>
            <Link href="/atlas" title="Periodic table and corpus panels"
               style={{
                 fontSize: 11, fontFamily: MONO, color: T.dim,
                 textDecoration: "none", padding: "3px 8px",
                 border: `1px solid ${T.border}`, borderRadius: 4,
               }}>atlas &rarr;</Link>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button
              onClick={() => setShowAsk((v) => !v)}
              title="Ask about a measured result"
              style={{
                padding: "3px 10px", background: "transparent",
                border: `1px solid ${T.border}`, borderRadius: 4,
                color: showAsk ? T.accent : T.dim, fontSize: 11,
                cursor: "pointer", fontFamily: MONO,
              }}
            >ask</button>
            <ConnectionBadge status={connection.status} onClick={() => setShowConn(true)} />
            <button
              onClick={doRun}
              disabled={busy || !active}
              title={active ? "Run this program (Ctrl+Enter)" : "Open a file to run"}
              style={{
                padding: "5px 16px",
                background: busy || !active ? "transparent" : T.ok,
                color: busy || !active ? T.dim : T.bg,
                border: busy || !active ? `1px solid ${T.border}` : "none",
                borderRadius: 4,
                fontSize: 11, fontWeight: 700, fontFamily: MONO,
                cursor: busy || !active ? "default" : "pointer",
                minWidth: 96,
              }}
            >
              {busy ? "running…" : "▶ Run"}
            </button>
          </div>
        </div>

        <AskPanel open={showAsk} onClose={() => setShowAsk(false)} />

        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          {/* explorer */}
          <div style={{
            width: leftW, background: T.surface,
            borderRight: `1px solid ${T.border}`, overflowY: "auto",
            flexShrink: 0,
          }}>
            <div style={{
              padding: "8px 12px", fontSize: 10, letterSpacing: 1,
              textTransform: "uppercase", color: T.dim,
            }}>
              Tutorials
            </div>
            {Object.entries(TUTORIALS).map(([lang, files]) => (
              <div key={lang}>
                <div
                  onClick={() => setExpanded((p) => {
                    const s = new Set(p);
                    s.has(lang) ? s.delete(lang) : s.add(lang);
                    return s;
                  })}
                  style={{
                    padding: "4px 12px", cursor: "pointer", fontSize: 12,
                    display: "flex", alignItems: "center", gap: 6,
                    userSelect: "none",
                  }}
                >
                  <span style={{
                    display: "inline-block", width: 8,
                    transform: expanded.has(lang) ? "rotate(90deg)" : "none",
                    transition: "transform .15s", color: T.dim,
                  }}>▸</span>
                  <span style={{ color: LANG_COLOR[lang] }}>{lang}</span>
                </div>
                {expanded.has(lang) && Object.keys(files).map((f) => {
                  const path = `${lang}/${f}`;
                  const on = active === path;
                  return (
                    <div
                      key={path}
                      onClick={() => openFile(path)}
                      style={{
                        padding: "3px 12px 3px 32px", cursor: "pointer",
                        fontSize: 11.5, color: on ? T.text : T.dim,
                        background: on ? T.active : "transparent",
                        whiteSpace: "nowrap", overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {f}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          <Grip onMouseDown={(e) => startResize("left", e)} vertical />

          {/* editor + terminal */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
            {/* tabs */}
            <div style={{
              height: 32, display: "flex", background: T.surface,
              borderBottom: `1px solid ${T.border}`, overflowX: "auto",
              flexShrink: 0,
            }}>
              {tabs.map((p) => {
                const on = p === active;
                return (
                  <div
                    key={p}
                    onClick={() => setActive(p)}
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "0 10px", fontSize: 11.5, cursor: "pointer",
                      background: on ? T.bg : "transparent",
                      color: on ? T.text : T.dim,
                      borderRight: `1px solid ${T.border}`,
                      borderTop: on ? `1px solid ${T.accent}` : "1px solid transparent",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {p.split("/").pop()}
                    <span
                      onClick={(e) => closeTab(p, e)}
                      style={{ color: T.muted, fontSize: 14, lineHeight: 1 }}
                    >×</span>
                  </div>
                );
              })}
            </div>

            {/* editor */}
            <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
              {!active ? (
                <Empty />
              ) : (
                <Editor
                  src={src}
                  markerByLine={markerByLine}
                  onChange={(v) => {
                    setEdits((p) => ({ ...p, [active]: v }));
                    setDirty((p) => new Set(p).add(active));
                  }}
                />
              )}
            </div>

            <Grip onMouseDown={(e) => startResize("term", e)} />

            {/* terminal */}
            <div style={{
              height: termH, background: T.surface,
              borderTop: `1px solid ${T.border}`, display: "flex",
              flexDirection: "column", flexShrink: 0,
            }}>
              <div style={{
                padding: "5px 12px", fontSize: 10, letterSpacing: 1,
                textTransform: "uppercase", color: T.dim,
                borderBottom: `1px solid ${T.border}`,
                display: "flex", justifyContent: "space-between",
              }}>
                <span>Output</span>
                <span
                  onClick={() => setTerm([])}
                  style={{ cursor: "pointer" }}
                >clear</span>
              </div>
              <div ref={termRef} style={{
                flex: 1, overflowY: "auto", padding: "6px 12px",
                fontSize: 11.5, lineHeight: 1.65,
              }}>
                {term.map((l, i) => (
                  <div key={i} style={{ color: TERM_COLOR[l.k] || T.text, whiteSpace: "pre-wrap" }}>
                    {l.k === "cmd" ? `$ ${l.t}` : l.t}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <Grip onMouseDown={(e) => startResize("right", e)} vertical />

          {/* results */}
          <div style={{
            width: rightW, background: T.surface,
            borderLeft: `1px solid ${T.border}`, display: "flex",
            flexDirection: "column", flexShrink: 0,
          }}>
            <div style={{
              display: "flex", borderBottom: `1px solid ${T.border}`,
              overflowX: "auto", flexShrink: 0,
            }}>
              {[
                ["run", "Run"],
                ["charts", "Charts"],
                ["plan", "Plan"],
                ["supplied", "Provenance"],
                ["capability", "Capability"],
                ["conformance", "Conformance"],
                ["interference", "Interference"],
                ["meibutsu", "Bulk"],
                ["papers", `Papers (${RESULT_COUNT})`],
              ].map(([id, label]) => (
                <div
                  key={id}
                  onClick={() => setOutTab(id)}
                  style={{
                    padding: "7px 12px", fontSize: 11, cursor: "pointer",
                    color: outTab === id ? T.text : T.dim,
                    borderBottom: outTab === id ? `2px solid ${T.accent}` : "2px solid transparent",
                    whiteSpace: "nowrap",
                  }}
                >{label}</div>
              ))}
            </div>
            <div style={{ flex: 1, overflowY: "auto" }}>
              <Results tab={outTab} lastRun={lastRun} lastPlan={lastPlan}
                       tutorial={tut} />
            </div>
          </div>
        </div>

        {/* status bar */}
        <div style={{
          height: 22, background: T.panel, borderTop: `1px solid ${T.border}`,
          display: "flex", alignItems: "center", padding: "0 12px", gap: 14,
          fontSize: 10.5, color: T.dim, flexShrink: 0,
        }}>
          <span>{active || "no file"}</span>
          {markers.length > 0 && (
            <span style={{ color: markers.some((m) => m.type === "error") ? T.err : T.warn }}>
              {markers.length} issue{markers.length > 1 ? "s" : ""}
            </span>
          )}
          <span style={{ flex: 1 }} />
          <span>
            engine:{" "}
            <span style={{ color: connection.token ? T.ok : T.dim }}>
              {connection.token ? "rust (local)" : "javascript (browser)"}
            </span>
          </span>
        </div>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- */

const LANG_COLOR = { masamune: T.keyword, honjo: T.accent, meibutsu: T.number };
const TERM_COLOR = {
  cmd: T.accent, ok: T.ok, err: T.err, warn: T.warn,
  dim: T.dim, out: T.text,
};

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function Grip({ onMouseDown, vertical }) {
  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        [vertical ? "width" : "height"]: 4,
        cursor: vertical ? "col-resize" : "row-resize",
        background: "transparent", flexShrink: 0,
      }}
    />
  );
}

function Empty() {
  return (
    <div style={{
      height: "100%", display: "flex", alignItems: "center",
      justifyContent: "center", flexDirection: "column", gap: 8,
      color: T.muted, fontSize: 12,
    }}>
      <div>Open a tutorial to begin</div>
      <div style={{ fontSize: 11 }}>Ctrl+Enter to run</div>
    </div>
  );
}

function Editor({ src, markerByLine, onChange }) {
  const lines = src.split("\n");
  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", overflow: "auto" }}>
      {/* gutter */}
      <div style={{
        padding: "10px 0", background: T.bg, color: T.muted,
        fontSize: 12, lineHeight: "19px", textAlign: "right",
        userSelect: "none", flexShrink: 0, minWidth: 46,
      }}>
        {lines.map((_, i) => {
          const mk = markerByLine[i];
          return (
            <div key={i} style={{ padding: "0 8px 0 10px", position: "relative" }}>
              {mk && (
                <span
                  title={mk.map((m) => m.msg).join("\n")}
                  style={{
                    position: "absolute", left: 2,
                    color: mk.some((m) => m.type === "error") ? T.err : T.warn,
                  }}
                >●</span>
              )}
              {i + 1}
            </div>
          );
        })}
      </div>
      {/* text */}
      <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
        <pre
          aria-hidden
          style={{
            margin: 0, padding: "10px 12px", fontSize: 12, lineHeight: "19px",
            fontFamily: MONO, whiteSpace: "pre", pointerEvents: "none",
          }}
          dangerouslySetInnerHTML={{ __html: highlight(src) + "\n" }}
        />
        <textarea
          value={src}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          style={{
            position: "absolute", inset: 0, width: "100%", height: "100%",
            padding: "10px 12px", margin: 0, border: "none", outline: "none",
            resize: "none", background: "transparent", color: "transparent",
            caretColor: T.text, fontSize: 12, lineHeight: "19px",
            fontFamily: MONO, whiteSpace: "pre", overflow: "hidden",
          }}
        />
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- */
/*  Result views — all values from generated data                   */
/* ---------------------------------------------------------------- */

function Results({ tab, lastRun, lastPlan, tutorial }) {
  if (tab === "run") return <RunView res={lastRun} tutorial={tutorial} />;
  if (tab === "charts") return <RunCharts res={lastRun} />;
  if (tab === "plan") return <PlanView res={lastPlan} tutorial={tutorial} />;
  if (tab === "supplied") return <SuppliedView />;
  if (tab === "capability") return <CapabilityView />;
  if (tab === "conformance") return <ConformanceView />;
  if (tab === "interference") return <InterferencePanel width={380} />;
  if (tab === "meibutsu") return <MeibutsuView />;
  if (tab === "papers") return <PapersView />;
  return null;
}

/**
 * The corpus figures, rendered live from the same results the papers plot.
 * The ladder tab additionally recomputes its invariants in the browser and
 * diffs them against the committed values.
 */
function PapersView() {
  const [paper, setPaper] = useState(
    PAPER_NAMES.includes("categorical-ladder") ? "categorical-ladder" : PAPER_NAMES[0]
  );
  if (!PAPER_NAMES.length) {
    return <Section title="No panel data" sub="Run python src/data/generate.py." />;
  }
  return (
    <div style={{ padding: 10 }}>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 12 }}>
        {PAPER_NAMES.map((n) => (
          <button
            key={n}
            onClick={() => setPaper(n)}
            style={{
              padding: "3px 9px", fontSize: 10, fontFamily: MONO,
              background: n === paper ? T.accent : "transparent",
              color: n === paper ? T.bg : T.dim,
              border: `1px solid ${n === paper ? T.accent : T.border}`,
              borderRadius: 3, cursor: "pointer",
            }}
          >{n.replace(/-/g, " ")}</button>
        ))}
      </div>
      <PaperPanels paper={paper} />
    </div>
  );
}


function Section({ title, sub, children }) {
  return (
    <div style={{ padding: 14 }}>
      <div style={{
        fontSize: 10, letterSpacing: 1, textTransform: "uppercase",
        color: T.dim, marginBottom: 4,
      }}>{title}</div>
      {sub && <div style={{ fontSize: 11.5, color: T.text, marginBottom: 12, lineHeight: 1.5 }}>{sub}</div>}
      {children}
    </div>
  );
}

function RunView({ res, tutorial }) {
  if (!res) {
    return (
      <Section title="No run yet" sub="Press Run, or Ctrl+Enter.">
        {tutorial?.expect && (
          <div style={{
            fontSize: 11, color: T.dim, lineHeight: 1.7,
            background: T.bg, border: `1px solid ${T.border}`,
            borderRadius: 4, padding: 10, whiteSpace: "pre-wrap",
          }}>
            <div style={{ color: T.warn, marginBottom: 6 }}>expected</div>
            {tutorial.expect}
          </div>
        )}
      </Section>
    );
  }
  return (
    <Section
      title="Run result"
      sub={
        <>
          {res.ok ? "completed" : "failed"} ·{" "}
          <span style={{ color: res.engine === ENGINE.LOCAL ? T.ok : T.warn }}>
            {res.engine === ENGINE.LOCAL ? "rust reference engine" : "browser engine"}
          </span>
        </>
      }
    >
      {res.error && (
        <div style={{
          fontSize: 11.5, color: T.err, background: T.bg,
          border: `1px solid ${T.err}`, borderRadius: 4, padding: 10,
          marginBottom: 10, whiteSpace: "pre-wrap",
        }}>{res.error}</div>
      )}
      {res.ok && (
        <>
          <KV k="cut count M" v={res.cutCount} />
          <KV k="floor" v={res.floor} />
          <KV k="bindings" v={Object.keys(res.named).length} />
          {res.log.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 10, color: T.dim, marginBottom: 4 }}>LOG</div>
              {res.log.map((l, i) => (
                <div key={i} style={{
                  fontSize: 11, color: T.text, background: T.bg,
                  padding: "5px 8px", borderRadius: 3, marginBottom: 3,
                  whiteSpace: "pre-wrap",
                }}>{l}</div>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  );
}


/**
 * The plan trace.
 *
 * A plan is a statement of what was asked for and in what order, so the
 * trace is the result — not just the final record set. Each step shows
 * what it did, and a refusal shows what was missing and what the format
 * does declare.
 */
function PlanView({ res, tutorial }) {
  if (!res) {
    return (
      <Section title="No plan run yet" sub="Press Run, or Ctrl+Enter.">
        {tutorial?.expect && (
          <div style={{
            fontSize: 11, color: T.dim, lineHeight: 1.7, background: T.bg,
            border: `1px solid ${T.border}`, borderRadius: 4, padding: 10,
            whiteSpace: "pre-wrap",
          }}>
            <div style={{ color: T.warn, marginBottom: 6 }}>expected</div>
            {tutorial.expect}
          </div>
        )}
      </Section>
    );
  }

  if (res.status === "parse-error") {
    return (
      <Section title="Plan" sub="did not parse">
        <div style={{
          fontSize: 11.5, color: T.err, background: T.bg,
          border: `1px solid ${T.err}`, borderRadius: 4, padding: 10,
          whiteSpace: "pre-wrap",
        }}>{res.error}</div>
      </Section>
    );
  }

  if (res.status === "refused") {
    const r = res.refusal;
    return (
      <Section
        title="Refused"
        sub={`before any record was read — ${res.records_read} opened`}
      >
        <KV k="reason" v={r.reason} />
        <KV k="line" v={r.step_line} />
        <KV k="source" v={`${r.source} : ${r.format}`} />
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 10, color: T.dim, marginBottom: 4 }}>
            REQUESTED BUT NOT DECLARED
          </div>
          {r.missing_features.map((f) => (
            <span key={f} style={chip(T.err)}>{f}</span>
          ))}
          <div style={{ fontSize: 10, color: T.dim, margin: "10px 0 4px" }}>
            {r.format.toUpperCase()} DECLARES
          </div>
          {r.source_capability.length === 0 ? (
            <span style={{ ...chip(T.muted), color: T.dim }}>nothing</span>
          ) : (
            r.source_capability.map((f) => (
              <span key={f} style={chip(T.stated)}>{f}</span>
            ))
          )}
        </div>
        <div style={{
          fontSize: 10, color: T.muted, marginTop: 12, lineHeight: 1.6,
          borderLeft: `2px solid ${T.warn}`, paddingLeft: 8,
        }}>
          The refusal is static: it follows from the declared capability
          set alone, so no file was opened and no record was parsed.
        </div>
      </Section>
    );
  }

  const emitted = res.steps.filter((s) => s.step === "emit")
    .flatMap((s) => s.emitted || []);

  return (
    <Section
      title={`Plan ${res.plan || ""}`}
      sub={
        <>
          {res.status} · {res.records_read} record
          {res.records_read === 1 ? "" : "s"} read ·{" "}
          <span style={{ color: T.warn }}>js-browser</span>
        </>
      }
    >
      {res.steps.map((st, i) => (
        <div key={i} style={{
          padding: "6px 0", borderBottom: `1px solid ${T.border}`,
          fontSize: 11,
        }}>
          <div style={{ display: "flex", gap: 8 }}>
            <span style={{ color: T.muted, width: 26, flexShrink: 0 }}>
              {st.line}
            </span>
            <span style={{ color: T.accent, width: 66, flexShrink: 0 }}>
              {st.step}
            </span>
            <span style={{ flex: 1, color: st.error ? T.err : T.text }}>
              {st.error ? st.error : stepSummary(st)}
            </span>
          </div>
          {st.step === "translate" && (
            <div style={{ marginLeft: 100, marginTop: 4 }}>
              {Object.entries(st.tally).map(([label, n]) => (
                <span key={label}
                      style={chip(label === "translated" ? T.stated : T.warn)}>
                  {n} {label}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}

      {(res.log || []).map((l, i) => (
        <div key={i} style={{
          fontSize: 10.5, color: T.warn, marginTop: 6,
          borderLeft: `2px solid ${T.warn}`, paddingLeft: 8,
        }}>
          line {l.step_line}: {l.message}
        </div>
      ))}

      {emitted.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 10, color: T.dim, marginBottom: 6 }}>
            EMITTED — {emitted.length} record{emitted.length === 1 ? "" : "s"}
          </div>
          {emitted.slice(0, 40).map((e, i) => (
            <div key={i} style={{ marginBottom: 4 }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 10.5, color: T.dim,
              }}>
                <span style={{ color: T.text }}>{e.record}</span>
                <span>
                  {e.payload?.supplied_fraction !== undefined
                    ? `φ ${e.payload.supplied_fraction.toFixed(3)}`
                    : e.verdict}
                </span>
              </div>
              {e.payload?.supplied_fraction !== undefined && (
                <div style={{
                  height: 5, background: T.bg, borderRadius: 3,
                  overflow: "hidden",
                }}>
                  <div style={{
                    width: `${e.payload.supplied_fraction * 100}%`,
                    height: "100%",
                    background: e.payload.supplied_fraction === 0
                      ? T.stated : T.supplied,
                  }} />
                </div>
              )}
            </div>
          ))}
          {emitted.length > 40 && (
            <div style={{ fontSize: 10, color: T.muted, marginTop: 4 }}>
              {emitted.length - 40} more not shown
            </div>
          )}
        </div>
      )}
    </Section>
  );
}

function stepSummary(st) {
  if (st.step === "read") return `${st.count} → ${st.target}`;
  if (st.step === "translate") return `→ ${st.target}`;
  if (st.step === "select") return `${st.kept} kept, ${st.dropped} dropped → ${st.target}`;
  if (st.step === "assert") {
    return `${st.condition.lhs} ${st.condition.op} ${st.condition.rhs} — observed ${st.observed}, ${st.passed ? "passed" : "failed"}`;
  }
  if (st.step === "emit") return `${st.name}: ${st.emitted.length}`;
  return "";
}

function chip(color) {
  return {
    display: "inline-block", padding: "1px 6px", marginRight: 4,
    marginBottom: 3, fontSize: 9.5, borderRadius: 3,
    background: "transparent", color, border: `1px solid ${color}`,
  };
}

function KV({ k, v }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      padding: "4px 0", borderBottom: `1px solid ${T.border}`,
      fontSize: 11.5,
    }}>
      <span style={{ color: T.dim }}>{k}</span>
      <span>{String(v)}</span>
    </div>
  );
}

function SuppliedView() {
  const s = MASAMUNE.supplied;
  const rows = [...s.rows].sort((a, b) => a.phi - b.phi);
  return (
    <Section
      title="Supplied fraction φ"
      sub={`Mean ${s.mean.toFixed(3)} over ${s.n_structures} structures — the majority of a SMILES-derived graph is convention, not record.`}
    >
      <div style={{ display: "flex", gap: 12, marginBottom: 10, fontSize: 11 }}>
        <Stat label="min" v={s.min.toFixed(3)} />
        <Stat label="median" v={s.median.toFixed(3)} />
        <Stat label="max" v={s.max.toFixed(3)} />
        <Stat label="σ" v={s.stdev.toFixed(3)} />
      </div>
      {rows.map((r) => (
        <div key={r.molecule} style={{ marginBottom: 3 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            fontSize: 10.5, color: T.dim, marginBottom: 1,
          }}>
            <span>{r.molecule}</span>
            <span>{r.phi.toFixed(3)}</span>
          </div>
          <div style={{ height: 6, background: T.bg, borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              width: `${r.phi * 100}%`, height: "100%",
              background: T.supplied,
            }} />
          </div>
        </div>
      ))}
      <div style={{ fontSize: 10, color: T.muted, marginTop: 10, lineHeight: 1.6 }}>
        No structure reaches φ = 0: every SMILES record in this corpus omits
        at least half of what the contact graph ends up containing, because
        the notation states no hydrogens explicitly.
      </div>
    </Section>
  );
}

function Stat({ label, v }) {
  return (
    <div>
      <div style={{ color: T.dim, fontSize: 9.5 }}>{label}</div>
      <div style={{ color: T.text }}>{v}</div>
    </div>
  );
}

function CapabilityView() {
  const { features, capability, containment } = MASAMUNE;
  const formats = Object.keys(capability);
  return (
    <Section
      title="Declared capability"
      sub={`${containment.n_refused_statically} of ${containment.n_pairs} requests decided before any record is read; static and post-read verdicts agree on ${containment.n_consistent}/${containment.n_pairs}.`}
    >
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left", padding: "3px 6px", color: T.dim }} />
              {formats.map((f) => (
                <th key={f} style={{ padding: "3px 5px", color: T.dim, fontWeight: 400 }}>{f}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {features.map((feat) => (
              <tr key={feat}>
                <td style={{ padding: "2px 6px", color: T.text, whiteSpace: "nowrap" }}>{feat}</td>
                {formats.map((f) => {
                  const has = capability[f].includes(feat);
                  return (
                    <td key={f} style={{ padding: "2px 5px", textAlign: "center" }}>
                      <span style={{
                        display: "inline-block", width: 9, height: 9,
                        borderRadius: 2, background: has ? T.stated : T.muted,
                      }} />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 10, color: T.muted, marginTop: 10, lineHeight: 1.6 }}>
        <strong style={{ color: T.warn }}>inchi</strong> declares nothing: no
        reader is implemented, so every request against it is refused
        statically. Under-declaring is safe; over-declaring would be unsound
        and silent. Note also that <strong>stereo</strong> is declared by no
        format here — the SMILES reader parses stereo tokens but builds no
        descriptors, so declaring it would be an over-declaration.
      </div>
    </Section>
  );
}

function ConformanceView() {
  return (
    <Section
      title="Conformance suite"
      sub="The eight items of the specification, executed against the reference interpreter."
    >
      {HONJO.conformance.map((c) => (
        <div key={c.item} style={{
          display: "flex", gap: 8, padding: "6px 0",
          borderBottom: `1px solid ${T.border}`, fontSize: 11,
        }}>
          <span style={{ color: T.accent, width: 24, flexShrink: 0 }}>{c.item}</span>
          <div style={{ flex: 1 }}>
            <div>{c.name}</div>
            <div style={{ color: T.dim, fontSize: 10 }}>
              {c.measured} · {c.programs} program{c.programs === 1 ? "" : "s"}
            </div>
          </div>
        </div>
      ))}
      <div style={{ fontSize: 10, color: T.muted, marginTop: 10, lineHeight: 1.6 }}>
        Two items could not have failed as first written and were restated:
        C1 (residue is clamped, so a sub-floor value is unconstructible) and
        C4 (two of the four programs bound values and so were not failures).
      </div>
    </Section>
  );
}

function MeibutsuView() {
  const m = MEIBUTSU;
  const cap = m.capacity.filter((r) => r.demod_vs_true_correlation !== null);
  return (
    <Section
      title="Interference"
      sub={`Self-comparison is exactly 1 for all ${m.self_visibility.n} structures (max deviation ${m.self_visibility.max_deviation}); inversion ranks the true structure first ${m.inversion.ranked_first}/${m.inversion.n}.`}
    >
      <div style={{ fontSize: 10, color: T.dim, marginBottom: 6 }}>
        BULK RECOVERY — correlation with true pairwise visibility
      </div>
      {cap.map((r) => {
        const v = r.demod_vs_true_correlation;
        const w = Math.min(Math.abs(v) * 140, 100);
        return (
          <div key={r.stack_size} style={{ marginBottom: 4 }}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              fontSize: 10, color: T.dim,
            }}>
              <span>n = {r.stack_size}</span>
              <span style={{ color: v > 0 ? T.ok : T.err }}>{v.toFixed(3)}</span>
            </div>
            <div style={{
              height: 6, background: T.bg, borderRadius: 3,
              position: "relative", overflow: "hidden",
            }}>
              <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: T.muted }} />
              <div style={{
                position: "absolute", top: 0, bottom: 0,
                left: v > 0 ? "50%" : `${50 - w / 2}%`,
                width: `${w / 2}%`,
                background: v > 0 ? T.ok : T.err,
              }} />
            </div>
          </div>
        );
      })}
      <div style={{
        fontSize: 10, color: T.muted, marginTop: 10, lineHeight: 1.6,
        borderLeft: `2px solid ${T.err}`, paddingLeft: 8,
      }}>
        The algebraic identity holds exactly — one superposition reproduces
        the sum over all {m.bulk_identity.n_pairs_implied} pairs with residual{" "}
        {m.bulk_identity.relative_residual}. But the cross-term bases are
        non-orthogonal, so no individual pair is recoverable from the stack.
        Bulk comparison is refuted, not supported.
      </div>
    </Section>
  );
}
