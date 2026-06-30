/**
 * /playground — the Honjo Masamune language, in the browser.
 * =========================================================
 * A live editor for `.hj` scripts. The honjo TypeScript compiler
 * (src/lib/honjo.js, bundled from ../honjo) runs entirely client-side:
 * source -> lex -> parse -> accountability check -> Cut-IR -> exact interpreter.
 *
 * Every operation is a cut. `cut N` individuates an atom; `a ~ b` is a bond;
 * `close X(...)` drives vacancies to zero; `track x in P ... yield r` propagates.
 * Nothing is free: every value carries the floor it was cut at, and the cut
 * count M is a monotone clock.
 */
import { useState, useCallback, useMemo } from "react";
import Head from "next/head";
import Link from "next/link";
import { evaluate, compile } from "@/lib/honjo";

const ACCENT = "#58E6D9";

const EXAMPLES = {
  carbon: `-- carbon.hj — individuating carbon (a length-one cut)
floor 1.0

C := cut 6                -- individuate Z=6 against the medium
observe C                 -- force the measurement now
`,
  water: `-- water.hj — cuts to closure, geometry by maximal separation
floor 1.0

O := cut 8
H := cut 1

-- a bond is a cut between two items, admitted only if it lowers thickness
OH := O ~ H when delta > 0
observe OH

-- close drives every vacancy to zero: stoichiometry + geometry follow
W := close O(H, H)        -- 2:1, bent, ~104.5 deg
observe W
assert W.valence == closed emit "water did not close"
`,
  track: `-- track.hj — tracking an item through a process (the causal table)
floor 1.0
import honjo.causal

O := cut 8
H := cut 1
W := close O(H, H)

-- propagate O's uncertainty through the process; admissible iff it
-- S-entropy-converges to the observed output. representation switching allowed.
path := track O in W
          with reps mass, charge, time
          until converge
          yield amalgamation

observe path              -- the amalgamation IS the result
`,
  salt: `-- salt.hj — ionic contact and a forbidden bond
floor 1.0

Na := cut 11
Cl := cut 17
observe Na
observe Cl

NaCl := close Na(Cl)      -- 1:1
observe NaCl

-- a closed-shell partner forms no bond (Ne has vacancy 0)
Ne := cut 10
dead := Na ~ Ne when delta > 0
observe dead              -- exists=false
`,
};

const TYPE_COLOR = {
  Atom: "#3b82f6",
  Bond: "#22c55e",
  Compound: "#f97316",
  Path: "#a855f7",
  Scalar: "#6b7280",
};

function ValueCard({ name, v }) {
  const color = TYPE_COLOR[v.ty] || "#6b7280";
  const rows = useMemo(() => {
    switch (v.ty) {
      case "Atom":
        return [
          ["Z", v.Z], ["symbol", v.symbol], ["config", v.config],
          ["term", v.term], ["vacancy", v.vacancy], ["valence", v.valence],
        ];
      case "Bond":
        return [
          ["pair", `${v.a}~${v.b}`], ["exists", String(v.exists)],
          ["Δthickness", fmt(v.delta)], ["shared", v.shared],
        ];
      case "Compound": {
        const lig = v.formula[1] > 1 ? v.ligand + v.formula[1] : v.formula[1] === 1 ? v.ligand : "";
        const formula = v.formula[0] === 2 ? v.central + "₂" : v.central + lig;
        return [
          ["formula", formula], ["geometry", v.geometry],
          ["angle", v.angleDeg == null ? "—" : fmt(v.angleDeg) + "°"],
          ["closed", String(v.valenceClosed)],
        ];
      }
      case "Path":
        return [
          ["item", v.item], ["steps", v.steps], ["converged", String(v.converged)],
          ["reps", v.reps.join(", ")], ["amalgamation", v.amalgamation.join(", ") || "—"],
        ];
      default:
        return [["value", fmt(v.value)]];
    }
  }, [v]);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-sm text-white">{name}</span>
        <span
          className="text-[10px] uppercase tracking-widest px-2 py-0.5 rounded-full"
          style={{ color, border: `1px solid ${color}55` }}
        >
          {v.ty}
        </span>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
        {rows.map(([k, val]) => (
          <div key={k} className="contents">
            <dt className="text-neutral-500">{k}</dt>
            <dd className="font-mono text-neutral-200 break-all">{String(val)}</dd>
          </div>
        ))}
        <dt className="text-neutral-500">residue</dt>
        <dd className="font-mono" style={{ color: ACCENT }}>{fmt(v.residue)} @ floor {fmt(v.floor)}</dd>
      </dl>
    </div>
  );
}

function fmt(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export default function Playground() {
  const [src, setSrc] = useState(EXAMPLES.water);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runScript = useCallback(() => {
    setError(null);
    setResult(null);
    try {
      // front-end-only first, so accountability errors are reported cleanly
      compile(src);
      const r = evaluate(src);
      setResult(r);
    } catch (e) {
      setError(e?.message || String(e));
    }
  }, [src]);

  const loadExample = useCallback((key) => {
    setSrc(EXAMPLES[key]);
    setResult(null);
    setError(null);
  }, []);

  const namedEntries = result ? Object.entries(result.named) : [];

  return (
    <>
      <Head>
        <title>honjo · Playground — Honjo Masamune</title>
        <meta name="description" content="Run honjo scripts in the browser: a cut-primitive DSL for cheminformatics by individuation." />
      </Head>
      <main className="min-h-screen bg-[#0a0a0a] text-neutral-200 pt-24 pb-16 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl md:text-4xl font-light tracking-tight text-white">
              honjo <span style={{ color: ACCENT }}>playground</span>
            </h1>
            <p className="mt-2 text-sm text-neutral-400 max-w-3xl">
              A cut-primitive language for cheminformatics by individuation. Every operation is a{" "}
              <span className="text-neutral-200">cut</span>: <code className="text-neutral-300">cut N</code> individuates an atom,{" "}
              <code className="text-neutral-300">a ~ b</code> is a bond, <code className="text-neutral-300">close X(…)</code>{" "}
              drives vacancies to zero, <code className="text-neutral-300">track x in P … yield r</code> propagates.
              Runs entirely in your browser. Nothing is free — every value carries the floor it was cut at.
            </p>
          </div>

          {/* Example buttons */}
          <div className="flex flex-wrap gap-2 mb-4">
            <span className="text-xs text-neutral-500 self-center mr-1">examples:</span>
            {Object.keys(EXAMPLES).map((k) => (
              <button
                key={k}
                onClick={() => loadExample(k)}
                className="text-xs font-mono px-3 py-1 rounded border border-neutral-700 text-neutral-300 hover:border-[#58E6D9] hover:text-[#58E6D9] transition-colors"
              >
                {k}.hj
              </button>
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Editor */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs uppercase tracking-widest text-neutral-500">source · .hj</span>
                <button
                  onClick={runScript}
                  className="text-sm font-medium px-5 py-1.5 rounded bg-[#58E6D9] text-[#0a0a0a] hover:brightness-110 transition"
                >
                  ▶ run
                </button>
              </div>
              <textarea
                value={src}
                onChange={(e) => setSrc(e.target.value)}
                spellCheck={false}
                className="w-full h-[28rem] font-mono text-sm leading-relaxed p-4 rounded-lg bg-neutral-950 border border-neutral-800 text-neutral-200 focus:outline-none focus:border-[#58E6D9] resize-none"
              />
            </div>

            {/* Output */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs uppercase tracking-widest text-neutral-500">measurement</span>
                {result && (
                  <span className="text-xs font-mono text-neutral-400">
                    clock M = <span style={{ color: ACCENT }}>{result.cutCount}</span>
                    {" · "}floor {fmt(result.floor)}
                    {" · "}
                    <span style={{ color: result.ok ? "#22c55e" : "#ef4444" }}>
                      {result.ok ? "ok" : "ABORTED"}
                    </span>
                  </span>
                )}
              </div>

              <div className="h-[28rem] overflow-auto rounded-lg bg-neutral-950 border border-neutral-800 p-4 space-y-3">
                {error && (
                  <div className="font-mono text-sm text-red-400 whitespace-pre-wrap">{error}</div>
                )}

                {!error && !result && (
                  <div className="text-neutral-600 text-sm italic">
                    Press <span className="text-neutral-400">run</span> to perform the cuts. The result is a measurement, not a simulation.
                  </div>
                )}

                {result && (
                  <>
                    {/* observation log */}
                    {result.log.length > 0 && (
                      <pre className="font-mono text-xs text-neutral-400 whitespace-pre-wrap border-b border-neutral-800 pb-3">
                        {result.log.join("\n")}
                      </pre>
                    )}
                    {/* structured value cards */}
                    <div className="grid sm:grid-cols-2 gap-3">
                      {namedEntries.map(([name, v]) => (
                        <ValueCard key={name} name={name} v={v} />
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Footer note */}
          <div className="mt-8 text-xs text-neutral-600 max-w-3xl">
            The language is specified in{" "}
            <span className="text-neutral-400">honjo-masamune-dsl.tex</span>. This playground runs the
            TypeScript reference compiler; a Rust target (cut = exact minimum cut) is the authoritative
            implementation. The two agree up to the floor —{" "}
            <Link href="/framework" className="text-[#58E6D9] hover:underline">read the framework</Link>.
          </div>
        </div>
      </main>
    </>
  );
}
