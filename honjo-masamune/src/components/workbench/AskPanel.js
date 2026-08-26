/**
 * Ask panel.
 *
 * This is not a chat assistant and does not pretend to be one. It
 * answers from the generated data files and from the language
 * definitions, and when it has no grounded answer it says so rather
 * than producing a fluent guess.
 *
 * The rule it follows is the one the framework applies to everything
 * else: a value-bearing answer carries a value and cites where the
 * value came from; anything else is a refusal that names what is
 * missing. There is no model here and no network call — a wrong number
 * stated confidently would be worse than no answer, and a local-first
 * tool should not be shipping questions to a third party.
 */

import { useState, useRef, useEffect } from "react";
import MASAMUNE from "@/data/masamune.json";
import HONJO from "@/data/honjo.json";
import MEIBUTSU from "@/data/meibutsu.json";
import SPECTRA from "@/data/spectra.json";
import { CAPABILITY, FEATURES } from "@/lib/masamune";

const T = {
  bg: "#1a1b26", panel: "#24253a", border: "#2f3146",
  text: "#c0caf5", dim: "#565f89", muted: "#3b3d57",
  accent: "#7dcfff", ok: "#9ece6a", warn: "#e0af68", err: "#f7768e",
};

/**
 * Grounded answers.
 *
 * Each entry names the source of its numbers. `match` is deliberately
 * literal — a fuzzy matcher that fired on the wrong question would
 * produce a confidently wrong answer, which is the failure mode this
 * panel exists to avoid.
 */
function answer(q) {
  const s = q.toLowerCase();
  const has = (...words) => words.every((w) => s.includes(w));

  /* ---- supplied fraction ---- */
  if (has("supplied") || has("phi") || has("convention")) {
    const d = MASAMUNE.supplied;
    return {
      kind: "value",
      source: "validation/results/exp_masamune.json (M1)",
      body:
        `Over ${d.n_structures} structures the mean supplied fraction is ` +
        `${d.mean.toFixed(3)}, median ${d.median.toFixed(3)}, ranging from ` +
        `${d.min.toFixed(3)} to ${d.max.toFixed(3)}.\n\n` +
        `No structure reaches 0: every SMILES record in the corpus omits ` +
        `at least half of what the contact graph ends up containing, ` +
        `because the notation states no hydrogens explicitly. A bracket ` +
        `atom like [NH3] does state them, and reaches exactly 0.`,
    };
  }

  /* ---- capability ---- */
  if (has("capability") || has("declare") || has("coords3d") ||
      has("stereo") || has("refuse")) {
    const c = MASAMUNE.containment;
    const undeclared = FEATURES.filter(
      (f) => !Object.values(CAPABILITY).some((set) => set.includes(f))
    );
    return {
      kind: "value",
      source: "honjo-py/hjm/masamune/capability.py + exp_masamune.json (M2)",
      body:
        `${c.n_refused_statically} of ${c.n_pairs} format/request pairs are ` +
        `decided before any record is read, and the static verdict agrees ` +
        `with the post-read outcome on ${c.n_consistent}/${c.n_pairs}.\n\n` +
        `smiles declares: ${CAPABILITY.smiles.join(", ")}.\n` +
        `inchi declares nothing — no reader is implemented, so every ` +
        `request against it is refused.\n\n` +
        (undeclared.length
          ? `Declared by no format here: ${undeclared.join(", ")}. The ` +
            `SMILES reader parses stereo tokens but builds no descriptors, ` +
            `so declaring stereo would be an over-declaration.`
          : ""),
    };
  }

  /* ---- interference / visibility ---- */
  if (has("visibility") || has("interference") || has("superpos") ||
      has("cross-term") || has("cross term")) {
    const m = MEIBUTSU;
    return {
      kind: "value",
      source: "graphical-chemistry-generator/results/exp_generator.json",
      body:
        `Self-comparison is exactly 1 for all ${m.self_visibility.n} ` +
        `structures, maximum deviation ${m.self_visibility.max_deviation}. ` +
        `That is Cauchy-Schwarz with equality, not an empirical ` +
        `coincidence.\n\n` +
        `Cross-visibility falls with coordinate separation: correlation ` +
        `${m.decay.corr_visibility_vs_distance} over ` +
        `${m.decay.n_pairs} pairs, mean ${m.decay.cross_mean}, max ` +
        `${m.decay.cross_max}. No cross-pair reaches 1.\n\n` +
        `|A+B|^2 = |A|^2 + |B|^2 + 2 Re<A,B>. The first two terms are ` +
        `properties of each structure alone; everything relational is the ` +
        `third.`,
    };
  }

  /* ---- bulk / stacking ---- */
  if (has("bulk") || has("stack") || has("demodulat") ||
      (has("scal") && has("compar"))) {
    const m = MEIBUTSU;
    const worst = m.capacity[m.capacity.length - 1];
    return {
      kind: "value",
      source: "exp_generator.json (G5, G6)",
      body:
        `The identity holds exactly: one superposition reproduces the sum ` +
        `over all ${m.bulk_identity.n_pairs_implied} pairs with relative ` +
        `residual ${m.bulk_identity.relative_residual}.\n\n` +
        `But recovery is refuted. Demodulation correlates with true ` +
        `pairwise visibility at ${m.capacity.find((r) => r.stack_size === 4)?.demod_vs_true_correlation} ` +
        `for four structures and turns negative from eight onward; at ` +
        `${worst.stack_size} structures the correlation is ` +
        `${worst.demod_vs_true_correlation} and ` +
        `${(worst.frac_diverged * 100).toFixed(1)}% of projections diverge.\n\n` +
        `The cross-term bases are non-orthogonal, so each projection ` +
        `absorbs energy belonging to other pairs. The relational content ` +
        `is present in the stack and not addressable within it.`,
    };
  }

  /* ---- conformance ---- */
  if (has("conformance") || has("floor") || has("resolution") ||
      has("c1") || has("c7")) {
    const ratios = HONJO.conformance[0].ratios;
    return {
      kind: "value",
      source: "validation/results/exp_honjo.json",
      body:
        `The eight conformance items were executed. Two could not have ` +
        `failed as first written and were restated.\n\n` +
        `C1: residue is floor x vacancy under a clamp, so a sub-floor ` +
        `value is unconstructible and looking for one cannot fail. What ` +
        `is measured instead is that residue/floor is constant per atom ` +
        `across a 16-fold floor sweep: ` +
        Object.entries(ratios)
          .map(([k, v]) => `${k} ${v[0]}`)
          .join(", ") + `.\n\n` +
        `C7: a floor below the target resolution is refused with both ` +
        `numbers in the verdict.`,
    };
  }

  /* ---- inversion ---- */
  if (has("invert") || has("inversion") || has("identif") ||
      has("address")) {
    const inv = MEIBUTSU.inversion;
    return {
      kind: "value",
      source: "exp_generator.json (G4)",
      body:
        `Two independent routes, and they disagree.\n\n` +
        `Interference ranking: ${inv.ranked_first}/${inv.n} — the ` +
        `generating structure ranks first in every case.\n` +
        `Address uniqueness: ${inv.address_unique}/${inv.n} — ` +
        `${inv.n - inv.address_unique} structures share a cell at full ` +
        `depth.\n\n` +
        `So the address is a screen, not an identification, and the ` +
        `interference route is doing the work. Reporting only the first ` +
        `number would conceal that.`,
    };
  }

  /* ---- spectra ---- */
  if (has("spectrum") || has("spectra") || has("mode") ||
      has("frequenc")) {
    const names = Object.keys(SPECTRA.spectra);
    return {
      kind: "value",
      source: "src/data/spectra.json (NIST CCCBDB)",
      body:
        `${names.length} reference spectra are available to name in a ` +
        `.mbt program: ${names.join(", ")}.\n\n` +
        `Write "spectrum a = H2O" to use one, or give the frequencies ` +
        `directly as "spectrum a [3657, 1595, 3756]". The reference ` +
        `frequency is fixed at ${SPECTRA.omega_ref} cm-1 by the corpus.`,
    };
  }

  /* ---- engines ---- */
  if (has("engine") || has("rust") || has("token") || has("local")) {
    return {
      kind: "value",
      source: "WORKBENCH.md",
      body:
        `.hnj runs on either the Rust reference compiler (via a local ` +
        `engine and a token) or the in-browser JavaScript build. The two ` +
        `agree on every tutorial.\n\n` +
        `.msm and .mbt run only in the browser: there is no Rust ` +
        `Masamune, so connecting a local engine does not change those ` +
        `paths. Every result is labelled with the engine that produced ` +
        `it.\n\n` +
        `Start the engine with "honjo serve" and paste the token it ` +
        `prints. Nothing is uploaded; the browser talks to a process on ` +
        `your machine over loopback.`,
    };
  }

  return null;
}

export default function AskPanel({ open, onClose }) {
  const [q, setQ] = useState("");
  const [log, setLog] = useState([]);
  const endRef = useRef(null);

  useEffect(() => {
    if (endRef.current) endRef.current.scrollIntoView({ block: "end" });
  }, [log]);

  if (!open) return null;

  const ask = () => {
    const question = q.trim();
    if (!question) return;
    setQ("");
    const a = answer(question);
    setLog((p) => [
      ...p,
      { role: "you", text: question },
      a
        ? { role: "src", ...a }
        : {
            role: "src",
            kind: "refusal",
            body:
              "I have no grounded answer for that. This panel answers " +
              "only from the generated result files, and refuses rather " +
              "than guessing.\n\nIt can answer about: the supplied " +
              "fraction, declared capability and static refusal, " +
              "interference and visibility, bulk stacking, the " +
              "conformance suite and the floor, inversion, the reference " +
              "spectra, and the engines.",
          },
    ]);
  };

  return (
    <div style={{
      position: "absolute", right: 0, top: 0, bottom: 0, width: 380,
      background: T.panel, borderLeft: `1px solid ${T.border}`,
      display: "flex", flexDirection: "column", zIndex: 60,
    }}>
      <div style={{
        padding: "8px 12px", borderBottom: `1px solid ${T.border}`,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: T.text }}>Ask</span>
        <span onClick={onClose}
              style={{ cursor: "pointer", color: T.dim, fontSize: 16 }}>×</span>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
        {log.length === 0 && (
          <div style={{ fontSize: 11, color: T.dim, lineHeight: 1.7 }}>
            Answers come from the generated result files, with the source
            named. There is no model here and no network call: a wrong
            number stated confidently would be worse than no answer.
            <div style={{ marginTop: 10, color: T.muted }}>
              Try: what is the supplied fraction · why was this refused ·
              does bulk stacking work · which spectra can I use
            </div>
          </div>
        )}

        {log.map((m, i) => (
          <div key={i} style={{ marginBottom: 10 }}>
            {m.role === "you" ? (
              <div style={{
                fontSize: 11.5, color: T.text, background: T.bg,
                padding: "6px 10px", borderRadius: 4,
              }}>{m.text}</div>
            ) : (
              <div style={{
                fontSize: 11.5, lineHeight: 1.65, whiteSpace: "pre-wrap",
                color: m.kind === "refusal" ? T.warn : T.text,
                borderLeft: `2px solid ${m.kind === "refusal" ? T.warn : T.ok}`,
                paddingLeft: 10,
              }}>
                {m.body}
                {m.source && (
                  <div style={{ fontSize: 10, color: T.muted, marginTop: 6 }}>
                    source: {m.source}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div style={{ padding: 8, borderTop: `1px solid ${T.border}` }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") ask(); }}
          placeholder="ask about a measured result…"
          style={{
            width: "100%", padding: "6px 10px", fontSize: 11.5,
            background: T.bg, border: `1px solid ${T.border}`,
            borderRadius: 4, color: T.text, outline: "none",
            fontFamily: "inherit", boxSizing: "border-box",
          }}
        />
      </div>
    </div>
  );
}
