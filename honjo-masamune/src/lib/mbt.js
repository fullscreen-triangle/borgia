/**
 * Meibutsu programs.
 *
 * A small language over the observation field: declare spectra, build
 * fields, superpose them, and read off what the superposition contains.
 *
 * There is no Python reference for this surface syntax — the reference
 * implementation is a library, not a language — so this parser is the
 * definition. What it must not do is invent semantics: every operation
 * below calls straight into src/lib/meibutsu.js, which IS checked
 * against the reference to 1e-13.
 *
 * Grammar:
 *
 *   program   := decl*
 *   decl      := "spectrum" ident "[" number ("," number)* "]"
 *              | "spectrum" ident "=" ident            -- from the library
 *              | "grid" number
 *              | "reference" number
 *              | "observe" ident
 *              | "superpose" ident ident
 *              | "invert" ident
 *              | "report" field ("," field)*
 *
 * Everything is evaluated in source order; there is no control flow.
 * A language that could branch would need a story about what a branch
 * means for a measurement, and there is not one yet.
 */

import {
  observe, visibility, crossTerm, superpose, energy, coordinates,
  OMEGA_REF,
} from "@/lib/meibutsu";
import SPECTRA from "@/data/spectra.json";

export class MbtError extends Error {
  constructor(msg, line) {
    super(`line ${line}: ${msg}`);
    this.line = line;
  }
}

/* ------------------------------------------------------------------ */

function lex(src) {
  const toks = [];
  src.split("\n").forEach((raw, li) => {
    const line = raw.replace(/--.*$/, "");
    let i = 0;
    while (i < line.length) {
      const c = line[i];
      if (/\s/.test(c)) { i += 1; continue; }
      if ("[],=".includes(c)) {
        toks.push({ kind: "op", text: c, line: li + 1 });
        i += 1;
        continue;
      }
      if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(line[i + 1] || ""))) {
        let j = i;
        while (j < line.length && /[0-9.eE+-]/.test(line[j])) {
          // stop a trailing sign that is not part of an exponent
          if ((line[j] === "+" || line[j] === "-") &&
              !/[eE]/.test(line[j - 1] || "")) break;
          j += 1;
        }
        toks.push({ kind: "number", text: line.slice(i, j), line: li + 1 });
        i = j;
        continue;
      }
      if (/[A-Za-z_]/.test(c)) {
        let j = i;
        while (j < line.length && /[A-Za-z0-9_]/.test(line[j])) j += 1;
        toks.push({ kind: "ident", text: line.slice(i, j), line: li + 1 });
        i = j;
        continue;
      }
      throw new MbtError(`unexpected character '${c}'`, li + 1);
    }
  });
  toks.push({ kind: "eof", text: "", line: 0 });
  return toks;
}

/* ------------------------------------------------------------------ */

const REPORTABLE = new Set([
  "coordinates", "energy", "visibility", "cross", "modes", "peak",
]);

export function runMbt(src) {
  let toks;
  try {
    toks = lex(src);
  } catch (err) {
    return { status: "parse-error", error: err.message, steps: [] };
  }

  let i = 0;
  const peek = () => toks[i];
  const next = () => toks[i++];
  const expect = (kind, text) => {
    const t = next();
    if (t.kind !== kind || (text !== undefined && t.text !== text)) {
      throw new MbtError(
        `expected '${text ?? kind}', got '${t.text || t.kind}'`, t.line
      );
    }
    return t;
  };

  const spectra = {};     // name -> {modes, bRot, source}
  const fields = {};      // name -> Observation
  const steps = [];
  let grid = 256;

  try {
    while (peek().kind !== "eof") {
      const tok = next();
      const ln = tok.line;

      if (tok.text === "grid") {
        grid = parseInt(expect("number").text, 10);
        if (!(grid > 1)) throw new MbtError("grid must exceed 1", ln);
        steps.push({ step: "grid", line: ln, value: grid });
        continue;
      }

      if (tok.text === "reference") {
        // The reference frequency is a property of the corpus, not a
        // free parameter: changing it would silently rescale every
        // address. Accepted, but only at its declared value.
        const v = parseFloat(expect("number").text);
        if (Math.abs(v - OMEGA_REF) > 1e-9) {
          throw new MbtError(
            `reference is fixed at ${OMEGA_REF} by the corpus; got ${v}`, ln
          );
        }
        steps.push({ step: "reference", line: ln, value: v });
        continue;
      }

      if (tok.text === "spectrum") {
        const name = expect("ident").text;
        if (peek().text === "=") {
          next();
          const from = expect("ident").text;
          const entry = SPECTRA.spectra[from];
          if (!entry) {
            throw new MbtError(
              `no spectrum named '${from}' in the reference set`, ln
            );
          }
          spectra[name] = {
            modes: entry.modes, bRot: entry.b_rot, source: from,
          };
        } else {
          expect("op", "[");
          const modes = [];
          for (;;) {
            modes.push(parseFloat(expect("number").text));
            if (peek().text === ",") { next(); continue; }
            break;
          }
          expect("op", "]");
          if (!modes.length) throw new MbtError("empty spectrum", ln);
          if (modes.some((m) => !(m > 0))) {
            throw new MbtError("mode frequencies must be positive", ln);
          }
          spectra[name] = { modes, bRot: null, source: null };
        }
        steps.push({
          step: "spectrum", line: ln, name,
          modes: spectra[name].modes, from: spectra[name].source,
        });
        continue;
      }

      if (tok.text === "observe") {
        const name = expect("ident").text;
        const sp = spectra[name];
        if (!sp) throw new MbtError(`no spectrum named '${name}'`, ln);
        const f = observe(sp.modes, { bRot: sp.bRot, grid, name });
        fields[name] = f;
        steps.push({
          step: "observe", line: ln, name,
          coords: f.coords, energy: energy(f), grid,
        });
        continue;
      }

      if (tok.text === "superpose") {
        const a = expect("ident").text;
        const b = expect("ident").text;
        const fa = fields[a];
        const fb = fields[b];
        if (!fa) throw new MbtError(`'${a}' has not been observed`, ln);
        if (!fb) throw new MbtError(`'${b}' has not been observed`, ln);
        const V = visibility(fa, fb);
        const cross = crossTerm(fa, fb);
        let relational = 0;
        let pos = 0;
        let neg = 0;
        for (const v of cross) {
          relational += v;
          if (v > 0) pos += 1;
          else if (v < 0) neg += 1;
        }
        steps.push({
          step: "superpose", line: ln, a, b,
          visibility: V,
          own_energy: energy(fa) + energy(fb),
          relational,
          constructive: pos,
          destructive: neg,
        });
        continue;
      }

      if (tok.text === "invert") {
        const name = expect("ident").text;
        const sp = spectra[name];
        if (!sp) throw new MbtError(`no spectrum named '${name}'`, ln);
        const q = observe(sp.modes, { bRot: sp.bRot, grid, name });
        const ranked = Object.entries(SPECTRA.spectra)
          .map(([k, v]) => [
            k,
            visibility(q, observe(v.modes, { bRot: v.b_rot, grid, name: k })),
          ])
          .sort((x, y) => y[1] - x[1]);
        steps.push({
          step: "invert", line: ln, query: name,
          ranked: ranked.slice(0, 5).map(([k, v]) => ({ name: k, visibility: v })),
          n_reference: ranked.length,
        });
        continue;
      }

      if (tok.text === "report") {
        const wanted = [];
        for (;;) {
          const f = expect("ident").text;
          if (!REPORTABLE.has(f)) {
            throw new MbtError(
              `cannot report '${f}'; try ${[...REPORTABLE].join(", ")}`, ln
            );
          }
          wanted.push(f);
          if (peek().text === ",") { next(); continue; }
          break;
        }
        const rows = Object.entries(fields).map(([name, f]) => {
          const row = { name };
          if (wanted.includes("coordinates")) row.coordinates = f.coords;
          if (wanted.includes("energy")) row.energy = energy(f);
          if (wanted.includes("modes")) row.modes = f.modes;
          if (wanted.includes("peak")) {
            let peakV = 0;
            let peakAt = 0;
            for (let k = 0; k < f.grid; k += 1) {
              if (f.amp[k] > peakV) { peakV = f.amp[k]; peakAt = k / (f.grid - 1); }
            }
            row.peak = { amplitude: peakV, at: peakAt };
          }
          return row;
        });
        steps.push({ step: "report", line: ln, fields: wanted, rows });
        continue;
      }

      throw new MbtError(`unknown statement '${tok.text}'`, ln);
    }
  } catch (err) {
    if (err instanceof MbtError) {
      return { status: "error", error: err.message, steps };
    }
    throw err;
  }

  return {
    status: "ok", steps,
    spectra: Object.keys(spectra),
    fields: Object.keys(fields),
    grid,
  };
}

export const REFERENCE_SPECTRA = Object.keys(SPECTRA.spectra);
