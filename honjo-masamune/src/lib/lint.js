/**
 * Editor diagnostics.
 *
 * The only checks here are ones the compiler would also make, computed
 * from the same declared capability sets. A marker in the gutter is a
 * prediction that the compiler will refuse, and it should be wrong only
 * if the compiler changes.
 *
 * Deliberately absent: style advice, naming conventions, and anything
 * that would produce a warning the compiler does not care about. A
 * linter that flags what the language permits trains people to ignore
 * it.
 */

/**
 * Parse `source <name> : <format> at "..."` declarations.
 * Returns a map from binding name to format.
 */
function sourceFormats(lines) {
  const out = {};
  lines.forEach((line) => {
    const m = line.match(/^\s*source\s+([A-Za-z_][\w]*)\s*:\s*([A-Za-z_][\w]*)/);
    if (m) out[m[1]] = m[2];
  });
  return out;
}

/**
 * Collect the features named by a `require` clause, which may run over
 * several lines until a clause keyword or a closing brace.
 */
function requiredFeatures(lines, startIdx) {
  const stop = /^\s*(expect|else|emit|let|assert|select|source|\}|$)/;
  let acc = lines[startIdx].replace(/^.*\brequire\b/, "");
  for (let i = startIdx + 1; i < lines.length; i += 1) {
    if (stop.test(lines[i])) break;
    acc += " " + lines[i];
  }
  return acc
    .split(/[,\s]+/)
    .map((s) => s.trim().replace(/[^\w]/g, ""))
    .filter(Boolean);
}

/**
 * Lint a source file.
 *
 * @param {string} src        file contents
 * @param {string} ext        extension without the dot
 * @param {object} capability format -> array of declared features
 */
export function lint(src, ext, capability) {
  if (!src) return [];
  const lines = src.split("\n");
  const markers = [];

  if (ext === "msm") {
    const formats = sourceFormats(lines);
    // A plan with exactly one source can attribute a `require` to it
    // even when the translate call does not name the binding.
    const only = Object.values(formats).length === 1
      ? Object.values(formats)[0]
      : null;

    lines.forEach((line, i) => {
      if (/^\s*--/.test(line)) return;

      if (/\brequire\b/.test(line)) {
        const feats = requiredFeatures(lines, i);
        // Attribute to whichever source binding is mentioned nearby,
        // else to the single source if there is one.
        let fmt = only;
        for (const [bind, f] of Object.entries(formats)) {
          const window = lines.slice(Math.max(0, i - 4), i + 1).join(" ");
          if (new RegExp(`\\b${bind}\\b`).test(window)) fmt = f;
        }
        if (!fmt) return;

        const declared = capability[fmt];
        if (!declared) {
          markers.push({
            line: i,
            type: "warning",
            msg: `unknown format '${fmt}' — no declared capability set`,
          });
          return;
        }
        const missing = feats.filter(
          (f) => f in FEATURE_SET && !declared.includes(f)
        );
        if (missing.length) {
          markers.push({
            line: i,
            type: "error",
            msg:
              `${fmt} cannot state ${missing.join(", ")} — this request is ` +
              `refused statically, before any record is read`,
          });
        }
        const unknown = feats.filter((f) => !(f in FEATURE_SET));
        if (unknown.length) {
          markers.push({
            line: i,
            type: "warning",
            msg: `not a known feature: ${unknown.join(", ")}`,
          });
        }
      }

      // `expect supplied < x` where x lies below every measured value
      // for the format is a threshold nothing can satisfy.
      const exp = line.match(/expect\s+supplied\s*<\s*([\d.]+)/);
      if (exp && only === "smiles") {
        const thr = parseFloat(exp[1]);
        if (thr <= 0.5) {
          markers.push({
            line: i,
            type: "warning",
            msg:
              `no SMILES-derived structure in the reference corpus reaches ` +
              `φ < 0.5 (measured minimum 0.500) — this threshold admits nothing`,
          });
        }
      }
    });
  }

  if (ext === "hnj") {
    lines.forEach((line, i) => {
      if (/^\s*--/.test(line)) return;
      const fl = line.match(/^\s*floor\s+([\d.eE+-]+)/);
      if (fl) {
        const v = parseFloat(fl[1]);
        if (Number.isFinite(v) && v < 1e-9) {
          markers.push({
            line: i,
            type: "error",
            msg:
              `floor ${fl[1]} is below the target resolution 1e-9 — the ` +
              `program will be refused with both numbers in the verdict`,
          });
        }
      }
    });
  }

  return markers;
}

/** The feature alphabet, as a set for membership tests. */
const FEATURE_SET = {
  element: 1, connectivity: 1, cellcount: 1, delocalisation: 1,
  charge: 1, isotope: 1, hcount: 1, stereo: 1, coords3d: 1,
  conformer: 1, provenance: 1,
};

export { FEATURE_SET };
