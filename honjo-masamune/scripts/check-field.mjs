/**
 * Verify the JavaScript field implementation against the Python
 * reference.
 *
 * The workbench computes interference in the browser. That is only
 * defensible if the browser computes the same numbers as the
 * implementation the paper's measurements came from. This script
 * compares coordinates, energies and pairwise visibilities against a
 * reference dump and fails on any disagreement beyond 1e-9.
 *
 * Regenerate the reference from the Python side:
 *
 *   cd ../dmitri/publications/graphical-chemistry-generator/src
 *   python -c "...see WORKBENCH.md..."
 *
 *   node scripts/check-field.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

const M = await import(pathToFileURL(join(ROOT, "src/lib/meibutsu.js")).href);
const REF = JSON.parse(readFileSync(join(HERE, "field-reference.json"), "utf8"));

const TOL = 1e-9;

let checks = 0;
let bad = 0;
let worst = 0;

function cmp(label, got, want) {
  checks += 1;
  const d = Math.abs(got - want);
  if (d > worst) worst = d;
  if (!(d <= TOL)) {
    bad += 1;
    console.log(`  MISMATCH ${label}`);
    console.log(`      js=${got}  py=${want}  |d|=${d.toExponential(3)}`);
  }
}

console.log("checking JS field against Python reference\n");

// --- coordinates and energy ---
const fields = {};
for (const [name, rec] of Object.entries(REF.compounds)) {
  const f = M.observe(rec.modes, { bRot: rec.b_rot, grid: 256, name });
  fields[name] = f;
  cmp(`${name} S_k`, f.coords[0], rec.coords[0]);
  cmp(`${name} S_t`, f.coords[1], rec.coords[1]);
  cmp(`${name} S_e`, f.coords[2], rec.coords[2]);
  cmp(`${name} energy`, M.energy(f), rec.energy);
}

// --- pairwise visibility ---
for (const [key, want] of Object.entries(REF.visibility)) {
  const [a, b] = key.split("|");
  cmp(`V(${a},${b})`, M.visibility(fields[a], fields[b]), want);
}

// --- the property the whole comparison rests on ---
let selfExact = 0;
for (const name of Object.keys(fields)) {
  const v = M.visibility(fields[name], fields[name]);
  if (Math.abs(v - 1) < 1e-12) selfExact += 1;
  else console.log(`  SELF NOT EXACT ${name}: ${v}`);
}

// --- the cross-term identity ---
// |A+B|^2 - |A|^2 - |B|^2 must equal the cross-term pointwise.
const names = Object.keys(fields);
let identityMax = 0;
for (let i = 0; i < names.length; i += 1) {
  for (let j = i + 1; j < names.length; j += 1) {
    const a = fields[names[i]];
    const b = fields[names[j]];
    const sup = M.superpose(a, b);
    const cross = M.crossTerm(a, b);
    for (let k = 0; k < a.grid; k += 1) {
      const own = a.amp[k] * a.amp[k] + b.amp[k] * b.amp[k];
      const d = Math.abs(sup[k] - own - cross[k]);
      if (d > identityMax) identityMax = d;
    }
  }
}

console.log();
console.log(`${checks - bad}/${checks} numeric checks agree (worst |d| = ${worst.toExponential(3)})`);
console.log(`self-visibility exactly 1: ${selfExact}/${names.length}`);
console.log(`cross-term identity residual: ${identityMax.toExponential(3)}`);

if (bad > 0 || selfExact !== names.length || identityMax > 1e-12) {
  console.error("\nFAIL: the browser implementation does not match the reference.");
  process.exit(1);
}
console.log("\nthe browser computes the same field as the reference.");
