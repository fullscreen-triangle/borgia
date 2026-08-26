/**
 * Verify the JavaScript Masamune port against the Python reference.
 *
 * The workbench translates SMILES in the browser. That is only
 * defensible if it produces the same graph as the implementation the
 * paper's measurements came from — the supplied fraction in particular,
 * since that is the number the whole provenance argument rests on.
 *
 * Compares, for every structure in the validation corpus and for a
 * probe set covering the verdict labels:
 *   the verdict label
 *   the supplied fraction, exactly
 *   atom, contact, medium-edge and delocalised-system counts
 *
 *   node scripts/check-masamune.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

const M = await import(pathToFileURL(join(ROOT, "src/lib/masamune.js")).href);
const REF = JSON.parse(readFileSync(join(HERE, "masamune-reference.json"), "utf8"));

let checks = 0;
let bad = 0;

function eq(label, got, want) {
  checks += 1;
  const same =
    typeof want === "number" && typeof got === "number"
      ? Math.abs(got - want) < 1e-12
      : got === want;
  if (!same) {
    bad += 1;
    console.log(`  MISMATCH ${label}`);
    console.log(`      js=${JSON.stringify(got)}  py=${JSON.stringify(want)}`);
  }
}

function stats(g) {
  const contacts = Object.values(g.contacts);
  return {
    phi: M.suppliedFraction(g),
    n_atoms: Object.keys(g.atoms).length,
    n_contacts: contacts.filter((c) => !c.isMedium).length,
    n_medium: contacts.filter((c) => c.isMedium).length,
    n_deloc: Object.keys(g.delocs).length,
    n_heavy: Object.values(g.atoms).filter((a) => a.z !== 1).length,
  };
}

console.log("checking JS Masamune against Python reference\n");

for (const [name, want] of Object.entries(REF.structures)) {
  const v = M.translateSmiles(want.smiles, {
    required: ["element", "connectivity"],
  });
  eq(`${name} label`, v.label, want.label);
  if (want.phi === undefined) continue;
  if (!v.value) {
    bad += 1;
    checks += 1;
    console.log(`  MISMATCH ${name}: js produced no graph`);
    continue;
  }
  const s = stats(v.value);
  eq(`${name} phi`, s.phi, want.phi);
  eq(`${name} n_atoms`, s.n_atoms, want.n_atoms);
  eq(`${name} n_contacts`, s.n_contacts, want.n_contacts);
  eq(`${name} n_medium`, s.n_medium, want.n_medium);
  eq(`${name} n_deloc`, s.n_deloc, want.n_deloc);
  eq(`${name} n_heavy`, s.n_heavy, want.n_heavy);
}

for (const [name, want] of Object.entries(REF.probes)) {
  const required =
    name === "stereo-required"
      ? ["element", "connectivity", "stereo"]
      : ["element", "connectivity"];
  const v = M.translateSmiles(want.smiles, { required });
  eq(`probe:${name} label`, v.label, want.label);
  if (want.phi !== null && v.value) {
    eq(`probe:${name} phi`, M.suppliedFraction(v.value), want.phi);
  }
}

// The invariant the verdict design exists to enforce.
let carriedOnFailure = 0;
for (const s of ["", "C(((", "Zz", "CCO"]) {
  const v = M.translateSmiles(s, { required: ["element", "connectivity"] });
  if (v.value !== null && !v.ok) carriedOnFailure += 1;
}

console.log();
console.log(`${checks - bad}/${checks} checks agree`);
console.log(`failure verdicts carrying a value: ${carriedOnFailure}`);

if (bad > 0 || carriedOnFailure > 0) {
  console.error("\nFAIL: the browser translation does not match the reference.");
  process.exit(1);
}
console.log("\nthe browser translates the same way as the reference.");
