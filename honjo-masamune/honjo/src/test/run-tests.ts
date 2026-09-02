// Honjo Masamune — test runner (no external deps).
// Exercises the pipeline, the four verbs, the accountability invariant, and
// cut-count monotonicity.

import { evaluate, compile } from "../index.js";
import { AtomVal, CompoundVal, PathVal, BondVal, shellCapacity } from "../stdlib.js";
import { deriveAtom, MAX_Z } from "../shell.js";

let pass = 0, fail = 0;
const failures: string[] = [];

function ok(name: string, cond: boolean, detail = "") {
  if (cond) { pass++; }
  else { fail++; failures.push(`${name}${detail ? "  (" + detail + ")" : ""}`); }
}
function throws(name: string, fn: () => void) {
  try { fn(); fail++; failures.push(`${name} (expected throw, none)`); }
  catch { pass++; }
}

// ---- 1. generate atom: carbon ----
{
  const r = evaluate(`floor 1.0\nC := cut 6\nobserve C`);
  const C = r.named["C"] as AtomVal;
  ok("carbon symbol", C.symbol === "C", C.symbol);
  ok("carbon config", C.config === "[He] 2s2 2p2", C.config);
  ok("carbon term", C.term === "3P_0", C.term);
  ok("carbon vacancy=4", C.vacancy === 4, String(C.vacancy));
  ok("carbon residue>=floor", C.residue >= 1.0, String(C.residue));
  ok("carbon cut count M=1", r.cutCount === 1, String(r.cutCount));
}

// ---- 2. build compound: water ----
{
  const r = evaluate(
    `floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)\nobserve W`
  );
  const W = r.named["W"] as CompoundVal;
  ok("water formula 1:2", W.formula[0] === 1 && W.formula[1] === 2, JSON.stringify(W.formula));
  ok("water bent", W.geometry === "bent", W.geometry);
  ok("water angle 104.5", W.angleDeg === 104.5, String(W.angleDeg));
  ok("water closed", W.valenceClosed === true);
}

// ---- 3. bonding criterion ----
{
  // open-shell pair bonds (delta > 0)
  const r1 = evaluate(`floor 1.0\nO := cut 8\nH := cut 1\nb := O ~ H\nobserve b`);
  ok("O~H bonds", (r1.named["b"] as BondVal).exists === true);
  // closed-shell partner does not bond
  const r2 = evaluate(`floor 1.0\nNa := cut 11\nNe := cut 10\nb := Na ~ Ne when delta > 0\nobserve b`);
  ok("Na~Ne no bond", (r2.named["b"] as BondVal).exists === false);
}

// ---- 4. track / causal table ----
{
  const r = evaluate(
    `floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)\n` +
    `path := track O in W with reps mass,charge until converge yield amalgamation\nobserve path`
  );
  const p = r.named["path"] as PathVal;
  ok("track converged", p.converged === true);
  ok("track steps == ligands(2)", p.steps === 2, String(p.steps));
  ok("track amalgamation nonempty", p.amalgamation.length === 2, String(p.amalgamation.length));
  ok("track reps preserved", p.reps.join(",") === "mass,charge", p.reps.join(","));
}

// ---- 5. accountability: no zero / negative floor ----
throws("floor 0 rejected", () => compile(`floor 0\nC := cut 6`));
throws("negative floor rejected", () => compile(`floor -1.0\nC := cut 6`));
throws("zero-floor literal rejected", () => compile(`floor 1.0\nx := 5.0#0`));
ok("positive floor accepted", (() => { compile(`floor 1.0\nC := cut 6`); return true; })());

// ---- 6. cut monotonicity (clock) ----
{
  const r = evaluate(`floor 1.0\nA := cut 6\nB := cut 8\nW := close B(A,A,A,A)`);
  // 2 individuations + close commits one cut per ligand interface
  ok("M monotone & >= individuations", r.cutCount >= 2, String(r.cutCount));
}

// ---- 7. shell capacity C(n)=2n^2 (pure verb) ----
{
  ok("C(1)=2", shellCapacity(1) === 2);
  ok("C(2)=8", shellCapacity(2) === 8);
  ok("C(3)=18", shellCapacity(3) === 18);
  ok("C(4)=32", shellCapacity(4) === 32);
}

// ---- 8. assert ok path; and stoichiometry NaCl is 1:1 ----
{
  const r = evaluate(`floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)\nassert W.valence == closed emit "no"`);
  ok("assert ok path", r.ok === true);
  const r2 = evaluate(`floor 1.0\nNa := cut 11\nCl := cut 17\nS := close Na(Cl)`);
  const S = r2.named["S"] as CompoundVal;
  ok("NaCl 1:1", S.formula[0] === 1 && S.formula[1] === 1, JSON.stringify(S.formula));
}

// ---- 9. examples parse, type-check, and run ----
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
{
  const here = dirname(fileURLToPath(import.meta.url));
  const exDir = join(here, "..", "..", "examples");
  let ran = 0;
  try {
    for (const f of readdirSync(exDir)) {
      if (!f.endsWith(".hj")) continue;
      const src = readFileSync(join(exDir, f), "utf8");
      const r = evaluate(src);
      ok(`example ${f} runs`, r.ok === true);
      ran++;
    }
  } catch (e) {
    failures.push(`examples dir: ${(e as Error).message}`);
  }
  ok("found example programs", ran >= 3, String(ran));
}

// ---- shell arithmetic: the atom is derived, not tabulated ----
// These check the port against the SAME committed benchmark the Python
// reference is validated on (atomic-derivation/results/*.json, NIST-checked).
// If the derivation drifts, this fails rather than quietly returning a
// plausible-looking configuration.
{
  const here2 = dirname(fileURLToPath(import.meta.url));
  const resDir = join(here2, "..", "..", "..", "..", "dmitri",
                      "publications", "atomic-derivation", "results");

  // C(n) = 2n^2 must hold as a computed sum, not by construction.
  ok("C(n)=2n^2", [1, 2, 3, 4, 5, 6].every((n) => shellCapacity(n) === 2 * n * n));

  try {
    const terms = JSON.parse(readFileSync(join(resDir, "term_symbols.json"), "utf8"));
    let tOk = 0;
    for (const r of terms.results) {
      const a = deriveAtom(r.Z);
      if (a.term === r.nist_term) tOk++;
      else failures.push(`term Z=${r.Z} ${r.symbol}: got ${a.term}, NIST ${r.nist_term}`);
    }
    ok("term symbols match NIST benchmark", tOk === terms.results.length,
       `${tOk}/${terms.results.length}`);
  } catch (e) {
    ok("term benchmark readable", false, (e as Error).message);
  }

  // Every named element must resolve. The old table stopped at 18; a
  // derivation that silently failed above some Z would be a table wearing a
  // function's clothes.
  let resolved = 0;
  for (let Z = 1; Z <= MAX_Z; Z++) {
    try { deriveAtom(Z); resolved++; } catch { /* counted by the assertion */ }
  }
  ok("all named elements resolve", resolved === MAX_Z, `${resolved}/${MAX_Z}`);

  // Spot-check placements the arithmetic has to get right, including the
  // cases that broke naive formulas: filled-d exceptions (Cu, Ag, Au) must
  // not read as alkali metals, and Pd must not fall a period short.
  const placements: [number, number, number][] = [
    [6, 2, 14], [17, 3, 17], [20, 4, 2], [21, 4, 3], [26, 4, 8],
    [29, 4, 11], [30, 4, 12], [46, 5, 10], [47, 5, 11], [79, 6, 11],
    [80, 6, 12], [2, 1, 18], [36, 4, 18],
  ];
  let placed = 0;
  for (const [Z, period, group] of placements) {
    const a = deriveAtom(Z);
    if (a.period === period && a.group === group) placed++;
    else failures.push(`placement Z=${Z} ${a.symbol}: got period ${a.period} group ${a.group}, want ${period}/${group}`);
  }
  ok("periods and groups derived correctly", placed === placements.length,
     `${placed}/${placements.length}`);

  // Elements the old eighteen-row table could not express at all.
  const fe = deriveAtom(26);
  ok("Z=26 beyond the old table", fe.configStr === "[Ar] 3d6 4s2" && fe.term === "5D_4",
     `${fe.configStr} ${fe.term}`);
}

// ---- report ----
console.log(`\nhonjo tests: ${pass} passed, ${fail} failed`);
if (fail) { console.log("failures:\n  " + failures.join("\n  ")); process.exit(1); }
process.exit(0);
