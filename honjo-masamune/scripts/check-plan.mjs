/**
 * Verify the JavaScript plan runner against the Python reference.
 *
 * Runs the same plans over the same records through both and compares
 * the step-by-step trace: what was read, what each translation
 * returned, what a select kept, whether an assertion passed, and what
 * was emitted.
 *
 * A plan is a statement about what was asked for and in what order, so
 * the trace is the thing that has to match — not just the final answer.
 *
 *   node scripts/check-plan.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

// The plan module imports masamune.js through the "@/" alias, which
// node does not resolve. Rewrite it to a relative specifier in memory
// rather than adding a bundler to the check path.
function loadAliased(relPath) {
  const src = readFileSync(join(ROOT, relPath), "utf8").replace(
    /from\s+"@\/lib\/([\w.]+)"/g,
    (_m, name) => {
      // the alias omits the extension; node needs it
      const file = name.endsWith(".js") ? name : `${name}.js`;
      return `from "${pathToFileURL(join(ROOT, "src/lib", file)).href}"`;
    }
  );
  return import("data:text/javascript," + encodeURIComponent(src));
}

const P = await loadAliased("src/lib/plan.js");
const REF = JSON.parse(readFileSync(join(HERE, "plan-reference.json"), "utf8"));

let checks = 0;
let bad = 0;

function eq(label, got, want) {
  checks += 1;
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g !== w) {
    bad += 1;
    console.log(`  MISMATCH ${label}`);
    console.log(`      js=${g}`);
    console.log(`      py=${w}`);
  }
}

console.log("checking JS plan runner against Python reference\n");

for (const [name, want] of Object.entries(REF.cases)) {
  const got = P.runPlan(want.source, REF.files);

  eq(`${name} status`, got.status, want.status);

  if (want.status === "refused") {
    eq(`${name} refusal.reason`, got.refusal?.reason, want.refusal.reason);
    eq(`${name} refusal.missing`, got.refusal?.missing_features,
       want.refusal.missing_features);
    eq(`${name} refusal.capability`, got.refusal?.source_capability,
       want.refusal.source_capability);
    continue;
  }
  if (want.status === "parse-error") {
    // messages are allowed to differ in wording; the line must match
    const gline = /line (\d+)/.exec(got.error || "")?.[1];
    const wline = /line (\d+)/.exec(want.error || "")?.[1];
    eq(`${name} error line`, gline, wline);
    continue;
  }

  eq(`${name} n_steps`, got.steps.length, want.steps.length);
  want.steps.forEach((ws, i) => {
    const gs = got.steps[i] || {};
    eq(`${name} step${i} op`, gs.step, ws.step);
    if (ws.count !== undefined) eq(`${name} step${i} count`, gs.count, ws.count);
    if (ws.tally !== undefined) eq(`${name} step${i} tally`, gs.tally, ws.tally);
    if (ws.kept !== undefined) {
      eq(`${name} step${i} kept`, gs.kept, ws.kept);
      eq(`${name} step${i} dropped`, gs.dropped, ws.dropped);
    }
    if (ws.passed !== undefined) {
      eq(`${name} step${i} passed`, gs.passed, ws.passed);
      eq(`${name} step${i} observed`, gs.observed, ws.observed);
    }
    if (ws.emitted !== undefined) {
      eq(`${name} step${i} n_emitted`, (gs.emitted || []).length,
         ws.emitted.length);
      ws.emitted.forEach((we, j) => {
        const ge = (gs.emitted || [])[j] || {};
        eq(`${name} step${i} emit${j} record`, ge.record, we.record);
        eq(`${name} step${i} emit${j} verdict`, ge.verdict, we.verdict);
      });
    }
  });
}

console.log();
console.log(`${checks - bad}/${checks} checks agree`);

if (bad > 0) {
  console.error("\nFAIL: the browser plan runner does not match the reference.");
  process.exit(1);
}
console.log("\nthe browser runs plans the same way as the reference.");
