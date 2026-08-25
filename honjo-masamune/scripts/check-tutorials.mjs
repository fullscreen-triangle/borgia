/**
 * Verify that every executable tutorial actually runs.
 *
 * A tutorial that does not compile teaches syntax the compiler
 * rejects, which is worse than no tutorial at all. This script runs
 * each one marked `executable: true` through the in-browser engine and
 * fails the build if any of them errors.
 *
 * Where a local Rust engine is reachable (env HONJO_ENDPOINT and
 * HONJO_TOKEN), each tutorial is run through that too and the two
 * engines are compared on cut count. A divergence is reported but does
 * not fail the run: the two are separate implementations and the
 * disagreement is the finding, not a build error.
 *
 *   node scripts/check-tutorials.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

// pathToFileURL: on Windows a bare absolute path is not a valid ESM specifier
const { evaluate } = await import(
  pathToFileURL(join(ROOT, "src/lib/honjo.js")).href
);

// tutorials.js uses the "@/..." alias, which node does not resolve.
// Read and strip the single import rather than pulling in a bundler.
const rawTutorials = readFileSync(join(ROOT, "src/lib/tutorials.js"), "utf8");
const mod = await import(
  "data:text/javascript," + encodeURIComponent(rawTutorials)
);
const { TUTORIALS } = mod;

const endpoint = process.env.HONJO_ENDPOINT || "";
const token = process.env.HONJO_TOKEN || "";
const useLocal = Boolean(endpoint && token);

async function runLocal(source) {
  const res = await fetch(`${endpoint.replace(/\/$/, "")}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ source }),
  });
  return res.json();
}

let checked = 0;
let failed = 0;
let diverged = 0;

console.log("checking executable tutorials\n");

for (const [lang, files] of Object.entries(TUTORIALS)) {
  for (const [name, entry] of Object.entries(files)) {
    if (!entry.executable) {
      console.log(`  ${lang}/${name}  — not executable, skipped`);
      continue;
    }
    checked += 1;

    let browser;
    try {
      browser = evaluate(entry.source);
    } catch (err) {
      failed += 1;
      console.log(`  ${lang}/${name}  FAILED (browser)`);
      console.log(`      ${err.message}`);
      continue;
    }

    if (!browser.ok) {
      failed += 1;
      console.log(`  ${lang}/${name}  FAILED (browser: not ok)`);
      console.log(`      ${(browser.log || []).join("\n      ")}`);
      continue;
    }

    let line = `  ${lang}/${name}  ok  M=${browser.cutCount}`;

    if (useLocal) {
      try {
        const local = await runLocal(entry.source);
        if (!local.ok) {
          failed += 1;
          console.log(`  ${lang}/${name}  FAILED (rust)`);
          console.log(`      ${local.error || "not ok"}`);
          continue;
        }
        if (local.cut_count !== browser.cutCount) {
          diverged += 1;
          line += `  DIVERGED rust M=${local.cut_count}`;
        } else {
          line += `  rust agrees`;
        }
      } catch (err) {
        line += `  (rust unreachable: ${err.message})`;
      }
    }

    console.log(line);
  }
}

console.log();
console.log(
  `${checked - failed}/${checked} executable tutorials run` +
    (useLocal ? ` · ${diverged} engine divergence(s)` : " · rust engine not checked")
);

if (failed > 0) {
  console.error("\nFAIL: a tutorial does not run.");
  process.exit(1);
}
if (!useLocal) {
  console.log(
    "\nSet HONJO_ENDPOINT and HONJO_TOKEN to also check the Rust engine."
  );
}
