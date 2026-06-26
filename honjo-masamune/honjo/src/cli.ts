#!/usr/bin/env node
// Honjo Masamune — CLI runner.  Usage: honjo <file.hj> [--ir] [--ast]

import { readFileSync } from "node:fs";
import { compile, evaluate } from "./index.js";

function main(argv: string[]): number {
  const args = argv.slice(2);
  if (args.length === 0 || args.includes("-h") || args.includes("--help")) {
    console.log("usage: honjo <file.hj> [--ir] [--ast]");
    return args.length === 0 ? 1 : 0;
  }
  const file = args.find((a) => !a.startsWith("--"));
  if (!file) { console.error("no input file"); return 1; }

  let src: string;
  try { src = readFileSync(file, "utf8"); }
  catch (e) { console.error(`cannot read ${file}: ${(e as Error).message}`); return 1; }

  try {
    if (args.includes("--ast")) {
      const { program } = compile(src);
      console.log(JSON.stringify(program, null, 2));
      return 0;
    }
    if (args.includes("--ir")) {
      const { ir } = compile(src);
      console.log(JSON.stringify(ir.instrs, null, 2));
      return 0;
    }
    const r = evaluate(src);
    for (const line of r.log) console.log(line);
    console.log(`-- cut count (clock) M = ${r.cutCount} ; floor = ${r.floor} ; ${r.ok ? "ok" : "ABORTED"}`);
    return r.ok ? 0 : 2;
  } catch (e) {
    console.error((e as Error).message);
    return 1;
  }
}

process.exit(main(process.argv));
