// Honjo Masamune — public API (the TypeScript playground compiler).
// Pipeline: source -> lex -> parse -> check -> lower (Cut-IR) -> run.

import { parse } from "./parser.js";
import { check } from "./types.js";
import { lower } from "./lower.js";
import { run, RunResult } from "./interp.js";
import { IRProgram } from "./ir.js";
import { Program } from "./ast.js";

export * from "./ast.js";
export type { Instr, IRProgram, IROperand, IRCond, IRValueRef } from "./ir.js";
export { lex } from "./lexer.js";
export { parse } from "./parser.js";
export { check } from "./types.js";
export { lower } from "./lower.js";
export { run, renderValue } from "./interp.js";
export type { RunResult, RVal } from "./interp.js";

export interface CompileResult {
  program: Program;
  ir: IRProgram;
}

/** Front end only: lex + parse + check + lower to Cut-IR (shared by both targets). */
export function compile(src: string): CompileResult {
  const program = parse(src);
  check(program);              // throws on accountability violation
  const ir = lower(program);
  return { program, ir };
}

/** Full pipeline on the reference (exact) back end. */
export function evaluate(src: string): RunResult {
  const { ir } = compile(src);
  return run(ir);
}

/** Convenience: run and return the human-readable observation log + clock. */
export function exec(src: string): { log: string[]; cutCount: number; ok: boolean } {
  const r = evaluate(src);
  return { log: r.log, cutCount: r.cutCount, ok: r.ok };
}
