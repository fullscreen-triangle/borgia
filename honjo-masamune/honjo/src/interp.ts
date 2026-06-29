// Honjo Masamune — Interpreter (operational semantics §6)
// Small-step evaluation over Cut-IR. Configuration = <graph state, cut-count M,
// instruction pointer>. Evaluation IS measurement: each cut event mutates state
// and strictly increments M (Thm "cut monotonicity"). This is the exact
// (reference) realisation of a cut — the back end the spec calls Target R in
// semantics, here in TypeScript for the playground; the WebGL render-pass back
// end (Target T) consumes the same IR.

import { IRProgram, IROperand, IRCond } from "./ir.js";
import {
  AtomVal, CompoundVal, PathVal, CutVal,
  individuate, bond, close, propagate,
} from "./stdlib.js";

export interface RunResult {
  cutCount: number;            // M, the program's intrinsic clock
  floor: number;
  registers: (RVal | undefined)[];
  named: Record<string, RVal>; // bound names -> values
  log: string[];               // observe / assert output
  ok: boolean;                 // false if an assert aborted
}

export type RVal = CutVal | { ty: "Scalar"; value: number; floor: number };

export class RuntimeError extends Error {
  constructor(msg: string) { super(`runtime error: ${msg}`); this.name = "RuntimeError"; }
}

export function run(ir: IRProgram): RunResult {
  const regs: (RVal | undefined)[] = [];
  const named: Record<string, RVal> = {};
  const log: string[] = [];
  let floor = Number.isNaN(ir.floor) ? 1 : ir.floor;
  let M = 0; // committed-cut count; monotone

  const regToName = new Map<number, string>();
  for (const [name, reg] of ir.nameToReg) regToName.set(reg, name);

  const getReg = (id: number): RVal => {
    const v = regs[id];
    if (v === undefined) throw new RuntimeError(`register r${id} read before write`);
    return v;
  };

  const asAtom = (id: number): AtomVal => {
    const v = getReg(id);
    if (v.ty !== "Atom") throw new RuntimeError(`expected Atom in r${id}, got ${v.ty}`);
    return v;
  };

  // resolve a numeric operand: literals, scalar regs, and measured fields.
  const numOperand = (o: IROperand): number => {
    switch (o.kind) {
      case "lit": return o.value;
      case "reg": {
        const v = getReg(o.id);
        return "value" in v ? (v as any).value : (v as any).residue ?? NaN;
      }
      case "field": return fieldValue(getReg(o.reg), o.field);
      case "name": throw new RuntimeError(`unresolved measured field '${o.name}' (no value in scope)`);
    }
  };

  // resolve a field of a cut value (delta, vacancy, valence, residue, angle...)
  const fieldValue = (v: RVal, field: string): number => {
    const anyv = v as any;
    if (field === "valence" || field === "closed") {
      if (v.ty === "Compound") return (v as CompoundVal).valenceClosed ? 1 : 0;
      if (v.ty === "Atom") return (v as AtomVal).vacancy === 0 ? 1 : 0;
    }
    if (field in anyv && typeof anyv[field] === "number") return anyv[field];
    throw new RuntimeError(`value of type ${v.ty} has no numeric field '${field}'`);
  };

  // evaluate a condition. A bare `name` operand refers to a field of the most
  // recent cut value when it matches (delta on a bond guard, etc.).
  const evalCond = (c: IRCond, ctx?: RVal): boolean => {
    const resolve = (o: IROperand): number => {
      if (o.kind === "name" && ctx) {
        // `delta`, `vacancy`, ... read off the contextual value
        try { return fieldValue(ctx, o.name); } catch { /* fall through */ }
      }
      if (o.kind === "name") {
        // closed-shell sentinel for `valence == closed`
        if (o.name === "closed") return ctx ? fieldValue(ctx, "valence") : 1;
      }
      return numOperand(o);
    };
    const L = resolve(c.left), R = resolve(c.right);
    switch (c.op) {
      case ">": return L > R;
      case "<": return L < R;
      case ">=": return L >= R;
      case "<=": return L <= R;
      case "==": return L === R;
    }
  };

  for (const ins of ir.instrs) {
    switch (ins.op) {
      case "Floor":
        if (!(ins.value > 0)) throw new RuntimeError("floor must be > 0");
        floor = ins.value;
        break;

      case "Mov":
        regs[ins.dst] = scalarFromOperand(ins.src, floor, numOperand);
        break;

      case "Ind": {
        const z = numOperand(ins.z);
        const atom = individuate(z, floor);
        regs[ins.dst] = atom;
        M += 1; // a cut event
        break;
      }

      case "Bnd": {
        const a = asAtom(ins.a), b = asAtom(ins.b);
        const bd = bond(a, b, floor);
        // guard, if present, reads `delta` off the bond just formed
        if (ins.guard && !evalCond(ins.guard, bd)) {
          // guard fails -> no bond committed (still a measured value, residue floor)
          regs[ins.dst] = { ...bd, exists: false };
          break;
        }
        regs[ins.dst] = bd;
        M += 1;
        break;
      }

      case "Cls": {
        const central = asAtom(ins.central);
        const ligs = ins.args.map((r) => asAtom(r));
        const comp = close(central, ligs, floor);
        regs[ins.dst] = comp;
        M += comp.ligands; // closure commits one cut per ligand interface
        break;
      }

      case "Prp": {
        const item = asAtom(ins.item);
        const proc = getReg(ins.process);
        if (proc.ty !== "Compound" && proc.ty !== "Path")
          throw new RuntimeError(`track process must be Compound or Path, got ${proc.ty}`);
        const admit =
          ins.admit.kind === "converge" ? "converge" as const :
          ins.admit.kind === "diverge" ? "diverge" as const :
          { holds: evalCond(ins.admit.cond, proc) };
        const path = propagate(item, proc as CompoundVal | PathVal, ins.reps ?? [], admit, floor);
        regs[ins.dst] = path;
        M += path.steps;
        break;
      }

      case "Obs": {
        const v = getReg(ins.reg);
        log.push(renderValue(v, regToName.get(ins.reg)));
        if (ins.as !== undefined) regs[ins.as] = v;
        break;
      }

      case "Assert": {
        const ctx = lastCutContext(regs);
        if (!evalCond(ins.cond, ctx)) {
          log.push(`ABORT: ${ins.message ?? "assertion failed"}`);
          return finalize(false);
        }
        break;
      }

      case "Call": {
        // module/stdlib call by name; supports the canonical verbs by qname.
        const verb = ins.name[ins.name.length - 1];
        if (verb === "individuate" || verb === "atom") {
          const z = numOperand(ins.args[0]);
          regs[ins.dst] = individuate(z, floor);
          M += 1;
        } else {
          throw new RuntimeError(`unknown call '${ins.name.join(".")}'`);
        }
        break;
      }
    }
  }

  return finalize(true);

  function finalize(ok: boolean): RunResult {
    for (const [name, reg] of ir.nameToReg) {
      const v = regs[reg];
      if (v !== undefined) named[name] = v;
    }
    return { cutCount: M, floor, registers: regs, named, log, ok };
  }
}

function scalarFromOperand(o: IROperand, floor: number, num: (o: IROperand) => number): RVal {
  if (o.kind === "lit") return { ty: "Scalar", value: o.value, floor: o.floor || floor };
  return { ty: "Scalar", value: num(o), floor };
}

function lastCutContext(regs: (RVal | undefined)[]): RVal | undefined {
  for (let i = regs.length - 1; i >= 0; i--) {
    const v = regs[i];
    if (v && v.ty !== "Scalar") return v;
  }
  return undefined;
}

export function renderValue(v: RVal, name?: string): string {
  const tag = name ? `${name} : ` : "";
  switch (v.ty) {
    case "Atom":
      return `${tag}Atom @ ${fmt(v.floor)}  Z=${v.Z} ${v.symbol}  ${v.config}  ${v.term}  vacancy=${v.vacancy}  valence=${v.valence}  residue=${fmt(v.residue)}`;
    case "Bond":
      return `${tag}Bond @ ${fmt(v.floor)}  ${v.a}~${v.b}  exists=${v.exists}  delta=${fmt(v.delta)}  shared=${v.shared}  residue=${fmt(v.residue)}`;
    case "Compound": {
      const lig = v.formula[1] > 1 ? v.ligand + v.formula[1] : v.formula[1] === 1 ? v.ligand : "";
      const formula = v.formula[0] === 2 ? v.central + "2" : v.central + lig;
      return `${tag}Compound @ ${fmt(v.floor)}  ${formula}  geometry=${v.geometry}  angle=${v.angleDeg ?? "-"}  closed=${v.valenceClosed}  residue=${fmt(v.residue)}`;
    }
    case "Path":
      return `${tag}Path @ ${fmt(v.floor)}  item=${v.item}  steps=${v.steps}  converged=${v.converged}  reps=[${v.reps.join(",")}]  amalgamation=[${v.amalgamation.join(", ")}]  residue=${fmt(v.residue)}`;
    case "Scalar":
      return `${tag}Scalar @ ${fmt(v.floor)}  ${fmt(v.value)}`;
  }
}

function fmt(n: number): string {
  if (Number.isNaN(n)) return "NaN";
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
