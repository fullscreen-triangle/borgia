// Honjo Masamune — Lowering: typed AST -> Cut-IR (§8)

import {
  Program, Decl, Stmt, Expr, Cond, Admit,
} from "./ast.js";
import { Instr, IRProgram, IROperand, IRCond, Admit as IRAdmit } from "./ir.js";
import { TyName } from "./types.js";

export class LowerError extends Error {
  constructor(msg: string) { super(`lower error: ${msg}`); this.name = "LowerError"; }
}

class Lowering {
  private instrs: Instr[] = [];
  private nameToReg = new Map<string, number>();
  private next = 0;
  private floor = NaN;

  private reg(): number { return this.next++; }

  lower(p: Program): IRProgram {
    for (const d of p.decls) this.lowerDecl(d);
    return { instrs: this.instrs, nameToReg: this.nameToReg, floor: this.floor };
  }

  private lowerDecl(d: Decl): void {
    switch (d.tag) {
      case "floor":
        this.floor = d.value;
        this.instrs.push({ op: "Floor", value: d.value });
        break;
      case "import":
        break; // imports affect name resolution only; stdlib is always available
      case "module":
        for (const inner of d.body) this.lowerDecl(inner);
        break;
      default:
        this.lowerStmt(d as Stmt);
    }
  }

  private lowerStmt(s: Stmt): void {
    switch (s.tag) {
      case "bind": {
        const r = this.lowerExpr(s.value);
        this.nameToReg.set(s.name, r);
        break;
      }
      case "exprStmt":
        this.lowerExpr(s.expr);
        break;
      case "observe": {
        const r = this.lowerExpr(s.expr);
        this.instrs.push({ op: "Obs", reg: r, as: s.as ? this.bindName(s.as, r) : undefined });
        break;
      }
      case "assert":
        this.instrs.push({ op: "Assert", cond: this.lowerCond(s.cond), message: s.emit });
        break;
      case "track": {
        const itemReg = this.lookup(s.item);
        const procReg = this.lowerExpr(s.process);
        const dst = this.reg();
        const admit = this.lowerAdmit(s.admit);
        this.instrs.push({
          op: "Prp", dst, item: itemReg, process: procReg,
          reps: s.reps, admit, ty: "Path", floor: this.floor,
        });
        this.nameToReg.set(s.yieldName, dst);
        break;
      }
    }
  }

  private bindName(name: string, reg: number): number {
    this.nameToReg.set(name, reg);
    return reg;
  }

  private lookup(name: string): number {
    const r = this.nameToReg.get(name);
    if (r === undefined) throw new LowerError(`unbound name '${name}'`);
    return r;
  }

  private lowerExpr(e: Expr): number {
    switch (e.tag) {
      case "num": {
        const dst = this.reg();
        this.instrs.push({ op: "Mov", dst, src: { kind: "lit", value: e.value, floor: e.floor ?? this.floor } });
        return dst;
      }
      case "str": {
        const dst = this.reg();
        // strings are passthrough literals; encoded via Mov with a sentinel
        this.instrs.push({ op: "Mov", dst, src: { kind: "lit", value: NaN, floor: this.floor } });
        return dst;
      }
      case "ref":
        return this.lookup(e.name);
      case "cut": {
        const z = this.operand(e.arg);
        const dst = this.reg();
        this.instrs.push({ op: "Ind", dst, z, ty: "Atom", floor: this.floor });
        return dst;
      }
      case "bond": {
        const a = this.lowerExpr(e.left);
        const b = this.lowerExpr(e.right);
        const dst = this.reg();
        const guard = e.guard ? this.lowerCond(e.guard) : undefined;
        this.instrs.push({ op: "Bnd", dst, a, b, guard, ty: "Bond", floor: this.floor });
        return dst;
      }
      case "close": {
        const central = this.lookup(e.central);
        const args = e.args.map((a) => this.lowerExpr(a.value));
        const dst = this.reg();
        this.instrs.push({ op: "Cls", dst, central, args, ty: "Compound", floor: this.floor });
        return dst;
      }
      case "trackExpr": {
        const itemReg = this.lookup(e.item);
        const procReg = this.lowerExpr(e.process);
        const dst = this.reg();
        this.instrs.push({
          op: "Prp", dst, item: itemReg, process: procReg,
          reps: e.reps, admit: this.lowerAdmit(e.admit), ty: "Path", floor: this.floor,
        });
        return dst;
      }
      case "call": {
        const args = e.args.map((a) => this.operand(a.value));
        const dst = this.reg();
        const verb = e.name[e.name.length - 1];
        const ty: TyName =
          verb === "atom" || verb === "individuate" ? "Atom" :
          verb === "bond" ? "Bond" :
          verb === "close" || verb === "compound" ? "Compound" :
          verb === "track" || verb === "propagate" || verb === "amalgamation" ? "Path" : "Cut";
        this.instrs.push({ op: "Call", dst, name: e.name, args, ty, floor: this.floor });
        return dst;
      }
    }
  }

  private operand(e: Expr): IROperand {
    switch (e.tag) {
      case "num": return { kind: "lit", value: e.value, floor: e.floor ?? this.floor };
      case "ref": {
        const r = this.nameToReg.get(e.name);
        if (r !== undefined) return { kind: "reg", id: r };
        // unbound bare name in a condition position = a measured field (delta, vacancy, ...)
        return { kind: "name", name: e.name };
      }
      case "call": {
        // dotted ref like W.valence becomes a field operand on the base register
        if (e.name.length === 2 && e.args.length === 0) {
          const base = this.nameToReg.get(e.name[0]);
          if (base !== undefined) return { kind: "field", reg: base, field: e.name[1] };
        }
        const r = this.lowerExpr(e);
        return { kind: "reg", id: r };
      }
      default: {
        const r = this.lowerExpr(e);
        return { kind: "reg", id: r };
      }
    }
  }

  private lowerCond(c: Cond): IRCond {
    return { left: this.operand(c.left), op: c.op, right: this.operand(c.right) };
  }

  private lowerAdmit(a: Admit): IRAdmit {
    if (a.kind === "converge") return { kind: "converge" };
    if (a.kind === "diverge") return { kind: "diverge" };
    return { kind: "cond", cond: this.lowerCond(a.cond) };
  }
}

export function lower(p: Program): IRProgram {
  return new Lowering().lower(p);
}
