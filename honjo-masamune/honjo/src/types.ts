// Honjo Masamune — Accountability type system (§5)
// Types record WHAT KIND of cut a value is and AT WHAT FLOOR.
// Static invariant (Thm "No zero-residue value"): every typed value has
// floor > 0; the sharp cut (floor <= 0) is not typeable.

import {
  Program, Decl, Stmt, Expr, Cond, Pos,
} from "./ast.js";

export type TyName = "Atom" | "Bond" | "Compound" | "Path" | "Scalar" | "Cut" | "Bool" | "Void";

export interface Ty {
  name: TyName;
  floor: number; // ambient/annotated floor at which the value is individuated; > 0 for cuts
}

export class TypeError_ extends Error {
  constructor(msg: string, public line: number, public col: number) {
    super(`type error at ${line}:${col}: ${msg}`);
    this.name = "TypeError";
  }
}

// Cut supertype: Atom/Bond/Compound/Path are arity-/role-refinements of Cut.
const CUT_LIKE = new Set<TyName>(["Atom", "Bond", "Compound", "Path", "Cut"]);

interface Env {
  ambientFloor: number;
  vars: Map<string, Ty>;
}

export interface TypedProgram {
  program: Program;
  // type of each top-level binding name, for reporting / lowering hints
  bindings: Map<string, Ty>;
  ambientFloorAt: Map<Stmt | Decl, number>;
}

export function check(program: Program): TypedProgram {
  const env: Env = { ambientFloor: NaN, vars: new Map() };
  const bindings = new Map<string, Ty>();
  const ambientFloorAt = new Map<Stmt | Decl, number>();

  function fail(msg: string, pos: Pos): never {
    throw new TypeError_(msg, pos.line, pos.col);
  }

  function requirePositiveFloor(f: number, pos: Pos): number {
    if (!(f > 0)) fail(`floor must be > 0 (got ${f}); the sharp cut is not expressible`, pos);
    return f;
  }

  function checkExpr(e: Expr): Ty {
    switch (e.tag) {
      case "num": {
        // a literal carries its annotated floor, or inherits the ambient floor.
        const f = e.floor !== undefined ? e.floor : env.ambientFloor;
        if (Number.isNaN(f)) fail("numeric literal used before any 'floor' declaration", e.pos);
        // A Scalar literal's "residue" is its floor; positivity enforced.
        requirePositiveFloor(f, e.pos);
        return { name: "Scalar", floor: f };
      }
      case "str":
        return { name: "Void", floor: env.ambientFloor };
      case "ref": {
        const t = env.vars.get(e.name);
        if (!t) {
          // allow dotted/unknown to be treated as scalar handle? No — names must be bound.
          fail(`unbound identifier '${e.name}'`, e.pos);
        }
        return t;
      }
      case "cut": {
        // cut Z : Atom @ ambient floor.  Z must be a positive scalar.
        const at = checkExpr(e.arg);
        if (at.name !== "Scalar") fail("cut expects an atomic number (Scalar)", e.pos);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Atom", floor: f };
      }
      case "bond": {
        const lt = checkExpr(e.left);
        const rt = checkExpr(e.right);
        if (!CUT_LIKE.has(lt.name) || !CUT_LIKE.has(rt.name))
          fail("a bond (~) joins two cut-like values (Atom/Compound)", e.pos);
        if (e.guard) checkCond(e.guard);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Bond", floor: f };
      }
      case "close": {
        const ct = env.vars.get(e.central);
        if (!ct) fail(`unbound central atom '${e.central}'`, e.pos);
        if (!CUT_LIKE.has(ct.name)) fail("close expects an Atom as the central item", e.pos);
        for (const a of e.args) {
          const t = checkExpr(a.value);
          if (!CUT_LIKE.has(t.name)) fail("close ligands must be cut-like (Atom)", e.pos);
        }
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Compound", floor: f };
      }
      case "trackExpr": {
        const it = env.vars.get(e.item);
        if (!it) fail(`tracking unbound item '${e.item}'`, e.pos);
        if (!CUT_LIKE.has(it.name)) fail("track expects an Atom/Compound item", e.pos);
        checkExpr(e.process);
        if (e.admit.kind === "cond") checkCond(e.admit.cond);
        const f = requirePositiveFloor(env.ambientFloor, e.pos);
        return { name: "Path", floor: f };
      }
      case "call": {
        // stdlib / module calls are checked structurally: arguments must type,
        // result type is inferred from the verb name where known.
        for (const a of e.args) checkExpr(a.value);
        const f = Number.isNaN(env.ambientFloor) ? 1 : env.ambientFloor;
        const verb = e.name[e.name.length - 1];
        const resultName: TyName =
          verb === "atom" || verb === "individuate" ? "Atom" :
          verb === "bond" ? "Bond" :
          verb === "close" || verb === "compound" ? "Compound" :
          verb === "track" || verb === "propagate" || verb === "amalgamation" ? "Path" :
          "Cut";
        return { name: resultName, floor: f };
      }
    }
  }

  function checkCond(c: Cond): Ty {
    // operands may be scalar-like or carry a numeric field (e.g. delta, vacancy).
    // We type each side leniently: refs to cut-like values expose numeric fields.
    typeCondOperand(c.left);
    typeCondOperand(c.right);
    return { name: "Bool", floor: env.ambientFloor };
  }

  function typeCondOperand(e: Expr): void {
    // A bare ref like `delta` or `W.valence` is a measured field; accept it.
    if (e.tag === "ref" || e.tag === "call") return;
    checkExpr(e);
  }

  function checkStmt(s: Stmt): void {
    ambientFloorAt.set(s, env.ambientFloor);
    switch (s.tag) {
      case "bind": {
        const t = checkExpr(s.value);
        env.vars.set(s.name, t);
        bindings.set(s.name, t);
        break;
      }
      case "exprStmt":
        checkExpr(s.expr);
        break;
      case "observe":
        // observe forces measurement; type is whatever the expr is.
        { const t = checkExpr(s.expr); if (s.as) env.vars.set(s.as, t); }
        break;
      case "assert":
        checkCond(s.cond);
        break;
      case "track": {
        const it = env.vars.get(s.item);
        if (!it) fail(`tracking unbound item '${s.item}'`, s.pos);
        if (!CUT_LIKE.has(it.name)) fail("track expects an Atom/Compound item", s.pos);
        checkExpr(s.process);
        if (s.admit.kind === "cond") checkCond(s.admit.cond);
        const f = requirePositiveFloor(env.ambientFloor, s.pos);
        env.vars.set(s.yieldName, { name: "Path", floor: f });
        bindings.set(s.yieldName, { name: "Path", floor: f });
        break;
      }
    }
  }

  function checkDecl(d: Decl): void {
    switch (d.tag) {
      case "floor":
        ambientFloorAt.set(d, d.value);
        requirePositiveFloor(d.value, d.pos);
        env.ambientFloor = d.value;
        break;
      case "import":
        break;
      case "module":
        for (const inner of d.body) checkDecl(inner);
        break;
      default:
        checkStmt(d as Stmt);
    }
  }

  for (const d of program.decls) checkDecl(d);
  return { program, bindings, ambientFloorAt };
}
