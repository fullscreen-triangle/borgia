// Honjo Masamune — Abstract Syntax Tree
// Mirrors the EBNF grammar of the spec (§4 "Grammar").

export interface Pos { line: number; col: number; }

// ---- Expressions ----
export type Expr =
  | NumLit
  | StrLit
  | Ref
  | CutExpr
  | BondExpr
  | CloseExpr
  | CallExpr;

export interface NumLit { tag: "num"; value: number; floor?: number; pos: Pos; }
export interface StrLit { tag: "str"; value: string; pos: Pos; }
export interface Ref { tag: "ref"; name: string; pos: Pos; }

// cut <number|ident>            — arity-1 cut (generation)
export interface CutExpr { tag: "cut"; arg: Expr; pos: Pos; }

// a ~ b  [when cond]            — arity-2 cut (bond)
export interface BondExpr { tag: "bond"; left: Expr; right: Expr; guard?: Cond; pos: Pos; }

// close X(a1,...,ak) [by ident] — cut-to-closure (compound)
export interface CloseExpr { tag: "close"; central: string; args: Arg[]; by?: string; pos: Pos; }

// qname(args)                   — stdlib / module call
export interface CallExpr { tag: "call"; name: string[]; args: Arg[]; pos: Pos; }

export interface Arg { label?: string; value: Expr; }

export interface Cond { tag: "cond"; left: Expr; op: RelOp; right: Expr; pos: Pos; }
export type RelOp = ">" | "<" | ">=" | "<=" | "==";

// ---- Statements / declarations ----
export type Decl = FloorDecl | ImportDecl | ModuleDecl | Stmt;
export type Stmt = BindStmt | TrackStmt | ObserveStmt | AssertStmt | ExprStmt;

export interface FloorDecl { tag: "floor"; value: number; pos: Pos; }
export interface ImportDecl { tag: "import"; name: string[]; pos: Pos; }
export interface ModuleDecl { tag: "module"; name: string; body: Decl[]; pos: Pos; }

export interface BindStmt { tag: "bind"; name: string; value: Expr; pos: Pos; }
export interface ExprStmt { tag: "exprStmt"; expr: Expr; pos: Pos; }
export interface ObserveStmt { tag: "observe"; expr: Expr; as?: string; pos: Pos; }
export interface AssertStmt { tag: "assert"; cond: Cond; emit?: string; pos: Pos; }

// track x in P [with reps r1,...] until <admit> yield r
export interface TrackStmt {
  tag: "track";
  item: string;
  process: Expr;
  reps?: string[];
  admit: Admit;
  yieldName: string;
  pos: Pos;
}
export type Admit = { kind: "converge" } | { kind: "diverge" } | { kind: "cond"; cond: Cond };

export interface Program { decls: Decl[]; }
