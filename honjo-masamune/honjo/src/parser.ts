// Honjo Masamune — Parser
// Recursive-descent parser implementing the grammar of the spec (§4).

import { Token, lex } from "./lexer.js";
import {
  Program, Decl, Stmt, Expr, Arg, Cond, RelOp, Admit, Pos,
} from "./ast.js";

export class ParseError extends Error {
  constructor(msg: string, public line: number, public col: number) {
    super(`parse error at ${line}:${col}: ${msg}`);
    this.name = "ParseError";
  }
}

const RELOPS = new Set<string>([">", "<", ">=", "<=", "=="]);

class Parser {
  private p = 0;
  constructor(private toks: Token[]) {}

  private peek(k = 0): Token { return this.toks[this.p + k]; }
  private at(kind: string, value?: string): boolean {
    const t = this.peek();
    if (t.kind !== kind && !(kind === "kw" && t.kind === "kw")) {
      // allow generic kind match below
    }
    return t.kind === kind && (value === undefined || t.value === value);
  }
  private isKw(v: string) { const t = this.peek(); return t.kind === "kw" && t.value === v; }
  private isOp(v: string) { const t = this.peek(); return t.kind === "op" && t.value === v; }
  private adv(): Token { return this.toks[this.p++]; }
  private pos(t = this.peek()): Pos { return { line: t.line, col: t.col }; }

  private expectOp(v: string): Token {
    if (!this.isOp(v)) this.fail(`expected '${v}'`);
    return this.adv();
  }
  private expectKw(v: string): Token {
    if (!this.isKw(v)) this.fail(`expected keyword '${v}'`);
    return this.adv();
  }
  private expectIdent(): string {
    const t = this.peek();
    if (t.kind !== "ident") this.fail("expected identifier");
    return this.adv().value;
  }
  private fail(msg: string): never {
    const t = this.peek();
    throw new ParseError(`${msg}, found ${t.kind} '${t.value}'`, t.line, t.col);
  }

  // program ::= { decl }
  parseProgram(): Program {
    const decls: Decl[] = [];
    while (!this.at("eof")) decls.push(this.parseDecl());
    return { decls };
  }

  private parseDecl(): Decl {
    if (this.isKw("floor")) return this.parseFloor();
    if (this.isKw("import")) return this.parseImport();
    if (this.isKw("module")) return this.parseModule();
    return this.parseStmt();
  }

  private parseFloor(): Decl {
    const pos = this.pos();
    this.expectKw("floor");
    const value = this.parseNumberValue();
    return { tag: "floor", value, pos };
  }

  private parseImport(): Decl {
    const pos = this.pos();
    this.expectKw("import");
    return { tag: "import", name: this.parseQName(), pos };
  }

  private parseModule(): Decl {
    const pos = this.pos();
    this.expectKw("module");
    const name = this.expectIdent();
    this.expectOp("{");
    const body: Decl[] = [];
    while (!this.isOp("}") && !this.at("eof")) body.push(this.parseDecl());
    this.expectOp("}");
    return { tag: "module", name, body, pos };
  }

  // stmt ::= bind | track | observe | assert | expr
  private parseStmt(): Stmt {
    if (this.isKw("track")) return this.parseTrack();
    if (this.isKw("observe")) return this.parseObserve();
    if (this.isKw("assert")) return this.parseAssert();
    // bind:  [let] ident := expr     (lookahead for ':=')
    if (this.isKw("let")) { this.adv(); return this.parseBindAfterName(); }
    if (this.peek().kind === "ident" &&
        this.peek(1).kind === "op" && this.peek(1).value === ":=") {
      return this.parseBindAfterName();
    }
    const pos = this.pos();
    return { tag: "exprStmt", expr: this.parseExpr(), pos };
  }

  private parseBindAfterName(): Stmt {
    const pos = this.pos();
    const name = this.expectIdent();
    this.expectOp(":=");
    // `name := track ...`  parses the track as an expression (yield role optional)
    if (this.isKw("track")) {
      return { tag: "bind", name, value: this.parseTrackExpr(), pos };
    }
    return { tag: "bind", name, value: this.parseExpr(), pos };
  }

  // track as an expression: track ident in expr [with reps ids] until <admit> [yield ident]
  private parseTrackExpr(): Expr {
    const pos = this.pos();
    this.expectKw("track");
    const item = this.expectIdent();
    this.expectKw("in");
    const process = this.parseExpr();
    let reps: string[] | undefined;
    if (this.isKw("with")) {
      this.adv();
      this.expectKw("reps");
      reps = [this.expectIdent()];
      while (this.isOp(",")) { this.adv(); reps.push(this.expectIdent()); }
    }
    this.expectKw("until");
    const admit = this.parseAdmit();
    let role: string | undefined;
    if (this.isKw("yield")) { this.adv(); role = this.expectIdent(); }
    return { tag: "trackExpr", item, process, reps, admit, role, pos };
  }

  private parseObserve(): Stmt {
    const pos = this.pos();
    this.expectKw("observe");
    const expr = this.parseExpr();
    let asName: string | undefined;
    if (this.isKw("as")) { this.adv(); asName = this.expectIdent(); }
    return { tag: "observe", expr, as: asName, pos };
  }

  private parseAssert(): Stmt {
    const pos = this.pos();
    this.expectKw("assert");
    const cond = this.parseCond();
    let emit: string | undefined;
    if (this.isKw("emit")) { this.adv(); emit = this.parseStringValue(); }
    return { tag: "assert", cond, emit, pos };
  }

  // track ident in expr [with reps idlist] until <admit> yield ident
  private parseTrack(): Stmt {
    const pos = this.pos();
    this.expectKw("track");
    const item = this.expectIdent();
    this.expectKw("in");
    const process = this.parseExpr();
    let reps: string[] | undefined;
    if (this.isKw("with")) {
      this.adv();
      this.expectKw("reps");
      reps = [this.expectIdent()];
      while (this.isOp(",")) { this.adv(); reps.push(this.expectIdent()); }
    }
    this.expectKw("until");
    const admit = this.parseAdmit();
    this.expectKw("yield");
    const yieldName = this.expectIdent();
    return { tag: "track", item, process, reps, admit, yieldName, pos };
  }

  private parseAdmit(): Admit {
    if (this.isKw("converge")) { this.adv(); return { kind: "converge" }; }
    if (this.isKw("diverge")) { this.adv(); return { kind: "diverge" }; }
    return { kind: "cond", cond: this.parseCond() };
  }

  // ---- expressions ----
  private parseExpr(): Expr {
    let e = this.parsePrimary();
    // bond:  expr ~ expr  [when cond]
    while (this.isOp("~")) {
      const pos = this.pos();
      this.adv();
      const right = this.parsePrimary();
      let guard: Cond | undefined;
      if (this.isKw("when")) { this.adv(); guard = this.parseCond(); }
      e = { tag: "bond", left: e, right, guard, pos };
    }
    return e;
  }

  private parsePrimary(): Expr {
    const t = this.peek();
    const pos = this.pos();

    if (this.isKw("cut")) {
      this.adv();
      return { tag: "cut", arg: this.parsePrimary(), pos };
    }
    if (this.isKw("close")) {
      this.adv();
      const central = this.expectIdent();
      this.expectOp("(");
      const args = this.parseArgList();
      this.expectOp(")");
      let by: string | undefined;
      if (this.isKw("by")) { this.adv(); by = this.expectIdent(); }
      return { tag: "close", central, args, by, pos };
    }
    if (this.isOp("(")) {
      this.adv();
      const e = this.parseExpr();
      this.expectOp(")");
      return e;
    }
    if (t.kind === "num") {
      const value = this.parseNumberValue();
      return { tag: "num", value: value, floor: this.lastFloor, pos };
    }
    if (t.kind === "str") {
      return { tag: "str", value: this.parseStringValue(), pos };
    }
    if (t.kind === "ident") {
      // qname, optionally a call  qname(args)
      const name = this.parseQName();
      if (this.isOp("(")) {
        this.adv();
        const args = this.parseArgList();
        this.expectOp(")");
        return { tag: "call", name, args, pos };
      }
      if (name.length === 1) return { tag: "ref", name: name[0], pos };
      // dotted ref with no call: treat as call with no args
      return { tag: "call", name, args: [], pos };
    }
    this.fail("expected expression");
  }

  private parseArgList(): Arg[] {
    const args: Arg[] = [];
    if (this.isOp(")")) return args;
    args.push(this.parseArg());
    while (this.isOp(",")) { this.adv(); args.push(this.parseArg()); }
    return args;
  }

  private parseArg(): Arg {
    // [ident :] expr   (named arg). Disambiguate ident ':' expr.
    if (this.peek().kind === "ident" &&
        this.peek(1).kind === "op" && this.peek(1).value === ":") {
      const label = this.adv().value;
      this.adv(); // ':'
      return { label, value: this.parseExpr() };
    }
    return { value: this.parseExpr() };
  }

  private parseCond(): Cond {
    const pos = this.pos();
    const left = this.parsePrimary();
    const t = this.peek();
    if (t.kind !== "op" || !RELOPS.has(t.value)) this.fail("expected relational operator");
    const op = this.adv().value as RelOp;
    const right = this.parsePrimary();
    return { tag: "cond", left, op, right, pos };
  }

  private parseQName(): string[] {
    const parts = [this.expectIdent()];
    while (this.isOp(".")) { this.adv(); parts.push(this.expectIdent()); }
    return parts;
  }

  // number  ::= numlit [ '#' numlit ]   (value at floor)
  private lastFloor: number | undefined;
  private parseNumberValue(): number {
    const t = this.peek();
    if (t.kind !== "num") this.fail("expected number");
    this.adv();
    let floorVal: number | undefined;
    if (this.isOp("#")) {
      this.adv();
      const ft = this.peek();
      if (ft.kind !== "num") this.fail("expected floor literal after '#'");
      this.adv();
      floorVal = parseFloat(ft.value);
    }
    this.lastFloor = floorVal;
    return parseFloat(t.value);
  }

  private parseStringValue(): string {
    const t = this.peek();
    if (t.kind !== "str") this.fail("expected string");
    this.adv();
    return t.value;
  }
}

export function parse(src: string): Program {
  return new Parser(lex(src)).parseProgram();
}
