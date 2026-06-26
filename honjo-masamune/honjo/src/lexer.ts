// Honjo Masamune — Lexer
// Implements the lexical structure of the language spec (§3 "Lexical structure").
// UTF-8 source, `.hj`. `--` line comments; whitespace is insignificant.

export type TokKind =
  | "kw"        // keyword
  | "ident"     // identifier
  | "num"       // numeric literal
  | "str"       // string literal
  | "op"        // operator / punctuation
  | "eof";

export interface Token {
  kind: TokKind;
  value: string;
  line: number;
  col: number;
}

// Keyword set from spec §3 (Definition: Token classes).
export const KEYWORDS = new Set<string>([
  "floor", "cut", "close", "track", "until", "yield", "when", "do",
  "emit", "observe", "in", "as", "let", "medium", "converge", "diverge",
  "with", "by", "import", "module", "export", "assert", "reps",
]);

// Multi-char operators, longest first so the scanner is greedy.
const MULTI_OPS = [":=", ">=", "<=", "==", "->"];
const SINGLE_OPS = new Set<string>([
  "~", "(", ")", "[", "]", "{", "}", ",", ":", ".", ">", "<", "#",
]);

export class LexError extends Error {
  constructor(msg: string, public line: number, public col: number) {
    super(`lex error at ${line}:${col}: ${msg}`);
    this.name = "LexError";
  }
}

const isIdentStart = (c: string) => /[A-Za-z_]/.test(c);
const isIdentPart = (c: string) => /[A-Za-z0-9_]/.test(c);
const isDigit = (c: string) => c >= "0" && c <= "9";

export function lex(src: string): Token[] {
  const toks: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const n = src.length;

  const peek = (k = 0) => src[i + k] ?? "";
  const adv = () => {
    const c = src[i++];
    if (c === "\n") { line++; col = 1; } else { col++; }
    return c;
  };

  while (i < n) {
    const c = peek();

    // whitespace
    if (c === " " || c === "\t" || c === "\r" || c === "\n") { adv(); continue; }

    // line comment  --  ...  to end of line
    if (c === "-" && peek(1) === "-") {
      while (i < n && peek() !== "\n") adv();
      continue;
    }

    const startLine = line, startCol = col;

    // string literal
    if (c === '"') {
      adv();
      let s = "";
      while (i < n && peek() !== '"') {
        const ch = adv();
        if (ch === "\\") {
          const e = adv();
          s += e === "n" ? "\n" : e === "t" ? "\t" : e;
        } else {
          s += ch;
        }
      }
      if (i >= n) throw new LexError("unterminated string", startLine, startCol);
      adv(); // closing quote
      toks.push({ kind: "str", value: s, line: startLine, col: startCol });
      continue;
    }

    // number: [0-9]+(.[0-9]+)?([eE][+-]?[0-9]+)?
    if (isDigit(c) || (c === "." && isDigit(peek(1)))) {
      let s = "";
      while (isDigit(peek())) s += adv();
      if (peek() === ".") { s += adv(); while (isDigit(peek())) s += adv(); }
      if (peek() === "e" || peek() === "E") {
        s += adv();
        if (peek() === "+" || peek() === "-") s += adv();
        if (!isDigit(peek())) throw new LexError("malformed exponent", startLine, startCol);
        while (isDigit(peek())) s += adv();
      }
      toks.push({ kind: "num", value: s, line: startLine, col: startCol });
      continue;
    }

    // identifier / keyword
    if (isIdentStart(c)) {
      let s = "";
      while (isIdentPart(peek())) s += adv();
      toks.push({ kind: KEYWORDS.has(s) ? "kw" : "ident", value: s, line: startLine, col: startCol });
      continue;
    }

    // multi-char operator
    const two = c + peek(1);
    if (MULTI_OPS.includes(two)) {
      adv(); adv();
      toks.push({ kind: "op", value: two, line: startLine, col: startCol });
      continue;
    }

    // single-char operator
    if (SINGLE_OPS.has(c)) {
      adv();
      toks.push({ kind: "op", value: c, line: startLine, col: startCol });
      continue;
    }

    throw new LexError(`unexpected character '${c}'`, startLine, startCol);
  }

  toks.push({ kind: "eof", value: "", line, col });
  return toks;
}
