//! Honjo Masamune — Lexer (spec §3). UTF-8 `.hj`, `--` line comments.

#[derive(Debug, Clone, PartialEq)]
pub enum TokKind {
    Kw,
    Ident,
    Num,
    Str,
    Op,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokKind,
    pub value: String,
    pub line: usize,
    pub col: usize,
}

pub const KEYWORDS: &[&str] = &[
    "floor", "cut", "close", "track", "until", "yield", "when", "do", "emit",
    "observe", "in", "as", "let", "medium", "converge", "diverge", "with",
    "by", "import", "module", "export", "assert", "reps",
];

const MULTI_OPS: &[&str] = &[":=", ">=", "<=", "==", "->"];
const SINGLE_OPS: &[char] =
    &['~', '(', ')', '[', ']', '{', '}', ',', ':', '.', '>', '<', '#'];

#[derive(Debug)]
pub struct LexError {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lex error at {}:{}: {}", self.line, self.col, self.msg)
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_part(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut col = 1usize;
    let mut toks = Vec::new();

    macro_rules! adv {
        () => {{
            let c = chars[i];
            i += 1;
            if c == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            c
        }};
    }

    while i < n {
        let c = chars[i];

        // whitespace
        if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            adv!();
            continue;
        }
        // line comment
        if c == '-' && i + 1 < n && chars[i + 1] == '-' {
            while i < n && chars[i] != '\n' {
                adv!();
            }
            continue;
        }

        let (sl, sc) = (line, col);

        // string
        if c == '"' {
            adv!();
            let mut s = String::new();
            while i < n && chars[i] != '"' {
                let ch = adv!();
                if ch == '\\' && i < n {
                    let e = adv!();
                    s.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        other => other,
                    });
                } else {
                    s.push(ch);
                }
            }
            if i >= n {
                return Err(LexError { msg: "unterminated string".into(), line: sl, col: sc });
            }
            adv!(); // closing quote
            toks.push(Token { kind: TokKind::Str, value: s, line: sl, col: sc });
            continue;
        }

        // number
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
            let mut s = String::new();
            while i < n && chars[i].is_ascii_digit() {
                s.push(adv!());
            }
            if i < n && chars[i] == '.' {
                s.push(adv!());
                while i < n && chars[i].is_ascii_digit() {
                    s.push(adv!());
                }
            }
            if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                s.push(adv!());
                if i < n && (chars[i] == '+' || chars[i] == '-') {
                    s.push(adv!());
                }
                if i >= n || !chars[i].is_ascii_digit() {
                    return Err(LexError { msg: "malformed exponent".into(), line: sl, col: sc });
                }
                while i < n && chars[i].is_ascii_digit() {
                    s.push(adv!());
                }
            }
            toks.push(Token { kind: TokKind::Num, value: s, line: sl, col: sc });
            continue;
        }

        // identifier / keyword
        if is_ident_start(c) {
            let mut s = String::new();
            while i < n && is_ident_part(chars[i]) {
                s.push(adv!());
            }
            let kind = if KEYWORDS.contains(&s.as_str()) { TokKind::Kw } else { TokKind::Ident };
            toks.push(Token { kind, value: s, line: sl, col: sc });
            continue;
        }

        // multi-char operator
        if i + 1 < n {
            let two: String = [chars[i], chars[i + 1]].iter().collect();
            if MULTI_OPS.contains(&two.as_str()) {
                adv!();
                adv!();
                toks.push(Token { kind: TokKind::Op, value: two, line: sl, col: sc });
                continue;
            }
        }

        // single-char operator
        if SINGLE_OPS.contains(&c) {
            adv!();
            toks.push(Token { kind: TokKind::Op, value: c.to_string(), line: sl, col: sc });
            continue;
        }

        return Err(LexError { msg: format!("unexpected character '{}'", c), line: sl, col: sc });
    }

    toks.push(Token { kind: TokKind::Eof, value: String::new(), line, col });
    Ok(toks)
}
