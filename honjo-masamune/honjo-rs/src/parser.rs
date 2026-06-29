//! Honjo Masamune — recursive-descent parser (grammar §4).

use crate::ast::*;
use crate::lexer::{lex, Token, TokKind};

#[derive(Debug)]
pub struct ParseError {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at {}:{}: {}", self.line, self.col, self.msg)
    }
}

type PResult<T> = Result<T, ParseError>;

const RELOPS: &[&str] = &[">", "<", ">=", "<=", "=="];

struct Parser {
    toks: Vec<Token>,
    p: usize,
    last_floor: Option<f64>,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.toks[self.p]
    }
    fn peek_at(&self, k: usize) -> &Token {
        &self.toks[(self.p + k).min(self.toks.len() - 1)]
    }
    fn is_kw(&self, v: &str) -> bool {
        let t = self.peek();
        t.kind == TokKind::Kw && t.value == v
    }
    fn is_op(&self, v: &str) -> bool {
        let t = self.peek();
        t.kind == TokKind::Op && t.value == v
    }
    fn is_eof(&self) -> bool {
        self.peek().kind == TokKind::Eof
    }
    fn adv(&mut self) -> Token {
        let t = self.toks[self.p].clone();
        self.p += 1;
        t
    }
    fn pos(&self) -> Pos {
        let t = self.peek();
        Pos { line: t.line, col: t.col }
    }
    fn fail<T>(&self, msg: &str) -> PResult<T> {
        let t = self.peek();
        Err(ParseError {
            msg: format!("{}, found {:?} '{}'", msg, t.kind, t.value),
            line: t.line,
            col: t.col,
        })
    }
    fn expect_op(&mut self, v: &str) -> PResult<()> {
        if self.is_op(v) {
            self.adv();
            Ok(())
        } else {
            self.fail(&format!("expected '{}'", v))
        }
    }
    fn expect_kw(&mut self, v: &str) -> PResult<()> {
        if self.is_kw(v) {
            self.adv();
            Ok(())
        } else {
            self.fail(&format!("expected keyword '{}'", v))
        }
    }
    fn expect_ident(&mut self) -> PResult<String> {
        if self.peek().kind == TokKind::Ident {
            Ok(self.adv().value)
        } else {
            self.fail("expected identifier")
        }
    }

    fn parse_program(&mut self) -> PResult<Program> {
        let mut decls = Vec::new();
        while !self.is_eof() {
            decls.push(self.parse_decl()?);
        }
        Ok(Program { decls })
    }

    fn parse_decl(&mut self) -> PResult<Decl> {
        if self.is_kw("floor") {
            let pos = self.pos();
            self.expect_kw("floor")?;
            let value = self.parse_number_value()?;
            return Ok(Decl::Floor { value, pos });
        }
        if self.is_kw("import") {
            let pos = self.pos();
            self.expect_kw("import")?;
            return Ok(Decl::Import { name: self.parse_qname()?, pos });
        }
        if self.is_kw("module") {
            let pos = self.pos();
            self.expect_kw("module")?;
            let name = self.expect_ident()?;
            self.expect_op("{")?;
            let mut body = Vec::new();
            while !self.is_op("}") && !self.is_eof() {
                body.push(self.parse_decl()?);
            }
            self.expect_op("}")?;
            return Ok(Decl::Module { name, body, pos });
        }
        Ok(Decl::Stmt(self.parse_stmt()?))
    }

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        if self.is_kw("track") {
            return self.parse_track_stmt();
        }
        if self.is_kw("observe") {
            let pos = self.pos();
            self.expect_kw("observe")?;
            let expr = self.parse_expr()?;
            let as_name = if self.is_kw("as") {
                self.adv();
                Some(self.expect_ident()?)
            } else {
                None
            };
            return Ok(Stmt::Observe { expr, as_name, pos });
        }
        if self.is_kw("assert") {
            let pos = self.pos();
            self.expect_kw("assert")?;
            let cond = self.parse_cond()?;
            let emit = if self.is_kw("emit") {
                self.adv();
                Some(self.parse_string_value()?)
            } else {
                None
            };
            return Ok(Stmt::Assert { cond, emit, pos });
        }
        if self.is_kw("let") {
            self.adv();
            return self.parse_bind_after_name();
        }
        if self.peek().kind == TokKind::Ident
            && self.peek_at(1).kind == TokKind::Op
            && self.peek_at(1).value == ":="
        {
            return self.parse_bind_after_name();
        }
        let pos = self.pos();
        Ok(Stmt::ExprStmt { expr: self.parse_expr()?, pos })
    }

    fn parse_bind_after_name(&mut self) -> PResult<Stmt> {
        let pos = self.pos();
        let name = self.expect_ident()?;
        self.expect_op(":=")?;
        if self.is_kw("track") {
            let value = self.parse_track_expr()?;
            return Ok(Stmt::Bind { name, value, pos });
        }
        Ok(Stmt::Bind { name, value: self.parse_expr()?, pos })
    }

    fn parse_track_stmt(&mut self) -> PResult<Stmt> {
        let pos = self.pos();
        self.expect_kw("track")?;
        let item = self.expect_ident()?;
        self.expect_kw("in")?;
        let process = self.parse_expr()?;
        let reps = self.parse_reps()?;
        self.expect_kw("until")?;
        let admit = self.parse_admit()?;
        self.expect_kw("yield")?;
        let yield_name = self.expect_ident()?;
        Ok(Stmt::Track { item, process, reps, admit, yield_name, pos })
    }

    fn parse_track_expr(&mut self) -> PResult<Expr> {
        let pos = self.pos();
        self.expect_kw("track")?;
        let item = self.expect_ident()?;
        self.expect_kw("in")?;
        let process = Box::new(self.parse_expr()?);
        let reps = self.parse_reps()?;
        self.expect_kw("until")?;
        let admit = self.parse_admit()?;
        let role = if self.is_kw("yield") {
            self.adv();
            Some(self.expect_ident()?)
        } else {
            None
        };
        Ok(Expr::TrackExpr { item, process, reps, admit, role, pos })
    }

    fn parse_reps(&mut self) -> PResult<Option<Vec<String>>> {
        if self.is_kw("with") {
            self.adv();
            self.expect_kw("reps")?;
            let mut reps = vec![self.expect_ident()?];
            while self.is_op(",") {
                self.adv();
                reps.push(self.expect_ident()?);
            }
            Ok(Some(reps))
        } else {
            Ok(None)
        }
    }

    fn parse_admit(&mut self) -> PResult<Admit> {
        if self.is_kw("converge") {
            self.adv();
            Ok(Admit::Converge)
        } else if self.is_kw("diverge") {
            self.adv();
            Ok(Admit::Diverge)
        } else {
            Ok(Admit::Cond(Box::new(self.parse_cond()?)))
        }
    }

    fn parse_expr(&mut self) -> PResult<Expr> {
        let mut e = self.parse_primary()?;
        while self.is_op("~") {
            let pos = self.pos();
            self.adv();
            let right = self.parse_primary()?;
            let guard = if self.is_kw("when") {
                self.adv();
                Some(Box::new(self.parse_cond()?))
            } else {
                None
            };
            e = Expr::Bond { left: Box::new(e), right: Box::new(right), guard, pos };
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        let pos = self.pos();
        if self.is_kw("cut") {
            self.adv();
            return Ok(Expr::Cut { arg: Box::new(self.parse_primary()?), pos });
        }
        if self.is_kw("close") {
            self.adv();
            let central = self.expect_ident()?;
            self.expect_op("(")?;
            let args = self.parse_arg_list()?;
            self.expect_op(")")?;
            let by = if self.is_kw("by") {
                self.adv();
                Some(self.expect_ident()?)
            } else {
                None
            };
            return Ok(Expr::Close { central, args, by, pos });
        }
        if self.is_op("(") {
            self.adv();
            let e = self.parse_expr()?;
            self.expect_op(")")?;
            return Ok(e);
        }
        let t = self.peek().clone();
        match t.kind {
            TokKind::Num => {
                let value = self.parse_number_value()?;
                Ok(Expr::Num { value, floor: self.last_floor, pos })
            }
            TokKind::Str => Ok(Expr::Str { value: self.parse_string_value()?, pos }),
            TokKind::Ident => {
                let name = self.parse_qname()?;
                if self.is_op("(") {
                    self.adv();
                    let args = self.parse_arg_list()?;
                    self.expect_op(")")?;
                    Ok(Expr::Call { name, args, pos })
                } else if name.len() == 1 {
                    Ok(Expr::Ref { name: name.into_iter().next().unwrap(), pos })
                } else {
                    Ok(Expr::Call { name, args: vec![], pos })
                }
            }
            _ => self.fail("expected expression"),
        }
    }

    fn parse_arg_list(&mut self) -> PResult<Vec<Arg>> {
        let mut args = Vec::new();
        if self.is_op(")") {
            return Ok(args);
        }
        args.push(self.parse_arg()?);
        while self.is_op(",") {
            self.adv();
            args.push(self.parse_arg()?);
        }
        Ok(args)
    }

    fn parse_arg(&mut self) -> PResult<Arg> {
        if self.peek().kind == TokKind::Ident
            && self.peek_at(1).kind == TokKind::Op
            && self.peek_at(1).value == ":"
        {
            let label = self.adv().value;
            self.adv(); // ':'
            return Ok(Arg { label: Some(label), value: self.parse_expr()? });
        }
        Ok(Arg { label: None, value: self.parse_expr()? })
    }

    fn parse_cond(&mut self) -> PResult<Cond> {
        let pos = self.pos();
        let left = self.parse_primary()?;
        let t = self.peek().clone();
        if t.kind != TokKind::Op || !RELOPS.contains(&t.value.as_str()) {
            return self.fail("expected relational operator");
        }
        let op = match self.adv().value.as_str() {
            ">" => RelOp::Gt,
            "<" => RelOp::Lt,
            ">=" => RelOp::Ge,
            "<=" => RelOp::Le,
            "==" => RelOp::Eq,
            _ => unreachable!(),
        };
        let right = self.parse_primary()?;
        Ok(Cond { left, op, right, pos })
    }

    fn parse_qname(&mut self) -> PResult<Vec<String>> {
        let mut parts = vec![self.expect_ident()?];
        while self.is_op(".") {
            self.adv();
            parts.push(self.expect_ident()?);
        }
        Ok(parts)
    }

    fn parse_number_value(&mut self) -> PResult<f64> {
        if self.peek().kind != TokKind::Num {
            return self.fail("expected number");
        }
        let v: f64 = self.adv().value.parse().unwrap();
        let mut floor_val = None;
        if self.is_op("#") {
            self.adv();
            if self.peek().kind != TokKind::Num {
                return self.fail("expected floor literal after '#'");
            }
            floor_val = Some(self.adv().value.parse::<f64>().unwrap());
        }
        self.last_floor = floor_val;
        Ok(v)
    }

    fn parse_string_value(&mut self) -> PResult<String> {
        if self.peek().kind != TokKind::Str {
            return self.fail("expected string");
        }
        Ok(self.adv().value)
    }
}

pub fn parse(src: &str) -> Result<Program, String> {
    let toks = lex(src).map_err(|e| e.to_string())?;
    let mut p = Parser { toks, p: 0, last_floor: None };
    p.parse_program().map_err(|e| e.to_string())
}
