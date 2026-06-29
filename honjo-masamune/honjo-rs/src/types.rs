//! Honjo Masamune — accountability type checker (§5).
//! Enforces the No-Zero-Residue invariant: every cut value has floor > 0.

use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TyName {
    Atom,
    Bond,
    Compound,
    Path,
    Scalar,
    Cut,
    Bool,
    Void,
}

#[derive(Debug, Clone, Copy)]
pub struct Ty {
    pub name: TyName,
    pub floor: f64,
}

fn cut_like(t: TyName) -> bool {
    matches!(t, TyName::Atom | TyName::Bond | TyName::Compound | TyName::Path | TyName::Cut)
}

struct Checker {
    ambient_floor: f64,
    vars: HashMap<String, Ty>,
}

type CResult<T> = Result<T, String>;

impl Checker {
    fn require_positive(&self, f: f64, pos: Pos) -> CResult<f64> {
        if f > 0.0 {
            Ok(f)
        } else {
            Err(format!(
                "type error at {}:{}: floor must be > 0 (got {}); the sharp cut is not expressible",
                pos.line, pos.col, f
            ))
        }
    }

    fn check_expr(&mut self, e: &Expr) -> CResult<Ty> {
        match e {
            Expr::Num { value: _, floor, pos } => {
                let f = floor.unwrap_or(self.ambient_floor);
                if f.is_nan() {
                    return Err(format!(
                        "type error at {}:{}: numeric literal used before any 'floor' declaration",
                        pos.line, pos.col
                    ));
                }
                self.require_positive(f, *pos)?;
                Ok(Ty { name: TyName::Scalar, floor: f })
            }
            Expr::Str { .. } => {
                let f = if self.ambient_floor.is_nan() { 1.0 } else { self.ambient_floor };
                Ok(Ty { name: TyName::Void, floor: f })
            }
            Expr::Ref { name, pos } => self
                .vars
                .get(name)
                .copied()
                .ok_or_else(|| format!("type error at {}:{}: unbound identifier '{}'", pos.line, pos.col, name)),
            Expr::Cut { arg, pos } => {
                let at = self.check_expr(arg)?;
                if at.name != TyName::Scalar {
                    return Err(format!("type error at {}:{}: cut expects an atomic number (Scalar)", pos.line, pos.col));
                }
                let f = self.require_positive(self.ambient_floor, *pos)?;
                Ok(Ty { name: TyName::Atom, floor: f })
            }
            Expr::Bond { left, right, guard, pos } => {
                let lt = self.check_expr(left)?;
                let rt = self.check_expr(right)?;
                if !cut_like(lt.name) || !cut_like(rt.name) {
                    return Err(format!("type error at {}:{}: a bond (~) joins two cut-like values", pos.line, pos.col));
                }
                if let Some(g) = guard {
                    self.check_cond(g)?;
                }
                let f = self.require_positive(self.ambient_floor, *pos)?;
                Ok(Ty { name: TyName::Bond, floor: f })
            }
            Expr::Close { central, args, pos, .. } => {
                let ct = self
                    .vars
                    .get(central)
                    .copied()
                    .ok_or_else(|| format!("type error at {}:{}: unbound central atom '{}'", pos.line, pos.col, central))?;
                if !cut_like(ct.name) {
                    return Err(format!("type error at {}:{}: close expects an Atom as the central item", pos.line, pos.col));
                }
                for a in args {
                    let t = self.check_expr(&a.value)?;
                    if !cut_like(t.name) {
                        return Err(format!("type error at {}:{}: close ligands must be cut-like (Atom)", pos.line, pos.col));
                    }
                }
                let f = self.require_positive(self.ambient_floor, *pos)?;
                Ok(Ty { name: TyName::Compound, floor: f })
            }
            Expr::TrackExpr { item, process, admit, pos, .. } => {
                let it = self
                    .vars
                    .get(item)
                    .copied()
                    .ok_or_else(|| format!("type error at {}:{}: tracking unbound item '{}'", pos.line, pos.col, item))?;
                if !cut_like(it.name) {
                    return Err(format!("type error at {}:{}: track expects an Atom/Compound item", pos.line, pos.col));
                }
                self.check_expr(process)?;
                if let Admit::Cond(c) = admit {
                    self.check_cond(c)?;
                }
                let f = self.require_positive(self.ambient_floor, *pos)?;
                Ok(Ty { name: TyName::Path, floor: f })
            }
            Expr::Call { name, args, pos } => {
                for a in args {
                    self.check_expr(&a.value)?;
                }
                let f = if self.ambient_floor.is_nan() { 1.0 } else { self.ambient_floor };
                let _ = pos;
                let verb = name.last().map(|s| s.as_str()).unwrap_or("");
                let n = match verb {
                    "atom" | "individuate" => TyName::Atom,
                    "bond" => TyName::Bond,
                    "close" | "compound" => TyName::Compound,
                    "track" | "propagate" | "amalgamation" => TyName::Path,
                    _ => TyName::Cut,
                };
                Ok(Ty { name: n, floor: f })
            }
        }
    }

    fn check_cond(&mut self, c: &Cond) -> CResult<()> {
        self.type_cond_operand(&c.left)?;
        self.type_cond_operand(&c.right)?;
        Ok(())
    }

    fn type_cond_operand(&mut self, e: &Expr) -> CResult<()> {
        // bare refs / calls are measured fields (delta, W.valence) — accepted.
        match e {
            Expr::Ref { .. } | Expr::Call { .. } => Ok(()),
            _ => self.check_expr(e).map(|_| ()),
        }
    }

    fn check_stmt(&mut self, s: &Stmt) -> CResult<()> {
        match s {
            Stmt::Bind { name, value, .. } => {
                let t = self.check_expr(value)?;
                self.vars.insert(name.clone(), t);
            }
            Stmt::ExprStmt { expr, .. } => {
                self.check_expr(expr)?;
            }
            Stmt::Observe { expr, as_name, .. } => {
                let t = self.check_expr(expr)?;
                if let Some(a) = as_name {
                    self.vars.insert(a.clone(), t);
                }
            }
            Stmt::Assert { cond, .. } => {
                self.check_cond(cond)?;
            }
            Stmt::Track { item, process, admit, yield_name, pos } => {
                let it = self
                    .vars
                    .get(item)
                    .copied()
                    .ok_or_else(|| format!("type error at {}:{}: tracking unbound item '{}'", pos.line, pos.col, item))?;
                if !cut_like(it.name) {
                    return Err(format!("type error at {}:{}: track expects an Atom/Compound item", pos.line, pos.col));
                }
                self.check_expr(process)?;
                if let Admit::Cond(c) = admit {
                    self.check_cond(c)?;
                }
                let f = self.require_positive(self.ambient_floor, *pos)?;
                self.vars.insert(yield_name.clone(), Ty { name: TyName::Path, floor: f });
            }
        }
        Ok(())
    }

    fn check_decl(&mut self, d: &Decl) -> CResult<()> {
        match d {
            Decl::Floor { value, pos } => {
                self.require_positive(*value, *pos)?;
                self.ambient_floor = *value;
            }
            Decl::Import { .. } => {}
            Decl::Module { body, .. } => {
                for inner in body {
                    self.check_decl(inner)?;
                }
            }
            Decl::Stmt(s) => self.check_stmt(s)?,
        }
        Ok(())
    }
}

pub fn check(program: &Program) -> Result<(), String> {
    let mut c = Checker { ambient_floor: f64::NAN, vars: HashMap::new() };
    for d in &program.decls {
        c.check_decl(d)?;
    }
    Ok(())
}
