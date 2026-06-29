//! Honjo Masamune — interpreter (operational semantics §6).
//! Evaluation IS measurement: each cut event mutates state and strictly
//! increments the cut count M (the intrinsic clock, monotone). This is the
//! exact (reference) realisation; a cut's residue is the exact boundary weight.

use crate::ast::*;
use crate::stdlib::{self, Admit as SAdmit, AtomVal, Value};
use std::collections::HashMap;

pub struct RunResult {
    pub cut_count: i64,
    pub floor: f64,
    pub named: HashMap<String, Value>,
    pub log: Vec<String>,
    pub ok: bool,
}

struct Interp {
    floor: f64,
    m: i64, // committed-cut count; monotone
    env: HashMap<String, Value>,
    log: Vec<String>,
    // insertion order of named values, for deterministic reporting
    order: Vec<String>,
}

type IResult<T> = Result<T, String>;

impl Interp {
    fn bind(&mut self, name: &str, v: Value) {
        if !self.env.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.env.insert(name.to_string(), v);
    }

    fn get_atom(&self, name: &str) -> IResult<AtomVal> {
        match self.env.get(name) {
            Some(Value::Atom(a)) => Ok(a.clone()),
            Some(other) => Err(format!("expected Atom for '{}', got {:?}", name, kind_of(other))),
            None => Err(format!("unbound name '{}'", name)),
        }
    }

    fn eval_expr(&mut self, e: &Expr) -> IResult<Value> {
        match e {
            Expr::Num { value, floor, .. } => {
                Ok(Value::Scalar { value: *value, floor: floor.unwrap_or(self.floor) })
            }
            Expr::Str { .. } => Ok(Value::Scalar { value: f64::NAN, floor: self.floor }),
            Expr::Ref { name, .. } => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| format!("unbound name '{}'", name)),
            Expr::Cut { arg, .. } => {
                let z = self.eval_scalar(arg)?;
                let atom = stdlib::individuate(z as i64, self.floor)?;
                self.m += 1;
                Ok(Value::Atom(atom))
            }
            Expr::Bond { left, right, guard, .. } => {
                let a = self.eval_atom(left)?;
                let b = self.eval_atom(right)?;
                let bd = stdlib::bond(&a, &b, self.floor);
                if let Some(g) = guard {
                    if !self.eval_cond(g, Some(&Value::Bond(bd.clone())))? {
                        let mut dead = bd.clone();
                        dead.exists = false;
                        return Ok(Value::Bond(dead));
                    }
                }
                self.m += 1;
                Ok(Value::Bond(bd))
            }
            Expr::Close { central, args, .. } => {
                let c = self.get_atom(central)?;
                let mut ligs = Vec::new();
                for a in args {
                    ligs.push(self.eval_atom(&a.value)?);
                }
                let comp = stdlib::close(&c, &ligs, self.floor)?;
                self.m += comp.ligands;
                Ok(Value::Compound(comp))
            }
            Expr::TrackExpr { item, process, reps, admit, .. } => {
                self.eval_track(item, process, reps, admit)
            }
            Expr::Call { name, args, .. } => {
                let verb = name.last().map(|s| s.as_str()).unwrap_or("");
                if verb == "individuate" || verb == "atom" {
                    let z = self.eval_scalar(&args[0].value)?;
                    let atom = stdlib::individuate(z as i64, self.floor)?;
                    self.m += 1;
                    Ok(Value::Atom(atom))
                } else {
                    Err(format!("unknown call '{}'", name.join(".")))
                }
            }
        }
    }

    fn eval_track(
        &mut self,
        item: &str,
        process: &Expr,
        reps: &Option<Vec<String>>,
        admit: &Admit,
    ) -> IResult<Value> {
        let it = self.get_atom(item)?;
        let proc = self.eval_expr(process)?;
        let sadmit = match admit {
            Admit::Converge => SAdmit::Converge,
            Admit::Diverge => SAdmit::Diverge,
            Admit::Cond(c) => SAdmit::Cond(self.eval_cond(c, Some(&proc))?),
        };
        let reps_v = reps.clone().unwrap_or_default();
        let path = stdlib::propagate(&it, &proc, &reps_v, sadmit, self.floor)?;
        self.m += path.steps;
        Ok(Value::Path(path))
    }

    fn eval_scalar(&mut self, e: &Expr) -> IResult<f64> {
        match self.eval_expr(e)? {
            Value::Scalar { value, .. } => Ok(value),
            other => Err(format!("expected Scalar, got {:?}", kind_of(&other))),
        }
    }

    fn eval_atom(&mut self, e: &Expr) -> IResult<AtomVal> {
        match self.eval_expr(e)? {
            Value::Atom(a) => Ok(a),
            other => Err(format!("expected Atom, got {:?}", kind_of(&other))),
        }
    }

    fn eval_cond(&mut self, c: &Cond, ctx: Option<&Value>) -> IResult<bool> {
        let l = self.eval_operand(&c.left, ctx)?;
        let r = self.eval_operand(&c.right, ctx)?;
        Ok(match c.op {
            RelOp::Gt => l > r,
            RelOp::Lt => l < r,
            RelOp::Ge => l >= r,
            RelOp::Le => l <= r,
            RelOp::Eq => l == r,
        })
    }

    fn eval_operand(&mut self, e: &Expr, ctx: Option<&Value>) -> IResult<f64> {
        match e {
            // bare name: a measured field of the context value (delta, valence...)
            Expr::Ref { name, .. } => {
                if let Some(v) = self.env.get(name) {
                    return Ok(scalar_of(v));
                }
                if name == "closed" {
                    // sentinel: `valence == closed` -> 1 when closed
                    return Ok(1.0);
                }
                if let Some(cx) = ctx {
                    return field_value(cx, name);
                }
                Err(format!("unresolved measured field '{}'", name))
            }
            // dotted field W.valence
            Expr::Call { name, args, .. } if name.len() == 2 && args.is_empty() => {
                let base = self
                    .env
                    .get(&name[0])
                    .cloned()
                    .ok_or_else(|| format!("unbound '{}'", name[0]))?;
                field_value(&base, &name[1])
            }
            _ => Ok(scalar_of(&self.eval_expr(e)?)),
        }
    }

    fn exec_stmt(&mut self, s: &Stmt) -> IResult<bool> {
        match s {
            Stmt::Bind { name, value, .. } => {
                let v = self.eval_expr(value)?;
                self.bind(name, v);
            }
            Stmt::ExprStmt { expr, .. } => {
                self.eval_expr(expr)?;
            }
            Stmt::Observe { expr, as_name, .. } => {
                let v = self.eval_expr(expr)?;
                let label = expr_name(expr);
                self.log.push(render_value(&v, label.as_deref()));
                if let Some(a) = as_name {
                    self.bind(a, v);
                }
            }
            Stmt::Assert { cond, emit, .. } => {
                let ctx = self.last_cut_context();
                if !self.eval_cond(cond, ctx.as_ref())? {
                    self.log.push(format!("ABORT: {}", emit.clone().unwrap_or_else(|| "assertion failed".into())));
                    return Ok(false);
                }
            }
            Stmt::Track { item, process, reps, admit, yield_name, .. } => {
                let v = self.eval_track(item, process, reps, admit)?;
                self.bind(yield_name, v);
            }
        }
        Ok(true)
    }

    fn last_cut_context(&self) -> Option<Value> {
        for name in self.order.iter().rev() {
            if let Some(v) = self.env.get(name) {
                if !matches!(v, Value::Scalar { .. }) {
                    return Some(v.clone());
                }
            }
        }
        None
    }

    fn exec_decl(&mut self, d: &Decl) -> IResult<bool> {
        match d {
            Decl::Floor { value, .. } => {
                if *value <= 0.0 {
                    return Err("floor must be > 0".into());
                }
                self.floor = *value;
            }
            Decl::Import { .. } => {}
            Decl::Module { body, .. } => {
                for inner in body {
                    if !self.exec_decl(inner)? {
                        return Ok(false);
                    }
                }
            }
            Decl::Stmt(s) => return self.exec_stmt(s),
        }
        Ok(true)
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Scalar { .. } => "Scalar",
        Value::Atom(_) => "Atom",
        Value::Bond(_) => "Bond",
        Value::Compound(_) => "Compound",
        Value::Path(_) => "Path",
    }
}

fn scalar_of(v: &Value) -> f64 {
    match v {
        Value::Scalar { value, .. } => *value,
        Value::Atom(a) => a.residue,
        Value::Bond(b) => b.residue,
        Value::Compound(c) => c.residue,
        Value::Path(p) => p.residue,
    }
}

fn field_value(v: &Value, field: &str) -> Result<f64, String> {
    if field == "valence" || field == "closed" {
        match v {
            Value::Compound(c) => return Ok(if c.valence_closed { 1.0 } else { 0.0 }),
            Value::Atom(a) => return Ok(if a.vacancy == 0 { 1.0 } else { 0.0 }),
            _ => {}
        }
    }
    match (v, field) {
        (Value::Atom(a), "vacancy") => Ok(a.vacancy as f64),
        (Value::Atom(a), "residue") => Ok(a.residue),
        (Value::Bond(b), "delta") => Ok(b.delta),
        (Value::Bond(b), "shared") => Ok(b.shared as f64),
        (Value::Bond(b), "residue") => Ok(b.residue),
        (Value::Compound(c), "angle") => Ok(c.angle_deg.unwrap_or(f64::NAN)),
        (Value::Compound(c), "ligands") => Ok(c.ligands as f64),
        (Value::Compound(c), "residue") => Ok(c.residue),
        (Value::Path(p), "steps") => Ok(p.steps as f64),
        (Value::Path(p), "residue") => Ok(p.residue),
        _ => Err(format!("value {} has no numeric field '{}'", kind_of(v), field)),
    }
}

fn expr_name(e: &Expr) -> Option<String> {
    match e {
        Expr::Ref { name, .. } => Some(name.clone()),
        _ => None,
    }
}

fn fmt(n: f64) -> String {
    if n.is_nan() {
        return "NaN".into();
    }
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        let s = format!("{:.4}", n);
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    }
}

pub fn render_value(v: &Value, name: Option<&str>) -> String {
    let tag = name.map(|n| format!("{} : ", n)).unwrap_or_default();
    match v {
        Value::Atom(a) => format!(
            "{}Atom @ {}  Z={} {}  {}  {}  vacancy={}  valence={}  residue={}",
            tag, fmt(a.floor), a.z, a.symbol, a.config, a.term, a.vacancy, a.valence, fmt(a.residue)
        ),
        Value::Bond(b) => format!(
            "{}Bond @ {}  {}~{}  exists={}  delta={}  shared={}  residue={}",
            tag, fmt(b.floor), b.a, b.b, b.exists, fmt(b.delta), b.shared, fmt(b.residue)
        ),
        Value::Compound(c) => {
            let lig = if c.formula.1 > 1 {
                format!("{}{}", c.ligand, c.formula.1)
            } else if c.formula.1 == 1 {
                c.ligand.clone()
            } else {
                String::new()
            };
            let formula = if c.formula.0 == 2 {
                format!("{}2", c.central)
            } else {
                format!("{}{}", c.central, lig)
            };
            let angle = c.angle_deg.map(fmt).unwrap_or_else(|| "-".into());
            format!(
                "{}Compound @ {}  {}  geometry={}  angle={}  closed={}  residue={}",
                tag, fmt(c.floor), formula, c.geometry, angle, c.valence_closed, fmt(c.residue)
            )
        }
        Value::Path(p) => format!(
            "{}Path @ {}  item={}  steps={}  converged={}  reps=[{}]  amalgamation=[{}]  residue={}",
            tag, fmt(p.floor), p.item, p.steps, p.converged, p.reps.join(","), p.amalgamation.join(", "), fmt(p.residue)
        ),
        Value::Scalar { value, floor } => format!("{}Scalar @ {}  {}", tag, fmt(*floor), fmt(*value)),
    }
}

pub fn run(program: &Program) -> Result<RunResult, String> {
    let mut it = Interp {
        floor: 1.0,
        m: 0,
        env: HashMap::new(),
        log: Vec::new(),
        order: Vec::new(),
    };
    let mut ok = true;
    for d in &program.decls {
        if !it.exec_decl(d)? {
            ok = false;
            break;
        }
    }
    Ok(RunResult { cut_count: it.m, floor: it.floor, named: it.env, log: it.log, ok })
}
