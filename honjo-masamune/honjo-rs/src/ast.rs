//! Honjo Masamune — AST (mirrors EBNF grammar §4).

#[derive(Debug, Clone, Copy)]
pub struct Pos {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
}

#[derive(Debug, Clone)]
pub struct Cond {
    pub left: Expr,
    pub op: RelOp,
    pub right: Expr,
    pub pos: Pos,
}

#[derive(Debug, Clone)]
pub enum Admit {
    Converge,
    Diverge,
    Cond(Box<Cond>),
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub label: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Num { value: f64, floor: Option<f64>, pos: Pos },
    Str { value: String, pos: Pos },
    Ref { name: String, pos: Pos },
    Cut { arg: Box<Expr>, pos: Pos },
    Bond { left: Box<Expr>, right: Box<Expr>, guard: Option<Box<Cond>>, pos: Pos },
    Close { central: String, args: Vec<Arg>, by: Option<String>, pos: Pos },
    Call { name: Vec<String>, args: Vec<Arg>, pos: Pos },
    TrackExpr {
        item: String,
        process: Box<Expr>,
        reps: Option<Vec<String>>,
        admit: Admit,
        role: Option<String>,
        pos: Pos,
    },
}

impl Expr {
    pub fn pos(&self) -> Pos {
        match self {
            Expr::Num { pos, .. }
            | Expr::Str { pos, .. }
            | Expr::Ref { pos, .. }
            | Expr::Cut { pos, .. }
            | Expr::Bond { pos, .. }
            | Expr::Close { pos, .. }
            | Expr::Call { pos, .. }
            | Expr::TrackExpr { pos, .. } => *pos,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Bind { name: String, value: Expr, pos: Pos },
    ExprStmt { expr: Expr, pos: Pos },
    Observe { expr: Expr, as_name: Option<String>, pos: Pos },
    Assert { cond: Cond, emit: Option<String>, pos: Pos },
    Track {
        item: String,
        process: Expr,
        reps: Option<Vec<String>>,
        admit: Admit,
        yield_name: String,
        pos: Pos,
    },
}

#[derive(Debug, Clone)]
pub enum Decl {
    Floor { value: f64, pos: Pos },
    Import { name: Vec<String>, pos: Pos },
    Module { name: String, body: Vec<Decl>, pos: Pos },
    Stmt(Stmt),
}

#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}
