//! Honjo Masamune — Rust reference compiler.
//! Pipeline: source -> lex -> parse -> check -> run (exact back end).
//! See ../docs/honjo-dsl/honjo-masamune-dsl.tex for the language spec.

pub mod ast;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod stdlib;
pub mod types;

use interp::RunResult;

/// Front end: parse + accountability check. Returns the typed program.
pub fn compile(src: &str) -> Result<ast::Program, String> {
    let program = parser::parse(src)?;
    types::check(&program)?;
    Ok(program)
}

/// Full pipeline on the reference (exact) back end.
pub fn evaluate(src: &str) -> Result<RunResult, String> {
    let program = compile(src)?;
    interp::run(&program)
}
