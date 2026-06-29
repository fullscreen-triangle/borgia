//! Honjo Masamune — CLI runner.  Usage: honjo <file.hj> [--ast]

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: honjo <file.hj> [--ast]");
        exit(if args.len() < 2 { 1 } else { 0 });
    }
    let file = match args.iter().skip(1).find(|a| !a.starts_with("--")) {
        Some(f) => f,
        None => {
            eprintln!("no input file");
            exit(1);
        }
    };
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {}", file, e);
            exit(1);
        }
    };

    if args.iter().any(|a| a == "--ast") {
        match honjo::compile(&src) {
            Ok(p) => {
                println!("{:#?}", p);
                exit(0);
            }
            Err(e) => {
                eprintln!("{}", e);
                exit(1);
            }
        }
    }

    match honjo::evaluate(&src) {
        Ok(r) => {
            for line in &r.log {
                println!("{}", line);
            }
            println!(
                "-- cut count (clock) M = {} ; floor = {} ; {}",
                r.cut_count,
                r.floor,
                if r.ok { "ok" } else { "ABORTED" }
            );
            exit(if r.ok { 0 } else { 2 });
        }
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
    }
}
