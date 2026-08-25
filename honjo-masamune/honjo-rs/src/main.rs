//! Honjo Masamune — CLI runner.  Usage: honjo <file.hj> [--ast]

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: honjo <file.hj> [--ast]");
        eprintln!("       honjo serve [--port N] [--origin URL]...");
        eprintln!();
        eprintln!("  serve  start a local engine for the web workbench and");
        eprintln!("         print a connection token. Computation stays on");
        eprintln!("         this machine; nothing is uploaded.");
        exit(if args.len() < 2 { 1 } else { 0 });
    }

    if args[1] == "serve" {
        let mut cfg = honjo::serve::Config::default();
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--port" => {
                    i += 1;
                    match args.get(i).and_then(|p| p.parse::<u16>().ok()) {
                        Some(p) => cfg.port = p,
                        None => {
                            eprintln!("--port requires a number 1-65535");
                            exit(1);
                        }
                    }
                }
                "--origin" => {
                    i += 1;
                    match args.get(i) {
                        Some(o) => cfg.allowed_origins.push(o.clone()),
                        None => {
                            eprintln!("--origin requires a URL");
                            exit(1);
                        }
                    }
                }
                other => {
                    eprintln!("unknown option: {}", other);
                    exit(1);
                }
            }
            i += 1;
        }
        if let Err(e) = honjo::serve::serve(cfg) {
            eprintln!("{}", e);
            exit(1);
        }
        exit(0);
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
