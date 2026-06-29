//! Integration tests for the Rust reference compiler.
//! Mirrors the TypeScript suite so the two targets are checked against the
//! same expectations (two-target equivalence, §8).

use honjo::evaluate;
use honjo::stdlib::{shell_capacity, Value};

fn named<'a>(r: &'a honjo::interp::RunResult, k: &str) -> &'a Value {
    r.named.get(k).expect("binding present")
}

#[test]
fn carbon() {
    let r = evaluate("floor 1.0\nC := cut 6\nobserve C").unwrap();
    if let Value::Atom(c) = named(&r, "C") {
        assert_eq!(c.symbol, "C");
        assert_eq!(c.config, "[He] 2s2 2p2");
        assert_eq!(c.term, "3P0");
        assert_eq!(c.vacancy, 4);
        assert!(c.residue >= 1.0);
    } else {
        panic!("C not an atom");
    }
    assert_eq!(r.cut_count, 1);
}

#[test]
fn water() {
    let r = evaluate("floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)\nobserve W").unwrap();
    if let Value::Compound(w) = named(&r, "W") {
        assert_eq!(w.formula, (1, 2));
        assert_eq!(w.geometry, "bent");
        assert_eq!(w.angle_deg, Some(104.5));
        assert!(w.valence_closed);
    } else {
        panic!("W not a compound");
    }
}

#[test]
fn bonding_criterion() {
    let r1 = evaluate("floor 1.0\nO := cut 8\nH := cut 1\nb := O ~ H\nobserve b").unwrap();
    if let Value::Bond(b) = named(&r1, "b") {
        assert!(b.exists);
    } else {
        panic!();
    }
    let r2 = evaluate("floor 1.0\nNa := cut 11\nNe := cut 10\nb := Na ~ Ne when delta > 0\nobserve b").unwrap();
    if let Value::Bond(b) = named(&r2, "b") {
        assert!(!b.exists);
    } else {
        panic!();
    }
}

#[test]
fn track_converges() {
    let src = "floor 1.0\nO := cut 8\nH := cut 1\nW := close O(H,H)\n\
               path := track O in W with reps mass,charge until converge yield amalgamation\nobserve path";
    let r = evaluate(src).unwrap();
    if let Value::Path(p) = named(&r, "path") {
        assert!(p.converged);
        assert_eq!(p.steps, 2);
        assert_eq!(p.amalgamation.len(), 2);
        assert_eq!(p.reps, vec!["mass".to_string(), "charge".to_string()]);
    } else {
        panic!("path not a Path");
    }
}

#[test]
fn no_zero_residue() {
    assert!(evaluate("floor 0\nC := cut 6").is_err());
    assert!(evaluate("floor -1.0\nC := cut 6").is_err());
    assert!(evaluate("floor 1.0\nx := 5.0#0").is_err());
    assert!(evaluate("floor 1.0\nC := cut 6").is_ok());
}

#[test]
fn cut_monotone() {
    let r = evaluate("floor 1.0\nA := cut 6\nB := cut 8\nW := close B(A,A,A,A)").unwrap();
    assert!(r.cut_count >= 2);
}

#[test]
fn shell_capacity_2n2() {
    assert_eq!(shell_capacity(1), 2);
    assert_eq!(shell_capacity(2), 8);
    assert_eq!(shell_capacity(3), 18);
    assert_eq!(shell_capacity(4), 32);
}

#[test]
fn nacl_one_to_one() {
    let r = evaluate("floor 1.0\nNa := cut 11\nCl := cut 17\nS := close Na(Cl)").unwrap();
    if let Value::Compound(s) = named(&r, "S") {
        assert_eq!(s.formula, (1, 1));
    } else {
        panic!();
    }
}

#[test]
fn examples_run() {
    for f in ["carbon", "water", "track", "salt"] {
        let src = std::fs::read_to_string(format!("examples/{}.hj", f)).unwrap();
        let r = evaluate(&src).unwrap();
        assert!(r.ok, "example {} aborted", f);
    }
}
