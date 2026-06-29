//! Honjo Masamune — standard library (§7). Bound to the framework's validated
//! operations; kept numerically identical to the TypeScript target so the
//! two-target equivalence theorem (§8) holds.

#[derive(Debug, Clone)]
pub enum Value {
    Scalar { value: f64, floor: f64 },
    Atom(AtomVal),
    Bond(BondVal),
    Compound(CompoundVal),
    Path(PathVal),
}

#[derive(Debug, Clone)]
pub struct AtomVal {
    pub z: i64,
    pub symbol: String,
    pub config: String,
    pub term: String,
    pub qv: i64,
    pub cap_v: i64,
    pub vacancy: i64,
    pub valence: i64,
    pub floor: f64,
    pub residue: f64,
}

#[derive(Debug, Clone)]
pub struct BondVal {
    pub a: String,
    pub b: String,
    pub delta: f64,
    pub shared: i64,
    pub exists: bool,
    pub floor: f64,
    pub residue: f64,
}

#[derive(Debug, Clone)]
pub struct CompoundVal {
    pub central: String,
    pub ligand: String,
    pub formula: (i64, i64),
    pub ligands: i64,
    pub geometry: String,
    pub angle_deg: Option<f64>,
    pub valence_closed: bool,
    pub floor: f64,
    pub residue: f64,
}

#[derive(Debug, Clone)]
pub struct PathVal {
    pub item: String,
    pub steps: i64,
    pub converged: bool,
    pub amalgamation: Vec<String>,
    pub reps: Vec<String>,
    pub floor: f64,
    pub residue: f64,
}

struct ElemRec {
    sym: &'static str,
    z: i64,
    qv: i64,
    cap_v: i64,
    config: &'static str,
    term: &'static str,
}

const ELEMENTS: &[ElemRec] = &[
    ElemRec { sym: "H",  z: 1,  qv: 1, cap_v: 2, config: "1s1",          term: "2S1/2" },
    ElemRec { sym: "He", z: 2,  qv: 2, cap_v: 2, config: "1s2",          term: "1S0" },
    ElemRec { sym: "Li", z: 3,  qv: 1, cap_v: 8, config: "[He] 2s1",     term: "2S1/2" },
    ElemRec { sym: "Be", z: 4,  qv: 2, cap_v: 8, config: "[He] 2s2",     term: "1S0" },
    ElemRec { sym: "B",  z: 5,  qv: 3, cap_v: 8, config: "[He] 2s2 2p1", term: "2P1/2" },
    ElemRec { sym: "C",  z: 6,  qv: 4, cap_v: 8, config: "[He] 2s2 2p2", term: "3P0" },
    ElemRec { sym: "N",  z: 7,  qv: 5, cap_v: 8, config: "[He] 2s2 2p3", term: "4S3/2" },
    ElemRec { sym: "O",  z: 8,  qv: 6, cap_v: 8, config: "[He] 2s2 2p4", term: "3P2" },
    ElemRec { sym: "F",  z: 9,  qv: 7, cap_v: 8, config: "[He] 2s2 2p5", term: "2P3/2" },
    ElemRec { sym: "Ne", z: 10, qv: 8, cap_v: 8, config: "[He] 2s2 2p6", term: "1S0" },
    ElemRec { sym: "Na", z: 11, qv: 1, cap_v: 8, config: "[Ne] 3s1",     term: "2S1/2" },
    ElemRec { sym: "Mg", z: 12, qv: 2, cap_v: 8, config: "[Ne] 3s2",     term: "1S0" },
    ElemRec { sym: "Al", z: 13, qv: 3, cap_v: 8, config: "[Ne] 3s2 3p1", term: "2P1/2" },
    ElemRec { sym: "Si", z: 14, qv: 4, cap_v: 8, config: "[Ne] 3s2 3p2", term: "3P0" },
    ElemRec { sym: "P",  z: 15, qv: 5, cap_v: 8, config: "[Ne] 3s2 3p3", term: "4S3/2" },
    ElemRec { sym: "S",  z: 16, qv: 6, cap_v: 8, config: "[Ne] 3s2 3p4", term: "3P2" },
    ElemRec { sym: "Cl", z: 17, qv: 7, cap_v: 8, config: "[Ne] 3s2 3p5", term: "2P3/2" },
    ElemRec { sym: "Ar", z: 18, qv: 8, cap_v: 8, config: "[Ne] 3s2 3p6", term: "1S0" },
];

/// Shell capacity C(n) = 2 n^2.
pub fn shell_capacity(n: i64) -> i64 {
    let mut c = 0;
    for l in 0..n {
        c += 2 * (2 * l + 1);
    }
    c
}

fn thickness(nu: i64, floor: f64, kappa: f64) -> f64 {
    floor + kappa * nu as f64
}

pub fn individuate(z: i64, floor: f64) -> Result<AtomVal, String> {
    if z < 1 {
        return Err(format!("cut: atomic number must be a positive integer (got {})", z));
    }
    let e = ELEMENTS
        .iter()
        .find(|e| e.z == z)
        .ok_or_else(|| format!("cut: element Z={} not in the light-element table (1..18 supported)", z))?;
    let vacancy = e.cap_v - e.qv;
    let valence = vacancy.min(e.cap_v - vacancy);
    let residue = thickness(vacancy, floor, 1.0);
    Ok(AtomVal {
        z, symbol: e.sym.into(), config: e.config.into(), term: e.term.into(),
        qv: e.qv, cap_v: e.cap_v, vacancy, valence, floor, residue,
    })
}

pub fn bond(a: &AtomVal, b: &AtomVal, floor: f64) -> BondVal {
    let kappa = 1.0;
    let shared = a.vacancy.min(b.vacancy);
    let sep = thickness(a.vacancy, floor, kappa) + thickness(b.vacancy, floor, kappa);
    let joined = thickness((a.vacancy - shared).max(0), floor, kappa)
        + thickness((b.vacancy - shared).max(0), floor, kappa);
    let delta = sep - joined;
    let exists = delta > 1e-12;
    BondVal {
        a: a.symbol.clone(), b: b.symbol.clone(), delta, shared, exists,
        floor, residue: delta.max(floor),
    }
}

pub fn close(central: &AtomVal, ligands: &[AtomVal], floor: f64) -> Result<CompoundVal, String> {
    if ligands.is_empty() {
        return Err("close: needs at least one ligand".into());
    }
    let lig = &ligands[0];

    if central.symbol == lig.symbol {
        return Ok(CompoundVal {
            central: central.symbol.clone(), ligand: lig.symbol.clone(),
            formula: (2, 0), ligands: 1, geometry: "linear".into(),
            angle_deg: Some(180.0), valence_closed: true,
            floor, residue: thickness(central.vacancy, floor, 1.0),
        });
    }

    let v_c = central.valence.max(1);
    let v_l = lig.valence.max(1);
    let n_lig = ((v_c as f64 / v_l as f64).round() as i64).max(1);

    let bonded = n_lig;
    let lone_pairs = if central.cap_v == 8 {
        ((central.qv - n_lig) / 2).max(0)
    } else {
        0
    };
    let k = bonded + lone_pairs;
    let ang_tet = round2((-1.0f64 / 3.0).acos().to_degrees());

    let (geometry, angle_deg): (&str, Option<f64>) = if k == 1 {
        ("terminal", None)
    } else if k == 2 {
        if lone_pairs == 0 { ("linear", Some(180.0)) } else { ("bent", Some(ang_tet)) }
    } else if k == 3 {
        if lone_pairs == 0 { ("trigonal", Some(120.0)) } else { ("bent", Some(117.0)) }
    } else {
        if lone_pairs == 0 {
            ("tetrahedral", Some(ang_tet))
        } else if lone_pairs == 1 {
            ("pyramidal", Some(107.0))
        } else {
            ("bent", Some(104.5))
        }
    };

    Ok(CompoundVal {
        central: central.symbol.clone(), ligand: lig.symbol.clone(),
        formula: (1, n_lig), ligands: n_lig, geometry: geometry.into(), angle_deg,
        valence_closed: true, floor, residue: thickness(0, floor, 1.0) + n_lig as f64 * floor,
    })
}

pub enum Admit {
    Converge,
    Diverge,
    Cond(bool),
}

pub fn propagate(
    item: &AtomVal,
    process: &Value,
    reps: &[String],
    admit: Admit,
    floor: f64,
) -> Result<PathVal, String> {
    let mut contacts = Vec::new();
    let mut steps = 0i64;
    let mut residue = 0.0f64;

    match process {
        Value::Compound(c) => {
            for i in 0..c.ligands {
                contacts.push(format!("{}~{}#{}", c.central, c.ligand, i + 1));
                steps += 1;
                residue += floor;
            }
        }
        Value::Path(p) => {
            steps = p.steps;
            residue = p.residue;
            contacts.extend(p.amalgamation.iter().cloned());
        }
        _ => return Err("track process must be Compound or Path".into()),
    }

    let converged = match admit {
        Admit::Converge => residue >= floor && steps > 0,
        Admit::Diverge => !(residue >= floor && steps > 0),
        Admit::Cond(b) => b,
    };

    let reps_v: Vec<String> = if reps.is_empty() {
        vec!["mass".into()]
    } else {
        reps.to_vec()
    };

    Ok(PathVal {
        item: item.symbol.clone(),
        steps,
        converged,
        amalgamation: if converged { contacts } else { vec![] },
        reps: reps_v,
        floor,
        residue,
    })
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
