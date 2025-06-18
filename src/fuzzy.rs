//! Fuzzy logic components for Borgia.

use crate::error::{BorgiaError, Result};
use serde::{Deserialize, Serialize};

/// Fuzzy set representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    pub name: String,
    pub membership_function: MembershipFunction,
}

/// Linguistic variable for fuzzy reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticVariable {
    pub name: String,
    pub fuzzy_sets: Vec<FuzzySet>,
}

/// Fuzzy rule for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub antecedent: String,
    pub consequent: String,
    pub weight: f64,
}

/// Membership function types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipFunction {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, std_dev: f64 },
}

impl MembershipFunction {
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            MembershipFunction::Triangular { a, b, c } => {
                if x <= *a || x >= *c {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else {
                    (c - x) / (c - b)
                }
            }
            MembershipFunction::Gaussian { mean, std_dev } => {
                let exp_arg = -0.5 * ((x - mean) / std_dev).powi(2);
                exp_arg.exp()
            }
            _ => 0.5, // Simplified for other types
        }
    }
} 