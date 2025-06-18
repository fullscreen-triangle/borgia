//! Fuzzy logic components for molecular reasoning and linguistic variables.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::ProbabilisticValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fuzzy set with membership function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySet {
    pub name: String,
    pub membership_function: MembershipFunction,
    pub universe_min: f64,
    pub universe_max: f64,
}

/// Linguistic variable composed of multiple fuzzy sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticVariable {
    pub name: String,
    pub universe_min: f64,
    pub universe_max: f64,
    pub fuzzy_sets: Vec<FuzzySet>,
}

/// Fuzzy rule for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRule {
    pub antecedent: FuzzyCondition,
    pub consequent: FuzzyConsequent,
    pub weight: f64,
    pub confidence: f64,
}

/// Fuzzy condition (antecedent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyCondition {
    pub variable: String,
    pub fuzzy_set: String,
    pub operator: FuzzyOperator,
    pub next_condition: Option<Box<FuzzyCondition>>,
}

/// Fuzzy consequent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConsequent {
    pub variable: String,
    pub fuzzy_set: String,
    pub certainty_factor: f64,
}

/// Fuzzy operators for combining conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FuzzyOperator {
    And,
    Or,
    Not,
    None, // For single conditions
}

/// Types of membership functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipFunction {
    Triangular { a: f64, b: f64, c: f64 },
    Trapezoidal { a: f64, b: f64, c: f64, d: f64 },
    Gaussian { mean: f64, std_dev: f64 },
    Sigmoid { a: f64, c: f64 },
    Bell { a: f64, b: f64, c: f64 },
}

/// Fuzzy inference engine
#[derive(Debug, Clone)]
pub struct FuzzyInferenceEngine {
    pub linguistic_variables: HashMap<String, LinguisticVariable>,
    pub rules: Vec<FuzzyRule>,
    pub defuzzification_method: DefuzzificationMethod,
}

/// Methods for defuzzification
#[derive(Debug, Clone)]
pub enum DefuzzificationMethod {
    Centroid,
    Bisector,
    MeanOfMaxima,
    SmallestOfMaxima,
    LargestOfMaxima,
}

/// Result of fuzzy inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyInferenceResult {
    pub crisp_value: f64,
    pub membership_degrees: HashMap<String, f64>,
    pub activated_rules: Vec<usize>,
    pub confidence: f64,
}

impl MembershipFunction {
    /// Evaluate membership function at given value
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
            MembershipFunction::Trapezoidal { a, b, c, d } => {
                if x <= *a || x >= *d {
                    0.0
                } else if x <= *b {
                    (x - a) / (b - a)
                } else if x <= *c {
                    1.0
                } else {
                    (d - x) / (d - c)
                }
            }
            MembershipFunction::Gaussian { mean, std_dev } => {
                let exp_arg = -0.5 * ((x - mean) / std_dev).powi(2);
                exp_arg.exp()
            }
            MembershipFunction::Sigmoid { a, c } => {
                1.0 / (1.0 + (-a * (x - c)).exp())
            }
            MembershipFunction::Bell { a, b, c } => {
                1.0 / (1.0 + ((x - c) / a).abs().powf(2.0 * b))
            }
        }
    }

    /// Get support of the membership function (where membership > 0)
    pub fn support(&self) -> (f64, f64) {
        match self {
            MembershipFunction::Triangular { a, c, .. } => (*a, *c),
            MembershipFunction::Trapezoidal { a, d, .. } => (*a, *d),
            MembershipFunction::Gaussian { mean, std_dev } => {
                (mean - 4.0 * std_dev, mean + 4.0 * std_dev)
            }
            MembershipFunction::Sigmoid { .. } => (f64::NEG_INFINITY, f64::INFINITY),
            MembershipFunction::Bell { a, c, .. } => (c - 4.0 * a, c + 4.0 * a),
        }
    }

    /// Get core of the membership function (where membership = 1)
    pub fn core(&self) -> (f64, f64) {
        match self {
            MembershipFunction::Triangular { b, .. } => (*b, *b),
            MembershipFunction::Trapezoidal { b, c, .. } => (*b, *c),
            MembershipFunction::Gaussian { mean, .. } => (*mean, *mean),
            MembershipFunction::Sigmoid { .. } => (f64::NEG_INFINITY, f64::INFINITY),
            MembershipFunction::Bell { c, .. } => (*c, *c),
        }
    }
}

impl FuzzySet {
    /// Create a new fuzzy set
    pub fn new(
        name: String,
        membership_function: MembershipFunction,
        universe_min: f64,
        universe_max: f64,
    ) -> Self {
        Self {
            name,
            membership_function,
            universe_min,
            universe_max,
        }
    }

    /// Evaluate membership degree for a given value
    pub fn membership(&self, value: f64) -> f64 {
        if value < self.universe_min || value > self.universe_max {
            0.0
        } else {
            self.membership_function.evaluate(value)
        }
    }

    /// Find the value that gives maximum membership
    pub fn centroid(&self, resolution: usize) -> f64 {
        let step = (self.universe_max - self.universe_min) / resolution as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..=resolution {
            let x = self.universe_min + i as f64 * step;
            let membership = self.membership(x);
            numerator += x * membership;
            denominator += membership;
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            (self.universe_min + self.universe_max) / 2.0
        }
    }

    /// Alpha-cut of the fuzzy set
    pub fn alpha_cut(&self, alpha: f64, resolution: usize) -> Vec<(f64, f64)> {
        let mut intervals = Vec::new();
        let step = (self.universe_max - self.universe_min) / resolution as f64;
        let mut in_cut = false;
        let mut start = 0.0;

        for i in 0..=resolution {
            let x = self.universe_min + i as f64 * step;
            let membership = self.membership(x);

            if membership >= alpha && !in_cut {
                start = x;
                in_cut = true;
            } else if membership < alpha && in_cut {
                intervals.push((start, x));
                in_cut = false;
            }
        }

        if in_cut {
            intervals.push((start, self.universe_max));
        }

        intervals
    }
}

impl LinguisticVariable {
    /// Create a new linguistic variable
    pub fn new(name: String, universe_min: f64, universe_max: f64) -> Self {
        Self {
            name,
            universe_min,
            universe_max,
            fuzzy_sets: Vec::new(),
        }
    }

    /// Add a fuzzy set to the linguistic variable
    pub fn add_fuzzy_set(&mut self, fuzzy_set: FuzzySet) {
        self.fuzzy_sets.push(fuzzy_set);
    }

    /// Get membership degrees for all fuzzy sets
    pub fn fuzzify(&self, value: f64) -> HashMap<String, f64> {
        let mut memberships = HashMap::new();
        
        for fuzzy_set in &self.fuzzy_sets {
            let membership = fuzzy_set.membership(value);
            memberships.insert(fuzzy_set.name.clone(), membership);
        }

        memberships
    }

    /// Find the fuzzy set with maximum membership for a value
    pub fn max_membership_set(&self, value: f64) -> Option<(String, f64)> {
        let memberships = self.fuzzify(value);
        
        memberships.into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Create standard similarity linguistic variable
    pub fn create_similarity_variable() -> Self {
        let mut var = LinguisticVariable::new("similarity".to_string(), 0.0, 1.0);

        // Very Low: triangular(0, 0, 0.2)
        var.add_fuzzy_set(FuzzySet::new(
            "very_low".to_string(),
            MembershipFunction::Triangular { a: 0.0, b: 0.0, c: 0.2 },
            0.0, 1.0,
        ));

        // Low: triangular(0.1, 0.25, 0.4)
        var.add_fuzzy_set(FuzzySet::new(
            "low".to_string(),
            MembershipFunction::Triangular { a: 0.1, b: 0.25, c: 0.4 },
            0.0, 1.0,
        ));

        // Medium: triangular(0.3, 0.5, 0.7)
        var.add_fuzzy_set(FuzzySet::new(
            "medium".to_string(),
            MembershipFunction::Triangular { a: 0.3, b: 0.5, c: 0.7 },
            0.0, 1.0,
        ));

        // High: triangular(0.6, 0.75, 0.9)
        var.add_fuzzy_set(FuzzySet::new(
            "high".to_string(),
            MembershipFunction::Triangular { a: 0.6, b: 0.75, c: 0.9 },
            0.0, 1.0,
        ));

        // Very High: triangular(0.8, 1.0, 1.0)
        var.add_fuzzy_set(FuzzySet::new(
            "very_high".to_string(),
            MembershipFunction::Triangular { a: 0.8, b: 1.0, c: 1.0 },
            0.0, 1.0,
        ));

        var
    }

    /// Create aromaticity linguistic variable
    pub fn create_aromaticity_variable() -> Self {
        let mut var = LinguisticVariable::new("aromaticity".to_string(), 0.0, 1.0);

        // Non-aromatic: triangular(0, 0, 0.3)
        var.add_fuzzy_set(FuzzySet::new(
            "non_aromatic".to_string(),
            MembershipFunction::Triangular { a: 0.0, b: 0.0, c: 0.3 },
            0.0, 1.0,
        ));

        // Weakly aromatic: triangular(0.2, 0.4, 0.6)
        var.add_fuzzy_set(FuzzySet::new(
            "weakly_aromatic".to_string(),
            MembershipFunction::Triangular { a: 0.2, b: 0.4, c: 0.6 },
            0.0, 1.0,
        ));

        // Aromatic: triangular(0.5, 0.7, 0.9)
        var.add_fuzzy_set(FuzzySet::new(
            "aromatic".to_string(),
            MembershipFunction::Triangular { a: 0.5, b: 0.7, c: 0.9 },
            0.0, 1.0,
        ));

        // Highly aromatic: triangular(0.8, 1.0, 1.0)
        var.add_fuzzy_set(FuzzySet::new(
            "highly_aromatic".to_string(),
            MembershipFunction::Triangular { a: 0.8, b: 1.0, c: 1.0 },
            0.0, 1.0,
        ));

        var
    }
}

impl FuzzyInferenceEngine {
    /// Create a new fuzzy inference engine
    pub fn new(defuzzification_method: DefuzzificationMethod) -> Self {
        Self {
            linguistic_variables: HashMap::new(),
            rules: Vec::new(),
            defuzzification_method,
        }
    }

    /// Add a linguistic variable
    pub fn add_variable(&mut self, variable: LinguisticVariable) {
        self.linguistic_variables.insert(variable.name.clone(), variable);
    }

    /// Add a fuzzy rule
    pub fn add_rule(&mut self, rule: FuzzyRule) {
        self.rules.push(rule);
    }

    /// Perform fuzzy inference
    pub fn infer(&self, inputs: &HashMap<String, f64>) -> Result<FuzzyInferenceResult> {
        let mut activated_rules = Vec::new();
        let mut output_memberships: HashMap<String, f64> = HashMap::new();

        // Evaluate each rule
        for (rule_index, rule) in self.rules.iter().enumerate() {
            let activation_strength = self.evaluate_condition(&rule.antecedent, inputs)?;
            
            if activation_strength > 0.0 {
                activated_rules.push(rule_index);
                
                // Apply rule weight and confidence
                let weighted_strength = activation_strength * rule.weight * rule.confidence;
                
                // Update output membership
                let output_key = format!("{}_{}", rule.consequent.variable, rule.consequent.fuzzy_set);
                let current_membership = output_memberships.get(&output_key).unwrap_or(&0.0);
                output_memberships.insert(
                    output_key,
                    current_membership.max(weighted_strength * rule.consequent.certainty_factor)
                );
            }
        }

        // Defuzzify to get crisp output
        let crisp_value = self.defuzzify(&output_memberships)?;
        
        // Calculate overall confidence
        let confidence = if activated_rules.is_empty() {
            0.0
        } else {
            activated_rules.iter()
                .map(|&i| self.rules[i].confidence)
                .sum::<f64>() / activated_rules.len() as f64
        };

        Ok(FuzzyInferenceResult {
            crisp_value,
            membership_degrees: output_memberships,
            activated_rules,
            confidence,
        })
    }

    /// Evaluate a fuzzy condition
    fn evaluate_condition(
        &self,
        condition: &FuzzyCondition,
        inputs: &HashMap<String, f64>,
    ) -> Result<f64> {
        let variable = self.linguistic_variables.get(&condition.variable)
            .ok_or_else(|| BorgiaError::fuzzy_logic(
                format!("Unknown variable: {}", condition.variable)
            ))?;

        let input_value = inputs.get(&condition.variable)
            .ok_or_else(|| BorgiaError::fuzzy_logic(
                format!("Missing input for variable: {}", condition.variable)
            ))?;

        let fuzzy_set = variable.fuzzy_sets.iter()
            .find(|fs| fs.name == condition.fuzzy_set)
            .ok_or_else(|| BorgiaError::fuzzy_logic(
                format!("Unknown fuzzy set: {}", condition.fuzzy_set)
            ))?;

        let membership = fuzzy_set.membership(*input_value);

        // Handle compound conditions
        if let Some(ref next_condition) = condition.next_condition {
            let next_membership = self.evaluate_condition(next_condition, inputs)?;
            
            match condition.operator {
                FuzzyOperator::And => Ok(membership.min(next_membership)),
                FuzzyOperator::Or => Ok(membership.max(next_membership)),
                FuzzyOperator::Not => Ok(1.0 - membership),
                FuzzyOperator::None => Ok(membership),
            }
        } else {
            match condition.operator {
                FuzzyOperator::Not => Ok(1.0 - membership),
                _ => Ok(membership),
            }
        }
    }

    /// Defuzzify output memberships to crisp value
    fn defuzzify(&self, memberships: &HashMap<String, f64>) -> Result<f64> {
        if memberships.is_empty() {
            return Ok(0.0);
        }

        match self.defuzzification_method {
            DefuzzificationMethod::Centroid => {
                self.centroid_defuzzification(memberships)
            }
            DefuzzificationMethod::MeanOfMaxima => {
                self.mean_of_maxima_defuzzification(memberships)
            }
            _ => {
                // Simplified: return weighted average
                let total_weight: f64 = memberships.values().sum();
                if total_weight > 0.0 {
                    let weighted_sum: f64 = memberships.iter()
                        .map(|(_, &membership)| membership * 0.5) // Simplified
                        .sum();
                    Ok(weighted_sum / total_weight)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Centroid defuzzification method
    fn centroid_defuzzification(&self, memberships: &HashMap<String, f64>) -> Result<f64> {
        // Simplified implementation - would need actual fuzzy set definitions
        let total_weight: f64 = memberships.values().sum();
        if total_weight > 0.0 {
            Ok(0.5) // Placeholder
        } else {
            Ok(0.0)
        }
    }

    /// Mean of maxima defuzzification method
    fn mean_of_maxima_defuzzification(&self, memberships: &HashMap<String, f64>) -> Result<f64> {
        let max_membership = memberships.values()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(&0.0);

        let max_keys: Vec<&String> = memberships.iter()
            .filter(|(_, &v)| (v - max_membership).abs() < 1e-6)
            .map(|(k, _)| k)
            .collect();

        // Return average of maximum membership positions (simplified)
        Ok(0.5)
    }
}

/// Create a molecular similarity fuzzy system
pub fn create_molecular_similarity_system() -> FuzzyInferenceEngine {
    let mut engine = FuzzyInferenceEngine::new(DefuzzificationMethod::Centroid);

    // Add linguistic variables
    engine.add_variable(LinguisticVariable::create_similarity_variable());
    engine.add_variable(LinguisticVariable::create_aromaticity_variable());

    // Add rules (simplified examples)
    engine.add_rule(FuzzyRule {
        antecedent: FuzzyCondition {
            variable: "similarity".to_string(),
            fuzzy_set: "high".to_string(),
            operator: FuzzyOperator::None,
            next_condition: None,
        },
        consequent: FuzzyConsequent {
            variable: "confidence".to_string(),
            fuzzy_set: "high".to_string(),
            certainty_factor: 0.9,
        },
        weight: 1.0,
        confidence: 0.95,
    });

    engine
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_membership() {
        let mf = MembershipFunction::Triangular { a: 0.0, b: 0.5, c: 1.0 };
        assert_eq!(mf.evaluate(0.0), 0.0);
        assert_eq!(mf.evaluate(0.5), 1.0);
        assert_eq!(mf.evaluate(1.0), 0.0);
        assert_eq!(mf.evaluate(0.25), 0.5);
    }

    #[test]
    fn test_gaussian_membership() {
        let mf = MembershipFunction::Gaussian { mean: 0.5, std_dev: 0.2 };
        assert_eq!(mf.evaluate(0.5), 1.0);
        assert!(mf.evaluate(0.3) < 1.0);
        assert!(mf.evaluate(0.7) < 1.0);
    }

    #[test]
    fn test_fuzzy_set() {
        let fs = FuzzySet::new(
            "medium".to_string(),
            MembershipFunction::Triangular { a: 0.3, b: 0.5, c: 0.7 },
            0.0, 1.0,
        );
        
        assert_eq!(fs.membership(0.5), 1.0);
        assert_eq!(fs.membership(0.4), 0.5);
        assert_eq!(fs.membership(0.6), 0.5);
    }

    #[test]
    fn test_linguistic_variable() {
        let var = LinguisticVariable::create_similarity_variable();
        let memberships = var.fuzzify(0.8);
        
        assert!(memberships.contains_key("very_high"));
        assert!(memberships.contains_key("high"));
        assert!(memberships["very_high"] > 0.0);
    }

    #[test]
    fn test_fuzzy_inference_engine() {
        let engine = create_molecular_similarity_system();
        assert!(!engine.linguistic_variables.is_empty());
        assert!(!engine.rules.is_empty());
    }
} 