//! Probabilistic computing foundation for Borgia.
//!
//! This module provides the core probabilistic data structures and algorithms
//! that enable uncertainty quantification throughout the cheminformatics pipeline.

use crate::error::{BorgiaError, Result};
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, Gamma};
use std::collections::HashMap;
use std::fmt;

/// A probabilistic value with uncertainty bounds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbabilisticValue {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Distribution type
    pub distribution: DistributionType,
    /// Sample data if available
    pub samples: Option<Vec<f64>>,
}

/// Types of probability distributions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    Beta,
    Gamma,
    Uniform,
    Empirical,
}

/// Confidence interval representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
    pub mean: f64,
}

/// Uncertainty bounds for probabilistic calculations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyBounds {
    pub lower: f64,
    pub upper: f64,
    pub probability: f64,
}

/// Bayesian inference engine for probabilistic updates
#[derive(Debug, Clone)]
pub struct BayesianInference {
    /// Prior distributions
    priors: HashMap<String, ProbabilisticValue>,
    /// Evidence accumulation
    evidence: Vec<Evidence>,
    /// Posterior distributions
    posteriors: HashMap<String, ProbabilisticValue>,
}

/// Evidence for Bayesian updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub parameter: String,
    pub observation: f64,
    pub uncertainty: f64,
    pub weight: f64,
    pub source: String,
}

/// Probabilistic similarity distribution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityDistribution {
    pub very_low: f64,
    pub low: f64,
    pub medium: f64,
    pub high: f64,
    pub very_high: f64,
}

impl ProbabilisticValue {
    /// Create a new probabilistic value with normal distribution
    pub fn new_normal(mean: f64, std_dev: f64, confidence_level: f64) -> Self {
        Self {
            mean,
            std_dev,
            confidence_level,
            distribution: DistributionType::Normal,
            samples: None,
        }
    }

    /// Create from empirical samples
    pub fn from_samples(samples: Vec<f64>, confidence_level: f64) -> Result<Self> {
        if samples.is_empty() {
            return Err(BorgiaError::probabilistic(
                "sample_creation",
                "Cannot create probabilistic value from empty samples",
            ));
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64;
        let std_dev = variance.sqrt();

        Ok(Self {
            mean,
            std_dev,
            confidence_level,
            distribution: DistributionType::Empirical,
            samples: Some(samples),
        })
    }

    /// Get confidence interval
    pub fn confidence_interval(&self) -> ConfidenceInterval {
        let z_score = match self.confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        };

        let margin = z_score * self.std_dev;

        ConfidenceInterval {
            lower_bound: self.mean - margin,
            upper_bound: self.mean + margin,
            confidence_level: self.confidence_level,
            mean: self.mean,
        }
    }

    /// Sample from the distribution
    pub fn sample(&self, n_samples: usize) -> Result<Vec<f64>> {
        let mut rng = thread_rng();
        let samples = match self.distribution {
            DistributionType::Normal => {
                let normal = Normal::new(self.mean, self.std_dev)
                    .map_err(|e| BorgiaError::probabilistic("normal_sampling", e.to_string()))?;
                (0..n_samples).map(|_| normal.sample(&mut rng)).collect()
            }
            DistributionType::Uniform => {
                let margin = self.std_dev * 1.732; // sqrt(3) for uniform distribution
                let uniform = Uniform::new(self.mean - margin, self.mean + margin);
                (0..n_samples).map(|_| uniform.sample(&mut rng)).collect()
            }
            DistributionType::Empirical => {
                if let Some(ref original_samples) = self.samples {
                    (0..n_samples)
                        .map(|_| original_samples[rng.gen_range(0..original_samples.len())])
                        .collect()
                } else {
                    return Err(BorgiaError::probabilistic(
                        "empirical_sampling",
                        "No samples available for empirical distribution",
                    ));
                }
            }
            _ => {
                return Err(BorgiaError::probabilistic(
                    "sampling",
                    format!("Sampling not implemented for {:?}", self.distribution),
                ))
            }
        };

        Ok(samples)
    }

    /// Propagate uncertainty through a function
    pub fn propagate_uncertainty<F>(&self, f: F, n_samples: usize) -> Result<ProbabilisticValue>
    where
        F: Fn(f64) -> f64,
    {
        let input_samples = self.sample(n_samples)?;
        let output_samples: Vec<f64> = input_samples.iter().map(|&x| f(x)).collect();

        ProbabilisticValue::from_samples(output_samples, self.confidence_level)
    }

    /// Combine with another probabilistic value
    pub fn combine_with<F>(
        &self,
        other: &ProbabilisticValue,
        f: F,
        n_samples: usize,
    ) -> Result<ProbabilisticValue>
    where
        F: Fn(f64, f64) -> f64,
    {
        let samples1 = self.sample(n_samples)?;
        let samples2 = other.sample(n_samples)?;

        let combined_samples: Vec<f64> = samples1
            .iter()
            .zip(samples2.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();

        ProbabilisticValue::from_samples(combined_samples, self.confidence_level.min(other.confidence_level))
    }

    /// Calculate probability that this value is greater than threshold
    pub fn probability_greater_than(&self, threshold: f64) -> f64 {
        if let Some(ref samples) = self.samples {
            samples.iter().filter(|&&x| x > threshold).count() as f64 / samples.len() as f64
        } else {
            // Use normal approximation
            let z = (threshold - self.mean) / self.std_dev;
            1.0 - normal_cdf(z)
        }
    }
}

impl BayesianInference {
    /// Create a new Bayesian inference engine
    pub fn new() -> Self {
        Self {
            priors: HashMap::new(),
            evidence: Vec::new(),
            posteriors: HashMap::new(),
        }
    }

    /// Set prior distribution for a parameter
    pub fn set_prior(&mut self, parameter: String, prior: ProbabilisticValue) {
        self.priors.insert(parameter, prior);
    }

    /// Add evidence
    pub fn add_evidence(&mut self, evidence: Evidence) {
        self.evidence.push(evidence);
    }

    /// Update posterior distributions using evidence
    pub fn update_posteriors(&mut self) -> Result<()> {
        for (param, prior) in &self.priors {
            let relevant_evidence: Vec<&Evidence> = self
                .evidence
                .iter()
                .filter(|e| e.parameter == *param)
                .collect();

            if relevant_evidence.is_empty() {
                // No evidence, posterior equals prior
                self.posteriors.insert(param.clone(), prior.clone());
                continue;
            }

            // Simple Bayesian update using weighted average
            let total_weight: f64 = relevant_evidence.iter().map(|e| e.weight).sum();
            let weighted_mean: f64 = relevant_evidence
                .iter()
                .map(|e| e.observation * e.weight)
                .sum::<f64>()
                / total_weight;

            // Combine prior and evidence
            let prior_weight = 1.0 / (prior.std_dev.powi(2));
            let evidence_weight = total_weight;

            let posterior_mean = (prior.mean * prior_weight + weighted_mean * evidence_weight)
                / (prior_weight + evidence_weight);

            let posterior_variance = 1.0 / (prior_weight + evidence_weight);
            let posterior_std_dev = posterior_variance.sqrt();

            let posterior = ProbabilisticValue::new_normal(
                posterior_mean,
                posterior_std_dev,
                prior.confidence_level,
            );

            self.posteriors.insert(param.clone(), posterior);
        }

        Ok(())
    }

    /// Get posterior distribution
    pub fn get_posterior(&self, parameter: &str) -> Option<&ProbabilisticValue> {
        self.posteriors.get(parameter)
    }

    /// Get all posteriors
    pub fn get_all_posteriors(&self) -> &HashMap<String, ProbabilisticValue> {
        &self.posteriors
    }
}

impl SimilarityDistribution {
    /// Create a new similarity distribution
    pub fn new(very_low: f64, low: f64, medium: f64, high: f64, very_high: f64) -> Result<Self> {
        let total = very_low + low + medium + high + very_high;
        if (total - 1.0).abs() > 1e-6 {
            return Err(BorgiaError::probabilistic(
                "similarity_distribution",
                format!("Probabilities must sum to 1.0, got {}", total),
            ));
        }

        Ok(Self {
            very_low,
            low,
            medium,
            high,
            very_high,
        })
    }

    /// Get the most likely similarity category
    pub fn most_likely(&self) -> &'static str {
        let max_prob = self.very_low.max(self.low.max(self.medium.max(self.high.max(self.very_high))));
        
        if (self.very_high - max_prob).abs() < 1e-10 {
            "very_high"
        } else if (self.high - max_prob).abs() < 1e-10 {
            "high"
        } else if (self.medium - max_prob).abs() < 1e-10 {
            "medium"
        } else if (self.low - max_prob).abs() < 1e-10 {
            "low"
        } else {
            "very_low"
        }
    }

    /// Get expected similarity value (0-1 scale)
    pub fn expected_value(&self) -> f64 {
        0.1 * self.very_low + 0.3 * self.low + 0.5 * self.medium + 0.7 * self.high + 0.9 * self.very_high
    }

    /// Get uncertainty (entropy)
    pub fn uncertainty(&self) -> f64 {
        let probs = [self.very_low, self.low, self.medium, self.high, self.very_high];
        -probs.iter().map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 }).sum::<f64>()
    }
}

impl fmt::Display for SimilarityDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{very_low: {:.3}, low: {:.3}, medium: {:.3}, high: {:.3}, very_high: {:.3}}}",
            self.very_low, self.low, self.medium, self.high, self.very_high
        )
    }
}

impl Default for BayesianInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate normal CDF using error function approximation
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Approximate error function
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_value_creation() {
        let pv = ProbabilisticValue::new_normal(0.5, 0.1, 0.95);
        assert_eq!(pv.mean, 0.5);
        assert_eq!(pv.std_dev, 0.1);
        assert_eq!(pv.confidence_level, 0.95);
    }

    #[test]
    fn test_confidence_interval() {
        let pv = ProbabilisticValue::new_normal(0.5, 0.1, 0.95);
        let ci = pv.confidence_interval();
        assert!((ci.lower_bound - 0.304).abs() < 0.01);
        assert!((ci.upper_bound - 0.696).abs() < 0.01);
    }

    #[test]
    fn test_similarity_distribution() {
        let dist = SimilarityDistribution::new(0.1, 0.2, 0.4, 0.2, 0.1).unwrap();
        assert_eq!(dist.most_likely(), "medium");
        assert!((dist.expected_value() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bayesian_inference() {
        let mut bayes = BayesianInference::new();
        let prior = ProbabilisticValue::new_normal(0.5, 0.2, 0.95);
        bayes.set_prior("similarity".to_string(), prior);

        let evidence = Evidence {
            parameter: "similarity".to_string(),
            observation: 0.7,
            uncertainty: 0.1,
            weight: 1.0,
            source: "test".to_string(),
        };
        bayes.add_evidence(evidence);

        bayes.update_posteriors().unwrap();
        let posterior = bayes.get_posterior("similarity").unwrap();
        
        // Posterior should be between prior and evidence
        assert!(posterior.mean > 0.5);
        assert!(posterior.mean < 0.7);
    }

    #[test]
    fn test_uncertainty_propagation() {
        let pv = ProbabilisticValue::new_normal(2.0, 0.1, 0.95);
        let result = pv.propagate_uncertainty(|x| x * x, 1000).unwrap();
        
        // E[X^2] ≈ 4.01 for X ~ N(2, 0.1^2)
        assert!((result.mean - 4.01).abs() < 0.1);
    }
} 