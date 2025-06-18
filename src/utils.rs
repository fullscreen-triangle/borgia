//! Utility functions for Borgia.

use crate::error::{BorgiaError, Result};

/// Utility functions for molecular operations
pub mod molecular_utils {
    use super::*;

    /// Validate SMILES string format
    pub fn validate_smiles(smiles: &str) -> Result<()> {
        if smiles.is_empty() {
            return Err(BorgiaError::invalid_smiles(smiles, "Empty SMILES string"));
        }

        // Basic validation - check for invalid characters
        if smiles.contains("X") || smiles.contains("*") {
            return Err(BorgiaError::invalid_smiles(smiles, "Contains invalid atoms"));
        }

        Ok(())
    }

    /// Calculate molecular weight from SMILES (simplified)
    pub fn estimate_molecular_weight(smiles: &str) -> f64 {
        let mut weight = 0.0;
        for ch in smiles.chars() {
            weight += match ch {
                'C' => 12.01,
                'N' => 14.01,
                'O' => 16.00,
                'S' => 32.07,
                'P' => 30.97,
                'F' => 19.00,
                _ => 0.0,
            };
        }
        weight
    }
}

/// Utility functions for probabilistic operations
pub mod probabilistic_utils {
    use super::*;
    use crate::probabilistic::ProbabilisticValue;

    /// Combine multiple probabilistic values
    pub fn combine_probabilistic_values(values: &[ProbabilisticValue]) -> Result<ProbabilisticValue> {
        if values.is_empty() {
            return Err(BorgiaError::probabilistic(
                "combination",
                "Cannot combine empty list of values",
            ));
        }

        let mean = values.iter().map(|v| v.mean).sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| v.std_dev.powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let confidence = values.iter().map(|v| v.confidence_level).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.95);

        Ok(ProbabilisticValue::new_normal(mean, std_dev, confidence))
    }
}

/// Utility functions for similarity calculations
pub mod similarity_utils {
    use super::*;

    /// Normalize similarity score to [0, 1] range
    pub fn normalize_similarity(score: f64) -> f64 {
        score.max(0.0).min(1.0)
    }

    /// Convert similarity to percentage
    pub fn similarity_to_percentage(score: f64) -> f64 {
        normalize_similarity(score) * 100.0
    }
} 