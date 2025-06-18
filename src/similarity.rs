//! Probabilistic and fuzzy similarity calculations for molecular comparison.

use crate::error::{BorgiaError, Result};
use crate::molecular::ProbabilisticMolecule;
use crate::probabilistic::{ProbabilisticValue, SimilarityDistribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Similarity calculation engine
#[derive(Debug, Clone)]
pub struct SimilarityEngine {
    /// Algorithm weights for different similarity types
    pub algorithm_weights: HashMap<String, f64>,
    /// Context-specific parameters
    pub context_params: HashMap<String, f64>,
}

/// Probabilistic similarity result with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticSimilarity {
    /// Mean similarity value
    pub similarity: ProbabilisticValue,
    /// Similarity distribution across linguistic categories
    pub distribution: SimilarityDistribution,
    /// Algorithm used
    pub algorithm: String,
    /// Confidence in the result
    pub confidence: f64,
}

/// Fuzzy similarity with linguistic variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySimilarity {
    /// Linguistic similarity assessment
    pub linguistic_value: String,
    /// Membership degrees for each category
    pub memberships: HashMap<String, f64>,
    /// Overall fuzzy similarity score
    pub fuzzy_score: f64,
}

/// Similarity calculation algorithms
#[derive(Debug, Clone, Copy)]
pub enum SimilarityAlgorithm {
    Tanimoto,
    Dice,
    Cosine,
    Jaccard,
    ProbabilisticTanimoto,
    FuzzyTanimoto,
    WeightedSimilarity,
}

impl SimilarityEngine {
    /// Create a new similarity engine
    pub fn new() -> Self {
        let mut algorithm_weights = HashMap::new();
        algorithm_weights.insert("tanimoto".to_string(), 1.0);
        algorithm_weights.insert("pharmacophoric".to_string(), 0.8);
        algorithm_weights.insert("quantum".to_string(), 0.6);
        algorithm_weights.insert("conformational".to_string(), 0.4);

        Self {
            algorithm_weights,
            context_params: HashMap::new(),
        }
    }

    /// Calculate probabilistic similarity between two molecules
    pub fn calculate_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        algorithm: SimilarityAlgorithm,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        match algorithm {
            SimilarityAlgorithm::Tanimoto => self.tanimoto_similarity(mol1, mol2),
            SimilarityAlgorithm::ProbabilisticTanimoto => self.probabilistic_tanimoto(mol1, mol2),
            SimilarityAlgorithm::FuzzyTanimoto => self.fuzzy_tanimoto(mol1, mol2, context),
            SimilarityAlgorithm::WeightedSimilarity => self.weighted_similarity(mol1, mol2, context),
            _ => Err(BorgiaError::similarity(
                format!("{:?}", algorithm),
                "Algorithm not implemented yet",
            )),
        }
    }

    /// Standard Tanimoto similarity
    fn tanimoto_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<ProbabilisticSimilarity> {
        let similarity_value = mol1.fingerprint.tanimoto_similarity(&mol2.fingerprint);
        
        let similarity = ProbabilisticValue::new_normal(
            similarity_value,
            0.05, // Standard uncertainty for Tanimoto
            0.95,
        );

        let distribution = self.similarity_to_distribution(similarity_value)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "tanimoto".to_string(),
            confidence: 0.9,
        })
    }

    /// Probabilistic Tanimoto with uncertainty propagation
    fn probabilistic_tanimoto(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<ProbabilisticSimilarity> {
        // Sample from fingerprint uncertainties
        let n_samples = 1000;
        let mut similarities = Vec::new();

        for _ in 0..n_samples {
            // Add noise to fingerprints based on uncertainties
            let mut fp1_noisy = mol1.fingerprint.combined.clone();
            let mut fp2_noisy = mol2.fingerprint.combined.clone();

            // Add Gaussian noise proportional to uncertainties
            for i in 0..fp1_noisy.len() {
                let noise1 = rand::random::<f64>() * mol1.fingerprint.uncertainties[i];
                let noise2 = rand::random::<f64>() * mol2.fingerprint.uncertainties[i];
                fp1_noisy[i] += noise1;
                fp2_noisy[i] += noise2;
            }

            // Calculate similarity with noisy fingerprints
            let intersection = fp1_noisy.dot(&fp2_noisy);
            let union = fp1_noisy.norm_squared() + fp2_noisy.norm_squared() - intersection;
            
            let sim = if union > 0.0 { intersection / union } else { 0.0 };
            similarities.push(sim.max(0.0).min(1.0)); // Clamp to [0, 1]
        }

        let similarity = ProbabilisticValue::from_samples(similarities, 0.95)?;
        let distribution = self.similarity_to_distribution(similarity.mean)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "probabilistic_tanimoto".to_string(),
            confidence: 0.95,
        })
    }

    /// Fuzzy Tanimoto with context awareness
    fn fuzzy_tanimoto(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        // Calculate different types of similarities
        let topo_sim = self.calculate_topological_similarity(mol1, mol2)?;
        let pharma_sim = self.calculate_pharmacophoric_similarity(mol1, mol2)?;
        let quantum_sim = self.calculate_quantum_similarity(mol1, mol2)?;

        // Context-dependent weighting
        let weights = self.get_context_weights(context);
        
        let weighted_similarity = 
            topo_sim * weights.get("topological").unwrap_or(&1.0) +
            pharma_sim * weights.get("pharmacophoric").unwrap_or(&0.5) +
            quantum_sim * weights.get("quantum").unwrap_or(&0.3);

        let total_weight = weights.values().sum::<f64>();
        let normalized_similarity = weighted_similarity / total_weight;

        let similarity = ProbabilisticValue::new_normal(
            normalized_similarity,
            0.1, // Higher uncertainty for fuzzy calculations
            0.85,
        );

        let distribution = self.similarity_to_distribution(normalized_similarity)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "fuzzy_tanimoto".to_string(),
            confidence: 0.85,
        })
    }

    /// Weighted similarity with adaptive feature importance
    fn weighted_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        let similarity_value = mol1.fingerprint.weighted_similarity(&mol2.fingerprint);
        
        // Adjust similarity based on molecular properties
        let prop_similarity = self.calculate_property_similarity(mol1, mol2)?;
        let combined_similarity = (similarity_value + prop_similarity) / 2.0;

        let similarity = ProbabilisticValue::new_normal(
            combined_similarity,
            0.08,
            0.90,
        );

        let distribution = self.similarity_to_distribution(combined_similarity)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "weighted_similarity".to_string(),
            confidence: 0.90,
        })
    }

    /// Calculate topological similarity
    fn calculate_topological_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.topological.dot(&mol2.fingerprint.topological);
        let union = mol1.fingerprint.topological.norm_squared() + 
                   mol2.fingerprint.topological.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate pharmacophoric similarity
    fn calculate_pharmacophoric_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.pharmacophoric.dot(&mol2.fingerprint.pharmacophoric);
        let union = mol1.fingerprint.pharmacophoric.norm_squared() + 
                   mol2.fingerprint.pharmacophoric.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate quantum similarity
    fn calculate_quantum_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.quantum.dot(&mol2.fingerprint.quantum);
        let union = mol1.fingerprint.quantum.norm_squared() + 
                   mol2.fingerprint.quantum.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate property-based similarity
    fn calculate_property_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        // Compare molecular properties with uncertainty
        let mw_diff = (mol1.properties.molecular_weight.mean - mol2.properties.molecular_weight.mean).abs();
        let mw_uncertainty = mol1.properties.molecular_weight.std_dev + mol2.properties.molecular_weight.std_dev;
        let mw_sim = 1.0 - (mw_diff / (mw_diff + mw_uncertainty + 1.0));

        let logp_diff = (mol1.properties.logp.mean - mol2.properties.logp.mean).abs();
        let logp_uncertainty = mol1.properties.logp.std_dev + mol2.properties.logp.std_dev;
        let logp_sim = 1.0 - (logp_diff / (logp_diff + logp_uncertainty + 1.0));

        // Average property similarities
        Ok((mw_sim + logp_sim) / 2.0)
    }

    /// Get context-specific weights
    fn get_context_weights(&self, context: &str) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        
        match context {
            "drug_metabolism" => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.8);
                weights.insert("quantum".to_string(), 0.6);
            }
            "structural_similarity" => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.3);
                weights.insert("quantum".to_string(), 0.2);
            }
            "binding_affinity" => {
                weights.insert("topological".to_string(), 0.7);
                weights.insert("pharmacophoric".to_string(), 1.0);
                weights.insert("quantum".to_string(), 0.8);
            }
            _ => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.5);
                weights.insert("quantum".to_string(), 0.3);
            }
        }
        
        weights
    }

    /// Convert similarity value to linguistic distribution
    fn similarity_to_distribution(&self, similarity: f64) -> Result<SimilarityDistribution> {
        // Convert continuous similarity to fuzzy linguistic categories
        let very_low = if similarity < 0.2 { 1.0 - similarity * 5.0 } else { 0.0 }.max(0.0);
        let low = if similarity >= 0.1 && similarity < 0.4 {
            if similarity < 0.25 { (similarity - 0.1) * 6.67 } else { (0.4 - similarity) * 6.67 }
        } else { 0.0 }.max(0.0);
        let medium = if similarity >= 0.3 && similarity < 0.7 {
            if similarity < 0.5 { (similarity - 0.3) * 5.0 } else { (0.7 - similarity) * 5.0 }
        } else { 0.0 }.max(0.0);
        let high = if similarity >= 0.6 && similarity < 0.9 {
            if similarity < 0.75 { (similarity - 0.6) * 6.67 } else { (0.9 - similarity) * 6.67 }
        } else { 0.0 }.max(0.0);
        let very_high = if similarity > 0.8 { (similarity - 0.8) * 5.0 } else { 0.0 }.max(0.0).min(1.0);

        // Normalize to ensure sum = 1.0
        let total = very_low + low + medium + high + very_high;
        if total > 0.0 {
            SimilarityDistribution::new(
                very_low / total,
                low / total,
                medium / total,
                high / total,
                very_high / total,
            )
        } else {
            SimilarityDistribution::new(1.0, 0.0, 0.0, 0.0, 0.0)
        }
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_engine() {
        let engine = SimilarityEngine::new();
        assert!(!engine.algorithm_weights.is_empty());
    }

    #[test]
    fn test_tanimoto_similarity() {
        let engine = SimilarityEngine::new();
        let mol1 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        let mol2 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        
        let result = engine.calculate_similarity(
            &mol1, &mol2, SimilarityAlgorithm::Tanimoto, "test"
        ).unwrap();
        
        assert!(result.similarity.mean > 0.8); // Should be very similar
        assert_eq!(result.algorithm, "tanimoto");
    }

    #[test]
    fn test_similarity_distribution() {
        let engine = SimilarityEngine::new();
        let dist = engine.similarity_to_distribution(0.8).unwrap();
        
        assert_eq!(dist.most_likely(), "very_high");
        assert!(dist.expected_value() > 0.7);
    }

    #[test]
    fn test_different_molecules() {
        let engine = SimilarityEngine::new();
        let mol1 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        let mol2 = ProbabilisticMolecule::from_smiles("CCCCCCCC").unwrap();
        
        let result = engine.calculate_similarity(
            &mol1, &mol2, SimilarityAlgorithm::Tanimoto, "test"
        ).unwrap();
        
        assert!(result.similarity.mean < 0.8); // Should be less similar
    }
} 