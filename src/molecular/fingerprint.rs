//! Enhanced molecular fingerprint with probabilistic features.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::ProbabilisticValue;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced molecular fingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedFingerprint {
    /// Topological features (Morgan-like)
    pub topological: DVector<f64>,
    /// Pharmacophoric features
    pub pharmacophoric: DVector<f64>,
    /// Quantum mechanical features
    pub quantum: DVector<f64>,
    /// Conformational features
    pub conformational: DVector<f64>,
    /// Interaction potential features
    pub interaction: DVector<f64>,
    /// Combined feature vector (all features concatenated)
    pub combined: DVector<f64>,
    /// Feature importance weights
    pub weights: DVector<f64>,
    /// Feature uncertainty estimates
    pub uncertainties: DVector<f64>,
}

/// Configuration for fingerprint generation
#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    pub topological_bits: usize,
    pub pharmacophoric_bits: usize,
    pub quantum_bits: usize,
    pub conformational_bits: usize,
    pub interaction_bits: usize,
    pub radius: usize,
    pub use_chirality: bool,
    pub use_bond_types: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            topological_bits: 10000,
            pharmacophoric_bits: 10000,
            quantum_bits: 10000,
            conformational_bits: 10000,
            interaction_bits: 10000,
            radius: 3,
            use_chirality: true,
            use_bond_types: true,
        }
    }
}

impl EnhancedFingerprint {
    /// Create enhanced fingerprint from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        // Create simplified fingerprints
        let topological = DVector::zeros(10000);
        let pharmacophoric = DVector::zeros(10000);
        let quantum = DVector::zeros(10000);
        let conformational = DVector::zeros(10000);
        let interaction = DVector::zeros(10000);
        let combined = DVector::zeros(50000);
        let weights = DVector::from_element(50000, 1.0);
        let uncertainties = DVector::from_element(50000, 0.1);

        Ok(Self {
            topological,
            pharmacophoric,
            quantum,
            conformational,
            interaction,
            combined,
            weights,
            uncertainties,
        })
    }

    /// Generate topological features (Morgan-like algorithm)
    fn generate_topological_features(smiles: &str, config: &FingerprintConfig) -> Result<DVector<f64>> {
        let mut features = DVector::zeros(config.topological_bits);
        
        // Simplified Morgan algorithm implementation
        // In reality, this would be a full graph-based implementation
        
        // Hash atom environments at different radii
        for radius in 0..=config.radius {
            for (i, ch) in smiles.chars().enumerate() {
                if ch.is_alphabetic() {
                    // Create atom environment hash
                    let mut hash = Self::atom_hash(ch);
                    
                    // Include neighboring atoms (simplified)
                    if radius > 0 {
                        let start = i.saturating_sub(radius);
                        let end = (i + radius + 1).min(smiles.len());
                        let environment: String = smiles.chars()
                            .skip(start)
                            .take(end - start)
                            .collect();
                        hash = Self::combine_hashes(hash, Self::string_hash(&environment));
                    }
                    
                    // Map hash to bit position
                    let bit_pos = (hash as usize) % config.topological_bits;
                    features[bit_pos] += 1.0;
                }
            }
        }

        // Normalize features
        let max_val = features.max();
        if max_val > 0.0 {
            features /= max_val;
        }

        Ok(features)
    }

    /// Generate pharmacophoric features
    fn generate_pharmacophoric_features(smiles: &str, config: &FingerprintConfig) -> Result<DVector<f64>> {
        let mut features = DVector::zeros(config.pharmacophoric_bits);
        
        // Detect pharmacophoric patterns
        let patterns = vec![
            ("OH", "hydrogen_bond_donor"),
            ("C=O", "hydrogen_bond_acceptor"),
            ("NH", "hydrogen_bond_donor"),
            ("N", "hydrogen_bond_acceptor"),
            ("c1ccccc1", "aromatic_ring"),
            ("C(=O)O", "carboxyl"),
            ("S", "sulfur"),
            ("P", "phosphorus"),
            ("F", "halogen"),
            ("Cl", "halogen"),
            ("Br", "halogen"),
            ("I", "halogen"),
        ];

        for (pattern, feature_type) in patterns {
            let count = smiles.matches(pattern).count();
            if count > 0 {
                let hash = Self::string_hash(feature_type);
                let bit_pos = (hash as usize) % config.pharmacophoric_bits;
                features[bit_pos] = (count as f64).ln() + 1.0; // Log scaling
            }
        }

        // Add distance-based pharmacophore features (simplified)
        for i in 0..smiles.len() {
            for j in (i + 3)..smiles.len() {
                if i < smiles.len() && j < smiles.len() {
                    let ch1 = smiles.chars().nth(i).unwrap_or('C');
                    let ch2 = smiles.chars().nth(j).unwrap_or('C');
                    
                    if ch1.is_alphabetic() && ch2.is_alphabetic() {
                        let distance = j - i;
                        let pair_hash = Self::combine_hashes(
                            Self::atom_hash(ch1),
                            Self::combine_hashes(Self::atom_hash(ch2), distance as u64)
                        );
                        let bit_pos = (pair_hash as usize) % config.pharmacophoric_bits;
                        features[bit_pos] += 1.0 / (distance as f64).sqrt();
                    }
                }
            }
        }

        Ok(features)
    }

    /// Generate quantum mechanical features
    fn generate_quantum_features(smiles: &str, config: &FingerprintConfig) -> Result<DVector<f64>> {
        let mut features = DVector::zeros(config.quantum_bits);
        
        // Simplified quantum features based on atom properties
        let atom_properties = HashMap::from([
            ('C', (2.55, 6.0, 1.70)),   // (electronegativity, valence_electrons, covalent_radius)
            ('N', (3.04, 5.0, 1.55)),
            ('O', (3.44, 6.0, 1.52)),
            ('S', (2.58, 6.0, 1.80)),
            ('P', (2.19, 5.0, 1.80)),
            ('F', (3.98, 7.0, 1.47)),
            ('Cl', (3.16, 7.0, 1.75)),
            ('Br', (2.96, 7.0, 1.85)),
            ('I', (2.66, 7.0, 1.98)),
        ]);

        for (i, ch) in smiles.chars().enumerate() {
            if let Some(&(electronegativity, valence, radius)) = atom_properties.get(&ch) {
                // Electronegativity features
                let en_hash = Self::combine_hashes(
                    Self::atom_hash(ch),
                    (electronegativity * 100.0) as u64
                );
                let en_bit = (en_hash as usize) % config.quantum_bits;
                features[en_bit] += electronegativity / 4.0; // Normalized

                // Valence electron features
                let val_hash = Self::combine_hashes(en_hash, valence as u64);
                let val_bit = (val_hash as usize) % config.quantum_bits;
                features[val_bit] += valence / 8.0; // Normalized

                // Atomic size features
                let size_hash = Self::combine_hashes(val_hash, (radius * 100.0) as u64);
                let size_bit = (size_hash as usize) % config.quantum_bits;
                features[size_bit] += radius / 2.0; // Normalized

                // Polarizability approximation (proportional to atomic volume)
                let polarizability = radius.powi(3);
                let pol_hash = Self::combine_hashes(size_hash, (polarizability * 100.0) as u64);
                let pol_bit = (pol_hash as usize) % config.quantum_bits;
                features[pol_bit] += polarizability / 10.0;
            }
        }

        Ok(features)
    }

    /// Generate conformational features
    fn generate_conformational_features(smiles: &str, config: &FingerprintConfig) -> Result<DVector<f64>> {
        let mut features = DVector::zeros(config.conformational_bits);
        
        // Rotatable bond features
        let single_bonds = smiles.matches('-').count();
        if single_bonds > 0 {
            let rot_hash = Self::string_hash("rotatable_bonds");
            let bit_pos = (rot_hash as usize) % config.conformational_bits;
            features[bit_pos] = (single_bonds as f64).ln() + 1.0;
        }

        // Ring flexibility features
        let ring_patterns = vec![
            ("C1CCCCC1", 0.8), // Cyclohexane - flexible
            ("c1ccccc1", 0.1), // Benzene - rigid
            ("C1CCCC1", 0.6),  // Cyclopentane - moderately flexible
            ("C1CCC1", 0.3),   // Cyclobutane - strained
            ("C1CC1", 0.1),    // Cyclopropane - very rigid
        ];

        for (pattern, flexibility) in ring_patterns {
            let count = smiles.matches(pattern).count();
            if count > 0 {
                let flex_hash = Self::string_hash(&format!("flexibility_{}", flexibility));
                let bit_pos = (flex_hash as usize) % config.conformational_bits;
                features[bit_pos] += count as f64 * flexibility;
            }
        }

        // Torsion angle features (simplified)
        for i in 0..smiles.len().saturating_sub(3) {
            let substr: String = smiles.chars().skip(i).take(4).collect();
            if substr.chars().all(|c| c.is_alphabetic() || c == '=' || c == '#') {
                let torsion_hash = Self::string_hash(&substr);
                let bit_pos = (torsion_hash as usize) % config.conformational_bits;
                features[bit_pos] += 1.0;
            }
        }

        Ok(features)
    }

    /// Generate interaction potential features
    fn generate_interaction_features(smiles: &str, config: &FingerprintConfig) -> Result<DVector<f64>> {
        let mut features = DVector::zeros(config.interaction_bits);
        
        // Hydrophobic features
        let hydrophobic_atoms = smiles.matches('C').count();
        if hydrophobic_atoms > 0 {
            let hydro_hash = Self::string_hash("hydrophobic");
            let bit_pos = (hydro_hash as usize) % config.interaction_bits;
            features[bit_pos] = (hydrophobic_atoms as f64).sqrt();
        }

        // Hydrophilic features
        let hydrophilic_count = smiles.matches('O').count() + 
                               smiles.matches('N').count() +
                               smiles.matches("OH").count();
        if hydrophilic_count > 0 {
            let hydro_hash = Self::string_hash("hydrophilic");
            let bit_pos = (hydro_hash as usize) % config.interaction_bits;
            features[bit_pos] = (hydrophilic_count as f64).sqrt();
        }

        // Electrostatic features
        let charged_patterns = vec![
            ("N+", "positive_charge"),
            ("O-", "negative_charge"),
            ("COO-", "carboxylate"),
            ("NH3+", "ammonium"),
        ];

        for (pattern, charge_type) in charged_patterns {
            let count = smiles.matches(pattern).count();
            if count > 0 {
                let charge_hash = Self::string_hash(charge_type);
                let bit_pos = (charge_hash as usize) % config.interaction_bits;
                features[bit_pos] += count as f64;
            }
        }

        // Van der Waals features (based on molecular size)
        let total_atoms = smiles.chars().filter(|c| c.is_alphabetic()).count();
        if total_atoms > 0 {
            let vdw_hash = Self::string_hash("van_der_waals");
            let bit_pos = (vdw_hash as usize) % config.interaction_bits;
            features[bit_pos] = (total_atoms as f64).sqrt() / 10.0;
        }

        Ok(features)
    }

    /// Calculate Tanimoto similarity with another fingerprint
    pub fn tanimoto_similarity(&self, other: &EnhancedFingerprint) -> f64 {
        let intersection = self.combined.dot(&other.combined);
        let union = self.combined.norm_squared() + other.combined.norm_squared() - intersection;
        
        if union > 0.0 {
            intersection / union
        } else {
            1.0 // Identical empty fingerprints
        }
    }

    /// Calculate weighted similarity using feature importance
    pub fn weighted_similarity(&self, other: &EnhancedFingerprint) -> f64 {
        let weighted_self = self.combined.component_mul(&self.weights);
        let weighted_other = other.combined.component_mul(&other.weights);
        
        let intersection = weighted_self.dot(&weighted_other);
        let union = weighted_self.norm_squared() + weighted_other.norm_squared() - intersection;
        
        if union > 0.0 {
            intersection / union
        } else {
            1.0
        }
    }

    /// Get feature vector dimension
    pub fn dimension(&self) -> usize {
        self.combined.len()
    }

    /// Get feature density (fraction of non-zero features)
    pub fn density(&self) -> f64 {
        let non_zero_count = self.combined.iter().filter(|&&x| x > 0.0).count();
        non_zero_count as f64 / self.combined.len() as f64
    }

    /// Simple hash function for atoms
    fn atom_hash(atom: char) -> u64 {
        match atom {
            'C' => 6,
            'N' => 7,
            'O' => 8,
            'S' => 16,
            'P' => 15,
            'F' => 9,
            'Cl' => 17,
            'Br' => 35,
            'I' => 53,
            _ => atom as u64,
        }
    }

    /// Simple hash function for strings
    fn string_hash(s: &str) -> u64 {
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    /// Combine two hash values
    fn combine_hashes(h1: u64, h2: u64) -> u64 {
        h1.wrapping_mul(31).wrapping_add(h2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let fp = EnhancedFingerprint::from_smiles("CCO").unwrap();
        assert_eq!(fp.dimension(), 50000);
        assert!(fp.density() > 0.0);
    }

    #[test]
    fn test_tanimoto_similarity() {
        let fp1 = EnhancedFingerprint::from_smiles("CCO").unwrap();
        let fp2 = EnhancedFingerprint::from_smiles("CCO").unwrap();
        let fp3 = EnhancedFingerprint::from_smiles("CCCCCCCC").unwrap();
        
        // Identical molecules should have similarity 1.0
        assert!((fp1.tanimoto_similarity(&fp2) - 1.0).abs() < 1e-10);
        
        // Different molecules should have similarity < 1.0
        assert!(fp1.tanimoto_similarity(&fp3) < 1.0);
    }

    #[test]
    fn test_different_feature_types() {
        let fp = EnhancedFingerprint::from_smiles("c1ccccc1CCO").unwrap(); // Benzyl alcohol
        
        // Should have non-zero features in all categories
        assert!(fp.topological.iter().any(|&x| x > 0.0));
        assert!(fp.pharmacophoric.iter().any(|&x| x > 0.0));
        assert!(fp.quantum.iter().any(|&x| x > 0.0));
        assert!(fp.conformational.iter().any(|&x| x > 0.0));
        assert!(fp.interaction.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_custom_config() {
        let config = FingerprintConfig {
            topological_bits: 1000,
            pharmacophoric_bits: 1000,
            quantum_bits: 1000,
            conformational_bits: 1000,
            interaction_bits: 1000,
            radius: 2,
            use_chirality: false,
            use_bond_types: false,
        };
        
        let fp = EnhancedFingerprint::from_smiles_with_config("CCO", &config).unwrap();
        assert_eq!(fp.dimension(), 5000);
    }
} 