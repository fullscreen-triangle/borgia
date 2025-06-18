//! Fuzzy molecular features with uncertainty quantification.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::ProbabilisticValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fuzzy aromaticity representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyAromaticity {
    /// Overall aromaticity score
    pub overall_aromaticity: ProbabilisticValue,
    /// Ring-specific aromaticity
    pub ring_aromaticity: HashMap<usize, ProbabilisticValue>,
    /// Electron delocalization scores
    pub delocalization: HashMap<usize, ProbabilisticValue>,
    /// Hückel rule compliance
    pub huckel_compliance: ProbabilisticValue,
}

/// Fuzzy ring systems representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyRingSystems {
    /// Detected rings
    pub rings: Vec<Ring>,
    /// Ring strain energies
    pub strain_energies: HashMap<usize, ProbabilisticValue>,
    /// Puckering probabilities
    pub puckering: HashMap<usize, ProbabilisticValue>,
    /// Ring flexibility measures
    pub flexibility: HashMap<usize, ProbabilisticValue>,
}

/// Ring representation with fuzzy properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ring {
    pub atoms: Vec<usize>,
    pub size: usize,
    pub aromaticity: ProbabilisticValue,
    pub planarity: ProbabilisticValue,
    pub substituents: Vec<usize>,
}

/// Fuzzy functional groups representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyFunctionalGroups {
    /// Detected functional groups
    pub groups: Vec<FunctionalGroup>,
    /// Hydrogen bonding capacity
    pub h_bonding: ProbabilisticValue,
    /// Reactivity potential
    pub reactivity: HashMap<String, ProbabilisticValue>,
    /// Pharmacophore features
    pub pharmacophores: Vec<Pharmacophore>,
}

/// Functional group representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalGroup {
    pub name: String,
    pub atoms: Vec<usize>,
    pub confidence: ProbabilisticValue,
    pub reactivity: ProbabilisticValue,
    pub polarity: ProbabilisticValue,
}

/// Pharmacophore feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pharmacophore {
    pub feature_type: String,
    pub center: Vec<f64>, // 3D coordinates
    pub radius: f64,
    pub strength: ProbabilisticValue,
}

impl FuzzyAromaticity {
    /// Create fuzzy aromaticity from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        // Detect aromatic patterns in SMILES
        let has_aromatic_chars = smiles.chars().any(|c| c.is_lowercase());
        let has_rings = smiles.contains('1') || smiles.contains('2') || smiles.contains('3');
        
        let aromaticity_mean = if has_aromatic_chars {
            0.8
        } else if has_rings {
            0.3
        } else {
            0.05
        };

        let overall_aromaticity = ProbabilisticValue::new_normal(
            aromaticity_mean,
            0.2,
            0.90
        );

        // Simplified Hückel rule compliance
        let huckel_compliance = if has_aromatic_chars {
            ProbabilisticValue::new_normal(0.7, 0.3, 0.85)
        } else {
            ProbabilisticValue::new_normal(0.1, 0.2, 0.85)
        };

        Ok(Self {
            overall_aromaticity,
            ring_aromaticity: HashMap::new(),
            delocalization: HashMap::new(),
            huckel_compliance,
        })
    }

    /// Calculate aromaticity uncertainty
    pub fn aromaticity_uncertainty(&self) -> f64 {
        self.overall_aromaticity.std_dev
    }

    /// Check if strongly aromatic (high confidence)
    pub fn is_strongly_aromatic(&self) -> bool {
        self.overall_aromaticity.mean > 0.7 && self.overall_aromaticity.std_dev < 0.2
    }
}

impl FuzzyRingSystems {
    /// Create fuzzy ring systems from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        let mut rings = Vec::new();
        let mut strain_energies = HashMap::new();
        let mut puckering = HashMap::new();
        let mut flexibility = HashMap::new();

        // Simple ring detection based on ring closure digits
        let ring_closures: Vec<char> = smiles.chars()
            .filter(|c| c.is_ascii_digit())
            .collect();

        // Create rings for each pair of ring closure digits
        for (i, &digit) in ring_closures.iter().enumerate() {
            if i % 2 == 0 && i + 1 < ring_closures.len() {
                let ring_size = 6; // Default to 6-membered rings
                
                let ring = Ring {
                    atoms: (0..ring_size).collect(),
                    size: ring_size,
                    aromaticity: ProbabilisticValue::new_normal(0.4, 0.3, 0.85),
                    planarity: ProbabilisticValue::new_normal(0.8, 0.2, 0.90),
                    substituents: Vec::new(),
                };

                // Ring strain energy (depends on ring size)
                let strain_energy = match ring_size {
                    3 => ProbabilisticValue::new_normal(27.5, 2.0, 0.95), // Cyclopropane
                    4 => ProbabilisticValue::new_normal(26.3, 2.0, 0.95), // Cyclobutane
                    5 => ProbabilisticValue::new_normal(6.2, 1.0, 0.95),  // Cyclopentane
                    6 => ProbabilisticValue::new_normal(0.1, 0.5, 0.95),  // Cyclohexane
                    7 => ProbabilisticValue::new_normal(6.2, 1.5, 0.95),  // Cycloheptane
                    _ => ProbabilisticValue::new_normal(10.0, 5.0, 0.80), // Larger rings
                };

                strain_energies.insert(rings.len(), strain_energy);
                
                // Puckering probability (higher for larger rings)
                let puckering_prob = if ring_size >= 6 {
                    ProbabilisticValue::new_normal(0.8, 0.2, 0.85)
                } else {
                    ProbabilisticValue::new_normal(0.2, 0.2, 0.85)
                };
                puckering.insert(rings.len(), puckering_prob);

                // Flexibility (inversely related to ring size)
                let flexibility_score = ProbabilisticValue::new_normal(
                    1.0 / ring_size as f64,
                    0.1,
                    0.80
                );
                flexibility.insert(rings.len(), flexibility_score);

                rings.push(ring);
            }
        }

        Ok(Self {
            rings,
            strain_energies,
            puckering,
            flexibility,
        })
    }

    /// Get total number of rings
    pub fn ring_count(&self) -> usize {
        self.rings.len()
    }

    /// Get average ring strain
    pub fn average_strain(&self) -> f64 {
        if self.strain_energies.is_empty() {
            return 0.0;
        }

        let total: f64 = self.strain_energies.values()
            .map(|strain| strain.mean)
            .sum();
        total / self.strain_energies.len() as f64
    }
}

impl FuzzyFunctionalGroups {
    /// Create fuzzy functional groups from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        let mut groups = Vec::new();
        let mut reactivity = HashMap::new();

        // Detect common functional groups
        if smiles.contains("OH") {
            groups.push(FunctionalGroup {
                name: "hydroxyl".to_string(),
                atoms: vec![0, 1], // Simplified
                confidence: ProbabilisticValue::new_normal(0.9, 0.1, 0.95),
                reactivity: ProbabilisticValue::new_normal(0.7, 0.2, 0.90),
                polarity: ProbabilisticValue::new_normal(0.8, 0.1, 0.95),
            });
            reactivity.insert("nucleophilic".to_string(), 
                ProbabilisticValue::new_normal(0.7, 0.2, 0.85));
        }

        if smiles.contains("C=O") {
            groups.push(FunctionalGroup {
                name: "carbonyl".to_string(),
                atoms: vec![0, 1], // Simplified
                confidence: ProbabilisticValue::new_normal(0.85, 0.15, 0.90),
                reactivity: ProbabilisticValue::new_normal(0.8, 0.2, 0.90),
                polarity: ProbabilisticValue::new_normal(0.9, 0.1, 0.95),
            });
            reactivity.insert("electrophilic".to_string(),
                ProbabilisticValue::new_normal(0.8, 0.2, 0.85));
        }

        if smiles.contains("NH") {
            groups.push(FunctionalGroup {
                name: "amine".to_string(),
                atoms: vec![0, 1], // Simplified
                confidence: ProbabilisticValue::new_normal(0.8, 0.2, 0.85),
                reactivity: ProbabilisticValue::new_normal(0.6, 0.3, 0.80),
                polarity: ProbabilisticValue::new_normal(0.7, 0.2, 0.90),
            });
            reactivity.insert("basic".to_string(),
                ProbabilisticValue::new_normal(0.6, 0.3, 0.80));
        }

        // Calculate hydrogen bonding capacity
        let oh_count = smiles.matches("OH").count() as f64;
        let nh_count = smiles.matches("NH").count() as f64;
        let carbonyl_count = smiles.matches("C=O").count() as f64;

        let h_bonding = ProbabilisticValue::new_normal(
            (oh_count + nh_count + carbonyl_count * 0.5) / 10.0, // Normalized
            0.2,
            0.85
        );

        Ok(Self {
            groups,
            h_bonding,
            reactivity,
            pharmacophores: Vec::new(), // Would be computed from 3D structure
        })
    }

    /// Get functional group count
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Check if has hydrogen bonding capability
    pub fn has_hydrogen_bonding(&self) -> bool {
        self.h_bonding.mean > 0.3
    }

    /// Get most reactive functional group
    pub fn most_reactive_group(&self) -> Option<&FunctionalGroup> {
        self.groups.iter()
            .max_by(|a, b| a.reactivity.mean.partial_cmp(&b.reactivity.mean).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_aromaticity() {
        let aromatic = FuzzyAromaticity::from_smiles("c1ccccc1").unwrap(); // Benzene
        assert!(aromatic.is_strongly_aromatic());
        
        let aliphatic = FuzzyAromaticity::from_smiles("CCCCCC").unwrap();
        assert!(!aliphatic.is_strongly_aromatic());
    }

    #[test]
    fn test_ring_systems() {
        let rings = FuzzyRingSystems::from_smiles("C1CCCCC1").unwrap(); // Cyclohexane
        assert!(rings.ring_count() > 0);
        assert!(rings.average_strain() >= 0.0);
    }

    #[test]
    fn test_functional_groups() {
        let groups = FuzzyFunctionalGroups::from_smiles("CCO").unwrap(); // Ethanol
        assert!(groups.has_hydrogen_bonding());
        assert!(groups.group_count() > 0);
    }

    #[test]
    fn test_carbonyl_detection() {
        let groups = FuzzyFunctionalGroups::from_smiles("CC(=O)C").unwrap(); // Acetone
        assert!(groups.groups.iter().any(|g| g.name == "carbonyl"));
    }
} 