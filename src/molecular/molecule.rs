//! Core probabilistic molecule representation.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::ProbabilisticValue;
use super::features::{FuzzyAromaticity, FuzzyRingSystems, FuzzyFunctionalGroups};
use super::fingerprint::EnhancedFingerprint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced molecular representation with probabilistic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticMolecule {
    /// SMILES string representation
    pub smiles: String,
    /// Fuzzy aromaticity information
    pub fuzzy_aromaticity: FuzzyAromaticity,
    /// Fuzzy ring systems
    pub fuzzy_rings: FuzzyRingSystems,
    /// Fuzzy functional groups
    pub fuzzy_groups: FuzzyFunctionalGroups,
    /// Enhanced molecular fingerprint
    pub fingerprint: EnhancedFingerprint,
    /// Probabilistic molecular properties
    pub properties: ProbabilisticProperties,
}

/// Probabilistic molecular properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticProperties {
    pub molecular_weight: ProbabilisticValue,
    pub logp: ProbabilisticValue,
    pub psa: ProbabilisticValue, // Polar Surface Area
    pub hbd: ProbabilisticValue, // Hydrogen Bond Donors
    pub hba: ProbabilisticValue, // Hydrogen Bond Acceptors
    pub rotatable_bonds: ProbabilisticValue,
}

impl ProbabilisticMolecule {
    /// Create a new probabilistic molecule from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        if smiles.is_empty() {
            return Err(BorgiaError::invalid_smiles(smiles, "Empty SMILES string"));
        }

        // Basic validation
        if smiles.contains("X") || smiles.contains("*") {
            return Err(BorgiaError::invalid_smiles(smiles, "Contains invalid atoms"));
        }

        // Create fuzzy features
        let fuzzy_aromaticity = FuzzyAromaticity::from_smiles(smiles)?;
        let fuzzy_rings = FuzzyRingSystems::from_smiles(smiles)?;
        let fuzzy_groups = FuzzyFunctionalGroups::from_smiles(smiles)?;
        let fingerprint = EnhancedFingerprint::from_smiles(smiles)?;
        let properties = Self::compute_properties(smiles)?;

        Ok(Self {
            smiles: smiles.to_string(),
            fuzzy_aromaticity,
            fuzzy_rings,
            fuzzy_groups,
            fingerprint,
            properties,
        })
    }

    /// Compute probabilistic molecular properties
    fn compute_properties(smiles: &str) -> Result<ProbabilisticProperties> {
        // Simplified property calculations with uncertainty
        // In reality, these would use quantum mechanical calculations,
        // machine learning models, or empirical correlations
        
        let atom_count = smiles.chars().filter(|c| c.is_alphabetic()).count() as f64;
        
        Ok(ProbabilisticProperties {
            molecular_weight: ProbabilisticValue::new_normal(
                atom_count * 12.0 + 20.0, // Rough estimate
                atom_count * 2.0,
                0.95
            ),
            logp: ProbabilisticValue::new_normal(
                (atom_count - 5.0) * 0.5, // Rough estimate
                0.5,
                0.90
            ),
            psa: ProbabilisticValue::new_normal(
                atom_count * 3.0,
                atom_count * 0.5,
                0.85
            ),
            hbd: ProbabilisticValue::new_normal(
                smiles.matches('O').count() as f64 * 0.5,
                0.3,
                0.80
            ),
            hba: ProbabilisticValue::new_normal(
                (smiles.matches('O').count() + smiles.matches('N').count()) as f64,
                0.5,
                0.80
            ),
            rotatable_bonds: ProbabilisticValue::new_normal(
                smiles.matches('-').count() as f64 * 0.7,
                1.0,
                0.75
            ),
        })
    }

    /// Get molecular formula (simplified)
    pub fn molecular_formula(&self) -> String {
        // Very simplified - count C, N, O in SMILES
        let c_count = self.smiles.matches('C').count();
        let n_count = self.smiles.matches('N').count();
        let o_count = self.smiles.matches('O').count();
        
        let mut formula = String::new();
        if c_count > 0 {
            if c_count == 1 {
                formula.push('C');
            } else {
                formula.push_str(&format!("C{}", c_count));
            }
        }
        if n_count > 0 {
            if n_count == 1 {
                formula.push('N');
            } else {
                formula.push_str(&format!("N{}", n_count));
            }
        }
        if o_count > 0 {
            if o_count == 1 {
                formula.push('O');
            } else {
                formula.push_str(&format!("O{}", o_count));
            }
        }
        
        if formula.is_empty() {
            formula = "Unknown".to_string();
        }
        
        formula
    }

    /// Check if molecule contains aromatic rings
    pub fn has_aromatic_rings(&self) -> bool {
        self.fuzzy_aromaticity.overall_aromaticity.mean > 0.5
    }

    /// Get uncertainty in molecular weight
    pub fn molecular_weight_uncertainty(&self) -> f64 {
        self.properties.molecular_weight.std_dev
    }

    /// Validate molecular structure
    pub fn validate(&self) -> Result<()> {
        if self.smiles.is_empty() {
            return Err(BorgiaError::validation("smiles", "SMILES string is empty"));
        }

        if self.properties.molecular_weight.mean <= 0.0 {
            return Err(BorgiaError::validation("molecular_weight", "Molecular weight must be positive"));
        }

        Ok(())
    }
}

impl std::fmt::Display for ProbabilisticMolecule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ProbabilisticMolecule(SMILES: {}, Formula: {}, MW: {:.1}±{:.1})",
            self.smiles,
            self.molecular_formula(),
            self.properties.molecular_weight.mean,
            self.properties.molecular_weight.std_dev
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molecule_creation() {
        let mol = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        assert_eq!(mol.smiles, "CCO");
        assert!(mol.properties.molecular_weight.mean > 0.0);
    }

    #[test]
    fn test_invalid_smiles() {
        let result = ProbabilisticMolecule::from_smiles("");
        assert!(result.is_err());
        
        let result = ProbabilisticMolecule::from_smiles("CCX");
        assert!(result.is_err());
    }

    #[test]
    fn test_molecular_formula() {
        let mol = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        let formula = mol.molecular_formula();
        assert!(formula.contains('C'));
        assert!(formula.contains('O'));
    }

    #[test]
    fn test_validation() {
        let mol = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        assert!(mol.validate().is_ok());
    }
} 