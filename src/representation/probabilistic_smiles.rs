use std::collections::HashMap;
use crate::error::BorgiaResult;
use crate::probabilistic::ProbabilisticValue;
use crate::representation::smiles_representation::{SMILESRepresentation, SMILESParser, SMILESAtom, SMILESBond};
use crate::algorithms::morgan::MolecularGraph;

/// Probabilistic SMILES representation that extends regular SMILES with uncertainty
/// This allows for fuzzy molecular structures and uncertain bond orders
#[derive(Debug, Clone)]
pub struct ProbabilisticSMILES {
    /// Base SMILES representation
    pub base_smiles: SMILESRepresentation,
    /// Probabilistic atom properties
    pub probabilistic_atoms: HashMap<usize, ProbabilisticAtom>,
    /// Probabilistic bond properties
    pub probabilistic_bonds: HashMap<(usize, usize), ProbabilisticBond>,
    /// Overall molecular uncertainty
    pub molecular_uncertainty: ProbabilisticValue,
    /// Confidence in the SMILES representation
    pub representation_confidence: ProbabilisticValue,
}

/// Probabilistic atom with uncertain properties
#[derive(Debug, Clone)]
pub struct ProbabilisticAtom {
    /// Atom ID
    pub id: usize,
    /// Probabilistic atomic number (for uncertain element identification)
    pub atomic_number: ProbabilisticValue,
    /// Probabilistic charge
    pub charge: ProbabilisticValue,
    /// Probabilistic hydrogen count
    pub hydrogen_count: ProbabilisticValue,
    /// Aromaticity probability
    pub aromaticity_probability: f64,
    /// Chirality confidence
    pub chirality_confidence: Option<ProbabilisticValue>,
    /// Isotope probability distribution
    pub isotope_distribution: HashMap<u16, f64>,
}

/// Probabilistic bond with uncertain properties
#[derive(Debug, Clone)]
pub struct ProbabilisticBond {
    /// Bond between atoms
    pub from_atom: usize,
    pub to_atom: usize,
    /// Probabilistic bond order
    pub bond_order: ProbabilisticValue,
    /// Bond type probabilities
    pub bond_type_probabilities: HashMap<String, f64>,
    /// Stereochemistry confidence
    pub stereochemistry_confidence: Option<ProbabilisticValue>,
}

/// Probabilistic SMILES parser that handles uncertain molecular structures
#[derive(Debug)]
pub struct ProbabilisticSMILESParser {
    /// Base SMILES parser
    base_parser: SMILESParser,
    /// Default uncertainty levels
    default_atomic_uncertainty: f64,
    default_bond_uncertainty: f64,
    default_charge_uncertainty: f64,
}

impl Default for ProbabilisticSMILESParser {
    fn default() -> Self {
        Self {
            base_parser: SMILESParser::new(),
            default_atomic_uncertainty: 0.05,
            default_bond_uncertainty: 0.1,
            default_charge_uncertainty: 0.2,
        }
    }
}

impl ProbabilisticSMILESParser {
    pub fn new(
        atomic_uncertainty: f64,
        bond_uncertainty: f64,
        charge_uncertainty: f64,
    ) -> Self {
        Self {
            base_parser: SMILESParser::new(),
            default_atomic_uncertainty: atomic_uncertainty,
            default_bond_uncertainty: bond_uncertainty,
            default_charge_uncertainty: charge_uncertainty,
        }
    }

    /// Parse SMILES string into probabilistic representation
    pub fn parse(&mut self, smiles: &str) -> BorgiaResult<ProbabilisticSMILES> {
        // Parse base SMILES
        let base_smiles = self.base_parser.parse(smiles)?;
        
        // Convert to probabilistic representation
        let probabilistic_atoms = self.create_probabilistic_atoms(&base_smiles)?;
        let probabilistic_bonds = self.create_probabilistic_bonds(&base_smiles)?;
        
        // Calculate molecular uncertainty
        let molecular_uncertainty = self.calculate_molecular_uncertainty(&probabilistic_atoms, &probabilistic_bonds);
        
        // Estimate representation confidence
        let representation_confidence = self.estimate_representation_confidence(&base_smiles);
        
        Ok(ProbabilisticSMILES {
            base_smiles,
            probabilistic_atoms,
            probabilistic_bonds,
            molecular_uncertainty,
            representation_confidence,
        })
    }

    /// Create probabilistic atoms from base SMILES atoms
    fn create_probabilistic_atoms(
        &self,
        base_smiles: &SMILESRepresentation,
    ) -> BorgiaResult<HashMap<usize, ProbabilisticAtom>> {
        let mut probabilistic_atoms = HashMap::new();
        
        for atom in &base_smiles.atoms {
            let prob_atom = self.create_probabilistic_atom(atom)?;
            probabilistic_atoms.insert(atom.id, prob_atom);
        }
        
        Ok(probabilistic_atoms)
    }

    /// Create a single probabilistic atom
    fn create_probabilistic_atom(&self, atom: &SMILESAtom) -> BorgiaResult<ProbabilisticAtom> {
        // Atomic number with uncertainty
        let atomic_number = ProbabilisticValue::new(
            atom.atomic_number as f64,
            self.default_atomic_uncertainty,
        );
        
        // Charge with uncertainty
        let charge = ProbabilisticValue::new(
            atom.charge as f64,
            self.default_charge_uncertainty,
        );
        
        // Hydrogen count with uncertainty
        let hydrogen_count = ProbabilisticValue::new(
            atom.hydrogen_count as f64,
            self.default_atomic_uncertainty,
        );
        
        // Aromaticity probability
        let aromaticity_probability = if atom.aromatic { 0.95 } else { 0.05 };
        
        // Chirality confidence
        let chirality_confidence = if atom.chirality.is_some() {
            Some(ProbabilisticValue::new(0.8, 0.2))
        } else {
            None
        };
        
        // Isotope distribution (simplified)
        let mut isotope_distribution = HashMap::new();
        if let Some(isotope) = atom.isotope {
            isotope_distribution.insert(isotope, 0.95);
            isotope_distribution.insert(isotope - 1, 0.03);
            isotope_distribution.insert(isotope + 1, 0.02);
        } else {
            // Natural isotope distribution for common elements
            match atom.atomic_number {
                6 => { // Carbon
                    isotope_distribution.insert(12, 0.989);
                    isotope_distribution.insert(13, 0.011);
                }
                7 => { // Nitrogen
                    isotope_distribution.insert(14, 0.996);
                    isotope_distribution.insert(15, 0.004);
                }
                8 => { // Oxygen
                    isotope_distribution.insert(16, 0.9976);
                    isotope_distribution.insert(17, 0.0004);
                    isotope_distribution.insert(18, 0.0020);
                }
                _ => {
                    isotope_distribution.insert(atom.atomic_number as u16, 1.0);
                }
            }
        }
        
        Ok(ProbabilisticAtom {
            id: atom.id,
            atomic_number,
            charge,
            hydrogen_count,
            aromaticity_probability,
            chirality_confidence,
            isotope_distribution,
        })
    }

    /// Create probabilistic bonds from base SMILES bonds
    fn create_probabilistic_bonds(
        &self,
        base_smiles: &SMILESRepresentation,
    ) -> BorgiaResult<HashMap<(usize, usize), ProbabilisticBond>> {
        let mut probabilistic_bonds = HashMap::new();
        
        for bond in &base_smiles.bonds {
            let prob_bond = self.create_probabilistic_bond(bond)?;
            let key = (bond.from_atom.min(bond.to_atom), bond.from_atom.max(bond.to_atom));
            probabilistic_bonds.insert(key, prob_bond);
        }
        
        Ok(probabilistic_bonds)
    }

    /// Create a single probabilistic bond
    fn create_probabilistic_bond(&self, bond: &SMILESBond) -> BorgiaResult<ProbabilisticBond> {
        // Bond order with uncertainty
        let bond_order = ProbabilisticValue::new(
            bond.bond_type.to_order(),
            self.default_bond_uncertainty,
        );
        
        // Bond type probabilities
        let mut bond_type_probabilities = HashMap::new();
        match bond.bond_type {
            crate::representation::smiles_representation::BondType::Single => {
                bond_type_probabilities.insert("single".to_string(), 0.9);
                bond_type_probabilities.insert("aromatic".to_string(), 0.1);
            }
            crate::representation::smiles_representation::BondType::Double => {
                bond_type_probabilities.insert("double".to_string(), 0.95);
                bond_type_probabilities.insert("single".to_string(), 0.05);
            }
            crate::representation::smiles_representation::BondType::Triple => {
                bond_type_probabilities.insert("triple".to_string(), 0.98);
                bond_type_probabilities.insert("double".to_string(), 0.02);
            }
            crate::representation::smiles_representation::BondType::Aromatic => {
                bond_type_probabilities.insert("aromatic".to_string(), 0.9);
                bond_type_probabilities.insert("single".to_string(), 0.1);
            }
            _ => {
                bond_type_probabilities.insert("single".to_string(), 0.8);
                bond_type_probabilities.insert("other".to_string(), 0.2);
            }
        }
        
        // Stereochemistry confidence
        let stereochemistry_confidence = if bond.stereochemistry.is_some() {
            Some(ProbabilisticValue::new(0.7, 0.3))
        } else {
            None
        };
        
        Ok(ProbabilisticBond {
            from_atom: bond.from_atom,
            to_atom: bond.to_atom,
            bond_order,
            bond_type_probabilities,
            stereochemistry_confidence,
        })
    }

    /// Calculate overall molecular uncertainty
    fn calculate_molecular_uncertainty(
        &self,
        atoms: &HashMap<usize, ProbabilisticAtom>,
        bonds: &HashMap<(usize, usize), ProbabilisticBond>,
    ) -> ProbabilisticValue {
        let mut total_uncertainty = 0.0;
        let mut component_count = 0;
        
        // Sum atomic uncertainties
        for atom in atoms.values() {
            total_uncertainty += atom.atomic_number.variance;
            total_uncertainty += atom.charge.variance;
            total_uncertainty += atom.hydrogen_count.variance;
            component_count += 3;
        }
        
        // Sum bond uncertainties
        for bond in bonds.values() {
            total_uncertainty += bond.bond_order.variance;
            component_count += 1;
        }
        
        let average_uncertainty = if component_count > 0 {
            total_uncertainty / component_count as f64
        } else {
            0.0
        };
        
        ProbabilisticValue::new(0.5, average_uncertainty.sqrt())
    }

    /// Estimate confidence in the SMILES representation
    fn estimate_representation_confidence(&self, base_smiles: &SMILESRepresentation) -> ProbabilisticValue {
        let mut confidence_factors = Vec::new();
        
        // Factor 1: Number of atoms (more atoms = more complexity = less confidence)
        let atom_factor = if base_smiles.atoms.len() < 10 {
            0.9
        } else if base_smiles.atoms.len() < 50 {
            0.8
        } else {
            0.7
        };
        confidence_factors.push(atom_factor);
        
        // Factor 2: Presence of aromatic systems
        let aromatic_count = base_smiles.atoms.iter().filter(|a| a.aromatic).count();
        let aromatic_factor = if aromatic_count == 0 {
            0.95
        } else if aromatic_count < 5 {
            0.85
        } else {
            0.75
        };
        confidence_factors.push(aromatic_factor);
        
        // Factor 3: Presence of stereochemistry
        let stereo_factor = if base_smiles.stereochemistry.is_empty() {
            0.9
        } else {
            0.8
        };
        confidence_factors.push(stereo_factor);
        
        // Factor 4: Ring complexity
        let ring_factor = if base_smiles.rings.len() < 2 {
            0.9
        } else if base_smiles.rings.len() < 5 {
            0.8
        } else {
            0.7
        };
        confidence_factors.push(ring_factor);
        
        // Combine factors
        let overall_confidence: f64 = confidence_factors.iter().product();
        let uncertainty = 1.0 - overall_confidence;
        
        ProbabilisticValue::new(overall_confidence, uncertainty * 0.5)
    }
}

/// Probabilistic SMILES generator that can output SMILES with uncertainty annotations
#[derive(Debug)]
pub struct ProbabilisticSMILESGenerator {
    /// Include uncertainty annotations in output
    pub include_uncertainty: bool,
    /// Uncertainty threshold for annotations
    pub uncertainty_threshold: f64,
}

impl Default for ProbabilisticSMILESGenerator {
    fn default() -> Self {
        Self {
            include_uncertainty: true,
            uncertainty_threshold: 0.1,
        }
    }
}

impl ProbabilisticSMILESGenerator {
    pub fn new(include_uncertainty: bool, uncertainty_threshold: f64) -> Self {
        Self {
            include_uncertainty,
            uncertainty_threshold,
        }
    }

    /// Generate SMILES string from probabilistic representation
    pub fn generate(&self, prob_smiles: &ProbabilisticSMILES) -> BorgiaResult<String> {
        let mut output = String::new();
        
        // Start with base SMILES
        output.push_str(&prob_smiles.base_smiles.smiles);
        
        // Add uncertainty annotations if requested
        if self.include_uncertainty {
            output.push_str(&self.generate_uncertainty_annotations(prob_smiles)?);
        }
        
        Ok(output)
    }

    /// Generate uncertainty annotations
    fn generate_uncertainty_annotations(&self, prob_smiles: &ProbabilisticSMILES) -> BorgiaResult<String> {
        let mut annotations = String::new();
        
        // Add molecular uncertainty annotation
        if prob_smiles.molecular_uncertainty.variance > self.uncertainty_threshold {
            annotations.push_str(&format!(
                " |UNCERTAINTY:{:.3}|",
                prob_smiles.molecular_uncertainty.variance
            ));
        }
        
        // Add confidence annotation
        if prob_smiles.representation_confidence.mean < 0.9 {
            annotations.push_str(&format!(
                " |CONFIDENCE:{:.3}|",
                prob_smiles.representation_confidence.mean
            ));
        }
        
        // Add atom-specific uncertainties
        for (atom_id, atom) in &prob_smiles.probabilistic_atoms {
            if atom.atomic_number.variance > self.uncertainty_threshold {
                annotations.push_str(&format!(
                    " |ATOM{}:±{:.3}|",
                    atom_id,
                    atom.atomic_number.variance
                ));
            }
        }
        
        // Add bond-specific uncertainties
        for ((from, to), bond) in &prob_smiles.probabilistic_bonds {
            if bond.bond_order.variance > self.uncertainty_threshold {
                annotations.push_str(&format!(
                    " |BOND{}-{}:±{:.3}|",
                    from, to,
                    bond.bond_order.variance
                ));
            }
        }
        
        Ok(annotations)
    }

    /// Generate canonical probabilistic SMILES
    pub fn generate_canonical(&self, prob_smiles: &ProbabilisticSMILES) -> BorgiaResult<String> {
        // This would implement canonical ordering considering uncertainties
        // For now, delegate to regular generation
        self.generate(prob_smiles)
    }
}

/// Utilities for working with probabilistic SMILES
pub struct ProbabilisticSMILESUtils;

impl ProbabilisticSMILESUtils {
    /// Compare two probabilistic SMILES representations
    pub fn compare(
        smiles1: &ProbabilisticSMILES,
        smiles2: &ProbabilisticSMILES,
    ) -> BorgiaResult<ProbabilisticValue> {
        // Compare molecular graphs
        let graph1 = smiles1.base_smiles.graph.as_ref().unwrap();
        let graph2 = smiles2.base_smiles.graph.as_ref().unwrap();
        
        // Basic structural comparison
        if graph1.nodes.len() != graph2.nodes.len() || graph1.edges.len() != graph2.edges.len() {
            return Ok(ProbabilisticValue::new(0.0, 0.1));
        }
        
        // Compare probabilistic properties
        let mut similarity_scores = Vec::new();
        
        // Atomic similarity
        for (id1, atom1) in &smiles1.probabilistic_atoms {
            if let Some(atom2) = smiles2.probabilistic_atoms.get(id1) {
                let atomic_similarity = Self::compare_atoms(atom1, atom2);
                similarity_scores.push(atomic_similarity);
            }
        }
        
        // Bond similarity
        for (key1, bond1) in &smiles1.probabilistic_bonds {
            if let Some(bond2) = smiles2.probabilistic_bonds.get(key1) {
                let bond_similarity = Self::compare_bonds(bond1, bond2);
                similarity_scores.push(bond_similarity);
            }
        }
        
        // Aggregate similarities
        if similarity_scores.is_empty() {
            Ok(ProbabilisticValue::new(0.0, 1.0))
        } else {
            let mean_similarity = similarity_scores.iter().map(|s| s.mean).sum::<f64>() / similarity_scores.len() as f64;
            let variance_sum = similarity_scores.iter().map(|s| s.variance).sum::<f64>();
            let combined_variance = variance_sum / (similarity_scores.len() as f64).powi(2);
            
            Ok(ProbabilisticValue::new(mean_similarity, combined_variance.sqrt()))
        }
    }

    /// Compare two probabilistic atoms
    fn compare_atoms(atom1: &ProbabilisticAtom, atom2: &ProbabilisticAtom) -> ProbabilisticValue {
        // Compare atomic numbers
        let atomic_diff = (atom1.atomic_number.mean - atom2.atomic_number.mean).abs();
        let atomic_uncertainty = (atom1.atomic_number.variance + atom2.atomic_number.variance).sqrt();
        
        // Compare charges
        let charge_diff = (atom1.charge.mean - atom2.charge.mean).abs();
        let charge_uncertainty = (atom1.charge.variance + atom2.charge.variance).sqrt();
        
        // Compare aromaticity
        let aromatic_diff = (atom1.aromaticity_probability - atom2.aromaticity_probability).abs();
        
        // Combine comparisons
        let similarity = if atomic_diff < 0.1 && charge_diff < 0.5 && aromatic_diff < 0.2 {
            0.9 - (atomic_diff + charge_diff * 0.1 + aromatic_diff)
        } else {
            0.1
        };
        
        let combined_uncertainty = (atomic_uncertainty + charge_uncertainty) * 0.5;
        
        ProbabilisticValue::new(similarity, combined_uncertainty)
    }

    /// Compare two probabilistic bonds
    fn compare_bonds(bond1: &ProbabilisticBond, bond2: &ProbabilisticBond) -> ProbabilisticValue {
        // Compare bond orders
        let order_diff = (bond1.bond_order.mean - bond2.bond_order.mean).abs();
        let order_uncertainty = (bond1.bond_order.variance + bond2.bond_order.variance).sqrt();
        
        // Compare bond type probabilities
        let mut type_similarity = 0.0;
        let mut type_count = 0;
        
        for (bond_type, prob1) in &bond1.bond_type_probabilities {
            if let Some(prob2) = bond2.bond_type_probabilities.get(bond_type) {
                type_similarity += 1.0 - (prob1 - prob2).abs();
                type_count += 1;
            }
        }
        
        let avg_type_similarity = if type_count > 0 {
            type_similarity / type_count as f64
        } else {
            0.0
        };
        
        // Combine comparisons
        let similarity = if order_diff < 0.2 {
            (0.8 - order_diff) * 0.7 + avg_type_similarity * 0.3
        } else {
            avg_type_similarity * 0.2
        };
        
        ProbabilisticValue::new(similarity, order_uncertainty)
    }

    /// Merge two probabilistic SMILES representations
    pub fn merge(
        smiles1: &ProbabilisticSMILES,
        smiles2: &ProbabilisticSMILES,
        weight1: f64,
        weight2: f64,
    ) -> BorgiaResult<ProbabilisticSMILES> {
        // This would implement a sophisticated merge operation
        // For now, return the first one with updated uncertainties
        let mut merged = smiles1.clone();
        
        // Increase uncertainty due to merging
        merged.molecular_uncertainty = ProbabilisticValue::new(
            merged.molecular_uncertainty.mean,
            merged.molecular_uncertainty.variance + 0.1,
        );
        
        merged.representation_confidence = ProbabilisticValue::new(
            merged.representation_confidence.mean * 0.9,
            merged.representation_confidence.variance + 0.05,
        );
        
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_smiles_parsing() {
        let mut parser = ProbabilisticSMILESParser::default();
        let result = parser.parse("CCO");
        
        assert!(result.is_ok());
        let prob_smiles = result.unwrap();
        assert_eq!(prob_smiles.probabilistic_atoms.len(), 3);
        assert_eq!(prob_smiles.probabilistic_bonds.len(), 2);
        assert!(prob_smiles.molecular_uncertainty.variance > 0.0);
    }

    #[test]
    fn test_probabilistic_smiles_generation() {
        let mut parser = ProbabilisticSMILESParser::default();
        let prob_smiles = parser.parse("c1ccccc1").unwrap();
        
        let generator = ProbabilisticSMILESGenerator::default();
        let result = generator.generate(&prob_smiles);
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("c1ccccc1"));
    }

    #[test]
    fn test_probabilistic_smiles_comparison() {
        let mut parser = ProbabilisticSMILESParser::default();
        let smiles1 = parser.parse("CCO").unwrap();
        let smiles2 = parser.parse("CCO").unwrap();
        
        let similarity = ProbabilisticSMILESUtils::compare(&smiles1, &smiles2).unwrap();
        assert!(similarity.mean > 0.8);
    }

    #[test]
    fn test_uncertainty_annotations() {
        let mut parser = ProbabilisticSMILESParser::new(0.2, 0.3, 0.4);
        let prob_smiles = parser.parse("CCO").unwrap();
        
        let generator = ProbabilisticSMILESGenerator::new(true, 0.1);
        let output = generator.generate(&prob_smiles).unwrap();
        
        assert!(output.contains("CCO"));
        // Should contain uncertainty annotations due to high uncertainty settings
        assert!(output.contains("UNCERTAINTY") || output.contains("CONFIDENCE"));
    }
}
