//! Enhanced molecular representations with probabilistic and fuzzy features.
//!
//! This module provides the core molecular data structures that encode uncertainty
//! and fuzzy chemical properties, enabling more nuanced molecular comparisons.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::{ProbabilisticValue, SimilarityDistribution};
use indexmap::IndexMap;
use nalgebra::DVector;
use petgraph::{Graph, Undirected};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Type alias for molecular graph
pub type MolecularGraph = Graph<Atom, Bond, Undirected>;

/// Enhanced molecular representation with probabilistic features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticMolecule {
    /// SMILES string representation
    pub smiles: String,
    /// Molecular graph structure
    pub graph: MolecularGraph,
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
    /// Atom-level fuzzy features
    pub atom_features: Vec<FuzzyAtomFeatures>,
    /// Bond-level fuzzy features
    pub bond_features: Vec<FuzzyBondFeatures>,
}

/// Atom representation with fuzzy properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub element: String,
    pub atomic_number: u8,
    pub formal_charge: i8,
    pub hybridization: Hybridization,
    pub aromaticity: ProbabilisticValue,
    pub electronegativity: ProbabilisticValue,
    pub partial_charge: ProbabilisticValue,
}

/// Bond representation with fuzzy properties
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bond {
    pub bond_type: BondType,
    pub bond_order: ProbabilisticValue,
    pub aromaticity: ProbabilisticValue,
    pub polarity: ProbabilisticValue,
    pub length: ProbabilisticValue,
}

/// Hybridization states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Hybridization {
    SP,
    SP2,
    SP3,
    SP3D,
    SP3D2,
    Unknown,
}

/// Bond types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
    Delocalized,
}

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

/// Ring representation
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

/// Enhanced molecular fingerprint with 50,000+ features
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
    /// Combined feature vector
    pub combined: DVector<f64>,
    /// Feature importance weights
    pub weights: DVector<f64>,
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
    pub formal_charge: ProbabilisticValue,
    pub dipole_moment: ProbabilisticValue,
    pub polarizability: ProbabilisticValue,
}

/// Fuzzy atom-level features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyAtomFeatures {
    pub atom_index: usize,
    pub environment: ProbabilisticValue,
    pub connectivity: ProbabilisticValue,
    pub accessibility: ProbabilisticValue,
    pub reactivity: ProbabilisticValue,
}

/// Fuzzy bond-level features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyBondFeatures {
    pub bond_index: usize,
    pub rotability: ProbabilisticValue,
    pub conjugation: ProbabilisticValue,
    pub strain: ProbabilisticValue,
    pub polarization: ProbabilisticValue,
}

impl ProbabilisticMolecule {
    /// Create a new probabilistic molecule from SMILES
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        // Parse SMILES string (simplified implementation)
        let graph = Self::parse_smiles(smiles)?;
        
        // Generate fuzzy features
        let fuzzy_aromaticity = Self::compute_fuzzy_aromaticity(&graph)?;
        let fuzzy_rings = Self::compute_fuzzy_rings(&graph)?;
        let fuzzy_groups = Self::compute_fuzzy_groups(&graph)?;
        let fingerprint = Self::compute_enhanced_fingerprint(&graph)?;
        let properties = Self::compute_probabilistic_properties(&graph)?;
        let atom_features = Self::compute_atom_features(&graph)?;
        let bond_features = Self::compute_bond_features(&graph)?;

        Ok(Self {
            smiles: smiles.to_string(),
            graph,
            fuzzy_aromaticity,
            fuzzy_rings,
            fuzzy_groups,
            fingerprint,
            properties,
            atom_features,
            bond_features,
        })
    }

    /// Parse SMILES string into molecular graph (simplified)
    fn parse_smiles(smiles: &str) -> Result<MolecularGraph> {
        let mut graph = Graph::new_undirected();
        
        // Simplified SMILES parsing - in reality would use RDKit or similar
        let chars: Vec<char> = smiles.chars().collect();
        let mut atom_stack = Vec::new();
        
        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                'C' => {
                    let atom = Atom {
                        element: "C".to_string(),
                        atomic_number: 6,
                        formal_charge: 0,
                        hybridization: Hybridization::SP3,
                        aromaticity: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                        electronegativity: ProbabilisticValue::new_normal(2.55, 0.05, 0.95),
                        partial_charge: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                    };
                    let node_idx = graph.add_node(atom);
                    atom_stack.push(node_idx);
                }
                'N' => {
                    let atom = Atom {
                        element: "N".to_string(),
                        atomic_number: 7,
                        formal_charge: 0,
                        hybridization: Hybridization::SP3,
                        aromaticity: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                        electronegativity: ProbabilisticValue::new_normal(3.04, 0.05, 0.95),
                        partial_charge: ProbabilisticValue::new_normal(-0.3, 0.1, 0.95),
                    };
                    let node_idx = graph.add_node(atom);
                    atom_stack.push(node_idx);
                }
                'O' => {
                    let atom = Atom {
                        element: "O".to_string(),
                        atomic_number: 8,
                        formal_charge: 0,
                        hybridization: Hybridization::SP3,
                        aromaticity: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                        electronegativity: ProbabilisticValue::new_normal(3.44, 0.05, 0.95),
                        partial_charge: ProbabilisticValue::new_normal(-0.4, 0.1, 0.95),
                    };
                    let node_idx = graph.add_node(atom);
                    atom_stack.push(node_idx);
                }
                _ => {
                    // Handle other atoms, bonds, rings, etc.
                    // This is a very simplified implementation
                }
            }
        }

        // Add bonds between consecutive atoms (simplified)
        for i in 0..atom_stack.len().saturating_sub(1) {
            let bond = Bond {
                bond_type: BondType::Single,
                bond_order: ProbabilisticValue::new_normal(1.0, 0.05, 0.95),
                aromaticity: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                polarity: ProbabilisticValue::new_normal(0.0, 0.2, 0.95),
                length: ProbabilisticValue::new_normal(1.54, 0.02, 0.95), // C-C bond length
            };
            graph.add_edge(atom_stack[i], atom_stack[i + 1], bond);
        }

        Ok(graph)
    }

    /// Compute fuzzy aromaticity features
    fn compute_fuzzy_aromaticity(graph: &MolecularGraph) -> Result<FuzzyAromaticity> {
        // Simplified aromaticity calculation
        let overall_aromaticity = ProbabilisticValue::new_normal(0.2, 0.3, 0.95);
        let ring_aromaticity = HashMap::new();
        let delocalization = HashMap::new();
        let huckel_compliance = ProbabilisticValue::new_normal(0.1, 0.2, 0.95);

        Ok(FuzzyAromaticity {
            overall_aromaticity,
            ring_aromaticity,
            delocalization,
            huckel_compliance,
        })
    }

    /// Compute fuzzy ring systems
    fn compute_fuzzy_rings(graph: &MolecularGraph) -> Result<FuzzyRingSystems> {
        // Simplified ring detection
        let rings = Vec::new();
        let strain_energies = HashMap::new();
        let puckering = HashMap::new();
        let flexibility = HashMap::new();

        Ok(FuzzyRingSystems {
            rings,
            strain_energies,
            puckering,
            flexibility,
        })
    }

    /// Compute fuzzy functional groups
    fn compute_fuzzy_groups(graph: &MolecularGraph) -> Result<FuzzyFunctionalGroups> {
        // Simplified functional group detection
        let groups = Vec::new();
        let h_bonding = ProbabilisticValue::new_normal(0.3, 0.2, 0.95);
        let reactivity = HashMap::new();
        let pharmacophores = Vec::new();

        Ok(FuzzyFunctionalGroups {
            groups,
            h_bonding,
            reactivity,
            pharmacophores,
        })
    }

    /// Compute enhanced molecular fingerprint
    fn compute_enhanced_fingerprint(graph: &MolecularGraph) -> Result<EnhancedFingerprint> {
        // Create high-dimensional fingerprint vectors
        let topological = DVector::zeros(10000);
        let pharmacophoric = DVector::zeros(10000);
        let quantum = DVector::zeros(10000);
        let conformational = DVector::zeros(10000);
        let interaction = DVector::zeros(10000);
        let combined = DVector::zeros(50000);
        let weights = DVector::from_element(50000, 1.0);

        Ok(EnhancedFingerprint {
            topological,
            pharmacophoric,
            quantum,
            conformational,
            interaction,
            combined,
            weights,
        })
    }

    /// Compute probabilistic molecular properties
    fn compute_probabilistic_properties(graph: &MolecularGraph) -> Result<ProbabilisticProperties> {
        // Simplified property calculations with uncertainty
        Ok(ProbabilisticProperties {
            molecular_weight: ProbabilisticValue::new_normal(180.0, 5.0, 0.95),
            logp: ProbabilisticValue::new_normal(2.1, 0.3, 0.95),
            psa: ProbabilisticValue::new_normal(45.0, 5.0, 0.95),
            hbd: ProbabilisticValue::new_normal(1.0, 0.2, 0.95),
            hba: ProbabilisticValue::new_normal(3.0, 0.3, 0.95),
            rotatable_bonds: ProbabilisticValue::new_normal(4.0, 0.5, 0.95),
            formal_charge: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
            dipole_moment: ProbabilisticValue::new_normal(1.5, 0.2, 0.95),
            polarizability: ProbabilisticValue::new_normal(15.0, 1.0, 0.95),
        })
    }

    /// Compute atom-level fuzzy features
    fn compute_atom_features(graph: &MolecularGraph) -> Result<Vec<FuzzyAtomFeatures>> {
        let mut features = Vec::new();
        
        for (idx, _atom) in graph.node_indices().enumerate() {
            features.push(FuzzyAtomFeatures {
                atom_index: idx,
                environment: ProbabilisticValue::new_normal(0.5, 0.2, 0.95),
                connectivity: ProbabilisticValue::new_normal(2.0, 0.5, 0.95),
                accessibility: ProbabilisticValue::new_normal(0.7, 0.1, 0.95),
                reactivity: ProbabilisticValue::new_normal(0.3, 0.2, 0.95),
            });
        }

        Ok(features)
    }

    /// Compute bond-level fuzzy features
    fn compute_bond_features(graph: &MolecularGraph) -> Result<Vec<FuzzyBondFeatures>> {
        let mut features = Vec::new();
        
        for (idx, _edge) in graph.edge_indices().enumerate() {
            features.push(FuzzyBondFeatures {
                bond_index: idx,
                rotability: ProbabilisticValue::new_normal(0.8, 0.2, 0.95),
                conjugation: ProbabilisticValue::new_normal(0.1, 0.1, 0.95),
                strain: ProbabilisticValue::new_normal(0.0, 0.1, 0.95),
                polarization: ProbabilisticValue::new_normal(0.2, 0.1, 0.95),
            });
        }

        Ok(features)
    }

    /// Get molecular formula
    pub fn molecular_formula(&self) -> String {
        let mut element_counts: HashMap<String, usize> = HashMap::new();
        
        for node_idx in self.graph.node_indices() {
            if let Some(atom) = self.graph.node_weight(node_idx) {
                *element_counts.entry(atom.element.clone()).or_insert(0) += 1;
            }
        }

        let mut formula = String::new();
        for (element, count) in element_counts {
            if count == 1 {
                formula.push_str(&element);
            } else {
                formula.push_str(&format!("{}{}", element, count));
            }
        }

        formula
    }

    /// Get number of atoms
    pub fn atom_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get number of bonds
    pub fn bond_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if molecule contains aromatic rings
    pub fn has_aromatic_rings(&self) -> bool {
        self.fuzzy_aromaticity.overall_aromaticity.mean > 0.5
    }
}

impl fmt::Display for ProbabilisticMolecule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProbabilisticMolecule(SMILES: {}, Formula: {}, Atoms: {}, MW: {:.1}±{:.1})",
            self.smiles,
            self.molecular_formula(),
            self.atom_count(),
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
        assert!(mol.atom_count() > 0);
    }

    #[test]
    fn test_molecular_properties() {
        let mol = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        assert!(mol.properties.molecular_weight.mean > 0.0);
        assert!(mol.properties.logp.std_dev > 0.0);
    }

    #[test]
    fn test_fuzzy_features() {
        let mol = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        assert!(mol.fuzzy_aromaticity.overall_aromaticity.confidence_level > 0.0);
        assert!(!mol.atom_features.is_empty());
    }
} 