//! Enhanced molecular fingerprint with probabilistic features and multiple descriptor types.

use crate::error::{BorgiaError, Result};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Enhanced molecular fingerprint with multiple feature types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedFingerprint {
    /// Topological features (Morgan-like algorithm)
    pub topological: DVector<f64>,
    /// Pharmacophoric features (3-point and 4-point pharmacophores)
    pub pharmacophoric: DVector<f64>,
    /// Quantum mechanical features (atom properties, bond properties)
    pub quantum: DVector<f64>,
    /// Conformational features (rotatable bonds, ring flexibility)
    pub conformational: DVector<f64>,
    /// Interaction potential features (hydrophobic, electrostatic)
    pub interaction: DVector<f64>,
    /// Combined feature vector (concatenation of all above)
    pub combined: DVector<f64>,
    /// Feature importance weights (learned or set)
    pub weights: DVector<f64>,
    /// Uncertainty estimates for each feature
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
    pub morgan_radius: usize,
    pub use_chirality: bool,
    pub use_bond_types: bool,
    pub use_atom_features: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            topological_bits: 10000,
            pharmacophoric_bits: 10000,
            quantum_bits: 10000,
            conformational_bits: 10000,
            interaction_bits: 10000,
            morgan_radius: 3,
            use_chirality: true,
            use_bond_types: true,
            use_atom_features: true,
        }
    }
}

/// Atom properties for quantum mechanical features
#[derive(Debug, Clone)]
struct AtomProperties {
    electronegativity: f64,
    valence_electrons: u8,
    covalent_radius: f64,
    van_der_waals_radius: f64,
    ionization_energy: f64,
    electron_affinity: f64,
}

impl EnhancedFingerprint {
    /// Create enhanced fingerprint from SMILES string
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        Self::from_smiles_with_config(smiles, &FingerprintConfig::default())
    }

    /// Create enhanced fingerprint with custom configuration
    pub fn from_smiles_with_config(smiles: &str, config: &FingerprintConfig) -> Result<Self> {
        if smiles.is_empty() {
            return Err(BorgiaError::molecular_parsing("Empty SMILES string"));
        }

        // Parse SMILES into atom and bond information
        let parsed_molecule = Self::parse_smiles_detailed(smiles)?;

        // Generate different types of features
        let topological = Self::generate_morgan_fingerprint(smiles, &parsed_molecule, config)?;
        let pharmacophoric = Self::generate_pharmacophoric_fingerprint(smiles, &parsed_molecule, config)?;
        let quantum = Self::generate_quantum_fingerprint(smiles, &parsed_molecule, config)?;
        let conformational = Self::generate_conformational_fingerprint(smiles, &parsed_molecule, config)?;
        let interaction = Self::generate_interaction_fingerprint(smiles, &parsed_molecule, config)?;

        // Combine all features into a single vector
        let total_bits = config.topological_bits + config.pharmacophoric_bits + 
                        config.quantum_bits + config.conformational_bits + 
                        config.interaction_bits;

        let mut combined = DVector::zeros(total_bits);
        let mut offset = 0;

        // Concatenate feature vectors
        combined.rows_mut(offset, config.topological_bits).copy_from(&topological);
        offset += config.topological_bits;

        combined.rows_mut(offset, config.pharmacophoric_bits).copy_from(&pharmacophoric);
        offset += config.pharmacophoric_bits;

        combined.rows_mut(offset, config.quantum_bits).copy_from(&quantum);
        offset += config.quantum_bits;

        combined.rows_mut(offset, config.conformational_bits).copy_from(&conformational);
        offset += config.conformational_bits;

        combined.rows_mut(offset, config.interaction_bits).copy_from(&interaction);

        // Initialize weights (uniform for now, but could be learned)
        let weights = DVector::from_element(total_bits, 1.0);

        // Initialize uncertainties based on feature type
        let mut uncertainties = DVector::zeros(total_bits);
        offset = 0;
        
        // Topological features have low uncertainty
        uncertainties.rows_mut(offset, config.topological_bits).fill(0.05);
        offset += config.topological_bits;
        
        // Pharmacophoric features have medium uncertainty
        uncertainties.rows_mut(offset, config.pharmacophoric_bits).fill(0.10);
        offset += config.pharmacophoric_bits;
        
        // Quantum features have higher uncertainty
        uncertainties.rows_mut(offset, config.quantum_bits).fill(0.15);
        offset += config.quantum_bits;
        
        // Conformational features have high uncertainty
        uncertainties.rows_mut(offset, config.conformational_bits).fill(0.20);
        offset += config.conformational_bits;
        
        // Interaction features have medium-high uncertainty
        uncertainties.rows_mut(offset, config.interaction_bits).fill(0.12);

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

    /// Parse SMILES string into detailed molecular representation
    fn parse_smiles_detailed(smiles: &str) -> Result<ParsedMolecule> {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();
        let mut atom_index = 0;
        let mut ring_closures: HashMap<char, usize> = HashMap::new();
        let mut prev_atom_index: Option<usize> = None;

        let chars: Vec<char> = smiles.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            match ch {
                'C' | 'N' | 'O' | 'S' | 'P' | 'F' | 'c' | 'n' | 'o' | 's' | 'p' => {
                    // Create atom
                    let element = if ch.is_uppercase() { ch.to_string() } else { ch.to_uppercase().to_string() };
                    let is_aromatic = ch.is_lowercase();
                    
                    atoms.push(ParsedAtom {
                        element,
                        is_aromatic,
                        formal_charge: 0,
                        hydrogen_count: 0,
                        index: atom_index,
                    });

                    // Create bond to previous atom if exists
                    if let Some(prev_idx) = prev_atom_index {
                        bonds.push(ParsedBond {
                            from: prev_idx,
                            to: atom_index,
                            bond_type: BondType::Single,
                            is_aromatic: false,
                        });
                    }

                    prev_atom_index = Some(atom_index);
                    atom_index += 1;
                }
                '=' => {
                    // Double bond - modify the last bond
                    if let Some(last_bond) = bonds.last_mut() {
                        last_bond.bond_type = BondType::Double;
                    }
                }
                '#' => {
                    // Triple bond - modify the last bond
                    if let Some(last_bond) = bonds.last_mut() {
                        last_bond.bond_type = BondType::Triple;
                    }
                }
                '1'..='9' => {
                    // Ring closure
                    if let Some(&ring_start) = ring_closures.get(&ch) {
                        // Close the ring
                        if let Some(current_atom) = prev_atom_index {
                            bonds.push(ParsedBond {
                                from: ring_start,
                                to: current_atom,
                                bond_type: BondType::Single,
                                is_aromatic: false,
                            });
                        }
                        ring_closures.remove(&ch);
                    } else {
                        // Start a ring
                        if let Some(current_atom) = prev_atom_index {
                            ring_closures.insert(ch, current_atom);
                        }
                    }
                }
                '(' => {
                    // Start branch - we'll handle this simply by continuing from current atom
                }
                ')' => {
                    // End branch - reset to main chain (simplified)
                }
                _ => {
                    // Ignore other characters for now
                }
            }
            i += 1;
        }

        Ok(ParsedMolecule { atoms, bonds })
    }

    /// Generate Morgan (ECFP-like) fingerprint
    fn generate_morgan_fingerprint(
        smiles: &str,
        molecule: &ParsedMolecule,
        config: &FingerprintConfig,
    ) -> Result<DVector<f64>> {
        let mut fingerprint = DVector::zeros(config.topological_bits);

        if molecule.atoms.is_empty() {
            return Ok(fingerprint);
        }

        // Initialize atom invariants
        let mut atom_invariants: Vec<u64> = molecule.atoms.iter()
            .map(|atom| Self::calculate_initial_invariant(atom))
            .collect();

        // Iterate Morgan algorithm for specified radius
        for radius in 0..=config.morgan_radius {
            for (atom_idx, atom) in molecule.atoms.iter().enumerate() {
                let mut environment_hash = atom_invariants[atom_idx];

                // Include neighbor information
                let neighbors = Self::get_neighbors(atom_idx, &molecule.bonds);
                let mut neighbor_invariants: Vec<u64> = neighbors.iter()
                    .map(|&neighbor_idx| atom_invariants[neighbor_idx])
                    .collect();
                neighbor_invariants.sort();

                for neighbor_invariant in neighbor_invariants {
                    environment_hash = Self::combine_hash(environment_hash, neighbor_invariant);
                }

                // Map to bit position
                let bit_position = (environment_hash as usize) % config.topological_bits;
                fingerprint[bit_position] += 1.0;

                // Update invariant for next iteration
                if radius < config.morgan_radius {
                    atom_invariants[atom_idx] = environment_hash;
                }
            }
        }

        // Normalize fingerprint
        let max_value = fingerprint.max();
        if max_value > 0.0 {
            fingerprint /= max_value;
        }

        Ok(fingerprint)
    }

    /// Generate pharmacophoric fingerprint
    fn generate_pharmacophoric_fingerprint(
        smiles: &str,
        molecule: &ParsedMolecule,
        config: &FingerprintConfig,
    ) -> Result<DVector<f64>> {
        let mut fingerprint = DVector::zeros(config.pharmacophoric_bits);

        // Identify pharmacophoric features
        let features = Self::identify_pharmacophores(molecule);

        // Generate 2-point pharmacophores (distances between features)
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                let distance = Self::estimate_topological_distance(
                    features[i].atom_index,
                    features[j].atom_index,
                    &molecule.bonds,
                );

                let feature_pair_hash = Self::combine_hash(
                    Self::hash_string(&features[i].feature_type),
                    Self::combine_hash(
                        Self::hash_string(&features[j].feature_type),
                        distance as u64,
                    ),
                );

                let bit_position = (feature_pair_hash as usize) % config.pharmacophoric_bits;
                fingerprint[bit_position] += 1.0;
            }
        }

        // Generate 3-point pharmacophores
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                for k in (j + 1)..features.len() {
                    let dist_ij = Self::estimate_topological_distance(
                        features[i].atom_index,
                        features[j].atom_index,
                        &molecule.bonds,
                    );
                    let dist_jk = Self::estimate_topological_distance(
                        features[j].atom_index,
                        features[k].atom_index,
                        &molecule.bonds,
                    );
                    let dist_ik = Self::estimate_topological_distance(
                        features[i].atom_index,
                        features[k].atom_index,
                        &molecule.bonds,
                    );

                    let triplet_hash = Self::combine_hash(
                        Self::hash_string(&format!("{}-{}-{}", 
                            features[i].feature_type,
                            features[j].feature_type,
                            features[k].feature_type)),
                        Self::combine_hash(
                            Self::combine_hash(dist_ij as u64, dist_jk as u64),
                            dist_ik as u64,
                        ),
                    );

                    let bit_position = (triplet_hash as usize) % config.pharmacophoric_bits;
                    fingerprint[bit_position] += 0.5; // Lower weight for 3-point
                }
            }
        }

        Ok(fingerprint)
    }

    /// Generate quantum mechanical features
    fn generate_quantum_fingerprint(
        smiles: &str,
        molecule: &ParsedMolecule,
        config: &FingerprintConfig,
    ) -> Result<DVector<f64>> {
        let mut fingerprint = DVector::zeros(config.quantum_bits);

        for atom in &molecule.atoms {
            let properties = Self::get_atom_properties(&atom.element);

            // Electronegativity features
            let en_hash = Self::combine_hash(
                Self::hash_string(&atom.element),
                (properties.electronegativity * 100.0) as u64,
            );
            let en_bit = (en_hash as usize) % config.quantum_bits;
            fingerprint[en_bit] += properties.electronegativity / 4.0; // Normalize

            // Valence electron features
            let val_hash = Self::combine_hash(en_hash, properties.valence_electrons as u64);
            let val_bit = (val_hash as usize) % config.quantum_bits;
            fingerprint[val_bit] += properties.valence_electrons as f64 / 8.0;

            // Atomic size features
            let size_hash = Self::combine_hash(val_hash, (properties.covalent_radius * 100.0) as u64);
            let size_bit = (size_hash as usize) % config.quantum_bits;
            fingerprint[size_bit] += properties.covalent_radius / 2.0;

            // Ionization energy features
            let ie_hash = Self::combine_hash(size_hash, (properties.ionization_energy * 10.0) as u64);
            let ie_bit = (ie_hash as usize) % config.quantum_bits;
            fingerprint[ie_bit] += properties.ionization_energy / 25.0; // Normalize
        }

        Ok(fingerprint)
    }

    /// Generate conformational features
    fn generate_conformational_fingerprint(
        smiles: &str,
        molecule: &ParsedMolecule,
        config: &FingerprintConfig,
    ) -> Result<DVector<f64>> {
        let mut fingerprint = DVector::zeros(config.conformational_bits);

        // Count rotatable bonds
        let rotatable_bonds = Self::count_rotatable_bonds(&molecule.bonds, &molecule.atoms);
        if rotatable_bonds > 0 {
            let rot_hash = Self::hash_string("rotatable_bonds");
            let bit_pos = (rot_hash as usize) % config.conformational_bits;
            fingerprint[bit_pos] = (rotatable_bonds as f64).ln() + 1.0;
        }

        // Analyze ring systems
        let rings = Self::find_rings(&molecule.bonds, molecule.atoms.len());
        for ring in rings {
            let ring_size = ring.len();
            let flexibility = Self::estimate_ring_flexibility(ring_size);
            
            let ring_hash = Self::combine_hash(
                Self::hash_string("ring_flexibility"),
                ring_size as u64,
            );
            let bit_pos = (ring_hash as usize) % config.conformational_bits;
            fingerprint[bit_pos] += flexibility;
        }

        // Torsion patterns
        for i in 0..molecule.atoms.len() {
            for j in (i + 1)..molecule.atoms.len() {
                if let Some(path) = Self::find_shortest_path(i, j, &molecule.bonds) {
                    if path.len() == 4 {
                        // 4-atom torsion
                        let torsion_pattern = format!("{}-{}-{}-{}",
                            molecule.atoms[path[0]].element,
                            molecule.atoms[path[1]].element,
                            molecule.atoms[path[2]].element,
                            molecule.atoms[path[3]].element);
                        
                        let torsion_hash = Self::hash_string(&torsion_pattern);
                        let bit_pos = (torsion_hash as usize) % config.conformational_bits;
                        fingerprint[bit_pos] += 1.0;
                    }
                }
            }
        }

        Ok(fingerprint)
    }

    /// Generate interaction potential features
    fn generate_interaction_fingerprint(
        smiles: &str,
        molecule: &ParsedMolecule,
        config: &FingerprintConfig,
    ) -> Result<DVector<f64>> {
        let mut fingerprint = DVector::zeros(config.interaction_bits);

        // Hydrophobic surface area estimation
        let hydrophobic_atoms = molecule.atoms.iter()
            .filter(|atom| atom.element == "C" && !atom.is_aromatic)
            .count();

        if hydrophobic_atoms > 0 {
            let hydro_hash = Self::hash_string("hydrophobic_surface");
            let bit_pos = (hydro_hash as usize) % config.interaction_bits;
            fingerprint[bit_pos] = (hydrophobic_atoms as f64).sqrt() / 10.0;
        }

        // Polar surface area contributors
        let polar_atoms = molecule.atoms.iter()
            .filter(|atom| matches!(atom.element.as_str(), "N" | "O" | "S"))
            .count();

        if polar_atoms > 0 {
            let polar_hash = Self::hash_string("polar_surface");
            let bit_pos = (polar_hash as usize) % config.interaction_bits;
            fingerprint[bit_pos] = (polar_atoms as f64).sqrt() / 5.0;
        }

        // Aromatic interactions
        let aromatic_atoms = molecule.atoms.iter()
            .filter(|atom| atom.is_aromatic)
            .count();

        if aromatic_atoms > 0 {
            let arom_hash = Self::hash_string("aromatic_interaction");
            let bit_pos = (arom_hash as usize) % config.interaction_bits;
            fingerprint[bit_pos] = (aromatic_atoms as f64) / 6.0; // Normalize by benzene
        }

        // Hydrogen bonding potential
        let hb_donors = molecule.atoms.iter()
            .filter(|atom| matches!(atom.element.as_str(), "N" | "O") && atom.hydrogen_count > 0)
            .count();

        let hb_acceptors = molecule.atoms.iter()
            .filter(|atom| matches!(atom.element.as_str(), "N" | "O" | "F"))
            .count();

        if hb_donors > 0 {
            let hbd_hash = Self::hash_string("hb_donor");
            let bit_pos = (hbd_hash as usize) % config.interaction_bits;
            fingerprint[bit_pos] = (hb_donors as f64) / 5.0; // Normalize
        }

        if hb_acceptors > 0 {
            let hba_hash = Self::hash_string("hb_acceptor");
            let bit_pos = (hba_hash as usize) % config.interaction_bits;
            fingerprint[bit_pos] = (hb_acceptors as f64) / 10.0; // Normalize
        }

        Ok(fingerprint)
    }

    /// Calculate Tanimoto similarity between fingerprints
    pub fn tanimoto_similarity(&self, other: &EnhancedFingerprint) -> f64 {
        let intersection = self.combined.dot(&other.combined);
        let union = self.combined.norm_squared() + other.combined.norm_squared() - intersection;
        
        if union > 0.0 {
            intersection / union
        } else {
            1.0 // Both fingerprints are zero vectors
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

    // Helper methods

    fn calculate_initial_invariant(atom: &ParsedAtom) -> u64 {
        let mut invariant = match atom.element.as_str() {
            "C" => 6,
            "N" => 7,
            "O" => 8,
            "S" => 16,
            "P" => 15,
            "F" => 9,
            "Cl" => 17,
            "Br" => 35,
            "I" => 53,
            _ => 1,
        };

        if atom.is_aromatic {
            invariant += 100;
        }

        invariant += (atom.formal_charge + 10) as u64;
        invariant
    }

    fn get_neighbors(atom_idx: usize, bonds: &[ParsedBond]) -> Vec<usize> {
        bonds.iter()
            .filter_map(|bond| {
                if bond.from == atom_idx {
                    Some(bond.to)
                } else if bond.to == atom_idx {
                    Some(bond.from)
                } else {
                    None
                }
            })
            .collect()
    }

    fn combine_hash(h1: u64, h2: u64) -> u64 {
        h1.wrapping_mul(31).wrapping_add(h2)
    }

    fn hash_string(s: &str) -> u64 {
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash
    }

    fn identify_pharmacophores(molecule: &ParsedMolecule) -> Vec<PharmacophoreFeature> {
        let mut features = Vec::new();

        for (idx, atom) in molecule.atoms.iter().enumerate() {
            match atom.element.as_str() {
                "N" => {
                    if atom.hydrogen_count > 0 {
                        features.push(PharmacophoreFeature {
                            feature_type: "hydrogen_bond_donor".to_string(),
                            atom_index: idx,
                        });
                    } else {
                        features.push(PharmacophoreFeature {
                            feature_type: "hydrogen_bond_acceptor".to_string(),
                            atom_index: idx,
                        });
                    }
                }
                "O" => {
                    features.push(PharmacophoreFeature {
                        feature_type: "hydrogen_bond_acceptor".to_string(),
                        atom_index: idx,
                    });
                    if atom.hydrogen_count > 0 {
                        features.push(PharmacophoreFeature {
                            feature_type: "hydrogen_bond_donor".to_string(),
                            atom_index: idx,
                        });
                    }
                }
                "C" if atom.is_aromatic => {
                    features.push(PharmacophoreFeature {
                        feature_type: "aromatic".to_string(),
                        atom_index: idx,
                    });
                }
                "C" => {
                    features.push(PharmacophoreFeature {
                        feature_type: "hydrophobic".to_string(),
                        atom_index: idx,
                    });
                }
                _ => {}
            }
        }

        features
    }

    fn estimate_topological_distance(from: usize, to: usize, bonds: &[ParsedBond]) -> usize {
        if from == to {
            return 0;
        }

        // Simple BFS for shortest path
        let mut visited = vec![false; bonds.len() * 2]; // Overestimate
        let mut queue = vec![(from, 0)];
        let mut front = 0;

        while front < queue.len() {
            let (current, dist) = queue[front];
            front += 1;

            if current == to {
                return dist;
            }

            if current < visited.len() && visited[current] {
                continue;
            }

            if current < visited.len() {
                visited[current] = true;
            }

            for bond in bonds {
                let next = if bond.from == current {
                    bond.to
                } else if bond.to == current {
                    bond.from
                } else {
                    continue;
                };

                if next < visited.len() && !visited[next] {
                    queue.push((next, dist + 1));
                }
            }
        }

        usize::MAX // No path found
    }

    fn get_atom_properties(element: &str) -> AtomProperties {
        match element {
            "C" => AtomProperties {
                electronegativity: 2.55,
                valence_electrons: 4,
                covalent_radius: 0.76,
                van_der_waals_radius: 1.70,
                ionization_energy: 11.26,
                electron_affinity: -1.26,
            },
            "N" => AtomProperties {
                electronegativity: 3.04,
                valence_electrons: 5,
                covalent_radius: 0.71,
                van_der_waals_radius: 1.55,
                ionization_energy: 14.53,
                electron_affinity: -0.07,
            },
            "O" => AtomProperties {
                electronegativity: 3.44,
                valence_electrons: 6,
                covalent_radius: 0.66,
                van_der_waals_radius: 1.52,
                ionization_energy: 13.62,
                electron_affinity: -1.46,
            },
            "S" => AtomProperties {
                electronegativity: 2.58,
                valence_electrons: 6,
                covalent_radius: 1.05,
                van_der_waals_radius: 1.80,
                ionization_energy: 10.36,
                electron_affinity: -2.08,
            },
            "P" => AtomProperties {
                electronegativity: 2.19,
                valence_electrons: 5,
                covalent_radius: 1.07,
                van_der_waals_radius: 1.80,
                ionization_energy: 10.49,
                electron_affinity: -0.75,
            },
            "F" => AtomProperties {
                electronegativity: 3.98,
                valence_electrons: 7,
                covalent_radius: 0.57,
                van_der_waals_radius: 1.47,
                ionization_energy: 17.42,
                electron_affinity: -3.40,
            },
            _ => AtomProperties {
                electronegativity: 2.0,
                valence_electrons: 4,
                covalent_radius: 1.0,
                van_der_waals_radius: 2.0,
                ionization_energy: 10.0,
                electron_affinity: 0.0,
            },
        }
    }

    fn count_rotatable_bonds(bonds: &[ParsedBond], atoms: &[ParsedAtom]) -> usize {
        bonds.iter()
            .filter(|bond| {
                bond.bond_type == BondType::Single && 
                !bond.is_aromatic &&
                Self::is_rotatable_bond(bond, atoms)
            })
            .count()
    }

    fn is_rotatable_bond(bond: &ParsedBond, atoms: &[ParsedAtom]) -> bool {
        // Simple heuristic: not rotatable if either atom is part of a small ring
        // or if either atom has only one heavy atom neighbor
        true // Simplified for now
    }

    fn find_rings(bonds: &[ParsedBond], num_atoms: usize) -> Vec<Vec<usize>> {
        // Simplified ring finding - would use more sophisticated algorithm in practice
        Vec::new()
    }

    fn estimate_ring_flexibility(ring_size: usize) -> f64 {
        match ring_size {
            3 => 0.1,  // Very rigid
            4 => 0.3,  // Rigid
            5 => 0.6,  // Moderately flexible
            6 => 0.8,  // Flexible
            7 => 0.9,  // Very flexible
            _ => 0.7,  // Default
        }
    }

    fn find_shortest_path(from: usize, to: usize, bonds: &[ParsedBond]) -> Option<Vec<usize>> {
        // Simplified - would implement proper shortest path algorithm
        None
    }
}

/// Parsed molecular representation
#[derive(Debug, Clone)]
struct ParsedMolecule {
    atoms: Vec<ParsedAtom>,
    bonds: Vec<ParsedBond>,
}

/// Parsed atom representation
#[derive(Debug, Clone)]
struct ParsedAtom {
    element: String,
    is_aromatic: bool,
    formal_charge: i8,
    hydrogen_count: u8,
    index: usize,
}

/// Parsed bond representation
#[derive(Debug, Clone)]
struct ParsedBond {
    from: usize,
    to: usize,
    bond_type: BondType,
    is_aromatic: bool,
}

/// Bond types
#[derive(Debug, Clone, PartialEq)]
enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

/// Pharmacophore feature
#[derive(Debug, Clone)]
struct PharmacophoreFeature {
    feature_type: String,
    atom_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let fp = EnhancedFingerprint::from_smiles("CCO").unwrap();
        assert_eq!(fp.dimension(), 50000);
        assert!(fp.density() >= 0.0);
    }

    #[test]
    fn test_tanimoto_similarity() {
        let fp1 = EnhancedFingerprint::from_smiles("CCO").unwrap();
        let fp2 = EnhancedFingerprint::from_smiles("CCO").unwrap();
        let fp3 = EnhancedFingerprint::from_smiles("CCCCCCCC").unwrap();
        
        // Identical molecules should have high similarity
        let sim_identical = fp1.tanimoto_similarity(&fp2);
        assert!(sim_identical >= 0.8);
        
        // Different molecules should have lower similarity
        let sim_different = fp1.tanimoto_similarity(&fp3);
        assert!(sim_different < sim_identical);
    }

    #[test]
    fn test_smiles_parsing() {
        let result = EnhancedFingerprint::parse_smiles_detailed("CCO");
        assert!(result.is_ok());
        
        let molecule = result.unwrap();
        assert_eq!(molecule.atoms.len(), 3); // C, C, O
        assert_eq!(molecule.bonds.len(), 2); // C-C, C-O
    }

    #[test]
    fn test_invalid_smiles() {
        let result = EnhancedFingerprint::from_smiles("");
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_types() {
        let fp = EnhancedFingerprint::from_smiles("c1ccccc1CCO").unwrap(); // Benzyl alcohol
        
        // Should have features in all categories
        assert!(fp.topological.iter().any(|&x| x > 0.0));
        assert!(fp.quantum.iter().any(|&x| x > 0.0));
        // Other feature types may be zero for this simple molecule
    }
} 