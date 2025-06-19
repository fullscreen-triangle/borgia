// =====================================================================================
// ENTROPY AS TANGIBLE DISTRIBUTION
// Revolutionary concept: entropy is tangible - it's where oscillations "land"
// =====================================================================================

use ndarray::{Array1, Array2, Array3};
use serde::{Serialize, Deserialize};

/// Entropy as statistical distribution of oscillation endpoints
/// Revolutionary concept: entropy is tangible - it's where oscillations "land"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntropyDistribution {
    /// Molecular configurations where oscillations settle
    pub configuration_endpoints: Vec<MolecularConfiguration>,
    
    /// Probability of landing at each endpoint
    pub landing_probabilities: Array1<f64>,
    
    /// Thermodynamic accessibility of each endpoint
    pub thermodynamic_accessibility: Array1<f64>,
    
    /// Patterns of how oscillations decay toward endpoints
    pub oscillation_decay_patterns: Vec<DecayPattern>,
    
    /// Clustering analysis of endpoint distributions
    pub endpoint_clustering: ClusteringAnalysis,
    
    /// Time evolution of endpoint probabilities
    pub temporal_evolution: Vec<Array1<f64>>,
}

/// Specific molecular configuration representing an oscillation endpoint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MolecularConfiguration {
    pub atom_positions: Vec<(f64, f64, f64)>,
    pub bond_lengths: Vec<f64>,
    pub bond_angles: Vec<f64>,
    pub dihedral_angles: Vec<f64>,
    pub electronic_state: ElectronicState,
    pub vibrational_modes: Vec<VibrationalMode>,
    pub energy: f64,
    pub stability_score: f64,
}

/// Electronic state information for quantum mechanical analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElectronicState {
    pub orbital_occupancies: Vec<f64>,
    pub spin_multiplicities: Vec<f64>,
    pub dipole_moment: (f64, f64, f64),
    pub polarizability: Array2<f64>,
    pub electron_density_distribution: Array3<f64>,
}

/// Vibrational mode information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VibrationalMode {
    pub frequency: f64,
    pub intensity: f64,
    pub displacement_vectors: Vec<(f64, f64, f64)>,
    pub quantum_number: u32,
}

/// Pattern of oscillation decay toward equilibrium
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayPattern {
    pub decay_constant: f64,
    pub oscillation_frequency: f64,
    pub phase_shift: f64,
    pub amplitude_modulation: Vec<f64>,
    pub pathway_atoms: Vec<usize>,
}

/// Analysis of how oscillation endpoints cluster in configuration space
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusteringAnalysis {
    pub cluster_centers: Vec<MolecularConfiguration>,
    pub cluster_assignments: Vec<usize>,
    pub cluster_probabilities: Array1<f64>,
    pub inter_cluster_transitions: Array2<f64>,
    pub cluster_stability_metrics: Vec<f64>,
}

impl EntropyDistribution {
    /// Create new entropy distribution with default parameters
    pub fn new(num_endpoints: usize) -> Self {
        Self {
            configuration_endpoints: Vec::with_capacity(num_endpoints),
            landing_probabilities: Array1::from_vec(vec![1.0 / num_endpoints as f64; num_endpoints]),
            thermodynamic_accessibility: Array1::from_vec(vec![1.0; num_endpoints]),
            oscillation_decay_patterns: Vec::new(),
            endpoint_clustering: ClusteringAnalysis::new(3), // Default 3 clusters
            temporal_evolution: Vec::new(),
        }
    }
    
    /// Calculate Shannon entropy of the endpoint distribution
    pub fn shannon_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for &prob in self.landing_probabilities.iter() {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }
    
    /// Calculate Boltzmann entropy using endpoint energies
    pub fn boltzmann_entropy(&self, temperature: f64) -> f64 {
        if self.configuration_endpoints.is_empty() {
            return 0.0;
        }
        
        let kb = 1.380649e-23; // Boltzmann constant
        let beta = 1.0 / (kb * temperature);
        
        // Calculate partition function
        let mut partition_function = 0.0;
        for config in &self.configuration_endpoints {
            partition_function += (-beta * config.energy).exp();
        }
        
        // Calculate entropy
        let mut entropy = 0.0;
        for config in &self.configuration_endpoints {
            let prob = (-beta * config.energy).exp() / partition_function;
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        
        entropy * kb
    }
    
    /// Add new molecular configuration endpoint
    pub fn add_configuration_endpoint(&mut self, config: MolecularConfiguration, probability: f64) {
        self.configuration_endpoints.push(config);
        
        // Expand probability arrays
        let new_size = self.configuration_endpoints.len();
        let mut new_probs = vec![0.0; new_size];
        let mut new_access = vec![1.0; new_size];
        
        // Copy existing values
        for (i, &prob) in self.landing_probabilities.iter().enumerate() {
            if i < new_size - 1 {
                new_probs[i] = prob;
            }
        }
        
        for (i, &access) in self.thermodynamic_accessibility.iter().enumerate() {
            if i < new_size - 1 {
                new_access[i] = access;
            }
        }
        
        // Add new probability
        new_probs[new_size - 1] = probability;
        
        // Renormalize probabilities
        let total: f64 = new_probs.iter().sum();
        if total > 0.0 {
            for prob in &mut new_probs {
                *prob /= total;
            }
        }
        
        self.landing_probabilities = Array1::from_vec(new_probs);
        self.thermodynamic_accessibility = Array1::from_vec(new_access);
    }
    
    /// Calculate most probable landing configuration
    pub fn most_probable_endpoint(&self) -> Option<&MolecularConfiguration> {
        if self.configuration_endpoints.is_empty() {
            return None;
        }
        
        let max_idx = self.landing_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)?;
        
        self.configuration_endpoints.get(max_idx)
    }
    
    /// Calculate effective number of accessible states
    pub fn effective_accessible_states(&self) -> f64 {
        let entropy = self.shannon_entropy();
        entropy.exp()
    }
    
    /// Update temporal evolution with new distribution
    pub fn update_temporal_evolution(&mut self) {
        self.temporal_evolution.push(self.landing_probabilities.clone());
        
        // Keep only recent history (last 1000 timesteps)
        if self.temporal_evolution.len() > 1000 {
            self.temporal_evolution.drain(0..100);
        }
    }
    
    /// Calculate entropy production rate
    pub fn entropy_production_rate(&self) -> f64 {
        if self.temporal_evolution.len() < 2 {
            return 0.0;
        }
        
        let current = self.temporal_evolution.last().unwrap();
        let previous = &self.temporal_evolution[self.temporal_evolution.len() - 2];
        
        // Calculate KL divergence as entropy production
        let mut production = 0.0;
        for (i, (&curr, &prev)) in current.iter().zip(previous.iter()).enumerate() {
            if curr > 0.0 && prev > 0.0 {
                production += curr * (curr / prev).ln();
            }
        }
        
        production
    }
}

impl ClusteringAnalysis {
    /// Create new clustering analysis
    pub fn new(num_clusters: usize) -> Self {
        Self {
            cluster_centers: Vec::with_capacity(num_clusters),
            cluster_assignments: Vec::new(),
            cluster_probabilities: Array1::from_vec(vec![1.0 / num_clusters as f64; num_clusters]),
            inter_cluster_transitions: Array2::eye(num_clusters),
            cluster_stability_metrics: vec![1.0; num_clusters],
        }
    }
    
    /// Calculate cluster entropy
    pub fn cluster_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        for &prob in self.cluster_probabilities.iter() {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        entropy
    }
}

impl MolecularConfiguration {
    /// Create minimal molecular configuration
    pub fn new_minimal() -> Self {
        Self {
            atom_positions: Vec::new(),
            bond_lengths: Vec::new(),
            bond_angles: Vec::new(),
            dihedral_angles: Vec::new(),
            electronic_state: ElectronicState {
                orbital_occupancies: Vec::new(),
                spin_multiplicities: Vec::new(),
                dipole_moment: (0.0, 0.0, 0.0),
                polarizability: Array2::eye(3),
                electron_density_distribution: Array3::zeros((10, 10, 10)),
            },
            vibrational_modes: Vec::new(),
            energy: 0.0,
            stability_score: 1.0,
        }
    }
    
    /// Calculate structural similarity to another configuration
    pub fn structural_similarity(&self, other: &MolecularConfiguration) -> f64 {
        if self.atom_positions.len() != other.atom_positions.len() {
            return 0.0;
        }
        
        // Calculate RMSD between atom positions
        let mut sum_squared_diff = 0.0;
        for (pos1, pos2) in self.atom_positions.iter().zip(other.atom_positions.iter()) {
            let dx = pos1.0 - pos2.0;
            let dy = pos1.1 - pos2.1;
            let dz = pos1.2 - pos2.2;
            sum_squared_diff += dx*dx + dy*dy + dz*dz;
        }
        
        let rmsd = (sum_squared_diff / self.atom_positions.len() as f64).sqrt();
        
        // Convert RMSD to similarity (0-1 scale)
        (-rmsd).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entropy_distribution_creation() {
        let entropy_dist = EntropyDistribution::new(4);
        assert_eq!(entropy_dist.configuration_endpoints.len(), 0);
        assert_eq!(entropy_dist.landing_probabilities.len(), 4);
        
        // Check uniform distribution
        for &prob in entropy_dist.landing_probabilities.iter() {
            assert!((prob - 0.25).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_shannon_entropy() {
        let mut entropy_dist = EntropyDistribution::new(4);
        
        // Uniform distribution should have maximum entropy
        let uniform_entropy = entropy_dist.shannon_entropy();
        assert!(uniform_entropy > 0.0);
        
        // Create non-uniform distribution
        entropy_dist.landing_probabilities = Array1::from_vec(vec![0.7, 0.2, 0.08, 0.02]);
        let nonuniform_entropy = entropy_dist.shannon_entropy();
        
        // Non-uniform should have less entropy
        assert!(nonuniform_entropy < uniform_entropy);
    }
    
    #[test]
    fn test_add_configuration_endpoint() {
        let mut entropy_dist = EntropyDistribution::new(2);
        let config = MolecularConfiguration::new_minimal();
        
        entropy_dist.add_configuration_endpoint(config, 0.3);
        assert_eq!(entropy_dist.configuration_endpoints.len(), 1);
        assert_eq!(entropy_dist.landing_probabilities.len(), 3);
        
        // Check probability normalization
        let sum: f64 = entropy_dist.landing_probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_effective_accessible_states() {
        let entropy_dist = EntropyDistribution::new(4);
        let effective_states = entropy_dist.effective_accessible_states();
        
        // For uniform distribution over 4 states, should be close to 4
        assert!((effective_states - 4.0).abs() < 0.1);
    }
} 