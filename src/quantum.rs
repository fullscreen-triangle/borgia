// =====================================================================================
// QUANTUM COMPUTATIONAL FRAMEWORK
// Implements the Membrane Quantum Computation Theorem and ENAQT principles
// =====================================================================================

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

/// Molecular representation as quantum computer using ENAQT principles
/// Revolutionary insight: molecules are room-temperature quantum computers
/// where environmental coupling enhances rather than destroys coherence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantumMolecularComputer {
    /// Environmental coupling strength γ - key ENAQT parameter
    pub environmental_coupling_strength: f64,
    
    /// Optimal coupling strength for maximum efficiency
    pub optimal_coupling: f64,
    
    /// Transport efficiency η = η₀ × (1 + αγ + βγ²)
    pub transport_efficiency: f64,
    
    /// Quantum coherence time at room temperature
    pub coherence_time: f64,
    
    /// Quantum beating frequencies from 2D electronic spectroscopy
    pub quantum_beating_frequencies: Array1<f64>,
    
    /// Tunneling pathways for electron transport
    pub tunneling_pathways: Vec<TunnelingPathway>,
    
    /// Electron transport chains
    pub electron_transport_chains: Vec<ElectronTransportChain>,
    
    /// Proton quantum channels
    pub proton_channels: Vec<ProtonChannel>,
    
    /// Inevitable radical generation rate (death-causing quantum leakage)
    pub radical_generation_rate: f64,
    
    /// Cross-section for quantum damage to biomolecules
    pub quantum_damage_cross_section: f64,
    
    /// Accumulated quantum damage over time
    pub accumulated_damage: f64,
    
    /// Membrane-like properties enabling quantum computation
    pub membrane_properties: MembraneProperties,
}

/// Quantum tunneling pathway through molecular barriers
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TunnelingPathway {
    /// Barrier height V₀ in electron volts
    pub barrier_height: f64,
    
    /// Barrier width a in nanometers
    pub barrier_width: f64,
    
    /// Tunneling probability P = (16E(V₀-E)/V₀²)exp(-2κa)
    pub tunneling_probability: f64,
    
    /// Electron energy E
    pub electron_energy: f64,
    
    /// Atoms forming the tunneling pathway
    pub pathway_atoms: Vec<usize>,
    
    /// Environmental assistance factors
    pub environmental_enhancement: f64,
}

/// Electron transport chain for quantum energy conversion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElectronTransportChain {
    /// Redox centers in the transport chain
    pub redox_centers: Vec<RedoxCenter>,
    
    /// Overall transport efficiency
    pub efficiency: f64,
    
    /// Quantum coherence effects
    pub coherence_contributions: Vec<f64>,
}

/// Individual redox center in electron transport
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedoxCenter {
    pub atom_index: usize,
    pub redox_potential: f64,
    pub reorganization_energy: f64,
    pub coupling_strength: f64,
    pub occupancy_probability: f64,
}

/// Proton quantum channel for proton transport
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtonChannel {
    /// Channel atoms forming the proton pathway
    pub channel_atoms: Vec<usize>,
    
    /// Quantized energy levels for proton states
    pub energy_levels: Array1<f64>,
    
    /// Transport rate through the channel
    pub transport_rate: f64,
    
    /// Channel selectivity for protons vs other ions
    pub selectivity: f64,
}

/// Membrane-like properties enabling quantum computation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MembraneProperties {
    /// Amphipathic character (hydrophilic/hydrophobic regions)
    pub amphipathic_score: f64,
    
    /// Self-assembly thermodynamics ΔG
    pub self_assembly_free_energy: f64,
    
    /// Critical micelle concentration
    pub critical_micelle_concentration: f64,
    
    /// Optimal tunneling distances (3-5 nm for biological membranes)
    pub optimal_tunneling_distances: Vec<f64>,
    
    /// Environmental coupling optimization
    pub coupling_optimization_score: f64,
    
    /// Room temperature quantum coherence potential
    pub room_temp_coherence_potential: f64,
}

impl QuantumMolecularComputer {
    /// Create new quantum molecular computer with default parameters
    pub fn new() -> Self {
        Self {
            environmental_coupling_strength: 0.5,
            optimal_coupling: 0.5,
            transport_efficiency: 0.7,
            coherence_time: 1e-12,
            quantum_beating_frequencies: Array1::from_vec(vec![1e12, 2e12, 3e12]),
            tunneling_pathways: Vec::new(),
            electron_transport_chains: Vec::new(),
            proton_channels: Vec::new(),
            radical_generation_rate: 1e-8,
            quantum_damage_cross_section: 1e-15,
            accumulated_damage: 0.0,
            membrane_properties: MembraneProperties::new(),
        }
    }
    
    /// Calculate ENAQT transport efficiency
    pub fn calculate_enaqt_efficiency(&self) -> f64 {
        let gamma = self.environmental_coupling_strength;
        let eta_0 = 0.4; // Base efficiency without environmental assistance
        let alpha = 1.2; // Coherent enhancement coefficient
        let beta = -0.3; // Overdamping coefficient
        
        eta_0 * (1.0 + alpha * gamma + beta * gamma.powi(2))
    }
    
    /// Calculate optimal environmental coupling strength
    pub fn calculate_optimal_coupling(&self) -> f64 {
        let alpha = 1.2;
        let beta = -0.3;
        
        // Optimal coupling: γ_opt = α/(2|β|)
        alpha / (2.0 * beta.abs())
    }
    
    /// Update transport efficiency based on current coupling
    pub fn update_transport_efficiency(&mut self) {
        self.transport_efficiency = self.calculate_enaqt_efficiency();
        self.optimal_coupling = self.calculate_optimal_coupling();
    }
    
    /// Calculate radical generation rate from quantum processes
    pub fn calculate_radical_generation_rate(&self) -> f64 {
        let base_rate = 1e-9; // Base radical generation rate per second
        let quantum_flux = self.transport_efficiency * 1e15; // Electrons per second
        let leakage_fraction = 1.0 - self.environmental_coupling_strength;
        
        base_rate * quantum_flux * leakage_fraction * self.quantum_damage_cross_section
    }
    
    /// Update accumulated quantum damage
    pub fn update_quantum_damage(&mut self, dt: f64) {
        let damage_rate = self.calculate_radical_generation_rate();
        self.accumulated_damage += damage_rate * dt;
        self.radical_generation_rate = damage_rate;
    }
    
    /// Check if molecule can function as membrane quantum computer
    pub fn is_membrane_quantum_computer(&self) -> bool {
        self.membrane_properties.amphipathic_score > 0.5 &&
        self.membrane_properties.optimal_tunneling_distances.iter()
            .any(|&distance| distance >= 3.0 && distance <= 5.0) &&
        self.transport_efficiency > 0.6 &&
        self.coherence_time > 1e-15
    }
    
    /// Calculate quantum computational advantage
    pub fn quantum_advantage(&self) -> f64 {
        if !self.is_membrane_quantum_computer() {
            return 1.0; // No quantum advantage
        }
        
        // Quantum advantage from coherence and ENAQT
        let coherence_advantage = (self.coherence_time * 1e12).ln().max(1.0);
        let transport_advantage = self.transport_efficiency / 0.4; // Relative to classical limit
        let coupling_advantage = 1.0 + self.environmental_coupling_strength;
        
        coherence_advantage * transport_advantage * coupling_advantage
    }
}

impl MembraneProperties {
    /// Create new membrane properties with default values
    pub fn new() -> Self {
        Self {
            amphipathic_score: 0.3,
            self_assembly_free_energy: -20.0, // kJ/mol
            critical_micelle_concentration: 1e-3, // M
            optimal_tunneling_distances: vec![4.0], // nm
            coupling_optimization_score: 0.5,
            room_temp_coherence_potential: 0.5,
        }
    }
    
    /// Check if properties are suitable for membrane quantum computation
    pub fn is_quantum_membrane(&self) -> bool {
        self.amphipathic_score > 0.5 &&
        self.critical_micelle_concentration < 1e-3 &&
        self.optimal_tunneling_distances.iter().any(|&d| d >= 3.0 && d <= 5.0) &&
        self.room_temp_coherence_potential > 0.7
    }
    
    /// Calculate membrane assembly probability
    pub fn assembly_probability(&self, temperature: f64) -> f64 {
        let kb = 8.314e-3; // kJ/(mol·K)
        let delta_g = self.self_assembly_free_energy;
        
        if delta_g < 0.0 {
            // Favorable assembly
            1.0 - (-delta_g / (kb * temperature)).exp().recip()
        } else {
            // Unfavorable assembly
            (-delta_g / (kb * temperature)).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_computer_creation() {
        let qmc = QuantumMolecularComputer::new();
        assert_eq!(qmc.transport_efficiency, 0.7);
        assert_eq!(qmc.coherence_time, 1e-12);
    }
    
    #[test]
    fn test_enaqt_efficiency() {
        let qmc = QuantumMolecularComputer::new();
        let efficiency = qmc.calculate_enaqt_efficiency();
        assert!(efficiency > 0.0);
        assert!(efficiency < 2.0); // Should be reasonable
    }
    
    #[test]
    fn test_optimal_coupling() {
        let qmc = QuantumMolecularComputer::new();
        let optimal = qmc.calculate_optimal_coupling();
        assert!(optimal > 0.0);
        assert!(optimal < 10.0); // Should be reasonable
    }
    
    #[test]
    fn test_membrane_properties() {
        let membrane = MembraneProperties::new();
        assert!(membrane.amphipathic_score >= 0.0);
        assert!(membrane.amphipathic_score <= 1.0);
        assert!(membrane.self_assembly_free_energy < 0.0); // Should be favorable
    }
    
    #[test]
    fn test_quantum_advantage() {
        let mut qmc = QuantumMolecularComputer::new();
        qmc.membrane_properties.amphipathic_score = 0.8;
        qmc.membrane_properties.optimal_tunneling_distances = vec![4.0];
        qmc.transport_efficiency = 0.8;
        qmc.coherence_time = 1e-9;
        
        let advantage = qmc.quantum_advantage();
        assert!(advantage > 1.0); // Should have quantum advantage
    }
} 