//! # Borgia - Revolutionary Probabilistic Cheminformatics Engine
//!
//! Borgia is a purpose-driven cheminformatics confirmation engine that serves as the molecular
//! evidence workhorse for biological intelligence systems. Unlike mainstream cheminformatics,
//! Borgia performs evidence-constrained, probabilistic molecular analysis on small sets of
//! highly relevant molecules.
//!
//! ## Key Features
//!
//! - **Probabilistic Molecular Representations**: Enhanced chemical descriptors with uncertainty
//! - **Evidence-Driven Processing**: Only analyze molecules that upstream evidence suggests
//! - **Fuzzy Logic Integration**: Linguistic variables for molecular reasoning
//! - **Adaptive Learning**: Representations evolve based on feedback
//!
//! ## Quick Start
//!
//! ```rust
//! use borgia::{BorgiaEngine, ProbabilisticMolecule, BorgiaRequest};
//!
//! // Initialize the engine
//! let engine = BorgiaEngine::new();
//!
//! // Create probabilistic molecular representations
//! let mol1 = ProbabilisticMolecule::from_smiles("CCO")?;  // Ethanol
//! let mol2 = ProbabilisticMolecule::from_smiles("CCN")?;  // Ethylamine
//!
//! // Compare with uncertainty quantification
//! let similarity = engine.compare_molecules(&mol1, &mol2, "drug_metabolism")?;
//! ```

pub mod probabilistic;
pub mod molecular;
pub mod similarity;
pub mod fuzzy;
pub mod evidence;
pub mod core;
pub mod engine;
pub mod integration;
pub mod utils;

// Algorithm modules
pub mod algorithms;

// Representation modules
pub mod representation;

// Error handling
pub mod error;
pub use error::{BorgiaError, Result};

// Core types
pub use core::{
    BorgiaEngine,
    BorgiaRequest,
    EvidenceType,
    ObjectiveFunction,
    UpstreamSystem,
};

// Advanced engine
pub use engine::{
    AdvancedBorgiaEngine,
    MolecularAnalysisResult,
    AnalysisRecommendation,
    ProcessingConfig,
};

// Molecular representations
pub use molecular::{
    ProbabilisticMolecule,
    FuzzyAromaticity,
    FuzzyRingSystems,
    FuzzyFunctionalGroups,
    EnhancedFingerprint,
    OscillatoryQuantumMolecule,
};

// Similarity and comparison
pub use similarity::{
    SimilarityEngine,
    ProbabilisticSimilarity,
    FuzzySimilarity,
    SimilarityDistribution,
    OscillatorySimilarityCalculator,
    QuantumComputationalSimilarityCalculator,
};

// Probabilistic computing
pub use probabilistic::{
    ProbabilisticValue,
    UncertaintyBounds,
    ConfidenceInterval,
    BayesianInference,
};

// Fuzzy logic
pub use fuzzy::{
    FuzzySet,
    LinguisticVariable,
    FuzzyRule,
    MembershipFunction,
    FuzzyInferenceEngine,
    create_molecular_similarity_system,
};

// Evidence processing
pub use evidence::{
    EvidenceProcessor,
    EvidenceContext,
    EvidencePropagation,
    EvidenceStrength,
    EvidenceFusionResult,
};

// Integration with upstream systems
pub use integration::{
    HegelIntegration,
    LavoisierIntegration,
    GospelIntegration,
    BeneGesseritIntegration,
    IntegrationManager,
    IntegrationRequest,
    IntegrationResponse,
    UpstreamFeedback,
};

// =====================================================================================
// BORGIA: Quantum-Oscillatory Molecular Representation System
// 
// This comprehensive system implements revolutionary molecular representations based on:
// 1. Universal Oscillatory Framework - Reality as nested oscillations
// 2. Membrane Quantum Computation Theorem - Life as quantum inevitability
// 3. Entropy as tangible oscillation endpoint distributions
// 4. Environment-Assisted Quantum Transport (ENAQT) principles
// 
// The system represents molecules not as static structures but as dynamic quantum
// oscillators embedded in the fundamental oscillatory fabric of reality itself.
// =====================================================================================

//! # Borgia: Quantum-Oscillatory Molecular Representation System
//!
//! Borgia is a revolutionary cheminformatics engine that represents molecules as dynamic
//! quantum oscillators embedded in the nested hierarchy of reality's oscillatory structure.
//!
//! ## Core Principles
//!
//! 1. **Universal Oscillatory Framework**: All bounded systems with nonlinear dynamics
//!    exhibit oscillatory behavior across nested hierarchies from quantum to organismal scales.
//!
//! 2. **Membrane Quantum Computation Theorem**: Amphipathic molecules with optimal
//!    tunneling distances (3-5 nm) function as room-temperature quantum computers.
//!
//! 3. **Entropy as Tangible Distribution**: Entropy represents statistical distribution
//!    of oscillation termination points - making it directly manipulable.
//!
//! 4. **Environment-Assisted Quantum Transport (ENAQT)**: Environmental coupling
//!    enhances rather than destroys quantum coherence when optimally tuned.
//!
//! ## Example Usage
//!
//! ```rust
//! use borgia::molecular::OscillatoryQuantumMolecule;
//! use borgia::similarity::{OscillatorySimilarityCalculator, QuantumComputationalSimilarityCalculator};
//!
//! // Create quantum-oscillatory molecules
//! let mut mol1 = OscillatoryQuantumMolecule::new("caffeine".to_string(), "CN1C=NC2=C1C(=O)N(C(=O)N2C)C".to_string());
//! let mut mol2 = OscillatoryQuantumMolecule::new("theobromine".to_string(), "CN1C=NC2=C1C(=O)NC(=O)N2C".to_string());
//!
//! // Update dynamics
//! mol1.update_dynamics(1e-12); // 1 ps timestep
//! mol2.update_dynamics(1e-12);
//!
//! // Calculate oscillatory synchronization similarity
//! let osc_calc = OscillatorySimilarityCalculator::new();
//! let sync_similarity = osc_calc.oscillatory_similarity(&mol1, &mol2);
//!
//! // Calculate quantum computational similarity
//! let quantum_calc = QuantumComputationalSimilarityCalculator::new();
//! let quantum_similarity = quantum_calc.quantum_computational_similarity(&mol1, &mol2);
//!
//! println!("Synchronization similarity: {:.3}", sync_similarity);
//! println!("Quantum computational similarity: {:.3}", quantum_similarity);
//! ```

pub mod core;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod fuzzy;
pub mod integration;
pub mod molecular;
pub mod probabilistic;
pub mod representation;
pub mod similarity;
pub mod utils;

// New quantum-oscillatory modules
pub mod oscillatory;
pub mod entropy;
pub mod quantum;
pub mod prediction;

// Re-export core types for convenience
pub use crate::molecular::OscillatoryQuantumMolecule;
pub use crate::oscillatory::{UniversalOscillator, OscillationState, SynchronizationParameters};
pub use crate::entropy::{EntropyDistribution, MolecularConfiguration, ClusteringAnalysis};
pub use crate::quantum::{QuantumMolecularComputer, TunnelingPathway, ElectronTransportChain, MembraneProperties};
pub use crate::prediction::{PropertyPredictions, BiologicalActivityPrediction, LongevityPrediction};
pub use crate::similarity::{OscillatorySimilarityCalculator, QuantumComputationalSimilarityCalculator};

// Re-export existing core functionality
pub use crate::core::*;
pub use crate::engine::*;
pub use crate::error::*;
pub use crate::evidence::*;
pub use crate::fuzzy::*;
pub use crate::integration::*;
pub use crate::probabilistic::*;
pub use crate::representation::*;
pub use crate::utils::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_oscillatory_molecule_creation() {
        let mol = OscillatoryQuantumMolecule::new("test_molecule".to_string(), "CCO".to_string());
        assert_eq!(mol.molecule_id, "test_molecule");
        assert_eq!(mol.smiles, "CCO");
        assert!(mol.oscillatory_state.natural_frequency > 0.0);
        assert!(mol.quantum_computer.transport_efficiency > 0.0);
    }

    #[test]
    fn test_oscillatory_synchronization() {
        let mol1 = OscillatoryQuantumMolecule::new("mol1".to_string(), "CCO".to_string());
        let mol2 = OscillatoryQuantumMolecule::new("mol2".to_string(), "CCN".to_string());
        
        let sync_potential = mol1.synchronization_potential(&mol2);
        assert!(sync_potential >= 0.0);
        assert!(sync_potential <= 1.0);
    }

    #[test]
    fn test_quantum_computational_similarity() {
        let mol1 = OscillatoryQuantumMolecule::new("mol1".to_string(), "CCO".to_string());
        let mol2 = OscillatoryQuantumMolecule::new("mol2".to_string(), "CCN".to_string());
        
        let quantum_calc = QuantumComputationalSimilarityCalculator::new();
        let similarity = quantum_calc.quantum_computational_similarity(&mol1, &mol2);
        
        assert!(similarity >= 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_entropy_distribution() {
        let mut entropy = EntropyDistribution::new(4);
        let shannon_entropy = entropy.shannon_entropy();
        assert!(shannon_entropy > 0.0);
        
        // Test entropy update
        entropy.update_temporal_evolution();
        assert_eq!(entropy.temporal_evolution.len(), 1);
    }

    #[test]
    fn test_quantum_computer_enaqt() {
        let mut qc = QuantumMolecularComputer::new();
        let initial_efficiency = qc.transport_efficiency;
        
        qc.update_transport_efficiency();
        let new_efficiency = qc.calculate_enaqt_efficiency();
        
        assert!(new_efficiency > 0.0);
        assert!(new_efficiency <= 2.0); // ENAQT can exceed classical limit
    }

    #[test]
    fn test_membrane_quantum_computation() {
        let mut qc = QuantumMolecularComputer::new();
        
        // Set membrane-like properties
        qc.membrane_properties.amphipathic_score = 0.8;
        qc.membrane_properties.optimal_tunneling_distances = vec![3.5, 4.0, 4.5];
        qc.transport_efficiency = 0.7;
        qc.coherence_time = 1e-12;
        
        assert!(qc.is_membrane_quantum_computer());
        
        let quantum_advantage = qc.quantum_advantage();
        assert!(quantum_advantage > 1.0);
    }

    #[test]
    fn test_temporal_dynamics() {
        let mut mol = OscillatoryQuantumMolecule::new("test".to_string(), "CCO".to_string());
        
        // Update dynamics for several timesteps
        for _ in 0..10 {
            mol.update_dynamics(1e-15); // 1 fs timesteps
        }
        
        assert_eq!(mol.temporal_dynamics.oscillation_time_series.len(), 10);
        assert_eq!(mol.temporal_dynamics.coherence_evolution.len(), 10);
        assert_eq!(mol.temporal_dynamics.radical_accumulation.len(), 10);
    }

    #[test]
    fn test_oscillatory_similarity_calculator() {
        let calc = OscillatorySimilarityCalculator::new();
        let mol1 = OscillatoryQuantumMolecule::new("mol1".to_string(), "CCO".to_string());
        let mol2 = OscillatoryQuantumMolecule::new("mol2".to_string(), "CCN".to_string());
        
        let similarity = calc.oscillatory_similarity(&mol1, &mol2);
        assert!(similarity >= 0.0);
        assert!(similarity <= 1.0);
        
        let entropy_similarity = calc.entropy_endpoint_similarity(&mol1, &mol2);
        assert!(entropy_similarity >= 0.0);
        assert!(entropy_similarity <= 1.0);
    }

    #[test]
    fn test_radical_generation() {
        let mut qc = QuantumMolecularComputer::new();
        let initial_damage = qc.accumulated_damage;
        
        // Simulate quantum damage accumulation
        qc.update_quantum_damage(1e-9); // 1 nanosecond
        
        assert!(qc.accumulated_damage >= initial_damage);
        assert!(qc.radical_generation_rate > 0.0);
    }

    #[test]
    fn test_property_predictions() {
        let predictions = PropertyPredictions::new();
        
        assert!(predictions.biological_activity.activity_score >= 0.0);
        assert!(predictions.longevity_impact.longevity_factor >= -1.0);
        assert!(predictions.toxicity_prediction.toxicity_score >= 0.0);
        assert!(predictions.drug_likeness.drug_likeness_score >= 0.0);
    }
} 