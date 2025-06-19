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

pub mod algorithms;
pub mod core;
pub mod engine;
pub mod entropy;
pub mod error;
pub mod evidence;
pub mod fuzzy;
pub mod integration;
pub mod molecular;
pub mod oscillatory;
pub mod prediction;
pub mod probabilistic;
pub mod quantum;
pub mod representation;
pub mod similarity;
pub mod utils;

// Re-export main components for easy access
pub use crate::core::BorgiaEngine;
pub use crate::engine::{BorgiaQuantumOscillatorySystem, QuantumOscillatoryAnalysisResult, DesignGoals};
pub use crate::molecular::{OscillatoryQuantumMolecule, HierarchyLevel, LevelDynamics};
pub use crate::oscillatory::{UniversalOscillator, OscillationState};
pub use crate::entropy::{EntropyDistribution, MolecularConfiguration, ClusteringAnalysis};
pub use crate::quantum::{QuantumMolecularComputer, MembraneProperties, TunnelingPathway, ElectronTransportChain, ProtonChannel};
pub use crate::prediction::{QuantumBiologicalPropertyPredictor, BiologicalActivityPrediction, LongevityPrediction, ToxicityPrediction, DrugLikenessPrediction, MembraneInteractionPrediction, QuantumEfficiencyPrediction};
pub use crate::similarity::{OscillatorySimilarityCalculator, QuantumComputationalSimilarityCalculator, ComprehensiveSimilarityResult};
pub use crate::representation::{SynchronizationParameters, InformationCatalyst, PropertyPredictions, TemporalDynamics};
pub use crate::molecular::molecule_database::{QuantumMolecularDatabase, SearchCriteria, QuantumSearchCriteria, OscillatorySearchCriteria, HierarchySearchCriteria, PropertySearchCriteria};

// Error handling
pub use crate::error::{BorgiaError, Result};

// Utility functions
pub use crate::utils::*;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Main entry point for the Borgia quantum-oscillatory molecular system
/// 
/// # Example
/// 
/// ```rust
/// use borgia::{BorgiaQuantumOscillatorySystem, SearchCriteria, QuantumSearchCriteria};
/// 
/// let mut system = BorgiaQuantumOscillatorySystem::new();
/// 
/// // Analyze a molecule
/// match system.complete_analysis("CCO") {
///     Ok(result) => {
///         println!("Quantum computational score: {}", result.quantum_computational_score);
///         println!("Oscillatory synchronization score: {}", result.oscillatory_synchronization_score);
///         println!("Death inevitability score: {}", result.death_inevitability_score);
///         println!("Membrane quantum computer potential: {}", result.membrane_quantum_computer_potential);
///         
///         for recommendation in &result.recommendations {
///             println!("Recommendation: {}", recommendation);
///         }
///     }
///     Err(e) => eprintln!("Analysis failed: {}", e),
/// }
/// 
/// // Search for quantum computers
/// let quantum_criteria = QuantumSearchCriteria {
///     min_enaqt_efficiency: Some(0.7),
///     min_membrane_score: Some(0.5),
///     max_radical_generation_rate: Some(1e-8),
///     min_coherence_time: Some(1e-12),
///     min_tunneling_pathways: Some(1),
///     min_match_threshold: Some(0.8),
///     weight: 1.0,
/// };
/// 
/// let results = system.database.search_quantum_computers(&quantum_criteria);
/// for (molecule_id, score) in results {
///     println!("Quantum computer candidate: {} (score: {:.3})", molecule_id, score);
/// }
/// ```
pub fn create_quantum_oscillatory_system() -> BorgiaQuantumOscillatorySystem {
    BorgiaQuantumOscillatorySystem::new()
}

/// Create a new Borgia engine (backward compatibility)
pub fn create_engine() -> BorgiaEngine {
    BorgiaEngine::new()
}

/// Quick analysis function for single molecules
pub fn analyze_molecule(smiles: &str) -> Result<QuantumOscillatoryAnalysisResult> {
    let mut system = BorgiaQuantumOscillatorySystem::new();
    system.complete_analysis(smiles).map_err(|e| BorgiaError::AnalysisError(e))
}

/// Batch analysis function for multiple molecules
pub fn analyze_molecules(smiles_list: Vec<String>) -> Vec<Result<QuantumOscillatoryAnalysisResult>> {
    let mut system = BorgiaQuantumOscillatorySystem::new();
    system.batch_analysis(smiles_list).into_iter()
        .map(|result| result.map_err(|e| BorgiaError::AnalysisError(e)))
        .collect()
}

/// Search for longevity-enhancing molecules
pub fn find_longevity_enhancers() -> Vec<(String, f64)> {
    let system = BorgiaQuantumOscillatorySystem::new();
    system.database.search_longevity_enhancers()
}

/// Search for death-accelerating molecules (high radical generation)
pub fn find_death_accelerators() -> Vec<(String, f64)> {
    let system = BorgiaQuantumOscillatorySystem::new();
    system.database.search_death_accelerators()
}

/// Find synchronization partners for a given molecule
pub fn find_synchronization_partners(molecule_id: &str, max_freq_diff: f64) -> Vec<(String, f64)> {
    let system = BorgiaQuantumOscillatorySystem::new();
    system.database.find_synchronization_partners(molecule_id, max_freq_diff)
}

/// Calculate comprehensive similarity between two molecules
pub fn calculate_comprehensive_similarity(mol1: &str, mol2: &str) -> ComprehensiveSimilarityResult {
    let system = BorgiaQuantumOscillatorySystem::new();
    system.comprehensive_similarity(mol1, mol2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let system = create_quantum_oscillatory_system();
        assert!(system.database.molecules.is_empty());
    }

    #[test]
    fn test_engine_creation() {
        let engine = create_engine();
        // Basic test to ensure engine creation works
    }

    #[test]
    fn test_molecule_analysis() {
        // Test basic molecule analysis functionality
        let result = analyze_molecule("CCO");
        // In a real implementation, this would test actual analysis
        // For now, we just ensure the function can be called
    }

    #[test]
    fn test_batch_analysis() {
        let molecules = vec!["CCO".to_string(), "CC".to_string()];
        let results = analyze_molecules(molecules);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_similarity_calculation() {
        let similarity = calculate_comprehensive_similarity("CCO", "CC");
        // Test that similarity calculation returns a valid result structure
        assert!(similarity.overall_similarity >= 0.0);
        assert!(similarity.overall_similarity <= 1.0);
    }

    #[test]
    fn test_search_functions() {
        // Test longevity enhancer search
        let enhancers = find_longevity_enhancers();
        assert!(enhancers.is_empty()); // Empty database initially

        // Test death accelerator search
        let accelerators = find_death_accelerators();
        assert!(accelerators.is_empty()); // Empty database initially

        // Test synchronization partner search
        let partners = find_synchronization_partners("test_mol", 0.1);
        assert!(partners.is_empty()); // Empty database initially
    }
} 