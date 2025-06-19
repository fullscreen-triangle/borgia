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
// This system implements revolutionary molecular representations based on:
// 1. Universal Oscillatory Framework - Reality as nested oscillations
// 2. Membrane Quantum Computation Theorem - Life as quantum inevitability
// 3. Entropy as tangible oscillation endpoint distributions
// 4. Environment-Assisted Quantum Transport (ENAQT) principles
// =====================================================================================

// Quantum-Oscillatory Framework
pub mod oscillatory;
pub mod quantum;
pub mod entropy;
pub mod membrane;

// Analysis Engines
pub mod prediction;
pub mod synchronization;

// Re-export main types
pub use oscillatory::UniversalOscillator;
pub use quantum::QuantumMolecularComputer;
pub use entropy::EntropyDistribution;
pub use prediction::QuantumBiologicalPropertyPredictor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let engine = BorgiaEngine::new();
        assert!(engine.is_initialized());
    }
} 