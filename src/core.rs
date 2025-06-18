//! Core types and structures for Borgia.

use crate::error::{BorgiaError, Result};
use crate::molecular::ProbabilisticMolecule;
use crate::similarity::{SimilarityEngine, ProbabilisticSimilarity};
use serde::{Deserialize, Serialize};

/// Main Borgia engine
#[derive(Debug, Clone)]
pub struct BorgiaEngine {
    pub similarity_engine: SimilarityEngine,
    pub initialized: bool,
}

/// Request structure for Borgia operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorgiaRequest {
    pub molecules: Vec<String>, // SMILES strings
    pub evidence_type: EvidenceType,
    pub objective_function: ObjectiveFunction,
    pub upstream_system: UpstreamSystem,
    pub context: String,
}

/// Types of evidence that can be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    StructuralSimilarity,
    PharmacologicalActivity,
    MetabolicPathway,
    MolecularInteraction,
    PropertyPrediction,
}

/// Objective functions for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectiveFunction {
    MaximizeSimilarity,
    MinimizeUncertainty,
    OptimizeBinding,
    PredictActivity,
    ClassifyMolecules,
}

/// Upstream systems that can integrate with Borgia
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpstreamSystem {
    Hegel,
    Lavoisier,
    Gospel,
    BeneGesserit,
    Other(String),
}

impl BorgiaEngine {
    pub fn new() -> Self {
        Self {
            similarity_engine: SimilarityEngine::new(),
            initialized: true,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn process_request(&self, request: &BorgiaRequest) -> Result<Vec<ProbabilisticSimilarity>> {
        if request.molecules.len() < 2 {
            return Err(BorgiaError::validation(
                "molecules",
                "At least 2 molecules required for comparison",
            ));
        }

        let mut results = Vec::new();
        let molecules: Result<Vec<ProbabilisticMolecule>> = request
            .molecules
            .iter()
            .map(|smiles| ProbabilisticMolecule::from_smiles(smiles))
            .collect();

        let molecules = molecules?;

        // Compare all pairs
        for i in 0..molecules.len() {
            for j in (i + 1)..molecules.len() {
                let similarity = self.similarity_engine.calculate_similarity(
                    &molecules[i],
                    &molecules[j],
                    crate::similarity::SimilarityAlgorithm::Tanimoto,
                    &request.context,
                )?;
                results.push(similarity);
            }
        }

        Ok(results)
    }
}

impl Default for BorgiaEngine {
    fn default() -> Self {
        Self::new()
    }
} 