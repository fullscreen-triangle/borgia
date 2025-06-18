//! Advanced molecular processing engine for Borgia.

use crate::error::{BorgiaError, Result};
use crate::molecular::{ProbabilisticMolecule, EnhancedFingerprint};
use crate::similarity::{SimilarityEngine, ProbabilisticSimilarity, SimilarityAlgorithm};
use crate::probabilistic::{ProbabilisticValue, BayesianInference, ConfidenceInterval};
use crate::fuzzy::{FuzzyInferenceEngine, LinguisticVariable, create_molecular_similarity_system};
use crate::evidence::{EvidenceProcessor, EvidenceContext};
use crate::core::{BorgiaRequest, EvidenceType, ObjectiveFunction, UpstreamSystem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced molecular processing engine
#[derive(Debug, Clone)]
pub struct AdvancedBorgiaEngine {
    pub similarity_engine: SimilarityEngine,
    pub fuzzy_engine: FuzzyInferenceEngine,
    pub bayesian_engine: BayesianInference,
    pub evidence_processor: EvidenceProcessor,
    pub molecular_cache: HashMap<String, ProbabilisticMolecule>,
    pub fingerprint_cache: HashMap<String, EnhancedFingerprint>,
    pub learning_enabled: bool,
    pub confidence_threshold: f64,
}

/// Comprehensive analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularAnalysisResult {
    pub similarity_matrix: Vec<Vec<ProbabilisticSimilarity>>,
    pub cluster_assignments: Vec<usize>,
    pub confidence_scores: Vec<f64>,
    pub linguistic_descriptions: Vec<String>,
    pub uncertainty_bounds: Vec<ConfidenceInterval>,
    pub evidence_support: f64,
    pub recommendation: AnalysisRecommendation,
}

/// Analysis recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisRecommendation {
    HighConfidenceMatch { confidence: f64, reason: String },
    ModerateConfidenceMatch { confidence: f64, uncertainty: f64, reason: String },
    LowConfidenceMatch { confidence: f64, uncertainty: f64, additional_evidence_needed: Vec<String> },
    NoSignificantMatch { reason: String, suggestions: Vec<String> },
    InsufficientEvidence { required_evidence: Vec<String> },
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub use_fuzzy_logic: bool,
    pub use_bayesian_inference: bool,
    pub enable_learning: bool,
    pub confidence_threshold: f64,
    pub similarity_algorithms: Vec<SimilarityAlgorithm>,
    pub max_molecules: usize,
    pub enable_caching: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            use_fuzzy_logic: true,
            use_bayesian_inference: true,
            enable_learning: true,
            confidence_threshold: 0.7,
            similarity_algorithms: vec![
                SimilarityAlgorithm::Tanimoto,
                SimilarityAlgorithm::Probabilistic,
                SimilarityAlgorithm::Fuzzy,
            ],
            max_molecules: 1000,
            enable_caching: true,
        }
    }
}

impl AdvancedBorgiaEngine {
    /// Create a new advanced engine
    pub fn new() -> Self {
        Self {
            similarity_engine: SimilarityEngine::new(),
            fuzzy_engine: create_molecular_similarity_system(),
            bayesian_engine: BayesianInference::new(),
            evidence_processor: EvidenceProcessor::new(),
            molecular_cache: HashMap::new(),
            fingerprint_cache: HashMap::new(),
            learning_enabled: true,
            confidence_threshold: 0.7,
        }
    }

    /// Create engine with custom configuration
    pub fn with_config(config: &ProcessingConfig) -> Self {
        let mut engine = Self::new();
        engine.learning_enabled = config.enable_learning;
        engine.confidence_threshold = config.confidence_threshold;
        engine
    }

    /// Process molecules with comprehensive analysis
    pub fn analyze_molecules(
        &mut self,
        request: &BorgiaRequest,
        config: &ProcessingConfig,
    ) -> Result<MolecularAnalysisResult> {
        // Validate request
        self.validate_request(request, config)?;

        // Create evidence context
        let evidence_context = EvidenceContext::new(
            request.evidence_type.clone(),
            request.upstream_system.clone(),
            request.context.clone(),
        );

        // Process molecules
        let molecules = self.process_molecule_batch(&request.molecules, &evidence_context)?;

        // Calculate comprehensive similarity matrix
        let similarity_matrix = self.calculate_similarity_matrix(&molecules, &evidence_context, config)?;

        // Perform clustering analysis
        let cluster_assignments = self.cluster_molecules(&similarity_matrix)?;

        // Calculate confidence scores
        let confidence_scores = self.calculate_confidence_scores(&similarity_matrix, &evidence_context)?;

        // Generate linguistic descriptions
        let linguistic_descriptions = self.generate_linguistic_descriptions(&similarity_matrix, &molecules)?;

        // Calculate uncertainty bounds
        let uncertainty_bounds = self.calculate_uncertainty_bounds(&similarity_matrix)?;

        // Assess evidence support
        let evidence_support = self.assess_evidence_support(&evidence_context, &molecules)?;

        // Generate recommendation
        let recommendation = self.generate_recommendation(
            &similarity_matrix,
            &confidence_scores,
            evidence_support,
            &request.objective_function,
        )?;

        // Update learning if enabled
        if self.learning_enabled {
            self.update_learning(&molecules, &similarity_matrix, &evidence_context)?;
        }

        Ok(MolecularAnalysisResult {
            similarity_matrix,
            cluster_assignments,
            confidence_scores,
            linguistic_descriptions,
            uncertainty_bounds,
            evidence_support,
            recommendation,
        })
    }

    /// Compare two molecules with full uncertainty quantification
    pub fn compare_molecules_advanced(
        &mut self,
        mol1_smiles: &str,
        mol2_smiles: &str,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        // Get or create molecules
        let mol1 = self.get_or_create_molecule(mol1_smiles)?;
        let mol2 = self.get_or_create_molecule(mol2_smiles)?;

        // Calculate multiple similarity measures
        let tanimoto = self.similarity_engine.calculate_similarity(
            &mol1,
            &mol2,
            SimilarityAlgorithm::Tanimoto,
            context,
        )?;

        let probabilistic = self.similarity_engine.calculate_similarity(
            &mol1,
            &mol2,
            SimilarityAlgorithm::Probabilistic,
            context,
        )?;

        let fuzzy = self.similarity_engine.calculate_similarity(
            &mol1,
            &mol2,
            SimilarityAlgorithm::Fuzzy,
            context,
        )?;

        // Combine results with Bayesian inference
        let combined_similarity = self.combine_similarity_measures(vec![tanimoto, probabilistic, fuzzy])?;

        Ok(combined_similarity)
    }

    /// Validate processing request
    fn validate_request(&self, request: &BorgiaRequest, config: &ProcessingConfig) -> Result<()> {
        if request.molecules.is_empty() {
            return Err(BorgiaError::validation("molecules", "No molecules provided"));
        }

        if request.molecules.len() > config.max_molecules {
            return Err(BorgiaError::validation(
                "molecules",
                &format!("Too many molecules: {} > {}", request.molecules.len(), config.max_molecules),
            ));
        }

        // Validate SMILES strings
        for (i, smiles) in request.molecules.iter().enumerate() {
            if smiles.is_empty() {
                return Err(BorgiaError::validation(
                    "molecules",
                    &format!("Empty SMILES string at index {}", i),
                ));
            }
        }

        Ok(())
    }

    /// Process a batch of molecules
    fn process_molecule_batch(
        &mut self,
        smiles_list: &[String],
        evidence_context: &EvidenceContext,
    ) -> Result<Vec<ProbabilisticMolecule>> {
        let mut molecules = Vec::new();

        for smiles in smiles_list {
            let molecule = self.get_or_create_molecule(smiles)?;
            
            // Apply evidence-based filtering/enhancement
            let enhanced_molecule = self.evidence_processor.enhance_molecule(&molecule, evidence_context)?;
            molecules.push(enhanced_molecule);
        }

        Ok(molecules)
    }

    /// Get molecule from cache or create new one
    fn get_or_create_molecule(&mut self, smiles: &str) -> Result<ProbabilisticMolecule> {
        if let Some(cached_mol) = self.molecular_cache.get(smiles) {
            Ok(cached_mol.clone())
        } else {
            let molecule = ProbabilisticMolecule::from_smiles(smiles)?;
            self.molecular_cache.insert(smiles.to_string(), molecule.clone());
            Ok(molecule)
        }
    }

    /// Calculate comprehensive similarity matrix
    fn calculate_similarity_matrix(
        &self,
        molecules: &[ProbabilisticMolecule],
        evidence_context: &EvidenceContext,
        config: &ProcessingConfig,
    ) -> Result<Vec<Vec<ProbabilisticSimilarity>>> {
        let n = molecules.len();
        let mut matrix = vec![vec![ProbabilisticSimilarity::default(); n]; n];

        for i in 0..n {
            for j in i..n {
                if i == j {
                    // Self-similarity is 1.0 with high confidence
                    matrix[i][j] = ProbabilisticSimilarity {
                        value: ProbabilisticValue::new(1.0, 0.0),
                        confidence: 1.0,
                        linguistic_description: "identical".to_string(),
                        context: evidence_context.context.clone(),
                        algorithm_used: "self".to_string(),
                    };
                } else {
                    // Calculate similarity using multiple algorithms
                    let mut similarities = Vec::new();
                    
                    for algorithm in &config.similarity_algorithms {
                        let sim = self.similarity_engine.calculate_similarity(
                            &molecules[i],
                            &molecules[j],
                            algorithm.clone(),
                            &evidence_context.context,
                        )?;
                        similarities.push(sim);
                    }

                    // Combine similarities
                    let combined = self.combine_similarity_measures(similarities)?;
                    matrix[i][j] = combined.clone();
                    matrix[j][i] = combined; // Symmetric
                }
            }
        }

        Ok(matrix)
    }

    /// Combine multiple similarity measures
    fn combine_similarity_measures(
        &self,
        similarities: Vec<ProbabilisticSimilarity>,
    ) -> Result<ProbabilisticSimilarity> {
        if similarities.is_empty() {
            return Err(BorgiaError::computation("No similarities to combine"));
        }

        if similarities.len() == 1 {
            return Ok(similarities[0].clone());
        }

        // Extract values and confidences
        let values: Vec<f64> = similarities.iter().map(|s| s.value.mean).collect();
        let uncertainties: Vec<f64> = similarities.iter().map(|s| s.value.std_dev).collect();
        let confidences: Vec<f64> = similarities.iter().map(|s| s.confidence).collect();

        // Weighted average based on confidence
        let total_confidence: f64 = confidences.iter().sum();
        if total_confidence == 0.0 {
            return Err(BorgiaError::computation("Zero total confidence"));
        }

        let weighted_mean: f64 = values.iter()
            .zip(confidences.iter())
            .map(|(v, c)| v * c)
            .sum::<f64>() / total_confidence;

        // Combined uncertainty (simplified)
        let combined_uncertainty: f64 = uncertainties.iter()
            .zip(confidences.iter())
            .map(|(u, c)| u * c)
            .sum::<f64>() / total_confidence;

        // Average confidence
        let avg_confidence = total_confidence / similarities.len() as f64;

        // Generate linguistic description
        let linguistic = if weighted_mean >= 0.8 {
            "very_high"
        } else if weighted_mean >= 0.6 {
            "high"
        } else if weighted_mean >= 0.4 {
            "medium"
        } else if weighted_mean >= 0.2 {
            "low"
        } else {
            "very_low"
        }.to_string();

        Ok(ProbabilisticSimilarity {
            value: ProbabilisticValue::new(weighted_mean, combined_uncertainty),
            confidence: avg_confidence,
            linguistic_description: linguistic,
            context: similarities[0].context.clone(),
            algorithm_used: "combined".to_string(),
        })
    }

    /// Perform molecular clustering
    fn cluster_molecules(&self, similarity_matrix: &[Vec<ProbabilisticSimilarity>]) -> Result<Vec<usize>> {
        let n = similarity_matrix.len();
        let mut clusters = vec![0; n];
        let mut cluster_id = 0;
        let mut visited = vec![false; n];

        // Simple threshold-based clustering
        let threshold = self.confidence_threshold;

        for i in 0..n {
            if !visited[i] {
                let mut current_cluster = vec![i];
                visited[i] = true;

                // Find all molecules similar to this one
                for j in (i + 1)..n {
                    if !visited[j] && similarity_matrix[i][j].value.mean >= threshold {
                        current_cluster.push(j);
                        visited[j] = true;
                    }
                }

                // Assign cluster IDs
                for &mol_idx in &current_cluster {
                    clusters[mol_idx] = cluster_id;
                }
                cluster_id += 1;
            }
        }

        Ok(clusters)
    }

    /// Calculate confidence scores for each molecule
    fn calculate_confidence_scores(
        &self,
        similarity_matrix: &[Vec<ProbabilisticSimilarity>],
        evidence_context: &EvidenceContext,
    ) -> Result<Vec<f64>> {
        let n = similarity_matrix.len();
        let mut scores = vec![0.0; n];

        for i in 0..n {
            let mut total_confidence = 0.0;
            let mut count = 0;

            for j in 0..n {
                if i != j {
                    total_confidence += similarity_matrix[i][j].confidence;
                    count += 1;
                }
            }

            scores[i] = if count > 0 {
                total_confidence / count as f64
            } else {
                0.0
            };
        }

        Ok(scores)
    }

    /// Generate linguistic descriptions
    fn generate_linguistic_descriptions(
        &self,
        similarity_matrix: &[Vec<ProbabilisticSimilarity>],
        molecules: &[ProbabilisticMolecule],
    ) -> Result<Vec<String>> {
        let n = molecules.len();
        let mut descriptions = Vec::new();

        for i in 0..n {
            let mut high_similarities = 0;
            let mut total_similarity = 0.0;

            for j in 0..n {
                if i != j {
                    let sim_value = similarity_matrix[i][j].value.mean;
                    total_similarity += sim_value;
                    if sim_value >= 0.7 {
                        high_similarities += 1;
                    }
                }
            }

            let avg_similarity = total_similarity / (n - 1) as f64;
            
            let description = if high_similarities >= (n - 1) / 2 {
                format!("Highly similar to most molecules (avg: {:.3})", avg_similarity)
            } else if avg_similarity >= 0.5 {
                format!("Moderately similar to other molecules (avg: {:.3})", avg_similarity)
            } else {
                format!("Distinct from other molecules (avg: {:.3})", avg_similarity)
            };

            descriptions.push(description);
        }

        Ok(descriptions)
    }

    /// Calculate uncertainty bounds
    fn calculate_uncertainty_bounds(
        &self,
        similarity_matrix: &[Vec<ProbabilisticSimilarity>],
    ) -> Result<Vec<ConfidenceInterval>> {
        let n = similarity_matrix.len();
        let mut bounds = Vec::new();

        for i in 0..n {
            let similarities: Vec<f64> = (0..n)
                .filter(|&j| i != j)
                .map(|j| similarity_matrix[i][j].value.mean)
                .collect();

            if similarities.is_empty() {
                bounds.push(ConfidenceInterval::new(0.0, 0.0, 0.95));
                continue;
            }

            let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
            let variance = similarities.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / similarities.len() as f64;
            let std_dev = variance.sqrt();

            // 95% confidence interval
            let margin = 1.96 * std_dev / (similarities.len() as f64).sqrt();
            bounds.push(ConfidenceInterval::new(mean - margin, mean + margin, 0.95));
        }

        Ok(bounds)
    }

    /// Assess evidence support
    fn assess_evidence_support(
        &self,
        evidence_context: &EvidenceContext,
        molecules: &[ProbabilisticMolecule],
    ) -> Result<f64> {
        // Simplified evidence assessment
        let base_support = match evidence_context.evidence_type {
            EvidenceType::StructuralSimilarity => 0.8,
            EvidenceType::PharmacologicalActivity => 0.9,
            EvidenceType::MetabolicPathway => 0.7,
            EvidenceType::MolecularInteraction => 0.85,
            EvidenceType::PropertyPrediction => 0.75,
        };

        // Adjust based on number of molecules and their complexity
        let complexity_factor = molecules.len() as f64 / 10.0;
        let adjusted_support = base_support * (1.0 - complexity_factor.min(0.3));

        Ok(adjusted_support.max(0.1).min(1.0))
    }

    /// Generate analysis recommendation
    fn generate_recommendation(
        &self,
        similarity_matrix: &[Vec<ProbabilisticSimilarity>],
        confidence_scores: &[f64],
        evidence_support: f64,
        objective: &ObjectiveFunction,
    ) -> Result<AnalysisRecommendation> {
        let avg_confidence = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        let max_confidence = confidence_scores.iter().cloned().fold(0.0, f64::max);

        // Find maximum similarity
        let max_similarity = similarity_matrix.iter()
            .flatten()
            .map(|s| s.value.mean)
            .fold(0.0, f64::max);

        let combined_score = (avg_confidence + evidence_support + max_similarity) / 3.0;

        let recommendation = if combined_score >= 0.8 && max_confidence >= 0.8 {
            AnalysisRecommendation::HighConfidenceMatch {
                confidence: combined_score,
                reason: format!("Strong evidence support ({:.3}) with high similarity ({:.3})", 
                    evidence_support, max_similarity),
            }
        } else if combined_score >= 0.6 {
            let uncertainty = 1.0 - combined_score;
            AnalysisRecommendation::ModerateConfidenceMatch {
                confidence: combined_score,
                uncertainty,
                reason: format!("Moderate confidence with some uncertainty ({:.3})", uncertainty),
            }
        } else if combined_score >= 0.3 {
            let uncertainty = 1.0 - combined_score;
            AnalysisRecommendation::LowConfidenceMatch {
                confidence: combined_score,
                uncertainty,
                additional_evidence_needed: vec![
                    "More molecular examples".to_string(),
                    "Additional structural data".to_string(),
                    "Experimental validation".to_string(),
                ],
            }
        } else if evidence_support < 0.4 {
            AnalysisRecommendation::InsufficientEvidence {
                required_evidence: vec![
                    "Stronger upstream evidence".to_string(),
                    "More contextual information".to_string(),
                    "Larger molecule set".to_string(),
                ],
            }
        } else {
            AnalysisRecommendation::NoSignificantMatch {
                reason: "Low similarity scores across all comparisons".to_string(),
                suggestions: vec![
                    "Try different similarity algorithms".to_string(),
                    "Expand molecular search space".to_string(),
                    "Consider different evidence types".to_string(),
                ],
            }
        };

        Ok(recommendation)
    }

    /// Update learning from analysis results
    fn update_learning(
        &mut self,
        molecules: &[ProbabilisticMolecule],
        similarity_matrix: &[Vec<ProbabilisticSimilarity>],
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Update Bayesian priors based on results
        for row in similarity_matrix {
            for similarity in row {
                self.bayesian_engine.update_evidence(similarity.value.mean, similarity.confidence)?;
            }
        }

        // Update fuzzy system parameters (simplified)
        // In a real implementation, this would adjust membership functions
        // based on observed performance

        Ok(())
    }
}

impl Default for AdvancedBorgiaEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export the core engine for backward compatibility
pub use crate::core::BorgiaEngine; 