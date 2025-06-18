//! Evidence processing and propagation for Borgia.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::{ProbabilisticValue, Evidence};
use crate::molecular::ProbabilisticMolecule;
use crate::core::{EvidenceType, UpstreamSystem};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Evidence processor for handling upstream information
#[derive(Debug, Clone)]
pub struct EvidenceProcessor {
    pub evidence_cache: HashMap<String, Evidence>,
    pub propagation_rules: HashMap<String, f64>,
    pub confidence_thresholds: HashMap<UpstreamSystem, f64>,
}

/// Context for evidence evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceContext {
    pub evidence_type: EvidenceType,
    pub upstream_system: UpstreamSystem,
    pub context: String,
    pub confidence_level: f64,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

/// Evidence propagation system
#[derive(Debug, Clone)]
pub struct EvidencePropagation {
    pub propagation_rules: HashMap<String, f64>,
    pub decay_factors: HashMap<EvidenceType, f64>,
    pub fusion_weights: HashMap<UpstreamSystem, f64>,
}

/// Evidence strength assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceStrength {
    pub raw_strength: f64,
    pub adjusted_strength: f64,
    pub confidence: f64,
    pub reliability: f64,
    pub supporting_factors: Vec<String>,
    pub limiting_factors: Vec<String>,
}

/// Evidence fusion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceFusionResult {
    pub fused_strength: f64,
    pub combined_confidence: f64,
    pub consistency_score: f64,
    pub source_contributions: HashMap<String, f64>,
    pub uncertainty_factors: Vec<String>,
}

impl EvidenceProcessor {
    pub fn new() -> Self {
        let mut confidence_thresholds = HashMap::new();
        confidence_thresholds.insert(UpstreamSystem::Hegel, 0.85);
        confidence_thresholds.insert(UpstreamSystem::Lavoisier, 0.80);
        confidence_thresholds.insert(UpstreamSystem::Gospel, 0.90);
        confidence_thresholds.insert(UpstreamSystem::BeneGesserit, 0.95);

        Self {
            evidence_cache: HashMap::new(),
            propagation_rules: HashMap::new(),
            confidence_thresholds,
        }
    }

    /// Add evidence to the cache
    pub fn add_evidence(&mut self, key: String, evidence: Evidence) {
        self.evidence_cache.insert(key, evidence);
    }

    /// Get evidence from cache
    pub fn get_evidence(&self, key: &str) -> Option<&Evidence> {
        self.evidence_cache.get(key)
    }

    /// Enhance molecule with evidence-based modifications
    pub fn enhance_molecule(
        &self,
        molecule: &ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<ProbabilisticMolecule> {
        let mut enhanced = molecule.clone();

        // Apply evidence-based enhancements
        match evidence_context.evidence_type {
            EvidenceType::StructuralSimilarity => {
                self.enhance_structural_features(&mut enhanced, evidence_context)?;
            }
            EvidenceType::PharmacologicalActivity => {
                self.enhance_pharmacological_features(&mut enhanced, evidence_context)?;
            }
            EvidenceType::MetabolicPathway => {
                self.enhance_metabolic_features(&mut enhanced, evidence_context)?;
            }
            EvidenceType::MolecularInteraction => {
                self.enhance_interaction_features(&mut enhanced, evidence_context)?;
            }
            EvidenceType::PropertyPrediction => {
                self.enhance_property_features(&mut enhanced, evidence_context)?;
            }
        }

        Ok(enhanced)
    }

    /// Assess evidence strength
    pub fn assess_evidence_strength(
        &self,
        evidence_context: &EvidenceContext,
        molecular_data: &[ProbabilisticMolecule],
    ) -> Result<EvidenceStrength> {
        let base_strength = self.calculate_base_strength(evidence_context)?;
        let reliability = self.assess_source_reliability(&evidence_context.upstream_system);
        let data_quality = self.assess_data_quality(molecular_data)?;

        let adjusted_strength = base_strength * reliability * data_quality;
        let confidence = self.calculate_evidence_confidence(evidence_context, adjusted_strength)?;

        let supporting_factors = self.identify_supporting_factors(evidence_context, molecular_data);
        let limiting_factors = self.identify_limiting_factors(evidence_context, molecular_data);

        Ok(EvidenceStrength {
            raw_strength: base_strength,
            adjusted_strength,
            confidence,
            reliability,
            supporting_factors,
            limiting_factors,
        })
    }

    /// Fuse multiple evidence sources
    pub fn fuse_evidence(
        &self,
        evidence_sources: &[EvidenceContext],
        molecular_data: &[ProbabilisticMolecule],
    ) -> Result<EvidenceFusionResult> {
        if evidence_sources.is_empty() {
            return Err(BorgiaError::validation("evidence_sources", "No evidence sources provided"));
        }

        let mut strengths = Vec::new();
        let mut confidences = Vec::new();
        let mut source_contributions = HashMap::new();

        // Assess each evidence source
        for context in evidence_sources {
            let strength = self.assess_evidence_strength(context, molecular_data)?;
            strengths.push(strength.adjusted_strength);
            confidences.push(strength.confidence);
            
            let source_key = format!("{:?}_{}", context.upstream_system, context.evidence_type as u8);
            source_contributions.insert(source_key, strength.adjusted_strength);
        }

        // Fuse evidence using weighted combination
        let total_confidence: f64 = confidences.iter().sum();
        let fused_strength = if total_confidence > 0.0 {
            strengths.iter()
                .zip(confidences.iter())
                .map(|(s, c)| s * c)
                .sum::<f64>() / total_confidence
        } else {
            0.0
        };

        let combined_confidence = total_confidence / evidence_sources.len() as f64;
        let consistency_score = self.calculate_consistency_score(&strengths);

        let uncertainty_factors = self.identify_uncertainty_factors(evidence_sources, molecular_data);

        Ok(EvidenceFusionResult {
            fused_strength,
            combined_confidence,
            consistency_score,
            source_contributions,
            uncertainty_factors,
        })
    }

    // Private helper methods

    fn enhance_structural_features(
        &self,
        molecule: &mut ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Enhance structural features based on evidence
        // This would modify the molecule's probabilistic properties
        // based on structural similarity evidence
        Ok(())
    }

    fn enhance_pharmacological_features(
        &self,
        molecule: &mut ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Enhance pharmacological features
        Ok(())
    }

    fn enhance_metabolic_features(
        &self,
        molecule: &mut ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Enhance metabolic pathway features
        Ok(())
    }

    fn enhance_interaction_features(
        &self,
        molecule: &mut ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Enhance molecular interaction features
        Ok(())
    }

    fn enhance_property_features(
        &self,
        molecule: &mut ProbabilisticMolecule,
        evidence_context: &EvidenceContext,
    ) -> Result<()> {
        // Enhance property prediction features
        Ok(())
    }

    fn calculate_base_strength(&self, evidence_context: &EvidenceContext) -> Result<f64> {
        let base_strength = match evidence_context.evidence_type {
            EvidenceType::StructuralSimilarity => 0.8,
            EvidenceType::PharmacologicalActivity => 0.9,
            EvidenceType::MetabolicPathway => 0.7,
            EvidenceType::MolecularInteraction => 0.85,
            EvidenceType::PropertyPrediction => 0.75,
        };

        Ok(base_strength * evidence_context.confidence_level)
    }

    fn assess_source_reliability(&self, upstream_system: &UpstreamSystem) -> f64 {
        self.confidence_thresholds.get(upstream_system).copied().unwrap_or(0.5)
    }

    fn assess_data_quality(&self, molecular_data: &[ProbabilisticMolecule]) -> Result<f64> {
        if molecular_data.is_empty() {
            return Ok(0.0);
        }

        let avg_confidence = molecular_data.iter()
            .map(|mol| mol.properties.molecular_weight.confidence)
            .sum::<f64>() / molecular_data.len() as f64;

        Ok(avg_confidence)
    }

    fn calculate_evidence_confidence(
        &self,
        evidence_context: &EvidenceContext,
        adjusted_strength: f64,
    ) -> Result<f64> {
        let time_factor = self.calculate_time_decay_factor(evidence_context.timestamp);
        let context_factor = if evidence_context.context.is_empty() { 0.8 } else { 1.0 };
        
        Ok(adjusted_strength * time_factor * context_factor)
    }

    fn calculate_time_decay_factor(&self, timestamp: u64) -> f64 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let age_hours = (current_time.saturating_sub(timestamp)) / 3600;
        
        // Exponential decay with half-life of 24 hours
        let decay_rate = 0.693 / 24.0; // ln(2) / half_life
        (-decay_rate * age_hours as f64).exp()
    }

    fn identify_supporting_factors(
        &self,
        evidence_context: &EvidenceContext,
        molecular_data: &[ProbabilisticMolecule],
    ) -> Vec<String> {
        let mut factors = Vec::new();

        if evidence_context.confidence_level > 0.8 {
            factors.push("High confidence level".to_string());
        }

        if molecular_data.len() > 10 {
            factors.push("Large dataset".to_string());
        }

        match evidence_context.upstream_system {
            UpstreamSystem::Gospel => factors.push("Highly reliable source".to_string()),
            UpstreamSystem::BeneGesserit => factors.push("Advanced analysis system".to_string()),
            _ => {}
        }

        factors
    }

    fn identify_limiting_factors(
        &self,
        evidence_context: &EvidenceContext,
        molecular_data: &[ProbabilisticMolecule],
    ) -> Vec<String> {
        let mut factors = Vec::new();

        if evidence_context.confidence_level < 0.5 {
            factors.push("Low confidence level".to_string());
        }

        if molecular_data.len() < 3 {
            factors.push("Small dataset".to_string());
        }

        if evidence_context.context.is_empty() {
            factors.push("Lack of context information".to_string());
        }

        factors
    }

    fn calculate_consistency_score(&self, strengths: &[f64]) -> f64 {
        if strengths.len() < 2 {
            return 1.0;
        }

        let mean = strengths.iter().sum::<f64>() / strengths.len() as f64;
        let variance = strengths.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / strengths.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Consistency score: higher when standard deviation is lower
        (1.0 - std_dev).max(0.0)
    }

    fn identify_uncertainty_factors(
        &self,
        evidence_sources: &[EvidenceContext],
        molecular_data: &[ProbabilisticMolecule],
    ) -> Vec<String> {
        let mut factors = Vec::new();

        if evidence_sources.len() < 2 {
            factors.push("Single evidence source".to_string());
        }

        let avg_confidence = evidence_sources.iter()
            .map(|ctx| ctx.confidence_level)
            .sum::<f64>() / evidence_sources.len() as f64;

        if avg_confidence < 0.7 {
            factors.push("Low average confidence".to_string());
        }

        if molecular_data.len() < 5 {
            factors.push("Limited molecular data".to_string());
        }

        factors
    }
}

impl EvidenceContext {
    /// Create a new evidence context
    pub fn new(
        evidence_type: EvidenceType,
        upstream_system: UpstreamSystem,
        context: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            evidence_type,
            upstream_system,
            context,
            confidence_level: 0.8, // Default confidence
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the context
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Set confidence level
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence_level = confidence.max(0.0).min(1.0);
    }
}

impl EvidencePropagation {
    pub fn new() -> Self {
        let mut decay_factors = HashMap::new();
        decay_factors.insert(EvidenceType::StructuralSimilarity, 0.95);
        decay_factors.insert(EvidenceType::PharmacologicalActivity, 0.90);
        decay_factors.insert(EvidenceType::MetabolicPathway, 0.85);
        decay_factors.insert(EvidenceType::MolecularInteraction, 0.92);
        decay_factors.insert(EvidenceType::PropertyPrediction, 0.88);

        let mut fusion_weights = HashMap::new();
        fusion_weights.insert(UpstreamSystem::Hegel, 0.8);
        fusion_weights.insert(UpstreamSystem::Lavoisier, 0.85);
        fusion_weights.insert(UpstreamSystem::Gospel, 0.95);
        fusion_weights.insert(UpstreamSystem::BeneGesserit, 1.0);

        Self {
            propagation_rules: HashMap::new(),
            decay_factors,
            fusion_weights,
        }
    }

    /// Propagate evidence through the system
    pub fn propagate_evidence(
        &self,
        evidence_strength: &EvidenceStrength,
        evidence_type: &EvidenceType,
        distance: usize,
    ) -> f64 {
        let decay_factor = self.decay_factors.get(evidence_type).copied().unwrap_or(0.9);
        let distance_decay = decay_factor.powi(distance as i32);
        
        evidence_strength.adjusted_strength * distance_decay
    }
}

impl Default for EvidenceProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for EvidencePropagation {
    fn default() -> Self {
        Self::new()
    }
} 