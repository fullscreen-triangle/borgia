//! Probabilistic and fuzzy similarity calculations for molecular comparison.

use crate::error::{BorgiaError, Result};
use crate::molecular::ProbabilisticMolecule;
use crate::probabilistic::{ProbabilisticValue, SimilarityDistribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::Array1;
use crate::molecular::OscillatoryQuantumMolecule;
use crate::oscillatory::{UniversalOscillator, OscillationState};
use crate::entropy::EntropyDistribution;
use crate::quantum::QuantumMolecularComputer;
use crate::representation::HierarchyLevel;

/// Similarity calculation engine
#[derive(Debug, Clone)]
pub struct SimilarityEngine {
    /// Algorithm weights for different similarity types
    pub algorithm_weights: HashMap<String, f64>,
    /// Context-specific parameters
    pub context_params: HashMap<String, f64>,
}

/// Probabilistic similarity result with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticSimilarity {
    /// Mean similarity value
    pub similarity: ProbabilisticValue,
    /// Similarity distribution across linguistic categories
    pub distribution: SimilarityDistribution,
    /// Algorithm used
    pub algorithm: String,
    /// Confidence in the result
    pub confidence: f64,
}

/// Fuzzy similarity with linguistic variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzySimilarity {
    /// Linguistic similarity assessment
    pub linguistic_value: String,
    /// Membership degrees for each category
    pub memberships: HashMap<String, f64>,
    /// Overall fuzzy similarity score
    pub fuzzy_score: f64,
}

/// Similarity calculation algorithms
#[derive(Debug, Clone, Copy)]
pub enum SimilarityAlgorithm {
    Tanimoto,
    Dice,
    Cosine,
    Jaccard,
    ProbabilisticTanimoto,
    FuzzyTanimoto,
    WeightedSimilarity,
}

/// Oscillatory similarity calculator based on synchronization potential
pub struct OscillatorySimilarityCalculator {
    pub synchronization_threshold: f64,
    pub frequency_weight: f64,
    pub amplitude_weight: f64,
    pub phase_weight: f64,
    pub entropy_weight: f64,
}

/// Quantum computational similarity calculator
pub struct QuantumComputationalSimilarityCalculator {
    pub enaqt_weight: f64,
    pub coupling_weight: f64,
    pub coherence_weight: f64,
    pub tunneling_weight: f64,
    pub membrane_weight: f64,
}

impl SimilarityEngine {
    /// Create a new similarity engine
    pub fn new() -> Self {
        let mut algorithm_weights = HashMap::new();
        algorithm_weights.insert("tanimoto".to_string(), 1.0);
        algorithm_weights.insert("pharmacophoric".to_string(), 0.8);
        algorithm_weights.insert("quantum".to_string(), 0.6);
        algorithm_weights.insert("conformational".to_string(), 0.4);

        Self {
            algorithm_weights,
            context_params: HashMap::new(),
        }
    }

    /// Calculate probabilistic similarity between two molecules
    pub fn calculate_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        algorithm: SimilarityAlgorithm,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        match algorithm {
            SimilarityAlgorithm::Tanimoto => self.tanimoto_similarity(mol1, mol2),
            SimilarityAlgorithm::ProbabilisticTanimoto => self.probabilistic_tanimoto(mol1, mol2),
            SimilarityAlgorithm::FuzzyTanimoto => self.fuzzy_tanimoto(mol1, mol2, context),
            SimilarityAlgorithm::WeightedSimilarity => self.weighted_similarity(mol1, mol2, context),
            _ => Err(BorgiaError::similarity(
                format!("{:?}", algorithm),
                "Algorithm not implemented yet",
            )),
        }
    }

    /// Standard Tanimoto similarity
    fn tanimoto_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<ProbabilisticSimilarity> {
        let similarity_value = mol1.fingerprint.tanimoto_similarity(&mol2.fingerprint);
        
        let similarity = ProbabilisticValue::new_normal(
            similarity_value,
            0.05, // Standard uncertainty for Tanimoto
            0.95,
        );

        let distribution = self.similarity_to_distribution(similarity_value)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "tanimoto".to_string(),
            confidence: 0.9,
        })
    }

    /// Probabilistic Tanimoto with uncertainty propagation
    fn probabilistic_tanimoto(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<ProbabilisticSimilarity> {
        // Sample from fingerprint uncertainties
        let n_samples = 1000;
        let mut similarities = Vec::new();

        for _ in 0..n_samples {
            // Add noise to fingerprints based on uncertainties
            let mut fp1_noisy = mol1.fingerprint.combined.clone();
            let mut fp2_noisy = mol2.fingerprint.combined.clone();

            // Add Gaussian noise proportional to uncertainties
            for i in 0..fp1_noisy.len() {
                let noise1 = rand::random::<f64>() * mol1.fingerprint.uncertainties[i];
                let noise2 = rand::random::<f64>() * mol2.fingerprint.uncertainties[i];
                fp1_noisy[i] += noise1;
                fp2_noisy[i] += noise2;
            }

            // Calculate similarity with noisy fingerprints
            let intersection = fp1_noisy.dot(&fp2_noisy);
            let union = fp1_noisy.norm_squared() + fp2_noisy.norm_squared() - intersection;
            
            let sim = if union > 0.0 { intersection / union } else { 0.0 };
            similarities.push(sim.max(0.0).min(1.0)); // Clamp to [0, 1]
        }

        let similarity = ProbabilisticValue::from_samples(similarities, 0.95)?;
        let distribution = self.similarity_to_distribution(similarity.mean)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "probabilistic_tanimoto".to_string(),
            confidence: 0.95,
        })
    }

    /// Fuzzy Tanimoto with context awareness
    fn fuzzy_tanimoto(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        // Calculate different types of similarities
        let topo_sim = self.calculate_topological_similarity(mol1, mol2)?;
        let pharma_sim = self.calculate_pharmacophoric_similarity(mol1, mol2)?;
        let quantum_sim = self.calculate_quantum_similarity(mol1, mol2)?;

        // Context-dependent weighting
        let weights = self.get_context_weights(context);
        
        let weighted_similarity = 
            topo_sim * weights.get("topological").unwrap_or(&1.0) +
            pharma_sim * weights.get("pharmacophoric").unwrap_or(&0.5) +
            quantum_sim * weights.get("quantum").unwrap_or(&0.3);

        let total_weight = weights.values().sum::<f64>();
        let normalized_similarity = weighted_similarity / total_weight;

        let similarity = ProbabilisticValue::new_normal(
            normalized_similarity,
            0.1, // Higher uncertainty for fuzzy calculations
            0.85,
        );

        let distribution = self.similarity_to_distribution(normalized_similarity)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "fuzzy_tanimoto".to_string(),
            confidence: 0.85,
        })
    }

    /// Weighted similarity with adaptive feature importance
    fn weighted_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
        context: &str,
    ) -> Result<ProbabilisticSimilarity> {
        let similarity_value = mol1.fingerprint.weighted_similarity(&mol2.fingerprint);
        
        // Adjust similarity based on molecular properties
        let prop_similarity = self.calculate_property_similarity(mol1, mol2)?;
        let combined_similarity = (similarity_value + prop_similarity) / 2.0;

        let similarity = ProbabilisticValue::new_normal(
            combined_similarity,
            0.08,
            0.90,
        );

        let distribution = self.similarity_to_distribution(combined_similarity)?;

        Ok(ProbabilisticSimilarity {
            similarity,
            distribution,
            algorithm: "weighted_similarity".to_string(),
            confidence: 0.90,
        })
    }

    /// Calculate topological similarity
    fn calculate_topological_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.topological.dot(&mol2.fingerprint.topological);
        let union = mol1.fingerprint.topological.norm_squared() + 
                   mol2.fingerprint.topological.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate pharmacophoric similarity
    fn calculate_pharmacophoric_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.pharmacophoric.dot(&mol2.fingerprint.pharmacophoric);
        let union = mol1.fingerprint.pharmacophoric.norm_squared() + 
                   mol2.fingerprint.pharmacophoric.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate quantum similarity
    fn calculate_quantum_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        let intersection = mol1.fingerprint.quantum.dot(&mol2.fingerprint.quantum);
        let union = mol1.fingerprint.quantum.norm_squared() + 
                   mol2.fingerprint.quantum.norm_squared() - intersection;
        
        Ok(if union > 0.0 { intersection / union } else { 0.0 })
    }

    /// Calculate property-based similarity
    fn calculate_property_similarity(
        &self,
        mol1: &ProbabilisticMolecule,
        mol2: &ProbabilisticMolecule,
    ) -> Result<f64> {
        // Compare molecular properties with uncertainty
        let mw_diff = (mol1.properties.molecular_weight.mean - mol2.properties.molecular_weight.mean).abs();
        let mw_uncertainty = mol1.properties.molecular_weight.std_dev + mol2.properties.molecular_weight.std_dev;
        let mw_sim = 1.0 - (mw_diff / (mw_diff + mw_uncertainty + 1.0));

        let logp_diff = (mol1.properties.logp.mean - mol2.properties.logp.mean).abs();
        let logp_uncertainty = mol1.properties.logp.std_dev + mol2.properties.logp.std_dev;
        let logp_sim = 1.0 - (logp_diff / (logp_diff + logp_uncertainty + 1.0));

        // Average property similarities
        Ok((mw_sim + logp_sim) / 2.0)
    }

    /// Get context-specific weights
    fn get_context_weights(&self, context: &str) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        
        match context {
            "drug_metabolism" => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.8);
                weights.insert("quantum".to_string(), 0.6);
            }
            "structural_similarity" => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.3);
                weights.insert("quantum".to_string(), 0.2);
            }
            "binding_affinity" => {
                weights.insert("topological".to_string(), 0.7);
                weights.insert("pharmacophoric".to_string(), 1.0);
                weights.insert("quantum".to_string(), 0.8);
            }
            _ => {
                weights.insert("topological".to_string(), 1.0);
                weights.insert("pharmacophoric".to_string(), 0.5);
                weights.insert("quantum".to_string(), 0.3);
            }
        }
        
        weights
    }

    /// Convert similarity value to linguistic distribution
    fn similarity_to_distribution(&self, similarity: f64) -> Result<SimilarityDistribution> {
        // Convert continuous similarity to fuzzy linguistic categories
        let very_low = if similarity < 0.2 { 1.0 - similarity * 5.0 } else { 0.0 }.max(0.0);
        let low = if similarity >= 0.1 && similarity < 0.4 {
            if similarity < 0.25 { (similarity - 0.1) * 6.67 } else { (0.4 - similarity) * 6.67 }
        } else { 0.0 }.max(0.0);
        let medium = if similarity >= 0.3 && similarity < 0.7 {
            if similarity < 0.5 { (similarity - 0.3) * 5.0 } else { (0.7 - similarity) * 5.0 }
        } else { 0.0 }.max(0.0);
        let high = if similarity >= 0.6 && similarity < 0.9 {
            if similarity < 0.75 { (similarity - 0.6) * 6.67 } else { (0.9 - similarity) * 6.67 }
        } else { 0.0 }.max(0.0);
        let very_high = if similarity > 0.8 { (similarity - 0.8) * 5.0 } else { 0.0 }.max(0.0).min(1.0);

        // Normalize to ensure sum = 1.0
        let total = very_low + low + medium + high + very_high;
        if total > 0.0 {
            SimilarityDistribution::new(
                very_low / total,
                low / total,
                medium / total,
                high / total,
                very_high / total,
            )
        } else {
            SimilarityDistribution::new(1.0, 0.0, 0.0, 0.0, 0.0)
        }
    }
}

impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OscillatorySimilarityCalculator {
    pub fn new() -> Self {
        Self {
            synchronization_threshold: 0.1,
            frequency_weight: 0.3,
            amplitude_weight: 0.2,
            phase_weight: 0.2,
            entropy_weight: 0.3,
        }
    }
    
    /// Calculate similarity based on oscillatory synchronization potential
    /// Following the Observer Synchronization Theorem
    pub fn oscillatory_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Calculate frequency difference
        let freq_diff = (mol1.oscillatory_state.natural_frequency - mol2.oscillatory_state.natural_frequency).abs();
        
        if freq_diff < self.synchronization_threshold {
            // Molecules can synchronize - calculate information transfer rate
            let phase_difference = self.calculate_phase_relationship(mol1, mol2);
            let coupling_strength = self.calculate_coupling_strength(mol1, mol2);
            let info_transfer_rate = coupling_strength * phase_difference.cos();
            
            // Similarity based on synchronization strength
            (-freq_diff / self.synchronization_threshold).exp() * info_transfer_rate.abs()
        } else {
            // No synchronization possible
            0.0
        }
    }
    
    /// Calculate entropy endpoint similarity
    /// Molecules with similar "landing patterns" are similar
    pub fn entropy_endpoint_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Compare endpoint distributions using Wasserstein distance
        let endpoint_distance = self.wasserstein_distance(
            &mol1.entropy_distribution.landing_probabilities,
            &mol2.entropy_distribution.landing_probabilities
        );
        
        // Convert distance to similarity
        (-endpoint_distance).exp()
    }
    
    /// Multi-scale similarity across the nested hierarchy
    pub fn nested_hierarchy_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> HashMap<u8, f64> {
        let mut similarities = HashMap::new();
        
        // Compare at each hierarchy level
        for (level, rep1) in &mol1.hierarchy_representations {
            if let Some(rep2) = mol2.hierarchy_representations.get(level) {
                let similarity = self.compare_hierarchy_level(rep1, rep2);
                similarities.insert(*level, similarity);
            }
        }
        
        similarities
    }
    
    /// Calculate phase relationship between two oscillators
    fn calculate_phase_relationship(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        mol1.oscillatory_state.current_state.phase - mol2.oscillatory_state.current_state.phase
    }
    
    /// Calculate coupling strength between oscillators
    fn calculate_coupling_strength(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Coupling strength based on molecular properties and spatial proximity
        let structural_coupling = self.calculate_structural_coupling(mol1, mol2);
        let electronic_coupling = self.calculate_electronic_coupling(mol1, mol2);
        let vibrational_coupling = self.calculate_vibrational_coupling(mol1, mol2);
        
        (structural_coupling * electronic_coupling * vibrational_coupling).sqrt()
    }
    
    /// Calculate Wasserstein distance between probability distributions
    fn wasserstein_distance(&self, dist1: &Array1<f64>, dist2: &Array1<f64>) -> f64 {
        // Simplified Wasserstein distance calculation
        let mut cumsum1 = 0.0;
        let mut cumsum2 = 0.0;
        let mut distance = 0.0;
        
        for i in 0..dist1.len().min(dist2.len()) {
            cumsum1 += dist1[i];
            cumsum2 += dist2[i];
            distance += (cumsum1 - cumsum2).abs();
        }
        
        distance / dist1.len() as f64
    }
    
    /// Compare similarity at specific hierarchy level
    fn compare_hierarchy_level(&self, level1: &HierarchyLevel, level2: &HierarchyLevel) -> f64 {
        let freq_similarity = 1.0 - (level1.characteristic_frequency - level2.characteristic_frequency).abs() / 
                             level1.characteristic_frequency.max(level2.characteristic_frequency);
        let amplitude_similarity = 1.0 - (level1.oscillation_amplitude - level2.oscillation_amplitude).abs() /
                                  level1.oscillation_amplitude.max(level2.oscillation_amplitude);
        
        (freq_similarity + amplitude_similarity) / 2.0
    }
    
    /// Calculate structural coupling between molecules
    fn calculate_structural_coupling(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Simplified structural coupling based on molecular size and shape
        let size_factor = (mol1.molecular_weight / mol2.molecular_weight).min(mol2.molecular_weight / mol1.molecular_weight);
        size_factor.sqrt()
    }
    
    /// Calculate electronic coupling between molecules
    fn calculate_electronic_coupling(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Electronic coupling based on quantum computational properties
        let efficiency_similarity = 1.0 - (mol1.quantum_computer.transport_efficiency - mol2.quantum_computer.transport_efficiency).abs();
        let coherence_similarity = 1.0 - (mol1.quantum_computer.coherence_time - mol2.quantum_computer.coherence_time).abs() /
                                  mol1.quantum_computer.coherence_time.max(mol2.quantum_computer.coherence_time);
        
        (efficiency_similarity + coherence_similarity) / 2.0
    }
    
    /// Calculate vibrational coupling between molecules
    fn calculate_vibrational_coupling(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Vibrational coupling based on oscillatory properties
        let damping_similarity = 1.0 - (mol1.oscillatory_state.damping_coefficient - mol2.oscillatory_state.damping_coefficient).abs();
        damping_similarity
    }
}

impl QuantumComputationalSimilarityCalculator {
    pub fn new() -> Self {
        Self {
            enaqt_weight: 0.3,
            coupling_weight: 0.25,
            coherence_weight: 0.25,
            tunneling_weight: 0.2,
            membrane_weight: 0.2,
        }
    }
    
    /// Calculate similarity based on quantum computational architecture
    /// Following the Membrane Quantum Computation Theorem
    pub fn quantum_computational_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        // Compare ENAQT capabilities
        let enaqt_similarity = self.compare_enaqt_architectures(mol1, mol2);
        
        // Compare environmental coupling optimization
        let coupling_similarity = self.compare_coupling_optimization(mol1, mol2);
        
        // Compare quantum coherence properties
        let coherence_similarity = self.compare_coherence_properties(mol1, mol2);
        
        // Compare tunneling pathway architectures
        let tunneling_similarity = self.compare_tunneling_pathways(mol1, mol2);
        
        // Compare membrane-like properties
        let membrane_similarity = self.compare_membrane_properties(mol1, mol2);
        
        // Weighted combination
        self.enaqt_weight * enaqt_similarity +
        self.coupling_weight * coupling_similarity +
        self.coherence_weight * coherence_similarity +
        self.tunneling_weight * tunneling_similarity +
        self.membrane_weight * membrane_similarity
    }
    
    /// Compare Environment-Assisted Quantum Transport capabilities
    pub fn compare_enaqt_architectures(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let eta1 = mol1.quantum_computer.transport_efficiency;
        let eta2 = mol2.quantum_computer.transport_efficiency;
        
        let gamma_opt1 = mol1.quantum_computer.optimal_coupling;
        let gamma_opt2 = mol2.quantum_computer.optimal_coupling;
        
        // Similarity based on quantum advantage
        let efficiency_similarity = 1.0 - (eta1 - eta2).abs() / eta1.max(eta2);
        let coupling_similarity = 1.0 - (gamma_opt1 - gamma_opt2).abs() / gamma_opt1.max(gamma_opt2);
        
        (efficiency_similarity + coupling_similarity) / 2.0
    }
    
    /// Compare environmental coupling optimization
    fn compare_coupling_optimization(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let coupling1 = mol1.quantum_computer.environmental_coupling_strength;
        let coupling2 = mol2.quantum_computer.environmental_coupling_strength;
        
        1.0 - (coupling1 - coupling2).abs() / coupling1.max(coupling2)
    }
    
    /// Compare quantum coherence properties
    fn compare_coherence_properties(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let coherence1 = mol1.quantum_computer.coherence_time;
        let coherence2 = mol2.quantum_computer.coherence_time;
        
        let beating1 = &mol1.quantum_computer.quantum_beating_frequencies;
        let beating2 = &mol2.quantum_computer.quantum_beating_frequencies;
        
        let coherence_similarity = 1.0 - (coherence1 - coherence2).abs() / coherence1.max(coherence2);
        let beating_similarity = self.compare_frequency_spectra(beating1, beating2);
        
        (coherence_similarity + beating_similarity) / 2.0
    }
    
    /// Compare tunneling pathway architectures
    fn compare_tunneling_pathways(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let pathways1 = &mol1.quantum_computer.tunneling_pathways;
        let pathways2 = &mol2.quantum_computer.tunneling_pathways;
        
        if pathways1.is_empty() && pathways2.is_empty() {
            return 1.0;
        }
        
        if pathways1.is_empty() || pathways2.is_empty() {
            return 0.0;
        }
        
        // Compare pathway characteristics
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        for pathway1 in pathways1 {
            for pathway2 in pathways2 {
                let barrier_similarity = 1.0 - (pathway1.barrier_height - pathway2.barrier_height).abs() / 
                                       pathway1.barrier_height.max(pathway2.barrier_height);
                let width_similarity = 1.0 - (pathway1.barrier_width - pathway2.barrier_width).abs() /
                                     pathway1.barrier_width.max(pathway2.barrier_width);
                let prob_similarity = 1.0 - (pathway1.tunneling_probability - pathway2.tunneling_probability).abs();
                
                total_similarity += (barrier_similarity + width_similarity + prob_similarity) / 3.0;
                count += 1;
            }
        }
        
        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }
    
    /// Compare membrane-like properties
    fn compare_membrane_properties(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let mem1 = &mol1.quantum_computer.membrane_properties;
        let mem2 = &mol2.quantum_computer.membrane_properties;
        
        let amphipathic_similarity = 1.0 - (mem1.amphipathic_score - mem2.amphipathic_score).abs();
        let assembly_similarity = 1.0 - (mem1.self_assembly_free_energy - mem2.self_assembly_free_energy).abs() / 100.0;
        let cmc_similarity = if mem1.critical_micelle_concentration > 0.0 && mem2.critical_micelle_concentration > 0.0 {
            1.0 - (mem1.critical_micelle_concentration.ln() - mem2.critical_micelle_concentration.ln()).abs() / 10.0
        } else {
            0.0
        };
        let coherence_similarity = 1.0 - (mem1.room_temp_coherence_potential - mem2.room_temp_coherence_potential).abs();
        
        (amphipathic_similarity + assembly_similarity + cmc_similarity + coherence_similarity) / 4.0
    }
    
    /// Compare frequency spectra
    fn compare_frequency_spectra(&self, freq1: &Array1<f64>, freq2: &Array1<f64>) -> f64 {
        if freq1.len() != freq2.len() {
            return 0.0;
        }
        
        let mut similarity = 0.0;
        for i in 0..freq1.len() {
            similarity += 1.0 - (freq1[i] - freq2[i]).abs() / freq1[i].max(freq2[i]);
        }
        
        similarity / freq1.len() as f64
    }
    
    /// Assess how membrane-like molecules are (quantum computational capability)
    pub fn membrane_like_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let membrane_scores = [mol1, mol2].iter().map(|mol| {
            let mut score = 0.0;
            
            // Amphipathic character (enables self-assembly)
            if mol.quantum_computer.membrane_properties.amphipathic_score > 0.5 {
                score += 0.2;
            }
            
            // Quantum tunneling pathways quality
            let tunneling_quality = self.assess_tunneling_pathway_quality(mol);
            score += 0.3 * tunneling_quality;
            
            // Environmental coupling optimization
            let coupling_optimization = self.assess_coupling_optimization(mol);
            score += 0.3 * coupling_optimization;
            
            // Room temperature quantum coherence potential
            score += 0.2 * mol.quantum_computer.membrane_properties.room_temp_coherence_potential;
            
            score
        }).collect::<Vec<_>>();
        
        // Similarity based on both being membrane-like
        1.0 - (membrane_scores[0] - membrane_scores[1]).abs()
    }
    
    /// Compare radical generation potential (death-causing quantum leakage)
    /// Following the Radical Inevitability Theorem
    pub fn death_inevitability_similarity(&self, mol1: &OscillatoryQuantumMolecule, mol2: &OscillatoryQuantumMolecule) -> f64 {
        let radical_rate1 = mol1.quantum_computer.radical_generation_rate;
        let radical_rate2 = mol2.quantum_computer.radical_generation_rate;
        
        // Molecules with similar death-causing potential are similar
        if radical_rate1 == 0.0 && radical_rate2 == 0.0 {
            1.0
        } else if radical_rate1 == 0.0 || radical_rate2 == 0.0 {
            0.0
        } else {
            1.0 - (radical_rate1.ln() - radical_rate2.ln()).abs() / 10.0
        }
    }
    
    /// Assess tunneling pathway quality
    fn assess_tunneling_pathway_quality(&self, mol: &OscillatoryQuantumMolecule) -> f64 {
        if mol.quantum_computer.tunneling_pathways.is_empty() {
            return 0.0;
        }
        
        let mut quality_score = 0.0;
        for pathway in &mol.quantum_computer.tunneling_pathways {
            // Optimal tunneling occurs at 3-5 nm distances
            let distance_score = if pathway.barrier_width >= 3.0 && pathway.barrier_width <= 5.0 {
                1.0
            } else if pathway.barrier_width < 3.0 {
                pathway.barrier_width / 3.0
            } else {
                5.0 / pathway.barrier_width
            };
            
            // Higher tunneling probability is better
            let probability_score = pathway.tunneling_probability;
            
            // Environmental enhancement is beneficial
            let enhancement_score = pathway.environmental_enhancement.min(1.0);
            
            quality_score += (distance_score + probability_score + enhancement_score) / 3.0;
        }
        
        quality_score / mol.quantum_computer.tunneling_pathways.len() as f64
    }
    
    /// Assess coupling optimization
    fn assess_coupling_optimization(&self, mol: &OscillatoryQuantumMolecule) -> f64 {
        let actual_coupling = mol.quantum_computer.environmental_coupling_strength;
        let optimal_coupling = mol.quantum_computer.optimal_coupling;
        
        if optimal_coupling > 0.0 {
            1.0 - (actual_coupling - optimal_coupling).abs() / optimal_coupling
        } else {
            0.0
        }
    }
}

/// Comprehensive similarity result combining all frameworks
#[derive(Clone, Debug)]
pub struct ComprehensiveSimilarityResult {
    pub oscillatory_similarity: f64,
    pub quantum_computational_similarity: f64,
    pub enaqt_similarity: f64,
    pub membrane_similarity: f64,
    pub death_inevitability_similarity: f64,
    pub entropy_endpoint_similarity: f64,
    pub hierarchical_similarities: HashMap<u8, f64>,
    pub overall_similarity: f64,
}

impl Default for ComprehensiveSimilarityResult {
    fn default() -> Self {
        Self {
            oscillatory_similarity: 0.0,
            quantum_computational_similarity: 0.0,
            enaqt_similarity: 0.0,
            membrane_similarity: 0.0,
            death_inevitability_similarity: 0.0,
            entropy_endpoint_similarity: 0.0,
            hierarchical_similarities: HashMap::new(),
            overall_similarity: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_engine() {
        let engine = SimilarityEngine::new();
        assert!(!engine.algorithm_weights.is_empty());
    }

    #[test]
    fn test_tanimoto_similarity() {
        let engine = SimilarityEngine::new();
        let mol1 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        let mol2 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        
        let result = engine.calculate_similarity(
            &mol1, &mol2, SimilarityAlgorithm::Tanimoto, "test"
        ).unwrap();
        
        assert!(result.similarity.mean > 0.8); // Should be very similar
        assert_eq!(result.algorithm, "tanimoto");
    }

    #[test]
    fn test_similarity_distribution() {
        let engine = SimilarityEngine::new();
        let dist = engine.similarity_to_distribution(0.8).unwrap();
        
        assert_eq!(dist.most_likely(), "very_high");
        assert!(dist.expected_value() > 0.7);
    }

    #[test]
    fn test_different_molecules() {
        let engine = SimilarityEngine::new();
        let mol1 = ProbabilisticMolecule::from_smiles("CCO").unwrap();
        let mol2 = ProbabilisticMolecule::from_smiles("CCCCCCCC").unwrap();
        
        let result = engine.calculate_similarity(
            &mol1, &mol2, SimilarityAlgorithm::Tanimoto, "test"
        ).unwrap();
        
        assert!(result.similarity.mean < 0.8); // Should be less similar
    }
} 