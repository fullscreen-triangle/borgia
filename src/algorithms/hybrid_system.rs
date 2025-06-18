use std::collections::HashMap;
use crate::error::BorgiaResult;
use crate::probabilistic::ProbabilisticValue;
use crate::algorithms::morgan::{MorganAlgorithm, MolecularGraph};
use crate::algorithms::vf2::VF2Algorithm;
use crate::algorithms::ullmann::UllmannAlgorithm;

/// Hybrid algorithm system that combines multiple graph isomorphism algorithms
/// Uses probabilistic decision making to select the best approach for each problem
#[derive(Debug, Clone)]
pub struct HybridAlgorithmSystem {
    /// Morgan algorithm instance
    morgan: MorganAlgorithm,
    /// VF2 algorithm instance
    vf2: VF2Algorithm,
    /// Ullmann algorithm instance
    ullmann: UllmannAlgorithm,
    /// Confidence threshold for algorithm selection
    confidence_threshold: f64,
    /// Whether to use ensemble methods
    use_ensemble: bool,
}

impl Default for HybridAlgorithmSystem {
    fn default() -> Self {
        Self {
            morgan: MorganAlgorithm::default(),
            vf2: VF2Algorithm::default(),
            ullmann: UllmannAlgorithm::default(),
            confidence_threshold: 0.8,
            use_ensemble: true,
        }
    }
}

/// Result from hybrid algorithm analysis
#[derive(Debug, Clone)]
pub struct HybridAnalysisResult {
    /// Primary algorithm used
    pub primary_algorithm: AlgorithmType,
    /// Isomorphism result
    pub is_isomorphic: bool,
    /// Confidence in the result
    pub confidence: ProbabilisticValue,
    /// Best mapping found (if any)
    pub best_mapping: Option<HashMap<usize, usize>>,
    /// Results from individual algorithms
    pub algorithm_results: HashMap<AlgorithmType, AlgorithmResult>,
    /// Ensemble consensus
    pub ensemble_consensus: Option<ProbabilisticValue>,
}

/// Individual algorithm result
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    pub is_isomorphic: bool,
    pub confidence: ProbabilisticValue,
    pub mapping: Option<HashMap<usize, usize>>,
    pub execution_time: f64,
}

/// Algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlgorithmType {
    Morgan,
    VF2,
    Ullmann,
    Hybrid,
}

impl std::fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmType::Morgan => write!(f, "Morgan"),
            AlgorithmType::VF2 => write!(f, "VF2"),
            AlgorithmType::Ullmann => write!(f, "Ullmann"),
            AlgorithmType::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Problem characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    pub query_size: usize,
    pub target_size: usize,
    pub query_density: f64,
    pub target_density: f64,
    pub size_ratio: f64,
    pub complexity_estimate: f64,
    pub uncertainty_level: f64,
}

impl HybridAlgorithmSystem {
    pub fn new(confidence_threshold: f64, use_ensemble: bool) -> Self {
        Self {
            morgan: MorganAlgorithm::default(),
            vf2: VF2Algorithm::default(),
            ullmann: UllmannAlgorithm::default(),
            confidence_threshold,
            use_ensemble,
        }
    }

    /// Comprehensive analysis using hybrid approach
    pub fn analyze_isomorphism(
        &self,
        query: &mut MolecularGraph,
        target: &mut MolecularGraph,
    ) -> BorgiaResult<HybridAnalysisResult> {
        // Analyze problem characteristics
        let characteristics = self.analyze_problem_characteristics(query, target);
        
        // Select primary algorithm
        let primary_algorithm = self.select_primary_algorithm(&characteristics);
        
        // Run individual algorithms
        let mut algorithm_results = HashMap::new();
        
        // Always run Morgan for fingerprinting
        let morgan_result = self.run_morgan_analysis(query, target)?;
        algorithm_results.insert(AlgorithmType::Morgan, morgan_result);
        
        // Run primary algorithm if not Morgan
        if primary_algorithm != AlgorithmType::Morgan {
            match primary_algorithm {
                AlgorithmType::VF2 => {
                    let vf2_result = self.run_vf2_analysis(query, target)?;
                    algorithm_results.insert(AlgorithmType::VF2, vf2_result);
                }
                AlgorithmType::Ullmann => {
                    let ullmann_result = self.run_ullmann_analysis(query, target)?;
                    algorithm_results.insert(AlgorithmType::Ullmann, ullmann_result);
                }
                _ => {}
            }
        }
        
        // Run ensemble if enabled
        let ensemble_consensus = if self.use_ensemble {
            self.run_ensemble_analysis(query, target, &mut algorithm_results)?
        } else {
            None
        };
        
        // Determine final result
        let (is_isomorphic, confidence, best_mapping) = self.determine_final_result(
            &algorithm_results,
            &ensemble_consensus,
            primary_algorithm,
        );
        
        Ok(HybridAnalysisResult {
            primary_algorithm,
            is_isomorphic,
            confidence,
            best_mapping,
            algorithm_results,
            ensemble_consensus,
        })
    }

    /// Analyze problem characteristics to guide algorithm selection
    fn analyze_problem_characteristics(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> ProblemCharacteristics {
        let query_size = query.nodes.len();
        let target_size = target.nodes.len();
        let query_edges = query.edges.len();
        let target_edges = target.edges.len();
        
        let query_density = if query_size > 1 {
            2.0 * query_edges as f64 / (query_size * (query_size - 1)) as f64
        } else {
            0.0
        };
        
        let target_density = if target_size > 1 {
            2.0 * target_edges as f64 / (target_size * (target_size - 1)) as f64
        } else {
            0.0
        };
        
        let size_ratio = if target_size > 0 {
            query_size as f64 / target_size as f64
        } else {
            1.0
        };
        
        // Estimate computational complexity
        let complexity_estimate = (query_size * target_size) as f64 * 
            (query_density + target_density) * 0.5;
        
        // Estimate uncertainty level based on node properties
        let uncertainty_level = self.estimate_uncertainty_level(query, target);
        
        ProblemCharacteristics {
            query_size,
            target_size,
            query_density,
            target_density,
            size_ratio,
            complexity_estimate,
            uncertainty_level,
        }
    }

    /// Estimate uncertainty level in the problem
    fn estimate_uncertainty_level(&self, query: &MolecularGraph, target: &MolecularGraph) -> f64 {
        let mut total_uncertainty = 0.0;
        let mut node_count = 0;
        
        for node in query.nodes.values() {
            total_uncertainty += node.morgan_number.variance + node.connectivity.variance;
            node_count += 1;
        }
        
        for node in target.nodes.values() {
            total_uncertainty += node.morgan_number.variance + node.connectivity.variance;
            node_count += 1;
        }
        
        if node_count > 0 {
            total_uncertainty / node_count as f64
        } else {
            0.0
        }
    }

    /// Select primary algorithm based on problem characteristics
    fn select_primary_algorithm(&self, characteristics: &ProblemCharacteristics) -> AlgorithmType {
        // Decision tree for algorithm selection
        
        // For small graphs with high uncertainty, use Morgan
        if characteristics.query_size <= 10 && characteristics.uncertainty_level > 0.5 {
            return AlgorithmType::Morgan;
        }
        
        // For subgraph isomorphism (size ratio < 1), prefer VF2
        if characteristics.size_ratio < 0.9 {
            return AlgorithmType::VF2;
        }
        
        // For exact isomorphism with moderate size, use Ullmann
        if characteristics.query_size == characteristics.target_size && 
           characteristics.query_size <= 50 &&
           characteristics.complexity_estimate < 1000.0 {
            return AlgorithmType::Ullmann;
        }
        
        // For large graphs or high complexity, use VF2
        if characteristics.complexity_estimate > 1000.0 || characteristics.query_size > 50 {
            return AlgorithmType::VF2;
        }
        
        // Default to Morgan for uncertain cases
        AlgorithmType::Morgan
    }

    /// Run Morgan algorithm analysis
    fn run_morgan_analysis(
        &self,
        query: &mut MolecularGraph,
        target: &mut MolecularGraph,
    ) -> BorgiaResult<AlgorithmResult> {
        let start_time = std::time::Instant::now();
        
        let is_isomorphic = self.morgan.are_isomorphic(query, target)?;
        let confidence = if is_isomorphic {
            ProbabilisticValue::new(0.85, 0.1) // Morgan gives good confidence for isomorphism
        } else {
            ProbabilisticValue::new(0.9, 0.05) // High confidence for non-isomorphism
        };
        
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(AlgorithmResult {
            is_isomorphic,
            confidence,
            mapping: None, // Morgan doesn't provide explicit mapping
            execution_time,
        })
    }

    /// Run VF2 algorithm analysis
    fn run_vf2_analysis(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<AlgorithmResult> {
        let start_time = std::time::Instant::now();
        
        let best_match = self.vf2.find_best_match(query, target)?;
        let (is_isomorphic, confidence, mapping) = if let Some((mapping, conf)) = best_match {
            (true, conf, Some(mapping))
        } else {
            (false, ProbabilisticValue::new(0.95, 0.02), None)
        };
        
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(AlgorithmResult {
            is_isomorphic,
            confidence,
            mapping,
            execution_time,
        })
    }

    /// Run Ullmann algorithm analysis
    fn run_ullmann_analysis(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<AlgorithmResult> {
        let start_time = std::time::Instant::now();
        
        let best_match = self.ullmann.find_best_isomorphism(query, target)?;
        let (is_isomorphic, confidence, mapping) = if let Some((mapping, conf)) = best_match {
            (true, conf, Some(mapping))
        } else {
            (false, ProbabilisticValue::new(0.92, 0.03), None)
        };
        
        let execution_time = start_time.elapsed().as_secs_f64();
        
        Ok(AlgorithmResult {
            is_isomorphic,
            confidence,
            mapping,
            execution_time,
        })
    }

    /// Run ensemble analysis combining multiple algorithms
    fn run_ensemble_analysis(
        &self,
        query: &mut MolecularGraph,
        target: &mut MolecularGraph,
        algorithm_results: &mut HashMap<AlgorithmType, AlgorithmResult>,
    ) -> BorgiaResult<Option<ProbabilisticValue>> {
        // Run all algorithms if not already run
        if !algorithm_results.contains_key(&AlgorithmType::VF2) {
            let vf2_result = self.run_vf2_analysis(query, target)?;
            algorithm_results.insert(AlgorithmType::VF2, vf2_result);
        }
        
        if !algorithm_results.contains_key(&AlgorithmType::Ullmann) {
            let ullmann_result = self.run_ullmann_analysis(query, target)?;
            algorithm_results.insert(AlgorithmType::Ullmann, ullmann_result);
        }
        
        // Compute ensemble consensus
        let consensus = self.compute_ensemble_consensus(algorithm_results);
        Ok(Some(consensus))
    }

    /// Compute ensemble consensus from multiple algorithm results
    fn compute_ensemble_consensus(
        &self,
        algorithm_results: &HashMap<AlgorithmType, AlgorithmResult>,
    ) -> ProbabilisticValue {
        let mut positive_votes = 0.0;
        let mut total_weight = 0.0;
        let mut total_uncertainty = 0.0;
        
        for (algorithm_type, result) in algorithm_results {
            let weight = self.get_algorithm_weight(*algorithm_type);
            total_weight += weight;
            
            if result.is_isomorphic {
                positive_votes += weight * result.confidence.mean;
            }
            
            total_uncertainty += result.confidence.variance * weight * weight;
        }
        
        let consensus_mean = if total_weight > 0.0 {
            positive_votes / total_weight
        } else {
            0.0
        };
        
        let consensus_uncertainty = if total_weight > 0.0 {
            (total_uncertainty / (total_weight * total_weight)).sqrt()
        } else {
            1.0
        };
        
        ProbabilisticValue::new(consensus_mean, consensus_uncertainty)
    }

    /// Get weight for each algorithm in ensemble
    fn get_algorithm_weight(&self, algorithm_type: AlgorithmType) -> f64 {
        match algorithm_type {
            AlgorithmType::Morgan => 0.3,   // Good for fingerprinting
            AlgorithmType::VF2 => 0.4,      // Good for subgraph matching
            AlgorithmType::Ullmann => 0.3,  // Good for exact matching
            AlgorithmType::Hybrid => 1.0,   // Not used in ensemble
        }
    }

    /// Determine final result from all analyses
    fn determine_final_result(
        &self,
        algorithm_results: &HashMap<AlgorithmType, AlgorithmResult>,
        ensemble_consensus: &Option<ProbabilisticValue>,
        primary_algorithm: AlgorithmType,
    ) -> (bool, ProbabilisticValue, Option<HashMap<usize, usize>>) {
        // Use ensemble consensus if available and confident
        if let Some(consensus) = ensemble_consensus {
            if consensus.mean > self.confidence_threshold {
                // Find best mapping from available results
                let best_mapping = self.find_best_mapping(algorithm_results);
                return (true, consensus.clone(), best_mapping);
            } else if consensus.mean < (1.0 - self.confidence_threshold) {
                return (false, 
                       ProbabilisticValue::new(1.0 - consensus.mean, consensus.variance), 
                       None);
            }
        }
        
        // Fall back to primary algorithm result
        if let Some(primary_result) = algorithm_results.get(&primary_algorithm) {
            return (
                primary_result.is_isomorphic,
                primary_result.confidence.clone(),
                primary_result.mapping.clone(),
            );
        }
        
        // Final fallback - use any available result
        for result in algorithm_results.values() {
            if result.confidence.mean > self.confidence_threshold {
                return (
                    result.is_isomorphic,
                    result.confidence.clone(),
                    result.mapping.clone(),
                );
            }
        }
        
        // No confident result found
        (false, ProbabilisticValue::new(0.5, 0.5), None)
    }

    /// Find best mapping from available algorithm results
    fn find_best_mapping(
        &self,
        algorithm_results: &HashMap<AlgorithmType, AlgorithmResult>,
    ) -> Option<HashMap<usize, usize>> {
        let mut best_mapping = None;
        let mut best_confidence = 0.0;
        
        for result in algorithm_results.values() {
            if let Some(ref mapping) = result.mapping {
                if result.confidence.mean > best_confidence {
                    best_confidence = result.confidence.mean;
                    best_mapping = Some(mapping.clone());
                }
            }
        }
        
        best_mapping
    }

    /// Generate comprehensive molecular fingerprint using hybrid approach
    pub fn generate_hybrid_fingerprint(
        &self,
        graph: &mut MolecularGraph,
        bits: usize,
    ) -> BorgiaResult<Vec<f64>> {
        // Use Morgan algorithm for fingerprint generation
        self.morgan.generate_fingerprint(graph, bits)
    }

    /// Compute similarity between two graphs using hybrid approach
    pub fn compute_similarity(
        &self,
        graph1: &mut MolecularGraph,
        graph2: &mut MolecularGraph,
    ) -> BorgiaResult<ProbabilisticValue> {
        // Generate fingerprints
        let fp1 = self.generate_hybrid_fingerprint(graph1, 1024)?;
        let fp2 = self.generate_hybrid_fingerprint(graph2, 1024)?;
        
        // Compute Tanimoto similarity
        let mut intersection = 0.0;
        let mut union = 0.0;
        
        for (a, b) in fp1.iter().zip(fp2.iter()) {
            let min_val = a.min(b);
            let max_val = a.max(b);
            intersection += min_val;
            union += max_val;
        }
        
        let similarity = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };
        
        // Estimate uncertainty based on fingerprint variance
        let fp1_variance: f64 = fp1.iter().map(|x| x * x).sum::<f64>() / fp1.len() as f64;
        let fp2_variance: f64 = fp2.iter().map(|x| x * x).sum::<f64>() / fp2.len() as f64;
        let uncertainty = (fp1_variance + fp2_variance).sqrt() * 0.1;
        
        Ok(ProbabilisticValue::new(similarity, uncertainty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_system_basic() {
        let mut query = MolecularGraph::new();
        let mut target = MolecularGraph::new();

        // Create simple graphs
        query.add_node(0, 6); // C
        query.add_node(1, 6); // C
        query.add_edge(0, 1, 1.0);

        target.add_node(0, 6); // C
        target.add_node(1, 6); // C
        target.add_edge(0, 1, 1.0);

        let hybrid = HybridAlgorithmSystem::default();
        let result = hybrid.analyze_isomorphism(&mut query, &mut target);

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert!(analysis.is_isomorphic);
        assert!(analysis.confidence.mean > 0.5);
    }

    #[test]
    fn test_algorithm_selection() {
        let hybrid = HybridAlgorithmSystem::default();
        
        // Test small graph characteristics
        let small_characteristics = ProblemCharacteristics {
            query_size: 5,
            target_size: 5,
            query_density: 0.3,
            target_density: 0.3,
            size_ratio: 1.0,
            complexity_estimate: 50.0,
            uncertainty_level: 0.6,
        };
        
        let algorithm = hybrid.select_primary_algorithm(&small_characteristics);
        assert_eq!(algorithm, AlgorithmType::Morgan);
        
        // Test subgraph characteristics
        let subgraph_characteristics = ProblemCharacteristics {
            query_size: 10,
            target_size: 20,
            query_density: 0.4,
            target_density: 0.3,
            size_ratio: 0.5,
            complexity_estimate: 200.0,
            uncertainty_level: 0.2,
        };
        
        let algorithm = hybrid.select_primary_algorithm(&subgraph_characteristics);
        assert_eq!(algorithm, AlgorithmType::VF2);
    }

    #[test]
    fn test_fingerprint_generation() {
        let mut graph = MolecularGraph::new();
        graph.add_node(0, 6); // C
        graph.add_node(1, 6); // C
        graph.add_edge(0, 1, 1.0);

        let hybrid = HybridAlgorithmSystem::default();
        let fingerprint = hybrid.generate_hybrid_fingerprint(&mut graph, 512);

        assert!(fingerprint.is_ok());
        let fp = fingerprint.unwrap();
        assert_eq!(fp.len(), 512);
    }

    #[test]
    fn test_similarity_computation() {
        let mut graph1 = MolecularGraph::new();
        let mut graph2 = MolecularGraph::new();

        // Create similar graphs
        graph1.add_node(0, 6); // C
        graph1.add_node(1, 6); // C
        graph1.add_edge(0, 1, 1.0);

        graph2.add_node(0, 6); // C
        graph2.add_node(1, 6); // C
        graph2.add_edge(0, 1, 1.0);

        let hybrid = HybridAlgorithmSystem::default();
        let similarity = hybrid.compute_similarity(&mut graph1, &mut graph2);

        assert!(similarity.is_ok());
        let sim = similarity.unwrap();
        assert!(sim.mean > 0.8); // Should be highly similar
    }
}
