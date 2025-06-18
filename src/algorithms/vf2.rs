use std::collections::{HashMap, HashSet};
use crate::error::BorgiaResult;
use crate::probabilistic::ProbabilisticValue;
use crate::algorithms::morgan::{MolecularGraph, MorganNode};

/// VF2 Algorithm implementation for subgraph isomorphism
/// Enhanced with probabilistic matching for uncertain molecular structures
#[derive(Debug, Clone)]
pub struct VF2Algorithm {
    /// Tolerance for probabilistic matching
    probability_threshold: f64,
    /// Maximum search depth
    max_depth: usize,
    /// Whether to use probabilistic node matching
    probabilistic_matching: bool,
}

impl Default for VF2Algorithm {
    fn default() -> Self {
        Self {
            probability_threshold: 0.8,
            max_depth: 1000,
            probabilistic_matching: true,
        }
    }
}

/// Represents the state of the VF2 matching process
#[derive(Debug, Clone)]
pub struct VF2State {
    /// Mapping from query graph nodes to target graph nodes
    pub core_1: HashMap<usize, usize>,
    /// Mapping from target graph nodes to query graph nodes
    pub core_2: HashMap<usize, usize>,
    /// In-neighbors of mapped nodes in query graph
    pub in_1: HashSet<usize>,
    /// Out-neighbors of mapped nodes in query graph
    pub out_1: HashSet<usize>,
    /// In-neighbors of mapped nodes in target graph
    pub in_2: HashSet<usize>,
    /// Out-neighbors of mapped nodes in target graph
    pub out_2: HashSet<usize>,
    /// Current depth of the search
    pub depth: usize,
    /// Probabilistic confidence of the current mapping
    pub confidence: ProbabilisticValue,
}

impl VF2State {
    pub fn new() -> Self {
        Self {
            core_1: HashMap::new(),
            core_2: HashMap::new(),
            in_1: HashSet::new(),
            out_1: HashSet::new(),
            in_2: HashSet::new(),
            out_2: HashSet::new(),
            depth: 0,
            confidence: ProbabilisticValue::new(1.0, 0.0),
        }
    }

    /// Check if the current state represents a complete mapping
    pub fn is_goal(&self, query_size: usize) -> bool {
        self.core_1.len() == query_size
    }

    /// Add a new pair to the mapping
    pub fn add_pair(&mut self, n: usize, m: usize, query: &MolecularGraph, target: &MolecularGraph) {
        self.core_1.insert(n, m);
        self.core_2.insert(m, n);
        self.depth += 1;

        // Update in/out sets
        self.update_sets_after_addition(n, m, query, target);
    }

    /// Remove a pair from the mapping (backtrack)
    pub fn remove_pair(&mut self, n: usize, m: usize, query: &MolecularGraph, target: &MolecularGraph) {
        self.core_1.remove(&n);
        self.core_2.remove(&m);
        self.depth -= 1;

        // Update in/out sets
        self.update_sets_after_removal(n, m, query, target);
    }

    /// Update the in/out sets after adding a pair
    fn update_sets_after_addition(&mut self, n: usize, m: usize, query: &MolecularGraph, target: &MolecularGraph) {
        // Update query graph sets
        if let Some(query_node) = query.nodes.get(&n) {
            for &neighbor in &query_node.neighbors {
                if !self.core_1.contains_key(&neighbor) {
                    self.out_1.insert(neighbor);
                }
            }
        }

        // Update target graph sets
        if let Some(target_node) = target.nodes.get(&m) {
            for &neighbor in &target_node.neighbors {
                if !self.core_2.contains_key(&neighbor) {
                    self.out_2.insert(neighbor);
                }
            }
        }

        // Remove newly mapped nodes from in/out sets
        self.in_1.remove(&n);
        self.out_1.remove(&n);
        self.in_2.remove(&m);
        self.out_2.remove(&m);
    }

    /// Update the in/out sets after removing a pair
    fn update_sets_after_removal(&mut self, n: usize, m: usize, query: &MolecularGraph, target: &MolecularGraph) {
        // Rebuild in/out sets (simplified approach)
        self.rebuild_sets(query, target);
    }

    /// Rebuild in/out sets from scratch
    fn rebuild_sets(&mut self, query: &MolecularGraph, target: &MolecularGraph) {
        self.in_1.clear();
        self.out_1.clear();
        self.in_2.clear();
        self.out_2.clear();

        // Rebuild query sets
        for &n in self.core_1.keys() {
            if let Some(node) = query.nodes.get(&n) {
                for &neighbor in &node.neighbors {
                    if !self.core_1.contains_key(&neighbor) {
                        self.out_1.insert(neighbor);
                    }
                }
            }
        }

        // Rebuild target sets
        for &m in self.core_2.keys() {
            if let Some(node) = target.nodes.get(&m) {
                for &neighbor in &node.neighbors {
                    if !self.core_2.contains_key(&neighbor) {
                        self.out_2.insert(neighbor);
                    }
                }
            }
        }
    }
}

/// Candidate pair for VF2 matching
#[derive(Debug, Clone)]
pub struct CandidatePair {
    pub query_node: usize,
    pub target_node: usize,
    pub compatibility_score: ProbabilisticValue,
}

impl VF2Algorithm {
    pub fn new(probability_threshold: f64, max_depth: usize) -> Self {
        Self {
            probability_threshold,
            max_depth,
            probabilistic_matching: true,
        }
    }

    /// Find all subgraph isomorphisms between query and target graphs
    pub fn find_isomorphisms(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<Vec<HashMap<usize, usize>>> {
        let mut isomorphisms = Vec::new();
        let mut state = VF2State::new();

        self.match_recursive(query, target, &mut state, &mut isomorphisms)?;

        Ok(isomorphisms)
    }

    /// Check if query is a subgraph of target
    pub fn is_subgraph(&self, query: &MolecularGraph, target: &MolecularGraph) -> BorgiaResult<bool> {
        let isomorphisms = self.find_isomorphisms(query, target)?;
        Ok(!isomorphisms.is_empty())
    }

    /// Recursive matching function
    fn match_recursive(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &mut VF2State,
        isomorphisms: &mut Vec<HashMap<usize, usize>>,
    ) -> BorgiaResult<()> {
        if state.depth > self.max_depth {
            return Ok(());
        }

        if state.is_goal(query.nodes.len()) {
            // Found a complete mapping
            isomorphisms.push(state.core_1.clone());
            return Ok(());
        }

        // Generate candidate pairs
        let candidates = self.generate_candidates(query, target, state)?;

        for candidate in candidates {
            if self.is_feasible(query, target, state, &candidate)? {
                // Add the pair and recurse
                state.add_pair(candidate.query_node, candidate.target_node, query, target);
                
                // Update confidence
                let old_confidence = state.confidence.clone();
                state.confidence = self.combine_confidence(&old_confidence, &candidate.compatibility_score);

                self.match_recursive(query, target, state, isomorphisms)?;

                // Backtrack
                state.remove_pair(candidate.query_node, candidate.target_node, query, target);
                state.confidence = old_confidence;
            }
        }

        Ok(())
    }

    /// Generate candidate pairs for matching
    fn generate_candidates(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &VF2State,
    ) -> BorgiaResult<Vec<CandidatePair>> {
        let mut candidates = Vec::new();

        // Priority 1: Pairs from out sets
        if !state.out_1.is_empty() && !state.out_2.is_empty() {
            for &n in &state.out_1 {
                for &m in &state.out_2 {
                    if let (Some(query_node), Some(target_node)) = (query.nodes.get(&n), target.nodes.get(&m)) {
                        let compatibility = self.compute_node_compatibility(query_node, target_node)?;
                        candidates.push(CandidatePair {
                            query_node: n,
                            target_node: m,
                            compatibility_score: compatibility,
                        });
                    }
                }
            }
        } else {
            // Priority 2: Unmapped nodes
            for (n, query_node) in &query.nodes {
                if !state.core_1.contains_key(n) {
                    for (m, target_node) in &target.nodes {
                        if !state.core_2.contains_key(m) {
                            let compatibility = self.compute_node_compatibility(query_node, target_node)?;
                            candidates.push(CandidatePair {
                                query_node: *n,
                                target_node: *m,
                                compatibility_score: compatibility,
                            });
                        }
                    }
                    break; // Only consider first unmapped query node
                }
            }
        }

        // Sort candidates by compatibility score (descending)
        candidates.sort_by(|a, b| {
            b.compatibility_score.mean.partial_cmp(&a.compatibility_score.mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    /// Check if a candidate pair is feasible
    fn is_feasible(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &VF2State,
        candidate: &CandidatePair,
    ) -> BorgiaResult<bool> {
        let n = candidate.query_node;
        let m = candidate.target_node;

        // Basic compatibility check
        if candidate.compatibility_score.mean < self.probability_threshold {
            return Ok(false);
        }

        // Check consistency with existing mapping
        if !self.check_consistency(query, target, state, n, m)? {
            return Ok(false);
        }

        // Check look-ahead conditions
        if !self.check_look_ahead(query, target, state, n, m)? {
            return Ok(false);
        }

        Ok(true)
    }

    /// Compute compatibility between two nodes
    fn compute_node_compatibility(
        &self,
        query_node: &MorganNode,
        target_node: &MorganNode,
    ) -> BorgiaResult<ProbabilisticValue> {
        // Atomic number must match exactly
        if query_node.atomic_number != target_node.atomic_number {
            return Ok(ProbabilisticValue::new(0.0, 0.0));
        }

        // Degree compatibility
        let degree_diff = (query_node.neighbors.len() as f64 - target_node.neighbors.len() as f64).abs();
        let degree_score = (-degree_diff * 0.5).exp();

        // Connectivity compatibility
        let connectivity_diff = (query_node.connectivity.mean - target_node.connectivity.mean).abs();
        let connectivity_uncertainty = (query_node.connectivity.variance + target_node.connectivity.variance).sqrt();
        let connectivity_score = if connectivity_uncertainty > 0.0 {
            (-connectivity_diff / connectivity_uncertainty).exp()
        } else {
            if connectivity_diff < 1e-6 { 1.0 } else { 0.0 }
        };

        // Morgan number compatibility
        let morgan_diff = (query_node.morgan_number.mean - target_node.morgan_number.mean).abs();
        let morgan_uncertainty = (query_node.morgan_number.variance + target_node.morgan_number.variance).sqrt();
        let morgan_score = if morgan_uncertainty > 0.0 {
            (-morgan_diff / morgan_uncertainty).exp()
        } else {
            if morgan_diff < 1e-6 { 1.0 } else { 0.0 }
        };

        // Combine scores
        let combined_score = degree_score * connectivity_score * morgan_score;
        let combined_uncertainty = (connectivity_uncertainty + morgan_uncertainty) * 0.1;

        Ok(ProbabilisticValue::new(combined_score, combined_uncertainty))
    }

    /// Check consistency with existing mapping
    fn check_consistency(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &VF2State,
        n: usize,
        m: usize,
    ) -> BorgiaResult<bool> {
        if let (Some(query_node), Some(target_node)) = (query.nodes.get(&n), target.nodes.get(&m)) {
            // Check edge consistency
            for (&query_neighbor, &query_bond_order) in query_node.neighbors.iter().zip(&query_node.bond_orders) {
                if let Some(&target_mapped) = state.core_1.get(&query_neighbor) {
                    // Find corresponding edge in target
                    let mut found_edge = false;
                    for (&target_neighbor, &target_bond_order) in target_node.neighbors.iter().zip(&target_node.bond_orders) {
                        if target_neighbor == target_mapped {
                            // Check bond order compatibility
                            let bond_diff = (query_bond_order - target_bond_order).abs();
                            if bond_diff > 0.5 { // Allow some tolerance
                                return Ok(false);
                            }
                            found_edge = true;
                            break;
                        }
                    }
                    if !found_edge {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    /// Check look-ahead conditions
    fn check_look_ahead(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &VF2State,
        n: usize,
        m: usize,
    ) -> BorgiaResult<bool> {
        // Simplified look-ahead: check if adding this pair would leave
        // enough candidates for remaining query nodes
        let remaining_query = query.nodes.len() - state.core_1.len() - 1;
        let remaining_target = target.nodes.len() - state.core_2.len() - 1;

        Ok(remaining_target >= remaining_query)
    }

    /// Combine confidence scores
    fn combine_confidence(
        &self,
        old_confidence: &ProbabilisticValue,
        new_score: &ProbabilisticValue,
    ) -> ProbabilisticValue {
        let combined_mean = old_confidence.mean * new_score.mean;
        let combined_variance = old_confidence.variance + new_score.variance;
        
        ProbabilisticValue::new(combined_mean, combined_variance.sqrt())
    }

    /// Find the best matching with highest confidence
    pub fn find_best_match(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<Option<(HashMap<usize, usize>, ProbabilisticValue)>> {
        let mut best_match = None;
        let mut best_confidence = ProbabilisticValue::new(0.0, 1.0);
        let mut state = VF2State::new();

        self.find_best_match_recursive(query, target, &mut state, &mut best_match, &mut best_confidence)?;

        Ok(best_match.map(|mapping| (mapping, best_confidence)))
    }

    /// Recursive function to find best match
    fn find_best_match_recursive(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        state: &mut VF2State,
        best_match: &mut Option<HashMap<usize, usize>>,
        best_confidence: &mut ProbabilisticValue,
    ) -> BorgiaResult<()> {
        if state.depth > self.max_depth {
            return Ok(());
        }

        if state.is_goal(query.nodes.len()) {
            if state.confidence.mean > best_confidence.mean {
                *best_match = Some(state.core_1.clone());
                *best_confidence = state.confidence.clone();
            }
            return Ok(());
        }

        let candidates = self.generate_candidates(query, target, state)?;

        for candidate in candidates {
            if self.is_feasible(query, target, state, &candidate)? {
                let old_confidence = state.confidence.clone();
                state.add_pair(candidate.query_node, candidate.target_node, query, target);
                state.confidence = self.combine_confidence(&old_confidence, &candidate.compatibility_score);

                self.find_best_match_recursive(query, target, state, best_match, best_confidence)?;

                state.remove_pair(candidate.query_node, candidate.target_node, query, target);
                state.confidence = old_confidence;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vf2_basic() {
        let mut query = MolecularGraph::new();
        let mut target = MolecularGraph::new();

        // Create simple graphs
        query.add_node(0, 6); // C
        query.add_node(1, 6); // C
        query.add_edge(0, 1, 1.0);

        target.add_node(0, 6); // C
        target.add_node(1, 6); // C
        target.add_node(2, 1); // H
        target.add_edge(0, 1, 1.0);
        target.add_edge(1, 2, 1.0);

        let vf2 = VF2Algorithm::default();
        let result = vf2.is_subgraph(&query, &target);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_vf2_no_match() {
        let mut query = MolecularGraph::new();
        let mut target = MolecularGraph::new();

        // Create incompatible graphs
        query.add_node(0, 7); // N
        target.add_node(0, 6); // C

        let vf2 = VF2Algorithm::default();
        let result = vf2.is_subgraph(&query, &target);

        assert!(result.is_ok());
        assert!(!result.unwrap());
    }
}
