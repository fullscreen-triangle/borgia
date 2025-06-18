use std::collections::HashMap;
use crate::error::BorgiaResult;
use crate::probabilistic::ProbabilisticValue;
use crate::algorithms::morgan::{MolecularGraph, MorganNode};

/// Ullmann Algorithm implementation for graph isomorphism
/// Enhanced with probabilistic compatibility matrices for uncertain molecular structures
#[derive(Debug, Clone)]
pub struct UllmannAlgorithm {
    /// Threshold for considering nodes compatible
    compatibility_threshold: f64,
    /// Maximum search depth
    max_depth: usize,
    /// Whether to use probabilistic refinement
    probabilistic_refinement: bool,
}

impl Default for UllmannAlgorithm {
    fn default() -> Self {
        Self {
            compatibility_threshold: 0.7,
            max_depth: 1000,
            probabilistic_refinement: true,
        }
    }
}

/// Compatibility matrix for Ullmann algorithm
#[derive(Debug, Clone)]
pub struct CompatibilityMatrix {
    /// Matrix dimensions (query_size x target_size)
    pub query_size: usize,
    pub target_size: usize,
    /// Compatibility values [query_node][target_node]
    pub matrix: Vec<Vec<ProbabilisticValue>>,
}

impl CompatibilityMatrix {
    pub fn new(query_size: usize, target_size: usize) -> Self {
        let matrix = vec![vec![ProbabilisticValue::new(0.0, 0.0); target_size]; query_size];
        Self {
            query_size,
            target_size,
            matrix,
        }
    }

    /// Set compatibility value
    pub fn set(&mut self, query_node: usize, target_node: usize, value: ProbabilisticValue) {
        if query_node < self.query_size && target_node < self.target_size {
            self.matrix[query_node][target_node] = value;
        }
    }

    /// Get compatibility value
    pub fn get(&self, query_node: usize, target_node: usize) -> ProbabilisticValue {
        if query_node < self.query_size && target_node < self.target_size {
            self.matrix[query_node][target_node].clone()
        } else {
            ProbabilisticValue::new(0.0, 0.0)
        }
    }

    /// Check if assignment is compatible
    pub fn is_compatible(&self, query_node: usize, target_node: usize, threshold: f64) -> bool {
        self.get(query_node, target_node).mean >= threshold
    }

    /// Refine the matrix by removing incompatible assignments
    pub fn refine(&mut self, query: &MolecularGraph, target: &MolecularGraph) -> bool {
        let mut changed = false;

        for q in 0..self.query_size {
            for t in 0..self.target_size {
                if self.matrix[q][t].mean > 0.0 {
                    let refinement_score = self.compute_refinement_score(q, t, query, target);
                    if refinement_score.mean < 0.5 {
                        self.matrix[q][t] = ProbabilisticValue::new(0.0, 0.0);
                        changed = true;
                    } else {
                        // Update with refined score
                        let old_score = self.matrix[q][t].clone();
                        let new_score = ProbabilisticValue::new(
                            old_score.mean * refinement_score.mean,
                            (old_score.variance + refinement_score.variance).sqrt(),
                        );
                        if (new_score.mean - old_score.mean).abs() > 1e-6 {
                            self.matrix[q][t] = new_score;
                            changed = true;
                        }
                    }
                }
            }
        }

        changed
    }

    /// Compute refinement score for a specific assignment
    fn compute_refinement_score(
        &self,
        q: usize,
        t: usize,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> ProbabilisticValue {
        let mut score = 1.0;
        let mut uncertainty = 0.0;

        if let (Some(query_node), Some(target_node)) = (query.nodes.get(&q), target.nodes.get(&t)) {
            // Check neighbor compatibility
            for &q_neighbor in &query_node.neighbors {
                let mut found_compatible = false;
                let mut max_neighbor_score = 0.0;

                for &t_neighbor in &target_node.neighbors {
                    let neighbor_compatibility = self.get(q_neighbor, t_neighbor);
                    if neighbor_compatibility.mean > 0.5 {
                        found_compatible = true;
                        max_neighbor_score = max_neighbor_score.max(neighbor_compatibility.mean);
                        uncertainty += neighbor_compatibility.variance;
                    }
                }

                if !found_compatible {
                    return ProbabilisticValue::new(0.0, 0.0);
                }

                score *= max_neighbor_score;
            }
        }

        ProbabilisticValue::new(score, uncertainty.sqrt())
    }
}

/// Permutation matrix for Ullmann algorithm
#[derive(Debug, Clone)]
pub struct PermutationMatrix {
    pub size: usize,
    pub matrix: Vec<Vec<bool>>,
}

impl PermutationMatrix {
    pub fn new(size: usize) -> Self {
        let matrix = vec![vec![false; size]; size];
        Self { size, matrix }
    }

    /// Set assignment
    pub fn set(&mut self, query_node: usize, target_node: usize, value: bool) {
        if query_node < self.size && target_node < self.size {
            self.matrix[query_node][target_node] = value;
        }
    }

    /// Get assignment
    pub fn get(&self, query_node: usize, target_node: usize) -> bool {
        if query_node < self.size && target_node < self.size {
            self.matrix[query_node][target_node]
        } else {
            false
        }
    }

    /// Check if matrix represents a valid permutation
    pub fn is_valid_permutation(&self) -> bool {
        // Each row should have exactly one true value
        for row in &self.matrix {
            if row.iter().filter(|&&x| x).count() != 1 {
                return false;
            }
        }

        // Each column should have exactly one true value
        for col in 0..self.size {
            let col_sum = (0..self.size).map(|row| self.matrix[row][col] as usize).sum::<usize>();
            if col_sum != 1 {
                return false;
            }
        }

        true
    }

    /// Convert to mapping
    pub fn to_mapping(&self) -> HashMap<usize, usize> {
        let mut mapping = HashMap::new();
        for (query_node, row) in self.matrix.iter().enumerate() {
            for (target_node, &assigned) in row.iter().enumerate() {
                if assigned {
                    mapping.insert(query_node, target_node);
                    break;
                }
            }
        }
        mapping
    }
}

impl UllmannAlgorithm {
    pub fn new(compatibility_threshold: f64, max_depth: usize) -> Self {
        Self {
            compatibility_threshold,
            max_depth,
            probabilistic_refinement: true,
        }
    }

    /// Find isomorphisms between two graphs
    pub fn find_isomorphisms(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<Vec<HashMap<usize, usize>>> {
        if query.nodes.len() > target.nodes.len() {
            return Ok(Vec::new());
        }

        // Build initial compatibility matrix
        let mut compatibility = self.build_compatibility_matrix(query, target)?;

        // Refine the compatibility matrix
        self.refine_compatibility_matrix(&mut compatibility, query, target)?;

        // Search for isomorphisms
        let mut isomorphisms = Vec::new();
        let mut permutation = PermutationMatrix::new(query.nodes.len());
        
        self.search_isomorphisms(
            query,
            target,
            &compatibility,
            &mut permutation,
            0,
            &mut isomorphisms,
        )?;

        Ok(isomorphisms)
    }

    /// Check if query is isomorphic to target
    pub fn is_isomorphic(&self, query: &MolecularGraph, target: &MolecularGraph) -> BorgiaResult<bool> {
        if query.nodes.len() != target.nodes.len() || query.edges.len() != target.edges.len() {
            return Ok(false);
        }

        let isomorphisms = self.find_isomorphisms(query, target)?;
        Ok(!isomorphisms.is_empty())
    }

    /// Build initial compatibility matrix
    fn build_compatibility_matrix(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<CompatibilityMatrix> {
        let mut matrix = CompatibilityMatrix::new(query.nodes.len(), target.nodes.len());

        // Convert node IDs to indices
        let query_nodes: Vec<_> = query.nodes.keys().cloned().collect();
        let target_nodes: Vec<_> = target.nodes.keys().cloned().collect();

        for (q_idx, &q_id) in query_nodes.iter().enumerate() {
            for (t_idx, &t_id) in target_nodes.iter().enumerate() {
                if let (Some(query_node), Some(target_node)) = (query.nodes.get(&q_id), target.nodes.get(&t_id)) {
                    let compatibility = self.compute_node_compatibility(query_node, target_node)?;
                    matrix.set(q_idx, t_idx, compatibility);
                }
            }
        }

        Ok(matrix)
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
        if query_node.neighbors.len() != target_node.neighbors.len() {
            return Ok(ProbabilisticValue::new(0.0, 0.0));
        }

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

        // Bond order compatibility
        let mut bond_score = 1.0;
        for (&q_bond, &t_bond) in query_node.bond_orders.iter().zip(&target_node.bond_orders) {
            let bond_diff = (q_bond - t_bond).abs();
            bond_score *= (-bond_diff).exp();
        }

        // Combine scores
        let combined_score = connectivity_score * morgan_score * bond_score;
        let combined_uncertainty = (connectivity_uncertainty + morgan_uncertainty) * 0.1;

        Ok(ProbabilisticValue::new(combined_score, combined_uncertainty))
    }

    /// Refine compatibility matrix iteratively
    fn refine_compatibility_matrix(
        &self,
        matrix: &mut CompatibilityMatrix,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<()> {
        if !self.probabilistic_refinement {
            return Ok(());
        }

        let mut iteration = 0;
        let max_iterations = 10;

        while iteration < max_iterations {
            let changed = matrix.refine(query, target);
            if !changed {
                break;
            }
            iteration += 1;
        }

        Ok(())
    }

    /// Search for isomorphisms using backtracking
    fn search_isomorphisms(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        compatibility: &CompatibilityMatrix,
        permutation: &mut PermutationMatrix,
        depth: usize,
        isomorphisms: &mut Vec<HashMap<usize, usize>>,
    ) -> BorgiaResult<()> {
        if depth > self.max_depth {
            return Ok(());
        }

        if depth == query.nodes.len() {
            // Found a complete assignment
            if self.is_valid_isomorphism(query, target, permutation)? {
                isomorphisms.push(permutation.to_mapping());
            }
            return Ok(());
        }

        // Try all compatible assignments for the current query node
        for target_node in 0..target.nodes.len() {
            if compatibility.is_compatible(depth, target_node, self.compatibility_threshold) {
                // Check if this target node is already assigned
                let mut already_assigned = false;
                for row in 0..depth {
                    if permutation.get(row, target_node) {
                        already_assigned = true;
                        break;
                    }
                }

                if !already_assigned {
                    // Make assignment
                    permutation.set(depth, target_node, true);

                    // Recurse
                    self.search_isomorphisms(
                        query,
                        target,
                        compatibility,
                        permutation,
                        depth + 1,
                        isomorphisms,
                    )?;

                    // Backtrack
                    permutation.set(depth, target_node, false);
                }
            }
        }

        Ok(())
    }

    /// Validate that a permutation represents a valid isomorphism
    fn is_valid_isomorphism(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        permutation: &PermutationMatrix,
    ) -> BorgiaResult<bool> {
        let mapping = permutation.to_mapping();

        // Convert node IDs to indices for mapping
        let query_nodes: Vec<_> = query.nodes.keys().cloned().collect();
        let target_nodes: Vec<_> = target.nodes.keys().cloned().collect();

        // Check edge preservation
        for (from_idx, to_idx, bond_order) in &query.edges {
            if let (Some(&from_mapped_idx), Some(&to_mapped_idx)) = (mapping.get(from_idx), mapping.get(to_idx)) {
                // Find corresponding edge in target
                let from_mapped_id = target_nodes.get(from_mapped_idx).copied().unwrap_or(0);
                let to_mapped_id = target_nodes.get(to_mapped_idx).copied().unwrap_or(0);

                let mut found_edge = false;
                for (t_from, t_to, t_bond_order) in &target.edges {
                    if (*t_from == from_mapped_id && *t_to == to_mapped_id) ||
                       (*t_from == to_mapped_id && *t_to == from_mapped_id) {
                        // Check bond order compatibility
                        if (bond_order - t_bond_order).abs() > 0.1 {
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

        Ok(true)
    }

    /// Find the best isomorphism with highest confidence
    pub fn find_best_isomorphism(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
    ) -> BorgiaResult<Option<(HashMap<usize, usize>, ProbabilisticValue)>> {
        let isomorphisms = self.find_isomorphisms(query, target)?;
        
        if isomorphisms.is_empty() {
            return Ok(None);
        }

        let mut best_mapping = None;
        let mut best_confidence = ProbabilisticValue::new(0.0, 1.0);

        for mapping in isomorphisms {
            let confidence = self.compute_mapping_confidence(query, target, &mapping)?;
            if confidence.mean > best_confidence.mean {
                best_mapping = Some(mapping);
                best_confidence = confidence;
            }
        }

        Ok(best_mapping.map(|m| (m, best_confidence)))
    }

    /// Compute confidence score for a mapping
    fn compute_mapping_confidence(
        &self,
        query: &MolecularGraph,
        target: &MolecularGraph,
        mapping: &HashMap<usize, usize>,
    ) -> BorgiaResult<ProbabilisticValue> {
        let mut total_score = 1.0;
        let mut total_uncertainty = 0.0;

        for (&q_id, &t_id) in mapping {
            if let (Some(query_node), Some(target_node)) = (query.nodes.get(&q_id), target.nodes.get(&t_id)) {
                let node_compatibility = self.compute_node_compatibility(query_node, target_node)?;
                total_score *= node_compatibility.mean;
                total_uncertainty += node_compatibility.variance;
            }
        }

        Ok(ProbabilisticValue::new(total_score, total_uncertainty.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ullmann_basic() {
        let mut query = MolecularGraph::new();
        let mut target = MolecularGraph::new();

        // Create identical simple graphs
        query.add_node(0, 6); // C
        query.add_node(1, 6); // C
        query.add_edge(0, 1, 1.0);

        target.add_node(0, 6); // C
        target.add_node(1, 6); // C
        target.add_edge(0, 1, 1.0);

        let ullmann = UllmannAlgorithm::default();
        let result = ullmann.is_isomorphic(&query, &target);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_ullmann_no_match() {
        let mut query = MolecularGraph::new();
        let mut target = MolecularGraph::new();

        // Create incompatible graphs
        query.add_node(0, 7); // N
        target.add_node(0, 6); // C

        let ullmann = UllmannAlgorithm::default();
        let result = ullmann.is_isomorphic(&query, &target);

        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_compatibility_matrix() {
        let mut matrix = CompatibilityMatrix::new(2, 2);
        let value = ProbabilisticValue::new(0.8, 0.1);
        
        matrix.set(0, 1, value.clone());
        let retrieved = matrix.get(0, 1);
        
        assert!((retrieved.mean - value.mean).abs() < 1e-6);
        assert!(matrix.is_compatible(0, 1, 0.7));
        assert!(!matrix.is_compatible(0, 1, 0.9));
    }
}
