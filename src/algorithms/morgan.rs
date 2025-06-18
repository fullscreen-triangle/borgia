use std::collections::{HashMap, HashSet};
use crate::error::BorgiaResult;
use crate::probabilistic::ProbabilisticValue;

/// Morgan Algorithm implementation with probabilistic extensions
/// This is a core algorithm for molecular fingerprinting and graph isomorphism
#[derive(Debug, Clone)]
pub struct MorganAlgorithm {
    /// Maximum number of iterations for convergence
    max_iterations: usize,
    /// Convergence threshold for probabilistic values
    convergence_threshold: f64,
    /// Whether to use probabilistic node values
    use_probabilistic: bool,
}

impl Default for MorganAlgorithm {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            use_probabilistic: true,
        }
    }
}

/// Represents a node in the molecular graph
#[derive(Debug, Clone)]
pub struct MorganNode {
    pub id: usize,
    pub atomic_number: u8,
    pub neighbors: Vec<usize>,
    pub bond_orders: Vec<f64>,
    pub connectivity: ProbabilisticValue,
    pub morgan_number: ProbabilisticValue,
}

impl MorganNode {
    pub fn new(id: usize, atomic_number: u8) -> Self {
        Self {
            id,
            atomic_number,
            neighbors: Vec::new(),
            bond_orders: Vec::new(),
            connectivity: ProbabilisticValue::new(0.0, 0.0),
            morgan_number: ProbabilisticValue::new(atomic_number as f64, 0.1),
        }
    }

    pub fn add_neighbor(&mut self, neighbor_id: usize, bond_order: f64) {
        self.neighbors.push(neighbor_id);
        self.bond_orders.push(bond_order);
        self.connectivity = ProbabilisticValue::new(
            self.neighbors.len() as f64,
            (self.neighbors.len() as f64).sqrt() * 0.1,
        );
    }
}

/// Molecular graph representation for Morgan algorithm
#[derive(Debug, Clone)]
pub struct MolecularGraph {
    pub nodes: HashMap<usize, MorganNode>,
    pub edges: Vec<(usize, usize, f64)>,
}

impl MolecularGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, id: usize, atomic_number: u8) {
        let node = MorganNode::new(id, atomic_number);
        self.nodes.insert(id, node);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, bond_order: f64) {
        self.edges.push((from, to, bond_order));
        
        // Update node neighbors
        if let Some(node) = self.nodes.get_mut(&from) {
            node.add_neighbor(to, bond_order);
        }
        if let Some(node) = self.nodes.get_mut(&to) {
            node.add_neighbor(from, bond_order);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl MorganAlgorithm {
    pub fn new(max_iterations: usize, convergence_threshold: f64) -> Self {
        Self {
            max_iterations,
            convergence_threshold,
            use_probabilistic: true,
        }
    }

    /// Compute Morgan numbers for all nodes in the graph
    pub fn compute_morgan_numbers(&self, graph: &mut MolecularGraph) -> BorgiaResult<Vec<ProbabilisticValue>> {
        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations && !converged {
            let mut new_morgan_numbers = HashMap::new();
            let mut max_change = 0.0;

            for (node_id, node) in &graph.nodes {
                let new_morgan = self.compute_node_morgan_number(node, &graph.nodes)?;
                let change = (new_morgan.mean - node.morgan_number.mean).abs();
                max_change = max_change.max(change);
                new_morgan_numbers.insert(*node_id, new_morgan);
            }

            // Update morgan numbers
            for (node_id, new_morgan) in new_morgan_numbers {
                if let Some(node) = graph.nodes.get_mut(&node_id) {
                    node.morgan_number = new_morgan;
                }
            }

            converged = max_change < self.convergence_threshold;
            iteration += 1;
        }

        if !converged {
            log::warn!("Morgan algorithm did not converge after {} iterations", self.max_iterations);
        }

        Ok(graph.nodes.values().map(|node| node.morgan_number.clone()).collect())
    }

    /// Compute Morgan number for a single node
    fn compute_node_morgan_number(
        &self,
        node: &MorganNode,
        all_nodes: &HashMap<usize, MorganNode>,
    ) -> BorgiaResult<ProbabilisticValue> {
        let mut sum_mean = node.atomic_number as f64;
        let mut sum_variance = 0.1; // Base uncertainty

        // Sum contributions from neighbors
        for (neighbor_id, bond_order) in node.neighbors.iter().zip(&node.bond_orders) {
            if let Some(neighbor) = all_nodes.get(neighbor_id) {
                let weighted_contribution = neighbor.morgan_number.mean * bond_order;
                sum_mean += weighted_contribution;
                
                // Propagate uncertainty
                let weighted_variance = neighbor.morgan_number.variance * bond_order * bond_order;
                sum_variance += weighted_variance;
            }
        }

        // Add connectivity contribution
        sum_mean += node.connectivity.mean * 0.1;
        sum_variance += node.connectivity.variance * 0.01;

        Ok(ProbabilisticValue::new(sum_mean, sum_variance.sqrt()))
    }

    /// Generate molecular fingerprint based on Morgan numbers
    pub fn generate_fingerprint(&self, graph: &mut MolecularGraph, bits: usize) -> BorgiaResult<Vec<f64>> {
        let morgan_numbers = self.compute_morgan_numbers(graph)?;
        let mut fingerprint = vec![0.0; bits];

        for morgan_num in morgan_numbers {
            // Hash the Morgan number to fingerprint positions
            let hash = self.hash_morgan_number(&morgan_num);
            let positions = self.get_fingerprint_positions(hash, bits, 3); // Set 3 bits per feature

            for pos in positions {
                fingerprint[pos] += morgan_num.mean / (morgan_num.variance + 1.0);
            }
        }

        // Normalize fingerprint
        let max_val = fingerprint.iter().fold(0.0, |a, &b| a.max(b));
        if max_val > 0.0 {
            for val in &mut fingerprint {
                *val /= max_val;
            }
        }

        Ok(fingerprint)
    }

    /// Hash a Morgan number for fingerprint generation
    fn hash_morgan_number(&self, morgan_num: &ProbabilisticValue) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash both mean and variance for probabilistic fingerprinting
        let mean_bits = morgan_num.mean.to_bits();
        let variance_bits = morgan_num.variance.to_bits();
        
        mean_bits.hash(&mut hasher);
        variance_bits.hash(&mut hasher);
        
        hasher.finish()
    }

    /// Get fingerprint bit positions from hash
    fn get_fingerprint_positions(&self, hash: u64, bits: usize, count: usize) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut current_hash = hash;

        for _ in 0..count {
            let pos = (current_hash as usize) % bits;
            positions.push(pos);
            current_hash = current_hash.wrapping_mul(1103515245).wrapping_add(12345);
        }

        positions
    }

    /// Check if two molecular graphs are isomorphic using Morgan numbers
    pub fn are_isomorphic(&self, graph1: &mut MolecularGraph, graph2: &mut MolecularGraph) -> BorgiaResult<bool> {
        if graph1.node_count() != graph2.node_count() || graph1.edge_count() != graph2.edge_count() {
            return Ok(false);
        }

        let morgan1 = self.compute_morgan_numbers(graph1)?;
        let morgan2 = self.compute_morgan_numbers(graph2)?;

        // Sort Morgan numbers for comparison
        let mut sorted1 = morgan1;
        let mut sorted2 = morgan2;
        
        sorted1.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
        sorted2.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        // Compare sorted Morgan numbers with uncertainty tolerance
        for (m1, m2) in sorted1.iter().zip(sorted2.iter()) {
            let diff = (m1.mean - m2.mean).abs();
            let combined_uncertainty = (m1.variance + m2.variance).sqrt();
            
            if diff > 3.0 * combined_uncertainty {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute canonical labeling using Morgan numbers
    pub fn canonical_labeling(&self, graph: &mut MolecularGraph) -> BorgiaResult<Vec<usize>> {
        let morgan_numbers = self.compute_morgan_numbers(graph)?;
        
        // Create (morgan_number, original_index) pairs
        let mut labeled_nodes: Vec<(ProbabilisticValue, usize)> = morgan_numbers
            .into_iter()
            .enumerate()
            .map(|(i, morgan)| (morgan, i))
            .collect();

        // Sort by Morgan number (mean value)
        labeled_nodes.sort_by(|a, b| {
            a.0.mean.partial_cmp(&b.0.mean).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return the canonical ordering
        Ok(labeled_nodes.into_iter().map(|(_, idx)| idx).collect())
    }

    /// Compute automorphism group size estimate
    pub fn estimate_automorphism_group_size(&self, graph: &mut MolecularGraph) -> BorgiaResult<usize> {
        let morgan_numbers = self.compute_morgan_numbers(graph)?;
        let mut equivalence_classes = HashMap::new();

        // Group nodes by similar Morgan numbers
        for (idx, morgan) in morgan_numbers.iter().enumerate() {
            let key = (morgan.mean * 1000.0) as i64; // Discretize for grouping
            equivalence_classes.entry(key).or_insert_with(Vec::new).push(idx);
        }

        // Estimate automorphism group size as product of factorials of class sizes
        let mut group_size = 1;
        for class in equivalence_classes.values() {
            if class.len() > 1 {
                group_size *= factorial(class.len());
            }
        }

        Ok(group_size)
    }
}

/// Helper function to compute factorial
fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morgan_algorithm_basic() {
        let mut graph = MolecularGraph::new();
        
        // Create a simple molecule: H-C-H
        graph.add_node(0, 1); // H
        graph.add_node(1, 6); // C
        graph.add_node(2, 1); // H
        
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);

        let algorithm = MorganAlgorithm::default();
        let result = algorithm.compute_morgan_numbers(&mut graph);
        
        assert!(result.is_ok());
        let morgan_numbers = result.unwrap();
        assert_eq!(morgan_numbers.len(), 3);
    }

    #[test]
    fn test_fingerprint_generation() {
        let mut graph = MolecularGraph::new();
        
        // Create a simple molecule
        graph.add_node(0, 6); // C
        graph.add_node(1, 6); // C
        graph.add_edge(0, 1, 1.0);

        let algorithm = MorganAlgorithm::default();
        let fingerprint = algorithm.generate_fingerprint(&mut graph, 1024);
        
        assert!(fingerprint.is_ok());
        let fp = fingerprint.unwrap();
        assert_eq!(fp.len(), 1024);
    }
}
