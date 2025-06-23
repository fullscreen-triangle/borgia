//! Core Data Structures for Turbulance Language
//! 
//! Implements evidence networks, motion tracking, and other fundamental
//! data structures that support the revolutionary paradigms.

use std::collections::{HashMap, HashSet, BTreeMap};
use serde::{Serialize, Deserialize};
use crate::error::{BorgiaError, BorgiaResult};

/// Evidence Network - Core data structure for handling conflicting evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceNetwork {
    /// Nodes in the network
    nodes: HashMap<String, EvidenceNode>,
    /// Edges representing relationships
    edges: HashMap<String, Vec<Edge>>,
    /// Belief values for each node
    beliefs: HashMap<String, f64>,
    /// Network metadata
    metadata: NetworkMetadata,
}

/// Evidence node types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceNode {
    /// Molecular evidence
    Molecule {
        structure: String,
        formula: String,
        motion: Motion,
    },
    
    /// Textual evidence
    Text {
        content: String,
        confidence: f64,
        motion: Motion,
    },
    
    /// Spectral evidence
    Spectra {
        peaks: Vec<(f64, f64)>, // (m/z, intensity)
        retention_time: f64,
        motion: Motion,
    },
    
    /// Genomic evidence
    Genomic {
        sequence: String,
        gene_id: String,
        motion: Motion,
    },
    
    /// Generic evidence with properties
    Generic {
        properties: HashMap<String, String>,
        motion: Motion,
    },
}

/// Motion - Tracks the "movement" or evolution of evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motion {
    /// Description of the motion
    pub description: String,
    /// Velocity of change
    pub velocity: f64,
    /// Direction vector
    pub direction: Vec<f64>,
    /// Temporal evolution
    pub temporal_signature: Vec<(f64, f64)>, // (time, intensity)
}

/// Edge types representing relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    /// Supporting relationship
    Supports { strength: f64 },
    /// Contradicting relationship
    Contradicts { strength: f64 },
    /// Causal relationship
    Causes { probability: f64 },
    /// Temporal relationship
    Precedes { time_delta: f64 },
    /// Similarity relationship
    Similar { similarity: f64 },
    /// Logical implication
    Implies { certainty: f64 },
}

/// Edge in the evidence network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub edge_type: EdgeType,
    pub uncertainty: f64,
    pub metadata: EdgeMetadata,
}

/// Metadata for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    pub created_at: String,
    pub source: String,
    pub confidence: f64,
}

/// Network metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    pub created_at: String,
    pub last_updated: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub domain: String,
}

/// Sensitivity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    pub node_sensitivities: HashMap<String, f64>,
    pub edge_sensitivities: HashMap<String, f64>,
    pub critical_paths: Vec<Vec<String>>,
    pub stability_score: f64,
}

impl Motion {
    /// Create new motion with description
    pub fn new(description: String) -> Self {
        Self {
            description,
            velocity: 0.0,
            direction: vec![0.0, 0.0, 0.0],
            temporal_signature: Vec::new(),
        }
    }
    
    /// Create motion with velocity
    pub fn with_velocity(description: String, velocity: f64) -> Self {
        Self {
            description,
            velocity,
            direction: vec![1.0, 0.0, 0.0], // Default direction
            temporal_signature: Vec::new(),
        }
    }
    
    /// Add temporal point to signature
    pub fn add_temporal_point(&mut self, time: f64, intensity: f64) {
        self.temporal_signature.push((time, intensity));
    }
    
    /// Calculate motion similarity
    pub fn similarity(&self, other: &Motion) -> f64 {
        // Simplified similarity based on velocity and direction
        let velocity_sim = 1.0 - (self.velocity - other.velocity).abs() / (self.velocity.max(other.velocity) + 1.0);
        
        let direction_sim = if self.direction.len() == other.direction.len() {
            let dot_product: f64 = self.direction.iter()
                .zip(other.direction.iter())
                .map(|(a, b)| a * b)
                .sum();
            let mag_self: f64 = self.direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mag_other: f64 = other.direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            if mag_self > 0.0 && mag_other > 0.0 {
                dot_product / (mag_self * mag_other)
            } else {
                1.0
            }
        } else {
            0.5
        };
        
        (velocity_sim + direction_sim) / 2.0
    }
}

impl EvidenceNetwork {
    /// Create new evidence network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            beliefs: HashMap::new(),
            metadata: NetworkMetadata {
                created_at: chrono::Utc::now().to_rfc3339(),
                last_updated: chrono::Utc::now().to_rfc3339(),
                node_count: 0,
                edge_count: 0,
                domain: "general".to_string(),
            },
        }
    }
    
    /// Add node to network
    pub fn add_node(&mut self, id: &str, node: EvidenceNode) {
        self.nodes.insert(id.to_string(), node);
        self.edges.insert(id.to_string(), Vec::new());
        self.metadata.node_count = self.nodes.len();
        self.metadata.last_updated = chrono::Utc::now().to_rfc3339();
    }
    
    /// Add edge to network
    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: EdgeType, uncertainty: f64) {
        let edge = Edge {
            from: from.to_string(),
            to: to.to_string(),
            edge_type,
            uncertainty,
            metadata: EdgeMetadata {
                created_at: chrono::Utc::now().to_rfc3339(),
                source: "system".to_string(),
                confidence: 1.0 - uncertainty,
            },
        };
        
        if let Some(edges) = self.edges.get_mut(from) {
            edges.push(edge);
            self.metadata.edge_count = self.edges.values().map(|v| v.len()).sum();
            self.metadata.last_updated = chrono::Utc::now().to_rfc3339();
        }
    }
    
    /// Set belief value for a node
    pub fn set_belief(&mut self, node_id: &str, belief: f64) {
        self.beliefs.insert(node_id.to_string(), belief.clamp(0.0, 1.0));
    }
    
    /// Get belief value for a node
    pub fn get_belief(&self, node_id: &str) -> Option<f64> {
        self.beliefs.get(node_id).copied()
    }
    
    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<String, EvidenceNode> {
        &self.nodes
    }
    
    /// Get all edges
    pub fn edges(&self) -> &HashMap<String, Vec<Edge>> {
        &self.edges
    }
    
    /// Propagate beliefs through the network
    pub fn propagate_beliefs(&mut self) {
        let mut new_beliefs = self.beliefs.clone();
        
        // Simple belief propagation algorithm
        for (node_id, edges) in &self.edges {
            if let Some(current_belief) = self.beliefs.get(node_id) {
                for edge in edges {
                    let influence = match &edge.edge_type {
                        EdgeType::Supports { strength } => strength * current_belief,
                        EdgeType::Contradicts { strength } => -strength * current_belief,
                        EdgeType::Causes { probability } => probability * current_belief,
                        EdgeType::Implies { certainty } => certainty * current_belief,
                        _ => 0.0,
                    };
                    
                    let target_belief = new_beliefs.get(&edge.to).unwrap_or(&0.5);
                    let updated_belief = (target_belief + influence * (1.0 - edge.uncertainty)).clamp(0.0, 1.0);
                    new_beliefs.insert(edge.to.clone(), updated_belief);
                }
            }
        }
        
        self.beliefs = new_beliefs;
    }
    
    /// Perform sensitivity analysis
    pub fn sensitivity_analysis(&self) -> SensitivityAnalysis {
        let mut node_sensitivities = HashMap::new();
        let mut edge_sensitivities = HashMap::new();
        
        // Calculate node sensitivities based on connectivity
        for (node_id, _) in &self.nodes {
            let in_degree = self.edges.values()
                .flat_map(|edges| edges.iter())
                .filter(|edge| edge.to == *node_id)
                .count() as f64;
            
            let out_degree = self.edges.get(node_id)
                .map(|edges| edges.len())
                .unwrap_or(0) as f64;
            
            let sensitivity = (in_degree + out_degree) / (self.metadata.node_count as f64).max(1.0);
            node_sensitivities.insert(node_id.clone(), sensitivity);
        }
        
        // Calculate edge sensitivities based on uncertainty and strength
        for edges in self.edges.values() {
            for edge in edges {
                let edge_key = format!("{}→{}", edge.from, edge.to);
                let sensitivity = match &edge.edge_type {
                    EdgeType::Supports { strength } | EdgeType::Contradicts { strength } => {
                        strength * (1.0 - edge.uncertainty)
                    },
                    EdgeType::Causes { probability } => probability * (1.0 - edge.uncertainty),
                    EdgeType::Implies { certainty } => certainty * (1.0 - edge.uncertainty),
                    _ => 0.5 * (1.0 - edge.uncertainty),
                };
                edge_sensitivities.insert(edge_key, sensitivity);
            }
        }
        
        // Calculate overall stability
        let avg_node_sensitivity: f64 = node_sensitivities.values().sum::<f64>() / node_sensitivities.len() as f64;
        let avg_edge_sensitivity: f64 = edge_sensitivities.values().sum::<f64>() / edge_sensitivities.len().max(1) as f64;
        let stability_score = 1.0 - (avg_node_sensitivity + avg_edge_sensitivity) / 2.0;
        
        SensitivityAnalysis {
            node_sensitivities,
            edge_sensitivities,
            critical_paths: Vec::new(), // Simplified - would implement path finding
            stability_score,
        }
    }
    
    /// Find shortest path between nodes
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        // Simplified Dijkstra's algorithm
        let mut distances: HashMap<String, f64> = HashMap::new();
        let mut previous: HashMap<String, String> = HashMap::new();
        let mut unvisited: HashSet<String> = self.nodes.keys().cloned().collect();
        
        // Initialize distances
        for node_id in &unvisited {
            distances.insert(node_id.clone(), if node_id == from { 0.0 } else { f64::INFINITY });
        }
        
        while !unvisited.is_empty() {
            // Find unvisited node with minimum distance
            let current = unvisited.iter()
                .min_by(|a, b| {
                    distances.get(*a).unwrap_or(&f64::INFINITY)
                        .partial_cmp(distances.get(*b).unwrap_or(&f64::INFINITY))
                        .unwrap()
                })
                .cloned()?;
            
            if current == to {
                break;
            }
            
            unvisited.remove(&current);
            
            // Update distances to neighbors
            if let Some(edges) = self.edges.get(&current) {
                for edge in edges {
                    if unvisited.contains(&edge.to) {
                        let weight = edge.uncertainty; // Use uncertainty as weight
                        let alt = distances.get(&current).unwrap_or(&f64::INFINITY) + weight;
                        
                        if alt < *distances.get(&edge.to).unwrap_or(&f64::INFINITY) {
                            distances.insert(edge.to.clone(), alt);
                            previous.insert(edge.to.clone(), current.clone());
                        }
                    }
                }
            }
        }
        
        // Reconstruct path
        if !previous.contains_key(to) && from != to {
            return None;
        }
        
        let mut path = Vec::new();
        let mut current = to.to_string();
        
        while let Some(prev) = previous.get(&current) {
            path.push(current.clone());
            current = prev.clone();
        }
        path.push(from.to_string());
        path.reverse();
        
        Some(path)
    }
    
    /// Export network to DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph EvidenceNetwork {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");
        
        // Add nodes
        for (node_id, node) in &self.nodes {
            let label = match node {
                EvidenceNode::Molecule { structure, .. } => format!("Molecule\\n{}", structure),
                EvidenceNode::Text { content, .. } => format!("Text\\n{}", content.chars().take(20).collect::<String>()),
                EvidenceNode::Spectra { peaks, .. } => format!("Spectra\\n{} peaks", peaks.len()),
                EvidenceNode::Genomic { gene_id, .. } => format!("Gene\\n{}", gene_id),
                EvidenceNode::Generic { .. } => "Generic".to_string(),
            };
            
            let belief = self.beliefs.get(node_id).unwrap_or(&0.5);
            let color = if *belief > 0.7 { "green" } else if *belief < 0.3 { "red" } else { "yellow" };
            
            dot.push_str(&format!("  \"{}\" [label=\"{}\\nBelief: {:.2}\", fillcolor={}, style=filled];\n", 
                                node_id, label, belief, color));
        }
        
        dot.push_str("\n");
        
        // Add edges
        for edges in self.edges.values() {
            for edge in edges {
                let style = match &edge.edge_type {
                    EdgeType::Supports { .. } => "solid",
                    EdgeType::Contradicts { .. } => "dashed",
                    EdgeType::Causes { .. } => "bold",
                    _ => "dotted",
                };
                
                let color = if edge.uncertainty < 0.3 { "black" } else if edge.uncertainty < 0.7 { "gray" } else { "lightgray" };
                
                dot.push_str(&format!("  \"{}\" -> \"{}\" [style={}, color={}];\n", 
                                    edge.from, edge.to, style, color));
            }
        }
        
        dot.push_str("}\n");
        dot
    }
}

impl Default for EvidenceNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuzzy container for probabilistic collections
#[derive(Debug, Clone)]
pub struct FuzzyContainer<T> {
    items: Vec<(T, f64)>, // (item, membership_degree)
    membership_function: Option<fn(&T) -> f64>,
}

impl<T> FuzzyContainer<T> {
    /// Create new fuzzy container
    pub fn new(membership_function: Option<fn(&T) -> f64>) -> Self {
        Self {
            items: Vec::new(),
            membership_function,
        }
    }
    
    /// Add item with automatic membership calculation
    pub fn add(&mut self, item: T) where T: Clone {
        let membership = if let Some(func) = self.membership_function {
            func(&item)
        } else {
            1.0
        };
        self.items.push((item, membership));
    }
    
    /// Add item with explicit membership
    pub fn add_with_membership(&mut self, item: T, membership: f64) {
        self.items.push((item, membership.clamp(0.0, 1.0)));
    }
    
    /// Filter by membership threshold
    pub fn filter_by_membership(&self, threshold: f64) -> Vec<&T> {
        self.items.iter()
            .filter(|(_, membership)| *membership >= threshold)
            .map(|(item, _)| item)
            .collect()
    }
    
    /// Get all items with memberships
    pub fn items(&self) -> &[(T, f64)] {
        &self.items
    }
    
    /// Calculate container entropy
    pub fn entropy(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        
        let total_membership: f64 = self.items.iter().map(|(_, m)| m).sum();
        if total_membership == 0.0 {
            return 0.0;
        }
        
        -self.items.iter()
            .map(|(_, membership)| {
                let p = membership / total_membership;
                if p > 0.0 { p * p.log2() } else { 0.0 }
            })
            .sum()
    }
}

/// Fuzzy map for key-value relationships with similarity
#[derive(Debug, Clone)]
pub struct FuzzyMap<K, V> {
    mappings: Vec<(K, V, f64)>, // (key, value, certainty)
    key_similarity_function: Option<fn(&K, &K) -> f64>,
}

impl<K, V> FuzzyMap<K, V> {
    /// Create new fuzzy map
    pub fn new(key_similarity_function: Option<fn(&K, &K) -> f64>) -> Self {
        Self {
            mappings: Vec::new(),
            key_similarity_function,
        }
    }
    
    /// Add mapping
    pub fn insert(&mut self, key: K, value: V, certainty: f64) {
        self.mappings.push((key, value, certainty.clamp(0.0, 1.0)));
    }
    
    /// Fuzzy lookup with similarity matching
    pub fn get(&self, query_key: &K) -> Option<(&V, f64)> where K: PartialEq {
        let mut best_match: Option<(&V, f64)> = None;
        let mut best_similarity = 0.0;
        
        for (key, value, certainty) in &self.mappings {
            let similarity = if let Some(sim_func) = self.key_similarity_function {
                sim_func(query_key, key)
            } else if query_key == key {
                1.0
            } else {
                0.0
            };
            
            let combined_score = similarity * certainty;
            if combined_score > best_similarity {
                best_similarity = combined_score;
                best_match = Some((value, combined_score));
            }
        }
        
        best_match
    }
    
    /// Get all mappings
    pub fn mappings(&self) -> &[(K, V, f64)] {
        &self.mappings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_evidence_network_creation() {
        let network = EvidenceNetwork::new();
        assert_eq!(network.nodes().len(), 0);
        assert_eq!(network.metadata.node_count, 0);
    }
    
    #[test]
    fn test_motion_similarity() {
        let motion1 = Motion::with_velocity("forward".to_string(), 1.0);
        let motion2 = Motion::with_velocity("forward".to_string(), 1.2);
        
        let similarity = motion1.similarity(&motion2);
        assert!(similarity > 0.8); // Should be similar
    }
    
    #[test]
    fn test_fuzzy_container() {
        let mut container = FuzzyContainer::new(None);
        container.add_with_membership("high_confidence", 0.9);
        container.add_with_membership("medium_confidence", 0.6);
        container.add_with_membership("low_confidence", 0.2);
        
        let high_items = container.filter_by_membership(0.8);
        assert_eq!(high_items.len(), 1);
        assert_eq!(high_items[0], &"high_confidence");
    }
    
    #[test]
    fn test_belief_propagation() {
        let mut network = EvidenceNetwork::new();
        
        network.add_node("A", EvidenceNode::Generic {
            properties: HashMap::new(),
            motion: Motion::new("Test A".to_string()),
        });
        
        network.add_node("B", EvidenceNode::Generic {
            properties: HashMap::new(),
            motion: Motion::new("Test B".to_string()),
        });
        
        network.add_edge("A", "B", EdgeType::Supports { strength: 0.8 }, 0.1);
        network.set_belief("A", 0.9);
        network.set_belief("B", 0.5);
        
        network.propagate_beliefs();
        
        let belief_b = network.get_belief("B").unwrap();
        assert!(belief_b > 0.5); // Should have increased due to support from A
    }
} 