//! Evidence processing and propagation for Borgia.

use crate::error::{BorgiaError, Result};
use crate::probabilistic::{ProbabilisticValue, Evidence};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evidence processor for handling upstream information
#[derive(Debug, Clone)]
pub struct EvidenceProcessor {
    pub evidence_cache: HashMap<String, Evidence>,
}

/// Context for evidence evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceContext {
    pub source_system: String,
    pub confidence_level: f64,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

/// Evidence propagation system
#[derive(Debug, Clone)]
pub struct EvidencePropagation {
    pub propagation_rules: HashMap<String, f64>,
}

impl EvidenceProcessor {
    pub fn new() -> Self {
        Self {
            evidence_cache: HashMap::new(),
        }
    }

    pub fn add_evidence(&mut self, key: String, evidence: Evidence) {
        self.evidence_cache.insert(key, evidence);
    }

    pub fn get_evidence(&self, key: &str) -> Option<&Evidence> {
        self.evidence_cache.get(key)
    }
}

impl Default for EvidenceProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EvidencePropagation {
    pub fn new() -> Self {
        Self {
            propagation_rules: HashMap::new(),
        }
    }
}

impl Default for EvidencePropagation {
    fn default() -> Self {
        Self::new()
    }
} 