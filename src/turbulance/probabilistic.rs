//! Probabilistic Text Processing - Points and Resolutions Paradigm
//! 
//! Core insight: "No point is 100% certain"
//! 
//! This module implements the revolutionary approach where all textual meaning
//! exists in probability space with inherent uncertainty.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{BorgiaError, BorgiaResult};

/// TextPoint represents text with uncertainty and multiple interpretations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPoint {
    pub content: String,
    pub confidence: f64,
    pub interpretations: Vec<TextInterpretation>,
    pub context_dependencies: HashMap<String, f64>,
    pub semantic_bounds: (f64, f64),
    pub evidence_strength: f64,
    pub contextual_relevance: f64,
}

/// Interpretation of a text point with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInterpretation {
    pub meaning: String,
    pub probability: f64,
    pub evidence: Vec<String>,
    pub semantic_category: SemanticCategory,
}

/// Semantic categories for interpretations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticCategory {
    Literal,
    Metaphorical,
    Technical,
    Colloquial,
    Domain(String),
    Temporal,
    Causal,
    Epistemic,
}

/// Resolution context for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionContext {
    pub domain: String,
    pub confidence_threshold: f64,
    pub strategy: ResolutionStrategy,
    pub temporal_context: Option<String>,
    pub cultural_context: Option<String>,
    pub purpose_context: Option<String>,
}

/// Resolution strategies for handling uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    MaximumLikelihood,
    Conservative,
    BayesianWeighted,
    FullDistribution,
    Exploratory,
}

/// Result of resolution function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionResult {
    /// High confidence, single interpretation
    Certain(Value),
    
    /// Multiple possibilities with probabilities
    Uncertain {
        possibilities: Vec<(Value, f64)>,
        confidence_interval: (f64, f64),
        aggregated_confidence: f64,
    },
    
    /// Context-dependent results
    Contextual {
        base_result: Value,
        context_variants: HashMap<String, (Value, f64)>,
        resolution_strategy: ResolutionStrategy,
    },
    
    /// Fuzzy results for inherently vague concepts
    Fuzzy {
        membership_function: Vec<(f64, f64)>,
        central_tendency: f64,
        spread: f64,
    },
}

/// Value types in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Point(TextPoint),
    Distribution(Vec<(Value, f64)>),
    Null,
}

/// Resolution function trait
pub trait ResolutionFunction {
    fn name(&self) -> &str;
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> BorgiaResult<ResolutionResult>;
    fn uncertainty_factor(&self) -> f64;
    fn can_handle(&self, point: &TextPoint) -> bool;
}

/// Debate platform for processing affirmations and contentions
#[derive(Debug, Clone)]
pub struct DebatePlatform {
    pub affirmations: Vec<Evidence>,
    pub contentions: Vec<Evidence>,
    pub resolution_strategies: Vec<ResolutionStrategy>,
    pub consensus_threshold: f64,
}

/// Evidence for debate platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub content: String,
    pub strength: f64,
    pub source: String,
    pub evidence_type: EvidenceType,
}

/// Types of evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    Empirical,
    Logical,
    Statistical,
    Expert,
    Contextual,
    Historical,
}

impl TextPoint {
    /// Create a new TextPoint
    pub fn new(content: String, confidence: f64) -> Self {
        Self {
            content,
            confidence,
            interpretations: Vec::new(),
            context_dependencies: HashMap::new(),
            semantic_bounds: (0.0, 1.0),
            evidence_strength: 0.5,
            contextual_relevance: 0.5,
        }
    }
    
    /// Add an interpretation to the point
    pub fn add_interpretation(&mut self, interpretation: TextInterpretation) {
        self.interpretations.push(interpretation);
    }
    
    /// Add context dependency
    pub fn add_context_dependency(&mut self, context: String, weight: f64) {
        self.context_dependencies.insert(context, weight);
    }
    
    /// Get most probable interpretation
    pub fn most_probable_interpretation(&self) -> Option<&TextInterpretation> {
        self.interpretations.iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
    }
    
    /// Calculate interpretation entropy (ambiguity measure)
    pub fn interpretation_entropy(&self) -> f64 {
        if self.interpretations.is_empty() {
            return 0.0;
        }
        
        let total_prob: f64 = self.interpretations.iter().map(|i| i.probability).sum();
        if total_prob == 0.0 {
            return 0.0;
        }
        
        -self.interpretations.iter()
            .map(|i| {
                let p = i.probability / total_prob;
                if p > 0.0 { p * p.log2() } else { 0.0 }
            })
            .sum::<f64>()
    }
}

impl DebatePlatform {
    /// Create new debate platform
    pub fn new() -> Self {
        Self {
            affirmations: Vec::new(),
            contentions: Vec::new(),
            resolution_strategies: vec![
                ResolutionStrategy::MaximumLikelihood,
                ResolutionStrategy::BayesianWeighted,
            ],
            consensus_threshold: 0.7,
        }
    }
    
    /// Add affirmation (supporting evidence)
    pub fn add_affirmation(&mut self, evidence: Evidence) {
        self.affirmations.push(evidence);
    }
    
    /// Add contention (challenging evidence)
    pub fn add_contention(&mut self, evidence: Evidence) {
        self.contentions.push(evidence);
    }
    
    /// Resolve debate to reach probabilistic consensus
    pub fn resolve_debate(&self, point: &TextPoint, strategy: &ResolutionStrategy) -> BorgiaResult<ResolutionResult> {
        match strategy {
            ResolutionStrategy::MaximumLikelihood => self.resolve_maximum_likelihood(point),
            ResolutionStrategy::Conservative => self.resolve_conservative(point),
            ResolutionStrategy::BayesianWeighted => self.resolve_bayesian(point),
            ResolutionStrategy::FullDistribution => self.resolve_full_distribution(point),
            ResolutionStrategy::Exploratory => self.resolve_exploratory(point),
        }
    }
    
    fn resolve_maximum_likelihood(&self, point: &TextPoint) -> BorgiaResult<ResolutionResult> {
        if let Some(best_interpretation) = point.most_probable_interpretation() {
            Ok(ResolutionResult::Certain(Value::String(best_interpretation.meaning.clone())))
        } else {
            Ok(ResolutionResult::Certain(Value::String(point.content.clone())))
        }
    }
    
    fn resolve_conservative(&self, point: &TextPoint) -> BorgiaResult<ResolutionResult> {
        // Choose interpretation with highest evidence strength
        let conservative_interpretation = point.interpretations.iter()
            .filter(|i| i.evidence.len() >= 2) // Require multiple evidence sources
            .max_by(|a, b| {
                let a_strength = a.probability * (a.evidence.len() as f64);
                let b_strength = b.probability * (b.evidence.len() as f64);
                a_strength.partial_cmp(&b_strength).unwrap()
            });
            
        if let Some(interpretation) = conservative_interpretation {
            Ok(ResolutionResult::Certain(Value::String(interpretation.meaning.clone())))
        } else {
            // Fall back to literal interpretation if no strong evidence
            Ok(ResolutionResult::Certain(Value::String(point.content.clone())))
        }
    }
    
    fn resolve_bayesian(&self, point: &TextPoint) -> BorgiaResult<ResolutionResult> {
        let affirmation_strength: f64 = self.affirmations.iter().map(|e| e.strength).sum();
        let contention_strength: f64 = self.contentions.iter().map(|e| e.strength).sum();
        
        let total_strength = affirmation_strength + contention_strength;
        if total_strength == 0.0 {
            return Ok(ResolutionResult::Certain(Value::String(point.content.clone())));
        }
        
        let bayesian_confidence = affirmation_strength / total_strength;
        
        let possibilities: Vec<(Value, f64)> = point.interpretations.iter()
            .map(|i| {
                let adjusted_prob = i.probability * bayesian_confidence;
                (Value::String(i.meaning.clone()), adjusted_prob)
            })
            .collect();
            
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (bayesian_confidence * 0.8, bayesian_confidence * 1.2),
            aggregated_confidence: bayesian_confidence,
        })
    }
    
    fn resolve_full_distribution(&self, point: &TextPoint) -> BorgiaResult<ResolutionResult> {
        let possibilities: Vec<(Value, f64)> = point.interpretations.iter()
            .map(|i| (Value::String(i.meaning.clone()), i.probability))
            .collect();
            
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: point.semantic_bounds,
            aggregated_confidence: point.confidence,
        })
    }
    
    fn resolve_exploratory(&self, point: &TextPoint) -> BorgiaResult<ResolutionResult> {
        // Include low-probability interpretations for discovery
        let mut possibilities: Vec<(Value, f64)> = point.interpretations.iter()
            .map(|i| (Value::String(i.meaning.clone()), i.probability))
            .collect();
            
        // Add speculative interpretations
        possibilities.push((Value::String(format!("Speculative: {}", point.content)), 0.1));
        
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (0.0, 1.0),
            aggregated_confidence: point.confidence * 0.8, // Lower confidence for exploration
        })
    }
}

/// Built-in resolution functions
pub struct ProbabilisticLengthResolution;

impl ResolutionFunction for ProbabilisticLengthResolution {
    fn name(&self) -> &str {
        "probabilistic_len"
    }
    
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> BorgiaResult<ResolutionResult> {
        let char_count = point.content.len() as f64;
        let word_count = point.content.split_whitespace().count() as f64;
        
        let possibilities = match context.domain.as_str() {
            "character_analysis" => vec![
                (Value::Float(char_count), 0.9),
                (Value::Float(word_count), 0.1),
            ],
            "linguistic" => vec![
                (Value::Float(word_count), 0.8),
                (Value::Float(char_count / 5.0), 0.2), // Average word length
            ],
            "informal" => {
                let contextual_length = if char_count < 50.0 { 0.3 } else if char_count < 200.0 { 0.6 } else { 0.9 };
                vec![
                    (Value::Float(contextual_length), 0.6),
                    (Value::Float(word_count), 0.4),
                ]
            },
            _ => vec![
                (Value::Float(char_count), 0.5),
                (Value::Float(word_count), 0.5),
            ],
        };
        
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (0.8, 0.95),
            aggregated_confidence: 0.87,
        })
    }
    
    fn uncertainty_factor(&self) -> f64 {
        0.13 // Length has relatively low uncertainty
    }
    
    fn can_handle(&self, _point: &TextPoint) -> bool {
        true // Can handle any text point
    }
}

/// Sentiment analysis resolution function
pub struct ProbabilisticSentimentResolution;

impl ResolutionFunction for ProbabilisticSentimentResolution {
    fn name(&self) -> &str {
        "probabilistic_sentiment"
    }
    
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> BorgiaResult<ResolutionResult> {
        let content = &point.content.to_lowercase();
        
        // Simple keyword-based sentiment with context adjustment
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"];
        
        let positive_count = positive_words.iter().filter(|&&word| content.contains(word)).count() as f64;
        let negative_count = negative_words.iter().filter(|&&word| content.contains(word)).count() as f64;
        
        let base_sentiment = if positive_count > negative_count {
            0.7 // Positive
        } else if negative_count > positive_count {
            0.3 // Negative
        } else {
            0.5 // Neutral
        };
        
        // Context adjustments
        let adjusted_sentiment = match context.domain.as_str() {
            "social_media" => {
                // Social media tends to be more expressive
                if base_sentiment > 0.5 { base_sentiment * 1.2 } else { base_sentiment * 0.8 }
            },
            "academic" => {
                // Academic text is more measured
                0.4 + (base_sentiment - 0.5) * 0.3
            },
            "business" => {
                // Business context is professional
                0.45 + (base_sentiment - 0.5) * 0.4
            },
            _ => base_sentiment,
        }.clamp(0.0, 1.0);
        
        let possibilities = vec![
            (Value::String("positive".to_string()), if adjusted_sentiment > 0.6 { adjusted_sentiment } else { 1.0 - adjusted_sentiment }),
            (Value::String("negative".to_string()), if adjusted_sentiment < 0.4 { 1.0 - adjusted_sentiment } else { adjusted_sentiment }),
            (Value::String("neutral".to_string()), 1.0 - (adjusted_sentiment - 0.5).abs() * 2.0),
        ];
        
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (0.6, 0.9),
            aggregated_confidence: point.confidence * 0.8,
        })
    }
    
    fn uncertainty_factor(&self) -> f64 {
        0.25 // Sentiment analysis has moderate uncertainty
    }
    
    fn can_handle(&self, point: &TextPoint) -> bool {
        !point.content.trim().is_empty()
    }
}

/// Merge multiple text points with uncertainty propagation
pub fn merge_points(points: &[TextPoint]) -> TextPoint {
    if points.is_empty() {
        return TextPoint::new("".to_string(), 0.0);
    }
    
    if points.len() == 1 {
        return points[0].clone();
    }
    
    let combined_content = points.iter()
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    
    // Calculate combined confidence using harmonic mean for conservative estimate
    let confidence_sum: f64 = points.iter().map(|p| 1.0 / p.confidence).sum();
    let combined_confidence = points.len() as f64 / confidence_sum;
    
    let mut merged = TextPoint::new(combined_content, combined_confidence);
    
    // Merge interpretations
    for point in points {
        for interpretation in &point.interpretations {
            merged.add_interpretation(interpretation.clone());
        }
    }
    
    // Merge context dependencies
    for point in points {
        for (context, weight) in &point.context_dependencies {
            let existing_weight = merged.context_dependencies.get(context).unwrap_or(&0.0);
            merged.context_dependencies.insert(context.clone(), existing_weight + weight);
        }
    }
    
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_point_creation() {
        let point = TextPoint::new("Hello world".to_string(), 0.9);
        assert_eq!(point.content, "Hello world");
        assert_eq!(point.confidence, 0.9);
        assert!(point.interpretations.is_empty());
    }
    
    #[test]
    fn test_interpretation_entropy() {
        let mut point = TextPoint::new("bank".to_string(), 0.8);
        
        point.add_interpretation(TextInterpretation {
            meaning: "financial institution".to_string(),
            probability: 0.6,
            evidence: vec!["context".to_string()],
            semantic_category: SemanticCategory::Technical,
        });
        
        point.add_interpretation(TextInterpretation {
            meaning: "river bank".to_string(),
            probability: 0.4,
            evidence: vec!["geographical".to_string()],
            semantic_category: SemanticCategory::Literal,
        });
        
        let entropy = point.interpretation_entropy();
        assert!(entropy > 0.0);
        assert!(entropy < 1.0); // Should be less than maximum entropy for 2 items
    }
    
    #[test]
    fn test_debate_platform() {
        let mut platform = DebatePlatform::new();
        
        platform.add_affirmation(Evidence {
            content: "Clinical trials show effectiveness".to_string(),
            strength: 0.8,
            source: "medical_journal".to_string(),
            evidence_type: EvidenceType::Empirical,
        });
        
        platform.add_contention(Evidence {
            content: "Sample size too small".to_string(),
            strength: 0.3,
            source: "peer_review".to_string(),
            evidence_type: EvidenceType::Statistical,
        });
        
        let point = TextPoint::new("Treatment is effective".to_string(), 0.7);
        let result = platform.resolve_debate(&point, &ResolutionStrategy::BayesianWeighted).unwrap();
        
        match result {
            ResolutionResult::Uncertain { aggregated_confidence, .. } => {
                assert!(aggregated_confidence > 0.5); // Should favor affirmations
            },
            _ => panic!("Expected uncertain result"),
        }
    }
} 