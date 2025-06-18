//! Error handling for the Borgia cheminformatics engine.

use std::fmt;
use thiserror::Error;

/// Result type alias for Borgia operations
pub type Result<T> = std::result::Result<T, BorgiaError>;

/// Comprehensive error types for Borgia operations
#[derive(Error, Debug)]
pub enum BorgiaError {
    /// Molecular parsing and representation errors
    #[error("Molecular parsing error: {message}")]
    MolecularParsingError { message: String },

    /// SMILES string parsing errors
    #[error("Invalid SMILES string: {smiles} - {reason}")]
    InvalidSmiles { smiles: String, reason: String },

    /// Probabilistic computation errors
    #[error("Probabilistic computation error: {operation} failed - {details}")]
    ProbabilisticError { operation: String, details: String },

    /// Fuzzy logic operation errors
    #[error("Fuzzy logic error: {message}")]
    FuzzyLogicError { message: String },

    /// Evidence processing errors
    #[error("Evidence processing error: {context} - {message}")]
    EvidenceError { context: String, message: String },

    /// Similarity calculation errors
    #[error("Similarity calculation error: {algorithm} failed - {reason}")]
    SimilarityError { algorithm: String, reason: String },

    /// Integration errors with upstream systems
    #[error("Integration error with {system}: {message}")]
    IntegrationError { system: String, message: String },

    /// Configuration and initialization errors
    #[error("Configuration error: {parameter} - {message}")]
    ConfigurationError { parameter: String, message: String },

    /// Mathematical computation errors
    #[error("Mathematical error: {operation} - {details}")]
    MathematicalError { operation: String, details: String },

    /// Memory and resource errors
    #[error("Resource error: {resource} - {message}")]
    ResourceError { resource: String, message: String },

    /// Validation errors
    #[error("Validation error: {field} - {message}")]
    ValidationError { field: String, message: String },

    /// Serialization/Deserialization errors
    #[error("Serialization error: {format} - {message}")]
    SerializationError { format: String, message: String },

    /// External library errors
    #[error("External library error: {library} - {message}")]
    ExternalError { library: String, message: String },

    /// Generic I/O errors
    #[error("I/O error: {operation} - {source}")]
    IoError {
        operation: String,
        #[source]
        source: std::io::Error,
    },

    /// Network and communication errors
    #[error("Network error: {endpoint} - {message}")]
    NetworkError { endpoint: String, message: String },

    /// Timeout errors
    #[error("Timeout error: {operation} exceeded {timeout_ms}ms")]
    TimeoutError { operation: String, timeout_ms: u64 },

    /// Insufficient data errors
    #[error("Insufficient data: {required} required, {available} available")]
    InsufficientDataError { required: String, available: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {operation} not supported for {context}")]
    UnsupportedOperationError { operation: String, context: String },
}

impl BorgiaError {
    /// Create a molecular parsing error
    pub fn molecular_parsing(message: impl Into<String>) -> Self {
        Self::MolecularParsingError {
            message: message.into(),
        }
    }

    /// Create an invalid SMILES error
    pub fn invalid_smiles(smiles: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidSmiles {
            smiles: smiles.into(),
            reason: reason.into(),
        }
    }

    /// Create a probabilistic computation error
    pub fn probabilistic(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::ProbabilisticError {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create a fuzzy logic error
    pub fn fuzzy_logic(message: impl Into<String>) -> Self {
        Self::FuzzyLogicError {
            message: message.into(),
        }
    }

    /// Create an evidence processing error
    pub fn evidence(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::EvidenceError {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Create a similarity calculation error
    pub fn similarity(algorithm: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::SimilarityError {
            algorithm: algorithm.into(),
            reason: reason.into(),
        }
    }

    /// Create an integration error
    pub fn integration(system: impl Into<String>, message: impl Into<String>) -> Self {
        Self::IntegrationError {
            system: system.into(),
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration(parameter: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            parameter: parameter.into(),
            message: message.into(),
        }
    }

    /// Create a mathematical error
    pub fn mathematical(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::MathematicalError {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create a validation error
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ValidationError {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::NetworkError { .. } => true,
            Self::TimeoutError { .. } => true,
            Self::ResourceError { .. } => true,
            Self::IoError { .. } => true,
            _ => false,
        }
    }

    /// Get error category for logging/monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::MolecularParsingError { .. } | Self::InvalidSmiles { .. } => "molecular",
            Self::ProbabilisticError { .. } => "probabilistic",
            Self::FuzzyLogicError { .. } => "fuzzy",
            Self::EvidenceError { .. } => "evidence",
            Self::SimilarityError { .. } => "similarity",
            Self::IntegrationError { .. } => "integration",
            Self::ConfigurationError { .. } => "configuration",
            Self::MathematicalError { .. } => "mathematical",
            Self::ResourceError { .. } => "resource",
            Self::ValidationError { .. } => "validation",
            Self::SerializationError { .. } => "serialization",
            Self::ExternalError { .. } => "external",
            Self::IoError { .. } => "io",
            Self::NetworkError { .. } => "network",
            Self::TimeoutError { .. } => "timeout",
            Self::InsufficientDataError { .. } => "data",
            Self::UnsupportedOperationError { .. } => "unsupported",
        }
    }
}

// Conversion from standard library errors
impl From<std::io::Error> for BorgiaError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            operation: "unknown".to_string(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for BorgiaError {
    fn from(err: serde_json::Error) -> Self {
        Self::SerializationError {
            format: "JSON".to_string(),
            message: err.to_string(),
        }
    }
}

impl From<toml::de::Error> for BorgiaError {
    fn from(err: toml::de::Error) -> Self {
        Self::SerializationError {
            format: "TOML".to_string(),
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = BorgiaError::molecular_parsing("Invalid molecule structure");
        assert_eq!(error.category(), "molecular");
        
        let error = BorgiaError::invalid_smiles("CCX", "Invalid atom X");
        assert!(error.to_string().contains("CCX"));
        
        let error = BorgiaError::probabilistic("bayesian_inference", "Insufficient samples");
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_error_categorization() {
        let error = BorgiaError::network_error("api.example.com", "Connection refused");
        assert_eq!(error.category(), "network");
        assert!(error.is_recoverable());
    }
}

impl BorgiaError {
    pub fn network_error(endpoint: impl Into<String>, message: impl Into<String>) -> Self {
        Self::NetworkError {
            endpoint: endpoint.into(),
            message: message.into(),
        }
    }
} 