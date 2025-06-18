//! Integration with upstream systems.

use crate::error::{BorgiaError, Result};
use serde::{Deserialize, Serialize};

/// Integration with Hegel system
#[derive(Debug, Clone)]
pub struct HegelIntegration {
    pub endpoint: String,
}

/// Integration with Lavoisier system
#[derive(Debug, Clone)]
pub struct LavoisierIntegration {
    pub endpoint: String,
}

/// Integration with Gospel system
#[derive(Debug, Clone)]
pub struct GospelIntegration {
    pub endpoint: String,
}

/// Integration with Bene Gesserit system
#[derive(Debug, Clone)]
pub struct BeneGesseritIntegration {
    pub endpoint: String,
}

impl HegelIntegration {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
}

impl LavoisierIntegration {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
}

impl GospelIntegration {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
}

impl BeneGesseritIntegration {
    pub fn new(endpoint: String) -> Self {
        Self { endpoint }
    }
} 