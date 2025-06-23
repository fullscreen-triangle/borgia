---
layout: page
title: "API Reference"
permalink: /api-reference/
---

# API Reference

## Core Types and Structures

### IntegratedBMDSystem

The main entry point for the Borgia framework, orchestrating all BMD scales and cross-scale coordination.

```rust
pub struct IntegratedBMDSystem {
    // Private fields
}

impl IntegratedBMDSystem {
    /// Creates a new integrated BMD system with default configuration
    pub fn new() -> Self

    /// Executes cross-scale analysis on molecular data
    pub fn execute_cross_scale_analysis(
        &mut self,
        molecules: Vec<String>,
        scales: Vec<BMDScale>
    ) -> BorgiaResult<BMDAnalysisResult>

    /// Demonstrates Mizraji's prisoner parable
    pub fn demonstrate_prisoner_parable(&mut self) -> BorgiaResult<PrisonerParableResult>

    /// Performs environmental noise-enhanced cheminformatics
    pub fn noise_enhanced_cheminformatics(
        &mut self,
        small_dataset: Vec<String>
    ) -> BorgiaResult<EnhancedDatasetResult>
}
```

### BMDScale

Enumeration of the five BMD scales implemented in the framework.

```rust
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum BMDScale {
    /// Quantum scale (10^-15 to 10^-12 seconds)
    Quantum,
    /// Molecular scale (10^-12 to 10^-9 seconds)
    Molecular,
    /// Cellular scale (10^-9 to 10^-6 seconds)
    Cellular,
    /// Environmental scale (10^-6 to 10^-3 seconds)
    Environmental,
    /// Hardware scale (10^-3 to 10^0 seconds)
    Hardware,
    /// Cognitive scale (10^0 to 10^2 seconds)
    Cognitive,
}
```

### BMDAnalysisResult

Result structure for cross-scale BMD analysis.

```rust
#[derive(Debug, Clone)]
pub struct BMDAnalysisResult {
    /// Thermodynamic amplification factor achieved
    pub amplification_factor: f64,
    /// Information catalysis results
    pub catalysis_results: Vec<CatalysisResult>,
    /// Cross-scale coordination results
    pub scale_coordination_results: Vec<CoordinationResult>,
    /// Thermodynamic consistency validation
    pub thermodynamic_consistency: bool,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}
```

## Molecular Representation

### Molecule

Core molecular representation structure.

```rust
#[derive(Debug, Clone)]
pub struct Molecule {
    /// Unique identifier for the molecule
    pub id: String,
    /// SMILES representation
    pub smiles: String,
    /// Molecular weight
    pub molecular_weight: f64,
    /// Calculated properties
    pub properties: MolecularProperties,
}

impl Molecule {
    /// Creates a molecule from SMILES string
    pub fn from_smiles(smiles: &str) -> BorgiaResult<Self>

    /// Calculates molecular fingerprint
    pub fn calculate_fingerprint(&self) -> BorgiaResult<MolecularFingerprint>

    /// Computes similarity to another molecule
    pub fn similarity_to(&self, other: &Molecule) -> BorgiaResult<f64>
}
```

### MolecularProperties

Calculated molecular properties.

```rust
#[derive(Debug, Clone)]
pub struct MolecularProperties {
    /// LogP (partition coefficient)
    pub log_p: f64,
    /// Topological polar surface area
    pub tpsa: f64,
    /// Number of hydrogen bond donors
    pub hbd_count: u32,
    /// Number of hydrogen bond acceptors
    pub hba_count: u32,
    /// Number of rotatable bonds
    pub rotatable_bonds: u32,
    /// Lipinski's Rule of Five compliance
    pub lipinski_compliant: bool,
}
```

### MolecularFingerprint

Molecular fingerprint for similarity analysis.

```rust
#[derive(Debug, Clone)]
pub struct MolecularFingerprint {
    /// Binary fingerprint data
    pub bits: Vec<bool>,
    /// Fingerprint algorithm used
    pub algorithm: FingerprintAlgorithm,
    /// Bit length
    pub length: usize,
}

impl MolecularFingerprint {
    /// Calculates Tanimoto similarity with another fingerprint
    pub fn tanimoto_similarity(&self, other: &MolecularFingerprint) -> f64

    /// Calculates Dice similarity with another fingerprint
    pub fn dice_similarity(&self, other: &MolecularFingerprint) -> f64
}
```

## BMD-Specific APIs

### QuantumBMD

Quantum-scale biological Maxwell's demon implementation.

```rust
pub struct QuantumBMD {
    // Private fields
}

impl QuantumBMD {
    /// Creates a new quantum BMD
    pub fn new() -> Self

    /// Creates a coherent quantum event
    pub fn create_coherent_event(&mut self, energy: f64) -> BorgiaResult<QuantumEvent>

    /// Processes quantum information
    pub fn process_quantum_information(&mut self, info: &QuantumInformation) 
        -> BorgiaResult<QuantumProcessingResult>
}
```

### MolecularBMD

Molecular-scale biological Maxwell's demon implementation.

```rust
pub struct MolecularBMD {
    // Private fields
}

impl MolecularBMD {
    /// Creates a new molecular BMD
    pub fn new() -> Self

    /// Analyzes substrate binding
    pub fn analyze_substrate_binding(&mut self, molecule: &Molecule) 
        -> BorgiaResult<BindingAnalysis>

    /// Performs molecular recognition
    pub fn recognize_molecular_pattern(&mut self, molecules: &[Molecule]) 
        -> BorgiaResult<RecognitionResult>
}
```

### EnvironmentalBMD

Environmental-scale biological Maxwell's demon implementation.

```rust
pub struct EnvironmentalBMD {
    // Private fields
}

impl EnvironmentalBMD {
    /// Creates a new environmental BMD
    pub fn new() -> Self

    /// Captures environmental noise from screen
    pub fn capture_environmental_noise(&mut self) -> BorgiaResult<EnvironmentalNoise>

    /// Enhances dataset using environmental noise
    pub fn enhance_dataset(&mut self, dataset: &[Molecule], noise: &EnvironmentalNoise) 
        -> BorgiaResult<Vec<Molecule>>
}
```

### HardwareBMD

Hardware-scale biological Maxwell's demon implementation.

```rust
pub struct HardwareBMD {
    // Private fields
}

impl HardwareBMD {
    /// Creates a new hardware BMD
    pub fn new() -> Self

    /// Performs LED-based molecular spectroscopy
    pub fn perform_led_spectroscopy(&mut self, sample: &MolecularSample) 
        -> BorgiaResult<SpectroscopyResult>

    /// Enhances consciousness coupling at 650nm
    pub fn enhance_consciousness_coupling(&mut self, wavelength: u32) 
        -> BorgiaResult<CouplingResult>
}
```

## Information Catalysis

### InformationCatalysisEngine

Implementation of Mizraji's information catalysis equation.

```rust
pub struct InformationCatalysisEngine {
    // Private fields
}

impl InformationCatalysisEngine {
    /// Creates a new information catalysis engine
    pub fn new() -> Self

    /// Executes information catalysis
    pub fn execute_catalysis(&mut self, input_info: &InformationPacket, 
                           context: &CatalysisContext) -> BorgiaResult<CatalysisResult>

    /// Calculates amplification factor
    pub fn calculate_amplification_factor(&self, input_energy: f64, output_consequences: f64) -> f64
}
```

### CatalysisResult

Result of information catalysis operation.

```rust
#[derive(Debug, Clone)]
pub struct CatalysisResult {
    /// Input information after filtering
    pub input_information: FilteredInformation,
    /// Output information after channeling
    pub output_information: ChanneledOutput,
    /// Achieved amplification factor
    pub amplification_factor: f64,
    /// Thermodynamic impact
    pub thermodynamic_impact: f64,
    /// Energy cost of the catalysis
    pub energy_cost: f64,
}
```

## Cross-Scale Coordination

### CrossScaleCoordinator

Manages information transfer and synchronization between scales.

```rust
pub struct CrossScaleCoordinator {
    // Private fields
}

impl CrossScaleCoordinator {
    /// Creates a new cross-scale coordinator
    pub fn new() -> Self

    /// Coordinates information transfer between two scales
    pub fn coordinate_scales(&mut self, scale1: BMDScale, scale2: BMDScale, 
                           information: &InformationPacket) -> BorgiaResult<CoordinationResult>

    /// Calculates coupling coefficient between scales
    pub fn calculate_coupling_coefficient(&self, scale1: BMDScale, scale2: BMDScale) -> f64
}
```

### CoordinationResult

Result of cross-scale coordination operation.

```rust
#[derive(Debug, Clone)]
pub struct CoordinationResult {
    /// Source scale
    pub source_scale: BMDScale,
    /// Target scale
    pub target_scale: BMDScale,
    /// Information transfer efficiency
    pub transfer_efficiency: f64,
    /// Synchronization window used
    pub synchronization_window: Duration,
    /// Coupling strength achieved
    pub coupling_strength: f64,
}
```

## Cheminformatics Integration

### SMILESProcessor

SMILES molecular representation processor.

```rust
pub struct SMILESProcessor {
    // Private fields
}

impl SMILESProcessor {
    /// Creates a new SMILES processor
    pub fn new() -> Self

    /// Parses SMILES string into molecule
    pub fn parse_smiles(&mut self, smiles: &str) -> BorgiaResult<Molecule>

    /// Validates SMILES string
    pub fn validate_smiles(&self, smiles: &str) -> bool

    /// Canonicalizes SMILES string
    pub fn canonicalize_smiles(&self, smiles: &str) -> BorgiaResult<String>
}
```

### SMARTSProcessor

SMARTS molecular pattern processor.

```rust
pub struct SMARTSProcessor {
    // Private fields
}

impl SMARTSProcessor {
    /// Creates a new SMARTS processor
    pub fn new() -> Self

    /// Parses SMARTS pattern
    pub fn parse_smarts(&mut self, smarts: &str) -> BorgiaResult<MolecularPattern>

    /// Matches SMARTS pattern against molecule
    pub fn match_pattern(&self, pattern: &MolecularPattern, molecule: &Molecule) 
        -> BorgiaResult<Vec<Match>>
}
```

### MolecularDatabase

Database for storing and querying molecules.

```rust
pub struct MolecularDatabase {
    // Private fields
}

impl MolecularDatabase {
    /// Creates a new molecular database
    pub fn new() -> Self

    /// Adds molecule to database
    pub fn add_molecule(&mut self, molecule: Molecule) -> BorgiaResult<()>

    /// Searches for similar molecules
    pub fn search_similar(&self, query: &Molecule, threshold: f64) 
        -> BorgiaResult<Vec<SimilarityMatch>>

    /// Performs substructure search
    pub fn substructure_search(&self, pattern: &MolecularPattern) 
        -> BorgiaResult<Vec<Molecule>>
}
```

## Performance and Metrics

### PerformanceMetrics

Performance metrics for BMD operations.

```rust
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total execution time
    pub execution_time: Duration,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of quantum cycles
    pub quantum_cycles: u64,
    /// Number of molecular cycles
    pub molecular_cycles: u64,
    /// Number of environmental cycles
    pub environmental_cycles: u64,
    /// Number of hardware cycles
    pub hardware_cycles: u64,
    /// Cross-scale coordination count
    pub cross_scale_coordinations: u64,
    /// Total amplification achieved
    pub total_amplification: f64,
}
```

### BMDMetrics

Specific metrics for BMD operations.

```rust
#[derive(Debug, Clone)]
pub struct BMDMetrics {
    /// Quantum BMD metrics
    pub quantum_metrics: QuantumMetrics,
    /// Molecular BMD metrics
    pub molecular_metrics: MolecularMetrics,
    /// Environmental BMD metrics
    pub environmental_metrics: EnvironmentalMetrics,
    /// Hardware BMD metrics
    pub hardware_metrics: HardwareMetrics,
    /// Overall system efficiency
    pub system_efficiency: f64,
}
```

## Error Handling

### BorgiaError

Comprehensive error type for the Borgia framework.

```rust
#[derive(Debug, thiserror::Error)]
pub enum BorgiaError {
    #[error("Quantum coherence lost: {message}")]
    QuantumCoherenceLoss { message: String },
    
    #[error("Molecular recognition failed: {molecule_id}")]
    MolecularRecognitionFailure { molecule_id: String },
    
    #[error("Environmental noise processing error: {details}")]
    EnvironmentalProcessingError { details: String },
    
    #[error("Hardware integration failure: {hardware_type}")]
    HardwareIntegrationFailure { hardware_type: String },
    
    #[error("Cross-scale coordination failed between {scale1:?} and {scale2:?}")]
    CrossScaleCoordinationFailure { scale1: BMDScale, scale2: BMDScale },
    
    #[error("Information catalysis error: {context}")]
    InformationCatalysisError { context: String },
    
    #[error("Thermodynamic consistency violation: {details}")]
    ThermodynamicInconsistency { details: String },
    
    #[error("SMILES parsing error: {smiles}")]
    SMILESParsingError { smiles: String },
    
    #[error("SMARTS pattern error: {pattern}")]
    SMARTSPatternError { pattern: String },
    
    #[error("Database error: {details}")]
    DatabaseError { details: String },
    
    #[error("I/O error: {source}")]
    IoError { #[from] source: std::io::Error },
    
    #[error("Serialization error: {source}")]
    SerializationError { #[from] source: serde_json::Error },
}
```

### BorgiaResult

Result type alias for convenient error handling.

```rust
pub type BorgiaResult<T> = Result<T, BorgiaError>;
```

## Utility Functions

### High-Level API Functions

Convenient functions for common operations.

```rust
/// Demonstrates Mizraji's prisoner parable
pub fn demonstrate_prisoner_parable() -> BorgiaResult<PrisonerParableResult>

/// Performs noise-enhanced cheminformatics analysis
pub fn noise_enhanced_cheminformatics(molecules: Vec<String>) 
    -> BorgiaResult<EnhancedAnalysisResult>

/// Executes hardware-integrated molecular spectroscopy
pub fn hardware_molecular_spectroscopy(molecules: Vec<String>) 
    -> BorgiaResult<SpectroscopyAnalysisResult>

/// Validates thermodynamic consistency across scales
pub fn validate_thermodynamic_consistency(system: &IntegratedBMDSystem) 
    -> BorgiaResult<ConsistencyReport>
```

### Molecular Utility Functions

```rust
/// Calculates molecular similarity using Tanimoto coefficient
pub fn calculate_tanimoto_similarity(mol1: &Molecule, mol2: &Molecule) -> BorgiaResult<f64>

/// Generates molecular descriptor vector
pub fn generate_molecular_descriptors(molecule: &Molecule) -> BorgiaResult<Vec<f64>>

/// Predicts molecular properties using BMD-enhanced methods
pub fn predict_molecular_properties(molecule: &Molecule) -> BorgiaResult<PropertyPrediction>

/// Validates Lipinski's Rule of Five
pub fn validate_lipinski_rule(molecule: &Molecule) -> BorgiaResult<LipinskiResult>
```

### Information Theory Utilities

```rust
/// Calculates Shannon entropy of information
pub fn calculate_shannon_entropy(probabilities: &[f64]) -> f64

/// Computes information transfer efficiency
pub fn compute_information_transfer_efficiency(
    input_info: &InformationPacket, 
    output_info: &InformationPacket
) -> f64

/// Measures cross-scale information coherence
pub fn measure_cross_scale_coherence(
    scale1_info: &ScaleInformation, 
    scale2_info: &ScaleInformation
) -> f64
```

## Configuration and Initialization

### BorgiaConfig

Configuration structure for the Borgia framework.

```rust
#[derive(Debug, Clone)]
pub struct BorgiaConfig {
    /// Enable quantum BMD
    pub enable_quantum_bmd: bool,
    /// Enable molecular BMD
    pub enable_molecular_bmd: bool,
    /// Enable environmental BMD
    pub enable_environmental_bmd: bool,
    /// Enable hardware BMD
    pub enable_hardware_bmd: bool,
    /// Cross-scale coordination threshold
    pub coordination_threshold: f64,
    /// Maximum amplification factor allowed
    pub max_amplification_factor: f64,
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for BorgiaConfig {
    fn default() -> Self {
        Self {
            enable_quantum_bmd: true,
            enable_molecular_bmd: true,
            enable_environmental_bmd: true,
            enable_hardware_bmd: true,
            coordination_threshold: 0.8,
            max_amplification_factor: 10000.0,
            optimization_level: OptimizationLevel::Balanced,
        }
    }
}
```

### Initialization Functions

```rust
/// Initializes Borgia framework with default configuration
pub fn initialize() -> BorgiaResult<IntegratedBMDSystem>

/// Initializes Borgia framework with custom configuration
pub fn initialize_with_config(config: BorgiaConfig) -> BorgiaResult<IntegratedBMDSystem>

/// Initializes specific BMD scales only
pub fn initialize_scales(scales: Vec<BMDScale>) -> BorgiaResult<IntegratedBMDSystem>
```

## Examples

### Basic Usage Example

```rust
use borgia::{initialize, BMDScale};

fn main() -> borgia::BorgiaResult<()> {
    // Initialize the framework
    let mut system = initialize()?;
    
    // Define molecules for analysis
    let molecules = vec![
        "CCO".to_string(),        // Ethanol
        "CC(=O)O".to_string(),    // Acetic acid
        "C6H12O6".to_string(),    // Glucose
    ];
    
    // Execute cross-scale analysis
    let result = system.execute_cross_scale_analysis(
        molecules,
        vec![BMDScale::Quantum, BMDScale::Molecular, BMDScale::Environmental]
    )?;
    
    println!("Amplification factor: {:.2}×", result.amplification_factor);
    println!("Thermodynamic consistency: {}", result.thermodynamic_consistency);
    
    Ok(())
}
```

### Advanced Analysis Example

```rust
use borgia::{initialize, demonstrate_prisoner_parable, noise_enhanced_cheminformatics};

fn main() -> borgia::BorgiaResult<()> {
    // Demonstrate Mizraji's prisoner parable
    let parable_result = demonstrate_prisoner_parable()?;
    println!("Prisoner parable amplification: {:.0}×", parable_result.amplification_factor);
    
    // Perform noise-enhanced cheminformatics
    let small_dataset = vec![
        "CCO".to_string(),
        "CC(=O)O".to_string(),
    ];
    
    let enhanced_result = noise_enhanced_cheminformatics(small_dataset)?;
    println!("Dataset enhanced from {} to {} molecules", 
             enhanced_result.original_size, 
             enhanced_result.enhanced_size);
    
    Ok(())
}
```

---

*This API reference provides comprehensive documentation for all public interfaces in the Borgia framework, enabling developers to effectively utilize biological Maxwell's demons for computational chemistry and bioinformatics applications.* 