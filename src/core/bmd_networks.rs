//! Multi-scale Biological Maxwell's Demon Networks
//! 
//! Implementation of multi-scale BMD networks operating across quantum (10^-15s),
//! molecular (10^-9s), and environmental (10^2s) timescales with cross-scale
//! coordination and thermodynamic amplification.

use crate::error::{BorgiaError, Result};
use crate::core::{InformationPacket, ProcessedInformation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use ndarray::{Array1, Array2};

/// Configuration for multi-scale BMD networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDNetworkConfiguration {
    /// Scale ratios between different levels
    pub scale_ratios: ScaleRatios,
    
    /// Cross-scale coupling strength
    pub coordination_strength: f64,
    
    /// Target amplification factor
    pub amplification_target: f64,
    
    /// Coherence requirements
    pub coherence_requirements: CoherenceRequirements,
    
    /// Network topology parameters
    pub topology: NetworkTopology,
    
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

impl Default for BMDNetworkConfiguration {
    fn default() -> Self {
        Self {
            scale_ratios: ScaleRatios::default(),
            coordination_strength: 0.85,
            amplification_target: 1000.0,
            coherence_requirements: CoherenceRequirements::default(),
            topology: NetworkTopology::default(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Scale ratios between different temporal levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleRatios {
    /// Quantum to molecular scale ratio
    pub quantum_molecular: f64,
    
    /// Molecular to environmental scale ratio
    pub molecular_environmental: f64,
    
    /// Direct quantum to environmental ratio
    pub quantum_environmental: f64,
}

impl Default for ScaleRatios {
    fn default() -> Self {
        Self {
            quantum_molecular: 1e6,     // 10^-15s to 10^-9s
            molecular_environmental: 1e11,  // 10^-9s to 10^2s
            quantum_environmental: 1e17,    // 10^-15s to 10^2s
        }
    }
}

/// Coherence requirements for different scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceRequirements {
    /// Quantum coherence time
    pub quantum_coherence_time: Duration,
    
    /// Molecular coherence time
    pub molecular_coherence_time: Duration,
    
    /// Environmental coherence time
    pub environmental_coherence_time: Duration,
    
    /// Cross-scale coherence preservation
    pub cross_scale_preservation: f64,
}

impl Default for CoherenceRequirements {
    fn default() -> Self {
        Self {
            quantum_coherence_time: Duration::from_nanos(1),
            molecular_coherence_time: Duration::from_micros(1),
            environmental_coherence_time: Duration::from_secs(100),
            cross_scale_preservation: 0.95,
        }
    }
}

/// Network topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkTopology {
    /// Number of BMD units per scale
    pub units_per_scale: HashMap<String, usize>,
    
    /// Connectivity patterns
    pub connectivity_patterns: ConnectivityPatterns,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        let mut units_per_scale = HashMap::new();
        units_per_scale.insert("quantum".to_string(), 1000);
        units_per_scale.insert("molecular".to_string(), 100);
        units_per_scale.insert("environmental".to_string(), 10);
        
        Self {
            units_per_scale,
            connectivity_patterns: ConnectivityPatterns::FullyConnected,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Connectivity patterns for BMD networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityPatterns {
    FullyConnected,
    HierarchicalTree,
    SmallWorld,
    ScaleFree,
    Custom(Vec<(usize, usize)>),
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    DynamicWeighting,
}

/// Performance thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Minimum amplification factor
    pub min_amplification: f64,
    
    /// Maximum processing latency
    pub max_latency: Duration,
    
    /// Minimum throughput
    pub min_throughput: f64,
    
    /// Minimum success rate
    pub min_success_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            min_amplification: 100.0,
            max_latency: Duration::from_millis(10),
            min_throughput: 1000.0, // operations per second
            min_success_rate: 0.95,
        }
    }
}

/// Multi-scale BMD network implementation
#[derive(Debug)]
pub struct MultiscaleBMDNetwork {
    /// Quantum scale BMD layer (10^-15s operations)
    pub quantum_layer: Arc<RwLock<QuantumBMDLayer>>,
    
    /// Molecular scale BMD layer (10^-9s operations)
    pub molecular_layer: Arc<RwLock<MolecularBMDLayer>>,
    
    /// Environmental scale BMD layer (10^2s operations)
    pub environmental_layer: Arc<RwLock<EnvironmentalBMDLayer>>,
    
    /// Cross-scale coordination protocol
    pub coordination_protocol: Arc<RwLock<ScaleCoordinationProtocol>>,
    
    /// Network configuration
    pub config: BMDNetworkConfiguration,
    
    /// Performance metrics
    pub metrics: Arc<RwLock<NetworkMetrics>>,
}

impl MultiscaleBMDNetwork {
    /// Create a new multi-scale BMD network
    pub async fn new(config: BMDNetworkConfiguration) -> Result<Self> {
        let quantum_layer = Arc::new(RwLock::new(
            QuantumBMDLayer::new(config.clone()).await?
        ));
        
        let molecular_layer = Arc::new(RwLock::new(
            MolecularBMDLayer::new(config.clone()).await?
        ));
        
        let environmental_layer = Arc::new(RwLock::new(
            EnvironmentalBMDLayer::new(config.clone()).await?
        ));
        
        let coordination_protocol = Arc::new(RwLock::new(
            ScaleCoordinationProtocol::new(config.clone()).await?
        ));
        
        let metrics = Arc::new(RwLock::new(NetworkMetrics::new()));
        
        Ok(Self {
            quantum_layer,
            molecular_layer,
            environmental_layer,
            coordination_protocol,
            config,
            metrics,
        })
    }
    
    /// Process information through multi-scale BMD network
    pub async fn process_multi_scale(
        &self,
        input: InformationPacket,
    ) -> Result<ProcessedInformation> {
        let start_time = Instant::now();
        
        // Step 1: Determine optimal processing scale
        let optimal_scale = self.determine_optimal_scale(&input).await?;
        
        // Step 2: Process through appropriate scale layers
        let processed = match optimal_scale {
            ProcessingScale::Quantum => {
                self.process_quantum_scale(input).await?
            },
            ProcessingScale::Molecular => {
                self.process_molecular_scale(input).await?
            },
            ProcessingScale::Environmental => {
                self.process_environmental_scale(input).await?
            },
            ProcessingScale::MultiScale => {
                self.process_all_scales(input).await?
            },
        };
        
        // Step 3: Cross-scale coordination and validation
        let coordinated = self.coordinate_across_scales(processed).await?;
        
        // Step 4: Update metrics
        let processing_time = start_time.elapsed();
        self.update_metrics(processing_time, &coordinated).await;
        
        Ok(coordinated)
    }
    
    /// Determine optimal processing scale for input
    async fn determine_optimal_scale(&self, input: &InformationPacket) -> Result<ProcessingScale> {
        let coordination = self.coordination_protocol.read().await;
        coordination.determine_optimal_scale(input).await
    }
    
    /// Process through quantum scale layer
    async fn process_quantum_scale(
        &self,
        input: InformationPacket,
    ) -> Result<ProcessedInformation> {
        let quantum = self.quantum_layer.read().await;
        quantum.process(input).await
    }
    
    /// Process through molecular scale layer
    async fn process_molecular_scale(
        &self,
        input: InformationPacket,
    ) -> Result<ProcessedInformation> {
        let molecular = self.molecular_layer.read().await;
        molecular.process(input).await
    }
    
    /// Process through environmental scale layer
    async fn process_environmental_scale(
        &self,
        input: InformationPacket,
    ) -> Result<ProcessedInformation> {
        let environmental = self.environmental_layer.read().await;
        environmental.process(input).await
    }
    
    /// Process through all scales with coordination
    async fn process_all_scales(
        &self,
        input: InformationPacket,
    ) -> Result<ProcessedInformation> {
        // Parallel processing across all scales
        let (quantum_result, molecular_result, environmental_result) = tokio::try_join!(
            self.process_quantum_scale(input.clone()),
            self.process_molecular_scale(input.clone()),
            self.process_environmental_scale(input.clone())
        )?;
        
        // Coordinate and merge results
        let coordination = self.coordination_protocol.read().await;
        coordination.merge_scale_results(vec![
            quantum_result,
            molecular_result,
            environmental_result,
        ]).await
    }
    
    /// Coordinate results across scales
    async fn coordinate_across_scales(
        &self,
        processed: ProcessedInformation,
    ) -> Result<ProcessedInformation> {
        let coordination = self.coordination_protocol.read().await;
        coordination.coordinate_results(processed).await
    }
    
    /// Update network metrics
    async fn update_metrics(&self, processing_time: Duration, result: &ProcessedInformation) {
        let mut metrics = self.metrics.write().await;
        metrics.update(processing_time, result);
    }
    
    /// Get network status
    pub fn status(&self) -> BMDNetworkStatus {
        BMDNetworkStatus {
            operational: true,
            performance_score: 0.95, // TODO: Calculate from actual metrics
            active_scales: 3,
            total_throughput: 1000.0, // TODO: Get from metrics
        }
    }
}

/// Processing scale determination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingScale {
    Quantum,
    Molecular,
    Environmental,
    MultiScale,
}

/// Quantum scale BMD layer (10^-15s operations)
#[derive(Debug)]
pub struct QuantumBMDLayer {
    /// Quantum BMD units
    pub bmd_units: Vec<QuantumBMDUnit>,
    
    /// Quantum state management
    pub quantum_state_manager: QuantumStateManager,
    
    /// Configuration
    pub config: BMDNetworkConfiguration,
}

impl QuantumBMDLayer {
    pub async fn new(config: BMDNetworkConfiguration) -> Result<Self> {
        let num_units = config.topology.units_per_scale
            .get("quantum")
            .copied()
            .unwrap_or(1000);
        
        let mut bmd_units = Vec::with_capacity(num_units);
        for i in 0..num_units {
            bmd_units.push(QuantumBMDUnit::new(i, &config).await?);
        }
        
        let quantum_state_manager = QuantumStateManager::new(&config).await?;
        
        Ok(Self {
            bmd_units,
            quantum_state_manager,
            config,
        })
    }
    
    pub async fn process(&self, input: InformationPacket) -> Result<ProcessedInformation> {
        // Implement quantum-scale processing
        // This operates at 10^-15s timescales with quantum coherence
        
        let start_time = Instant::now();
        
        // Step 1: Quantum state preparation
        let quantum_state = self.quantum_state_manager.prepare_state(&input).await?;
        
        // Step 2: Distribute across quantum BMD units
        let unit_index = self.select_optimal_unit(&input).await?;
        let result = self.bmd_units[unit_index].process(quantum_state).await?;
        
        // Step 3: Quantum measurement and decoherence management
        let measured_result = self.quantum_state_manager.measure_result(result).await?;
        
        Ok(ProcessedInformation {
            processed_data: measured_result.data,
            amplification_achieved: measured_result.amplification,
            quality_metrics: measured_result.quality,
            processing_time: start_time.elapsed(),
            metadata: measured_result.metadata,
        })
    }
    
    async fn select_optimal_unit(&self, _input: &InformationPacket) -> Result<usize> {
        // TODO: Implement intelligent unit selection
        Ok(0)
    }
}

/// Molecular scale BMD layer (10^-9s operations)
#[derive(Debug)]
pub struct MolecularBMDLayer {
    /// Molecular BMD units
    pub bmd_units: Vec<MolecularBMDUnit>,
    
    /// Molecular interaction manager
    pub interaction_manager: MolecularInteractionManager,
    
    /// Configuration
    pub config: BMDNetworkConfiguration,
}

impl MolecularBMDLayer {
    pub async fn new(config: BMDNetworkConfiguration) -> Result<Self> {
        let num_units = config.topology.units_per_scale
            .get("molecular")
            .copied()
            .unwrap_or(100);
        
        let mut bmd_units = Vec::with_capacity(num_units);
        for i in 0..num_units {
            bmd_units.push(MolecularBMDUnit::new(i, &config).await?);
        }
        
        let interaction_manager = MolecularInteractionManager::new(&config).await?;
        
        Ok(Self {
            bmd_units,
            interaction_manager,
            config,
        })
    }
    
    pub async fn process(&self, input: InformationPacket) -> Result<ProcessedInformation> {
        // Implement molecular-scale processing
        // This operates at 10^-9s timescales with molecular interactions
        
        let start_time = Instant::now();
        
        // Step 1: Molecular configuration preparation
        let molecular_config = self.interaction_manager.prepare_configuration(&input).await?;
        
        // Step 2: Distribute across molecular BMD units
        let unit_index = self.select_optimal_unit(&input).await?;
        let result = self.bmd_units[unit_index].process(molecular_config).await?;
        
        // Step 3: Molecular interaction analysis
        let analyzed_result = self.interaction_manager.analyze_result(result).await?;
        
        Ok(ProcessedInformation {
            processed_data: analyzed_result.data,
            amplification_achieved: analyzed_result.amplification,
            quality_metrics: analyzed_result.quality,
            processing_time: start_time.elapsed(),
            metadata: analyzed_result.metadata,
        })
    }
    
    async fn select_optimal_unit(&self, _input: &InformationPacket) -> Result<usize> {
        // TODO: Implement intelligent unit selection
        Ok(0)
    }
}

/// Environmental scale BMD layer (10^2s operations)
#[derive(Debug)]
pub struct EnvironmentalBMDLayer {
    /// Environmental BMD units
    pub bmd_units: Vec<EnvironmentalBMDUnit>,
    
    /// Environmental context manager
    pub context_manager: EnvironmentalContextManager,
    
    /// Configuration
    pub config: BMDNetworkConfiguration,
}

impl EnvironmentalBMDLayer {
    pub async fn new(config: BMDNetworkConfiguration) -> Result<Self> {
        let num_units = config.topology.units_per_scale
            .get("environmental")
            .copied()
            .unwrap_or(10);
        
        let mut bmd_units = Vec::with_capacity(num_units);
        for i in 0..num_units {
            bmd_units.push(EnvironmentalBMDUnit::new(i, &config).await?);
        }
        
        let context_manager = EnvironmentalContextManager::new(&config).await?;
        
        Ok(Self {
            bmd_units,
            context_manager,
            config,
        })
    }
    
    pub async fn process(&self, input: InformationPacket) -> Result<ProcessedInformation> {
        // Implement environmental-scale processing
        // This operates at 10^2s timescales with environmental context
        
        let start_time = Instant::now();
        
        // Step 1: Environmental context preparation
        let env_context = self.context_manager.prepare_context(&input).await?;
        
        // Step 2: Distribute across environmental BMD units
        let unit_index = self.select_optimal_unit(&input).await?;
        let result = self.bmd_units[unit_index].process(env_context).await?;
        
        // Step 3: Environmental impact analysis
        let analyzed_result = self.context_manager.analyze_result(result).await?;
        
        Ok(ProcessedInformation {
            processed_data: analyzed_result.data,
            amplification_achieved: analyzed_result.amplification,
            quality_metrics: analyzed_result.quality,
            processing_time: start_time.elapsed(),
            metadata: analyzed_result.metadata,
        })
    }
    
    async fn select_optimal_unit(&self, _input: &InformationPacket) -> Result<usize> {
        // TODO: Implement intelligent unit selection
        Ok(0)
    }
}

/// Cross-scale coordination protocol
#[derive(Debug)]
pub struct ScaleCoordinationProtocol {
    /// Coordination algorithms
    pub coordination_algorithms: CoordinationAlgorithms,
    
    /// Scale interaction models
    pub interaction_models: ScaleInteractionModels,
    
    /// Configuration
    pub config: BMDNetworkConfiguration,
}

impl ScaleCoordinationProtocol {
    pub async fn new(config: BMDNetworkConfiguration) -> Result<Self> {
        let coordination_algorithms = CoordinationAlgorithms::new(&config).await?;
        let interaction_models = ScaleInteractionModels::new(&config).await?;
        
        Ok(Self {
            coordination_algorithms,
            interaction_models,
            config,
        })
    }
    
    pub async fn determine_optimal_scale(
        &self,
        input: &InformationPacket,
    ) -> Result<ProcessingScale> {
        // Analyze input characteristics to determine optimal processing scale
        match input.metadata.information_type {
            crate::core::InformationType::QuantumState => Ok(ProcessingScale::Quantum),
            crate::core::InformationType::MolecularStructure => Ok(ProcessingScale::Molecular),
            crate::core::InformationType::EnvironmentalData => Ok(ProcessingScale::Environmental),
            crate::core::InformationType::TemporalCoordinate => Ok(ProcessingScale::MultiScale),
            _ => Ok(ProcessingScale::MultiScale),
        }
    }
    
    pub async fn merge_scale_results(
        &self,
        results: Vec<ProcessedInformation>,
    ) -> Result<ProcessedInformation> {
        // Merge results from multiple scales
        self.coordination_algorithms.merge_results(results).await
    }
    
    pub async fn coordinate_results(
        &self,
        processed: ProcessedInformation,
    ) -> Result<ProcessedInformation> {
        // Apply cross-scale coordination
        self.coordination_algorithms.coordinate(processed).await
    }
}

// Placeholder implementations for supporting types
// These would be fully implemented in separate modules

#[derive(Debug)]
pub struct QuantumBMDUnit {
    pub id: usize,
}

impl QuantumBMDUnit {
    pub async fn new(id: usize, _config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self { id })
    }
    
    pub async fn process(&self, _state: QuantumState) -> Result<QuantumResult> {
        Ok(QuantumResult::default())
    }
}

#[derive(Debug)]
pub struct MolecularBMDUnit {
    pub id: usize,
}

impl MolecularBMDUnit {
    pub async fn new(id: usize, _config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self { id })
    }
    
    pub async fn process(&self, _config: MolecularConfiguration) -> Result<MolecularResult> {
        Ok(MolecularResult::default())
    }
}

#[derive(Debug)]
pub struct EnvironmentalBMDUnit {
    pub id: usize,
}

impl EnvironmentalBMDUnit {
    pub async fn new(id: usize, _config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self { id })
    }
    
    pub async fn process(&self, _context: EnvironmentalContext) -> Result<EnvironmentalResult> {
        Ok(EnvironmentalResult::default())
    }
}

// Supporting type placeholders
#[derive(Debug)]
pub struct QuantumStateManager;
#[derive(Debug)]
pub struct MolecularInteractionManager;
#[derive(Debug)]
pub struct EnvironmentalContextManager;
#[derive(Debug)]
pub struct CoordinationAlgorithms;
#[derive(Debug)]
pub struct ScaleInteractionModels;

#[derive(Debug, Default)]
pub struct QuantumState;
#[derive(Debug, Default)]
pub struct MolecularConfiguration;
#[derive(Debug, Default)]
pub struct EnvironmentalContext;
#[derive(Debug, Default)]
pub struct QuantumResult {
    pub data: Vec<u8>,
    pub amplification: f64,
    pub quality: crate::core::QualityMetrics,
    pub metadata: crate::core::ProcessedMetadata,
}
#[derive(Debug, Default)]
pub struct MolecularResult {
    pub data: Vec<u8>,
    pub amplification: f64,
    pub quality: crate::core::QualityMetrics,
    pub metadata: crate::core::ProcessedMetadata,
}
#[derive(Debug, Default)]
pub struct EnvironmentalResult {
    pub data: Vec<u8>,
    pub amplification: f64,
    pub quality: crate::core::QualityMetrics,
    pub metadata: crate::core::ProcessedMetadata,
}

impl QuantumStateManager {
    pub async fn new(_config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self)
    }
    
    pub async fn prepare_state(&self, _input: &InformationPacket) -> Result<QuantumState> {
        Ok(QuantumState::default())
    }
    
    pub async fn measure_result(&self, result: QuantumResult) -> Result<QuantumResult> {
        Ok(result)
    }
}

impl MolecularInteractionManager {
    pub async fn new(_config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self)
    }
    
    pub async fn prepare_configuration(&self, _input: &InformationPacket) -> Result<MolecularConfiguration> {
        Ok(MolecularConfiguration::default())
    }
    
    pub async fn analyze_result(&self, result: MolecularResult) -> Result<MolecularResult> {
        Ok(result)
    }
}

impl EnvironmentalContextManager {
    pub async fn new(_config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self)
    }
    
    pub async fn prepare_context(&self, _input: &InformationPacket) -> Result<EnvironmentalContext> {
        Ok(EnvironmentalContext::default())
    }
    
    pub async fn analyze_result(&self, result: EnvironmentalResult) -> Result<EnvironmentalResult> {
        Ok(result)
    }
}

impl CoordinationAlgorithms {
    pub async fn new(_config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self)
    }
    
    pub async fn merge_results(&self, results: Vec<ProcessedInformation>) -> Result<ProcessedInformation> {
        // Simple merge implementation - combine first result
        Ok(results.into_iter().next().unwrap_or_default())
    }
    
    pub async fn coordinate(&self, processed: ProcessedInformation) -> Result<ProcessedInformation> {
        Ok(processed)
    }
}

impl ScaleInteractionModels {
    pub async fn new(_config: &BMDNetworkConfiguration) -> Result<Self> {
        Ok(Self)
    }
}

impl Default for ProcessedInformation {
    fn default() -> Self {
        Self {
            processed_data: Vec::new(),
            amplification_achieved: 1.0,
            quality_metrics: crate::core::QualityMetrics::default(),
            processing_time: Duration::from_millis(0),
            metadata: crate::core::ProcessedMetadata::default(),
        }
    }
}

impl Default for crate::core::QualityMetrics {
    fn default() -> Self {
        Self {
            fidelity: 1.0,
            signal_to_noise_ratio: 1.0,
            coherence_preservation: 1.0,
            information_preservation: 1.0,
        }
    }
}

impl Default for crate::core::ProcessedMetadata {
    fn default() -> Self {
        Self {
            processing_path: Vec::new(),
            resource_usage: crate::core::ResourceUsage::default(),
            performance_metrics: crate::core::PerformanceMetrics::default(),
        }
    }
}

impl Default for crate::core::ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_time: Duration::from_millis(0),
            memory_peak: 0,
            quantum_resources: 0.0,
            energy_consumption: 0.0,
        }
    }
}

impl Default for crate::core::PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_millis(0),
            efficiency: 1.0,
            success_rate: 1.0,
        }
    }
}

/// Network performance metrics
#[derive(Debug)]
pub struct NetworkMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub average_processing_time: Duration,
    pub average_amplification: f64,
    pub last_updated: Instant,
}

impl NetworkMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            average_processing_time: Duration::from_millis(0),
            average_amplification: 1.0,
            last_updated: Instant::now(),
        }
    }
    
    pub fn update(&mut self, processing_time: Duration, result: &ProcessedInformation) {
        self.total_operations += 1;
        self.successful_operations += 1; // TODO: Check for errors
        
        // Update running averages
        let alpha = 0.1; // Exponential moving average factor
        let current_time_ms = processing_time.as_millis() as f64;
        let avg_time_ms = self.average_processing_time.as_millis() as f64;
        let new_avg_time_ms = alpha * current_time_ms + (1.0 - alpha) * avg_time_ms;
        self.average_processing_time = Duration::from_millis(new_avg_time_ms as u64);
        
        self.average_amplification = alpha * result.amplification_achieved 
            + (1.0 - alpha) * self.average_amplification;
        
        self.last_updated = Instant::now();
    }
}

/// BMD network status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDNetworkStatus {
    pub operational: bool,
    pub performance_score: f64,
    pub active_scales: usize,
    pub total_throughput: f64,
} 