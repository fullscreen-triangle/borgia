"""
Borgia Test Framework - Configuration Module
===========================================

Configuration management for the Borgia biological Maxwell demons (BMD)
cheminformatics test/validation framework.

This module provides configuration classes and utilities for:
- Test execution parameters
- Molecular generation settings  
- BMD network configurations
- Hardware integration parameters
- Performance benchmark settings
- Visualization configuration
- Output and export settings

Author: Borgia Development Team
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestConfig:
    """
    Base test configuration class with common settings.
    
    Attributes:
        test_name: Name of the test configuration
        enabled: Whether this test is enabled
        timeout_seconds: Maximum execution time in seconds
        retry_attempts: Number of retry attempts on failure
        parallel_execution: Whether to run in parallel
        log_level: Logging level for this test
    """
    test_name: str = "default_test"
    enabled: bool = True
    timeout_seconds: float = 300.0  # 5 minutes default
    retry_attempts: int = 3
    parallel_execution: bool = True
    log_level: str = "INFO"
    

@dataclass
class MolecularGenerationConfig(TestConfig):
    """
    Configuration for molecular generation tests.
    
    Attributes:
        molecule_count: Number of molecules to generate
        precision_targets: List of precision targets in seconds
        processing_capacities: List of processing capacity targets (ops/sec)
        dual_functionality_required: Whether dual functionality is mandatory
        quality_control_tolerance: Quality control tolerance (0.0 = zero tolerance)
        generation_algorithms: List of generation algorithms to test
        validation_strictness: Validation strictness level (0.0-1.0)
        export_molecules: Whether to export generated molecules
        cache_results: Whether to cache generation results
    """
    test_name: str = "molecular_generation"
    molecule_count: int = 1000
    precision_targets: List[float] = field(default_factory=lambda: [1e-30, 1e-35, 1e-40])
    processing_capacities: List[float] = field(default_factory=lambda: [1e6, 1e7, 1e8])
    dual_functionality_required: bool = True
    quality_control_tolerance: float = 0.0  # Zero tolerance
    generation_algorithms: List[str] = field(default_factory=lambda: ["bmde", "quantum_coherent", "thermal_equilibrium"])
    validation_strictness: float = 1.0  # Maximum strictness
    export_molecules: bool = True
    cache_results: bool = True


@dataclass
class BMDNetworkConfig(TestConfig):
    """
    Configuration for BMD network tests.
    
    Attributes:
        quantum_timescale: Quantum layer timescale in seconds
        molecular_timescale: Molecular layer timescale in seconds  
        environmental_timescale: Environmental layer timescale in seconds
        amplification_target: Target amplification factor
        efficiency_target: Target efficiency (0.0-1.0)
        coherence_time_target: Target coherence time in seconds
        network_topologies: List of network topologies to test
        synchronization_tolerance: Synchronization tolerance
        noise_levels: List of noise levels to test
        temperature_ranges: Temperature ranges for testing (Kelvin)
    """
    test_name: str = "bmd_networks"
    quantum_timescale: float = 1e-15  # femtoseconds
    molecular_timescale: float = 1e-9   # nanoseconds
    environmental_timescale: float = 100.0  # 100 seconds
    amplification_target: float = 1000.0
    efficiency_target: float = 0.95
    coherence_time_target: float = 247e-6  # microseconds
    network_topologies: List[str] = field(default_factory=lambda: ["hierarchical", "mesh", "hybrid"])
    synchronization_tolerance: float = 1e-12
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    temperature_ranges: List[List[float]] = field(default_factory=lambda: [[77, 300], [300, 400]])


@dataclass  
class HardwareIntegrationConfig(TestConfig):
    """
    Configuration for hardware integration tests.
    
    Attributes:
        led_wavelengths: LED wavelengths for spectroscopy (nm)
        led_power_levels: LED power levels to test (mW)
        cpu_timing_enabled: Whether to test CPU timing integration
        noise_enhancement_enabled: Whether to test noise enhancement
        zero_cost_validation: Whether to validate zero-cost operation
        performance_improvement_target: Target performance improvement factor
        memory_reduction_target: Target memory reduction factor
        supported_hardware: List of supported hardware configurations
        calibration_required: Whether hardware calibration is required
        safety_checks: Whether to perform safety checks
    """
    test_name: str = "hardware_integration"
    led_wavelengths: List[int] = field(default_factory=lambda: [470, 525, 625])  # Blue, Green, Red
    led_power_levels: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])
    cpu_timing_enabled: bool = True
    noise_enhancement_enabled: bool = True
    zero_cost_validation: bool = True
    performance_improvement_target: float = 3.0
    memory_reduction_target: float = 150.0
    supported_hardware: List[str] = field(default_factory=lambda: ["standard_led", "rgb_arrays", "spectrometers"])
    calibration_required: bool = False
    safety_checks: bool = True


@dataclass
class PerformanceBenchmarkConfig(TestConfig):
    """
    Configuration for performance benchmark tests.
    
    Attributes:
        duration_seconds: Benchmark duration in seconds
        warmup_iterations: Number of warmup iterations
        benchmark_iterations: Number of benchmark iterations
        memory_limit_gb: Memory limit in gigabytes
        cpu_cores: Number of CPU cores to use (-1 for all)
        benchmark_types: List of benchmark types to run
        profiling_enabled: Whether to enable detailed profiling
        resource_monitoring: Whether to monitor system resources
        statistical_analysis: Whether to perform statistical analysis
        comparison_baselines: Baseline performance metrics for comparison
    """
    test_name: str = "performance_benchmarks"
    duration_seconds: float = 300.0
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    memory_limit_gb: float = 32.0
    cpu_cores: int = -1  # Use all available
    benchmark_types: List[str] = field(default_factory=lambda: [
        "molecular_generation", "bmd_coordination", "hardware_integration", 
        "information_catalysis", "system_scalability"
    ])
    profiling_enabled: bool = True
    resource_monitoring: bool = True
    statistical_analysis: bool = True
    comparison_baselines: Dict[str, float] = field(default_factory=dict)


@dataclass
class DownstreamIntegrationConfig(TestConfig):
    """
    Configuration for downstream system integration tests.
    
    Attributes:
        masunda_temporal_config: Configuration for Masunda Temporal integration
        buhera_foundry_config: Configuration for Buhera Foundry integration
        kambuzuma_config: Configuration for Kambuzuma integration
        integration_timeout: Timeout for integration tests
        data_transfer_validation: Whether to validate data transfer
        security_validation: Whether to validate security protocols
        compatibility_checks: Whether to perform compatibility checks
        failover_testing: Whether to test failover mechanisms
    """
    test_name: str = "downstream_integration"
    masunda_temporal_config: Dict[str, Any] = field(default_factory=lambda: {
        "precision_requirement": 1e-18,  # attoseconds
        "oscillation_frequency": 9192631770,  # Cesium standard
        "synchronization_protocol": "ntp_enhanced",
        "validation_duration": 3600  # 1 hour
    })
    buhera_foundry_config: Dict[str, Any] = field(default_factory=lambda: {
        "substrate_quality_requirement": 0.999999,  # 6 nines
        "manufacturing_precision": 1e-9,  # nanometer scale
        "batch_size_limits": [100, 1000, 10000],
        "quality_assurance_protocols": ["spectroscopy", "quantum_validation", "thermal_analysis"]
    })
    kambuzuma_config: Dict[str, Any] = field(default_factory=lambda: {
        "quantum_coherence_requirement": 247e-6,  # microseconds
        "consciousness_integration_level": 0.95,
        "biological_quantum_protocols": ["membrane_oscillation", "protein_folding", "dna_resonance"],
        "safety_validation": True
    })
    integration_timeout: float = 1800.0  # 30 minutes
    data_transfer_validation: bool = True
    security_validation: bool = True
    compatibility_checks: bool = True
    failover_testing: bool = True


@dataclass
class QualityControlConfig(TestConfig):
    """
    Configuration for quality control validation tests.
    
    Attributes:
        zero_tolerance_enabled: Whether zero tolerance is enforced
        defect_detection_sensitivity: Sensitivity for defect detection (0.0-1.0)
        false_positive_threshold: Maximum acceptable false positive rate
        false_negative_threshold: Maximum acceptable false negative rate (should be 0.0)
        validation_algorithms: List of validation algorithms to test
        statistical_confidence: Statistical confidence level required
        batch_validation_size: Size of validation batches
        continuous_monitoring: Whether to enable continuous monitoring
        alert_thresholds: Thresholds for quality alerts
    """
    test_name: str = "quality_control"
    zero_tolerance_enabled: bool = True
    defect_detection_sensitivity: float = 1.0  # Maximum sensitivity
    false_positive_threshold: float = 0.01  # 1% max
    false_negative_threshold: float = 0.0   # Zero tolerance
    validation_algorithms: List[str] = field(default_factory=lambda: [
        "spectroscopic_analysis", "quantum_coherence_check", "thermal_stability", "information_integrity"
    ])
    statistical_confidence: float = 0.99999  # 5 nines confidence
    batch_validation_size: int = 1000
    continuous_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "defect_rate": 0.0,
        "processing_speed": 0.95,  # Minimum relative performance
        "memory_efficiency": 0.9   # Minimum efficiency
    })


@dataclass
class VisualizationConfig:
    """
    Configuration for visualization and reporting.
    
    Attributes:
        generate_plots: Whether to generate plots
        interactive_dashboards: Whether to create interactive dashboards
        export_formats: List of export formats
        plot_style: Matplotlib style to use
        figure_size: Default figure size [width, height]
        dpi: Resolution in dots per inch
        color_scheme: Color scheme for plots
        animation_enabled: Whether to generate animations
        real_time_updates: Whether to enable real-time updates
        custom_themes: Custom visualization themes
    """
    generate_plots: bool = True
    interactive_dashboards: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['png', 'svg', 'html', 'pdf'])
    plot_style: str = 'seaborn-v0_8'
    figure_size: List[float] = field(default_factory=lambda: [12, 8])
    dpi: int = 300
    color_scheme: str = 'viridis'
    animation_enabled: bool = False
    real_time_updates: bool = True
    custom_themes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """
    Configuration for output and export settings.
    
    Attributes:
        base_output_dir: Base directory for outputs
        export_json: Whether to export JSON results
        export_csv: Whether to export CSV summaries
        export_hdf5: Whether to export HDF5 detailed data
        export_pickle: Whether to export pickle files
        generate_report: Whether to generate comprehensive reports
        timestamp_results: Whether to add timestamps to result directories
        compression_enabled: Whether to enable output compression
        backup_results: Whether to create backup copies
        retention_policy: Data retention policy in days
    """
    base_output_dir: str = 'borgia_results'
    export_json: bool = True
    export_csv: bool = True
    export_hdf5: bool = True
    export_pickle: bool = False
    generate_report: bool = True
    timestamp_results: bool = True
    compression_enabled: bool = False
    backup_results: bool = True
    retention_policy: int = 30  # 30 days


class ConfigurationManager:
    """
    Manager for loading, validating, and managing configurations.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.configs = {}
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default configurations for all components."""
        self.configs = {
            'molecular_generation': MolecularGenerationConfig(),
            'bmd_networks': BMDNetworkConfig(),
            'hardware_integration': HardwareIntegrationConfig(),
            'performance_benchmarks': PerformanceBenchmarkConfig(),
            'downstream_integration': DownstreamIntegrationConfig(),
            'quality_control': QualityControlConfig(),
            'visualization': VisualizationConfig(),
            'output': OutputConfig()
        }
    
    def get_config(self, config_name: str) -> Union[TestConfig, VisualizationConfig, OutputConfig]:
        """
        Get configuration by name.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration object
            
        Raises:
            KeyError: If configuration name not found
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' not found. Available: {list(self.configs.keys())}")
        
        return self.configs[config_name]
    
    def update_config(self, config_name: str, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            config_name: Name of the configuration to update
            **kwargs: Configuration parameters to update
        """
        if config_name not in self.configs:
            raise KeyError(f"Configuration '{config_name}' not found")
        
        config = self.configs[config_name]
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise AttributeError(f"Configuration '{config_name}' has no attribute '{key}'")
    
    def validate_configuration(self, config_name: str) -> List[str]:
        """
        Validate configuration parameters.
        
        Args:
            config_name: Name of the configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        config = self.get_config(config_name)
        errors = []
        
        # Validate common TestConfig parameters
        if isinstance(config, TestConfig):
            if config.timeout_seconds <= 0:
                errors.append(f"{config_name}: timeout_seconds must be positive")
            
            if config.retry_attempts < 0:
                errors.append(f"{config_name}: retry_attempts must be non-negative")
        
        # Validate specific configurations
        if isinstance(config, MolecularGenerationConfig):
            if config.molecule_count <= 0:
                errors.append("molecular_generation: molecule_count must be positive")
            
            if not (0.0 <= config.quality_control_tolerance <= 1.0):
                errors.append("molecular_generation: quality_control_tolerance must be between 0.0 and 1.0")
            
            if not config.precision_targets:
                errors.append("molecular_generation: precision_targets cannot be empty")
        
        elif isinstance(config, BMDNetworkConfig):
            if config.amplification_target <= 1.0:
                errors.append("bmd_networks: amplification_target must be > 1.0")
            
            if not (0.0 <= config.efficiency_target <= 1.0):
                errors.append("bmd_networks: efficiency_target must be between 0.0 and 1.0")
            
            timescales = [config.quantum_timescale, config.molecular_timescale, config.environmental_timescale]
            if not all(t > 0 for t in timescales):
                errors.append("bmd_networks: all timescales must be positive")
        
        elif isinstance(config, HardwareIntegrationConfig):
            if config.performance_improvement_target <= 1.0:
                errors.append("hardware_integration: performance_improvement_target must be > 1.0")
            
            if not config.led_wavelengths:
                errors.append("hardware_integration: led_wavelengths cannot be empty")
            
            for wavelength in config.led_wavelengths:
                if not (300 <= wavelength <= 1000):  # Visible to near-IR range
                    errors.append(f"hardware_integration: wavelength {wavelength}nm outside valid range (300-1000nm)")
        
        elif isinstance(config, PerformanceBenchmarkConfig):
            if config.duration_seconds <= 0:
                errors.append("performance_benchmarks: duration_seconds must be positive")
            
            if config.benchmark_iterations <= 0:
                errors.append("performance_benchmarks: benchmark_iterations must be positive")
            
            if config.memory_limit_gb <= 0:
                errors.append("performance_benchmarks: memory_limit_gb must be positive")
        
        elif isinstance(config, QualityControlConfig):
            if config.false_negative_threshold > 0.0 and config.zero_tolerance_enabled:
                errors.append("quality_control: false_negative_threshold must be 0.0 for zero tolerance")
            
            if not (0.0 <= config.defect_detection_sensitivity <= 1.0):
                errors.append("quality_control: defect_detection_sensitivity must be between 0.0 and 1.0")
        
        return errors
    
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """
        Validate all configurations.
        
        Returns:
            Dictionary mapping configuration names to validation errors
        """
        all_errors = {}
        
        for config_name in self.configs.keys():
            errors = self.validate_configuration(config_name)
            if errors:
                all_errors[config_name] = errors
        
        return all_errors
    
    def load_from_file(self, config_path: Union[str, Path]):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        import json
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update configurations with loaded data
        for config_name, params in config_data.items():
            if config_name in self.configs:
                for key, value in params.items():
                    if hasattr(self.configs[config_name], key):
                        setattr(self.configs[config_name], key, value)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """
        Save current configuration to JSON file.
        
        Args:
            config_path: Path to save configuration file
        """
        import json
        from dataclasses import asdict
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert all configurations to dictionaries
        config_data = {}
        for name, config in self.configs.items():
            config_data[name] = asdict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def get_summary(self) -> str:
        """
        Get a summary of current configuration.
        
        Returns:
            Configuration summary string
        """
        summary_lines = ["Borgia Test Framework Configuration Summary", "=" * 50]
        
        for name, config in self.configs.items():
            summary_lines.append(f"\n{name.upper()}:")
            
            if isinstance(config, TestConfig):
                summary_lines.append(f"  • Enabled: {config.enabled}")
                summary_lines.append(f"  • Timeout: {config.timeout_seconds}s")
                summary_lines.append(f"  • Parallel: {config.parallel_execution}")
                
                # Add specific configuration details
                if isinstance(config, MolecularGenerationConfig):
                    summary_lines.append(f"  • Molecules: {config.molecule_count}")
                    summary_lines.append(f"  • Zero Tolerance: {config.quality_control_tolerance == 0.0}")
                
                elif isinstance(config, BMDNetworkConfig):
                    summary_lines.append(f"  • Amplification Target: {config.amplification_target}×")
                    summary_lines.append(f"  • Efficiency Target: {config.efficiency_target:.1%}")
                
                elif isinstance(config, HardwareIntegrationConfig):
                    summary_lines.append(f"  • LED Wavelengths: {config.led_wavelengths}")
                    summary_lines.append(f"  • Zero Cost: {config.zero_cost_validation}")
                
                elif isinstance(config, PerformanceBenchmarkConfig):
                    summary_lines.append(f"  • Duration: {config.duration_seconds}s")
                    summary_lines.append(f"  • Iterations: {config.benchmark_iterations}")
            
            elif isinstance(config, VisualizationConfig):
                summary_lines.append(f"  • Generate Plots: {config.generate_plots}")
                summary_lines.append(f"  • Interactive: {config.interactive_dashboards}")
                summary_lines.append(f"  • Formats: {config.export_formats}")
            
            elif isinstance(config, OutputConfig):
                summary_lines.append(f"  • Output Dir: {config.base_output_dir}")
                summary_lines.append(f"  • Export JSON: {config.export_json}")
                summary_lines.append(f"  • Generate Report: {config.generate_report}")
        
        return "\n".join(summary_lines)


# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for direct access
def get_config(config_name: str):
    """Get configuration by name using global manager."""
    return config_manager.get_config(config_name)

def update_config(config_name: str, **kwargs):
    """Update configuration using global manager."""
    return config_manager.update_config(config_name, **kwargs)

def validate_all_configs():
    """Validate all configurations using global manager."""
    return config_manager.validate_all_configurations()
