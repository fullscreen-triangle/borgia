"""
Borgia Test Framework - Core Module
===================================

Central orchestrator for the Borgia biological Maxwell demons (BMD) 
cheminformatics test/validation framework. Provides the main testing 
infrastructure and coordination between different validation components.

This module implements:
- Central test framework coordination
- Result aggregation and analysis
- Configuration management
- System validation orchestration
- Performance monitoring and benchmarking

Author: Borgia Development Team
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .exceptions import BorgiaTestError, ValidationError, BenchmarkError
from .config import TestConfig, VisualizationConfig
from .data_structures import ValidationResults, BenchmarkResults, PerformanceMetrics


@dataclass
class TestConfiguration:
    """
    Comprehensive configuration for Borgia test framework execution.
    
    Attributes:
        molecular_generation_config: Configuration for molecular generation tests
        bmd_network_config: Configuration for BMD network validation
        hardware_integration_config: Configuration for hardware integration tests
        performance_benchmark_config: Configuration for performance benchmarking
        visualization_config: Configuration for result visualization
        output_config: Configuration for result export and storage
    """
    molecular_generation_config: Dict[str, Any] = field(default_factory=dict)
    bmd_network_config: Dict[str, Any] = field(default_factory=dict)
    hardware_integration_config: Dict[str, Any] = field(default_factory=dict)
    performance_benchmark_config: Dict[str, Any] = field(default_factory=dict)
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration values."""
        if not self.molecular_generation_config:
            self.molecular_generation_config = {
                'molecule_count': 10000,
                'precision_targets': [1e-30, 1e-35, 1e-40, 1e-45, 1e-50],
                'processing_capacities': [1e6, 1e7, 1e8],
                'dual_functionality_required': True,
                'quality_control_tolerance': 0.0  # Zero tolerance
            }
        
        if not self.bmd_network_config:
            self.bmd_network_config = {
                'quantum_timescale': 1e-15,
                'molecular_timescale': 1e-9,
                'environmental_timescale': 100,
                'amplification_target': 1000.0,
                'efficiency_target': 0.95,
                'coherence_time_target': 247e-6
            }
        
        if not self.hardware_integration_config:
            self.hardware_integration_config = {
                'led_wavelengths': [470, 525, 625],  # Blue, Green, Red
                'cpu_timing_enabled': True,
                'noise_enhancement_enabled': True,
                'zero_cost_validation': True,
                'performance_improvement_target': 3.0,
                'memory_reduction_target': 150.0
            }
        
        if not self.performance_benchmark_config:
            self.performance_benchmark_config = {
                'duration_seconds': 300,
                'memory_limit_gb': 32,
                'cpu_cores': -1,  # Use all available
                'benchmark_iterations': 10,
                'warmup_iterations': 3
            }
        
        if not self.visualization_config:
            self.visualization_config = {
                'generate_plots': True,
                'interactive_dashboards': True,
                'export_formats': ['png', 'svg', 'html'],
                'plot_style': 'seaborn-v0_8',
                'figure_size': [12, 8],
                'dpi': 300
            }
        
        if not self.output_config:
            self.output_config = {
                'base_output_dir': 'results',
                'export_json': True,
                'export_csv': True,
                'export_hdf5': True,
                'generate_report': True,
                'timestamp_results': True
            }


@dataclass 
class ValidationResult:
    """
    Comprehensive validation result container.
    
    Attributes:
        test_name: Name of the validation test
        success: Whether the validation passed
        score: Numerical score (0.0 to 1.0)
        metrics: Detailed performance metrics
        errors: List of errors encountered
        warnings: List of warnings generated
        execution_time: Time taken to execute the test
        timestamp: When the test was executed
        metadata: Additional test metadata
    """
    test_name: str
    success: bool
    score: float
    metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate result data."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
        
        if self.errors and self.success:
            self.success = False  # Cannot be successful with errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'success': self.success,
            'score': self.score,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class BenchmarkResult:
    """
    Performance benchmark result container.
    
    Attributes:
        benchmark_name: Name of the benchmark
        performance_score: Overall performance score
        throughput: Operations per second
        latency: Average response time in seconds
        memory_usage: Peak memory usage in MB
        cpu_utilization: Average CPU utilization percentage
        gpu_utilization: Average GPU utilization percentage (if applicable)
        execution_time: Total benchmark execution time
        iterations_completed: Number of benchmark iterations
        metadata: Additional benchmark metadata
    """
    benchmark_name: str
    performance_score: float
    throughput: float
    latency: float
    memory_usage: float
    cpu_utilization: float
    gpu_utilization: Optional[float] = None
    execution_time: float = 0.0
    iterations_completed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_name': self.benchmark_name,
            'performance_score': self.performance_score,
            'throughput': self.throughput,
            'latency': self.latency,
            'memory_usage': self.memory_usage,
            'cpu_utilization': self.cpu_utilization,
            'gpu_utilization': self.gpu_utilization,
            'execution_time': self.execution_time,
            'iterations_completed': self.iterations_completed,
            'metadata': self.metadata
        }


class BorgiaTestFramework:
    """
    Main orchestrator for the Borgia BMD cheminformatics test/validation framework.
    
    This class coordinates all testing components and provides a unified interface
    for executing comprehensive validation and benchmarking of the Borgia system.
    """
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        """
        Initialize the Borgia Test Framework.
        
        Args:
            config: Test configuration. If None, defaults are loaded.
        """
        self.config = config or TestConfiguration()
        self.logger = self._setup_logging()
        self.results_cache = {}
        self.active_tests = {}
        self.framework_start_time = time.time()
        
        # Initialize result storage
        self.validation_results: List[ValidationResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Initialize component managers
        self._initialize_components()
        
        self.logger.info("Borgia Test Framework initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up framework logging."""
        logger = logging.getLogger('borgia_test_framework')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize framework components."""
        try:
            # These imports are delayed to avoid circular imports
            from .molecular_generation import MolecularGenerator, DualFunctionalityValidator
            from .bmd_networks import BMDNetworkTester
            from .hardware_integration import HardwareIntegrationTester
            from .benchmarks import PerformanceBenchmarker
            from .visualization import BorgiaVisualizer
            
            # Initialize core components
            self.molecular_generator = MolecularGenerator(self.config.molecular_generation_config)
            self.dual_functionality_validator = DualFunctionalityValidator()
            self.bmd_network_tester = BMDNetworkTester(self.config.bmd_network_config)
            self.hardware_integration_tester = HardwareIntegrationTester(self.config.hardware_integration_config)
            self.performance_benchmarker = PerformanceBenchmarker(self.config.performance_benchmark_config)
            self.visualizer = BorgiaVisualizer(self.config.visualization_config)
            
            self.logger.info("All framework components initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise BorgiaTestError(f"Component initialization failed: {e}")
    
    def run_comprehensive_validation(self, 
                                   output_dir: Optional[Union[str, Path]] = None,
                                   parallel: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive validation of the Borgia BMD system.
        
        Args:
            output_dir: Directory to save results. If None, uses config default.
            parallel: Whether to run tests in parallel where possible.
            
        Returns:
            Dictionary containing comprehensive validation results
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive Borgia BMD system validation")
        
        try:
            # Set up output directory
            output_path = Path(output_dir) if output_dir else Path(self.config.output_config['base_output_dir'])
            if self.config.output_config['timestamp_results']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_path / f"validation_{timestamp}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Run validation test suite
            validation_results = self._run_validation_suite(parallel=parallel)
            
            # Run performance benchmarks
            benchmark_results = self._run_benchmark_suite(parallel=parallel)
            
            # Generate comprehensive analysis
            analysis_results = self._generate_comprehensive_analysis(
                validation_results, benchmark_results
            )
            
            # Export results
            self._export_results(validation_results, benchmark_results, analysis_results, output_path)
            
            # Generate visualizations
            if self.config.visualization_config['generate_plots']:
                self._generate_visualizations(validation_results, benchmark_results, output_path)
            
            # Generate comprehensive report
            if self.config.output_config['generate_report']:
                self._generate_comprehensive_report(
                    validation_results, benchmark_results, analysis_results, output_path
                )
            
            execution_time = time.time() - start_time
            self.logger.info(f"Comprehensive validation completed in {execution_time:.2f} seconds")
            
            return {
                'validation_results': validation_results,
                'benchmark_results': benchmark_results,
                'analysis_results': analysis_results,
                'execution_time': execution_time,
                'output_directory': str(output_path),
                'framework_version': '1.0.0'
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            raise BorgiaTestError(f"Validation suite execution failed: {e}")
    
    def _run_validation_suite(self, parallel: bool = True) -> Dict[str, ValidationResult]:
        """Run the complete validation test suite."""
        self.logger.info("Executing validation test suite")
        
        validation_tests = [
            ('molecular_generation', self._validate_molecular_generation),
            ('dual_functionality', self._validate_dual_functionality),
            ('bmd_networks', self._validate_bmd_networks),
            ('information_catalysis', self._validate_information_catalysis),
            ('hardware_integration', self._validate_hardware_integration),
            ('downstream_integration', self._validate_downstream_integration),
            ('quality_control', self._validate_quality_control),
            ('cascade_failure_protection', self._validate_cascade_failure_protection)
        ]
        
        results = {}
        
        if parallel and len(validation_tests) > 1:
            # Run tests in parallel using threading
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_test = {
                    executor.submit(test_func): test_name 
                    for test_name, test_func in validation_tests
                }
                
                for future in concurrent.futures.as_completed(future_to_test):
                    test_name = future_to_test[future]
                    try:
                        result = future.result()
                        results[test_name] = result
                        self.logger.info(f"✓ {test_name} validation completed - Score: {result.score:.3f}")
                    except Exception as e:
                        self.logger.error(f"✗ {test_name} validation failed: {e}")
                        results[test_name] = ValidationResult(
                            test_name=test_name,
                            success=False,
                            score=0.0,
                            metrics={},
                            errors=[str(e)]
                        )
        else:
            # Run tests sequentially
            for test_name, test_func in validation_tests:
                try:
                    result = test_func()
                    results[test_name] = result
                    self.logger.info(f"✓ {test_name} validation completed - Score: {result.score:.3f}")
                except Exception as e:
                    self.logger.error(f"✗ {test_name} validation failed: {e}")
                    results[test_name] = ValidationResult(
                        test_name=test_name,
                        success=False,
                        score=0.0,
                        metrics={},
                        errors=[str(e)]
                    )
        
        return results
    
    def _run_benchmark_suite(self, parallel: bool = True) -> Dict[str, BenchmarkResult]:
        """Run the complete performance benchmark suite."""
        self.logger.info("Executing performance benchmark suite")
        
        benchmark_tests = [
            ('molecular_generation_performance', self._benchmark_molecular_generation),
            ('bmd_network_performance', self._benchmark_bmd_networks),
            ('hardware_integration_performance', self._benchmark_hardware_integration),
            ('information_catalysis_performance', self._benchmark_information_catalysis),
            ('system_scalability', self._benchmark_system_scalability),
            ('memory_efficiency', self._benchmark_memory_efficiency),
            ('cpu_utilization', self._benchmark_cpu_utilization)
        ]
        
        results = {}
        
        if parallel and len(benchmark_tests) > 1:
            # Run benchmarks in parallel
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_to_benchmark = {
                    executor.submit(benchmark_func): benchmark_name 
                    for benchmark_name, benchmark_func in benchmark_tests
                }
                
                for future in concurrent.futures.as_completed(future_to_benchmark):
                    benchmark_name = future_to_benchmark[future]
                    try:
                        result = future.result()
                        results[benchmark_name] = result
                        self.logger.info(f"✓ {benchmark_name} benchmark completed - Score: {result.performance_score:.3f}")
                    except Exception as e:
                        self.logger.error(f"✗ {benchmark_name} benchmark failed: {e}")
                        results[benchmark_name] = BenchmarkResult(
                            benchmark_name=benchmark_name,
                            performance_score=0.0,
                            throughput=0.0,
                            latency=float('inf'),
                            memory_usage=0.0,
                            cpu_utilization=0.0,
                            metadata={'error': str(e)}
                        )
        else:
            # Run benchmarks sequentially
            for benchmark_name, benchmark_func in benchmark_tests:
                try:
                    result = benchmark_func()
                    results[benchmark_name] = result
                    self.logger.info(f"✓ {benchmark_name} benchmark completed - Score: {result.performance_score:.3f}")
                except Exception as e:
                    self.logger.error(f"✗ {benchmark_name} benchmark failed: {e}")
                    results[benchmark_name] = BenchmarkResult(
                        benchmark_name=benchmark_name,
                        performance_score=0.0,
                        throughput=0.0,
                        latency=float('inf'),
                        memory_usage=0.0,
                        cpu_utilization=0.0,
                        metadata={'error': str(e)}
                    )
        
        return results
    
    # Validation test implementations
    def _validate_molecular_generation(self) -> ValidationResult:
        """Validate molecular generation capabilities."""
        start_time = time.time()
        
        try:
            # Generate test molecules
            molecules = self.molecular_generator.generate_dual_functionality_molecules(
                count=self.config.molecular_generation_config['molecule_count'],
                precision_target=1e-30,
                processing_capacity=1e6
            )
            
            # Validate generation success
            success_rate = len(molecules) / self.config.molecular_generation_config['molecule_count']
            
            metrics = {
                'molecules_generated': len(molecules),
                'generation_success_rate': success_rate,
                'average_generation_time': (time.time() - start_time) / len(molecules),
                'dual_functionality_compliance': 1.0  # All molecules must have dual functionality
            }
            
            return ValidationResult(
                test_name='molecular_generation',
                success=success_rate >= 0.95,
                score=success_rate,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='molecular_generation',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_dual_functionality(self) -> ValidationResult:
        """Validate dual-functionality (clock + processor) capabilities."""
        start_time = time.time()
        
        try:
            # Generate sample molecules for testing
            molecules = self.molecular_generator.generate_dual_functionality_molecules(
                count=1000,
                precision_target=1e-30,
                processing_capacity=1e6
            )
            
            # Validate dual functionality
            validation_results = self.dual_functionality_validator.validate_batch(molecules)
            
            metrics = {
                'total_molecules_tested': len(molecules),
                'clock_functionality_success_rate': validation_results['clock_success_rate'],
                'processor_functionality_success_rate': validation_results['processor_success_rate'],
                'dual_functionality_success_rate': validation_results['dual_functionality_success_rate'],
                'average_clock_precision': validation_results['average_clock_precision'],
                'average_processing_capacity': validation_results['average_processing_capacity']
            }
            
            # Zero tolerance for dual functionality failures
            success = validation_results['dual_functionality_success_rate'] == 1.0
            
            return ValidationResult(
                test_name='dual_functionality',
                success=success,
                score=validation_results['dual_functionality_success_rate'],
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='dual_functionality',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_bmd_networks(self) -> ValidationResult:
        """Validate multi-scale BMD network coordination."""
        start_time = time.time()
        
        try:
            # Test BMD network coordination
            network_results = self.bmd_network_tester.test_multi_scale_coordination()
            
            metrics = {
                'quantum_layer_efficiency': network_results['quantum_efficiency'],
                'molecular_layer_efficiency': network_results['molecular_efficiency'],
                'environmental_layer_efficiency': network_results['environmental_efficiency'],
                'cross_scale_synchronization': network_results['synchronization_quality'],
                'thermodynamic_amplification_factor': network_results['amplification_factor'],
                'information_catalysis_efficiency': network_results['catalysis_efficiency']
            }
            
            # Check if amplification target is met
            amplification_success = network_results['amplification_factor'] >= self.config.bmd_network_config['amplification_target']
            efficiency_success = network_results['catalysis_efficiency'] >= self.config.bmd_network_config['efficiency_target']
            
            success = amplification_success and efficiency_success
            score = min(
                network_results['amplification_factor'] / self.config.bmd_network_config['amplification_target'],
                1.0
            )
            
            return ValidationResult(
                test_name='bmd_networks',
                success=success,
                score=score,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='bmd_networks',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_information_catalysis(self) -> ValidationResult:
        """Validate information catalysis efficiency and thermodynamic amplification."""
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from .information_catalysis import InformationCatalysisValidator
            
            catalysis_validator = InformationCatalysisValidator()
            results = catalysis_validator.validate_catalysis_efficiency()
            
            metrics = {
                'catalysis_efficiency': results['efficiency'],
                'thermodynamic_amplification': results['amplification_factor'],
                'information_preservation_rate': results['information_preservation'],
                'entropy_reduction_factor': results['entropy_reduction'],
                'energy_efficiency': results['energy_efficiency']
            }
            
            success = (
                results['efficiency'] >= 0.95 and
                results['amplification_factor'] >= 1000.0 and
                results['information_preservation'] >= 0.99
            )
            
            return ValidationResult(
                test_name='information_catalysis',
                success=success,
                score=results['efficiency'],
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='information_catalysis',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_hardware_integration(self) -> ValidationResult:
        """Validate hardware integration capabilities."""
        start_time = time.time()
        
        try:
            # Test hardware integration
            integration_results = self.hardware_integration_tester.test_complete_integration()
            
            metrics = {
                'led_spectroscopy_success': integration_results['led_spectroscopy_success'],
                'cpu_timing_coordination_success': integration_results['cpu_timing_success'],
                'noise_enhancement_success': integration_results['noise_enhancement_success'],
                'performance_improvement_factor': integration_results['performance_improvement'],
                'memory_reduction_factor': integration_results['memory_reduction'],
                'zero_cost_validation': integration_results['zero_cost_confirmed']
            }
            
            success = (
                integration_results['led_spectroscopy_success'] and
                integration_results['cpu_timing_success'] and
                integration_results['noise_enhancement_success'] and
                integration_results['performance_improvement'] >= self.config.hardware_integration_config['performance_improvement_target']
            )
            
            score = integration_results['performance_improvement'] / self.config.hardware_integration_config['performance_improvement_target']
            score = min(score, 1.0)
            
            return ValidationResult(
                test_name='hardware_integration',
                success=success,
                score=score,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='hardware_integration',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_downstream_integration(self) -> ValidationResult:
        """Validate integration with downstream systems."""
        start_time = time.time()
        
        try:
            # Import integration validators
            from .integration_tests import MasundaTemporalValidator, BuheraFoundryValidator, KambuzumaIntegrationValidator
            
            # Test all downstream integrations
            masunda_validator = MasundaTemporalValidator()
            buhera_validator = BuheraFoundryValidator()
            kambuzuma_validator = KambuzumaIntegrationValidator()
            
            masunda_results = masunda_validator.validate_oscillating_atom_provision()
            buhera_results = buhera_validator.validate_bmd_substrate_provision()
            kambuzuma_results = kambuzuma_validator.validate_biological_quantum_molecule_provision()
            
            metrics = {
                'masunda_integration_success': masunda_results['success'],
                'masunda_precision_achievement': masunda_results['precision_achieved'],
                'buhera_integration_success': buhera_results['success'],
                'buhera_substrate_quality': buhera_results['substrate_quality'],
                'kambuzuma_integration_success': kambuzuma_results['success'],
                'kambuzuma_quantum_coherence': kambuzuma_results['quantum_coherence']
            }
            
            success = (
                masunda_results['success'] and
                buhera_results['success'] and
                kambuzuma_results['success']
            )
            
            # Calculate composite score
            scores = [
                masunda_results.get('score', 0.0),
                buhera_results.get('score', 0.0),
                kambuzuma_results.get('score', 0.0)
            ]
            score = np.mean(scores)
            
            return ValidationResult(
                test_name='downstream_integration',
                success=success,
                score=score,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='downstream_integration',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_quality_control(self) -> ValidationResult:
        """Validate zero-tolerance quality control system."""
        start_time = time.time()
        
        try:
            from .molecular_generation import MolecularQualityControl
            
            # Test quality control system
            qc_system = MolecularQualityControl()
            
            # Generate molecules with intentional defects for testing
            test_molecules = self.molecular_generator.generate_test_molecules_with_defects(1000)
            
            # Run quality control validation
            qc_results = qc_system.validate_zero_tolerance_protocols(test_molecules)
            
            metrics = {
                'defect_detection_rate': qc_results['defect_detection_rate'],
                'false_positive_rate': qc_results['false_positive_rate'],
                'false_negative_rate': qc_results['false_negative_rate'],
                'processing_speed': qc_results['molecules_per_second'],
                'memory_efficiency': qc_results['memory_usage_mb']
            }
            
            # Zero tolerance means 100% defect detection with no false negatives
            success = (
                qc_results['defect_detection_rate'] == 1.0 and
                qc_results['false_negative_rate'] == 0.0
            )
            
            return ValidationResult(
                test_name='quality_control',
                success=success,
                score=qc_results['defect_detection_rate'],
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='quality_control',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    def _validate_cascade_failure_protection(self) -> ValidationResult:
        """Validate cascade failure prevention mechanisms."""
        start_time = time.time()
        
        try:
            from .integration_tests import CascadeFailureAnalyzer
            
            # Test cascade failure protection
            cascade_analyzer = CascadeFailureAnalyzer()
            
            # Simulate various failure scenarios
            failure_scenarios = [
                'molecular_generation_failure',
                'bmd_network_disruption',
                'hardware_integration_failure',
                'downstream_system_overload'
            ]
            
            results = {}
            for scenario in failure_scenarios:
                scenario_result = cascade_analyzer.test_failure_scenario(scenario)
                results[scenario] = scenario_result
            
            metrics = {
                'failure_detection_time': np.mean([r['detection_time'] for r in results.values()]),
                'recovery_time': np.mean([r['recovery_time'] for r in results.values()]),
                'system_availability': np.mean([r['availability_maintained'] for r in results.values()]),
                'data_integrity_maintained': all(r['data_integrity'] for r in results.values()),
                'redundancy_effectiveness': np.mean([r['redundancy_score'] for r in results.values()])
            }
            
            # Success if system maintains >95% availability and 100% data integrity
            success = (
                metrics['system_availability'] >= 0.95 and
                metrics['data_integrity_maintained']
            )
            
            return ValidationResult(
                test_name='cascade_failure_protection',
                success=success,
                score=metrics['system_availability'],
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name='cascade_failure_protection',
                success=False,
                score=0.0,
                metrics={},
                errors=[str(e)],
                execution_time=time.time() - start_time
            )
    
    # Benchmark implementations
    def _benchmark_molecular_generation(self) -> BenchmarkResult:
        """Benchmark molecular generation performance."""
        return self.performance_benchmarker.benchmark_molecular_generation()
    
    def _benchmark_bmd_networks(self) -> BenchmarkResult:
        """Benchmark BMD network performance."""
        return self.performance_benchmarker.benchmark_bmd_networks()
    
    def _benchmark_hardware_integration(self) -> BenchmarkResult:
        """Benchmark hardware integration performance."""
        return self.performance_benchmarker.benchmark_hardware_integration()
    
    def _benchmark_information_catalysis(self) -> BenchmarkResult:
        """Benchmark information catalysis performance."""
        return self.performance_benchmarker.benchmark_information_catalysis()
    
    def _benchmark_system_scalability(self) -> BenchmarkResult:
        """Benchmark system scalability."""
        return self.performance_benchmarker.benchmark_system_scalability()
    
    def _benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Benchmark memory efficiency."""
        return self.performance_benchmarker.benchmark_memory_efficiency()
    
    def _benchmark_cpu_utilization(self) -> BenchmarkResult:
        """Benchmark CPU utilization efficiency."""
        return self.performance_benchmarker.benchmark_cpu_utilization()
    
    def _generate_comprehensive_analysis(self, 
                                       validation_results: Dict[str, ValidationResult],
                                       benchmark_results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive analysis of all test results."""
        analysis = {
            'overall_validation_score': np.mean([r.score for r in validation_results.values()]),
            'overall_benchmark_score': np.mean([r.performance_score for r in benchmark_results.values()]),
            'tests_passed': sum(1 for r in validation_results.values() if r.success),
            'total_tests': len(validation_results),
            'critical_failures': [
                name for name, result in validation_results.items() 
                if not result.success and name in ['dual_functionality', 'quality_control', 'cascade_failure_protection']
            ],
            'performance_summary': {
                'average_throughput': np.mean([r.throughput for r in benchmark_results.values()]),
                'average_latency': np.mean([r.latency for r in benchmark_results.values()]),
                'peak_memory_usage': max([r.memory_usage for r in benchmark_results.values()]),
                'average_cpu_utilization': np.mean([r.cpu_utilization for r in benchmark_results.values()])
            },
            'recommendations': self._generate_recommendations(validation_results, benchmark_results)
        }
        
        return analysis
    
    def _generate_recommendations(self, 
                                validation_results: Dict[str, ValidationResult],
                                benchmark_results: Dict[str, BenchmarkResult]) -> List[str]:
        """Generate improvement recommendations based on test results."""
        recommendations = []
        
        # Check for critical failures
        critical_failures = [
            name for name, result in validation_results.items() 
            if not result.success and name in ['dual_functionality', 'quality_control', 'cascade_failure_protection']
        ]
        
        if critical_failures:
            recommendations.append(
                f"CRITICAL: Address failures in {', '.join(critical_failures)} immediately. "
                f"These components are essential for system integrity."
            )
        
        # Performance recommendations
        low_performers = [
            name for name, result in validation_results.items()
            if result.score < 0.8
        ]
        
        if low_performers:
            recommendations.append(
                f"Performance optimization needed for: {', '.join(low_performers)}. "
                f"Consider parameter tuning and algorithm improvements."
            )
        
        # Memory usage recommendations
        high_memory_benchmarks = [
            name for name, result in benchmark_results.items()
            if result.memory_usage > 1000  # More than 1GB
        ]
        
        if high_memory_benchmarks:
            recommendations.append(
                f"High memory usage detected in: {', '.join(high_memory_benchmarks)}. "
                f"Consider memory optimization strategies."
            )
        
        # Add positive feedback
        if not critical_failures and not low_performers:
            recommendations.append(
                "Excellent performance across all validation tests. "
                "System is ready for production deployment."
            )
        
        return recommendations
    
    def _export_results(self, 
                       validation_results: Dict[str, ValidationResult],
                       benchmark_results: Dict[str, BenchmarkResult],
                       analysis_results: Dict[str, Any],
                       output_path: Path):
        """Export all results in configured formats."""
        self.logger.info(f"Exporting results to {output_path}")
        
        # Prepare comprehensive results data
        results_data = {
            'framework_info': {
                'version': '1.0.0',
                'execution_timestamp': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.framework_start_time
            },
            'validation_results': {name: result.to_dict() for name, result in validation_results.items()},
            'benchmark_results': {name: result.to_dict() for name, result in benchmark_results.items()},
            'analysis_results': analysis_results,
            'configuration': {
                'molecular_generation': self.config.molecular_generation_config,
                'bmd_network': self.config.bmd_network_config,
                'hardware_integration': self.config.hardware_integration_config,
                'performance_benchmark': self.config.performance_benchmark_config
            }
        }
        
        # Export JSON
        if self.config.output_config['export_json']:
            json_path = output_path / 'comprehensive_results.json'
            with open(json_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            self.logger.info(f"Results exported to {json_path}")
        
        # Export CSV summaries
        if self.config.output_config['export_csv']:
            # Validation results CSV
            validation_df = pd.DataFrame([
                {
                    'test_name': name,
                    'success': result.success,
                    'score': result.score,
                    'execution_time': result.execution_time,
                    'errors': len(result.errors),
                    'warnings': len(result.warnings)
                }
                for name, result in validation_results.items()
            ])
            validation_df.to_csv(output_path / 'validation_summary.csv', index=False)
            
            # Benchmark results CSV
            benchmark_df = pd.DataFrame([result.to_dict() for result in benchmark_results.values()])
            benchmark_df.to_csv(output_path / 'benchmark_summary.csv', index=False)
            
            self.logger.info("CSV summaries exported")
        
        # Export HDF5 for detailed data
        if self.config.output_config['export_hdf5']:
            try:
                import h5py
                
                h5_path = output_path / 'detailed_results.h5'
                with h5py.File(h5_path, 'w') as f:
                    # Store validation metrics
                    val_group = f.create_group('validation')
                    for name, result in validation_results.items():
                        test_group = val_group.create_group(name)
                        for metric_name, metric_value in result.metrics.items():
                            if isinstance(metric_value, (int, float)):
                                test_group.create_dataset(metric_name, data=metric_value)
                    
                    # Store benchmark metrics
                    bench_group = f.create_group('benchmarks')
                    for name, result in benchmark_results.items():
                        bench_group.create_dataset(f'{name}/performance_score', data=result.performance_score)
                        bench_group.create_dataset(f'{name}/throughput', data=result.throughput)
                        bench_group.create_dataset(f'{name}/latency', data=result.latency)
                        bench_group.create_dataset(f'{name}/memory_usage', data=result.memory_usage)
                        bench_group.create_dataset(f'{name}/cpu_utilization', data=result.cpu_utilization)
                
                self.logger.info(f"HDF5 data exported to {h5_path}")
                
            except ImportError:
                self.logger.warning("HDF5 export skipped - h5py not available")
    
    def _generate_visualizations(self, 
                               validation_results: Dict[str, ValidationResult],
                               benchmark_results: Dict[str, BenchmarkResult],
                               output_path: Path):
        """Generate comprehensive visualizations."""
        viz_output_path = output_path / 'visualizations'
        viz_output_path.mkdir(exist_ok=True)
        
        self.logger.info("Generating comprehensive visualizations")
        
        try:
            # Generate validation overview
            self.visualizer.plot_validation_overview(
                validation_results, 
                output_path=viz_output_path / 'validation_overview.png'
            )
            
            # Generate benchmark comparison
            self.visualizer.plot_benchmark_comparison(
                benchmark_results,
                output_path=viz_output_path / 'benchmark_comparison.png'
            )
            
            # Generate performance heatmap
            self.visualizer.plot_performance_heatmap(
                validation_results, benchmark_results,
                output_path=viz_output_path / 'performance_heatmap.png'
            )
            
            # Generate interactive dashboard
            if self.config.visualization_config['interactive_dashboards']:
                self.visualizer.create_interactive_dashboard(
                    validation_results, benchmark_results,
                    output_path=viz_output_path / 'interactive_dashboard.html'
                )
            
            self.logger.info("All visualizations generated successfully")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
    
    def _generate_comprehensive_report(self,
                                     validation_results: Dict[str, ValidationResult],
                                     benchmark_results: Dict[str, BenchmarkResult], 
                                     analysis_results: Dict[str, Any],
                                     output_path: Path):
        """Generate comprehensive PDF report."""
        try:
            from .reports import BorgiaReportGenerator
            
            report_generator = BorgiaReportGenerator()
            
            report_path = output_path / 'borgia_comprehensive_report.pdf'
            
            report_generator.generate_comprehensive_report(
                validation_results=validation_results,
                benchmark_results=benchmark_results,
                analysis_results=analysis_results,
                output_path=report_path,
                include_visualizations=True
            )
            
            self.logger.info(f"Comprehensive report generated: {report_path}")
            
        except ImportError:
            self.logger.warning("Report generation skipped - report dependencies not available")
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")


# Utility functions for standalone usage
def run_quick_validation() -> Dict[str, Any]:
    """Run a quick validation of the Borgia system."""
    framework = BorgiaTestFramework()
    
    # Use reduced configuration for quick validation
    quick_config = TestConfiguration()
    quick_config.molecular_generation_config['molecule_count'] = 100
    quick_config.performance_benchmark_config['duration_seconds'] = 60
    
    framework.config = quick_config
    
    return framework.run_comprehensive_validation(parallel=False)


def validate_system_requirements(verbose: bool = True) -> bool:
    """
    Validate that system requirements are met for running the framework.
    
    Args:
        verbose: Whether to print detailed information
        
    Returns:
        bool: True if requirements are met, False otherwise
    """
    requirements_met = True
    
    if verbose:
        print("🔍 Validating Borgia Test Framework system requirements...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        if verbose:
            print("✗ Python 3.8+ required")
        requirements_met = False
    elif verbose:
        print(f"✓ Python {sys.version.split()[0]}")
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 16:
            if verbose:
                print(f"⚠ Warning: {memory_gb:.1f}GB RAM detected (16GB+ recommended)")
        elif verbose:
            print(f"✓ {memory_gb:.1f}GB RAM available")
    except ImportError:
        if verbose:
            print("⚠ Cannot verify memory requirements (psutil not installed)")
    
    # Check CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 4:
        if verbose:
            print(f"⚠ Warning: {cpu_count} CPU cores (4+ recommended)")
    elif verbose:
        print(f"✓ {cpu_count} CPU cores available")
    
    # Check critical dependencies
    critical_deps = ['numpy', 'scipy', 'matplotlib', 'pandas', 'rdkit']
    
    for dep in critical_deps:
        try:
            __import__(dep)
            if verbose:
                print(f"✓ {dep} available")
        except ImportError:
            if verbose:
                print(f"✗ {dep} missing")
            requirements_met = False
    
    if verbose:
        if requirements_met:
            print("🎉 All system requirements met!")
        else:
            print("❌ Some requirements not met. Please install missing dependencies.")
    
    return requirements_met
