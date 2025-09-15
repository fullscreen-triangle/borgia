"""
Borgia Test Framework - Data Structures Module
==============================================

Core data structures for the Borgia biological Maxwell demons (BMD)
cheminformatics test framework. Provides standardized data containers
for molecular data, validation results, and performance metrics.

Author: Borgia Development Team
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import pickle
import h5py


@dataclass
class MolecularData:
    """Container for molecular data and properties."""
    molecular_id: str
    smiles: str
    formula: str
    molecular_weight: float
    
    # Dual functionality properties
    clock_properties: Dict[str, Any] = field(default_factory=dict)
    processor_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    dual_functionality_score: float = 0.0
    thermodynamic_efficiency: float = 0.0
    information_catalysis_capability: float = 0.0
    
    # Validation status
    validation_passed: bool = False
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    generation_timestamp: datetime = field(default_factory=datetime.now)
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data['generation_timestamp'] = self.generation_timestamp.isoformat()
        return data
    
    def to_json(self, filepath: Optional[Path] = None) -> str:
        """Export to JSON format."""
        json_data = json.dumps(self.to_dict(), indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
        
        return json_data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MolecularData':
        """Create instance from dictionary."""
        if 'generation_timestamp' in data and isinstance(data['generation_timestamp'], str):
            data['generation_timestamp'] = datetime.fromisoformat(data['generation_timestamp'])
        return cls(**data)


@dataclass
class BMDNetworkData:
    """Container for BMD network data and metrics."""
    network_id: str
    network_size: int
    coordination_protocol: str
    
    # Layer performance data
    quantum_layer_data: Dict[str, Any] = field(default_factory=dict)
    molecular_layer_data: Dict[str, Any] = field(default_factory=dict)
    environmental_layer_data: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-scale coordination data
    synchronization_quality: float = 0.0
    information_flow_efficiency: float = 0.0
    amplification_factor: float = 0.0
    
    # Network topology
    adjacency_matrix: Optional[np.ndarray] = None
    connectivity_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    overall_efficiency: float = 0.0
    execution_time: float = 0.0
    
    # Metadata
    test_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with numpy array handling."""
        data = asdict(self)
        data['test_timestamp'] = self.test_timestamp.isoformat()
        
        if self.adjacency_matrix is not None:
            data['adjacency_matrix'] = self.adjacency_matrix.tolist()
        
        return data
    
    def save_hdf5(self, filepath: Path):
        """Save to HDF5 format with efficient array storage."""
        with h5py.File(filepath, 'w') as f:
            # Basic attributes
            f.attrs['network_id'] = self.network_id
            f.attrs['network_size'] = self.network_size
            f.attrs['coordination_protocol'] = self.coordination_protocol
            f.attrs['synchronization_quality'] = self.synchronization_quality
            f.attrs['information_flow_efficiency'] = self.information_flow_efficiency
            f.attrs['amplification_factor'] = self.amplification_factor
            f.attrs['overall_efficiency'] = self.overall_efficiency
            f.attrs['execution_time'] = self.execution_time
            f.attrs['test_timestamp'] = self.test_timestamp.isoformat()
            
            # Layer data groups
            quantum_group = f.create_group('quantum_layer')
            for key, value in self.quantum_layer_data.items():
                if isinstance(value, np.ndarray):
                    quantum_group.create_dataset(key, data=value)
                else:
                    quantum_group.attrs[key] = value
            
            molecular_group = f.create_group('molecular_layer')
            for key, value in self.molecular_layer_data.items():
                if isinstance(value, np.ndarray):
                    molecular_group.create_dataset(key, data=value)
                else:
                    molecular_group.attrs[key] = value
            
            environmental_group = f.create_group('environmental_layer')
            for key, value in self.environmental_layer_data.items():
                if isinstance(value, np.ndarray):
                    environmental_group.create_dataset(key, data=value)
                else:
                    environmental_group.attrs[key] = value
            
            # Adjacency matrix
            if self.adjacency_matrix is not None:
                f.create_dataset('adjacency_matrix', data=self.adjacency_matrix)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics and benchmarks."""
    test_name: str
    test_category: str
    
    # Core performance metrics
    throughput: float = 0.0                    # Operations per second
    latency: float = 0.0                       # Average response time (seconds)
    memory_usage: float = 0.0                  # Peak memory usage (MB)
    cpu_utilization: float = 0.0               # Average CPU utilization (%)
    gpu_utilization: Optional[float] = None    # GPU utilization if applicable
    
    # Quality metrics
    accuracy: float = 0.0                      # Result accuracy (0-1)
    precision: float = 0.0                     # Result precision (0-1)
    recall: float = 0.0                        # Result recall (0-1)
    f1_score: float = 0.0                      # F1 score (0-1)
    
    # Scalability metrics
    scalability_factor: float = 1.0           # Performance scaling factor
    bottleneck_component: Optional[str] = None # Performance bottleneck
    
    # Resource efficiency
    energy_efficiency: float = 0.0            # Operations per joule
    memory_efficiency: float = 0.0            # Operations per MB
    
    # Statistical data
    measurement_count: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: float = 0.0
    
    # Metadata
    measurement_timestamp: datetime = field(default_factory=datetime.now)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_composite_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted combination of performance metrics
        throughput_score = min(self.throughput / 1e6, 1.0)  # Normalize to 1M ops/sec
        latency_score = max(1.0 - self.latency, 0.0)  # Lower latency is better
        accuracy_score = self.accuracy
        efficiency_score = (self.energy_efficiency + self.memory_efficiency) / 2
        
        weights = [0.3, 0.2, 0.3, 0.2]  # Throughput, Latency, Accuracy, Efficiency
        scores = [throughput_score, latency_score, accuracy_score, efficiency_score]
        
        composite_score = sum(w * s for w, s in zip(weights, scores))
        return composite_score
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series for easy analysis."""
        data_dict = asdict(self)
        data_dict['measurement_timestamp'] = self.measurement_timestamp.isoformat()
        data_dict['composite_score'] = self.calculate_composite_score()
        
        return pd.Series(data_dict)


@dataclass
class ValidationResults:
    """Container for validation test results."""
    validation_id: str
    test_suite: str
    
    # Overall results
    overall_passed: bool = False
    overall_score: float = 0.0
    success_rate: float = 0.0
    
    # Individual test results
    individual_results: List[Dict[str, Any]] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    
    # Quality metrics
    dual_functionality_success_rate: float = 0.0
    amplification_factor_achieved: float = 0.0
    information_preservation_rate: float = 0.0
    
    # Performance data
    total_execution_time: float = 0.0
    molecules_tested: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    
    # Error analysis
    error_distribution: Dict[str, int] = field(default_factory=dict)
    cascade_failure_risk: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def add_individual_result(self, test_name: str, passed: bool, score: float, 
                            details: Optional[Dict[str, Any]] = None):
        """Add individual test result."""
        result = {
            'test_name': test_name,
            'passed': passed,
            'score': score,
            'details': details or {}
        }
        
        self.individual_results.append(result)
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
            self.failed_tests.append(test_name)
        
        # Update overall metrics
        total_tests = self.tests_passed + self.tests_failed
        self.success_rate = self.tests_passed / total_tests if total_tests > 0 else 0.0
        
        # Update overall score (average of individual scores)
        scores = [r['score'] for r in self.individual_results]
        self.overall_score = np.mean(scores) if scores else 0.0
        
        # Overall pass determination
        self.overall_passed = self.success_rate >= 0.8 and len(self.critical_failures) == 0
    
    def export_summary(self) -> Dict[str, Any]:
        """Export summary of validation results."""
        return {
            'validation_id': self.validation_id,
            'test_suite': self.test_suite,
            'overall_passed': self.overall_passed,
            'overall_score': self.overall_score,
            'success_rate': self.success_rate,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'critical_failures': len(self.critical_failures),
            'molecules_tested': self.molecules_tested,
            'execution_time': self.total_execution_time,
            'cascade_failure_risk': self.cascade_failure_risk,
            'validation_timestamp': self.validation_timestamp.isoformat()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert individual results to DataFrame."""
        if not self.individual_results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.individual_results)


@dataclass 
class BenchmarkResults:
    """Container for benchmark test results."""
    benchmark_id: str
    benchmark_suite: str
    
    # Performance results
    performance_metrics: List[PerformanceMetrics] = field(default_factory=list)
    
    # Statistical analysis
    mean_performance: Dict[str, float] = field(default_factory=dict)
    std_performance: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Comparative analysis
    baseline_comparison: Dict[str, float] = field(default_factory=dict)
    improvement_factors: Dict[str, float] = field(default_factory=dict)
    
    # Resource utilization
    peak_memory_usage: float = 0.0
    average_cpu_utilization: float = 0.0
    total_energy_consumption: float = 0.0
    
    # Scalability analysis
    scalability_metrics: Dict[str, Any] = field(default_factory=dict)
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    benchmark_timestamp: datetime = field(default_factory=datetime.now)
    benchmark_parameters: Dict[str, Any] = field(default_factory=dict)
    system_configuration: Dict[str, Any] = field(default_factory=dict)
    
    def add_performance_metric(self, metric: PerformanceMetrics):
        """Add performance metric to results."""
        self.performance_metrics.append(metric)
        self._update_statistical_analysis()
    
    def _update_statistical_analysis(self):
        """Update statistical analysis based on current metrics."""
        if not self.performance_metrics:
            return
        
        # Convert metrics to DataFrame for analysis
        df = pd.DataFrame([metric.to_series() for metric in self.performance_metrics])
        
        # Calculate statistical measures
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            self.mean_performance[col] = df[col].mean()
            self.std_performance[col] = df[col].std()
            
            # 95% confidence interval
            n = len(df)
            if n > 1:
                sem = df[col].std() / np.sqrt(n)  # Standard error of mean
                ci = 1.96 * sem  # 95% CI
                self.confidence_intervals[col] = (
                    self.mean_performance[col] - ci,
                    self.mean_performance[col] + ci
                )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_metrics:
            return {'error': 'No performance metrics available'}
        
        return {
            'benchmark_summary': {
                'benchmark_id': self.benchmark_id,
                'benchmark_suite': self.benchmark_suite,
                'total_measurements': len(self.performance_metrics),
                'benchmark_timestamp': self.benchmark_timestamp.isoformat()
            },
            'performance_statistics': {
                'mean_performance': self.mean_performance,
                'std_performance': self.std_performance,
                'confidence_intervals': self.confidence_intervals
            },
            'resource_utilization': {
                'peak_memory_usage': self.peak_memory_usage,
                'average_cpu_utilization': self.average_cpu_utilization,
                'total_energy_consumption': self.total_energy_consumption
            },
            'comparative_analysis': {
                'baseline_comparison': self.baseline_comparison,
                'improvement_factors': self.improvement_factors
            },
            'scalability_analysis': self.scalability_metrics
        }


class ResultsDatabase:
    """Database for storing and retrieving test results."""
    
    def __init__(self, db_path: Path):
        """Initialize results database."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        (self.db_path / 'molecular_data').mkdir(exist_ok=True)
        (self.db_path / 'validation_results').mkdir(exist_ok=True)
        (self.db_path / 'benchmark_results').mkdir(exist_ok=True)
        (self.db_path / 'bmd_networks').mkdir(exist_ok=True)
        (self.db_path / 'performance_metrics').mkdir(exist_ok=True)
    
    def store_molecular_data(self, data: MolecularData):
        """Store molecular data."""
        filepath = self.db_path / 'molecular_data' / f"{data.molecular_id}.json"
        data.to_json(filepath)
    
    def store_validation_results(self, results: ValidationResults):
        """Store validation results."""
        filepath = self.db_path / 'validation_results' / f"{results.validation_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def store_benchmark_results(self, results: BenchmarkResults):
        """Store benchmark results."""
        filepath = self.db_path / 'benchmark_results' / f"{results.benchmark_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def store_bmd_network_data(self, data: BMDNetworkData):
        """Store BMD network data."""
        filepath = self.db_path / 'bmd_networks' / f"{data.network_id}.h5"
        data.save_hdf5(filepath)
    
    def load_molecular_data(self, molecular_id: str) -> Optional[MolecularData]:
        """Load molecular data by ID."""
        filepath = self.db_path / 'molecular_data' / f"{molecular_id}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            return MolecularData.from_dict(data)
        return None
    
    def load_validation_results(self, validation_id: str) -> Optional[ValidationResults]:
        """Load validation results by ID."""
        filepath = self.db_path / 'validation_results' / f"{validation_id}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_benchmark_results(self, benchmark_id: str) -> Optional[BenchmarkResults]:
        """Load benchmark results by ID."""
        filepath = self.db_path / 'benchmark_results' / f"{benchmark_id}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available data in the database."""
        available_data = {}
        
        # List molecular data
        molecular_files = list((self.db_path / 'molecular_data').glob('*.json'))
        available_data['molecular_data'] = [f.stem for f in molecular_files]
        
        # List validation results
        validation_files = list((self.db_path / 'validation_results').glob('*.pkl'))
        available_data['validation_results'] = [f.stem for f in validation_files]
        
        # List benchmark results
        benchmark_files = list((self.db_path / 'benchmark_results').glob('*.pkl'))
        available_data['benchmark_results'] = [f.stem for f in benchmark_files]
        
        # List BMD network data
        network_files = list((self.db_path / 'bmd_networks').glob('*.h5'))
        available_data['bmd_networks'] = [f.stem for f in network_files]
        
        return available_data
    
    def export_summary_report(self, output_path: Path):
        """Export summary report of all stored data."""
        available_data = self.list_available_data()
        
        summary_report = {
            'database_path': str(self.db_path),
            'export_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'molecular_data_count': len(available_data['molecular_data']),
                'validation_results_count': len(available_data['validation_results']),
                'benchmark_results_count': len(available_data['benchmark_results']),
                'bmd_networks_count': len(available_data['bmd_networks'])
            },
            'available_data': available_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        return summary_report
