"""
Borgia Test Framework - Utility Functions
=========================================

Utility functions for the Borgia biological Maxwell demons (BMD)
cheminformatics test framework. Provides common functionality
for data processing, file operations, and system validation.

Author: Borgia Development Team
"""

import os
import sys
import json
import pickle
import logging
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import gzip
import shutil


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging for the Borgia test framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('borgia_tests')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_system_requirements(verbose: bool = True) -> bool:
    """
    Validate that system requirements are met for the test framework.
    
    Args:
        verbose: Whether to print detailed information
        
    Returns:
        True if requirements are met, False otherwise
    """
    requirements_met = True
    
    if verbose:
        print("🔍 Validating Borgia Test Framework system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        if verbose:
            print("✗ Python 3.8+ required")
        requirements_met = False
    elif verbose:
        print(f"✓ Python {sys.version.split()[0]}")
    
    # Check memory
    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 16:
            if verbose:
                print(f"⚠ Warning: {memory_gb:.1f}GB RAM detected (16GB+ recommended)")
        elif verbose:
            print(f"✓ {memory_gb:.1f}GB RAM available")
    except Exception:
        if verbose:
            print("⚠ Cannot verify memory requirements")
    
    # Check CPU cores
    cpu_count = os.cpu_count() or 1
    if cpu_count < 4:
        if verbose:
            print(f"⚠ Warning: {cpu_count} CPU cores (4+ recommended)")
    elif verbose:
        print(f"✓ {cpu_count} CPU cores available")
    
    # Check critical dependencies
    critical_deps = ['numpy', 'scipy', 'matplotlib', 'pandas']
    
    for dep in critical_deps:
        try:
            __import__(dep)
            if verbose:
                print(f"✓ {dep} available")
        except ImportError:
            if verbose:
                print(f"✗ {dep} missing")
            requirements_met = False
    
    # Check optional dependencies
    optional_deps = ['rdkit', 'plotly', 'networkx', 'seaborn']
    missing_optional = []
    
    for dep in optional_deps:
        try:
            __import__(dep)
            if verbose:
                print(f"✓ {dep} available")
        except ImportError:
            missing_optional.append(dep)
            if verbose:
                print(f"⚠ {dep} optional but recommended")
    
    if verbose:
        if requirements_met:
            print("🎉 Core system requirements met!")
            if missing_optional:
                print(f"📝 Optional dependencies missing: {', '.join(missing_optional)}")
                print("   Install with: pip install " + " ".join(missing_optional))
        else:
            print("❌ Some requirements not met. Please install missing dependencies.")
    
    return requirements_met


def load_test_data(data_path: Union[str, Path], 
                  data_type: str = "auto") -> Any:
    """
    Load test data from various file formats.
    
    Args:
        data_path: Path to data file
        data_type: Type of data ("json", "pickle", "csv", "hdf5", "auto")
        
    Returns:
        Loaded data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Auto-detect format if not specified
    if data_type == "auto":
        suffix = data_path.suffix.lower()
        format_map = {
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.csv': 'csv',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5'
        }
        data_type = format_map.get(suffix, 'json')
    
    # Load based on type
    if data_type == "json":
        with open(data_path, 'r') as f:
            return json.load(f)
    
    elif data_type == "pickle":
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    
    elif data_type == "csv":
        return pd.read_csv(data_path)
    
    elif data_type == "hdf5":
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                data = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data[key] = f[key][...]
                    elif isinstance(f[key], h5py.Group):
                        data[key] = {}
                        for subkey in f[key].keys():
                            data[key][subkey] = f[key][subkey][...]
                return data
        except ImportError:
            raise ImportError("h5py required for HDF5 data loading")
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def export_results(results: Any, 
                  output_path: Union[str, Path],
                  format_type: str = "auto",
                  compress: bool = False) -> Path:
    """
    Export results to various file formats.
    
    Args:
        results: Results data to export
        output_path: Output file path
        format_type: Export format ("json", "pickle", "csv", "hdf5", "auto")
        compress: Whether to compress the output
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format if not specified
    if format_type == "auto":
        suffix = output_path.suffix.lower()
        format_map = {
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle', 
            '.csv': 'csv',
            '.h5': 'hdf5',
            '.hdf5': 'hdf5'
        }
        format_type = format_map.get(suffix, 'json')
        if suffix not in format_map:
            output_path = output_path.with_suffix('.json')
    
    # Export based on type
    if format_type == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format_type == "pickle":
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format_type == "csv":
        if isinstance(results, pd.DataFrame):
            results.to_csv(output_path, index=False)
        elif isinstance(results, dict):
            pd.DataFrame(results).to_csv(output_path, index=False)
        else:
            pd.DataFrame([results]).to_csv(output_path, index=False)
    
    elif format_type == "hdf5":
        try:
            import h5py
            with h5py.File(output_path, 'w') as f:
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (np.ndarray, list)):
                            f.create_dataset(key, data=value)
                        else:
                            f.attrs[key] = str(value)
                else:
                    f.create_dataset('data', data=results)
        except ImportError:
            raise ImportError("h5py required for HDF5 export")
    
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    
    # Compress if requested
    if compress and output_path.suffix != '.gz':
        compressed_path = Path(str(output_path) + '.gz')
        with open(output_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original uncompressed file
        output_path.unlink()
        output_path = compressed_path
    
    return output_path


def generate_test_molecules(count: int = 100, 
                          seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate test molecular data for framework validation.
    
    Args:
        count: Number of molecules to generate
        seed: Random seed for reproducible generation
        
    Returns:
        List of molecular data dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    molecules = []
    
    # Common molecular templates
    templates = [
        "c1ccccc1",  # Benzene
        "CCO",       # Ethanol
        "CC(=O)O",   # Acetic acid
        "c1ccc2ccccc2c1",  # Naphthalene
        "CC(C)O",    # Isopropanol
        "c1ccc(cc1)O",  # Phenol
        "CC(C)(C)O",  # tert-butanol
        "c1coc2ccccc12",  # Benzofuran
    ]
    
    for i in range(count):
        # Select random template
        template = np.random.choice(templates)
        
        # Generate random properties
        molecular_data = {
            'molecular_id': f'TEST_MOL_{i:04d}',
            'smiles': template,
            'formula': f'C{np.random.randint(6, 20)}H{np.random.randint(6, 30)}O{np.random.randint(0, 5)}',
            'molecular_weight': np.random.uniform(100, 400),
            
            # Dual functionality properties
            'base_frequency': np.random.uniform(1e12, 1e15),
            'frequency_stability': np.random.uniform(0.9, 0.99),
            'phase_coherence': np.random.uniform(0.85, 0.98),
            'temporal_precision': np.random.uniform(1e-35, 1e-25),
            
            'instruction_set_size': np.random.randint(1000, 100000),
            'memory_capacity': np.random.randint(10000, 1000000),
            'processing_rate': np.random.uniform(1e5, 1e7),
            'parallel_processing_capability': np.random.choice([True, False]),
            
            'recursive_enhancement_factor': np.random.uniform(1.2, 3.0),
            'network_coordination_capability': np.random.choice([True, False], p=[0.8, 0.2]),
            
            'dual_functionality_score': np.random.uniform(0.7, 0.95),
            'thermodynamic_efficiency': np.random.uniform(0.8, 0.95),
            'information_catalysis_capability': np.random.uniform(0.75, 0.95),
            
            'generation_timestamp': datetime.now().isoformat()
        }
        
        molecules.append(molecular_data)
    
    return molecules


def calculate_performance_metrics(data: Dict[str, Any], 
                                reference_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate performance metrics from test data.
    
    Args:
        data: Test data dictionary
        reference_data: Optional reference data for comparison
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Basic statistical metrics
    if 'scores' in data:
        scores = np.array(data['scores'])
        metrics.update({
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'q25_score': np.percentile(scores, 25),
            'q75_score': np.percentile(scores, 75)
        })
    
    # Success rate metrics
    if 'success_flags' in data:
        success_flags = np.array(data['success_flags'])
        metrics['success_rate'] = np.mean(success_flags)
        metrics['failure_rate'] = 1.0 - metrics['success_rate']
    
    # Performance metrics
    if 'execution_times' in data:
        times = np.array(data['execution_times'])
        metrics.update({
            'mean_execution_time': np.mean(times),
            'total_execution_time': np.sum(times),
            'throughput': len(times) / np.sum(times) if np.sum(times) > 0 else 0
        })
    
    # Memory metrics
    if 'memory_usage' in data:
        memory = np.array(data['memory_usage'])
        metrics.update({
            'peak_memory_usage': np.max(memory),
            'average_memory_usage': np.mean(memory),
            'memory_efficiency': len(memory) / np.sum(memory) if np.sum(memory) > 0 else 0
        })
    
    # Comparative metrics if reference provided
    if reference_data:
        for key in ['mean_score', 'success_rate', 'throughput']:
            if key in metrics and key in reference_data:
                improvement_key = f'{key}_improvement'
                if reference_data[key] > 0:
                    metrics[improvement_key] = metrics[key] / reference_data[key]
                else:
                    metrics[improvement_key] = float('inf') if metrics[key] > 0 else 1.0
    
    return metrics


def create_summary_report(results_data: Dict[str, Any], 
                         output_path: Optional[Path] = None) -> str:
    """
    Create a comprehensive summary report.
    
    Args:
        results_data: Dictionary containing all test results
        output_path: Optional path to save report
        
    Returns:
        Report content as string
    """
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "BORGIA BMD FRAMEWORK - COMPREHENSIVE TEST REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Framework Version: 1.0.0",
        ""
    ])
    
    # System Information
    report_lines.extend([
        "SYSTEM INFORMATION",
        "-" * 40,
        f"Python Version: {sys.version.split()[0]}",
        f"Platform: {sys.platform}",
        f"CPU Cores: {os.cpu_count()}",
    ])
    
    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        report_lines.append(f"Memory: {memory_gb:.1f} GB")
    except Exception:
        report_lines.append("Memory: Unknown")
    
    report_lines.append("")
    
    # Test Results Summary
    if 'validation_results' in results_data:
        validation = results_data['validation_results']
        report_lines.extend([
            "VALIDATION RESULTS SUMMARY",
            "-" * 40,
            f"Total Tests: {validation.get('total_tests', 0)}",
            f"Passed: {validation.get('tests_passed', 0)}",
            f"Failed: {validation.get('tests_failed', 0)}",
            f"Success Rate: {validation.get('success_rate', 0):.1%}",
            f"Overall Score: {validation.get('overall_score', 0):.3f}",
            ""
        ])
    
    # Benchmark Results Summary
    if 'benchmark_results' in results_data:
        benchmark = results_data['benchmark_results']
        report_lines.extend([
            "BENCHMARK RESULTS SUMMARY",
            "-" * 40,
            f"Average Throughput: {benchmark.get('avg_throughput', 0):.0f} ops/sec",
            f"Average Latency: {benchmark.get('avg_latency', 0):.3f} seconds",
            f"Peak Memory Usage: {benchmark.get('peak_memory', 0):.1f} MB",
            f"Average CPU Usage: {benchmark.get('avg_cpu', 0):.1f}%",
            ""
        ])
    
    # Key Performance Indicators
    if 'key_metrics' in results_data:
        metrics = results_data['key_metrics']
        report_lines.extend([
            "KEY PERFORMANCE INDICATORS",
            "-" * 40,
            f"Dual-Functionality Success: {metrics.get('dual_functionality_rate', 0):.1%}",
            f"Amplification Factor: {metrics.get('amplification_factor', 0):.1f}×",
            f"Information Catalysis Efficiency: {metrics.get('catalysis_efficiency', 0):.1%}",
            f"BMD Network Coordination: {metrics.get('network_coordination', 0):.1%}",
            ""
        ])
    
    # Recommendations
    if 'recommendations' in results_data:
        recommendations = results_data['recommendations']
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 40
        ])
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")
    
    # Footer
    report_lines.extend([
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    # Join all lines
    report_content = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_content)
    
    return report_content


def calculate_file_hash(file_path: Union[str, Path], 
                       algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("md5", "sha1", "sha256", "sha512")
        
    Returns:
        Hexadecimal hash string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get hash function
    hash_func = getattr(hashlib, algorithm.lower())()
    
    # Read file in chunks to handle large files
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def cleanup_temp_files(temp_dir: Union[str, Path], 
                      max_age_hours: float = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of files deleted
    """
    temp_dir = Path(temp_dir)
    
    if not temp_dir.exists():
        return 0
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    for file_path in temp_dir.rglob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    pass  # Skip files that can't be deleted
    
    return deleted_count


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def progress_callback(current: int, total: int, operation: str = "Processing") -> None:
    """
    Simple progress callback function.
    
    Args:
        current: Current item number
        total: Total number of items
        operation: Description of operation
    """
    if total > 0:
        percentage = (current / total) * 100
        bar_length = 50
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f'\r{operation}: |{bar}| {percentage:.1f}% ({current}/{total})', 
              end='', flush=True)
        
        if current == total:
            print()  # New line when complete
