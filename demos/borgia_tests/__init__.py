"""
Borgia Comprehensive Test/Validation Framework
==============================================

A sophisticated testing and validation framework for the Borgia biological Maxwell demons 
(BMD) cheminformatics engine implementing Eduardo Mizraji's biological Maxwell demons theory.

This framework provides comprehensive validation of:
- Multi-scale BMD networks across quantum, molecular, and environmental timescales  
- Dual-functionality molecules functioning as both precision clocks and computational processors
- Information catalysis efficiency with >1000× thermodynamic amplification
- Hardware integration with LED spectroscopy and CPU timing synchronization
- Integration with downstream systems for temporal navigation and quantum processing

Author: Borgia Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Borgia Development Team"
__email__ = "research@borgia-framework.org"
__license__ = "MIT"

# Core framework imports
from .core import (
    BorgiaTestFramework,
    ValidationResult,
    BenchmarkResult,
    TestConfiguration
)

# Molecular generation testing
from .molecular_generation import (
    MolecularGenerator,
    DualFunctionalityValidator,
    ClockValidator,
    ProcessorValidator,
    MolecularQualityControl
)

# BMD network testing
from .bmd_networks import (
    BMDNetworkTester,
    QuantumBMDLayer,
    MolecularBMDLayer, 
    EnvironmentalBMDLayer,
    CrossScaleCoordinator
)

# Hardware integration testing
from .hardware_integration import (
    HardwareIntegrationTester,
    LEDSpectroscopyValidator,
    CPUTimingCoordinator,
    NoiseEnhancementProcessor
)

# Information catalysis testing
from .information_catalysis import (
    InformationCatalysisValidator,
    ThermodynamicAmplificationTester,
    EntropyAnalyzer,
    BMDEfficiencyAnalyzer
)

# Downstream system integration
from .integration_tests import (
    MasundaTemporalValidator,
    BuheraFoundryValidator,
    KambuzumaIntegrationValidator,
    CascadeFailureAnalyzer
)

# Performance benchmarking
from .benchmarks import (
    PerformanceBenchmarker,
    MolecularGenerationBenchmark,
    BMDNetworkBenchmark,
    HardwareIntegrationBenchmark
)

# Visualization framework
from .visualization import (
    BorgiaVisualizer,
    MolecularVisualizer,
    BMDNetworkVisualizer,
    PerformanceVisualizer,
    InteractiveDashboard
)

# Utility functions
from .utils import (
    load_test_data,
    export_results,
    validate_system_requirements,
    generate_test_molecules,
    calculate_performance_metrics
)

# Configuration management
from .config import (
    TestConfig,
    VisualizationConfig,
    BenchmarkConfig,
    ValidationConfig
)

# Test data structures
from .data_structures import (
    MolecularData,
    BMDNetworkData,
    PerformanceMetrics,
    ValidationResults,
    BenchmarkResults
)

# Exception classes
from .exceptions import (
    BorgiaTestError,
    ValidationError,
    BenchmarkError,
    ConfigurationError,
    HardwareError
)

# Main framework classes available at package level
__all__ = [
    # Core framework
    'BorgiaTestFramework',
    'ValidationResult', 
    'BenchmarkResult',
    'TestConfiguration',
    
    # Molecular generation
    'MolecularGenerator',
    'DualFunctionalityValidator',
    'ClockValidator',
    'ProcessorValidator',
    'MolecularQualityControl',
    
    # BMD networks
    'BMDNetworkTester',
    'QuantumBMDLayer',
    'MolecularBMDLayer',
    'EnvironmentalBMDLayer', 
    'CrossScaleCoordinator',
    
    # Hardware integration
    'HardwareIntegrationTester',
    'LEDSpectroscopyValidator',
    'CPUTimingCoordinator',
    'NoiseEnhancementProcessor',
    
    # Information catalysis
    'InformationCatalysisValidator',
    'ThermodynamicAmplificationTester',
    'EntropyAnalyzer',
    'BMDEfficiencyAnalyzer',
    
    # Integration testing
    'MasundaTemporalValidator',
    'BuheraFoundryValidator', 
    'KambuzumaIntegrationValidator',
    'CascadeFailureAnalyzer',
    
    # Benchmarking
    'PerformanceBenchmarker',
    'MolecularGenerationBenchmark',
    'BMDNetworkBenchmark',
    'HardwareIntegrationBenchmark',
    
    # Visualization
    'BorgiaVisualizer',
    'MolecularVisualizer',
    'BMDNetworkVisualizer',
    'PerformanceVisualizer',
    'InteractiveDashboard',
    
    # Utilities
    'load_test_data',
    'export_results',
    'validate_system_requirements',
    'generate_test_molecules',
    'calculate_performance_metrics',
    
    # Configuration
    'TestConfig',
    'VisualizationConfig',
    'BenchmarkConfig',
    'ValidationConfig',
    
    # Data structures
    'MolecularData',
    'BMDNetworkData',
    'PerformanceMetrics',
    'ValidationResults',
    'BenchmarkResults',
    
    # Exceptions
    'BorgiaTestError',
    'ValidationError',
    'BenchmarkError', 
    'ConfigurationError',
    'HardwareError'
]

# Framework metadata
FRAMEWORK_INFO = {
    'name': 'Borgia Test Framework',
    'version': __version__,
    'description': 'Comprehensive test/validation framework for Borgia BMD cheminformatics engine',
    'author': __author__,
    'license': __license__,
    'url': 'https://github.com/fullscreen-triangle/borgia',
    'keywords': [
        'cheminformatics',
        'biological-maxwell-demons',
        'molecular-generation',
        'quantum-chemistry', 
        'information-catalysis',
        'thermodynamic-amplification',
        'multi-scale-networks',
        'hardware-integration'
    ]
}

def get_framework_info():
    """Return comprehensive framework information."""
    return FRAMEWORK_INFO.copy()

def validate_installation():
    """
    Validate that the Borgia Test Framework is properly installed.
    
    Returns:
        bool: True if installation is valid, False otherwise
        
    Raises:
        ImportError: If critical dependencies are missing
    """
    try:
        # Test core dependencies
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        import pandas as pd
        import rdkit
        
        # Test framework modules
        from . import core
        from . import molecular_generation
        from . import bmd_networks
        from . import hardware_integration
        from . import visualization
        
        print("✓ Borgia Test Framework installation validated successfully")
        print(f"✓ Framework version: {__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ SciPy version: {scipy.__version__}")
        print(f"✓ RDKit version: {rdkit.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Installation validation failed: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

# Quick start function
def quick_start_demo():
    """
    Run a quick demonstration of the Borgia Test Framework capabilities.
    
    This function provides a rapid overview of the framework's key features
    for new users to verify installation and understand basic functionality.
    """
    try:
        print("🧬 Borgia Test Framework - Quick Start Demo")
        print("=" * 50)
        
        # Initialize core framework
        framework = BorgiaTestFramework()
        print("✓ Framework initialized")
        
        # Generate sample dual-functionality molecules
        generator = MolecularGenerator()
        molecules = generator.generate_dual_functionality_molecules(
            count=100,
            precision_target=1e-30,
            processing_capacity=1e6
        )
        print(f"✓ Generated {len(molecules)} dual-functionality molecules")
        
        # Quick validation
        validator = DualFunctionalityValidator()
        results = validator.quick_validation(molecules[:10])
        print(f"✓ Validation complete - Success rate: {results.success_rate:.1%}")
        
        # Generate basic visualization
        visualizer = BorgiaVisualizer()
        visualizer.plot_quick_overview(results)
        print("✓ Quick visualization generated")
        
        print("\n🎉 Quick start demo completed successfully!")
        print("📚 See README.md for comprehensive usage examples")
        
    except Exception as e:
        print(f"✗ Quick start demo failed: {e}")
        print("Please check your installation and try again")

# Initialize framework on import
def _initialize_framework():
    """Initialize framework components and validate system requirements."""
    try:
        # Validate system requirements silently
        validate_system_requirements(verbose=False)
        
        # Set default configurations
        TestConfig.load_defaults()
        VisualizationConfig.load_defaults()
        
    except Exception as e:
        # Non-critical initialization errors should not prevent import
        pass

# Call initialization
_initialize_framework()
