#!/usr/bin/env python3
"""
Gonfanolier: Comprehensive Validation Framework
==============================================

A comprehensive validation framework for the Borgia molecular computing system,
designed to validate fuzzy molecular representations, S-entropy coordinates,
and oscillatory cheminformatics through rigorous experimental protocols.
"""

__version__ = "1.0.0"
__author__ = "Borgia Framework Team"
__email__ = "team@borgia.dev"

# Core validation modules
from . import src

# Visualization modules (lazy import to avoid dependencies)
try:
    from .viz_information_density import InformationDensityVisualizer
    from .viz_s_entropy_coordinates import SEntropyCoordinateVisualizer
    from .viz_bmd_equivalence import BMDEquivalenceVisualizer
    from .viz_spectroscopy_cv import SpectroscopyCVVisualizer
    _VIZ_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Visualization modules not available: {e}")
    _VIZ_AVAILABLE = False

# Master orchestrators (lazy import)
try:
    from .generate_scientific_visualizations import ScientificVisualizationMaster
    _VIZ_MASTER_AVAILABLE = True
except ImportError:
    _VIZ_MASTER_AVAILABLE = False

try:
    from .run_all_validations import MasterValidator as ValidationMaster
    _VALIDATION_MASTER_AVAILABLE = True  
except ImportError:
    _VALIDATION_MASTER_AVAILABLE = False

__all__ = [
    # Core framework
    'src',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    
    # Functions
    'get_validation_summary',
    'list_available_datasets',
    'list_validation_categories', 
    'get_framework_info',
]

# Conditionally add visualization classes if available
if _VIZ_AVAILABLE:
    __all__.extend([
        'InformationDensityVisualizer',
        'SEntropyCoordinateVisualizer', 
        'BMDEquivalenceVisualizer',
        'SpectroscopyCVVisualizer',
    ])

# Conditionally add master orchestrators if available
if _VIZ_MASTER_AVAILABLE:
    __all__.append('ScientificVisualizationMaster')
    
if _VALIDATION_MASTER_AVAILABLE:
    __all__.append('ValidationMaster')

# Validation framework summary
VALIDATION_SUMMARY = {
    'total_scripts': 19,
    'information_scripts': 6,
    'oscillatory_scripts': 6,
    'spectroscopy_scripts': 7,
    'visualization_panels': 64,
    'datasets_included': 5,
    'output_formats': ['PNG', 'PDF', 'JSON'],
    'publication_ready': True
}

def get_validation_summary():
    """Return summary of the validation framework capabilities."""
    return VALIDATION_SUMMARY

def list_available_datasets():
    """List all available SMARTS datasets."""
    datasets = [
        'agrafiotis',
        'ahmed-bajorath', 
        'daylight',
        'hann',
        'walters'
    ]
    return datasets

def list_validation_categories():
    """List all validation categories."""
    categories = {
        'information': [
            'molecular_representation_information_density',
            'meta_information_extraction', 
            'chemical_reaction_prediction',
            'compression_information_retention',
            'dynamic_information_database',
            'situational_utility_analysis'
        ],
        'oscillatory': [
            'st_stellas_entropy_coordinates',
            'bmd_equivalence',
            'dual_functionality', 
            'information_catalysis',
            'strategic_optimization',
            'oscilatory_molecular_architecture'
        ],
        'spectroscopy': [
            'molecule_to_drip',
            'computer_vision_chemical_analysis',
            'led_spectroscopy',
            'hardware_clock_synchronization',
            'noise_enhanced_processing',
            'pixel_chemical_modification',
            'rgb_chemical_mapping'
        ]
    }
    return categories

def get_framework_info():
    """Get comprehensive framework information."""
    return {
        'name': 'Gonfanolier',
        'version': __version__,
        'description': 'Comprehensive Validation Framework for Fuzzy Molecular Representations',
        'validation_summary': get_validation_summary(),
        'available_datasets': list_available_datasets(),
        'validation_categories': list_validation_categories(),
        'citation': {
            'title': 'Gonfanolier: Comprehensive Validation of Fuzzy Molecular Representations and S-Entropy Coordinates in Oscillatory Cheminformatics',
            'authors': 'Borgia Framework Team',
            'year': 2024,
            'journal': 'Journal of Cheminformatics',
            'status': 'In preparation'
        }
    }

# Display banner on import
print(f"""
🎯 Gonfanolier v{__version__} - Comprehensive Validation Framework
=====================================================
✅ {VALIDATION_SUMMARY['total_scripts']} validation scripts loaded
✅ {VALIDATION_SUMMARY['visualization_panels']} visualization panels available
✅ {len(list_available_datasets())} SMARTS datasets ready
✅ Publication-quality outputs enabled

Quick Start:
  - Run all validations: python run_all_validations.py
  - Generate visualizations: python generate_scientific_visualizations.py
  - Framework info: gonfanolier.get_framework_info()
""")
