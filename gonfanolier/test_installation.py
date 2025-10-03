#!/usr/bin/env python3
"""
Installation and Setup Test
===========================

Quick test to verify that the Gonfanolier validation framework
is properly installed and configured.
"""

import os
import sys
import importlib
from pathlib import Path

def test_basic_imports():
    """Test basic Python package imports"""
    print("🔍 Testing basic imports...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'scipy', 'json', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} - MISSING")
    
    return len(missing_packages) == 0, missing_packages

def test_data_availability():
    """Test that SMARTS datasets are available"""
    print("\n🔍 Testing data availability...")
    
    datasets = {
        'agrafiotis': 'public/agrafiotis-smarts-tar/agrafiotis.smarts',
        'ahmed': 'public/ahmed-smarts-tar/ahmed.smarts', 
        'hann': 'public/hann-smarts-tar/hann.smarts',
        'walters': 'public/walters-smarts-tar/walters.smarts'
    }
    
    missing_datasets = []
    for name, path in datasets.items():
        if os.path.exists(path):
            # Check if file has content
            with open(path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"  ✅ {name} ({len(lines)} patterns)")
                else:
                    missing_datasets.append(name)
                    print(f"  ⚠️ {name} - EMPTY")
        else:
            missing_datasets.append(name)
            print(f"  ❌ {name} - MISSING")
    
    return len(missing_datasets) == 0, missing_datasets

def test_validation_scripts():
    """Test that validation scripts are available and importable"""
    print("\n🔍 Testing validation scripts...")
    
    script_paths = {
        'information': [
            'src/information/molecular_representation_information_density.py',
            'src/information/meta_information_extraction.py',
            'src/information/chemical_reaction_prediction.py'
        ],
        'oscillatory': [
            'src/oscillatory/st_stellas_entropy_coordinates.py',
            'src/oscillatory/bmd_equivalence.py',
            'src/oscillatory/dual_functionality.py'
        ],
        'spectroscopy': [
            'src/spectroscopy/molecule_to_drip.py',
            'src/spectroscopy/computer_vision_chemical_analysis.py',
            'src/spectroscopy/led_spectroscopy.py'
        ]
    }
    
    missing_scripts = []
    total_scripts = 0
    
    for category, scripts in script_paths.items():
        print(f"  📁 {category}:")
        for script in scripts:
            total_scripts += 1
            if os.path.exists(script):
                print(f"    ✅ {os.path.basename(script)}")
            else:
                missing_scripts.append(script)
                print(f"    ❌ {os.path.basename(script)} - MISSING")
    
    return len(missing_scripts) == 0, missing_scripts, total_scripts

def test_visualization_modules():
    """Test visualization modules"""
    print("\n🔍 Testing visualization modules...")
    
    viz_modules = [
        'viz_information_density.py',
        'viz_s_entropy_coordinates.py',
        'viz_bmd_equivalence.py',
        'viz_spectroscopy_cv.py'
    ]
    
    missing_viz = []
    for module in viz_modules:
        if os.path.exists(module):
            print(f"  ✅ {module}")
        else:
            missing_viz.append(module)
            print(f"  ❌ {module} - MISSING")
    
    return len(missing_viz) == 0, missing_viz

def test_master_scripts():
    """Test master orchestrator scripts"""
    print("\n🔍 Testing master scripts...")
    
    master_scripts = [
        'run_all_validations.py',
        'generate_scientific_visualizations.py'
    ]
    
    missing_masters = []
    for script in master_scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            missing_masters.append(script)
            print(f"  ❌ {script} - MISSING")
    
    return len(missing_masters) == 0, missing_masters

def test_infrastructure_files():
    """Test infrastructure files"""
    print("\n🔍 Testing infrastructure files...")
    
    infra_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        '__init__.py',
        'MANIFEST.in',
        'pyproject.toml'
    ]
    
    missing_infra = []
    for file in infra_files:
        if os.path.exists(file):
            # Check if file has content
            size = os.path.getsize(file)
            if size > 10:  # At least 10 bytes
                print(f"  ✅ {file} ({size} bytes)")
            else:
                missing_infra.append(file)
                print(f"  ⚠️ {file} - EMPTY")
        else:
            missing_infra.append(file)
            print(f"  ❌ {file} - MISSING")
    
    return len(missing_infra) == 0, missing_infra

def main():
    """Run comprehensive installation test"""
    print("🎯 Gonfanolier Installation Test")
    print("=" * 40)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    test_results = {}
    
    # Run all tests
    test_results['imports'], missing_packages = test_basic_imports()
    test_results['data'], missing_datasets = test_data_availability()
    test_results['scripts'], missing_scripts, total_scripts = test_validation_scripts()
    test_results['visualization'], missing_viz = test_visualization_modules()
    test_results['masters'], missing_masters = test_master_scripts()
    test_results['infrastructure'], missing_infra = test_infrastructure_files()
    
    # Overall results
    all_passed = all(test_results.values())
    
    print("\n" + "=" * 40)
    print("📊 INSTALLATION TEST SUMMARY")
    print("=" * 40)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.capitalize()}: {status}")
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ {total_scripts} validation scripts available")
        print(f"✅ All datasets loaded and ready")
        print(f"✅ All visualization modules available")
        print(f"✅ Framework is ready for use!")
        
        print(f"\nQuick Start:")
        print(f"  python run_all_validations.py")
        print(f"  python generate_scientific_visualizations.py")
        
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        if missing_packages:
            print(f"Missing packages: {missing_packages}")
            print("Run: pip install -r requirements.txt")
        if missing_datasets:
            print(f"Missing datasets: {missing_datasets}")
        if missing_scripts:
            print(f"Missing scripts: {len(missing_scripts)} scripts")
        if missing_viz:
            print(f"Missing visualization: {missing_viz}")
        if missing_masters:
            print(f"Missing master scripts: {missing_masters}")
        if missing_infra:
            print(f"Missing infrastructure: {missing_infra}")
    
    # Save test results
    test_summary = {
        'test_results': test_results,
        'all_passed': all_passed,
        'missing_components': {
            'packages': missing_packages if not test_results['imports'] else [],
            'datasets': missing_datasets if not test_results['data'] else [],
            'scripts': missing_scripts if not test_results['scripts'] else [],
            'visualization': missing_viz if not test_results['visualization'] else [],
            'masters': missing_masters if not test_results['masters'] else [],
            'infrastructure': missing_infra if not test_results['infrastructure'] else []
        },
        'total_scripts_found': total_scripts,
        'timestamp': sys.version,
        'working_directory': os.getcwd()
    }
    
    with open('results/installation_test_results.json', 'w') as f:
        import json
        json.dump(test_summary, f, indent=2)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
