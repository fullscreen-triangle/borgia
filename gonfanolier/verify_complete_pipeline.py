#!/usr/bin/env python3
"""
Complete Pipeline Verification
==============================

Verifies that ALL functions in gonfanolier/src are properly integrated 
and used in the validation and visualization pipeline.
"""

import os
import ast
import importlib.util
from pathlib import Path
import json

class PipelineVerifier:
    """Comprehensive pipeline verification"""
    
    def __init__(self):
        self.src_dir = Path("gonfanolier/src")
        self.analysis = {
            'all_functions': {},
            'used_functions': {},
            'orphaned_functions': {},
            'import_chains': {},
            'validation_coverage': {}
        }
    
    def extract_functions_from_file(self, filepath):
        """Extract all function and class definitions from a Python file"""
        functions = []
        classes = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    # Get methods within classes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            functions.append(f"{node.name}.{item.name}")
            
            return functions, classes
        except Exception as e:
            print(f"❌ Error parsing {filepath}: {e}")
            return [], []
    
    def analyze_all_src_files(self):
        """Analyze all files in src/ directories"""
        print("🔍 Analyzing all src files...")
        
        categories = ['information', 'oscillatory', 'spectroscopy']
        
        for category in categories:
            category_dir = self.src_dir / category
            if not category_dir.exists():
                continue
            
            self.analysis['all_functions'][category] = {}
            
            for py_file in category_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                print(f"  📄 {py_file}")
                functions, classes = self.extract_functions_from_file(py_file)
                
                self.analysis['all_functions'][category][py_file.stem] = {
                    'file_path': str(py_file),
                    'functions': functions,
                    'classes': classes,
                    'total_definitions': len(functions) + len(classes)
                }
    
    def check_validation_script_usage(self):
        """Check which scripts are actually called by run_all_validations.py"""
        print("\n🔍 Checking validation script usage...")
        
        # Read run_all_validations.py to find script list
        validation_file = Path("gonfanolier/run_all_validations.py")
        if not validation_file.exists():
            print("❌ run_all_validations.py not found!")
            return
        
        with open(validation_file, 'r') as f:
            content = f.read()
        
        # Extract validation_scripts list
        used_scripts = []
        in_validation_scripts = False
        
        for line in content.split('\n'):
            if 'validation_scripts = [' in line:
                in_validation_scripts = True
                continue
            elif in_validation_scripts and ']' in line and 'validation_scripts' not in line:
                break
            elif in_validation_scripts and 'gonfanolier/src/' in line:
                # Extract script path
                if "'" in line or '"' in line:
                    path = line.split("'")[1] if "'" in line else line.split('"')[1]
                    if path.startswith('gonfanolier/src/'):
                        script_name = Path(path).stem
                        category = path.split('/')[2]  # information/oscillatory/spectroscopy
                        used_scripts.append((category, script_name))
        
        self.analysis['validation_coverage']['used_scripts'] = used_scripts
        self.analysis['validation_coverage']['total_used'] = len(used_scripts)
        
        # Calculate orphaned scripts
        all_scripts = []
        for category in self.analysis['all_functions']:
            for script in self.analysis['all_functions'][category]:
                all_scripts.append((category, script))
        
        orphaned_scripts = []
        for script in all_scripts:
            if script not in used_scripts:
                orphaned_scripts.append(script)
        
        self.analysis['validation_coverage']['orphaned_scripts'] = orphaned_scripts
        self.analysis['validation_coverage']['total_scripts'] = len(all_scripts)
        self.analysis['validation_coverage']['orphan_percentage'] = len(orphaned_scripts) / len(all_scripts) * 100
        
        print(f"  ✅ Used scripts: {len(used_scripts)}")
        print(f"  ❌ Orphaned scripts: {len(orphaned_scripts)}")
        print(f"  📊 Orphan percentage: {self.analysis['validation_coverage']['orphan_percentage']:.1f}%")
    
    def check_visualization_integration(self):
        """Check if visualization modules use src functions"""
        print("\n🔍 Checking visualization integration...")
        
        viz_files = ['viz_information_density.py', 'viz_s_entropy_coordinates.py', 
                    'viz_bmd_equivalence.py', 'viz_spectroscopy_cv.py']
        
        integration_status = {}
        
        for viz_file in viz_files:
            viz_path = Path(f"gonfanolier/{viz_file}")
            if not viz_path.exists():
                integration_status[viz_file] = {'status': 'missing', 'imports_src': False}
                continue
            
            with open(viz_path, 'r') as f:
                content = f.read()
            
            # Check for imports from src
            imports_src = 'from gonfanolier.src' in content or 'import gonfanolier.src' in content
            uses_synthetic = 'generate_' in content and 'synthetic' in content.lower()
            
            integration_status[viz_file] = {
                'status': 'exists',
                'imports_src': imports_src, 
                'uses_synthetic': uses_synthetic,
                'integration_level': 'integrated' if imports_src else 'isolated'
            }
            
            print(f"  📊 {viz_file}: {integration_status[viz_file]['integration_level']}")
        
        self.analysis['import_chains']['visualization_integration'] = integration_status
    
    def generate_function_usage_report(self):
        """Generate comprehensive function usage report"""
        print("\n📊 Generating function usage report...")
        
        total_functions = 0
        total_classes = 0
        
        # Count all functions and classes
        for category in self.analysis['all_functions']:
            for script in self.analysis['all_functions'][category]:
                script_data = self.analysis['all_functions'][category][script]
                total_functions += len(script_data['functions'])
                total_classes += len(script_data['classes'])
        
        # Calculate usage statistics
        used_scripts = len(self.analysis['validation_coverage']['used_scripts'])
        total_scripts = self.analysis['validation_coverage']['total_scripts']
        orphan_percentage = self.analysis['validation_coverage']['orphan_percentage']
        
        report = {
            'summary': {
                'total_scripts': total_scripts,
                'used_scripts': used_scripts,
                'orphaned_scripts': total_scripts - used_scripts,
                'orphan_percentage': orphan_percentage,
                'total_functions': total_functions,
                'total_classes': total_classes,
                'integration_status': 'INCOMPLETE' if orphan_percentage > 30 else 'COMPLETE'
            },
            'detailed_analysis': self.analysis,
            'critical_issues': []
        }
        
        # Identify critical issues
        if orphan_percentage > 50:
            report['critical_issues'].append(f"CRITICAL: {orphan_percentage:.1f}% of scripts are orphaned")
        
        if not any(viz['imports_src'] for viz in self.analysis['import_chains']['visualization_integration'].values()):
            report['critical_issues'].append("CRITICAL: Visualization modules don't import src functions")
        
        return report
    
    def print_detailed_report(self, report):
        """Print detailed analysis report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE PIPELINE VERIFICATION REPORT")
        print("="*60)
        
        summary = report['summary']
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Total Scripts: {summary['total_scripts']}")
        print(f"Used Scripts: {summary['used_scripts']} ✅")
        print(f"Orphaned Scripts: {summary['orphaned_scripts']} ❌")
        print(f"Orphan Percentage: {summary['orphan_percentage']:.1f}%")
        print(f"Total Functions: {summary['total_functions']}")
        print(f"Total Classes: {summary['total_classes']}")
        print(f"Integration Status: {summary['integration_status']}")
        
        print(f"\n🚨 CRITICAL ISSUES:")
        if report['critical_issues']:
            for issue in report['critical_issues']:
                print(f"  • {issue}")
        else:
            print("  ✅ No critical issues found")
        
        print(f"\n📁 ORPHANED SCRIPTS:")
        orphaned = self.analysis['validation_coverage']['orphaned_scripts']
        for category, script in orphaned:
            script_info = self.analysis['all_functions'][category][script]
            print(f"  ❌ {category}/{script}.py ({script_info['total_definitions']} definitions)")
        
        print(f"\n📊 VISUALIZATION INTEGRATION:")
        for viz_file, status in self.analysis['import_chains']['visualization_integration'].items():
            integration = "✅ INTEGRATED" if status.get('imports_src', False) else "❌ ISOLATED"
            print(f"  {viz_file}: {integration}")
        
        print(f"\n✅ USED SCRIPTS:")
        for category, script in self.analysis['validation_coverage']['used_scripts']:
            script_info = self.analysis['all_functions'][category][script]
            print(f"  ✅ {category}/{script}.py ({script_info['total_definitions']} definitions)")
    
    def save_report(self, report):
        """Save analysis report to file"""
        os.makedirs('gonfanolier/results', exist_ok=True)
        
        with open('gonfanolier/results/pipeline_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to: gonfanolier/results/pipeline_verification_report.json")

def main():
    """Run complete pipeline verification"""
    print("🔍 GONFANOLIER PIPELINE VERIFICATION")
    print("=" * 40)
    
    verifier = PipelineVerifier()
    
    # Run all verification steps
    verifier.analyze_all_src_files()
    verifier.check_validation_script_usage()
    verifier.check_visualization_integration()
    
    # Generate and display report
    report = verifier.generate_function_usage_report()
    verifier.print_detailed_report(report)
    verifier.save_report(report)
    
    # Return status
    is_complete = report['summary']['orphan_percentage'] < 30
    print(f"\n🎯 PIPELINE STATUS: {'COMPLETE' if is_complete else 'INCOMPLETE'}")
    
    return is_complete

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
