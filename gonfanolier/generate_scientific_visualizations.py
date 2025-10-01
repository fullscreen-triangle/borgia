#!/usr/bin/env python3
"""
Scientific Visualizations Generator
=================================

Comprehensive visualization system implementing the complete scientific visualization template.
Generates publication-quality plots for all validation results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for publication quality
plt.style.use('default')
sns.set_palette("husl")

class ScientificVisualizationMaster:
    """Master class orchestrating all scientific visualizations"""
    
    def __init__(self, results_dir: str = "gonfanolier/results"):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "scientific_visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Track completion status
        self.completed_tasks = set()
        
        # Load all validation results
        self.results_data = self.load_all_validation_results()
        
        print("🎨 Scientific Visualization Master initialized")
        print(f"📊 Results directory: {self.results_dir}")
        print(f"🖼️ Visualizations output: {self.viz_dir}")
    
    def load_all_validation_results(self) -> Dict[str, Any]:
        """Load all validation results from JSON files"""
        results = {}
        
        # Define expected result files
        result_files = {
            'information_density': 'molecular_representation_information_density_results.json',
            's_entropy_coordinates': 's_entropy_results.json', 
            'bmd_equivalence': 'bmd_catalysis_results.json',
            'meta_information': 'meta_information_results.json',
            'compression_analysis': 'compression_analysis_results.json',
            'dynamic_database': 'dynamic_database_results.json',
            'reaction_prediction': 'reaction_prediction_results.json',
            'situational_utility': 'situational_utility_results.json',
            'dual_functionality': 'dual_functionality_results.json',
            'information_catalysis': 'information_catalysis_results.json',
            'strategic_optimization': 'strategic_optimization_results.json',
            'oscillatory_architecture': 'oscillatory_architecture_results.json',
            'hardware_sync': 'hardware_sync_results.json',
            'led_spectroscopy': 'led_spectroscopy_results.json',
            'noise_enhancement': 'noise_enhancement_results.json',
            'pixel_chemical': 'pixel_chemical_results.json',
            'rgb_chemical': 'rgb_chemical_results.json',
            'spectral_analysis': 'spectral_analysis_results.json',
            'molecule_to_drip': 'molecule_to_drip_results.json',
            'cv_chemical_analysis': 'cv_chemical_insights.json'
        }
        
        for key, filename in result_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        results[key] = json.load(f)
                    print(f"✅ Loaded {key}")
                except Exception as e:
                    print(f"❌ Failed to load {key}: {e}")
                    results[key] = {}
            else:
                print(f"⚠️ Missing {key} results file")
                results[key] = {}
        
        return results
    
    def generate_all_visualizations(self):
        """Generate all scientific visualizations according to template"""
        print("\n🎨 Generating Scientific Visualizations")
        print("=" * 50)
        
        # 1. Information Density Visualizations ✓
        self.generate_information_density_visualizations()
        
        # 2. S-Entropy Coordinate Visualizations ✓  
        self.generate_s_entropy_visualizations()
        
        # 3. BMD Equivalence Validation Plots ✓
        self.generate_bmd_equivalence_plots()
        
        # 4. Spectroscopy and Computer Vision Analysis ✓
        self.generate_spectroscopy_cv_plots()
        
        # Generate summary report
        self.generate_visualization_summary()
        
        print(f"\n🎉 All visualizations generated!")
        print(f"📁 Saved to: {self.viz_dir}")
    
    def mark_completed(self, task: str):
        """Mark a visualization task as completed"""
        self.completed_tasks.add(task)
        print(f"✅ Completed: {task}")

# Import visualization modules
try:
    from viz_information_density import InformationDensityVisualizer
    from viz_s_entropy_coordinates import SEntropyCoordinateVisualizer  
    from viz_bmd_equivalence import BMDEquivalenceVisualizer
    from viz_spectroscopy_cv import SpectroscopyCVVisualizer
except ImportError:
    # Handle relative imports for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from viz_information_density import InformationDensityVisualizer
    from viz_s_entropy_coordinates import SEntropyCoordinateVisualizer  
    from viz_bmd_equivalence import BMDEquivalenceVisualizer
    from viz_spectroscopy_cv import SpectroscopyCVVisualizer

    def generate_information_density_visualizations(self):
        """Generate Information Density Visualizations (Section 1)"""
        print("\n📊 Generating Information Density Visualizations...")
        
        visualizer = InformationDensityVisualizer(self.results_data, self.viz_dir)
        
        # Panel A: Shannon Entropy Comparisons
        visualizer.generate_panel_a_shannon_entropy_comparisons()
        self.mark_completed("1.A - Shannon Entropy Comparisons")
        
        # Panel B: Information Density Heat Maps  
        visualizer.generate_panel_b_information_density_heatmaps()
        self.mark_completed("1.B - Information Density Heat Maps")
        
        # Panel C: Compression Ratio Analysis
        visualizer.generate_panel_c_compression_analysis()
        self.mark_completed("1.C - Compression Ratio Analysis")
        
        # Panel D: Meta-Information Extraction
        visualizer.generate_panel_d_meta_information_extraction()
        self.mark_completed("1.D - Meta-Information Extraction")
    
    def generate_s_entropy_visualizations(self):
        """Generate S-Entropy Coordinate Visualizations (Section 2)"""
        print("\n🌌 Generating S-Entropy Coordinate Visualizations...")
        
        visualizer = SEntropyCoordinateVisualizer(self.results_data, self.viz_dir)
        
        # Panel A: 3D Coordinate Space Mapping
        visualizer.generate_panel_a_3d_coordinate_mapping()
        self.mark_completed("2.A - 3D Coordinate Space Mapping")
        
        # Panel B: Molecular Trajectory Visualization
        visualizer.generate_panel_b_trajectory_visualization()
        self.mark_completed("2.B - Molecular Trajectory Visualization")
        
        # Panel C: Strategic Chess-like Analysis
        visualizer.generate_panel_c_strategic_chess_analysis()
        self.mark_completed("2.C - Strategic Chess-like Analysis")
        
        # Panel D: Coordinate Transformation Animations
        visualizer.generate_panel_d_transformation_animations()
        self.mark_completed("2.D - Coordinate Transformation Animations")
    
    def generate_bmd_equivalence_plots(self):
        """Generate BMD Equivalence Validation Plots (Section 3)"""
        print("\n⚖️ Generating BMD Equivalence Validation Plots...")
        
        visualizer = BMDEquivalenceVisualizer(self.results_data, self.viz_dir)
        
        # Panel A: Cross-Modal Variance Analysis
        visualizer.generate_panel_a_cross_modal_variance()
        self.mark_completed("3.A - Cross-Modal Variance Analysis")
        
        # Panel B: Multi-Pathway Validation
        visualizer.generate_panel_b_multi_pathway_validation()
        self.mark_completed("3.B - Multi-Pathway Validation")
        
        # Panel C: Equivalence Threshold Testing
        visualizer.generate_panel_c_threshold_testing()
        self.mark_completed("3.C - Equivalence Threshold Testing")
        
        # Panel D: Authentication vs Artifact Discrimination
        visualizer.generate_panel_d_authentication_artifacts()
        self.mark_completed("3.D - Authentication vs Artifact Discrimination")
    
    def generate_spectroscopy_cv_plots(self):
        """Generate Spectroscopy and Computer Vision Analysis (Section 4)"""
        print("\n🔬 Generating Spectroscopy & Computer Vision Plots...")
        
        visualizer = SpectroscopyCVVisualizer(self.results_data, self.viz_dir)
        
        # Panel A: Molecule-to-Drip Pattern Visualizations
        visualizer.generate_panel_a_molecule_to_drip()
        self.mark_completed("4.A - Molecule-to-Drip Visualizations")
        
        # Panel B: Computer Vision Classification Performance
        visualizer.generate_panel_b_cv_classification()
        self.mark_completed("4.B - Computer Vision Classification")
        
        # Panel C: Visual-Chemical Information Preservation
        visualizer.generate_panel_c_information_preservation()
        self.mark_completed("4.C - Visual-Chemical Information Preservation")
        
        # Panel D: Pattern Recognition Performance
        visualizer.generate_panel_d_pattern_recognition()
        self.mark_completed("4.D - Pattern Recognition Performance")
    
    def generate_visualization_summary(self):
        """Generate summary of all completed visualizations"""
        summary = {
            'total_visualizations': len(self.completed_tasks),
            'completed_tasks': sorted(list(self.completed_tasks)),
            'results_used': list(self.results_data.keys()),
            'output_directory': str(self.viz_dir)
        }
        
        with open(self.viz_dir / 'visualization_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Visualization Summary:")
        print(f"Total visualizations completed: {summary['total_visualizations']}")
        print(f"Results datasets used: {len(summary['results_used'])}")

def main():
    """Main function to generate all scientific visualizations"""
    print("🎨 Scientific Visualization Generator")
    print("=" * 40)
    
    # Initialize visualization master
    viz_master = ScientificVisualizationMaster()
    
    # Generate all visualizations
    viz_master.generate_all_visualizations()
    
    print("\n🏁 Scientific visualization generation complete!")

if __name__ == "__main__":
    main()
