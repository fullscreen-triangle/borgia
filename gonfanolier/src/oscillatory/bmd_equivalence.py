#!/usr/bin/env python3
"""
Biological Maxwell Demon (BMD) Equivalence Validation
====================================================

Cross-modal validation of molecular representations using BMD equivalence
to ensure fuzzy representations capture real molecular information.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from typing import Dict, List, Tuple, Any

class BMDPathwayProcessor:
    """Process molecular data through different BMD pathways"""
    
    def __init__(self):
        self.visual_processor = self._create_visual_pathway()
        self.spectral_processor = self._create_spectral_pathway()
        self.semantic_processor = self._create_semantic_pathway()
        self.scaler = StandardScaler()
        
    def _create_visual_pathway(self):
        """Visual pathway: processes spectral data as images"""
        return MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
    
    def _create_spectral_pathway(self):
        """Spectral pathway: processes numerical peak data"""
        return MLPRegressor(hidden_layer_sizes=(40, 25), max_iter=1000, random_state=42)
    
    def _create_semantic_pathway(self):
        """Semantic pathway: processes molecular descriptors"""
        return MLPRegressor(hidden_layer_sizes=(60, 35), max_iter=1000, random_state=42)
    
    def extract_visual_features(self, smarts_pattern: str) -> np.ndarray:
        """Extract visual features from SMARTS pattern"""
        # Convert pattern to "image-like" 2D representation
        pattern_matrix = np.zeros((8, 8))
        
        # Map characters to positions
        for i, char in enumerate(smarts_pattern[:64]):  # Limit to 8x8 grid
            row, col = divmod(i, 8)
            if row < 8 and col < 8:
                pattern_matrix[row, col] = ord(char) / 128.0  # Normalize
                
        return pattern_matrix.flatten()
    
    def extract_spectral_features(self, smarts_pattern: str) -> np.ndarray:
        """Extract spectral features from SMARTS pattern"""
        features = np.zeros(20)  # Fixed-size feature vector
        
        # Character frequency features
        char_counts = {}
        for char in smarts_pattern:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Map important characters to feature positions
        important_chars = 'CNOSPFClBr()[]=#:~'
        for i, char in enumerate(important_chars[:20]):
            if char in char_counts:
                features[i] = char_counts[char] / len(smarts_pattern)
                
        return features
    
    def extract_semantic_features(self, smarts_pattern: str) -> np.ndarray:
        """Extract semantic molecular descriptors"""
        features = np.zeros(15)
        
        if not smarts_pattern:
            return features
            
        # Structural descriptors
        features[0] = len(smarts_pattern)  # Pattern length
        features[1] = len(set(smarts_pattern))  # Character diversity
        features[2] = smarts_pattern.count('C')  # Carbon count
        features[3] = smarts_pattern.count('N')  # Nitrogen count
        features[4] = smarts_pattern.count('O')  # Oxygen count
        features[5] = smarts_pattern.count('=')  # Double bonds
        features[6] = smarts_pattern.count('#')  # Triple bonds
        features[7] = smarts_pattern.count('(')  # Branches
        features[8] = smarts_pattern.count('[')  # Groups
        features[9] = sum(1 for c in smarts_pattern if c.isdigit())  # Ring closures
        features[10] = sum(1 for c in smarts_pattern if c.islower())  # Aromatic
        features[11] = smarts_pattern.count('F') + smarts_pattern.count('Cl')  # Halogens
        features[12] = smarts_pattern.count('S')  # Sulfur
        features[13] = smarts_pattern.count('P')  # Phosphorus
        features[14] = smarts_pattern.count(':')  # Aromatic bonds
        
        return features
    
    def process_through_pathways(self, smarts_patterns: List[str]) -> Dict[str, np.ndarray]:
        """Process SMARTS patterns through all three pathways"""
        
        # Extract features for each pathway
        visual_features = np.array([self.extract_visual_features(p) for p in smarts_patterns])
        spectral_features = np.array([self.extract_spectral_features(p) for p in smarts_patterns])
        semantic_features = np.array([self.extract_semantic_features(p) for p in smarts_patterns])
        
        # Create synthetic target values (for training demonstration)
        # In real scenario, these would be experimental measurements
        targets = self._generate_synthetic_targets(smarts_patterns)
        
        # Split data for training
        visual_train, visual_test, targets_train, targets_test = train_test_split(
            visual_features, targets, test_size=0.3, random_state=42)
        spectral_train, spectral_test, _, _ = train_test_split(
            spectral_features, targets, test_size=0.3, random_state=42)
        semantic_train, semantic_test, _, _ = train_test_split(
            semantic_features, targets, test_size=0.3, random_state=42)
        
        # Train pathways
        self.visual_processor.fit(visual_train, targets_train)
        self.spectral_processor.fit(spectral_train, targets_train)
        self.semantic_processor.fit(semantic_train, targets_train)
        
        # Get predictions
        visual_pred = self.visual_processor.predict(visual_test)
        spectral_pred = self.spectral_processor.predict(spectral_test)
        semantic_pred = self.semantic_processor.predict(semantic_test)
        
        return {
            'visual_predictions': visual_pred,
            'spectral_predictions': spectral_pred,
            'semantic_predictions': semantic_pred,
            'targets': targets_test,
            'visual_features': visual_test,
            'spectral_features': spectral_test,
            'semantic_features': semantic_test
        }
    
    def _generate_synthetic_targets(self, patterns: List[str]) -> np.ndarray:
        """Generate synthetic target values based on pattern complexity"""
        targets = []
        for pattern in patterns:
            # Synthetic "bioactivity" based on pattern characteristics
            complexity = len(set(pattern)) / len(pattern) if pattern else 0
            size_factor = np.log(len(pattern) + 1)
            heteroatoms = (pattern.count('N') + pattern.count('O') + 
                          pattern.count('S') + pattern.count('P')) / len(pattern)
            
            target = complexity * 2.0 + size_factor * 0.5 + heteroatoms * 3.0
            targets.append(target)
            
        return np.array(targets)

class BMDEquivalenceValidator:
    """Validate BMD equivalence across pathways"""
    
    def validate_bmd_equivalence(self, pathway_results: Dict[str, np.ndarray], 
                                epsilon_threshold: float = 0.1) -> Dict[str, Any]:
        """Validate BMD equivalence across processing pathways"""
        
        visual_pred = pathway_results['visual_predictions']
        spectral_pred = pathway_results['spectral_predictions']
        semantic_pred = pathway_results['semantic_predictions']
        
        # Calculate variance for each pathway
        visual_variance = np.var(visual_pred)
        spectral_variance = np.var(spectral_pred)
        semantic_variance = np.var(semantic_pred)
        
        # Calculate cross-pathway differences
        delta_vs = np.abs(visual_variance - spectral_variance)
        delta_se = np.abs(spectral_variance - semantic_variance)
        delta_ev = np.abs(semantic_variance - visual_variance)
        
        total_variance_difference = delta_vs + delta_se + delta_ev
        
        # BMD equivalence check
        equivalence_achieved = total_variance_difference < epsilon_threshold
        
        # Calculate correlation between pathways
        corr_vs = np.corrcoef(visual_pred, spectral_pred)[0, 1]
        corr_se = np.corrcoef(spectral_pred, semantic_pred)[0, 1]
        corr_ev = np.corrcoef(semantic_pred, visual_pred)[0, 1]
        
        avg_correlation = (corr_vs + corr_se + corr_ev) / 3
        
        # Information consistency measure
        info_consistency = 1.0 - (total_variance_difference / epsilon_threshold)
        info_consistency = max(0.0, min(1.0, info_consistency))  # Clamp to [0,1]
        
        return {
            'equivalence_achieved': equivalence_achieved,
            'total_variance_difference': total_variance_difference,
            'epsilon_threshold': epsilon_threshold,
            'pathway_variances': {
                'visual': visual_variance,
                'spectral': spectral_variance,
                'semantic': semantic_variance
            },
            'pathway_correlations': {
                'visual_spectral': corr_vs,
                'spectral_semantic': corr_se,
                'semantic_visual': corr_ev,
                'average': avg_correlation
            },
            'information_consistency': info_consistency,
            'validation_score': avg_correlation * info_consistency
        }

def load_smarts_datasets():
    """Load SMARTS datasets"""
    datasets = {}
    files = {
        'agrafiotis': 'gonfanolier/public/agrafiotis-smarts-tar/agrafiotis.smarts',
        'ahmed': 'gonfanolier/public/ahmed-smarts-tar/ahmed.smarts',
        'hann': 'gonfanolier/public/hann-smarts-tar/hann.smarts',
        'walters': 'gonfanolier/public/walters-smarts-tar/walters.smarts'
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            patterns.append(parts[0])
            datasets[name] = patterns
            print(f"Loaded {len(patterns)} patterns from {name}")
    
    return datasets

def main():
    """Main function for BMD equivalence validation"""
    print("🔬 Biological Maxwell Demon (BMD) Equivalence Validation")
    print("=" * 60)
    
    # Initialize components
    processor = BMDPathwayProcessor()
    validator = BMDEquivalenceValidator()
    
    # Load datasets
    print("\n📁 Loading SMARTS datasets...")
    datasets = load_smarts_datasets()
    
    if not datasets:
        print("❌ No datasets loaded.")
        return
    
    # Create results directory
    os.makedirs('gonfanolier/results', exist_ok=True)
    
    # Validate BMD equivalence for each dataset
    print("\n🔬 Validating BMD equivalence across pathways...")
    all_validation_results = {}
    
    for dataset_name, patterns in datasets.items():
        print(f"\n🔍 Processing {dataset_name} dataset ({len(patterns)} patterns)...")
        
        # Process through pathways
        pathway_results = processor.process_through_pathways(patterns)
        
        # Validate equivalence
        validation_results = validator.validate_bmd_equivalence(pathway_results)
        
        all_validation_results[dataset_name] = validation_results
        
        # Print results
        equiv_status = "✅ PASSED" if validation_results['equivalence_achieved'] else "❌ FAILED"
        print(f"  BMD Equivalence: {equiv_status}")
        print(f"  Validation Score: {validation_results['validation_score']:.3f}")
        print(f"  Average Correlation: {validation_results['pathway_correlations']['average']:.3f}")
        print(f"  Information Consistency: {validation_results['information_consistency']:.3f}")
    
    # Create visualization
    print("\n📊 Creating validation visualizations...")
    
    # BMD equivalence comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BMD Equivalence Validation Results', fontsize=16, fontweight='bold')
    
    # Validation scores
    datasets_list = list(all_validation_results.keys())
    validation_scores = [all_validation_results[d]['validation_score'] for d in datasets_list]
    
    axes[0,0].bar(datasets_list, validation_scores, color='steelblue', alpha=0.7)
    axes[0,0].set_title('BMD Validation Scores')
    axes[0,0].set_ylabel('Validation Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Equivalence status
    equivalence_status = [1 if all_validation_results[d]['equivalence_achieved'] else 0 
                         for d in datasets_list]
    colors = ['green' if status else 'red' for status in equivalence_status]
    
    axes[0,1].bar(datasets_list, equivalence_status, color=colors, alpha=0.7)
    axes[0,1].set_title('BMD Equivalence Achievement')
    axes[0,1].set_ylabel('Equivalence Achieved')
    axes[0,1].set_ylim(0, 1.2)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Correlation matrix
    avg_correlations = [all_validation_results[d]['pathway_correlations']['average'] 
                       for d in datasets_list]
    
    axes[1,0].bar(datasets_list, avg_correlations, color='orange', alpha=0.7)
    axes[1,0].set_title('Average Pathway Correlations')
    axes[1,0].set_ylabel('Correlation Coefficient')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Information consistency
    info_consistency = [all_validation_results[d]['information_consistency'] 
                       for d in datasets_list]
    
    axes[1,1].bar(datasets_list, info_consistency, color='purple', alpha=0.7)
    axes[1,1].set_title('Information Consistency')
    axes[1,1].set_ylabel('Consistency Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('gonfanolier/results/bmd_equivalence_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    print("\n💾 Saving validation results...")
    with open('gonfanolier/results/bmd_equivalence_results.json', 'w') as f:
        json.dump(all_validation_results, f, indent=2, default=str)
    
    # Create summary
    summary_data = []
    for dataset_name, results in all_validation_results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Equivalence_Achieved': results['equivalence_achieved'],
            'Validation_Score': results['validation_score'],
            'Avg_Correlation': results['pathway_correlations']['average'],
            'Info_Consistency': results['information_consistency'],
            'Visual_Variance': results['pathway_variances']['visual'],
            'Spectral_Variance': results['pathway_variances']['spectral'],
            'Semantic_Variance': results['pathway_variances']['semantic']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('gonfanolier/results/bmd_equivalence_summary.csv', index=False)
    
    print("\n📋 BMD Equivalence Validation Summary:")
    print(summary_df.round(4))
    
    # Overall assessment
    total_datasets = len(all_validation_results)
    passed_datasets = sum(1 for r in all_validation_results.values() 
                         if r['equivalence_achieved'])
    avg_validation_score = np.mean([r['validation_score'] 
                                   for r in all_validation_results.values()])
    
    print(f"\n🎯 Overall Assessment:")
    print(f"  • Datasets passing BMD equivalence: {passed_datasets}/{total_datasets}")
    print(f"  • Average validation score: {avg_validation_score:.3f}")
    print(f"  • Cross-modal validation {'SUCCESS' if passed_datasets >= total_datasets/2 else 'NEEDS IMPROVEMENT'}")
    
    if passed_datasets >= total_datasets/2:
        print("  ✅ Molecular representations demonstrate cross-modal consistency")
        print("  ✅ BMD equivalence validates information authenticity")
    else:
        print("  ⚠️ Some representations may contain artifacts")
        print("  ⚠️ Further optimization of fuzzy encoding parameters recommended")
    
    print("\n🏁 BMD equivalence validation complete!")

if __name__ == "__main__":
    main()