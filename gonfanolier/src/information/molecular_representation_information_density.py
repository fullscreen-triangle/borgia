#!/usr/bin/env python3
"""
Molecular Representation Information Density Analysis
====================================================

This script quantifies information density across different molecular encodings,
comparing traditional SMARTS/SMILES representations with fuzzy variants.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple, Any

# Add parent directory to path for common utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common_utils import get_base_directory, load_smarts_datasets, ensure_results_directory, safe_divide, save_results

class SMARTSDataLoader:
    """Load and parse SMARTS datasets from University of Hamburg"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets = {}
        
    def load_all_datasets(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load all SMARTS datasets using common utility"""
        raw_datasets = load_smarts_datasets()
        
        # Convert to expected format (pattern, description) tuples
        for name, patterns in raw_datasets.items():
            formatted_patterns = []
            for i, pattern in enumerate(patterns):
                formatted_patterns.append((pattern, f"{name}_pattern_{i}"))
            self.datasets[name] = formatted_patterns
            
        return self.datasets
    
    def _parse_smarts_file(self, filepath: str) -> List[Tuple[str, str]]:
        """Parse SMARTS file and return list of (pattern, identifier) tuples"""
        patterns = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        pattern = parts[0]
                        identifier = parts[1]
                        patterns.append((pattern, identifier))
        return patterns

class InformationDensityAnalyzer:
    """Analyze information density of molecular representations"""
    
    def calculate_shannon_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of a sequence"""
        if not sequence:
            return 0.0
            
        counter = Counter(sequence)
        total = len(sequence)
        
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def measure_representation_information(self, patterns: List[str]) -> Dict[str, float]:
        """Quantify information density across different molecular encodings"""
        if not patterns:
            return {}
            
        combined_sequence = ''.join(patterns)
        
        metrics = {
            'entropy_content': self.calculate_shannon_entropy(combined_sequence),
            'avg_pattern_length': np.mean([len(p) for p in patterns]),
            'pattern_complexity': np.std([len(p) for p in patterns]),
            'unique_characters': len(set(combined_sequence)),
            'pattern_diversity': len(set(patterns)) / len(patterns),
        }
        
        return metrics

def main():
    """Main function to run the analysis"""
    print("🧬 Molecular Representation Information Density Analysis")
    print("=" * 60)
    
    # Initialize components
    loader = SMARTSDataLoader("")  # data_dir not needed with common utils
    analyzer = InformationDensityAnalyzer()
    
    # Load SMARTS datasets
    print("\n📁 Loading SMARTS datasets...")
    datasets = loader.load_all_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Please check data directory.")
        return
    
    # Analyze each dataset
    results = {}
    
    for dataset_name, patterns_list in datasets.items():
        print(f"\n🔍 Analyzing {dataset_name} dataset ({len(patterns_list)} patterns)...")
        
        # Extract patterns
        patterns = [pattern[0] for pattern in patterns_list]
        
        # Analyze information density
        metrics = analyzer.measure_representation_information(patterns)
        results[dataset_name] = metrics
        
        print(f"  Shannon Entropy: {metrics['entropy_content']:.3f} bits")
        print(f"  Avg Pattern Length: {metrics['avg_pattern_length']:.1f}")
        print(f"  Pattern Diversity: {metrics['pattern_diversity']:.3f}")
    
    # Save results
    save_results(results, 'information_density_results.json')
    
    # Create summary
    summary_data = []
    for dataset_name, metrics in results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Entropy_Content': metrics['entropy_content'],
            'Avg_Length': metrics['avg_pattern_length'],
            'Complexity': metrics['pattern_complexity'],
            'Diversity': metrics['pattern_diversity']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('gonfanolier/results/information_density_summary.csv', index=False)
    
    print("\n📋 Analysis Summary:")
    print(summary_df.round(3))
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()