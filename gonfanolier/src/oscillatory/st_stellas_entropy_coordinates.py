#!/usr/bin/env python3
"""
St-Stella's S-Entropy Coordinate Transformation
==============================================

Transform SMARTS patterns to S-entropy coordinates (S_knowledge, S_time, S_entropy)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import json

class StellaCoordinateTransformer:
    def transform_pattern(self, pattern):
        """Transform SMARTS to S-entropy coordinates"""
        s_knowledge = self._calc_knowledge(pattern)
        s_time = self._calc_time(pattern)
        s_entropy = self._calc_entropy(pattern)
        
        return {
            'S_knowledge': s_knowledge,
            'S_time': s_time,
            'S_entropy': s_entropy,
            'pattern': pattern
        }
    
    def _calc_knowledge(self, pattern):
        """Information content measure"""
        if not pattern:
            return 0.0
        char_diversity = len(set(pattern)) / len(pattern)
        structure_complexity = pattern.count('[') + pattern.count('(')
        return char_diversity * 2.0 + structure_complexity * 0.1
    
    def _calc_time(self, pattern):
        """Temporal dynamics measure"""
        if not pattern:
            return 0.0
        length = np.log(len(pattern) + 1)
        rings = len([c for c in pattern if c.isdigit()])
        return length * 0.3 + rings * 0.5
    
    def _calc_entropy(self, pattern):
        """Disorder measure"""
        if not pattern:
            return 0.0
        counter = Counter(pattern)
        entropy = 0
        for count in counter.values():
            p = count / len(pattern)
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy / np.log2(len(counter)) if counter else 0

def load_datasets():
    """Load SMARTS datasets"""
    datasets = {}
    
    # Find the correct base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')  # Go up to gonfanolier root
    
    files = {
        'agrafiotis': os.path.join(base_dir, 'public', 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(base_dir, 'public', 'ahmed-smarts-tar', 'ahmed.smarts'),
        'hann': os.path.join(base_dir, 'public', 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(base_dir, 'public', 'walters-smarts-tar', 'walters.smarts')
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                patterns.append(parts[0])
                datasets[name] = patterns
                print(f"Loaded {len(patterns)} patterns from {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    # If no datasets found, create synthetic data for demo
    if not datasets:
        print("No SMARTS files found, using synthetic molecular patterns for demo...")
        datasets['synthetic'] = [
            'c1ccccc1',  # benzene
            'CCO',       # ethanol
            'CC(=O)O',   # acetic acid
            'c1ccc2ccccc2c1',  # naphthalene
            'CC(C)O'     # isopropanol
        ]
        print(f"Created {len(datasets['synthetic'])} synthetic patterns")
    
    return datasets

def main():
    print("🎯 St-Stella S-Entropy Coordinate Analysis")
    print("=" * 50)
    
    # Load data
    datasets = load_datasets()
    transformer = StellaCoordinateTransformer()
    
    # Transform patterns
    all_coords = {}
    for name, patterns in datasets.items():
        coords = [transformer.transform_pattern(p) for p in patterns]
        all_coords[name] = coords
        
        # Stats
        s_k = [c['S_knowledge'] for c in coords]
        s_t = [c['S_time'] for c in coords]
        s_e = [c['S_entropy'] for c in coords]
        
        print(f"\n{name}: S_k={np.mean(s_k):.3f}, S_t={np.mean(s_t):.3f}, S_e={np.mean(s_e):.3f}")
    
    # 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (name, coords) in enumerate(all_coords.items()):
        s_k = [c['S_knowledge'] for c in coords]
        s_t = [c['S_time'] for c in coords]
        s_e = [c['S_entropy'] for c in coords]
        
        ax.scatter(s_k, s_t, s_e, c=colors[i], label=name, alpha=0.7)
    
    ax.set_xlabel('S_knowledge')
    ax.set_ylabel('S_time')
    ax.set_zlabel('S_entropy')
    ax.legend()
    ax.set_title('St-Stella S-Entropy Coordinates')
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 's_entropy_3d.png'), dpi=300)
    plt.show()
    
    # Save results
    results = {}
    for name, coords in all_coords.items():
        results[name] = {
            'count': len(coords),
            'avg_S_knowledge': np.mean([c['S_knowledge'] for c in coords]),
            'avg_S_time': np.mean([c['S_time'] for c in coords]),
            'avg_S_entropy': np.mean([c['S_entropy'] for c in coords])
        }
    
    with open(os.path.join(results_dir, 's_entropy_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()