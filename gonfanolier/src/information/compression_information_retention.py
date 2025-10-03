#!/usr/bin/env python3
"""
Compression vs Information Retention Analysis
============================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class CompressionAnalyzer:
    def analyze_compression_tradeoffs(self, patterns):
        levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for level in levels:
            compressed = self.compress_patterns(patterns, level)
            
            results[level] = {
                'storage_reduction': self.calc_storage_reduction(patterns, compressed),
                'info_retention': self.calc_info_retention(patterns, compressed),
                'reconstruction_fidelity': self.calc_fidelity(patterns, compressed)
            }
        
        return results
    
    def compress_patterns(self, patterns, level):
        compressed = []
        for pattern in patterns:
            if level >= 0.7:
                comp = ''.join(c for c in pattern if c in 'CNO')[:int(len(pattern)*(1-level))]
            else:
                comp = pattern[:int(len(pattern)*(1-level*0.5))]
            compressed.append(comp if comp else pattern[:1])
        return compressed
    
    def calc_storage_reduction(self, orig, comp):
        orig_size = sum(len(p) for p in orig)
        comp_size = sum(len(p) for p in comp)
        return (orig_size - comp_size) / orig_size * 100 if orig_size > 0 else 0
    
    def calc_info_retention(self, orig, comp):
        orig_chars = set(''.join(orig))
        comp_chars = set(''.join(comp))
        return len(orig_chars.intersection(comp_chars)) / len(orig_chars) * 100 if orig_chars else 0
    
    def calc_fidelity(self, orig, comp):
        fidelities = []
        for o, c in zip(orig, comp):
            fidelity = len(set(o).intersection(set(c))) / len(set(o)) if o else 0
            fidelities.append(fidelity)
        return np.mean(fidelities) * 100

def load_datasets():
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
                        if line.strip() and not line.startswith('#'):
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
    print("🗜️ Compression vs Information Retention")
    print("=" * 45)
    
    datasets = load_datasets()
    analyzer = CompressionAnalyzer()
    
    results = {}
    for name, patterns in datasets.items():
        results[name] = analyzer.analyze_compression_tradeoffs(patterns)
        print(f"{name}: analyzed {len(patterns)} patterns")
    
    # Average results across datasets
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_storage = []
    avg_retention = []
    avg_fidelity = []
    
    for level in levels:
        storage_vals = [results[d][level]['storage_reduction'] for d in results]
        retention_vals = [results[d][level]['info_retention'] for d in results] 
        fidelity_vals = [results[d][level]['reconstruction_fidelity'] for d in results]
        
        avg_storage.append(np.mean(storage_vals))
        avg_retention.append(np.mean(retention_vals))
        avg_fidelity.append(np.mean(fidelity_vals))
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(levels, avg_storage, 'bo-', label='Storage Reduction')
    ax1.plot(levels, [100-r for r in avg_retention], 'ro-', label='Info Loss')
    ax1.set_xlabel('Compression Level')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Storage vs Information Tradeoff')
    ax1.legend()
    ax1.grid(True)
    
    ax2.bar(levels, avg_fidelity, color='green', alpha=0.7)
    ax2.set_xlabel('Compression Level')
    ax2.set_ylabel('Reconstruction Fidelity (%)')
    ax2.set_title('Reconstruction Quality')
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'compression_analysis.png'), dpi=300)
    plt.show()
    
    # Save results
    with open(os.path.join(results_dir, 'compression_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find optimal level (balance storage and retention)
    balance_scores = [s + r for s, r in zip(avg_storage, avg_retention)]
    optimal_idx = np.argmax(balance_scores)
    optimal_level = levels[optimal_idx]
    
    print(f"\n🎯 Results:")
    print(f"Optimal compression level: {optimal_level}")
    print(f"Storage reduction: {avg_storage[optimal_idx]:.1f}%")
    print(f"Information retention: {avg_retention[optimal_idx]:.1f}%")
    print(f"Reconstruction fidelity: {avg_fidelity[optimal_idx]:.1f}%")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()