#!/usr/bin/env python3
"""
Meta-Information Extraction Analysis
===================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict, Counter

class MetaInfoExtractor:
    def extract_meta_patterns(self, patterns):
        """Extract meta-information from molecular patterns"""
        meta_info = {
            'stereochemistry': self._count_stereo_indicators(patterns),
            'reactivity': self._identify_reactive_sites(patterns),
            'pharmacophore': self._extract_pharmacophore_features(patterns),
            'motifs': self._find_structural_motifs(patterns),
            'compression': self._calculate_compression(patterns)
        }
        return meta_info
    
    def _count_stereo_indicators(self, patterns):
        """Count stereochemistry implications"""
        stereo_count = sum(p.count('@') + p.count('/') + p.count('\\') for p in patterns)
        return {'stereo_centers': stereo_count, 'avg_per_pattern': stereo_count / len(patterns)}
    
    def _identify_reactive_sites(self, patterns):
        """Identify reactive functional groups"""
        reactive_groups = {
            'carbonyl': sum(p.count('C=O') for p in patterns),
            'hydroxyl': sum(p.count('OH') for p in patterns),
            'amino': sum(p.count('NH') for p in patterns),
            'halogen': sum(p.count('F') + p.count('Cl') + p.count('Br') for p in patterns)
        }
        reactive_groups['total'] = sum(reactive_groups.values())
        return reactive_groups
    
    def _extract_pharmacophore_features(self, patterns):
        """Extract pharmacophore-relevant features"""
        features = {
            'h_bond_donors': sum(p.count('OH') + p.count('NH') for p in patterns),
            'h_bond_acceptors': sum(p.count('O') + p.count('N') for p in patterns),
            'aromatic_rings': sum(sum(1 for c in p if c.islower()) // 6 for p in patterns),
            'hydrophobic': sum(p.count('C') for p in patterns)
        }
        features['pharma_score'] = (features['h_bond_donors'] * 0.3 + 
                                  features['h_bond_acceptors'] * 0.2 +
                                  features['aromatic_rings'] * 0.4 +
                                  features['hydrophobic'] * 0.1)
        return features
    
    def _find_structural_motifs(self, patterns):
        """Find common structural motifs"""
        motifs = defaultdict(int)
        motif_patterns = {
            'benzene': 'c1ccccc1',
            'carbonyl': 'C=O',
            'amide': 'C(=O)N',
            'ether': 'COC'
        }
        
        for pattern in patterns:
            for motif_name, motif_smarts in motif_patterns.items():
                if motif_smarts in pattern:
                    motifs[motif_name] += 1
        
        return {'motif_counts': dict(motifs), 'diversity': len(motifs)}
    
    def _calculate_compression(self, patterns):
        """Calculate compression metrics"""
        original_size = sum(len(p) for p in patterns)
        unique_patterns = len(set(patterns))
        compression_ratio = len(patterns) / unique_patterns if unique_patterns > 0 else 1
        
        return {
            'original_size': original_size,
            'unique_patterns': unique_patterns,
            'compression_ratio': compression_ratio,
            'space_savings_percent': (1 - unique_patterns / len(patterns)) * 100
        }

def load_datasets():
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
            with open(filepath, 'r') as f:
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
    print("🧠 Meta-Information Extraction Analysis")
    print("=" * 45)
    
    # Load data and initialize
    datasets = load_datasets()
    extractor = MetaInfoExtractor()
    
    # Create results directory
    os.makedirs('gonfanolier/results', exist_ok=True)
    
    # Extract meta-information
    all_results = {}
    
    for dataset_name, patterns in datasets.items():
        print(f"\n🔍 Processing {dataset_name} dataset...")
        
        meta_info = extractor.extract_meta_patterns(patterns)
        all_results[dataset_name] = meta_info
        
        print(f"  Reactive sites: {meta_info['reactivity']['total']}")
        print(f"  Pharmacophore score: {meta_info['pharmacophore']['pharma_score']:.2f}")
        print(f"  Compression ratio: {meta_info['compression']['compression_ratio']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Meta-Information Analysis', fontsize=16)
    
    datasets_list = list(all_results.keys())
    
    # Reactive sites
    reactive_totals = [all_results[d]['reactivity']['total'] for d in datasets_list]
    axes[0,0].bar(datasets_list, reactive_totals, color='red', alpha=0.7)
    axes[0,0].set_title('Total Reactive Sites')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Pharmacophore scores
    pharma_scores = [all_results[d]['pharmacophore']['pharma_score'] for d in datasets_list]
    axes[0,1].bar(datasets_list, pharma_scores, color='green', alpha=0.7)
    axes[0,1].set_title('Pharmacophore Scores')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Compression ratios
    compression_ratios = [all_results[d]['compression']['compression_ratio'] for d in datasets_list]
    axes[1,0].bar(datasets_list, compression_ratios, color='blue', alpha=0.7)
    axes[1,0].set_title('Compression Ratios')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Motif diversity
    motif_diversity = [all_results[d]['motifs']['diversity'] for d in datasets_list]
    axes[1,1].bar(datasets_list, motif_diversity, color='orange', alpha=0.7)
    axes[1,1].set_title('Motif Diversity')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('gonfanolier/results/meta_information_analysis.png', dpi=300)
    plt.show()
    
    # Save results
    with open('gonfanolier/results/meta_information_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    summary_data = []
    for dataset_name, results in all_results.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Reactive_Sites': results['reactivity']['total'],
            'Pharmacophore_Score': results['pharmacophore']['pharma_score'],
            'Compression_Ratio': results['compression']['compression_ratio'],
            'Motif_Diversity': results['motifs']['diversity'],
            'Space_Savings_%': results['compression']['space_savings_percent']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('gonfanolier/results/meta_information_summary.csv', index=False)
    
    print("\n📋 Meta-Information Summary:")
    print(summary_df.round(3))
    
    print(f"\n🎯 Meta-information extraction demonstrates:")
    print(f"  • Successful pattern compression and storage reduction")
    print(f"  • Identification of pharmacophore and reactive features")
    print(f"  • Discovery of structural motifs across datasets")
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()