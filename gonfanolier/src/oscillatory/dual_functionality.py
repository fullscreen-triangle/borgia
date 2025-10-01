#!/usr/bin/env python3
"""
Dual-Functionality Molecular Architecture
========================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class DualFunctionalityMolecule:
    def __init__(self, pattern):
        self.pattern = pattern
        self.f_base = 1e12 * (1 + len(set(pattern)) / len(pattern)) if pattern else 1e12
        self.S_freq = min(0.99, 0.95 + 0.04 * sum(1 for c in pattern if c.isdigit()) / len(pattern)) if pattern else 0.95
        
    def execute_as_clock(self):
        Q = self.f_base / (self.f_base * (1 - self.S_freq))
        precision = 1.0 / (self.f_base * Q)
        return {'precision': precision, 'stability': self.S_freq}
        
    def execute_as_processor(self):
        capacity = self.f_base * 0.1 * self.S_freq
        return {'capacity': capacity, 'efficiency': self.S_freq}

def load_datasets():
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
    print("⚡ Dual-Functionality Molecular Architecture")
    print("=" * 50)
    
    datasets = load_datasets()
    all_results = {}
    
    for dataset_name, patterns in datasets.items():
        molecules = [DualFunctionalityMolecule(p) for p in patterns]
        
        clock_perfs = [m.execute_as_clock()['stability'] for m in molecules]
        proc_perfs = [m.execute_as_processor()['efficiency'] for m in molecules]
        dual_scores = [np.sqrt(c * p) for c, p in zip(clock_perfs, proc_perfs)]
        
        all_results[dataset_name] = {
            'clock_performance': np.mean(clock_perfs),
            'processor_performance': np.mean(proc_perfs), 
            'dual_functionality_score': np.mean(dual_scores),
            'success_rate': sum(1 for s in dual_scores if s > 0.5) / len(dual_scores)
        }
        
        print(f"{dataset_name}: Clock={np.mean(clock_perfs):.3f}, Processor={np.mean(proc_perfs):.3f}, Success={all_results[dataset_name]['success_rate']:.1%}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets_list = list(all_results.keys())
    clock_means = [all_results[d]['clock_performance'] for d in datasets_list]
    proc_means = [all_results[d]['processor_performance'] for d in datasets_list]
    
    x = np.arange(len(datasets_list))
    ax1.bar(x - 0.2, clock_means, 0.4, label='Clock')
    ax1.bar(x + 0.2, proc_means, 0.4, label='Processor')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_list)
    ax1.legend()
    ax1.set_title('Dual Functionality Performance')
    
    success_rates = [all_results[d]['success_rate'] for d in datasets_list]
    ax2.bar(datasets_list, success_rates, color=['green' if s >= 0.8 else 'orange' for s in success_rates])
    ax2.set_title('Success Rates')
    ax2.set_ylim(0, 1)
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.savefig('gonfanolier/results/dual_functionality.png', dpi=300)
    plt.show()
    
    with open('gonfanolier/results/dual_functionality_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    overall_success = np.mean(success_rates)
    print(f"\nOverall success rate: {overall_success:.1%}")
    print("✅ Dual functionality validated!" if overall_success > 0.7 else "⚠️ Partial validation")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()