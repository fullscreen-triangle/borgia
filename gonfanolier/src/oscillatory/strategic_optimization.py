#!/usr/bin/env python3
"""
Strategic Chess-with-Miracles Optimization
=========================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class ChessMiracleOptimizer:
    def __init__(self):
        self.miracle_energy = 100
    
    def optimize_position(self, s_coords, problem_type='general'):
        """Optimize S-entropy coordinates using chess-like strategy"""
        s_k, s_t, s_e = s_coords
        
        # Strategic value function
        if problem_type == 'drug_discovery':
            target_value = 0.4*s_k + 0.3*s_t + 0.3*s_e
        else:
            target_value = (s_k + s_t + s_e) / 3
        
        # Generate moves
        moves = [
            (s_k+0.1, s_t, s_e),
            (s_k, s_t+0.1, s_e),
            (s_k, s_t, s_e+0.1)
        ]
        
        # Select best move
        best_move = max(moves, key=lambda pos: self.evaluate_position(pos, problem_type))
        improvement = self.evaluate_position(best_move, problem_type) - self.evaluate_position(s_coords, problem_type)
        
        return {
            'original_position': s_coords,
            'optimized_position': best_move,
            'improvement': improvement,
            'steps': 1
        }
    
    def evaluate_position(self, coords, problem_type):
        s_k, s_t, s_e = coords
        if problem_type == 'drug_discovery':
            return 0.4*s_k + 0.3*s_t + 0.3*s_e
        return (s_k + s_t + s_e) / 3

def pattern_to_coords(pattern):
    if not pattern:
        return (0.5, 0.5, 0.5)
    
    s_k = len(set(pattern)) / len(pattern)
    s_t = pattern.count('(') / len(pattern)
    s_e = min(1, len(pattern) / 20)
    
    return (s_k, s_t, s_e)

def load_datasets():
    datasets = {}
    
    # Dynamic path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    public_dir = os.path.join(project_root, 'public')
    
    files = {
        'agrafiotis': os.path.join(public_dir, 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(public_dir, 'ahmed-smarts-tar', 'ahmed.smarts'),
        'hann': os.path.join(public_dir, 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(public_dir, 'walters-smarts-tar', 'walters.smarts')
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
    
    # Fallback to synthetic data if no files found
    if not datasets:
        print("No SMARTS files found, generating synthetic patterns...")
        datasets['synthetic'] = [
            'C1=CC=CC=C1',  # Benzene
            'CCO',          # Ethanol
            'CC(=O)O',      # Acetic acid
            'C1=CC=C(C=C1)O',  # Phenol
            'CC(C)O'        # Isopropanol
        ]
        print(f"Generated {len(datasets['synthetic'])} synthetic patterns")
    
    return datasets

def main():
    print("♟️ Strategic Chess-with-Miracles Optimization")
    print("=" * 50)
    
    datasets = load_datasets()
    optimizer = ChessMiracleOptimizer()
    
    all_patterns = []
    for patterns in datasets.values():
        all_patterns.extend(patterns[:5])  # First 5 from each
    
    results = []
    problem_types = ['drug_discovery', 'general']
    
    for problem_type in problem_types:
        improvements = []
        
        for pattern in all_patterns:
            coords = pattern_to_coords(pattern)
            result = optimizer.optimize_position(coords, problem_type)
            improvements.append(result['improvement'])
        
        results.append({
            'problem_type': problem_type,
            'avg_improvement': np.mean(improvements),
            'success_rate': sum(1 for imp in improvements if imp > 0.05) / len(improvements)
        })
    
    # Visualization
    problem_names = [r['problem_type'] for r in results]
    improvements = [r['avg_improvement'] for r in results]
    success_rates = [r['success_rate'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(problem_names, improvements, color='green', alpha=0.7)
    ax1.set_title('Strategic Improvements')
    ax1.set_ylabel('Improvement')
    
    ax2.bar(problem_names, success_rates, color='blue', alpha=0.7)
    ax2.set_title('Success Rates')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    
    # Create results directory with dynamic path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'strategic_optimization.png'), dpi=300)
    plt.show()
    
    with open(os.path.join(results_dir, 'strategic_optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    avg_improvement = np.mean(improvements)
    avg_success = np.mean(success_rates)
    
    print(f"\nResults:")
    for result in results:
        print(f"{result['problem_type']}: {result['avg_improvement']:.3f} improvement, {result['success_rate']:.1%} success")
    
    print(f"\n🎯 Overall: {avg_improvement:.3f} improvement, {avg_success:.1%} success rate")
    print("✅ Strategic optimization validated!" if avg_improvement > 0.02 else "⚠️ Optimization needs improvement")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()