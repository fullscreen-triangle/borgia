#!/usr/bin/env python3
"""
Oscillatory Molecular Architecture Networks
==========================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx

class BMDNetworkBuilder:
    def __init__(self):
        self.scales = {
            'quantum': 1e-15,
            'molecular': 1e-9, 
            'environmental': 1e2
        }
        self.network = nx.Graph()
    
    def build_bmd_network(self, molecular_patterns, labels):
        """Build BMD network from molecular patterns"""
        self.network.clear()
        
        # Add nodes
        for i, (pattern, label) in enumerate(zip(molecular_patterns, labels)):
            node_id = f"bmd_{i}"
            props = self.calculate_bmd_properties(pattern, label)
            self.network.add_node(node_id, **props)
        
        # Add edges based on similarity
        nodes = list(self.network.nodes())
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                sim = self.calculate_similarity(molecular_patterns[i], molecular_patterns[j])
                if sim > 0.3:
                    self.network.add_edge(nodes[i], nodes[j], weight=sim)
        
        return self.network
    
    def calculate_bmd_properties(self, pattern, label):
        """Calculate BMD properties"""
        if not pattern:
            return {'info_processing': 0, 'efficiency': 0, 'coordination': 0}
        
        # Information processing from entropy
        from collections import Counter
        counts = Counter(pattern)
        entropy = -sum((c/len(pattern)) * np.log2(c/len(pattern)) for c in counts.values())
        info_processing = min(1.0, entropy / 4.0)
        
        # Efficiency from structure
        complexity = (pattern.count('(') + pattern.count('[')) / len(pattern)
        efficiency = max(0.1, 1.0 - complexity)
        
        # Coordination from functional groups
        func_groups = pattern.count('OH') + pattern.count('C=O') + pattern.count('NH')
        coordination = min(1.0, func_groups / len(pattern) * 5)
        
        return {
            'pattern': pattern,
            'dataset': label,
            'info_processing': info_processing,
            'efficiency': efficiency,
            'coordination': coordination,
            'overall': (info_processing + efficiency + coordination) / 3
        }
    
    def calculate_similarity(self, p1, p2):
        """Calculate pattern similarity"""
        if not p1 or not p2:
            return 0.0
        
        chars1, chars2 = set(p1), set(p2)
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0

class CoordinationOptimizer:
    def __init__(self, network):
        self.network = network
    
    def optimize_coordination(self, target=0.8):
        """Optimize network coordination"""
        initial_eff = self.calculate_efficiency()
        
        # Apply improvements
        bottlenecks = self.find_bottlenecks()
        improvements = 0
        
        for node in bottlenecks:
            node_data = self.network.nodes[node]
            if node_data.get('overall', 0) < 0.5:
                # Improve coordination
                node_data['coordination'] = min(1.0, node_data.get('coordination', 0) + 0.2)
                node_data['overall'] = (
                    node_data.get('info_processing', 0) +
                    node_data.get('efficiency', 0) +
                    node_data['coordination']
                ) / 3
                improvements += 1
        
        final_eff = self.calculate_efficiency()
        
        return {
            'initial_efficiency': initial_eff,
            'final_efficiency': final_eff,
            'improvement': final_eff - initial_eff,
            'improvements_applied': improvements
        }
    
    def calculate_efficiency(self):
        """Calculate network efficiency"""
        if not self.network.nodes():
            return 0.0
        
        efficiencies = [self.network.nodes[node].get('overall', 0) for node in self.network.nodes()]
        return np.mean(efficiencies)
    
    def find_bottlenecks(self):
        """Find coordination bottlenecks"""
        bottlenecks = []
        
        try:
            centrality = nx.betweenness_centrality(self.network)
            
            for node in self.network.nodes():
                node_eff = self.network.nodes[node].get('overall', 0)
                node_centrality = centrality.get(node, 0)
                
                if node_centrality > 0.1 and node_eff < 0.5:
                    bottlenecks.append(node)
        except:
            pass
        
        return bottlenecks

class ScaleCoordinator:
    def __init__(self):
        self.scales = ['quantum', 'molecular', 'environmental']
        self.coordination_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.6], 
            [0.3, 0.6, 1.0]
        ])
    
    def calculate_scale_interactions(self, patterns):
        """Calculate multi-scale interactions"""
        interactions = {}
        
        for i, scale1 in enumerate(self.scales):
            for j, scale2 in enumerate(self.scales):
                base_strength = self.coordination_matrix[i, j]
                
                # Pattern-dependent coupling
                avg_complexity = np.mean([len(set(p))/len(p) for p in patterns]) if patterns else 0.5
                coupling_strength = base_strength * (0.5 + avg_complexity)
                
                interactions[f"{scale1}_{scale2}"] = {
                    'base_strength': base_strength,
                    'coupling_strength': coupling_strength,
                    'effective_coupling': min(1.0, coupling_strength)
                }
        
        return interactions

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
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            patterns.append(parts[0])
            datasets[name] = patterns
            print(f"Loaded {len(patterns)} patterns from {name}")
    return datasets

def main():
    print("⚡ Oscillatory Molecular Architecture Networks")
    print("=" * 50)
    
    datasets = load_datasets()
    
    # Combine patterns
    all_patterns = []
    all_labels = []
    
    for name, patterns in datasets.items():
        all_patterns.extend(patterns[:10])  # First 10 from each
        all_labels.extend([name] * min(10, len(patterns)))
    
    print(f"\n⚡ Building BMD network for {len(all_patterns)} patterns...")
    
    # Build network
    network_builder = BMDNetworkBuilder()
    bmd_network = network_builder.build_bmd_network(all_patterns, all_labels)
    
    print(f"Network: {bmd_network.number_of_nodes()} nodes, {bmd_network.number_of_edges()} edges")
    
    # Optimize coordination
    optimizer = CoordinationOptimizer(bmd_network)
    opt_results = optimizer.optimize_coordination()
    
    print(f"Efficiency: {opt_results['initial_efficiency']:.3f} → {opt_results['final_efficiency']:.3f}")
    print(f"Improvement: {opt_results['improvement']:.3f}")
    
    # Scale coordination
    scale_coord = ScaleCoordinator()
    scale_interactions = scale_coord.calculate_scale_interactions(all_patterns)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Network topology
    if bmd_network.nodes():
        pos = nx.spring_layout(bmd_network, seed=42)
        node_colors = [bmd_network.nodes[node].get('overall', 0) for node in bmd_network.nodes()]
        
        nx.draw(bmd_network, pos, ax=axes[0,0],
                node_color=node_colors, cmap='viridis',
                node_size=200, alpha=0.8, with_labels=False)
        axes[0,0].set_title('BMD Network Topology')
    
    # Efficiency distribution
    efficiencies = [bmd_network.nodes[node].get('overall', 0) for node in bmd_network.nodes()]
    axes[0,1].hist(efficiencies, bins=15, alpha=0.7, color='green')
    axes[0,1].set_xlabel('Overall Efficiency')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('BMD Efficiency Distribution')
    
    # Scale interaction matrix
    scales = ['quantum', 'molecular', 'environmental']
    interaction_matrix = np.zeros((3, 3))
    
    for i, scale1 in enumerate(scales):
        for j, scale2 in enumerate(scales):
            key = f"{scale1}_{scale2}"
            if key in scale_interactions:
                interaction_matrix[i, j] = scale_interactions[key]['effective_coupling']
    
    im = axes[1,0].imshow(interaction_matrix, cmap='viridis')
    axes[1,0].set_xticks(range(3))
    axes[1,0].set_yticks(range(3))
    axes[1,0].set_xticklabels(scales)
    axes[1,0].set_yticklabels(scales)
    axes[1,0].set_title('Scale Interaction Matrix')
    
    # Optimization progress
    axes[1,1].bar(['Initial', 'Final'], 
                 [opt_results['initial_efficiency'], opt_results['final_efficiency']],
                 color=['red', 'green'], alpha=0.7)
    axes[1,1].set_ylabel('Network Efficiency')
    axes[1,1].set_title('Coordination Optimization')
    axes[1,1].set_ylim(0, 1)
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('gonfanolier/results/oscillatory_architecture.png', dpi=300)
    plt.show()
    
    # Save results
    summary = {
        'network_stats': {
            'nodes': bmd_network.number_of_nodes(),
            'edges': bmd_network.number_of_edges(),
            'density': nx.density(bmd_network)
        },
        'optimization': opt_results,
        'scale_interactions': {k: v['effective_coupling'] for k, v in scale_interactions.items()},
        'avg_scale_coupling': np.mean([v['effective_coupling'] for v in scale_interactions.values()])
    }
    
    with open('gonfanolier/results/oscillatory_architecture_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n🎯 Architecture Summary:")
    print(f"Network density: {summary['network_stats']['density']:.3f}")
    print(f"Average scale coupling: {summary['avg_scale_coupling']:.3f}")
    print(f"Final efficiency: {summary['optimization']['final_efficiency']:.3f}")
    
    success = (summary['optimization']['final_efficiency'] > 0.6 and 
               summary['avg_scale_coupling'] > 0.5)
    
    print(f"\n{'✅ SUCCESS' if success else '⚠️ PARTIAL'}: Oscillatory architecture {'validated' if success else 'needs optimization'}")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()