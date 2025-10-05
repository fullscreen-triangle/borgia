"""
Molecular Hierarchy Navigation Through Oscillatory Gear Ratios

Implements O(1) complexity molecular structure navigation using hierarchical
oscillatory systems and gear ratio calculations for direct level jumping.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import os

class MolecularHierarchyNavigator:
    """
    Hierarchical molecular structure navigation using oscillatory gear ratios.
    Achieves O(1) complexity for structural level transitions.
    """

    def __init__(self):
        self.hierarchy_levels = {
            'atom': {'frequency': 1.0, 'level': 1},
            'bond': {'frequency': 2.3, 'level': 2},
            'fragment': {'frequency': 5.7, 'level': 3},
            'functional_group': {'frequency': 12.4, 'level': 4},
            'substructure': {'frequency': 28.9, 'level': 5},
            'molecule': {'frequency': 67.2, 'level': 6}
        }
        self.gear_ratios = {}
        self.observers = {}
        self.navigation_cache = {}

    def compute_gear_ratios(self):
        """Pre-compute all gear ratios for O(1) navigation."""
        levels = list(self.hierarchy_levels.keys())

        for i, level_i in enumerate(levels):
            for j, level_j in enumerate(levels):
                freq_i = self.hierarchy_levels[level_i]['frequency']
                freq_j = self.hierarchy_levels[level_j]['frequency']

                # Gear ratio R_{i→j} = ω_i / ω_j
                ratio = freq_i / freq_j
                self.gear_ratios[(level_i, level_j)] = ratio

        print(f"✅ Computed {len(self.gear_ratios)} gear ratios")

    def initialize_observers(self, smarts_data: List[str]):
        """Initialize finite observers for each hierarchical level."""
        for level in self.hierarchy_levels.keys():
            self.observers[level] = {
                'current_level': level,
                'information_acquired': [],
                'observation_duration': 0.0,
                'processed_molecules': 0
            }

        # Process SMARTS data through observers
        for i, smarts in enumerate(smarts_data[:100]):  # Sample for efficiency
            self._process_molecule_through_observers(smarts, i)

    def _process_molecule_through_observers(self, smarts: str, mol_id: int):
        """Process a molecule through all hierarchical observers."""
        start_time = time.time()

        # Simulate hierarchical analysis
        molecular_features = self._extract_hierarchical_features(smarts)

        for level, features in molecular_features.items():
            if level in self.observers:
                self.observers[level]['information_acquired'].extend(features)
                self.observers[level]['processed_molecules'] += 1

        duration = time.time() - start_time
        for observer in self.observers.values():
            observer['observation_duration'] += duration

    def _extract_hierarchical_features(self, smarts: str) -> Dict[str, List]:
        """Extract features at each hierarchical level."""
        features = {
            'atom': [],
            'bond': [],
            'fragment': [],
            'functional_group': [],
            'substructure': [],
            'molecule': []
        }

        # Atom level - count different atom types
        atom_chars = set(c for c in smarts if c.isalpha() and c.isupper())
        features['atom'] = list(atom_chars)

        # Bond level - identify bond types
        bond_types = []
        if '=' in smarts: bond_types.append('double')
        if '#' in smarts: bond_types.append('triple')
        if '-' in smarts: bond_types.append('single')
        features['bond'] = bond_types

        # Fragment level - ring detection
        if '1' in smarts or '2' in smarts: features['fragment'].append('ring')
        if len(smarts) > 10: features['fragment'].append('chain')

        # Functional group level
        if 'O' in smarts and 'H' in smarts: features['functional_group'].append('hydroxyl')
        if 'C=O' in smarts: features['functional_group'].append('carbonyl')
        if 'N' in smarts: features['functional_group'].append('amino')

        # Substructure level
        if len(features['functional_group']) > 1: features['substructure'].append('multifunctional')
        if len(features['fragment']) > 0: features['substructure'].append('structured')

        # Molecule level
        complexity = len(smarts) + len(features['atom']) + len(features['functional_group'])
        if complexity > 20: features['molecule'].append('complex')
        else: features['molecule'].append('simple')

        return features

    def navigate_direct(self, source_level: str, target_level: str,
                       molecular_state: Dict) -> Dict:
        """
        O(1) direct navigation between hierarchical levels using gear ratios.
        """
        if (source_level, target_level) not in self.gear_ratios:
            raise ValueError(f"No gear ratio computed for {source_level} → {target_level}")

        # O(1) gear ratio lookup
        ratio = self.gear_ratios[(source_level, target_level)]

        # O(1) state transformation
        new_state = molecular_state.copy()
        new_state['level'] = target_level
        new_state['frequency'] = self.hierarchy_levels[target_level]['frequency']
        new_state['transformation_ratio'] = ratio
        new_state['navigation_time'] = time.time()

        return new_state

    def transcendent_navigate(self, source_level: str, target_level: str,
                            molecular_context: Dict) -> Dict:
        """
        Transcendent observer navigation with stochastic fallback.
        """
        try:
            # Attempt direct navigation
            result = self.navigate_direct(source_level, target_level, molecular_context)
            result['navigation_method'] = 'direct'
            return result

        except Exception:
            # Stochastic sampling fallback
            return self._stochastic_navigate(source_level, target_level, molecular_context)

    def _stochastic_navigate(self, source_level: str, target_level: str,
                           molecular_context: Dict) -> Dict:
        """Stochastic navigation for ambiguous scenarios."""
        # Implement constrained random walk
        current_level = source_level
        path = [current_level]

        # Navigate through intermediate levels
        source_idx = self.hierarchy_levels[source_level]['level']
        target_idx = self.hierarchy_levels[target_level]['level']

        step_direction = 1 if target_idx > source_idx else -1

        while current_level != target_level:
            current_idx = self.hierarchy_levels[current_level]['level']
            next_idx = current_idx + step_direction

            # Find level with matching index
            next_level = None
            for level, props in self.hierarchy_levels.items():
                if props['level'] == next_idx:
                    next_level = level
                    break

            if next_level is None:
                break

            current_level = next_level
            path.append(current_level)

        result = molecular_context.copy()
        result['level'] = target_level
        result['navigation_method'] = 'stochastic'
        result['navigation_path'] = path
        result['navigation_time'] = time.time()

        return result

    def analyze_navigation_performance(self, test_molecules: List[str]) -> Dict:
        """Analyze navigation performance across hierarchical levels."""
        results = {
            'direct_navigation_times': [],
            'stochastic_navigation_times': [],
            'gear_ratio_efficiency': {},
            'observer_statistics': {},
            'complexity_analysis': {}
        }

        # Test direct navigation performance
        levels = list(self.hierarchy_levels.keys())

        for i in range(50):  # Performance test iterations
            source = np.random.choice(levels)
            target = np.random.choice(levels)

            molecular_state = {
                'molecule_id': i,
                'level': source,
                'complexity': np.random.randint(5, 25),
                'features': np.random.randint(1, 8)
            }

            # Time direct navigation
            start_time = time.time()
            direct_result = self.navigate_direct(source, target, molecular_state)
            direct_time = time.time() - start_time
            results['direct_navigation_times'].append(direct_time)

            # Time stochastic navigation
            start_time = time.time()
            stochastic_result = self._stochastic_navigate(source, target, molecular_state)
            stochastic_time = time.time() - start_time
            results['stochastic_navigation_times'].append(stochastic_time)

        # Analyze gear ratio efficiency
        for (source, target), ratio in self.gear_ratios.items():
            key = f"{source}→{target}"
            results['gear_ratio_efficiency'][key] = {
                'ratio': ratio,
                'efficiency': 1.0 / max(ratio, 1e-6),  # Inverse for efficiency
                'frequency_difference': abs(
                    self.hierarchy_levels[source]['frequency'] -
                    self.hierarchy_levels[target]['frequency']
                )
            }

        # Observer statistics
        for level, observer in self.observers.items():
            results['observer_statistics'][level] = {
                'molecules_processed': observer['processed_molecules'],
                'total_observation_time': observer['observation_duration'],
                'features_acquired': len(observer['information_acquired']),
                'avg_time_per_molecule': (
                    observer['observation_duration'] / max(observer['processed_molecules'], 1)
                )
            }

        # Complexity analysis
        results['complexity_analysis'] = {
            'avg_direct_navigation_time': np.mean(results['direct_navigation_times']),
            'avg_stochastic_navigation_time': np.mean(results['stochastic_navigation_times']),
            'speedup_factor': (
                np.mean(results['stochastic_navigation_times']) /
                max(np.mean(results['direct_navigation_times']), 1e-9)
            ),
            'total_gear_ratios': len(self.gear_ratios),
            'navigation_efficiency': len(self.gear_ratios) / len(self.hierarchy_levels)**2
        }

        return results

def load_smarts_datasets():
    """Load SMARTS datasets from the public directory."""
    datasets = {}
    base_path = Path("gonfanolier/public")

    dataset_files = {
        'agrafiotis': base_path / "agrafiotis-smarts-tar" / "agrafiotis.smarts",
        'ahmed': base_path / "ahmed-smarts-tar" / "ahmed.smarts",
        'daylight': base_path / "daylight-smarts-tar" / "daylight.smarts",
        'hann': base_path / "hann-smarts-tar" / "hann.smarts",
        'walters': base_path / "walters-smarts-tar" / "walters.smarts"
    }

    all_smarts = []

    for name, file_path in dataset_files.items():
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    smarts_list = [line.strip() for line in f if line.strip()]
                    datasets[name] = smarts_list
                    all_smarts.extend(smarts_list)
                    print(f"✅ Loaded {len(smarts_list)} SMARTS from {name}")
            else:
                print(f"⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    if not all_smarts:
        print("⚠️ No SMARTS files found, generating synthetic data...")
        all_smarts = [
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)O",      # Acetic acid
            "CCO",          # Ethanol
            "C1=CC=C(C=C1)O", # Phenol
            "CC(C)O",       # Isopropanol
            "C1=CC=C2C(=C1)C=CC=C2", # Naphthalene
            "CC(=O)N",      # Acetamide
            "C1=CC=C(C=C1)N", # Aniline
        ] * 20  # Replicate for testing

    return all_smarts, datasets

def create_visualizations(results: Dict, output_dir: Path):
    """Create comprehensive visualizations of navigation performance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Navigation Time Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Navigation times histogram
    ax1.hist(results['direct_navigation_times'], bins=20, alpha=0.7,
             label='Direct Navigation', color='blue')
    ax1.hist(results['stochastic_navigation_times'], bins=20, alpha=0.7,
             label='Stochastic Navigation', color='red')
    ax1.set_xlabel('Navigation Time (seconds)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Navigation Time Distribution')
    ax1.legend()
    ax1.set_yscale('log')

    # Gear ratio efficiency heatmap
    gear_data = []
    levels = ['atom', 'bond', 'fragment', 'functional_group', 'substructure', 'molecule']

    for source in levels:
        row = []
        for target in levels:
            key = f"{source}→{target}"
            if key in results['gear_ratio_efficiency']:
                efficiency = results['gear_ratio_efficiency'][key]['efficiency']
                row.append(efficiency)
            else:
                row.append(0)
        gear_data.append(row)

    im = ax2.imshow(gear_data, cmap='viridis', aspect='auto')
    ax2.set_xticks(range(len(levels)))
    ax2.set_yticks(range(len(levels)))
    ax2.set_xticklabels(levels, rotation=45)
    ax2.set_yticklabels(levels)
    ax2.set_title('Gear Ratio Efficiency Matrix')
    plt.colorbar(im, ax=ax2)

    # Observer performance
    observer_levels = list(results['observer_statistics'].keys())
    molecules_processed = [results['observer_statistics'][level]['molecules_processed']
                          for level in observer_levels]

    ax3.bar(observer_levels, molecules_processed, color='green', alpha=0.7)
    ax3.set_xlabel('Hierarchical Level')
    ax3.set_ylabel('Molecules Processed')
    ax3.set_title('Observer Processing Statistics')
    ax3.tick_params(axis='x', rotation=45)

    # Complexity analysis
    complexity_metrics = ['avg_direct_navigation_time', 'avg_stochastic_navigation_time',
                         'speedup_factor', 'navigation_efficiency']
    complexity_values = [results['complexity_analysis'][metric] for metric in complexity_metrics]

    ax4.bar(range(len(complexity_metrics)), complexity_values, color='orange', alpha=0.7)
    ax4.set_xticks(range(len(complexity_metrics)))
    ax4.set_xticklabels([m.replace('_', '\n') for m in complexity_metrics], rotation=45)
    ax4.set_ylabel('Value')
    ax4.set_title('Complexity Analysis Metrics')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'molecular_hierarchy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detailed gear ratio analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gear ratios by level difference
    level_diffs = []
    ratios = []

    for (source, target), ratio in results['gear_ratio_efficiency'].items():
        source_level = levels.index(source.split('→')[0])
        target_level = levels.index(source.split('→')[1])
        level_diff = abs(target_level - source_level)
        level_diffs.append(level_diff)
        ratios.append(ratio['ratio'])

    ax1.scatter(level_diffs, ratios, alpha=0.6, s=50)
    ax1.set_xlabel('Hierarchical Level Difference')
    ax1.set_ylabel('Gear Ratio')
    ax1.set_title('Gear Ratios vs Level Difference')
    ax1.set_yscale('log')

    # Observer efficiency
    observer_efficiency = []
    observer_names = []

    for level, stats in results['observer_statistics'].items():
        if stats['molecules_processed'] > 0:
            efficiency = stats['features_acquired'] / stats['total_observation_time']
            observer_efficiency.append(efficiency)
            observer_names.append(level)

    ax2.bar(observer_names, observer_efficiency, color='purple', alpha=0.7)
    ax2.set_xlabel('Observer Level')
    ax2.set_ylabel('Features per Second')
    ax2.set_title('Observer Processing Efficiency')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'gear_ratio_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results: Dict, datasets: Dict, output_dir: Path):
    """Save analysis results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON serialization
    json_results = {
        'navigation_performance': {
            'avg_direct_time': float(np.mean(results['direct_navigation_times'])),
            'avg_stochastic_time': float(np.mean(results['stochastic_navigation_times'])),
            'speedup_factor': float(results['complexity_analysis']['speedup_factor']),
            'total_tests': len(results['direct_navigation_times'])
        },
        'gear_ratio_summary': {
            'total_ratios': len(results['gear_ratio_efficiency']),
            'avg_efficiency': float(np.mean([
                gr['efficiency'] for gr in results['gear_ratio_efficiency'].values()
            ])),
            'max_ratio': float(max([
                gr['ratio'] for gr in results['gear_ratio_efficiency'].values()
            ])),
            'min_ratio': float(min([
                gr['ratio'] for gr in results['gear_ratio_efficiency'].values()
            ]))
        },
        'observer_summary': {
            'total_observers': len(results['observer_statistics']),
            'total_molecules_processed': sum([
                obs['molecules_processed'] for obs in results['observer_statistics'].values()
            ]),
            'total_features_acquired': sum([
                obs['features_acquired'] for obs in results['observer_statistics'].values()
            ])
        },
        'dataset_info': {
            name: len(smarts_list) for name, smarts_list in datasets.items()
        },
        'complexity_analysis': results['complexity_analysis']
    }

    # Save results
    with open(output_dir / 'molecular_hierarchy_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✅ Results saved to {output_dir}")

def main():
    """Main execution function for molecular hierarchy navigation analysis."""
    print("🧬 Molecular Hierarchy Navigation Analysis")
    print("=" * 50)

    # Load datasets
    print("\n📊 Loading SMARTS datasets...")
    all_smarts, datasets = load_smarts_datasets()
    print(f"✅ Loaded {len(all_smarts)} total SMARTS patterns")

    # Initialize navigator
    print("\n🔧 Initializing Molecular Hierarchy Navigator...")
    navigator = MolecularHierarchyNavigator()

    # Compute gear ratios
    print("\n⚙️ Computing gear ratios...")
    navigator.compute_gear_ratios()

    # Initialize observers
    print("\n👁️ Initializing hierarchical observers...")
    navigator.initialize_observers(all_smarts)

    # Analyze performance
    print("\n📈 Analyzing navigation performance...")
    results = navigator.analyze_navigation_performance(all_smarts)

    # Create output directory
    output_dir = Path("gonfanolier/results/hierarchy")

    # Generate visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results, output_dir)

    # Save results
    print("\n💾 Saving results...")
    save_results(results, datasets, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("🎯 MOLECULAR HIERARCHY NAVIGATION SUMMARY")
    print("=" * 50)
    print(f"⚡ Average Direct Navigation Time: {results['complexity_analysis']['avg_direct_navigation_time']:.2e} seconds")
    print(f"🔄 Average Stochastic Navigation Time: {results['complexity_analysis']['avg_stochastic_navigation_time']:.2e} seconds")
    print(f"🚀 Speedup Factor: {results['complexity_analysis']['speedup_factor']:.1f}×")
    print(f"⚙️ Total Gear Ratios Computed: {results['complexity_analysis']['total_gear_ratios']}")
    print(f"🎯 Navigation Efficiency: {results['complexity_analysis']['navigation_efficiency']:.3f}")
    print(f"📁 Results saved to: {output_dir}")

    return results

if __name__ == "__main__":
    results = main()
