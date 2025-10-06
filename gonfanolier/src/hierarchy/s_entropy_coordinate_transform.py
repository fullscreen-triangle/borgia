"""
S-Entropy Coordinate Transformation for Molecular Analysis

Converts SMILES/molecular graphs to tri-dimensional S-entropy coordinates
(S_knowledge, S_time, S_entropy) enabling strategic molecular navigation
and chess-like exploration with miracle operations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

class SEntropyCoordinateTransformer:
    """
    Transform molecular representations to S-entropy coordinate space for
    strategic navigation and chess-like molecular exploration.
    """

    def __init__(self):
        self.coordinate_space = {
            'S_knowledge': {'min': 0.0, 'max': 1.0},
            'S_time': {'min': 0.0, 'max': 1.0},
            'S_entropy': {'min': 0.0, 'max': 1.0}
        }
        self.molecular_coordinates = {}
        self.miracle_windows = {}
        self.strategic_positions = {}

    def transform_to_s_entropy(self, smarts_list: List[str]) -> Dict[str, Tuple[float, float, float]]:
        """
        Transform molecular SMARTS to S-entropy coordinates (S_knowledge, S_time, S_entropy).
        """
        coordinates = {}

        for i, smarts in enumerate(smarts_list):
            mol_id = f"mol_{i}"

            # Calculate S_knowledge: Information content measure
            s_knowledge = self._calculate_knowledge_coordinate(smarts)

            # Calculate S_time: Temporal process coordinate
            s_time = self._calculate_time_coordinate(smarts, i)

            # Calculate S_entropy: Disorder/organization measure
            s_entropy = self._calculate_entropy_coordinate(smarts)

            coordinates[mol_id] = (s_knowledge, s_time, s_entropy)

        self.molecular_coordinates = coordinates
        return coordinates

    def _calculate_knowledge_coordinate(self, smarts: str) -> float:
        """Calculate S_knowledge based on information content."""
        # Information entropy of character distribution
        char_counts = Counter(smarts)
        total_chars = len(smarts)

        if total_chars == 0:
            return 0.0

        entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)

        # Normalize to [0, 1] range
        max_entropy = np.log2(min(total_chars, 26))  # Assuming max 26 different characters
        s_knowledge = entropy / max(max_entropy, 1e-6)

        # Add complexity factors
        complexity_factors = {
            'unique_chars': len(char_counts) / max(total_chars, 1),
            'ring_structures': smarts.count('1') + smarts.count('2') + smarts.count('3'),
            'functional_groups': smarts.count('O') + smarts.count('N') + smarts.count('S'),
            'bond_diversity': smarts.count('=') + smarts.count('#') + smarts.count('-')
        }

        complexity_score = sum(complexity_factors.values()) / 4.0
        s_knowledge = 0.7 * s_knowledge + 0.3 * min(complexity_score, 1.0)

        return min(max(s_knowledge, 0.0), 1.0)

    def _calculate_time_coordinate(self, smarts: str, position: int) -> float:
        """Calculate S_time based on temporal dynamics and position."""
        # Sequential position influence
        position_factor = position / 1000.0  # Normalize assuming max 1000 molecules

        # Molecular "age" based on complexity
        complexity = len(smarts) + smarts.count('(') + smarts.count('[')
        age_factor = complexity / 50.0  # Normalize assuming max complexity of 50

        # Temporal patterns in structure
        temporal_patterns = {
            'cyclic_time': (smarts.count('1') + smarts.count('2')) / max(len(smarts), 1),
            'linear_time': smarts.count('C') / max(len(smarts), 1),
            'branching_time': smarts.count('(') / max(len(smarts), 1),
            'aromatic_time': smarts.count(':') / max(len(smarts), 1)
        }

        temporal_score = sum(temporal_patterns.values()) / 4.0

        # Combine factors
        s_time = 0.4 * position_factor + 0.3 * age_factor + 0.3 * temporal_score

        return min(max(s_time, 0.0), 1.0)

    def _calculate_entropy_coordinate(self, smarts: str) -> float:
        """Calculate S_entropy based on disorder/organization measure."""
        if not smarts:
            return 0.0

        # Structural organization measures
        organization_factors = {
            'symmetry': self._calculate_symmetry_score(smarts),
            'regularity': self._calculate_regularity_score(smarts),
            'predictability': self._calculate_predictability_score(smarts),
            'compactness': self._calculate_compactness_score(smarts)
        }

        # Higher organization = lower entropy
        organization_score = sum(organization_factors.values()) / 4.0
        s_entropy = 1.0 - organization_score  # Invert for entropy

        return min(max(s_entropy, 0.0), 1.0)

    def _calculate_symmetry_score(self, smarts: str) -> float:
        """Calculate molecular symmetry score."""
        # Simple symmetry detection based on character patterns
        mid = len(smarts) // 2
        if len(smarts) < 2:
            return 0.0

        # Check for palindromic patterns
        left_half = smarts[:mid]
        right_half = smarts[mid:] if len(smarts) % 2 == 0 else smarts[mid+1:]

        matches = sum(1 for a, b in zip(left_half, right_half[::-1]) if a == b)
        symmetry = matches / max(len(left_half), 1)

        return min(symmetry, 1.0)

    def _calculate_regularity_score(self, smarts: str) -> float:
        """Calculate structural regularity score."""
        # Look for repeating patterns
        pattern_lengths = [2, 3, 4]
        max_regularity = 0.0

        for length in pattern_lengths:
            if len(smarts) >= length * 2:
                patterns = {}
                for i in range(len(smarts) - length + 1):
                    pattern = smarts[i:i+length]
                    patterns[pattern] = patterns.get(pattern, 0) + 1

                if patterns:
                    max_count = max(patterns.values())
                    regularity = max_count / max(len(smarts) - length + 1, 1)
                    max_regularity = max(max_regularity, regularity)

        return min(max_regularity, 1.0)

    def _calculate_predictability_score(self, smarts: str) -> float:
        """Calculate structural predictability score."""
        # Measure how predictable the next character is given previous ones
        if len(smarts) < 2:
            return 0.0

        transitions = {}
        for i in range(len(smarts) - 1):
            current = smarts[i]
            next_char = smarts[i + 1]

            if current not in transitions:
                transitions[current] = {}
            transitions[current][next_char] = transitions[current].get(next_char, 0) + 1

        # Calculate average predictability
        total_predictability = 0.0
        total_transitions = 0

        for current, next_chars in transitions.items():
            total_next = sum(next_chars.values())
            if total_next > 0:
                max_prob = max(next_chars.values()) / total_next
                total_predictability += max_prob
                total_transitions += 1

        if total_transitions == 0:
            return 0.0

        return total_predictability / total_transitions

    def _calculate_compactness_score(self, smarts: str) -> float:
        """Calculate molecular compactness score."""
        # Measure how "compact" the structure is
        if not smarts:
            return 0.0

        # Count structural features that indicate compactness
        compact_features = {
            'rings': smarts.count('1') + smarts.count('2') + smarts.count('3'),
            'aromatic': smarts.count(':'),
            'double_bonds': smarts.count('='),
            'triple_bonds': smarts.count('#')
        }

        # Linear features that reduce compactness
        linear_features = {
            'single_bonds': smarts.count('-'),
            'branches': smarts.count('('),
            'chain_length': len(re.findall(r'C+', smarts))
        }

        compact_score = sum(compact_features.values())
        linear_score = sum(linear_features.values())

        if compact_score + linear_score == 0:
            return 0.0

        compactness = compact_score / (compact_score + linear_score)
        return min(compactness, 1.0)

    def generate_strategic_positions(self) -> Dict[str, Dict]:
        """Generate strategic positions for chess-like molecular exploration."""
        strategic_positions = {}

        for mol_id, (s_k, s_t, s_e) in self.molecular_coordinates.items():
            # Calculate strategic value
            strategic_value = self._calculate_strategic_value(s_k, s_t, s_e)

            # Determine strategic strength
            strategic_strength = self._determine_strategic_strength(strategic_value)

            # Estimate solution metrics
            time_to_solution = self._estimate_time_to_solution(s_k, s_t, s_e)
            entropy_cost = self._calculate_entropy_cost(s_k, s_t, s_e)

            strategic_positions[mol_id] = {
                'coordinates': (s_k, s_t, s_e),
                'strategic_value': strategic_value,
                'strategic_strength': strategic_strength,
                'information_level': s_k,
                'time_to_solution': time_to_solution,
                'entropy_cost': entropy_cost,
                'position_type': self._classify_position_type(s_k, s_t, s_e)
            }

        self.strategic_positions = strategic_positions
        return strategic_positions

    def _calculate_strategic_value(self, s_k: float, s_t: float, s_e: float) -> float:
        """Calculate strategic value of position."""
        # Weighted combination of coordinates
        alpha, beta, gamma = 0.4, 0.35, 0.25

        # Higher knowledge and time are better, lower entropy is better
        value = alpha * s_k + beta * s_t + gamma * (1.0 - s_e)

        return min(max(value, 0.0), 1.0)

    def _determine_strategic_strength(self, strategic_value: float) -> str:
        """Determine strategic strength category."""
        if strategic_value >= 0.8:
            return 'dominant'
        elif strategic_value >= 0.65:
            return 'advantageous'
        elif strategic_value >= 0.45:
            return 'balanced'
        elif strategic_value >= 0.25:
            return 'weak'
        else:
            return 'critical'

    def _estimate_time_to_solution(self, s_k: float, s_t: float, s_e: float) -> float:
        """Estimate time to reach solution from current position."""
        # Higher knowledge and time reduce solution time
        # Higher entropy increases solution time
        base_time = 1.0

        knowledge_factor = 1.0 - s_k  # Less knowledge = more time
        time_factor = 1.0 - s_t      # Less temporal progress = more time
        entropy_factor = s_e         # More entropy = more time

        estimated_time = base_time * (1.0 + knowledge_factor + time_factor + entropy_factor)

        return max(estimated_time, 0.1)  # Minimum time

    def _calculate_entropy_cost(self, s_k: float, s_t: float, s_e: float) -> float:
        """Calculate entropy cost to reach solution."""
        # Cost is related to current entropy and required organization
        target_entropy = 0.2  # Target low entropy for solution

        entropy_gap = max(s_e - target_entropy, 0.0)
        organization_cost = entropy_gap * (2.0 - s_k)  # Less knowledge = higher cost

        return max(organization_cost, 0.1)

    def _classify_position_type(self, s_k: float, s_t: float, s_e: float) -> str:
        """Classify the type of strategic position."""
        if s_k > 0.7 and s_t > 0.7 and s_e < 0.3:
            return 'solution_ready'
        elif s_k > 0.6 and s_e < 0.4:
            return 'knowledge_rich'
        elif s_t > 0.6 and s_e < 0.4:
            return 'time_advanced'
        elif s_e > 0.7:
            return 'chaotic'
        elif s_k < 0.3 and s_t < 0.3:
            return 'early_stage'
        else:
            return 'intermediate'

    def generate_miracle_windows(self) -> Dict[str, List[Dict]]:
        """Generate miracle window operations for strategic navigation."""
        miracle_windows = {}

        miracle_types = ['knowledge', 'time', 'entropy', 'dimensional', 'synthesis']

        for mol_id, position in self.strategic_positions.items():
            s_k, s_t, s_e = position['coordinates']
            miracles = []

            for miracle_type in miracle_types:
                miracle = self._generate_miracle_operation(miracle_type, s_k, s_t, s_e, position)
                miracles.append(miracle)

            miracle_windows[mol_id] = miracles

        self.miracle_windows = miracle_windows
        return miracle_windows

    def _generate_miracle_operation(self, miracle_type: str, s_k: float, s_t: float,
                                  s_e: float, position: Dict) -> Dict:
        """Generate a specific miracle operation."""
        base_strength = 0.3
        base_duration = 5
        base_cost = 0.2

        if miracle_type == 'knowledge':
            # Knowledge breakthrough miracle
            strength = base_strength * (1.0 - s_k)  # More effective when knowledge is low
            target_dimension = 'knowledge'
            transformation = (0.5 * strength, 0, 0)

        elif miracle_type == 'time':
            # Time acceleration miracle
            strength = base_strength * (1.0 - s_t)  # More effective when time is low
            target_dimension = 'time'
            transformation = (0, 0.4 * strength, 0)

        elif miracle_type == 'entropy':
            # Entropy organization miracle
            strength = base_strength * s_e  # More effective when entropy is high
            target_dimension = 'entropy'
            transformation = (0, 0, -0.6 * strength)  # Negative to reduce entropy

        elif miracle_type == 'dimensional':
            # Dimensional shift miracle
            strength = base_strength
            target_dimension = 'all'
            noise = np.random.normal(0, 0.1, 3)
            transformation = tuple(strength * noise)

        else:  # synthesis
            # Synthesis miracle
            strength = base_strength
            target_dimension = 'all'
            transformation = (0.2 * strength, 0.2 * strength, 0.2 * strength)

        return {
            'type': miracle_type,
            'target_dimension': target_dimension,
            'strength': strength,
            'duration': base_duration,
            'cost': base_cost * (1.0 + strength),
            'transformation': transformation,
            'effectiveness': self._calculate_miracle_effectiveness(position, miracle_type)
        }

    def _calculate_miracle_effectiveness(self, position: Dict, miracle_type: str) -> float:
        """Calculate the effectiveness of a miracle operation."""
        strategic_value = position['strategic_value']
        strategic_strength = position['strategic_strength']

        # Base effectiveness depends on current position strength
        strength_multipliers = {
            'critical': 1.5,    # Miracles more effective in critical positions
            'weak': 1.3,
            'balanced': 1.0,
            'advantageous': 0.8,
            'dominant': 0.5     # Miracles less needed in dominant positions
        }

        base_effectiveness = strength_multipliers.get(strategic_strength, 1.0)

        # Miracle-specific effectiveness
        miracle_multipliers = {
            'knowledge': 1.2 if position['information_level'] < 0.5 else 0.8,
            'time': 1.2 if position['time_to_solution'] > 1.0 else 0.8,
            'entropy': 1.2 if position['entropy_cost'] > 0.5 else 0.8,
            'dimensional': 1.0,
            'synthesis': 1.1
        }

        miracle_multiplier = miracle_multipliers.get(miracle_type, 1.0)

        effectiveness = base_effectiveness * miracle_multiplier * (0.5 + strategic_value)

        return min(max(effectiveness, 0.1), 2.0)

    def analyze_coordinate_space(self) -> Dict:
        """Analyze the S-entropy coordinate space distribution and properties."""
        if not self.molecular_coordinates:
            return {'error': 'No coordinates computed'}

        coordinates_array = np.array(list(self.molecular_coordinates.values()))

        analysis = {
            'coordinate_statistics': {
                'S_knowledge': {
                    'mean': float(np.mean(coordinates_array[:, 0])),
                    'std': float(np.std(coordinates_array[:, 0])),
                    'min': float(np.min(coordinates_array[:, 0])),
                    'max': float(np.max(coordinates_array[:, 0]))
                },
                'S_time': {
                    'mean': float(np.mean(coordinates_array[:, 1])),
                    'std': float(np.std(coordinates_array[:, 1])),
                    'min': float(np.min(coordinates_array[:, 1])),
                    'max': float(np.max(coordinates_array[:, 1]))
                },
                'S_entropy': {
                    'mean': float(np.mean(coordinates_array[:, 2])),
                    'std': float(np.std(coordinates_array[:, 2])),
                    'min': float(np.min(coordinates_array[:, 2])),
                    'max': float(np.max(coordinates_array[:, 2]))
                }
            },
            'space_coverage': {
                'volume_occupied': self._calculate_occupied_volume(coordinates_array),
                'density_distribution': self._analyze_density_distribution(coordinates_array),
                'coordinate_correlations': self._calculate_coordinate_correlations(coordinates_array)
            },
            'strategic_analysis': self._analyze_strategic_distribution(),
            'miracle_analysis': self._analyze_miracle_effectiveness()
        }

        return analysis

    def _calculate_occupied_volume(self, coordinates: np.ndarray) -> float:
        """Calculate the volume of coordinate space occupied by molecules."""
        if len(coordinates) == 0:
            return 0.0

        # Calculate bounding box volume
        mins = np.min(coordinates, axis=0)
        maxs = np.max(coordinates, axis=0)
        ranges = maxs - mins

        volume = np.prod(ranges)
        return float(volume)

    def _analyze_density_distribution(self, coordinates: np.ndarray) -> Dict:
        """Analyze the density distribution in coordinate space."""
        if len(coordinates) == 0:
            return {}

        # Divide space into bins and count molecules
        bins = 5  # 5x5x5 grid
        hist, edges = np.histogramdd(coordinates, bins=bins)

        total_bins = bins ** 3
        occupied_bins = np.count_nonzero(hist)

        return {
            'total_bins': total_bins,
            'occupied_bins': int(occupied_bins),
            'occupancy_ratio': float(occupied_bins / total_bins),
            'max_density': float(np.max(hist)),
            'avg_density': float(np.mean(hist[hist > 0])) if occupied_bins > 0 else 0.0
        }

    def _calculate_coordinate_correlations(self, coordinates: np.ndarray) -> Dict:
        """Calculate correlations between S-entropy coordinates."""
        if len(coordinates) < 2:
            return {}

        corr_matrix = np.corrcoef(coordinates.T)

        return {
            'S_knowledge_S_time': float(corr_matrix[0, 1]),
            'S_knowledge_S_entropy': float(corr_matrix[0, 2]),
            'S_time_S_entropy': float(corr_matrix[1, 2])
        }

    def _analyze_strategic_distribution(self) -> Dict:
        """Analyze the distribution of strategic positions."""
        if not self.strategic_positions:
            return {}

        strength_counts = Counter(pos['strategic_strength'] for pos in self.strategic_positions.values())
        position_types = Counter(pos['position_type'] for pos in self.strategic_positions.values())

        strategic_values = [pos['strategic_value'] for pos in self.strategic_positions.values()]

        return {
            'strength_distribution': dict(strength_counts),
            'position_type_distribution': dict(position_types),
            'strategic_value_stats': {
                'mean': float(np.mean(strategic_values)),
                'std': float(np.std(strategic_values)),
                'min': float(np.min(strategic_values)),
                'max': float(np.max(strategic_values))
            }
        }

    def _analyze_miracle_effectiveness(self) -> Dict:
        """Analyze miracle operation effectiveness."""
        if not self.miracle_windows:
            return {}

        all_miracles = []
        for miracles in self.miracle_windows.values():
            all_miracles.extend(miracles)

        if not all_miracles:
            return {}

        effectiveness_by_type = {}
        for miracle in all_miracles:
            miracle_type = miracle['type']
            if miracle_type not in effectiveness_by_type:
                effectiveness_by_type[miracle_type] = []
            effectiveness_by_type[miracle_type].append(miracle['effectiveness'])

        effectiveness_stats = {}
        for miracle_type, values in effectiveness_by_type.items():
            effectiveness_stats[miracle_type] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        return {
            'effectiveness_by_type': effectiveness_stats,
            'total_miracles': len(all_miracles),
            'avg_effectiveness': float(np.mean([m['effectiveness'] for m in all_miracles]))
        }

def load_smarts_datasets():
    """Load SMARTS datasets from the public directory."""
    datasets = {}

    # Find the correct base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')  # Go up to gonfanolier root

    dataset_files = {
        'agrafiotis': os.path.join(base_dir, 'public', 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(base_dir, 'public', 'ahmed-smarts-tar', 'ahmed.smarts'),
        'daylight': os.path.join(base_dir, 'public', 'daylight-smarts-tar', 'daylight.smarts'),
        'hann': os.path.join(base_dir, 'public', 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(base_dir, 'public', 'walters-smarts-tar', 'walters.smarts')
    }

    all_smarts = []

    for name, file_path in dataset_files.items():
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    smarts_list = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                smarts_list.append(parts[0])
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

def create_visualizations(results: Dict, coordinates: Dict, strategic_positions: Dict, output_dir: Path):
    """Create comprehensive visualizations of S-entropy coordinate analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. 3D scatter plot of S-entropy coordinates
    fig = plt.figure(figsize=(20, 15))

    # 3D coordinate space
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    coords_array = np.array(list(coordinates.values()))
    if len(coords_array) > 0:
        ax1.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2],
                   c=coords_array[:, 2], cmap='viridis', alpha=0.6, s=50)
        ax1.set_xlabel('S_knowledge')
        ax1.set_ylabel('S_time')
        ax1.set_zlabel('S_entropy')
        ax1.set_title('S-Entropy Coordinate Space')

    # 2D projections
    ax2 = fig.add_subplot(2, 3, 2)
    if len(coords_array) > 0:
        ax2.scatter(coords_array[:, 0], coords_array[:, 1],
                   c=coords_array[:, 2], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('S_knowledge')
        ax2.set_ylabel('S_time')
        ax2.set_title('S_knowledge vs S_time')
        plt.colorbar(ax2.collections[0], ax=ax2, label='S_entropy')

    ax3 = fig.add_subplot(2, 3, 3)
    if len(coords_array) > 0:
        ax3.scatter(coords_array[:, 0], coords_array[:, 2],
                   c=coords_array[:, 1], cmap='plasma', alpha=0.6)
        ax3.set_xlabel('S_knowledge')
        ax3.set_ylabel('S_entropy')
        ax3.set_title('S_knowledge vs S_entropy')
        plt.colorbar(ax3.collections[0], ax=ax3, label='S_time')

    # Strategic position distribution
    ax4 = fig.add_subplot(2, 3, 4)
    if 'strategic_analysis' in results and 'strength_distribution' in results['strategic_analysis']:
        strength_dist = results['strategic_analysis']['strength_distribution']
        strengths = list(strength_dist.keys())
        counts = list(strength_dist.values())

        ax4.bar(strengths, counts, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('Strategic Strength')
        ax4.set_ylabel('Count')
        ax4.set_title('Strategic Position Distribution')
        ax4.tick_params(axis='x', rotation=45)

    # Coordinate statistics
    ax5 = fig.add_subplot(2, 3, 5)
    if 'coordinate_statistics' in results:
        coord_stats = results['coordinate_statistics']
        dimensions = ['S_knowledge', 'S_time', 'S_entropy']
        means = [coord_stats[dim]['mean'] for dim in dimensions]
        stds = [coord_stats[dim]['std'] for dim in dimensions]

        x = np.arange(len(dimensions))
        ax5.bar(x, means, yerr=stds, capsize=5, color='lightblue', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(dimensions)
        ax5.set_ylabel('Value')
        ax5.set_title('Coordinate Statistics (Mean ± Std)')

    # Miracle effectiveness
    ax6 = fig.add_subplot(2, 3, 6)
    if 'miracle_analysis' in results and 'effectiveness_by_type' in results['miracle_analysis']:
        miracle_stats = results['miracle_analysis']['effectiveness_by_type']
        miracle_types = list(miracle_stats.keys())
        effectiveness = [miracle_stats[mtype]['mean'] for mtype in miracle_types]

        ax6.bar(miracle_types, effectiveness, color='lightgreen', alpha=0.7)
        ax6.set_xlabel('Miracle Type')
        ax6.set_ylabel('Average Effectiveness')
        ax6.set_title('Miracle Operation Effectiveness')
        ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 's_entropy_coordinate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Strategic value heatmap
    if strategic_positions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Strategic value vs coordinates
        strategic_values = [pos['strategic_value'] for pos in strategic_positions.values()]
        coords_for_strategy = [pos['coordinates'] for pos in strategic_positions.values()]

        if coords_for_strategy:
            coords_strategy_array = np.array(coords_for_strategy)

            scatter = ax1.scatter(coords_strategy_array[:, 0], coords_strategy_array[:, 1],
                                c=strategic_values, cmap='RdYlGn', alpha=0.7, s=60)
            ax1.set_xlabel('S_knowledge')
            ax1.set_ylabel('S_time')
            ax1.set_title('Strategic Value Distribution')
            plt.colorbar(scatter, ax=ax1, label='Strategic Value')

            # Position type distribution
            if 'strategic_analysis' in results and 'position_type_distribution' in results['strategic_analysis']:
                pos_types = results['strategic_analysis']['position_type_distribution']
                types = list(pos_types.keys())
                type_counts = list(pos_types.values())

                ax2.pie(type_counts, labels=types, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Position Type Distribution')

        plt.tight_layout()
        plt.savefig(output_dir / 'strategic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def save_results(results: Dict, coordinates: Dict, strategic_positions: Dict,
                miracle_windows: Dict, datasets: Dict, output_dir: Path):
    """Save analysis results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON serialization
    json_results = {
        'coordinate_transformation_summary': {
            'total_molecules': len(coordinates),
            'coordinate_space_volume': results.get('space_coverage', {}).get('volume_occupied', 0),
            'strategic_positions_generated': len(strategic_positions),
            'miracle_windows_created': len(miracle_windows)
        },
        'coordinate_statistics': results.get('coordinate_statistics', {}),
        'space_analysis': results.get('space_coverage', {}),
        'strategic_analysis': results.get('strategic_analysis', {}),
        'miracle_analysis': results.get('miracle_analysis', {}),
        'sample_coordinates': {
            mol_id: coords for mol_id, coords in list(coordinates.items())[:10]
        },
        'sample_strategic_positions': {
            mol_id: {
                'coordinates': pos['coordinates'],
                'strategic_value': pos['strategic_value'],
                'strategic_strength': pos['strategic_strength'],
                'position_type': pos['position_type']
            } for mol_id, pos in list(strategic_positions.items())[:10]
        },
        'dataset_info': {
            name: len(smarts_list) for name, smarts_list in datasets.items()
        }
    }

    # Save results
    with open(output_dir / 's_entropy_coordinate_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✅ Results saved to {output_dir}")

def main():
    """Main execution function for S-entropy coordinate transformation analysis."""
    print("🎯 S-Entropy Coordinate Transformation Analysis")
    print("=" * 50)

    # Load datasets
    print("\n📊 Loading SMARTS datasets...")
    all_smarts, datasets = load_smarts_datasets()
    print(f"✅ Loaded {len(all_smarts)} total SMARTS patterns")

    # Initialize transformer
    print("\n🔧 Initializing S-Entropy Coordinate Transformer...")
    transformer = SEntropyCoordinateTransformer()

    # Transform to S-entropy coordinates
    print("\n🎯 Transforming to S-entropy coordinates...")
    coordinates = transformer.transform_to_s_entropy(all_smarts[:100])  # Sample for efficiency

    # Generate strategic positions
    print("\n♟️ Generating strategic positions...")
    strategic_positions = transformer.generate_strategic_positions()

    # Generate miracle windows
    print("\n✨ Generating miracle windows...")
    miracle_windows = transformer.generate_miracle_windows()

    # Analyze coordinate space
    print("\n📈 Analyzing coordinate space...")
    results = transformer.analyze_coordinate_space()

    # Create output directory
    output_dir = Path("gonfanolier/results/hierarchy")

    # Generate visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results, coordinates, strategic_positions, output_dir)

    # Save results
    print("\n💾 Saving results...")
    save_results(results, coordinates, strategic_positions, miracle_windows, datasets, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("🎯 S-ENTROPY COORDINATE TRANSFORMATION SUMMARY")
    print("=" * 50)
    print(f"🧬 Molecules Transformed: {len(coordinates)}")
    print(f"♟️ Strategic Positions Generated: {len(strategic_positions)}")
    print(f"✨ Miracle Windows Created: {len(miracle_windows)}")

    if 'coordinate_statistics' in results:
        coord_stats = results['coordinate_statistics']
        print(f"📊 S_knowledge Range: [{coord_stats['S_knowledge']['min']:.3f}, {coord_stats['S_knowledge']['max']:.3f}]")
        print(f"⏰ S_time Range: [{coord_stats['S_time']['min']:.3f}, {coord_stats['S_time']['max']:.3f}]")
        print(f"🌀 S_entropy Range: [{coord_stats['S_entropy']['min']:.3f}, {coord_stats['S_entropy']['max']:.3f}]")

    if 'space_coverage' in results:
        space_stats = results['space_coverage']
        print(f"📦 Coordinate Space Volume: {space_stats.get('volume_occupied', 0):.6f}")
        if 'density_distribution' in space_stats:
            density = space_stats['density_distribution']
            print(f"🎯 Space Occupancy: {density.get('occupancy_ratio', 0):.1%}")

    if 'strategic_analysis' in results and 'strategic_value_stats' in results['strategic_analysis']:
        strategy_stats = results['strategic_analysis']['strategic_value_stats']
        print(f"⚡ Average Strategic Value: {strategy_stats['mean']:.3f}")

    if 'miracle_analysis' in results:
        miracle_stats = results['miracle_analysis']
        print(f"✨ Average Miracle Effectiveness: {miracle_stats.get('avg_effectiveness', 0):.3f}")

    print(f"📁 Results saved to: {output_dir}")

    return results

if __name__ == "__main__":
    results = main()
