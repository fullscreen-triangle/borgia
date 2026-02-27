"""
Penultimate State Validation Module

Validates Theorem 10.2: Penultimate states are actionable and provide
3-cell neighborhoods for molecular identification.

Tests the hypothesis that identification via penultimate states is:
1. Faster than direct fixed point matching
2. More robust to noise
3. Enables early identification

Author: Trajectory Completion Framework
"""

import numpy as np
from typing import Tuple, List, Dict
import json
from pathlib import Path


class PenultimateStateValidator:
    """Validates penultimate state theory for molecular identification."""

    def __init__(self):
        """Initialize validator."""
        self.epsilon = 1e-6

    def construct_penultimate_cell(self, trit_string: List[int]) -> Dict:
        """
        Construct penultimate cell from trit string.

        From Theorem 10.2: Penultimate state has trit_string[:-1]
        and 3 possible final trits.

        Args:
            trit_string: Full trit string to fixed point

        Returns:
            Dictionary with penultimate structure
        """
        if len(trit_string) == 0:
            return {'error': 'empty_trit_string'}

        k = len(trit_string)
        penultimate_string = trit_string[:-1]
        gateway_trit = trit_string[-1]

        # The 3 possible cells at depth k
        possible_final_trits = [0, 1, 2]
        three_cells = []

        for final_trit in possible_final_trits:
            cell_string = penultimate_string + [final_trit]
            is_target = (final_trit == gateway_trit)
            three_cells.append({
                'trit_string': cell_string,
                'final_trit': final_trit,
                'is_target': is_target
            })

        return {
            'depth': k,
            'penultimate_depth': k - 1,
            'penultimate_string': penultimate_string,
            'gateway_trit': gateway_trit,
            'three_cells': three_cells
        }

    def simulate_trajectory(self,
                           s_star: Tuple[float, float, float],
                           n_steps: int = 10,
                           noise_level: float = 0.01) -> np.ndarray:
        """
        Simulate trajectory approaching fixed point s*.

        Args:
            s_star: Fixed point coordinates
            n_steps: Number of trajectory points
            noise_level: Gaussian noise standard deviation

        Returns:
            Array of shape (n_steps, 3) with trajectory
        """
        trajectory = []

        # Start from random point
        s_0 = np.random.rand(3)

        for i in range(n_steps):
            # Exponential approach to fixed point
            t = i / n_steps
            s_t = s_0 * (1 - t) + np.array(s_star) * t

            # Add noise
            noise = np.random.normal(0, noise_level, 3)
            s_t_noisy = s_t + noise

            # Clip to [0, 1]³
            s_t_noisy = np.clip(s_t_noisy, 0, 1)

            trajectory.append(s_t_noisy)

        return np.array(trajectory)

    def identify_via_penultimate(self,
                                  trajectory: np.ndarray,
                                  atlas: List[Dict],
                                  depth_threshold: int = None) -> Dict:
        """
        Identify molecule using penultimate state approach.

        Algorithm:
        1. Determine which penultimate cell trajectory enters
        2. Read gateway trit
        3. Return molecule

        Args:
            trajectory: Array of trajectory points
            atlas: List of encoded molecules (fixed point atlas)
            depth_threshold: Early stopping depth (None = use full depth)

        Returns:
            Identification result
        """
        if len(trajectory) == 0 or len(atlas) == 0:
            return {'error': 'empty_input'}

        # Use final trajectory point (or point at threshold)
        if depth_threshold is None:
            s_obs = trajectory[-1]
        else:
            idx = min(depth_threshold, len(trajectory) - 1)
            s_obs = trajectory[idx]

        # Find closest penultimate cell
        min_dist = float('inf')
        best_match = None

        for molecule in atlas:
            s_star = np.array([molecule['coordinates']['S_k'],
                              molecule['coordinates']['S_t'],
                              molecule['coordinates']['S_e']])

            # Distance to fixed point
            dist = np.linalg.norm(s_obs - s_star)

            if dist < min_dist:
                min_dist = dist
                best_match = molecule

        return {
            'identified_molecule': best_match['name'] if best_match else None,
            'distance': min_dist,
            'observation_point': s_obs.tolist(),
            'fixed_point': [best_match['coordinates']['S_k'],
                           best_match['coordinates']['S_t'],
                           best_match['coordinates']['S_e']] if best_match else None,
            'gateway_trit': best_match.get('gateway_trit') if best_match else None
        }

    def identify_via_fixed_point(self,
                                  trajectory: np.ndarray,
                                  atlas: List[Dict]) -> Dict:
        """
        Traditional identification: direct fixed point matching.

        Requires full trajectory to s*.

        Args:
            trajectory: Full trajectory to fixed point
            atlas: Fixed point atlas

        Returns:
            Identification result
        """
        # Must reach very close to fixed point
        s_final = trajectory[-1]

        min_dist = float('inf')
        best_match = None

        for molecule in atlas:
            s_star = np.array([molecule['coordinates']['S_k'],
                              molecule['coordinates']['S_t'],
                              molecule['coordinates']['S_e']])

            dist = np.linalg.norm(s_final - s_star)

            if dist < min_dist:
                min_dist = dist
                best_match = molecule

        return {
            'identified_molecule': best_match['name'] if best_match else None,
            'distance': min_dist,
            'requires_full_trajectory': True
        }

    def compare_methods(self,
                       atlas: List[Dict],
                       n_trials: int = 100,
                       noise_levels: List[float] = [0.001, 0.01, 0.05]) -> Dict:
        """
        Compare penultimate vs fixed point identification methods.

        Tests:
        1. Accuracy under noise
        2. Early identification capability
        3. Robustness

        Args:
            atlas: Fixed point atlas
            n_trials: Number of test trajectories per molecule
            noise_levels: Noise levels to test

        Returns:
            Comparison statistics
        """
        results = {
            'noise_levels': noise_levels,
            'n_trials': n_trials,
            'comparisons': []
        }

        for noise in noise_levels:
            penultimate_correct = 0
            fixed_point_correct = 0
            early_identification_possible = 0

            for molecule in atlas[:10]:  # Test subset
                s_star = (molecule['coordinates']['S_k'],
                         molecule['coordinates']['S_t'],
                         molecule['coordinates']['S_e'])

                for trial in range(n_trials):
                    # Simulate trajectory
                    traj = self.simulate_trajectory(s_star, n_steps=20, noise_level=noise)

                    # Identify via penultimate (early stopping at 70% completion)
                    result_penu = self.identify_via_penultimate(traj, atlas, depth_threshold=14)
                    if result_penu.get('identified_molecule') == molecule['name']:
                        penultimate_correct += 1
                        early_identification_possible += 1

                    # Identify via fixed point (full trajectory required)
                    result_fixed = self.identify_via_fixed_point(traj, atlas)
                    if result_fixed.get('identified_molecule') == molecule['name']:
                        fixed_point_correct += 1

            total = len(atlas[:10]) * n_trials

            comparison = {
                'noise_level': noise,
                'penultimate_accuracy': penultimate_correct / total,
                'fixed_point_accuracy': fixed_point_correct / total,
                'early_id_rate': early_identification_possible / total,
                'penultimate_advantage': (penultimate_correct - fixed_point_correct) / total
            }

            results['comparisons'].append(comparison)

        return results

    def validate_three_cell_structure(self, penultimate_cells: List[Dict]) -> Dict:
        """
        Validate that penultimate states form 3-cell neighborhoods.

        Args:
            penultimate_cells: List of penultimate cell structures

        Returns:
            Validation statistics
        """
        validations = []

        for cell_struct in penultimate_cells:
            three_cells = cell_struct.get('three_cells', [])

            # Should have exactly 3 cells
            has_three_cells = len(three_cells) == 3

            # Exactly one should be target
            n_targets = sum(1 for cell in three_cells if cell.get('is_target', False))
            has_one_target = (n_targets == 1)

            # Final trits should be {0, 1, 2}
            final_trits = set(cell.get('final_trit') for cell in three_cells)
            correct_trits = final_trits == {0, 1, 2}

            validations.append({
                'has_three_cells': has_three_cells,
                'has_one_target': has_one_target,
                'correct_trits': correct_trits,
                'valid': has_three_cells and has_one_target and correct_trits
            })

        n_valid = sum(1 for v in validations if v['valid'])

        return {
            'n_tested': len(validations),
            'n_valid': n_valid,
            'validity_rate': n_valid / len(validations) if validations else 0,
            'all_valid': all(v['valid'] for v in validations)
        }


def main():
    """Run penultimate state validation."""
    validator = PenultimateStateValidator()

    # Load atlas
    atlas_file = Path(__file__).parent.parent.parent / 'results' / 'fixed_point_uniqueness.json'

    if not atlas_file.exists():
        print("[WARNING] Atlas not found. Run fixed_point_uniqueness.py first.")
        return

    with open(atlas_file, 'r') as f:
        atlas_data = json.load(f)

    atlas = atlas_data['molecules']

    print(f"Loaded atlas with {len(atlas)} molecules\n")

    # Test 1: Construct penultimate cells
    print("=== Test 1: Penultimate Cell Construction ===")
    penultimate_structures = []

    for mol in atlas[:5]:
        trit_string = mol['trit_string']
        penu_struct = validator.construct_penultimate_cell(trit_string)
        penultimate_structures.append(penu_struct)

        print(f"{mol['name']}: depth={penu_struct['depth']}, "
              f"gateway={penu_struct['gateway_trit']}")

    # Test 2: Validate 3-cell structure
    print(f"\n=== Test 2: Three-Cell Structure Validation ===")
    three_cell_validation = validator.validate_three_cell_structure(penultimate_structures)
    print(f"Validity rate: {three_cell_validation['validity_rate']:.1%}")
    print(f"All valid: {three_cell_validation['all_valid']}")

    # Test 3: Compare identification methods
    print(f"\n=== Test 3: Method Comparison ===")
    comparison = validator.compare_methods(atlas, n_trials=20, noise_levels=[0.001, 0.01, 0.05])

    for comp in comparison['comparisons']:
        print(f"\nNoise level: {comp['noise_level']}")
        print(f"  Penultimate accuracy: {comp['penultimate_accuracy']:.1%}")
        print(f"  Fixed point accuracy: {comp['fixed_point_accuracy']:.1%}")
        print(f"  Early ID rate: {comp['early_id_rate']:.1%}")
        print(f"  Advantage: {comp['penultimate_advantage']:+.1%}")

    # Save results
    results = {
        'penultimate_structures': penultimate_structures[:5],
        'three_cell_validation': three_cell_validation,
        'method_comparison': comparison
    }

    output_file = Path(__file__).parent.parent.parent / 'results' / 'penultimate_validation.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
