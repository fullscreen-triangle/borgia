"""
Ternary Trajectory Validation for ESDVS

Validates ternary state trajectory completion for electronic states.
Analyzes fidelity and convergence over emission lifetime.

Author: Emission-Strobed Dual-Mode Spectroscopy Framework
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List


class TernaryTrajectoryValidator:
    """Validates ternary state trajectory reconstruction."""

    def __init__(self, data_file: str = None):
        """Initialize with ESDVS results data."""
        if data_file is None:
            data_file = Path(__file__).parent.parent.parent.parent / 'esdvs_results.json'

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.molecule = self.data['molecule']

    def extract_trajectory_data(self) -> Dict:
        """Extract ternary trajectory data."""
        traj = self.data['ternary_trajectory']

        return {
            'n_points': traj['n_points'],
            'fidelity_at_tau_em': traj['fidelity_at_tau_em'],
            'average_fidelity': traj['average_fidelity']
        }

    def simulate_ternary_evolution(self, tau_em: float = 850e-12,
                                   tau_vib: float = 100e-12,
                                   n_points: int = 100) -> Dict:
        """
        Simulate ternary state evolution.

        States: |0> = ground, |1> = natural, |2> = excited

        Rate equations:
        dc2/dt = -c2/tau_em - c2/tau_vib
        dc1/dt = c2/tau_vib - c1/tau_vib
        dc0/dt = c2/tau_em + c1/tau_vib

        Args:
            tau_em: Emission lifetime (s)
            tau_vib: Vibrational relaxation time (s)
            n_points: Number of time points

        Returns:
            Trajectory dict with amplitudes
        """
        # Time points
        t = np.linspace(0, 3 * tau_em, n_points)

        # Initialize: pure excited state
        c0 = np.zeros(n_points)
        c1 = np.zeros(n_points)
        c2 = np.zeros(n_points)

        c2[0] = 1.0  # Start in excited state

        # Solve coupled ODEs
        dt = t[1] - t[0]

        for i in range(1, n_points):
            # Rates
            dc2_dt = -c2[i-1]/tau_em - c2[i-1]/tau_vib
            dc1_dt = c2[i-1]/tau_vib - c1[i-1]/tau_vib
            dc0_dt = c2[i-1]/tau_em + c1[i-1]/tau_vib

            # Euler step
            c2[i] = c2[i-1] + dc2_dt * dt
            c1[i] = c1[i-1] + dc1_dt * dt
            c0[i] = c0[i-1] + dc0_dt * dt

            # Normalize
            norm = c0[i] + c1[i] + c2[i]
            c0[i] /= norm
            c1[i] /= norm
            c2[i] /= norm

        return {
            'time': t.tolist(),
            'c0': c0.tolist(),
            'c1': c1.tolist(),
            'c2': c2.tolist(),
            'tau_em': tau_em,
            'tau_vib': tau_vib
        }

    def compute_fidelity(self, c_measured: np.ndarray, c_theory: np.ndarray) -> float:
        """
        Compute state fidelity.

        F = |<psi_measured|psi_theory>|^2

        Args:
            c_measured: Measured amplitudes [c0, c1, c2]
            c_theory: Theoretical amplitudes [c0, c1, c2]

        Returns:
            Fidelity (0 to 1)
        """
        overlap = np.abs(np.sum(c_measured * c_theory))
        return overlap ** 2

    def analyze_trajectory_quality(self) -> Dict:
        """Analyze trajectory reconstruction quality."""
        traj_data = self.extract_trajectory_data()

        # Thresholds
        high_fidelity = 0.95
        excellent_fidelity = 0.98

        avg_fid = traj_data['average_fidelity']
        tau_fid = traj_data['fidelity_at_tau_em']

        quality_avg = 'excellent' if avg_fid > excellent_fidelity else ('high' if avg_fid > high_fidelity else 'moderate')
        quality_tau = 'excellent' if tau_fid > excellent_fidelity else ('high' if tau_fid > high_fidelity else 'moderate')

        return {
            'average_fidelity': avg_fid,
            'fidelity_at_tau_em': tau_fid,
            'quality_average': quality_avg,
            'quality_at_emission': quality_tau,
            'meets_high_threshold': avg_fid > high_fidelity,
            'meets_excellent_threshold': avg_fid > excellent_fidelity,
            'fidelity_drop_at_emission': avg_fid - tau_fid
        }

    def comprehensive_validation(self) -> Dict:
        """Run comprehensive ternary trajectory validation."""
        traj_data = self.extract_trajectory_data()
        simulated = self.simulate_ternary_evolution(n_points=traj_data['n_points'])
        quality = self.analyze_trajectory_quality()

        return {
            'molecule': self.molecule,
            'trajectory_data': traj_data,
            'simulated_trajectory': simulated,
            'quality_analysis': quality,
            'validation_pass': quality['meets_high_threshold']
        }


def main():
    """Run ternary trajectory validation."""
    validator = TernaryTrajectoryValidator()

    print(f"\n=== Ternary Trajectory Validation for {validator.molecule} ===\n")

    results = validator.comprehensive_validation()

    print(f"Trajectory Points: {results['trajectory_data']['n_points']}")
    print(f"\nFidelity Analysis:")
    print(f"  Average fidelity: {results['quality_analysis']['average_fidelity']:.4f} ({results['quality_analysis']['quality_average']})")
    print(f"  Fidelity at tau_em: {results['quality_analysis']['fidelity_at_tau_em']:.4f} ({results['quality_analysis']['quality_at_emission']})")
    print(f"  Fidelity drop: {results['quality_analysis']['fidelity_drop_at_emission']:.4f}")

    print(f"\nValidation:")
    print(f"  High fidelity (>0.95): {results['quality_analysis']['meets_high_threshold']}")
    print(f"  Excellent fidelity (>0.98): {results['quality_analysis']['meets_excellent_threshold']}")
    print(f"  Overall: {'PASS' if results['validation_pass'] else 'FAIL'}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'esdvs'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'ternary_trajectory_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
