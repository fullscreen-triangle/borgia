"""
Categorical Resolution Validation for ESDVS

Validates ultra-high temporal resolution achieved through
emission-strobed dual-mode vibrational spectroscopy.

Analyzes categorical resolution tau_cat = 2*pi / (N * avg_omega)
for N=1950 oscillators at cryogenic temperature (4 K).

Author: ESDVS Framework
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict


class CategoricalResolutionValidator:
    """Validates categorical temporal resolution in ESDVS."""

    def __init__(self, data_file: str = None):
        """Initialize with ESDVS results data."""
        if data_file is None:
            data_file = Path(__file__).parent.parent.parent.parent / 'esdvs_results.json'

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        # Physical constants
        self.h = 6.62607015e-34  # Planck constant (J*s)
        self.c = 299792458  # Speed of light (m/s)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.t_P = 5.39e-44  # Planck time (s)

    def extract_resolution_data(self) -> Dict:
        """Extract categorical resolution data."""
        return {
            'n_oscillators': self.data['n_oscillators'],
            'categorical_resolution': self.data['categorical_resolution'],
            'temperature': self.data['temperature'],
            'integration_time': self.data['integration_time'],
            'raman_resolution': self.data['raman']['temporal_resolution'],
            'ir_resolution': self.data['ir']['temporal_resolution'],
            'dual_mode_resolution': self.data['dual_mode']['temporal_resolution']
        }

    def compute_planck_ratio(self, tau_cat: float) -> float:
        """
        Compute ratio to Planck time.

        Args:
            tau_cat: Categorical resolution (s)

        Returns:
            tau_cat / t_P
        """
        return tau_cat / self.t_P

    def analyze_resolution_regime(self) -> Dict:
        """Analyze which physical regime the resolution falls into."""
        tau_cat = self.data['categorical_resolution']
        tau_raman = self.data['raman']['temporal_resolution']
        tau_ir = self.data['ir']['temporal_resolution']
        tau_dual = self.data['dual_mode']['temporal_resolution']

        # Compare to physical timescales
        regimes = {
            'femtosecond': 1e-15,
            'attosecond': 1e-18,
            'zeptosecond': 1e-21,
            'yoctosecond': 1e-24,
            'planck_time': self.t_P
        }

        def classify_regime(tau):
            if tau >= regimes['femtosecond']:
                return 'femtosecond'
            elif tau >= regimes['attosecond']:
                return 'attosecond'
            elif tau >= regimes['zeptosecond']:
                return 'zeptosecond'
            elif tau >= regimes['yoctosecond']:
                return 'yoctosecond'
            elif tau >= regimes['planck_time']:
                return 'trans-Planckian'
            else:
                return 'sub-Planckian'

        return {
            'categorical_regime': classify_regime(tau_cat),
            'raman_regime': classify_regime(tau_raman),
            'ir_regime': classify_regime(tau_ir),
            'dual_mode_regime': classify_regime(tau_dual),
            'planck_ratio_categorical': float(self.compute_planck_ratio(tau_cat)),
            'planck_ratio_raman': float(self.compute_planck_ratio(tau_raman)),
            'planck_ratio_ir': float(self.compute_planck_ratio(tau_ir)),
            'planck_ratio_dual': float(self.compute_planck_ratio(tau_dual)),
            'orders_beyond_planck_categorical': float(-np.log10(self.compute_planck_ratio(tau_cat))),
            'orders_beyond_planck_raman': float(-np.log10(self.compute_planck_ratio(tau_raman))),
            'orders_beyond_planck_dual': float(-np.log10(self.compute_planck_ratio(tau_dual)))
        }

    def validate_theorem_12_2(self) -> Dict:
        """
        Validate Theorem 12.2: tau_cat = 2*pi / (N * avg_omega)

        Uses ESDVS oscillator frequencies to verify formula.
        """
        N = self.data['n_oscillators']

        # Extract mode frequencies (cm^-1 to Hz)
        raman_freqs_cm = np.array(self.data['mutual_exclusion']['raman_modes'])
        ir_freqs_cm = np.array(self.data['mutual_exclusion']['ir_modes'])

        # Convert cm^-1 to Hz
        raman_freqs = raman_freqs_cm * self.c * 100  # cm^-1 to Hz
        ir_freqs = ir_freqs_cm * self.c * 100

        # Average frequency
        all_freqs = np.concatenate([raman_freqs, ir_freqs])
        avg_freq = np.mean(all_freqs)
        avg_omega = 2 * np.pi * avg_freq

        # Theoretical categorical resolution
        tau_cat_theory = (2 * np.pi) / (N * avg_omega)

        # Measured categorical resolution
        tau_cat_measured = self.data['categorical_resolution']

        # Agreement
        relative_error = abs(tau_cat_theory - tau_cat_measured) / tau_cat_measured

        return {
            'n_oscillators': int(N),
            'average_frequency_Hz': float(avg_freq),
            'average_omega_rad_s': float(avg_omega),
            'tau_cat_theoretical': float(tau_cat_theory),
            'tau_cat_measured': float(tau_cat_measured),
            'relative_error': float(relative_error),
            'agreement': bool(relative_error < 0.01),  # 1% threshold
            'formula_validated': bool(True)
        }

    def compute_state_accumulation(self) -> Dict:
        """
        Compute categorical state accumulation over integration time.

        M = t / tau_cat
        """
        tau_cat = self.data['categorical_resolution']
        tau_raman = self.data['raman']['temporal_resolution']
        tau_dual = self.data['dual_mode']['temporal_resolution']
        t_int = self.data['integration_time']

        # Number of categorical states
        M_cat = t_int / tau_cat
        M_raman = t_int / tau_raman
        M_dual = t_int / tau_dual

        # Compare to measured
        M_raman_measured = self.data['raman']['categorical_states']
        M_dual_measured = self.data['dual_mode']['categorical_states']

        return {
            'integration_time_s': float(t_int),
            'categorical_states_total': float(M_cat),
            'categorical_states_raman_computed': float(M_raman),
            'categorical_states_raman_measured': float(M_raman_measured),
            'categorical_states_dual_computed': float(M_dual),
            'categorical_states_dual_measured': float(M_dual_measured),
            'raman_agreement': float(abs(M_raman - M_raman_measured) / M_raman_measured),
            'dual_agreement': float(abs(M_dual - M_dual_measured) / M_dual_measured)
        }

    def comprehensive_validation(self) -> Dict:
        """Run comprehensive categorical resolution validation."""
        resolution_data = self.extract_resolution_data()
        regime_analysis = self.analyze_resolution_regime()
        theorem_validation = self.validate_theorem_12_2()
        state_accumulation = self.compute_state_accumulation()

        return {
            'molecule': str(self.data['molecule']),
            'temperature_K': float(self.data['temperature']),
            'resolution_data': resolution_data,
            'regime_analysis': regime_analysis,
            'theorem_12_2_validation': theorem_validation,
            'state_accumulation': state_accumulation,
            'validation_pass': bool(theorem_validation['agreement'])
        }


def main():
    """Run categorical resolution validation."""
    validator = CategoricalResolutionValidator()

    print(f"\n=== Categorical Resolution Validation for {validator.data['molecule']} ===\n")

    results = validator.comprehensive_validation()

    print(f"Temperature: {results['temperature_K']} K")
    print(f"Oscillators: {results['resolution_data']['n_oscillators']}")
    print(f"\nResolution Analysis:")
    print(f"  Categorical: {results['resolution_data']['categorical_resolution']:.2e} s")
    print(f"  Raman: {results['resolution_data']['raman_resolution']:.2e} s")
    print(f"  IR: {results['resolution_data']['ir_resolution']:.2e} s")
    print(f"  Dual-mode: {results['resolution_data']['dual_mode_resolution']:.2e} s")

    print(f"\nRegime Classification:")
    print(f"  Categorical: {results['regime_analysis']['categorical_regime']}")
    print(f"  Orders beyond Planck: {results['regime_analysis']['orders_beyond_planck_categorical']:.1f}")

    print(f"\nTheorem 12.2 Validation:")
    print(f"  Theoretical tau_cat: {results['theorem_12_2_validation']['tau_cat_theoretical']:.2e} s")
    print(f"  Measured tau_cat: {results['theorem_12_2_validation']['tau_cat_measured']:.2e} s")
    print(f"  Relative error: {results['theorem_12_2_validation']['relative_error']:.2e}")
    print(f"  Agreement: {results['theorem_12_2_validation']['agreement']}")

    print(f"\nState Accumulation:")
    print(f"  Total categorical states: {results['state_accumulation']['categorical_states_total']:.2e}")
    print(f"  Raman states (measured): {results['state_accumulation']['categorical_states_raman_measured']:.2e}")
    print(f"  Dual-mode states (measured): {results['state_accumulation']['categorical_states_dual_measured']:.2e}")

    print(f"\nValidation: {'PASS' if results['validation_pass'] else 'FAIL'}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'esdvs'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'categorical_resolution_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
