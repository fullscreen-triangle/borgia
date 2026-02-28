"""
Dual-Mode Enhancement Validation for ESDVS

Validates enhancement factor from simultaneous Raman + IR operation.
Analyzes complementary information and resolution improvement.

Enhancement factor = tau_single / tau_dual

Author: ESDVS Framework
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict


class DualModeEnhancementValidator:
    """Validates dual-mode enhancement in ESDVS."""

    def __init__(self, data_file: str = None):
        """Initialize with ESDVS results data."""
        if data_file is None:
            data_file = Path(__file__).parent.parent.parent.parent / 'esdvs_results.json'

        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def extract_enhancement_data(self) -> Dict:
        """Extract dual-mode enhancement data."""
        return {
            'raman_states': self.data['raman']['categorical_states'],
            'ir_states': self.data['ir']['categorical_states'],
            'dual_mode_states': self.data['dual_mode']['categorical_states'],
            'raman_resolution': self.data['raman']['temporal_resolution'],
            'ir_resolution': self.data['ir']['temporal_resolution'],
            'dual_mode_resolution': self.data['dual_mode']['temporal_resolution'],
            'enhancement_factor': self.data['dual_mode']['enhancement_factor']
        }

    def validate_enhancement_formula(self) -> Dict:
        """
        Validate enhancement factor formula.

        For independent modes: M_dual = M_raman + M_ir
        Enhancement = tau_avg / tau_dual
        """
        M_raman = self.data['raman']['categorical_states']
        M_ir = self.data['ir']['categorical_states']
        M_dual = self.data['dual_mode']['categorical_states']

        tau_raman = self.data['raman']['temporal_resolution']
        tau_ir = self.data['ir']['temporal_resolution']
        tau_dual = self.data['dual_mode']['temporal_resolution']

        # Expected dual-mode states (additive for independent modes)
        M_dual_expected = M_raman + M_ir

        # Average single-mode resolution
        tau_avg = (tau_raman + tau_ir) / 2

        # Theoretical enhancement
        enhancement_theoretical = tau_avg / tau_dual

        # Measured enhancement
        enhancement_measured = self.data['dual_mode']['enhancement_factor']

        # Agreement
        state_relative_error = abs(M_dual - M_dual_expected) / M_dual_expected
        enhancement_relative_error = abs(enhancement_theoretical - enhancement_measured) / enhancement_measured

        return {
            'M_dual_expected': float(M_dual_expected),
            'M_dual_measured': float(M_dual),
            'state_agreement': float(1 - state_relative_error),
            'enhancement_theoretical': float(enhancement_theoretical),
            'enhancement_measured': float(enhancement_measured),
            'enhancement_agreement': float(1 - enhancement_relative_error),
            'formula_validated': bool(enhancement_relative_error < 0.01)
        }

    def analyze_complementarity(self) -> Dict:
        """
        Analyze complementarity between Raman and IR modes.

        Mutual exclusion rule for Td symmetry ensures modes are complementary.
        """
        n_raman = self.data['mutual_exclusion']['n_raman']
        n_ir = self.data['mutual_exclusion']['n_ir']
        n_overlap = self.data['mutual_exclusion']['n_overlap']
        n_expected = self.data['mutual_exclusion']['n_expected']

        # Complementarity index: fraction of non-overlapping modes
        # For perfect complementarity with expected overlaps: = 1
        # Unexpected overlaps reduce this
        n_unexpected = self.data['mutual_exclusion']['n_unexpected']
        complementarity = 1 - (n_unexpected / max(n_raman, n_ir))

        # Information redundancy (overlap fraction)
        redundancy = n_overlap / (n_raman + n_ir)

        # Effective mode count
        n_effective = n_raman + n_ir - n_overlap

        return {
            'n_raman_modes': int(n_raman),
            'n_ir_modes': int(n_ir),
            'n_overlap_modes': int(n_overlap),
            'n_expected_overlap': int(n_expected),
            'n_unexpected_overlap': int(n_unexpected),
            'n_effective_modes': int(n_effective),
            'complementarity_index': float(complementarity),
            'redundancy_fraction': float(redundancy),
            'is_complementary': bool(complementarity > 0.9)
        }

    def compute_information_gain(self) -> Dict:
        """
        Compute information gain from dual-mode operation.

        I_dual = log2(M_dual / M_single)
        """
        M_raman = self.data['raman']['categorical_states']
        M_ir = self.data['ir']['categorical_states']
        M_dual = self.data['dual_mode']['categorical_states']

        # Average single-mode states
        M_avg_single = (M_raman + M_ir) / 2

        # Information gain (bits)
        I_gain = np.log2(M_dual / M_avg_single)

        # Relative improvement
        improvement = (M_dual - M_avg_single) / M_avg_single

        return {
            'M_avg_single': float(M_avg_single),
            'M_dual': float(M_dual),
            'information_gain_bits': float(I_gain),
            'relative_improvement': float(improvement),
            'improvement_percent': float(improvement * 100)
        }

    def validate_resolution_improvement(self) -> Dict:
        """
        Validate that dual-mode improves temporal resolution.

        tau_dual < tau_raman and tau_dual < tau_ir
        """
        tau_raman = self.data['raman']['temporal_resolution']
        tau_ir = self.data['ir']['temporal_resolution']
        tau_dual = self.data['dual_mode']['temporal_resolution']

        improvement_over_raman = (tau_raman - tau_dual) / tau_raman
        improvement_over_ir = (tau_ir - tau_dual) / tau_ir

        # Average improvement
        avg_improvement = (improvement_over_raman + improvement_over_ir) / 2

        return {
            'tau_raman': float(tau_raman),
            'tau_ir': float(tau_ir),
            'tau_dual': float(tau_dual),
            'improvement_over_raman': float(improvement_over_raman),
            'improvement_over_ir': float(improvement_over_ir),
            'average_improvement': float(avg_improvement),
            'improvement_percent': float(avg_improvement * 100),
            'is_improved': bool(tau_dual < tau_raman and tau_dual < tau_ir)
        }

    def comprehensive_validation(self) -> Dict:
        """Run comprehensive dual-mode enhancement validation."""
        enhancement_data = self.extract_enhancement_data()
        formula_validation = self.validate_enhancement_formula()
        complementarity = self.analyze_complementarity()
        information_gain = self.compute_information_gain()
        resolution_improvement = self.validate_resolution_improvement()

        return {
            'molecule': str(self.data['molecule']),
            'enhancement_data': enhancement_data,
            'formula_validation': formula_validation,
            'complementarity_analysis': complementarity,
            'information_gain': information_gain,
            'resolution_improvement': resolution_improvement,
            'validation_pass': bool(
                formula_validation['formula_validated'] and
                complementarity['is_complementary'] and
                resolution_improvement['is_improved']
            )
        }


def main():
    """Run dual-mode enhancement validation."""
    validator = DualModeEnhancementValidator()

    print(f"\n=== Dual-Mode Enhancement Validation for {validator.data['molecule']} ===\n")

    results = validator.comprehensive_validation()

    print(f"Enhancement Factor: {results['enhancement_data']['enhancement_factor']:.4f}x")
    print(f"\nFormula Validation:")
    print(f"  Enhancement (theoretical): {results['formula_validation']['enhancement_theoretical']:.4f}x")
    print(f"  Enhancement (measured): {results['formula_validation']['enhancement_measured']:.4f}x")
    print(f"  Agreement: {results['formula_validation']['enhancement_agreement']:.4f}")
    print(f"  Formula validated: {results['formula_validation']['formula_validated']}")

    print(f"\nComplementarity Analysis:")
    print(f"  Raman modes: {results['complementarity_analysis']['n_raman_modes']}")
    print(f"  IR modes: {results['complementarity_analysis']['n_ir_modes']}")
    print(f"  Effective modes: {results['complementarity_analysis']['n_effective_modes']}")
    print(f"  Complementarity index: {results['complementarity_analysis']['complementarity_index']:.4f}")
    print(f"  Is complementary: {results['complementarity_analysis']['is_complementary']}")

    print(f"\nInformation Gain:")
    print(f"  Information gain: {results['information_gain']['information_gain_bits']:.2f} bits")
    print(f"  Relative improvement: {results['information_gain']['improvement_percent']:.2f}%")

    print(f"\nResolution Improvement:")
    print(f"  tau_raman: {results['resolution_improvement']['tau_raman']:.2e} s")
    print(f"  tau_ir: {results['resolution_improvement']['tau_ir']:.2e} s")
    print(f"  tau_dual: {results['resolution_improvement']['tau_dual']:.2e} s")
    print(f"  Average improvement: {results['resolution_improvement']['improvement_percent']:.2f}%")
    print(f"  Is improved: {results['resolution_improvement']['is_improved']}")

    print(f"\nValidation: {'PASS' if results['validation_pass'] else 'FAIL'}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'esdvs'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'dual_mode_enhancement_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
