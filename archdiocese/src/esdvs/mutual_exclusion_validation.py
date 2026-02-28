"""
Mutual Exclusion Validation for ESDVS

Validates symmetry-based mutual exclusion principle for CH4+ (Td symmetry).
Analyzes Raman-IR mode separation and cross-prediction accuracy.

Author: Emission-Strobed Dual-Mode Spectroscopy Framework
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


class MutualExclusionValidator:
    """Validates mutual exclusion and cross-prediction for vibrational modes."""

    def __init__(self, data_file: str = None):
        """Initialize with ESDVS results data."""
        if data_file is None:
            data_file = Path(__file__).parent.parent.parent.parent / 'esdvs_results.json'

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.molecule = self.data['molecule']
        self.temperature = self.data['temperature']

    def extract_mode_data(self) -> Dict:
        """Extract Raman and IR mode data."""
        mutual_ex = self.data['mutual_exclusion']

        return {
            'raman_modes': list(mutual_ex['raman_modes']),
            'ir_modes': list(mutual_ex['ir_modes']),
            'overlaps': mutual_ex['overlaps'],
            'expected_overlaps': mutual_ex['expected_overlaps'],
            'n_raman': int(mutual_ex['n_raman']),
            'n_ir': int(mutual_ex['n_ir']),
            'n_overlap': int(mutual_ex['n_overlap']),
            'n_expected': int(mutual_ex['n_expected']),
            'n_unexpected': int(mutual_ex['n_unexpected']),
            'violation_metric': float(mutual_ex['violation_metric']),
            'violation_metric_strict': float(mutual_ex['violation_metric_strict']),
            'passes': bool(mutual_ex['passes'])
        }

    def analyze_mode_separation(self) -> Dict:
        """Analyze frequency separation between modes."""
        mode_data = self.extract_mode_data()

        raman = np.array(mode_data['raman_modes'])
        ir = np.array(mode_data['ir_modes'])

        # All modes combined
        all_modes = np.concatenate([raman, ir])

        # Pairwise distances
        raman_spacing = np.diff(np.sort(raman))
        ir_spacing = np.diff(np.sort(ir)) if len(ir) > 1 else np.array([])
        all_spacing = np.diff(np.sort(all_modes))

        return {
            'raman_min_spacing': float(np.min(raman_spacing)) if len(raman_spacing) > 0 else 0.0,
            'raman_max_spacing': float(np.max(raman_spacing)) if len(raman_spacing) > 0 else 0.0,
            'ir_min_spacing': float(np.min(ir_spacing)) if len(ir_spacing) > 0 else 0.0,
            'ir_max_spacing': float(np.max(ir_spacing)) if len(ir_spacing) > 0 else 0.0,
            'all_min_spacing': float(np.min(all_spacing)),
            'all_max_spacing': float(np.max(all_spacing)),
            'frequency_range': [float(np.min(all_modes)), float(np.max(all_modes))],
            'raman_range': [float(np.min(raman)), float(np.max(raman))],
            'ir_range': [float(np.min(ir)), float(np.max(ir))]
        }

    def validate_cross_prediction(self) -> Dict:
        """Validate cross-prediction accuracy."""
        cross_pred = self.data['cross_prediction']

        raman_from_ir = float(cross_pred['raman_from_ir'])
        ir_from_raman = float(cross_pred['ir_from_raman'])
        average = float(cross_pred['average'])

        # Check if accuracy meets threshold (>95%)
        threshold = 0.95
        passes_threshold = bool(average > threshold)

        return {
            'raman_from_ir_accuracy': raman_from_ir,
            'ir_from_raman_accuracy': ir_from_raman,
            'average_accuracy': average,
            'threshold': float(threshold),
            'passes_threshold': passes_threshold,
            'accuracy_percentage': float(average * 100)
        }

    def analyze_overlaps(self) -> Dict:
        """Analyze expected vs unexpected overlaps."""
        mode_data = self.extract_mode_data()

        overlaps = mode_data['overlaps']
        expected = mode_data['expected_overlaps']

        # Calculate frequency differences in overlaps
        overlap_diffs = []
        for overlap in overlaps:
            diff = float(abs(overlap[1] - overlap[0]))
            overlap_diffs.append(diff)

        return {
            'n_overlaps': int(len(overlaps)),
            'n_expected': int(len(expected)),
            'n_unexpected': int(mode_data['n_unexpected']),
            'overlap_frequency_diffs': overlap_diffs,
            'mean_overlap_diff': float(np.mean(overlap_diffs)) if overlap_diffs else 0.0,
            'all_expected': bool(mode_data['n_unexpected'] == 0)
        }

    def comprehensive_validation(self) -> Dict:
        """Run comprehensive validation suite."""
        mode_data = self.extract_mode_data()
        separation = self.analyze_mode_separation()
        cross_pred = self.validate_cross_prediction()
        overlaps = self.analyze_overlaps()

        # Overall pass/fail
        mutual_exclusion_pass = bool(mode_data['passes'])
        cross_prediction_pass = bool(cross_pred['passes_threshold'])
        no_unexpected_overlaps = bool(overlaps['all_expected'])

        overall_pass = bool(mutual_exclusion_pass and cross_prediction_pass and no_unexpected_overlaps)

        return {
            'molecule': str(self.molecule),
            'temperature_K': float(self.temperature),
            'mode_data': mode_data,
            'mode_separation': separation,
            'cross_prediction': cross_pred,
            'overlap_analysis': overlaps,
            'validation_summary': {
                'mutual_exclusion_pass': mutual_exclusion_pass,
                'cross_prediction_pass': cross_prediction_pass,
                'no_unexpected_overlaps': no_unexpected_overlaps,
                'overall_pass': overall_pass,
                'violation_metric': float(mode_data['violation_metric']),
                'strict_violation_metric': float(mode_data['violation_metric_strict'])
            }
        }


def main():
    """Run mutual exclusion validation."""
    validator = MutualExclusionValidator()

    print(f"\n=== Mutual Exclusion Validation for {validator.molecule} ===\n")

    results = validator.comprehensive_validation()

    # Print results
    print(f"Temperature: {results['temperature_K']} K")
    print(f"\nMode Counts:")
    print(f"  Raman modes: {results['mode_data']['n_raman']}")
    print(f"  IR modes: {results['mode_data']['n_ir']}")
    print(f"  Expected overlaps: {results['mode_data']['n_expected']}")
    print(f"  Unexpected overlaps: {results['mode_data']['n_unexpected']}")

    print(f"\nCross-Prediction Accuracy:")
    print(f"  Raman from IR: {results['cross_prediction']['raman_from_ir_accuracy']:.4f}")
    print(f"  IR from Raman: {results['cross_prediction']['ir_from_raman_accuracy']:.4f}")
    print(f"  Average: {results['cross_prediction']['accuracy_percentage']:.2f}%")

    print(f"\nMutual Exclusion Metrics:")
    print(f"  Violation metric: {results['validation_summary']['violation_metric']}")
    print(f"  Strict violation: {results['validation_summary']['strict_violation_metric']}")

    print(f"\nValidation Summary:")
    print(f"  Mutual exclusion: {'PASS' if results['validation_summary']['mutual_exclusion_pass'] else 'FAIL'}")
    print(f"  Cross-prediction: {'PASS' if results['validation_summary']['cross_prediction_pass'] else 'FAIL'}")
    print(f"  Overall: {'PASS' if results['validation_summary']['overall_pass'] else 'FAIL'}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'esdvs'
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / 'mutual_exclusion_validation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
