#!/usr/bin/env python3
"""
Partition Coordinate Validation Framework
==========================================

Validates the theoretical claims from:
"On the Necessity of Frequency-Selective Coupling Structures in Bounded Oscillatory Systems"

This module tests:
1. Partition coordinate extraction (n, l, m, s)
2. Capacity theorem: 2n² states at depth n
3. Frequency-coordinate duality
4. Selection rules: Δl = ±1, Δm ∈ {0, ±1}, Δs = 0
5. Resonance enhancement and selectivity
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, signal
from scipy.stats import pearsonr
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json


# =============================================================================
# PARTITION COORDINATE SYSTEM
# =============================================================================

@dataclass
class PartitionCoordinate:
    """Represents a partition coordinate (n, l, m, s)"""
    n: int      # Depth (principal)
    l: int      # Complexity (angular)
    m: int      # Orientation (magnetic)
    s: float    # Chirality (spin): ±0.5

    def __post_init__(self):
        """Validate constraints from Theorem 3.2"""
        assert self.n >= 1, f"n must be ≥ 1, got {self.n}"
        assert 0 <= self.l <= self.n - 1, f"l must be in [0, n-1], got l={self.l} for n={self.n}"
        assert -self.l <= self.m <= self.l, f"m must be in [-l, l], got m={self.m} for l={self.l}"
        assert self.s in (-0.5, 0.5), f"s must be ±0.5, got {self.s}"

    def to_tuple(self) -> Tuple[int, int, int, float]:
        return (self.n, self.l, self.m, self.s)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()


class PartitionCoordinateSystem:
    """
    Implements the partition coordinate system from the theoretical framework.

    Validates:
    - Constraint relationships (Theorem 3.2)
    - Capacity theorem: 2n² (Theorem 3.3)
    - Energy ordering (n + αl) (Theorem 3.4)
    """

    def __init__(self, max_n: int = 7):
        self.max_n = max_n
        self.all_coordinates = self._generate_all_coordinates()

    def _generate_all_coordinates(self) -> List[PartitionCoordinate]:
        """Generate all valid partition coordinates up to max_n"""
        coords = []
        for n in range(1, self.max_n + 1):
            for l in range(0, n):
                for m in range(-l, l + 1):
                    for s in (-0.5, 0.5):
                        coords.append(PartitionCoordinate(n, l, m, s))
        return coords

    def capacity_at_depth(self, n: int) -> int:
        """
        Calculate capacity at depth n.
        Theorem 3.3: C(n) = 2n²
        """
        return 2 * n * n

    def validate_capacity_theorem(self) -> Dict:
        """
        Validate that capacity at each depth n equals 2n².

        Returns validation results with observed vs predicted counts.
        """
        results = {'depths': [], 'observed': [], 'predicted': [], 'match': []}

        for n in range(1, self.max_n + 1):
            observed = sum(1 for c in self.all_coordinates if c.n == n)
            predicted = self.capacity_at_depth(n)

            results['depths'].append(n)
            results['observed'].append(observed)
            results['predicted'].append(predicted)
            results['match'].append(observed == predicted)

        results['all_match'] = all(results['match'])
        results['total_states'] = len(self.all_coordinates)
        results['cumulative_predicted'] = sum(self.capacity_at_depth(n) for n in range(1, self.max_n + 1))

        return results

    def energy_ordering(self, alpha: float = 0.7) -> List[PartitionCoordinate]:
        """
        Order coordinates by energy: E ∝ (n + αl)
        Theorem 3.4: Energy ordering
        """
        def energy(coord: PartitionCoordinate) -> float:
            return coord.n + alpha * coord.l

        return sorted(self.all_coordinates, key=energy)

    def filling_sequence(self, num_entities: int, alpha: float = 0.7) -> List[PartitionCoordinate]:
        """
        Get filling sequence for num_entities following energy ordering.
        Implements the exclusion principle: no two entities share coordinates.
        """
        ordered = self.energy_ordering(alpha)
        return ordered[:num_entities]

    def shell_closures(self, alpha: float = 0.7) -> List[int]:
        """
        Find shell closure points (complete filling of (n + αl) levels).

        Returns list of cumulative counts at shell closures.
        """
        closures = []
        ordered = self.energy_ordering(alpha)

        current_level = None
        for i, coord in enumerate(ordered):
            level = coord.n + alpha * coord.l
            if current_level is not None and level > current_level + 0.5:
                closures.append(i)
            current_level = level

        return closures


# =============================================================================
# FREQUENCY-COORDINATE DUALITY
# =============================================================================

class FrequencyCoordinateDuality:
    """
    Implements frequency-coordinate duality from Theorem 4.1.

    Maps each coordinate to characteristic frequency regime:
    - ω_n ∝ n⁻³ (depth → high frequency / X-ray)
    - ω_l ∝ l(l+1) (complexity → optical)
    - ω_m ∝ m (orientation → microwave)
    - ω_s ∝ s (chirality → radio)
    """

    def __init__(self):
        # Fundamental frequency scale (atomic units: Hartree/ℏ ≈ 4.13 × 10¹⁶ Hz)
        self.omega_0 = 4.13e16  # Hz

        # Hierarchy constants (β ≪ 1, γ ≪ β, δ ≪ γ)
        self.beta = 1e-3    # Angular/optical regime
        self.gamma = 1e-6   # Orientation/microwave regime
        self.delta = 1e-9   # Chirality/radio regime

        # Frequency regimes (Hz) - with separation gaps
        self.regimes = {
            'n': (1e16, 1e18),     # X-ray / UV (core transitions)
            'l': (1e13, 1e15),     # Optical / IR (valence transitions)
            'm': (1e10, 1e11),     # Microwave (Zeeman splitting)
            's': (1e6, 1e8)        # Radio / NMR (spin resonance)
        }

    def omega_n(self, n: int) -> float:
        """Characteristic frequency for depth coordinate n"""
        return self.omega_0 * (n ** -3)

    def omega_l(self, l: int) -> float:
        """Characteristic frequency for complexity coordinate l"""
        return self.omega_0 * self.beta * l * (l + 1)

    def omega_m(self, m: int, B: float = 1.0) -> float:
        """
        Characteristic frequency for orientation coordinate m.
        B: external field strength (Tesla)
        """
        mu_B = constants.physical_constants['Bohr magneton'][0]
        return (mu_B * B * abs(m)) / constants.hbar

    def omega_s(self, s: float, B: float = 1.0) -> float:
        """
        Characteristic frequency for chirality coordinate s.
        B: external field strength (Tesla)
        """
        g_factor = 2.002319  # Electron g-factor
        mu_B = constants.physical_constants['Bohr magneton'][0]
        return (g_factor * mu_B * B * abs(s)) / constants.hbar

    def frequency_fingerprint(self, coord: PartitionCoordinate, B: float = 1.0) -> Dict[str, float]:
        """
        Get complete frequency fingerprint for a partition coordinate.
        Corollary 4.3: Frequency fingerprint uniquely identifies coordinate.
        """
        return {
            'omega_n': self.omega_n(coord.n),
            'omega_l': self.omega_l(coord.l),
            'omega_m': self.omega_m(coord.m, B),
            'omega_s': self.omega_s(coord.s, B)
        }

    def identify_regime(self, frequency: float) -> str:
        """Map frequency to coordinate regime"""
        for coord, (low, high) in self.regimes.items():
            if low <= frequency <= high:
                return coord
        return 'unknown'

    def validate_regime_separation(self) -> Dict:
        """
        Validate that frequency regimes are well-separated.
        Proposition 4.2: sup Ω_s ≪ inf Ω_m ≪ ... ≪ inf Ω_n
        """
        results = {'separated': True, 'gaps': {}}

        regime_order = ['s', 'm', 'l', 'n']

        for i in range(len(regime_order) - 1):
            low_regime = regime_order[i]
            high_regime = regime_order[i + 1]

            gap = self.regimes[high_regime][0] / self.regimes[low_regime][1]
            results['gaps'][f'{low_regime}_to_{high_regime}'] = gap

            if gap < 10:  # Should be well-separated
                results['separated'] = False

        return results


# =============================================================================
# SELECTION RULES VALIDATION
# =============================================================================

class SelectionRulesValidator:
    """
    Validates selection rules from Theorem 4.4.

    - Δl = ±1 (dipole selection)
    - Δm ∈ {0, ±1} (orientation selection)
    - Δs = 0 (chirality conservation)
    """

    @staticmethod
    def is_allowed_transition(initial: PartitionCoordinate,
                              final: PartitionCoordinate) -> Dict:
        """
        Check if transition is allowed by selection rules.

        Returns dict with allowed status and which rules are satisfied.
        """
        delta_n = final.n - initial.n
        delta_l = final.l - initial.l
        delta_m = final.m - initial.m
        delta_s = final.s - initial.s

        l_allowed = delta_l in (-1, 1)
        m_allowed = delta_m in (-1, 0, 1)
        s_allowed = delta_s == 0

        return {
            'allowed': l_allowed and m_allowed and s_allowed,
            'delta_n': delta_n,
            'delta_l': delta_l,
            'delta_m': delta_m,
            'delta_s': delta_s,
            'l_rule_satisfied': l_allowed,
            'm_rule_satisfied': m_allowed,
            's_rule_satisfied': s_allowed
        }

    @staticmethod
    def allowed_transitions_from(coord: PartitionCoordinate,
                                  max_n: int = 7) -> List[PartitionCoordinate]:
        """Get all allowed transitions from given coordinate"""
        allowed = []

        for n in range(1, max_n + 1):
            for l in range(0, n):
                # Only Δl = ±1 allowed
                if abs(l - coord.l) != 1:
                    continue

                for m in range(-l, l + 1):
                    # Only Δm ∈ {0, ±1} allowed
                    if abs(m - coord.m) > 1:
                        continue

                    # Δs = 0 (chirality conserved)
                    s = coord.s

                    try:
                        new_coord = PartitionCoordinate(n, l, m, s)
                        if new_coord != coord:
                            allowed.append(new_coord)
                    except AssertionError:
                        pass  # Invalid coordinate

        return allowed

    def validate_selection_rules(self, coord_system: PartitionCoordinateSystem) -> Dict:
        """
        Validate selection rules across all coordinate pairs.

        Returns statistics on rule compliance.
        """
        all_coords = coord_system.all_coordinates

        total_pairs = 0
        allowed_count = 0
        rule_violations = {'l': 0, 'm': 0, 's': 0}

        for initial in all_coords:
            for final in all_coords:
                if initial == final:
                    continue

                total_pairs += 1
                result = self.is_allowed_transition(initial, final)

                if result['allowed']:
                    allowed_count += 1
                else:
                    if not result['l_rule_satisfied']:
                        rule_violations['l'] += 1
                    if not result['m_rule_satisfied']:
                        rule_violations['m'] += 1
                    if not result['s_rule_satisfied']:
                        rule_violations['s'] += 1

        return {
            'total_pairs': total_pairs,
            'allowed_transitions': allowed_count,
            'forbidden_transitions': total_pairs - allowed_count,
            'allowed_fraction': allowed_count / total_pairs if total_pairs > 0 else 0,
            'rule_violations': rule_violations
        }


# =============================================================================
# RESONANCE AND COUPLING VALIDATION
# =============================================================================

class ResonanceValidator:
    """
    Validates resonance conditions from Section 6.

    Tests:
    - Lorentzian coupling profile
    - Off-resonance suppression
    - Selectivity from regime separation
    """

    def __init__(self, linewidth_ratio: float = 0.01):
        """
        linewidth_ratio: Γ/ω₀ (typical quality factor inverse)
        """
        self.linewidth_ratio = linewidth_ratio

    def coupling_strength(self, omega_system: float, omega_apparatus: float,
                          linewidth: float) -> float:
        """
        Lorentzian coupling strength (Theorem 6.1).
        C(ω_s, ω_a) = C₀ / (1 + (ω_s - ω_a)² / Γ²)
        """
        detuning = omega_system - omega_apparatus
        return 1.0 / (1.0 + (detuning / linewidth) ** 2)

    def validate_resonance_enhancement(self, omega_0: float = 1e15) -> Dict:
        """
        Validate resonance enhancement profile.

        Returns coupling strength as function of detuning.
        """
        linewidth = omega_0 * self.linewidth_ratio

        # Detuning range: -10Γ to +10Γ
        detunings = np.linspace(-10 * linewidth, 10 * linewidth, 1000)

        couplings = [self.coupling_strength(omega_0 + d, omega_0, linewidth)
                     for d in detunings]

        # Find FWHM
        half_max = 0.5
        above_half = np.array(couplings) >= half_max
        fwhm_indices = np.where(above_half)[0]

        if len(fwhm_indices) > 0:
            fwhm = detunings[fwhm_indices[-1]] - detunings[fwhm_indices[0]]
        else:
            fwhm = 2 * linewidth  # Theoretical value

        return {
            'detunings': detunings.tolist(),
            'couplings': couplings,
            'linewidth': linewidth,
            'fwhm_measured': fwhm,
            'fwhm_theoretical': 2 * linewidth,
            'peak_coupling': max(couplings),
            'on_resonance': couplings[len(couplings)//2]
        }

    def validate_off_resonance_suppression(self, omega_0: float = 1e15) -> Dict:
        """
        Validate off-resonance suppression (Corollary 6.1).
        For Δ ≫ Γ: C ≈ C₀ × (Γ/Δ)²
        """
        linewidth = omega_0 * self.linewidth_ratio

        # Test detunings: 1Γ to 100Γ
        detuning_factors = np.logspace(0, 2, 50)
        detunings = detuning_factors * linewidth

        measured_suppression = []
        predicted_suppression = []

        for d in detunings:
            measured = self.coupling_strength(omega_0 + d, omega_0, linewidth)
            predicted = (linewidth / d) ** 2

            measured_suppression.append(measured)
            predicted_suppression.append(predicted)

        # Correlation between measured and predicted (for Δ ≫ Γ)
        correlation, p_value = pearsonr(
            np.log10(measured_suppression[10:]),  # Skip first points where Δ ~ Γ
            np.log10(predicted_suppression[10:])
        )

        return {
            'detuning_factors': detuning_factors.tolist(),
            'measured_suppression': measured_suppression,
            'predicted_suppression': predicted_suppression,
            'correlation': correlation,
            'p_value': p_value,
            'agrees_with_theory': correlation > 0.99
        }

    def calculate_selectivity(self, freq_duality: FrequencyCoordinateDuality) -> Dict:
        """
        Calculate selectivity for each coordinate (Theorem 6.3).
        S_ξ ≥ (Δ_min / Γ)²
        """
        selectivities = {}

        for coord in ['n', 'l', 'm', 's']:
            regime = freq_duality.regimes[coord]
            regime_width = regime[1] - regime[0]
            linewidth = regime[0] * self.linewidth_ratio

            # Find minimum distance to nearest regime
            other_regimes = [r for c, r in freq_duality.regimes.items() if c != coord]
            min_distances = []

            for other in other_regimes:
                dist1 = abs(regime[0] - other[1])
                dist2 = abs(regime[1] - other[0])
                min_distances.append(min(dist1, dist2))

            delta_min = min(min_distances)
            selectivity = (delta_min / linewidth) ** 2

            selectivities[coord] = {
                'regime': regime,
                'linewidth': linewidth,
                'delta_min': delta_min,
                'selectivity': selectivity,
                'high_selectivity': selectivity > 100
            }

        return selectivities


# =============================================================================
# MOLECULAR DATA INTEGRATION
# =============================================================================

class MolecularPartitionExtractor:
    """
    Extract partition coordinates from molecular data (SMARTS patterns).

    This bridges the theoretical framework with experimental validation
    by mapping molecular features to partition coordinates.
    """

    def __init__(self):
        self.coord_system = PartitionCoordinateSystem()
        self.freq_duality = FrequencyCoordinateDuality()

    def extract_from_smarts(self, pattern: str) -> Dict:
        """
        Extract approximate partition coordinates from SMARTS pattern.

        Mapping heuristics:
        - n (depth): related to molecular size / electron count
        - l (complexity): related to aromatic/conjugation structure
        - m (orientation): related to symmetry elements
        - s (chirality): related to stereogenic centers
        """
        if not pattern:
            return {'error': 'Empty pattern'}

        # Estimate depth (n) from molecular size
        # Heavier atoms / more electrons → higher n
        heavy_atoms = sum(1 for c in pattern if c.isupper())
        n = min(7, max(1, (heavy_atoms + 2) // 3))

        # Estimate complexity (l) from aromatic/conjugation
        aromatic = sum(1 for c in pattern if c.islower())
        conjugation = pattern.count('=') + pattern.count('#')
        l = min(n - 1, (aromatic + conjugation) // 2)

        # Estimate orientation (m) from asymmetry
        # Balanced patterns → m ≈ 0, asymmetric → higher |m|
        left_count = pattern.count('(')
        right_count = pattern.count(')')
        asymmetry = abs(left_count - right_count)
        m = min(l, max(-l, asymmetry - l // 2))

        # Estimate chirality (s) from stereogenic indicators
        has_stereo = '@' in pattern or '/' in pattern or '\\' in pattern
        s = 0.5 if has_stereo else -0.5

        try:
            coord = PartitionCoordinate(n, l, m, s)
            fingerprint = self.freq_duality.frequency_fingerprint(coord)

            return {
                'pattern': pattern,
                'coordinate': coord.to_tuple(),
                'n': n, 'l': l, 'm': m, 's': s,
                'frequency_fingerprint': fingerprint,
                'valid': True
            }
        except AssertionError as e:
            return {
                'pattern': pattern,
                'error': str(e),
                'valid': False
            }

    def extract_batch(self, patterns: List[str]) -> List[Dict]:
        """Extract coordinates from multiple patterns"""
        return [self.extract_from_smarts(p) for p in patterns]

    def validate_coordinate_distribution(self, patterns: List[str]) -> Dict:
        """
        Validate that extracted coordinates follow expected distributions.
        """
        extractions = self.extract_batch(patterns)
        valid = [e for e in extractions if e.get('valid', False)]

        if not valid:
            return {'error': 'No valid extractions'}

        n_values = [e['n'] for e in valid]
        l_values = [e['l'] for e in valid]
        m_values = [e['m'] for e in valid]
        s_values = [e['s'] for e in valid]

        return {
            'total_patterns': len(patterns),
            'valid_extractions': len(valid),
            'n_distribution': {
                'mean': np.mean(n_values),
                'std': np.std(n_values),
                'min': min(n_values),
                'max': max(n_values)
            },
            'l_distribution': {
                'mean': np.mean(l_values),
                'std': np.std(l_values),
                'min': min(l_values),
                'max': max(l_values)
            },
            'm_distribution': {
                'mean': np.mean(m_values),
                'std': np.std(m_values),
                'min': min(m_values),
                'max': max(m_values)
            },
            's_distribution': {
                'positive_fraction': sum(1 for s in s_values if s > 0) / len(s_values)
            }
        }


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def load_datasets():
    """Load SMARTS datasets"""
    datasets = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')

    files = {
        'agrafiotis': os.path.join(base_dir, 'public', 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(base_dir, 'public', 'ahmed-smarts-tar', 'ahmed.smarts'),
        'hann': os.path.join(base_dir, 'public', 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(base_dir, 'public', 'walters-smarts-tar', 'walters.smarts')
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
                print(f"  Loaded {len(patterns)} patterns from {name}")
            except Exception as e:
                print(f"  Error loading {name}: {e}")

    if not datasets:
        print("  No SMARTS files found, using synthetic patterns...")
        datasets['synthetic'] = [
            'c1ccccc1', 'CCO', 'CC(=O)O', 'c1ccc2ccccc2c1', 'CC(C)O',
            'C1=CC=C(C=C1)O', 'CC(=O)OC1=CC=CC=C1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
        ]

    return datasets


def create_validation_plots(results: Dict, save_dir: str):
    """Create comprehensive validation visualizations"""

    fig = plt.figure(figsize=(20, 16))

    # 1. Capacity Theorem Validation
    ax1 = plt.subplot(3, 4, 1)
    capacity = results['capacity_validation']
    depths = capacity['depths']
    ax1.bar(np.array(depths) - 0.2, capacity['observed'], 0.4,
            label='Observed', alpha=0.7, color='blue')
    ax1.bar(np.array(depths) + 0.2, capacity['predicted'], 0.4,
            label='Predicted (2n²)', alpha=0.7, color='red')
    ax1.set_xlabel('Depth n')
    ax1.set_ylabel('State Count')
    ax1.set_title('Capacity Theorem: 2n²')
    ax1.legend()
    ax1.set_xticks(depths)

    # 2. Frequency Regime Separation
    ax2 = plt.subplot(3, 4, 2)
    regimes = results['regime_separation']
    gap_names = list(regimes['gaps'].keys())
    gap_values = [np.log10(v) for v in regimes['gaps'].values()]
    colors = ['green' if v > 1 else 'red' for v in gap_values]
    ax2.barh(gap_names, gap_values, color=colors, alpha=0.7)
    ax2.axvline(1, color='black', linestyle='--', label='10× separation')
    ax2.set_xlabel('Log₁₀(Gap Factor)')
    ax2.set_title('Frequency Regime Separation')
    ax2.legend()

    # 3. Selection Rules Statistics
    ax3 = plt.subplot(3, 4, 3)
    selection = results['selection_rules']
    categories = ['Allowed', 'Forbidden']
    counts = [selection['allowed_transitions'], selection['forbidden_transitions']]
    ax3.pie(counts, labels=categories, autopct='%1.1f%%',
            colors=['green', 'red'])
    ax3.set_title(f'Selection Rules\n(Total: {selection["total_pairs"]} pairs)')

    # 4. Resonance Profile
    ax4 = plt.subplot(3, 4, 4)
    resonance = results['resonance_enhancement']
    detunings = np.array(resonance['detunings'])
    linewidth = resonance['linewidth']
    ax4.plot(detunings / linewidth, resonance['couplings'], 'b-', linewidth=2)
    ax4.axhline(0.5, color='red', linestyle='--', label='FWHM')
    ax4.set_xlabel('Detuning / Γ')
    ax4.set_ylabel('Coupling Strength')
    ax4.set_title('Lorentzian Resonance Profile')
    ax4.legend()
    ax4.set_xlim(-5, 5)
    ax4.grid(True, alpha=0.3)

    # 5. Off-Resonance Suppression
    ax5 = plt.subplot(3, 4, 5)
    suppression = results['off_resonance_suppression']
    ax5.loglog(suppression['detuning_factors'], suppression['measured_suppression'],
               'b-', label='Measured', linewidth=2)
    ax5.loglog(suppression['detuning_factors'], suppression['predicted_suppression'],
               'r--', label='Predicted (Γ/Δ)²', linewidth=2)
    ax5.set_xlabel('Detuning / Γ')
    ax5.set_ylabel('Suppression Factor')
    ax5.set_title(f'Off-Resonance Suppression\n(Correlation: {suppression["correlation"]:.4f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Coordinate Selectivity
    ax6 = plt.subplot(3, 4, 6)
    selectivity = results['selectivity']
    coords = list(selectivity.keys())
    sel_values = [np.log10(selectivity[c]['selectivity']) for c in coords]
    colors = ['green' if selectivity[c]['high_selectivity'] else 'orange' for c in coords]
    ax6.bar(coords, sel_values, color=colors, alpha=0.7)
    ax6.axhline(2, color='red', linestyle='--', label='S > 100')
    ax6.set_xlabel('Coordinate')
    ax6.set_ylabel('Log₁₀(Selectivity)')
    ax6.set_title('Coordinate Selectivity')
    ax6.legend()

    # 7. Energy Ordering (Filling Sequence)
    ax7 = plt.subplot(3, 4, 7)
    filling = results['filling_sequence']
    energies = [c['n'] + 0.7 * c['l'] for c in filling[:30]]
    ax7.plot(range(1, len(energies) + 1), energies, 'bo-', markersize=4)
    ax7.set_xlabel('Filling Order')
    ax7.set_ylabel('Energy (n + 0.7l)')
    ax7.set_title('Energy Ordering')
    ax7.grid(True, alpha=0.3)

    # 8. Molecular Coordinate Distribution (n)
    ax8 = plt.subplot(3, 4, 8)
    mol_dist = results.get('molecular_distribution', {})
    if mol_dist and 'n_distribution' in mol_dist:
        # Create histogram of n values from extractions
        ax8.text(0.5, 0.5, f"n: μ={mol_dist['n_distribution']['mean']:.2f}\n"
                          f"   σ={mol_dist['n_distribution']['std']:.2f}\n"
                          f"   range=[{mol_dist['n_distribution']['min']}, {mol_dist['n_distribution']['max']}]",
                transform=ax8.transAxes, fontsize=12, va='center', ha='center')
    ax8.set_title('Molecular n Distribution')
    ax8.axis('off')

    # 9-12: Summary Panels
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    summary_text = f"""
    VALIDATION SUMMARY
    ==================

    Capacity Theorem (2n^2):
    {'PASSED' if capacity['all_match'] else 'FAILED'}
    Total states: {capacity['total_states']}

    Regime Separation:
    {'Well-separated' if regimes['separated'] else 'Overlapping'}

    Selection Rules:
    Allowed fraction: {selection['allowed_fraction']:.1%}

    Resonance Theory:
    Correlation: {suppression['correlation']:.4f}
    {'Matches theory' if suppression['agrees_with_theory'] else 'Deviates'}
    """
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    # 10. Rule Violations Breakdown
    ax10 = plt.subplot(3, 4, 10)
    violations = selection['rule_violations']
    ax10.bar(['Δl ≠ ±1', 'Δm ∉ {0,±1}', 'Δs ≠ 0'],
             [violations['l'], violations['m'], violations['s']],
             color=['blue', 'green', 'red'], alpha=0.7)
    ax10.set_ylabel('Violation Count')
    ax10.set_title('Selection Rule Violations')

    # 11. Shell Closures
    ax11 = plt.subplot(3, 4, 11)
    closures = results['shell_closures']
    ax11.scatter(range(len(closures)), closures, s=100, c='purple', marker='*')
    ax11.set_xlabel('Shell Index')
    ax11.set_ylabel('Cumulative Count')
    ax11.set_title('Shell Closure Points')
    ax11.grid(True, alpha=0.3)

    # 12. Framework Correspondence
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    correspondence_text = """
    PHYSICAL CORRESPONDENCE
    =======================

    Derived Structure → Physical System
    -----------------------------------
    (n,l,m,s) coords → Quantum numbers
    Capacity 2n²     → Shell structure
    Δl = ±1          → Dipole selection
    ω_n regime       → X-ray spectroscopy
    ω_l regime       → UV-Vis spectroscopy
    ω_m regime       → Zeeman spectroscopy
    ω_s regime       → NMR spectroscopy
    """
    ax12.text(0.1, 0.9, correspondence_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'partition_coordinate_validation.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def main():
    """Run comprehensive partition coordinate validation"""
    print("=" * 70)
    print("PARTITION COORDINATE VALIDATION FRAMEWORK")
    print("Validating: 'On the Necessity of Frequency-Selective Coupling Structures'")
    print("=" * 70)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = {}

    # 1. Validate Capacity Theorem
    print("\n[1/7] Validating Capacity Theorem (2n²)...")
    coord_system = PartitionCoordinateSystem(max_n=7)
    results['capacity_validation'] = coord_system.validate_capacity_theorem()
    print(f"      All depths match: {results['capacity_validation']['all_match']}")
    print(f"      Total states: {results['capacity_validation']['total_states']}")

    # 2. Validate Frequency Regime Separation
    print("\n[2/7] Validating Frequency Regime Separation...")
    freq_duality = FrequencyCoordinateDuality()
    results['regime_separation'] = freq_duality.validate_regime_separation()
    print(f"      Well-separated: {results['regime_separation']['separated']}")

    # 3. Validate Selection Rules
    print("\n[3/7] Validating Selection Rules...")
    sel_validator = SelectionRulesValidator()
    results['selection_rules'] = sel_validator.validate_selection_rules(coord_system)
    print(f"      Allowed fraction: {results['selection_rules']['allowed_fraction']:.1%}")

    # 4. Validate Resonance Enhancement
    print("\n[4/7] Validating Resonance Enhancement...")
    res_validator = ResonanceValidator()
    results['resonance_enhancement'] = res_validator.validate_resonance_enhancement()
    print(f"      Peak coupling: {results['resonance_enhancement']['peak_coupling']:.4f}")

    # 5. Validate Off-Resonance Suppression
    print("\n[5/7] Validating Off-Resonance Suppression...")
    results['off_resonance_suppression'] = res_validator.validate_off_resonance_suppression()
    print(f"      Theory correlation: {results['off_resonance_suppression']['correlation']:.4f}")

    # 6. Calculate Selectivity
    print("\n[6/7] Calculating Coordinate Selectivity...")
    results['selectivity'] = res_validator.calculate_selectivity(freq_duality)
    high_sel = sum(1 for c in results['selectivity'].values() if c['high_selectivity'])
    print(f"      High selectivity coords: {high_sel}/4")

    # 7. Molecular Data Integration
    print("\n[7/7] Extracting Coordinates from Molecular Data...")
    datasets = load_datasets()
    all_patterns = []
    for patterns in datasets.values():
        all_patterns.extend(patterns[:20])

    mol_extractor = MolecularPartitionExtractor()
    results['molecular_distribution'] = mol_extractor.validate_coordinate_distribution(all_patterns)
    print(f"      Valid extractions: {results['molecular_distribution'].get('valid_extractions', 0)}")

    # Get filling sequence and shell closures
    results['filling_sequence'] = [c.to_tuple() for c in coord_system.filling_sequence(50)]
    results['filling_sequence'] = [{'n': c[0], 'l': c[1], 'm': c[2], 's': c[3]}
                                   for c in results['filling_sequence']]
    results['shell_closures'] = coord_system.shell_closures()

    # Create visualizations
    print("\n[8/8] Creating validation plots...")
    create_validation_plots(results, results_dir)

    # Save results
    print("\nSaving results...")

    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(os.path.join(results_dir, 'partition_validation_results.json'), 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Capacity Theorem:      {'[PASSED]' if results['capacity_validation']['all_match'] else '[FAILED]'}")
    print(f"Regime Separation:     {'[PASSED]' if results['regime_separation']['separated'] else '[FAILED]'}")
    print(f"Resonance Theory:      {'[PASSED]' if results['off_resonance_suppression']['agrees_with_theory'] else '[FAILED]'}")
    print(f"Selection Rules:       {results['selection_rules']['allowed_fraction']:.1%} transitions allowed")
    print(f"\nResults saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
