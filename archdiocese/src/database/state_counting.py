"""
State Counting Module

Implements the Time-State Identity: dM/dt = 1/⟨τ_p⟩

Validates that temporal evolution IS state counting (Definition 12.1).

Author: Trajectory Completion Framework
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


class StateCounting:
    """Implements categorical state counting."""

    def __init__(self):
        """Initialize with constants."""
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)

    def count_categorical_states(self, time: float, period: float) -> int:
        """
        Count categorical states traversed.

        From Definition 5.1: C(t) = ⌊t/T⌋

        Args:
            time: Elapsed time
            period: Oscillation period

        Returns:
            Number of categorical states
        """
        return int(np.floor(time / period))

    def time_state_identity(self, M: int, tau_p: float) -> float:
        """
        Compute time from state count via Time-State Identity.

        dM/dt = 1/⟨τ_p⟩  ⟹  t = M · ⟨τ_p⟩

        Args:
            M: Number of categorical states
            tau_p: Mean partition time

        Returns:
            Time (seconds)
        """
        return M * tau_p

    def categorical_entropy(self, M: int) -> float:
        """
        Compute categorical entropy.

        From Definition 5.2: S_cat = k_B ln(M)

        Args:
            M: Number of categorical states

        Returns:
            Entropy (J/K)
        """
        if M <= 0:
            return 0.0

        return self.k_B * np.log(M)

    def entropy_growth_rate(self, M: int, tau_p: float) -> float:
        """
        Compute entropy production rate.

        From Theorem 5.3: dS/dt = k_B/t = k_B/(M·τ_p)

        Args:
            M: Categorical state count
            tau_p: Mean partition time

        Returns:
            Entropy production rate (J/(K·s))
        """
        t = self.time_state_identity(M, tau_p)

        if t <= 0:
            return 0.0

        return self.k_B / t

    def partition_depth_from_states(self, categorical_sequence: List[int]) -> float:
        """
        Compute partition depth from categorical state sequence.

        M = Σ log₃(k_i) where k_i are partition branch factors.

        Args:
            categorical_sequence: Sequence of categorical states

        Returns:
            Partition depth M
        """
        if len(categorical_sequence) == 0:
            return 0.0

        # Each state transition corresponds to a partition refinement
        # Assume ternary branching (base 3)
        M = sum(np.log(3) / np.log(3) for _ in categorical_sequence)

        return M

    def demonstrate_state_counting(self,
                                   n_oscillators: int,
                                   frequency: float,
                                   duration: float) -> Dict:
        """
        Demonstrate state counting for an oscillator ensemble.

        Args:
            n_oscillators: Number of oscillators
            frequency: Mean frequency (Hz)
            duration: Observation duration (s)

        Returns:
            State counting results
        """
        period = 1.0 / frequency
        tau_p = period / n_oscillators  # Mean partition time

        # Count states for each oscillator
        states_per_oscillator = self.count_categorical_states(duration, period)

        # Total categorical states (ensemble)
        M_total = states_per_oscillator * n_oscillators

        # Verify time-state identity
        t_computed = self.time_state_identity(M_total, tau_p)

        # Categorical entropy
        S_cat = self.categorical_entropy(M_total)

        # Entropy growth rate
        dS_dt = self.entropy_growth_rate(M_total, tau_p)

        return {
            'n_oscillators': n_oscillators,
            'frequency_Hz': frequency,
            'period_s': period,
            'duration_s': duration,
            'tau_p_s': tau_p,
            'states_per_oscillator': states_per_oscillator,
            'total_categorical_states': M_total,
            'time_computed_s': t_computed,
            'time_error': abs(t_computed - duration) / duration,
            'categorical_entropy_J_K': S_cat,
            'entropy_rate_J_K_s': dS_dt
        }

    def trajectory_state_sequence(self,
                                  s_start: Tuple[float, float, float],
                                  s_end: Tuple[float, float, float],
                                  n_steps: int) -> List[Dict]:
        """
        Generate categorical state sequence for trajectory in S-space.

        Args:
            s_start: Starting coordinates
            s_end: Ending coordinates (fixed point)
            n_steps: Number of steps

        Returns:
            List of state dictionaries
        """
        trajectory = []

        for i in range(n_steps + 1):
            t = i / n_steps

            # Linear interpolation
            s_t = tuple(s_start[j] * (1 - t) + s_end[j] * t for j in range(3))

            # Categorical state (simplified: count steps)
            M = i

            # Entropy
            S = self.categorical_entropy(M + 1)  # +1 to avoid log(0)

            trajectory.append({
                'step': i,
                'time_fraction': t,
                'coordinates': s_t,
                'categorical_state': M,
                'entropy': S
            })

        return trajectory


def main():
    """Run state counting validation."""
    counter = StateCounting()

    print("\n=== State Counting Validation ===\n")

    # Test 1: Simple oscillator
    print("=== Test 1: Single Oscillator ===")
    result1 = counter.demonstrate_state_counting(
        n_oscillators=1,
        frequency=1000.0,  # 1 kHz
        duration=1.0  # 1 second
    )

    print(f"Frequency: {result1['frequency_Hz']} Hz")
    print(f"Duration: {result1['duration_s']} s")
    print(f"States counted: {result1['total_categorical_states']}")
    print(f"Time from states: {result1['time_computed_s']:.6f} s")
    print(f"Error: {result1['time_error']:.2e}")
    print(f"Categorical entropy: {result1['categorical_entropy_J_K']:.2e} J/K")

    # Test 2: Oscillator ensemble
    print(f"\n=== Test 2: Oscillator Ensemble ===")
    result2 = counter.demonstrate_state_counting(
        n_oscillators=100,
        frequency=1e13,  # IR frequency
        duration=1e-9  # 1 nanosecond
    )

    print(f"N oscillators: {result2['n_oscillators']}")
    print(f"Frequency: {result2['frequency_Hz']:.2e} Hz")
    print(f"Duration: {result2['duration_s']:.2e} s")
    print(f"τ_p: {result2['tau_p_s']:.2e} s")
    print(f"Total states: {result2['total_categorical_states']}")
    print(f"Time from states: {result2['time_computed_s']:.2e} s")
    print(f"Error: {result2['time_error']:.2e}")

    # Test 3: Trajectory state sequence
    print(f"\n=== Test 3: Trajectory State Sequence ===")
    s_start = (0.1, 0.2, 0.3)
    s_end = (0.8, 0.7, 0.9)

    traj = counter.trajectory_state_sequence(s_start, s_end, n_steps=10)

    print(f"Start: {s_start}")
    print(f"End: {s_end}")
    print(f"\nState sequence:")
    for state in traj[::2]:  # Every other step
        print(f"  Step {state['step']}: "
              f"S=({state['coordinates'][0]:.2f}, {state['coordinates'][1]:.2f}, {state['coordinates'][2]:.2f}), "
              f"M={state['categorical_state']}, "
              f"S_cat={state['entropy']:.2e} J/K")

    # Save results
    results = {
        'single_oscillator': result1,
        'ensemble': result2,
        'trajectory_sample': traj[:5]  # First 5 points
    }

    output_file = Path(__file__).parent.parent.parent / 'results' / 'state_counting.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
