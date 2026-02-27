"""
Oscillator Coordinate Mapping Module

Maps spectral data (UV-Vis, chromatography) to entropy coordinate space S = [0,1]³.
Implements the spectroscopy → oscillator → S-space pipeline from trajectory completion theory.

Author: Trajectory Completion Framework
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram
from scipy.fft import fft, fftfreq
from typing import Tuple, Dict, List
import json
from pathlib import Path


class SpectralOscillatorMapper:
    """Maps spectral measurements to oscillatory coordinates in S-space."""

    def __init__(self, max_frequency: float = 1e15):
        """
        Initialize mapper.

        Args:
            max_frequency: Maximum characteristic frequency for normalization (Hz)
        """
        self.max_frequency = max_frequency
        self.c = 299792458  # Speed of light (m/s)
        self.h = 6.62607015e-34  # Planck constant (J·s)
        self.kb = 1.380649e-23  # Boltzmann constant (J/K)

    def wavelength_to_frequency(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """Convert wavelength (nm) to frequency (Hz)."""
        wavelength_m = wavelength_nm * 1e-9
        return self.c / wavelength_m

    def extract_oscillators_from_spectrum(self,
                                         wavelengths: np.ndarray,
                                         absorbances: np.ndarray,
                                         prominence: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Extract oscillator frequencies from absorption spectrum.

        Peaks in absorption spectrum correspond to resonant frequencies.

        Args:
            wavelengths: Wavelength values (nm)
            absorbances: Absorbance values
            prominence: Minimum peak prominence

        Returns:
            Dictionary with frequencies, amplitudes, and peak indices
        """
        # Find peaks (resonances)
        peaks, properties = find_peaks(absorbances, prominence=prominence)

        if len(peaks) == 0:
            # If no peaks found, use prominent features
            peaks = np.argsort(absorbances)[-5:]  # Top 5 absorbance points

        peak_wavelengths = wavelengths[peaks]
        peak_absorbances = absorbances[peaks]
        peak_frequencies = self.wavelength_to_frequency(peak_wavelengths)

        return {
            'frequencies': peak_frequencies,
            'amplitudes': peak_absorbances,
            'wavelengths': peak_wavelengths,
            'indices': peaks
        }

    def extract_oscillators_from_chromatogram(self,
                                             times: np.ndarray,
                                             signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract oscillatory components from chromatographic signal.

        Uses FFT to identify frequency components.

        Args:
            times: Time values (min)
            signal: Signal intensity (mAU)

        Returns:
            Dictionary with frequencies and amplitudes
        """
        # Convert time to seconds
        times_s = times * 60

        # Remove DC component
        signal_centered = signal - np.mean(signal)

        # Compute FFT
        N = len(signal_centered)
        dt = np.mean(np.diff(times_s))

        fft_vals = fft(signal_centered)
        freqs = fftfreq(N, dt)

        # Only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        amplitudes_pos = np.abs(fft_vals[pos_mask])

        # Find dominant frequencies
        top_k = min(20, len(freqs_pos))
        top_indices = np.argsort(amplitudes_pos)[-top_k:]

        return {
            'frequencies': freqs_pos[top_indices],
            'amplitudes': amplitudes_pos[top_indices]
        }

    def compute_Sk(self, oscillators: Dict[str, np.ndarray]) -> float:
        """
        Compute kinetic/knowledge coordinate S_k.

        S_k encodes partition configuration - weighted by amplitude distribution.

        Args:
            oscillators: Dictionary with 'frequencies' and 'amplitudes'

        Returns:
            S_k ∈ [0, 1]
        """
        frequencies = oscillators['frequencies']
        amplitudes = oscillators['amplitudes']

        if len(frequencies) == 0:
            return 0.0

        # Normalize amplitudes
        amp_norm = amplitudes / (np.sum(amplitudes) + 1e-10)

        # S_k: weighted mean of normalized frequencies (partition occupancy)
        freq_norm = frequencies / self.max_frequency
        freq_norm = np.clip(freq_norm, 0, 1)

        Sk = np.sum(amp_norm * freq_norm)
        return float(np.clip(Sk, 0, 1))

    def compute_St(self, oscillators: Dict[str, np.ndarray]) -> float:
        """
        Compute temporal coordinate S_t.

        S_t encodes characteristic oscillation frequency.

        Args:
            oscillators: Dictionary with 'frequencies' and 'amplitudes'

        Returns:
            S_t ∈ [0, 1]
        """
        frequencies = oscillators['frequencies']
        amplitudes = oscillators['amplitudes']

        if len(frequencies) == 0:
            return 0.0

        # Normalize amplitudes
        amp_norm = amplitudes / (np.sum(amplitudes) + 1e-10)

        # S_t: Mean frequency weighted by amplitude
        mean_freq = np.sum(frequencies * amp_norm)
        St = mean_freq / self.max_frequency

        return float(np.clip(St, 0, 1))

    def compute_Se(self, oscillators: Dict[str, np.ndarray]) -> float:
        """
        Compute evolution coordinate S_e.

        S_e encodes partition depth (number of distinct oscillators).

        Args:
            oscillators: Dictionary with 'frequencies' and 'amplitudes'

        Returns:
            S_e ∈ [0, 1]
        """
        frequencies = oscillators['frequencies']
        amplitudes = oscillators['amplitudes']

        if len(frequencies) == 0:
            return 0.0

        # Partition depth M = Σ log₃(kᵢ)
        # Approximate by logarithm of number of oscillators
        N_osc = len(frequencies)

        # Also consider frequency diversity (entropy)
        amp_norm = amplitudes / (np.sum(amplitudes) + 1e-10)
        entropy = -np.sum(amp_norm * np.log(amp_norm + 1e-10))

        # Combine count and entropy
        max_N = 100  # Assumed maximum oscillators
        max_entropy = np.log(max_N)

        Se = 0.5 * (N_osc / max_N) + 0.5 * (entropy / max_entropy)

        return float(np.clip(Se, 0, 1))

    def map_to_sspace(self, oscillators: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """
        Map oscillator data to S-space coordinates.

        Args:
            oscillators: Dictionary with 'frequencies' and 'amplitudes'

        Returns:
            Tuple (S_k, S_t, S_e)
        """
        Sk = self.compute_Sk(oscillators)
        St = self.compute_St(oscillators)
        Se = self.compute_Se(oscillators)

        return (Sk, St, Se)

    def compute_categorical_resolution(self, oscillators: Dict[str, np.ndarray]) -> float:
        """
        Compute categorical temporal resolution τ_cat.

        From Theorem 12.2: τ_cat = 2π / (N * ⟨ω⟩)

        Args:
            oscillators: Dictionary with 'frequencies' and 'amplitudes'

        Returns:
            Categorical resolution τ_cat (seconds)
        """
        frequencies = oscillators['frequencies']
        amplitudes = oscillators['amplitudes']

        if len(frequencies) == 0:
            return float('inf')

        N_osc = len(frequencies)
        amp_norm = amplitudes / (np.sum(amplitudes) + 1e-10)
        mean_omega = 2 * np.pi * np.sum(frequencies * amp_norm)

        tau_cat = (2 * np.pi) / (N_osc * mean_omega)

        return tau_cat

    def process_uv_vis_file(self, filepath: str) -> Dict:
        """
        Process UV-Vis spectrum file and extract S-space coordinates.

        Args:
            filepath: Path to CSV file (wavelength, absorbance)

        Returns:
            Dictionary with coordinates and metadata
        """
        df = pd.read_csv(filepath)

        # Parse columns (handle different formats)
        if 'Wavelength (nm)' in df.columns:
            wavelengths = df['Wavelength (nm)'].values
            absorbances = df.iloc[:, 1].values
        else:
            wavelengths = df.iloc[:, 0].values
            absorbances = df.iloc[:, 1].values

        # Extract oscillators
        oscillators = self.extract_oscillators_from_spectrum(wavelengths, absorbances)

        # Map to S-space
        Sk, St, Se = self.map_to_sspace(oscillators)

        # Compute resolution
        tau_cat = self.compute_categorical_resolution(oscillators)

        return {
            'file': Path(filepath).name,
            'type': 'uv_vis',
            'coordinates': {
                'S_k': Sk,
                'S_t': St,
                'S_e': Se
            },
            'categorical_resolution': tau_cat,
            'n_oscillators': len(oscillators['frequencies']),
            'oscillators': {
                'frequencies': oscillators['frequencies'].tolist(),
                'amplitudes': oscillators['amplitudes'].tolist(),
                'wavelengths': oscillators['wavelengths'].tolist()
            }
        }

    def process_chromatogram_file(self, filepath: str) -> Dict:
        """
        Process chromatography file and extract S-space coordinates.

        Args:
            filepath: Path to CSV file (time, signal)

        Returns:
            Dictionary with coordinates and metadata
        """
        df = pd.read_csv(filepath)

        # Parse columns
        times = df.iloc[:, 0].values
        signal = df.iloc[:, 1].values

        # Extract oscillators
        oscillators = self.extract_oscillators_from_chromatogram(times, signal)

        # Map to S-space
        Sk, St, Se = self.map_to_sspace(oscillators)

        # Compute resolution
        tau_cat = self.compute_categorical_resolution(oscillators)

        return {
            'file': Path(filepath).name,
            'type': 'chromatogram',
            'coordinates': {
                'S_k': Sk,
                'S_t': St,
                'S_e': Se
            },
            'categorical_resolution': tau_cat,
            'n_oscillators': len(oscillators['frequencies']),
            'oscillators': {
                'frequencies': oscillators['frequencies'].tolist(),
                'amplitudes': oscillators['amplitudes'].tolist()
            }
        }


def main():
    """Example usage."""
    mapper = SpectralOscillatorMapper()

    # Process UV-Vis spectrum
    results = []

    data_dir = Path(__file__).parent.parent.parent / 'public' / 'spectra'

    # Process UV_Vis_1.csv
    uv_file = data_dir / 'UV_Vis_1.csv'
    if uv_file.exists():
        result = mapper.process_uv_vis_file(str(uv_file))
        results.append(result)
        print(f"UV-Vis spectrum: S = ({result['coordinates']['S_k']:.4f}, "
              f"{result['coordinates']['S_t']:.4f}, {result['coordinates']['S_e']:.4f})")
        print(f"  N_oscillators = {result['n_oscillators']}")
        print(f"  tau_cat = {result['categorical_resolution']:.2e} s")

    # Process chromatogram files
    for chrom_file in ['214.csv', '280.csv', '300.csv', '400.csv', '420.csv']:
        filepath = data_dir / chrom_file
        if filepath.exists():
            result = mapper.process_chromatogram_file(str(filepath))
            results.append(result)
            print(f"\nChromatogram {chrom_file}: S = ({result['coordinates']['S_k']:.4f}, "
                  f"{result['coordinates']['S_t']:.4f}, {result['coordinates']['S_e']:.4f})")

    # Save results
    output_file = Path(__file__).parent.parent.parent / 'results' / 'oscillator_mapping_results.json'
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
