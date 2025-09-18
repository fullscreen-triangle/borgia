import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import json


class TimeSeriesAnalyzer:
    def __init__(self, clock_time_series_data):
        self.time_series_data = clock_time_series_data

    def frequency_analysis(self, molecule_idx=0, max_points=5000):
        """Perform FFT analysis on oscillation data"""
        mol_data = self.time_series_data[molecule_idx]

        # Get time and oscillation data
        time_points = np.array(mol_data['time_points_seconds'][:max_points])
        oscillations = np.array(mol_data['oscillation_measurements'][:max_points])

        # Calculate sampling frequency
        dt = time_points[1] - time_points[0]
        fs = 1 / dt

        # Perform FFT
        fft_vals = fft(oscillations)
        freqs = fftfreq(len(oscillations), dt)

        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = np.abs(fft_vals[pos_mask])

        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Time domain
        ax1.plot(time_points * 1e12, oscillations, linewidth=0.8)
        ax1.set_title(f'Time Domain - {mol_data["molecule_id"]}')
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        # Frequency domain
        ax2.loglog(freqs_pos, fft_pos)
        ax2.set_title('Frequency Domain (FFT)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True, alpha=0.3)

        # Spectrogram
        f, t, Sxx = signal.spectrogram(oscillations, fs, nperseg=256)
        im = ax3.pcolormesh(t * 1e12, f / 1e12, 10 * np.log10(Sxx), shading='gouraud')
        ax3.set_title('Spectrogram')
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Frequency (THz)')
        plt.colorbar(im, ax=ax3, label='Power (dB)')

        plt.tight_layout()
        return fig

    def compare_molecules_spectra(self, molecule_indices=[0, 1, 2, 3], max_points=2000):
        """Compare frequency spectra of multiple molecules"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Molecular Oscillation Spectra Comparison', fontsize=16)

        for i, mol_idx in enumerate(molecule_indices):
            row, col = i // 2, i % 2

            mol_data = self.time_series_data[mol_idx]
            oscillations = np.array(mol_data['oscillation_measurements'][:max_points])
            time_points = np.array(mol_data['time_points_seconds'][:max_points])

            dt = time_points[1] - time_points[0]

            # FFT
            fft_vals = fft(oscillations)
            freqs = fftfreq(len(oscillations), dt)

            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask] / 1e12  # Convert to THz
            fft_pos = np.abs(fft_vals[pos_mask])

            axes[row, col].semilogy(freqs_pos, fft_pos)
            axes[row, col].set_title(f'{mol_data["molecule_id"]}')
            axes[row, col].set_xlabel('Frequency (THz)')
            axes[row, col].set_ylabel('Magnitude')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlim(0, 10)  # Focus on first 10 THz

        plt.tight_layout()
        return fig


# Usage
def analyze_time_series(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    ts_analyzer = TimeSeriesAnalyzer(data['clock_time_series'])

    # Analyze first molecule
    fig1 = ts_analyzer.frequency_analysis(0)
    fig1.savefig('frequency_analysis_mol1.png', dpi=300, bbox_inches='tight')

    # Compare multiple molecules
    fig2 = ts_analyzer.compare_molecules_spectra()
    fig2.savefig('molecules_spectra_comparison.png', dpi=300, bbox_inches='tight')

    return fig1, fig2
