import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import json


class NetworkTimeSeriesAnalyzer:
    def __init__(self, json_file_path):
        """Initialize with network topology data"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)

        # Extract amplification dynamics time series
        self.time_series = self.data['amplification_dynamics']
        self.time_points = np.array(self.time_series['time_seconds'])
        self.amplification_values = np.array(self.time_series['amplification_factor'])

        # Extract layer efficiencies for analysis
        self.layer_efficiencies = self.data['layer_efficiencies']

        # Create synthetic time series for each layer based on efficiency
        self.create_layer_time_series()

    def create_layer_time_series(self):
        """Create synthetic time series for each network layer"""
        n_points = len(self.time_points)

        # Generate layer-specific signals based on efficiency and characteristics
        quantum_base = self.layer_efficiencies['quantum']['efficiency']
        molecular_base = self.layer_efficiencies['molecular']['efficiency']
        environmental_base = self.layer_efficiencies['environmental']['efficiency']

        # Add some realistic variations and correlations
        np.random.seed(42)  # For reproducibility

        # Quantum layer: high frequency, low amplitude variations
        quantum_noise = 0.1 * np.random.randn(n_points)
        quantum_trend = quantum_base * (1 + 0.05 * np.sin(2 * np.pi * self.time_points / 100))
        self.quantum_signal = quantum_trend + quantum_noise

        # Molecular layer: medium frequency variations
        molecular_noise = 0.05 * np.random.randn(n_points)
        molecular_trend = molecular_base * (1 + 0.03 * np.cos(2 * np.pi * self.time_points / 200))
        self.molecular_signal = molecular_trend + molecular_noise

        # Environmental layer: low frequency, large amplitude variations
        environmental_noise = 0.02 * np.random.randn(n_points)
        environmental_trend = environmental_base * (1 + 0.08 * np.sin(2 * np.pi * self.time_points / 500))
        self.environmental_signal = environmental_trend + environmental_noise

    def amplification_frequency_analysis(self, max_points=5000):
        """Perform FFT analysis on amplification dynamics"""
        # Limit points if necessary
        if len(self.time_points) > max_points:
            time_data = self.time_points[:max_points]
            amp_data = self.amplification_values[:max_points]
        else:
            time_data = self.time_points
            amp_data = self.amplification_values

        # Calculate sampling frequency
        dt = time_data[1] - time_data[0]
        fs = 1 / dt

        # Perform FFT
        fft_vals = fft(amp_data - np.mean(amp_data))  # Remove DC component
        freqs = fftfreq(len(amp_data), dt)

        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = np.abs(fft_vals[pos_mask])

        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Time domain
        ax1.plot(time_data, amp_data, linewidth=0.8, color='blue')
        ax1.set_title('Amplification Factor Time Series')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplification Factor')
        ax1.grid(True, alpha=0.3)

        # Frequency domain
        ax2.loglog(freqs_pos, fft_pos, color='red')
        ax2.set_title('Frequency Domain (FFT)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.grid(True, alpha=0.3)

        # Spectrogram
        f, t, Sxx = signal.spectrogram(amp_data, fs, nperseg=min(256, len(amp_data) // 4))
        im = ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        ax3.set_title('Spectrogram')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax3, label='Power (dB)')

        plt.tight_layout()
        return fig

    def compare_layer_signals(self):
        """Compare time series signals from different network layers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Layer Signal Comparison', fontsize=16)

        signals = [self.quantum_signal, self.molecular_signal,
                   self.environmental_signal, self.amplification_values]
        signal_names = ['Quantum Layer', 'Molecular Layer',
                        'Environmental Layer', 'Amplification Factor']
        colors = ['blue', 'red', 'green', 'purple']

        for i, (signal_data, name, color) in enumerate(zip(signals, signal_names, colors)):
            row, col = i // 2, i % 2

            # Time series plot
            axes[row, col].plot(self.time_points, signal_data,
                                color=color, linewidth=0.8, alpha=0.8)
            axes[row, col].set_title(f'{name} Signal')
            axes[row, col].set_xlabel('Time (seconds)')
            axes[row, col].set_ylabel('Signal Value')
            axes[row, col].grid(True, alpha=0.3)

            # Add statistics
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            axes[row, col].text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                                transform=axes[row, col].transAxes,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def cross_correlation_analysis(self):
        """Analyze cross-correlations between different signals"""
        signals = {
            'Quantum': self.quantum_signal,
            'Molecular': self.molecular_signal,
            'Environmental': self.environmental_signal,
            'Amplification': self.amplification_values
        }

        # Calculate cross-correlations
        signal_names = list(signals.keys())
        n_signals = len(signal_names)

        fig, axes = plt.subplots(n_signals, n_signals, figsize=(16, 16))
        fig.suptitle('Cross-Correlation Analysis Between Network Signals', fontsize=16)

        for i, name1 in enumerate(signal_names):
            for j, name2 in enumerate(signal_names):
                if i == j:
                    # Auto-correlation
                    correlation = np.correlate(signals[name1], signals[name1], mode='full')
                    lags = np.arange(-len(signals[name1]) + 1, len(signals[name1]))
                    axes[i, j].plot(lags, correlation / np.max(correlation))
                    axes[i, j].set_title(f'{name1} Autocorrelation')
                else:
                    # Cross-correlation
                    correlation = np.correlate(signals[name1], signals[name2], mode='full')
                    lags = np.arange(-len(signals[name1]) + 1, len(signals[name1]))
                    axes[i, j].plot(lags, correlation / np.max(np.abs(correlation)))
                    axes[i, j].set_title(f'{name1} vs {name2}')

                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_xlabel('Lag')
                axes[i, j].set_ylabel('Correlation')

        plt.tight_layout()
        return fig

    def wavelet_analysis(self):
        """Perform wavelet analysis on the amplification signal"""
        from scipy import signal as scipy_signal

        # Use continuous wavelet transform
        widths = np.arange(1, 31)
        cwt_matrix = scipy_signal.cwt(self.amplification_values,
                                      scipy_signal.ricker, widths)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Original signal
        ax1.plot(self.time_points, self.amplification_values)
        ax1.set_title('Original Amplification Signal')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplification Factor')
        ax1.grid(True, alpha=0.3)

        # Wavelet transform
        im = ax2.imshow(np.abs(cwt_matrix), extent=[self.time_points[0], self.time_points[-1],
                                                    widths[0], widths[-1]],
                        cmap='jet', aspect='auto', origin='lower')
        ax2.set_title('Continuous Wavelet Transform')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Scale')
        plt.colorbar(im, ax=ax2, label='Magnitude')

        plt.tight_layout()
        return fig


# Usage function
def analyze_network_time_series(json_file_path):
    """Main function to run all time series analyses"""
    ts_analyzer = NetworkTimeSeriesAnalyzer(json_file_path)

    # Analyze amplification dynamics
    fig1 = ts_analyzer.amplification_frequency_analysis()
    fig1.savefig('amplification_frequency_analysis.png', dpi=300, bbox_inches='tight')

    # Compare layer signals
    fig2 = ts_analyzer.compare_layer_signals()
    fig2.savefig('layer_signals_comparison.png', dpi=300, bbox_inches='tight')

    # Cross-correlation analysis
    fig3 = ts_analyzer.cross_correlation_analysis()
    fig3.savefig('cross_correlation_analysis.png', dpi=300, bbox_inches='tight')

    # Wavelet analysis
    fig4 = ts_analyzer.wavelet_analysis()
    fig4.savefig('wavelet_analysis.png', dpi=300, bbox_inches='tight')

    return fig1, fig2, fig3, fig4


if __name__ == "__main__":
    # Run the analysis
    figures = analyze_network_time_series('real_borgia_results/network_data_1758140958.json')
    plt.show()
