import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MolecularCheminformaticsAnalyzer:
    def __init__(self, json_file_path):
        """Initialize with molecular cheminformatics data"""
        print(f"Loading molecular data from {json_file_path}...")
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)

        self.molecules = self.data['molecular_data']['molecules']
        self.n_molecules = len(self.molecules)
        print(f"Loaded {self.n_molecules} molecules for analysis")

        # Extract global molecular properties
        self.extract_molecular_properties()

    def extract_molecular_properties(self):
        """Extract key molecular properties for global analysis"""
        self.molecular_properties = {
            'molecular_id': [],
            'smiles': [],
            'molecular_weight': [],
            'logp': [],
            'tpsa': [],
            'base_frequency_hz': [],
            'temporal_precision_seconds': [],
            'frequency_stability': [],
            'processing_rate_ops_per_sec': [],
            'memory_capacity_bits': [],
            'parallel_processing': []
        }

        for mol in self.molecules:
            self.molecular_properties['molecular_id'].append(mol['molecular_id'])
            self.molecular_properties['smiles'].append(mol['smiles'])
            self.molecular_properties['molecular_weight'].append(mol['molecular_weight'])
            self.molecular_properties['logp'].append(mol['logp'])
            self.molecular_properties['tpsa'].append(mol['tpsa'])
            self.molecular_properties['base_frequency_hz'].append(mol['clock_properties']['base_frequency_hz'])
            self.molecular_properties['temporal_precision_seconds'].append(
                mol['clock_properties']['temporal_precision_seconds'])
            self.molecular_properties['frequency_stability'].append(mol['clock_properties']['frequency_stability'])
            self.molecular_properties['processing_rate_ops_per_sec'].append(
                mol['processor_properties']['processing_rate_ops_per_sec'])
            self.molecular_properties['memory_capacity_bits'].append(
                mol['processor_properties']['memory_capacity_bits'])
            self.molecular_properties['parallel_processing'].append(mol['processor_properties']['parallel_processing'])

        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.molecular_properties)
        print("Molecular properties extracted successfully")

    def global_molecular_overview(self):
        """Create comprehensive overview of molecular properties"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Global Molecular Cheminformatics Overview', fontsize=16, fontweight='bold')

        # Molecular Weight Distribution
        axes[0, 0].hist(self.df['molecular_weight'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Molecular Weight Distribution')
        axes[0, 0].set_xlabel('Molecular Weight (g/mol)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # LogP Distribution
        axes[0, 1].hist(self.df['logp'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('LogP Distribution')
        axes[0, 1].set_xlabel('LogP')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # TPSA Distribution
        axes[0, 2].hist(self.df['tpsa'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('TPSA Distribution')
        axes[0, 2].set_xlabel('TPSA (Ų)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)

        # Base Frequency Distribution
        axes[1, 0].hist(self.df['base_frequency_hz'] / 1e12, bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 0].set_title('Base Frequency Distribution')
        axes[1, 0].set_xlabel('Base Frequency (THz)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Temporal Precision Distribution
        axes[1, 1].hist(self.df['temporal_precision_seconds'] * 1e26, bins=30, alpha=0.7, color='purple',
                        edgecolor='black')
        axes[1, 1].set_title('Temporal Precision Distribution')
        axes[1, 1].set_xlabel('Temporal Precision (×10⁻²⁶ s)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # Frequency Stability Distribution
        axes[1, 2].hist(self.df['frequency_stability'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 2].set_title('Frequency Stability Distribution')
        axes[1, 2].set_xlabel('Frequency Stability')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)

        # Processing Rate Distribution
        axes[2, 0].hist(self.df['processing_rate_ops_per_sec'] / 1e6, bins=30, alpha=0.7, color='cyan',
                        edgecolor='black')
        axes[2, 0].set_title('Processing Rate Distribution')
        axes[2, 0].set_xlabel('Processing Rate (MOps/s)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)

        # Memory Capacity Distribution
        axes[2, 1].hist(self.df['memory_capacity_bits'] / 1e3, bins=30, alpha=0.7, color='pink', edgecolor='black')
        axes[2, 1].set_title('Memory Capacity Distribution')
        axes[2, 1].set_xlabel('Memory Capacity (kbits)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)

        # Parallel Processing Distribution
        parallel_counts = self.df['parallel_processing'].value_counts()
        axes[2, 2].pie(parallel_counts.values, labels=['Parallel', 'Sequential'],
                       autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[2, 2].set_title('Parallel Processing Distribution')

        plt.tight_layout()
        return fig

    def correlation_analysis(self):
        """Analyze correlations between molecular properties"""
        # Select numerical columns for correlation
        numerical_cols = ['molecular_weight', 'logp', 'tpsa', 'base_frequency_hz',
                          'temporal_precision_seconds', 'frequency_stability',
                          'processing_rate_ops_per_sec', 'memory_capacity_bits']

        correlation_matrix = self.df[numerical_cols].corr()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Molecular Properties Correlation Analysis', fontsize=16, fontweight='bold')

        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax1, fmt='.3f')
        ax1.set_title('Correlation Matrix Heatmap')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # Strongest correlations scatter plot
        # Find the strongest correlation (excluding diagonal)
        corr_values = correlation_matrix.values
        np.fill_diagonal(corr_values, 0)
        max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_values)), corr_values.shape)

        col1 = numerical_cols[max_corr_idx[0]]
        col2 = numerical_cols[max_corr_idx[1]]
        corr_value = correlation_matrix.loc[col1, col2]

        ax2.scatter(self.df[col1], self.df[col2], alpha=0.6, s=50)
        ax2.set_xlabel(col1.replace('_', ' ').title())
        ax2.set_ylabel(col2.replace('_', ' ').title())
        ax2.set_title(f'Strongest Correlation: {col1} vs {col2}\nr = {corr_value:.3f}')
        ax2.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(self.df[col1], self.df[col2], 1)
        p = np.poly1d(z)
        ax2.plot(self.df[col1], p(self.df[col1]), "r--", alpha=0.8)

        plt.tight_layout()
        return fig, correlation_matrix

    def molecular_clustering_analysis(self, n_clusters=4):
        """Perform clustering analysis on molecular properties"""
        # Prepare data for clustering
        feature_cols = ['molecular_weight', 'logp', 'tpsa', 'base_frequency_hz',
                        'frequency_stability', 'processing_rate_ops_per_sec', 'memory_capacity_bits']

        X = self.df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Molecular Clustering Analysis (k={n_clusters})', fontsize=16, fontweight='bold')

        # PCA scatter plot with clusters
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                                     cmap='tab10', alpha=0.7, s=50)
        axes[0, 0].set_title('PCA Visualization of Molecular Clusters')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 0])

        # Cluster characteristics
        self.df['cluster'] = cluster_labels
        cluster_stats = self.df.groupby('cluster')[feature_cols].mean()

        sns.heatmap(cluster_stats.T, annot=True, cmap='viridis', ax=axes[0, 1], fmt='.2f')
        axes[0, 1].set_title('Cluster Characteristics (Mean Values)')
        axes[0, 1].set_ylabel('Features')

        # Cluster size distribution
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        axes[1, 0].bar(range(n_clusters), cluster_counts.values, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Molecules')
        axes[1, 0].grid(True, alpha=0.3)

        # Feature importance in clustering (based on cluster centers spread)
        feature_importance = np.std(kmeans.cluster_centers_, axis=0)
        axes[1, 1].bar(range(len(feature_cols)), feature_importance, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Feature Importance in Clustering')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Standard Deviation of Cluster Centers')
        axes[1, 1].set_xticks(range(len(feature_cols)))
        axes[1, 1].set_xticklabels([col.replace('_', '\n') for col in feature_cols], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, kmeans, pca

    def clock_time_series_global_analysis(self):
        """Analyze clock time series data globally across all molecules"""
        print("Analyzing clock time series data...")

        # Extract time series data from first few molecules (for performance)
        sample_molecules = self.data['molecular_data']['clock_time_series'][:10]  # Sample first 10

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Global Clock Time Series Analysis', fontsize=16, fontweight='bold')

        all_oscillations = []
        all_frequencies = []

        # Process each molecule's time series
        for i, mol_ts in enumerate(sample_molecules):
            if i >= 5:  # Limit to first 5 for visualization
                break

            time_points = np.array(mol_ts['time_points_seconds'][:1000])  # First 1000 points
            oscillations = np.array(mol_ts['oscillation_measurements'][:1000])

            all_oscillations.extend(oscillations)

            # Plot individual time series
            axes[0, 0].plot(time_points * 1e12, oscillations, alpha=0.7, linewidth=0.8,
                            label=f"Mol {i + 1}")

        axes[0, 0].set_title('Sample Molecular Oscillations')
        axes[0, 0].set_xlabel('Time (ps)')
        axes[0, 0].set_ylabel('Oscillation Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Global oscillation distribution
        axes[0, 1].hist(all_oscillations, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Global Oscillation Amplitude Distribution')
        axes[0, 1].set_xlabel('Oscillation Amplitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Frequency analysis for one representative molecule
        if sample_molecules:
            mol_data = sample_molecules[0]
            time_points = np.array(mol_data['time_points_seconds'][:2000])
            oscillations = np.array(mol_data['oscillation_measurements'][:2000])

            dt = time_points[1] - time_points[0]
            fft_vals = fft(oscillations - np.mean(oscillations))
            freqs = fftfreq(len(oscillations), dt)

            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask] / 1e12  # Convert to THz
            fft_pos = np.abs(fft_vals[pos_mask])

            axes[0, 2].loglog(freqs_pos, fft_pos)
            axes[0, 2].set_title('Representative Frequency Spectrum')
            axes[0, 2].set_xlabel('Frequency (THz)')
            axes[0, 2].set_ylabel('Magnitude')
            axes[0, 2].grid(True, alpha=0.3)

        # Base frequency vs other properties
        axes[1, 0].scatter(self.df['base_frequency_hz'] / 1e12, self.df['molecular_weight'],
                           alpha=0.6, s=50, c=self.df['logp'], cmap='viridis')
        axes[1, 0].set_xlabel('Base Frequency (THz)')
        axes[1, 0].set_ylabel('Molecular Weight')
        axes[1, 0].set_title('Base Frequency vs Molecular Weight')
        axes[1, 0].grid(True, alpha=0.3)

        # Frequency stability vs processing rate
        axes[1, 1].scatter(self.df['frequency_stability'], self.df['processing_rate_ops_per_sec'] / 1e6,
                           alpha=0.6, s=50, c=self.df['tpsa'], cmap='plasma')
        axes[1, 1].set_xlabel('Frequency Stability')
        axes[1, 1].set_ylabel('Processing Rate (MOps/s)')
        axes[1, 1].set_title('Frequency Stability vs Processing Rate')
        axes[1, 1].grid(True, alpha=0.3)

        # Temporal precision distribution
        axes[1, 2].boxplot([self.df[self.df['parallel_processing'] == True]['temporal_precision_seconds'] * 1e26,
                            self.df[self.df['parallel_processing'] == False]['temporal_precision_seconds'] * 1e26],
                           labels=['Parallel', 'Sequential'])
        axes[1, 2].set_title('Temporal Precision by Processing Type')
        axes[1, 2].set_ylabel('Temporal Precision (×10⁻²⁶ s)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def interactive_molecular_dashboard(self):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Molecular Weight vs LogP', 'Base Frequency vs TPSA',
                            'Processing Rate vs Memory', 'Frequency Stability Distribution',
                            'Molecular Property Correlations', 'Cluster Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        # Molecular Weight vs LogP
        fig.add_trace(
            go.Scatter(x=self.df['molecular_weight'], y=self.df['logp'],
                       mode='markers', name='Molecules',
                       marker=dict(size=8, color=self.df['tpsa'], colorscale='Viridis',
                                   showscale=True, colorbar=dict(title="TPSA"))),
            row=1, col=1
        )

        # Base Frequency vs TPSA
        fig.add_trace(
            go.Scatter(x=self.df['base_frequency_hz'] / 1e12, y=self.df['tpsa'],
                       mode='markers', name='Frequency vs TPSA',
                       marker=dict(size=8, color=self.df['frequency_stability'],
                                   colorscale='Plasma', showscale=False)),
            row=1, col=2
        )

        # Processing Rate vs Memory
        fig.add_trace(
            go.Scatter(x=self.df['processing_rate_ops_per_sec'] / 1e6,
                       y=self.df['memory_capacity_bits'] / 1e3,
                       mode='markers', name='Processing vs Memory',
                       marker=dict(size=8, color=self.df['molecular_weight'],
                                   colorscale='Cividis', showscale=False)),
            row=1, col=3
        )

        # Frequency Stability Distribution
        fig.add_trace(
            go.Histogram(x=self.df['frequency_stability'], name='Frequency Stability',
                         marker_color='lightblue', opacity=0.7),
            row=2, col=1
        )

        # Add more traces for remaining subplots...

        fig.update_layout(
            title_text="Interactive Molecular Cheminformatics Dashboard",
            showlegend=False,
            height=800
        )

        return fig

    def generate_global_summary_report(self):
        """Generate comprehensive summary statistics"""
        report = {
            'dataset_info': {
                'total_molecules': self.n_molecules,
                'validation_type': self.data['validation_type'],
                'execution_time': self.data['total_execution_time']
            },
            'molecular_statistics': {},
            'clock_statistics': {},
            'processor_statistics': {}
        }

        # Molecular property statistics
        mol_props = ['molecular_weight', 'logp', 'tpsa']
        for prop in mol_props:
            report['molecular_statistics'][prop] = {
                'mean': float(self.df[prop].mean()),
                'std': float(self.df[prop].std()),
                'min': float(self.df[prop].min()),
                'max': float(self.df[prop].max()),
                'median': float(self.df[prop].median())
            }

        # Clock property statistics
        clock_props = ['base_frequency_hz', 'temporal_precision_seconds', 'frequency_stability']
        for prop in clock_props:
            report['clock_statistics'][prop] = {
                'mean': float(self.df[prop].mean()),
                'std': float(self.df[prop].std()),
                'min': float(self.df[prop].min()),
                'max': float(self.df[prop].max())
            }

        # Processor property statistics
        proc_props = ['processing_rate_ops_per_sec', 'memory_capacity_bits']
        for prop in proc_props:
            report['processor_statistics'][prop] = {
                'mean': float(self.df[prop].mean()),
                'std': float(self.df[prop].std()),
                'min': float(self.df[prop].min()),
                'max': float(self.df[prop].max())
            }

        # Parallel processing distribution
        parallel_dist = self.df['parallel_processing'].value_counts()
        report['processor_statistics']['parallel_processing_distribution'] = {
            'parallel': int(parallel_dist.get(True, 0)),
            'sequential': int(parallel_dist.get(False, 0)),
            'parallel_percentage': float(parallel_dist.get(True, 0) / len(self.df) * 100)
        }

        return report


# Usage function
def analyze_molecular_cheminformatics(json_file_path):
    """Main function to run all molecular analyses"""
    analyzer = MolecularCheminformaticsAnalyzer(json_file_path)

    print("1. Generating global molecular overview...")
    fig1 = analyzer.global_molecular_overview()
    fig1.savefig('molecular_global_overview.png', dpi=300, bbox_inches='tight')

    print("2. Performing correlation analysis...")
    fig2, corr_matrix = analyzer.correlation_analysis()
    fig2.savefig('molecular_correlation_analysis.png', dpi=300, bbox_inches='tight')

    print("3. Running clustering analysis...")
    fig3, kmeans_model, pca_model = analyzer.molecular_clustering_analysis()
    fig3.savefig('molecular_clustering_analysis.png', dpi=300, bbox_inches='tight')

    print("4. Analyzing clock time series...")
    fig4 = analyzer.clock_time_series_global_analysis()
    fig4.savefig('clock_time_series_global.png', dpi=300, bbox_inches='tight')

    print("5. Creating interactive dashboard...")
    interactive_fig = analyzer.interactive_molecular_dashboard()
    interactive_fig.write_html('molecular_interactive_dashboard.html')

    print("6. Generating summary report...")
    report = analyzer.generate_global_summary_report()

    print("\n=== MOLECULAR CHEMINFORMATICS GLOBAL ANALYSIS REPORT ===")
    print(f"Dataset: {report['dataset_info']['total_molecules']} molecules")
    print(f"Validation Type: {report['dataset_info']['validation_type']}")
    print(f"Execution Time: {report['dataset_info']['execution_time']:.3f} seconds")

    print("\nMolecular Properties Summary:")
    for prop, stats in report['molecular_statistics'].items():
        print(f"  {prop.replace('_', ' ').title()}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"    Std: {stats['std']:.2f}")

    print("\nClock Properties Summary:")
    for prop, stats in report['clock_statistics'].items():
        if 'frequency' in prop and 'hz' in prop:
            print(f"  {prop.replace('_', ' ').title()}: {stats['mean'] / 1e12:.2f} ± {stats['std'] / 1e12:.2f} THz")
        elif 'precision' in prop:
            print(
                f"  {prop.replace('_', ' ').title()}: {stats['mean'] * 1e26:.2f} ± {stats['std'] * 1e26:.2f} ×10⁻²⁶ s")
        else:
            print(f"  {prop.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print(f"\nProcessor Properties:")
    print(
        f"  Parallel Processing: {report['processor_statistics']['parallel_processing_distribution']['parallel_percentage']:.1f}% parallel")
    print(
        f"  Average Processing Rate: {report['processor_statistics']['processing_rate_ops_per_sec']['mean'] / 1e6:.2f} MOps/s")
    print(
        f"  Average Memory Capacity: {report['processor_statistics']['memory_capacity_bits']['mean'] / 1e3:.2f} kbits")

    return fig1, fig2, fig3, fig4, interactive_fig, report


if __name__ == "__main__":
    # Run the complete analysis
    results = analyze_molecular_cheminformatics('real_borgia_results/real_validation_1758140958.json')
    plt.show()
