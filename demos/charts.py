import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MolecularDataAnalyzer:
    def __init__(self, json_file_path):
        """Initialize the analyzer with molecular data"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)

        self.molecules_df = pd.DataFrame(self.data['molecules'])
        self.prepare_data()

    def prepare_data(self):
        """Prepare and clean the data for analysis"""
        # Extract clock properties into separate columns
        clock_props = pd.json_normalize(self.molecules_df['clock_properties'])
        clock_props.columns = ['clock_' + col for col in clock_props.columns]

        # Extract processor properties
        proc_props = pd.json_normalize(self.molecules_df['processor_properties'])
        proc_props.columns = ['proc_' + col for col in proc_props.columns]

        # Combine all properties
        self.molecules_df = pd.concat([
            self.molecules_df.drop(['clock_properties', 'processor_properties'], axis=1),
            clock_props,
            proc_props
        ], axis=1)

        # Convert boolean to numeric for correlation analysis
        self.molecules_df['proc_parallel_processing_num'] = self.molecules_df['proc_parallel_processing'].astype(int)

    def molecular_properties_overview(self):
        """Create overview of molecular properties"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Molecular Properties Overview', fontsize=16, fontweight='bold')

        # Molecular weight distribution
        axes[0, 0].hist(self.molecules_df['molecular_weight'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Molecular Weight Distribution')
        axes[0, 0].set_xlabel('Molecular Weight')
        axes[0, 0].set_ylabel('Frequency')

        # LogP distribution
        axes[0, 1].hist(self.molecules_df['logp'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('LogP Distribution')
        axes[0, 1].set_xlabel('LogP')
        axes[0, 1].set_ylabel('Frequency')

        # TPSA distribution
        axes[0, 2].hist(self.molecules_df['tpsa'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('TPSA Distribution')
        axes[0, 2].set_xlabel('TPSA')
        axes[0, 2].set_ylabel('Frequency')

        # Molecular weight vs LogP
        scatter = axes[1, 0].scatter(self.molecules_df['molecular_weight'], self.molecules_df['logp'],
                                     c=self.molecules_df['tpsa'], cmap='viridis', alpha=0.7)
        axes[1, 0].set_title('Molecular Weight vs LogP (colored by TPSA)')
        axes[1, 0].set_xlabel('Molecular Weight')
        axes[1, 0].set_ylabel('LogP')
        plt.colorbar(scatter, ax=axes[1, 0], label='TPSA')

        # Clock frequency distribution
        axes[1, 1].hist(self.molecules_df['clock_base_frequency_hz'] / 1e12, bins=20,
                        alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_title('Clock Base Frequency Distribution')
        axes[1, 1].set_xlabel('Frequency (THz)')
        axes[1, 1].set_ylabel('Frequency')

        # Processing rate distribution
        axes[1, 2].hist(self.molecules_df['proc_processing_rate_ops_per_sec'] / 1e6, bins=20,
                        alpha=0.7, color='mediumpurple', edgecolor='black')
        axes[1, 2].set_title('Processing Rate Distribution')
        axes[1, 2].set_xlabel('Processing Rate (Mops/sec)')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        return fig

    def correlation_analysis(self):
        """Analyze correlations between different properties"""
        # Select numeric columns for correlation
        numeric_cols = ['molecular_weight', 'logp', 'tpsa', 'clock_base_frequency_hz',
                        'clock_temporal_precision_seconds', 'clock_frequency_stability',
                        'proc_processing_rate_ops_per_sec', 'proc_memory_capacity_bits',
                        'proc_parallel_processing_num']

        corr_data = self.molecules_df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Correlation Matrix of Molecular and Computational Properties',
                     fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return fig

    def clock_properties_analysis(self):
        """Analyze clock-related properties"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clock Properties Analysis', fontsize=16, fontweight='bold')

        # Frequency vs Stability
        scatter = axes[0, 0].scatter(self.molecules_df['clock_base_frequency_hz'] / 1e12,
                                     self.molecules_df['clock_frequency_stability'],
                                     c=self.molecules_df['molecular_weight'], cmap='plasma', alpha=0.7)
        axes[0, 0].set_xlabel('Base Frequency (THz)')
        axes[0, 0].set_ylabel('Frequency Stability')
        axes[0, 0].set_title('Frequency vs Stability (colored by MW)')
        plt.colorbar(scatter, ax=axes[0, 0], label='Molecular Weight')

        # Precision vs LogP
        axes[0, 1].scatter(self.molecules_df['logp'],
                           self.molecules_df['clock_temporal_precision_seconds'] * 1e26,
                           alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('LogP')
        axes[0, 1].set_ylabel('Temporal Precision (×10⁻²⁶ s)')
        axes[0, 1].set_title('Temporal Precision vs LogP')

        # Frequency distribution by parallel processing
        parallel_yes = self.molecules_df[self.molecules_df['proc_parallel_processing']][
                           'clock_base_frequency_hz'] / 1e12
        parallel_no = self.molecules_df[~self.molecules_df['proc_parallel_processing']][
                          'clock_base_frequency_hz'] / 1e12

        axes[1, 0].hist([parallel_yes, parallel_no], bins=15, alpha=0.7,
                        label=['Parallel Processing', 'Sequential Processing'],
                        color=['green', 'red'])
        axes[1, 0].set_xlabel('Base Frequency (THz)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Frequency Distribution by Processing Type')
        axes[1, 0].legend()

        # 3D relationship
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        scatter_3d = ax_3d.scatter(self.molecules_df['molecular_weight'],
                                   self.molecules_df['clock_base_frequency_hz'] / 1e12,
                                   self.molecules_df['proc_processing_rate_ops_per_sec'] / 1e6,
                                   c=self.molecules_df['tpsa'], cmap='viridis', alpha=0.6)
        ax_3d.set_xlabel('Molecular Weight')
        ax_3d.set_ylabel('Clock Frequency (THz)')
        ax_3d.set_zlabel('Processing Rate (Mops/sec)')
        ax_3d.set_title('3D Relationship')

        plt.tight_layout()
        return fig

    def time_series_analysis(self):
        """Analyze time series data for selected molecules"""
        # Get time series data for first few molecules
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Molecular Clock Time Series Analysis', fontsize=16, fontweight='bold')

        # Plot oscillation patterns for first 4 molecules
        for i, mol_data in enumerate(self.data['clock_time_series'][:4]):
            row, col = i // 2, i % 2

            time_points = np.array(mol_data['time_points_seconds']) * 1e12  # Convert to picoseconds
            oscillations = mol_data['oscillation_measurements']

            # Plot first 1000 points for clarity
            axes[row, col].plot(time_points[:1000], oscillations[:1000],
                                linewidth=0.8, alpha=0.8)
            axes[row, col].set_title(f'Molecule {mol_data["molecule_id"]} Oscillations')
            axes[row, col].set_xlabel('Time (ps)')
            axes[row, col].set_ylabel('Oscillation Amplitude')
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def interactive_dashboard(self):
        """Create an interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Molecular Properties', 'Clock vs Processing Rate',
                            'Frequency Stability', 'Memory vs Processing'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Molecular properties scatter
        fig.add_trace(
            go.Scatter(
                x=self.molecules_df['molecular_weight'],
                y=self.molecules_df['logp'],
                mode='markers',
                marker=dict(
                    size=self.molecules_df['tpsa'] / 5,
                    color=self.molecules_df['clock_base_frequency_hz'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Clock Frequency")
                ),
                text=self.molecules_df['molecular_id'],
                hovertemplate='<b>%{text}</b><br>MW: %{x}<br>LogP: %{y}<extra></extra>',
                name='Molecules'
            ),
            row=1, col=1
        )

        # Clock frequency vs processing rate
        fig.add_trace(
            go.Scatter(
                x=self.molecules_df['clock_base_frequency_hz'] / 1e12,
                y=self.molecules_df['proc_processing_rate_ops_per_sec'] / 1e6,
                mode='markers',
                marker=dict(
                    color=self.molecules_df['molecular_weight'],
                    colorscale='Plasma',
                    size=8
                ),
                text=self.molecules_df['molecular_id'],
                name='Clock vs Processing'
            ),
            row=1, col=2
        )

        # Frequency stability
        fig.add_trace(
            go.Histogram(
                x=self.molecules_df['clock_frequency_stability'],
                nbinsx=20,
                name='Frequency Stability'
            ),
            row=2, col=1
        )

        # Memory vs Processing (colored by parallel processing)
        colors = ['red' if not pp else 'blue' for pp in self.molecules_df['proc_parallel_processing']]
        fig.add_trace(
            go.Scatter(
                x=self.molecules_df['proc_memory_capacity_bits'] / 1000,
                y=self.molecules_df['proc_processing_rate_ops_per_sec'] / 1e6,
                mode='markers',
                marker=dict(color=colors, size=8),
                text=self.molecules_df['molecular_id'],
                name='Memory vs Processing'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Interactive Molecular Data Dashboard",
            showlegend=False,
            height=800
        )

        # Update axis labels
        fig.update_xaxes(title_text="Molecular Weight", row=1, col=1)
        fig.update_yaxes(title_text="LogP", row=1, col=1)
        fig.update_xaxes(title_text="Clock Frequency (THz)", row=1, col=2)
        fig.update_yaxes(title_text="Processing Rate (Mops/sec)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency Stability", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Memory Capacity (KB)", row=2, col=2)
        fig.update_yaxes(title_text="Processing Rate (Mops/sec)", row=2, col=2)

        return fig

    def generate_summary_stats(self):
        """Generate summary statistics"""
        stats = {
            'total_molecules': len(self.molecules_df),
            'molecular_weight': {
                'mean': self.molecules_df['molecular_weight'].mean(),
                'std': self.molecules_df['molecular_weight'].std(),
                'min': self.molecules_df['molecular_weight'].min(),
                'max': self.molecules_df['molecular_weight'].max()
            },
            'clock_frequency': {
                'mean': self.molecules_df['clock_base_frequency_hz'].mean(),
                'std': self.molecules_df['clock_base_frequency_hz'].std(),
                'min': self.molecules_df['clock_base_frequency_hz'].min(),
                'max': self.molecules_df['clock_base_frequency_hz'].max()
            },
            'parallel_processing_ratio': self.molecules_df['proc_parallel_processing'].mean()
        }
        return stats


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MolecularDataAnalyzer('real_borgia_results/molecular_data_1758140958.json')

    # Generate all visualizations
    fig1 = analyzer.molecular_properties_overview()
    fig1.savefig('molecular_properties_overview.png', dpi=300, bbox_inches='tight')

    fig2 = analyzer.correlation_analysis()
    fig2.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')

    fig3 = analyzer.clock_properties_analysis()
    fig3.savefig('clock_properties_analysis.png', dpi=300, bbox_inches='tight')

    fig4 = analyzer.time_series_analysis()
    fig4.savefig('time_series_analysis.png', dpi=300, bbox_inches='tight')

    # Generate interactive dashboard
    interactive_fig = analyzer.interactive_dashboard()
    interactive_fig.write_html('interactive_dashboard.html')

    # Print summary statistics
    stats = analyzer.generate_summary_stats()
    print("Summary Statistics:")
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Molecular weight range: {stats['molecular_weight']['min']:.2f} - {stats['molecular_weight']['max']:.2f}")
    print(f"Average clock frequency: {stats['clock_frequency']['mean'] / 1e12:.2f} THz")
    print(f"Parallel processing ratio: {stats['parallel_processing_ratio']:.2%}")

    plt.show()
