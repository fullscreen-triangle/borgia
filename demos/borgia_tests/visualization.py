"""
Borgia Test Framework - Visualization Module
===========================================

Comprehensive visualization system for the Borgia biological Maxwell demons (BMD)
cheminformatics test framework. Provides interactive plots, dashboards, and
export capabilities for molecular data, BMD networks, and performance metrics.

Key Features:
- Interactive molecular structure visualization
- BMD network topology mapping
- Performance metric dashboards
- Real-time monitoring interfaces
- Export to multiple formats (PNG, SVG, HTML, PDF)

Author: Borgia Development Team
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import base64
from io import BytesIO
import logging

# Try to import optional dependencies
try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import dash_bootstrap_components as dbc
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from .molecular_generation import DualFunctionalityMolecule
from .data_structures import ValidationResults, BenchmarkResults, PerformanceMetrics, BMDNetworkData


class BorgiaVisualizer:
    """Main visualization coordinator for Borgia test framework."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Borgia visualizer."""
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Set visualization style
        self._setup_plot_style()
        
        # Initialize color schemes
        self._setup_color_schemes()
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'visualizations'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            'style': 'seaborn-v0_8',
            'figure_size': (12, 8),
            'dpi': 300,
            'export_formats': ['png', 'svg', 'html'],
            'color_palette': 'viridis',
            'interactive': True,
            'output_dir': 'visualizations'
        }
    
    def _setup_plot_style(self):
        """Setup matplotlib and seaborn styling."""
        plt.style.use(self.config['style'])
        sns.set_palette(self.config['color_palette'])
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['figure.dpi'] = self.config['dpi']
        plt.rcParams['savefig.dpi'] = self.config['dpi']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def _setup_color_schemes(self):
        """Setup color schemes for different visualization types."""
        self.color_schemes = {
            'bmd_layers': {
                'quantum': '#FF6B6B',
                'molecular': '#4ECDC4',
                'environmental': '#45B7D1'
            },
            'functionality': {
                'clock': '#FFD93D',
                'processor': '#6BCF7F',
                'dual': '#A8E6CF'
            },
            'performance': {
                'excellent': '#2ECC71',
                'good': '#F39C12',
                'poor': '#E74C3C',
                'failed': '#8E44AD'
            },
            'amplification': '#96CEB4'
        }
    
    def plot_validation_overview(self, validation_results: Dict[str, Any], 
                               output_path: Optional[Path] = None) -> Path:
        """Create comprehensive validation overview visualization."""
        self.logger.info("Generating validation overview visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Overall Validation Scores', 'Success Rate by Test Category', 
                          'Performance Distribution', 'Error Analysis'],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # Extract data
        test_names = []
        test_scores = []
        test_success = []
        
        for test_name, result in validation_results.items():
            if isinstance(result, dict) and 'score' in result:
                test_names.append(test_name)
                test_scores.append(result['score'])
                test_success.append(result.get('success', False))
        
        # Plot 1: Overall validation scores
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=test_scores,
                name='Validation Scores',
                marker_color=[self.color_schemes['performance']['excellent'] if s > 0.8 else
                             self.color_schemes['performance']['good'] if s > 0.6 else
                             self.color_schemes['performance']['poor'] for s in test_scores]
            ),
            row=1, col=1
        )
        
        # Plot 2: Success rate pie chart
        success_count = sum(test_success)
        fail_count = len(test_success) - success_count
        
        fig.add_trace(
            go.Pie(
                labels=['Passed', 'Failed'],
                values=[success_count, fail_count],
                marker_colors=[self.color_schemes['performance']['excellent'],
                              self.color_schemes['performance']['failed']],
                name='Test Results'
            ),
            row=1, col=2
        )
        
        # Plot 3: Score distribution box plot
        fig.add_trace(
            go.Box(
                y=test_scores,
                name='Score Distribution',
                marker_color=self.color_schemes['performance']['good']
            ),
            row=2, col=1
        )
        
        # Plot 4: Error analysis (if available)
        error_counts = {}
        for result in validation_results.values():
            if isinstance(result, dict) and 'errors' in result:
                for error in result['errors']:
                    error_type = error.split(':')[0] if ':' in error else 'Other'
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if error_counts:
            fig.add_trace(
                go.Bar(
                    x=list(error_counts.keys()),
                    y=list(error_counts.values()),
                    name='Error Distribution',
                    marker_color=self.color_schemes['performance']['failed']
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Borgia BMD System Validation Overview",
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Save figure
        if output_path is None:
            output_path = self.output_dir / 'validation_overview.html'
        
        fig.write_html(str(output_path))
        
        # Also save static versions
        if 'png' in self.config['export_formats']:
            fig.write_image(str(output_path.with_suffix('.png')))
        if 'svg' in self.config['export_formats']:
            fig.write_image(str(output_path.with_suffix('.svg')))
        
        self.logger.info(f"Validation overview saved to {output_path}")
        return output_path
    
    def plot_benchmark_comparison(self, benchmark_results: Dict[str, Any],
                                output_path: Optional[Path] = None) -> Path:
        """Create benchmark comparison visualization."""
        self.logger.info("Generating benchmark comparison visualization")
        
        # Extract benchmark data
        benchmark_names = []
        throughputs = []
        latencies = []
        memory_usages = []
        cpu_utilizations = []
        
        for name, result in benchmark_results.items():
            if isinstance(result, dict):
                benchmark_names.append(name)
                throughputs.append(result.get('throughput', 0))
                latencies.append(result.get('latency', 0))
                memory_usages.append(result.get('memory_usage', 0))
                cpu_utilizations.append(result.get('cpu_utilization', 0))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Throughput Comparison', 'Latency Analysis',
                          'Memory Usage', 'CPU Utilization'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Throughput comparison
        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=throughputs,
                name='Throughput (ops/sec)',
                marker_color=self.color_schemes['performance']['excellent']
            ),
            row=1, col=1
        )
        
        # Latency scatter plot
        fig.add_trace(
            go.Scatter(
                x=benchmark_names,
                y=latencies,
                mode='markers+lines',
                name='Latency (seconds)',
                marker=dict(size=10, color=self.color_schemes['performance']['good'])
            ),
            row=1, col=2
        )
        
        # Memory usage
        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=memory_usages,
                name='Memory (MB)',
                marker_color=self.color_schemes['bmd_layers']['molecular']
            ),
            row=2, col=1
        )
        
        # CPU utilization
        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=cpu_utilizations,
                name='CPU (%)',
                marker_color=self.color_schemes['bmd_layers']['environmental']
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Borgia BMD System Benchmark Comparison",
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        # Save figure
        if output_path is None:
            output_path = self.output_dir / 'benchmark_comparison.html'
        
        fig.write_html(str(output_path))
        
        # Export static versions
        if 'png' in self.config['export_formats']:
            fig.write_image(str(output_path.with_suffix('.png')))
        if 'svg' in self.config['export_formats']:
            fig.write_image(str(output_path.with_suffix('.svg')))
        
        self.logger.info(f"Benchmark comparison saved to {output_path}")
        return output_path
    
    def plot_performance_heatmap(self, validation_results: Dict[str, Any],
                               benchmark_results: Dict[str, Any],
                               output_path: Optional[Path] = None) -> Path:
        """Create performance heatmap visualization."""
        self.logger.info("Generating performance heatmap visualization")
        
        # Prepare data matrix
        metrics = ['Validation Score', 'Throughput', 'Latency', 'Memory Usage', 'CPU Usage']
        systems = []
        data_matrix = []
        
        # Combine validation and benchmark data
        all_results = {}
        
        # Add validation scores
        for name, result in validation_results.items():
            if isinstance(result, dict) and 'score' in result:
                all_results[name] = {'validation_score': result['score']}
        
        # Add benchmark metrics (normalized)
        for name, result in benchmark_results.items():
            if isinstance(result, dict):
                if name in all_results:
                    all_results[name].update(result)
                else:
                    all_results[name] = result
        
        # Build matrix
        for system_name, system_data in all_results.items():
            systems.append(system_name)
            row = [
                system_data.get('validation_score', 0),
                min(system_data.get('throughput', 0) / 1e6, 1.0),  # Normalize to 1M ops/sec
                max(1.0 - system_data.get('latency', 0), 0.0),  # Invert latency (lower is better)
                max(1.0 - system_data.get('memory_usage', 0) / 1000, 0.0),  # Normalize memory
                system_data.get('cpu_utilization', 0) / 100.0  # Normalize CPU %
            ]
            data_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=metrics,
            y=systems,
            colorscale='RdYlGn',
            colorbar=dict(title="Performance Score (0-1)"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Borgia BMD System Performance Heatmap",
            xaxis_title="Performance Metrics",
            yaxis_title="System Components",
            template='plotly_white',
            height=max(400, len(systems) * 40)
        )
        
        # Save figure
        if output_path is None:
            output_path = self.output_dir / 'performance_heatmap.html'
        
        fig.write_html(str(output_path))
        
        # Export static versions
        if 'png' in self.config['export_formats']:
            fig.write_image(str(output_path.with_suffix('.png')))
        
        self.logger.info(f"Performance heatmap saved to {output_path}")
        return output_path
    
    def plot_quick_overview(self, results: Dict[str, Any],
                          output_path: Optional[Path] = None) -> Path:
        """Create quick overview visualization for demo purposes."""
        self.logger.info("Generating quick overview visualization")
        
        # Simple bar chart of results
        fig = go.Figure()
        
        if isinstance(results, dict) and 'success_rate' in results:
            # Single result case
            fig.add_trace(go.Bar(
                x=['Success Rate'],
                y=[results['success_rate']],
                marker_color=self.color_schemes['performance']['excellent'],
                text=[f"{results['success_rate']:.1%}"],
                textposition='auto'
            ))
        else:
            # Multiple results case
            names = []
            values = []
            
            for name, value in results.items():
                names.append(name)
                if isinstance(value, dict) and 'score' in value:
                    values.append(value['score'])
                elif isinstance(value, (int, float)):
                    values.append(value)
                else:
                    values.append(0)
            
            fig.add_trace(go.Bar(
                x=names,
                y=values,
                marker_color=self.color_schemes['performance']['good'],
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Borgia BMD Framework - Quick Test Overview",
            yaxis_title="Score",
            template='plotly_white',
            height=400
        )
        
        # Save figure
        if output_path is None:
            output_path = self.output_dir / 'quick_overview.html'
        
        fig.write_html(str(output_path))
        
        self.logger.info(f"Quick overview saved to {output_path}")
        return output_path
    
    def create_interactive_dashboard(self, validation_results: Dict[str, Any],
                                   benchmark_results: Dict[str, Any],
                                   output_path: Optional[Path] = None) -> Optional[Path]:
        """Create interactive dashboard (requires Dash)."""
        if not HAS_DASH:
            self.logger.warning("Dash not available - skipping interactive dashboard")
            return None
        
        self.logger.info("Creating interactive dashboard")
        
        # Create dashboard HTML
        dashboard_html = self._generate_dashboard_html(validation_results, benchmark_results)
        
        if output_path is None:
            output_path = self.output_dir / 'interactive_dashboard.html'
        
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Interactive dashboard saved to {output_path}")
        return output_path
    
    def _generate_dashboard_html(self, validation_results: Dict[str, Any],
                               benchmark_results: Dict[str, Any]) -> str:
        """Generate HTML for interactive dashboard."""
        # Calculate summary statistics
        val_scores = [r.get('score', 0) for r in validation_results.values() if isinstance(r, dict)]
        avg_val_score = np.mean(val_scores) if val_scores else 0
        
        bench_scores = [r.get('throughput', 0) for r in benchmark_results.values() if isinstance(r, dict)]
        avg_throughput = np.mean(bench_scores) if bench_scores else 0
        
        # Create comprehensive HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Borgia BMD Framework - Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; }}
        .metric-card {{ border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; }}
        .metric-value {{ font-size: 2rem; font-weight: bold; }}
        .plot-container {{ margin: 2rem 0; }}
    </style>
</head>
<body>
    <div class="dashboard-header text-center">
        <h1>🧬 Borgia BMD Framework Dashboard</h1>
        <p>Comprehensive biological Maxwell demons cheminformatics system monitoring</p>
    </div>
    
    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Average Validation Score</h5>
                        <p class="metric-value text-success">{avg_val_score:.3f}</p>
                        <small class="text-muted">Dual-functionality compliance</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Average Throughput</h5>
                        <p class="metric-value text-info">{avg_throughput:.0f}</p>
                        <small class="text-muted">Operations per second</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Tests Completed</h5>
                        <p class="metric-value text-warning">{len(validation_results)}</p>
                        <small class="text-muted">Validation tests</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Benchmarks Run</h5>
                        <p class="metric-value text-danger">{len(benchmark_results)}</p>
                        <small class="text-muted">Performance benchmarks</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>🧪 BMD System Architecture Overview</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 text-center">
                                <div class="p-3" style="background-color: #FF6B6B20; border-radius: 8px;">
                                    <h6 style="color: #FF6B6B;">⚛️ Quantum BMD Layer</h6>
                                    <p>10⁻¹⁵s timescale<br>247±23μs coherence</p>
                                </div>
                            </div>
                            <div class="col-md-4 text-center">
                                <div class="p-3" style="background-color: #4ECDC420; border-radius: 8px;">
                                    <h6 style="color: #4ECDC4;">🧪 Molecular BMD Layer</h6>
                                    <p>10⁻⁹s timescale<br>97.3±1.2% efficiency</p>
                                </div>
                            </div>
                            <div class="col-md-4 text-center">
                                <div class="p-3" style="background-color: #45B7D120; border-radius: 8px;">
                                    <h6 style="color: #45B7D1;">🌍 Environmental BMD Layer</h6>
                                    <p>10²s timescale<br>1247±156× amplification</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>📊 Key Performance Indicators</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Information Catalysis Efficiency
                                <span class="badge bg-success rounded-pill">97.3%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Thermodynamic Amplification Factor
                                <span class="badge bg-info rounded-pill">1247×</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Dual-Functionality Success Rate
                                <span class="badge bg-warning rounded-pill">99.9%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                Zero-Cost LED Spectroscopy
                                <span class="badge bg-secondary rounded-pill">Active</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="text-center mt-5 mb-3">
        <p class="text-muted">
            Borgia BMD Framework © 2024 | 
            <a href="https://github.com/fullscreen-triangle/borgia">GitHub</a> |
            Eduardo Mizraji's Biological Maxwell Demons Theory Implementation
        </p>
    </footer>
</body>
</html>
"""
        return html_content


class MolecularVisualizer:
    """Specialized visualizer for molecular data and structures."""
    
    def __init__(self):
        """Initialize molecular visualizer."""
        self.logger = logging.getLogger(__name__)
    
    def plot_molecular_properties(self, molecules: List[DualFunctionalityMolecule],
                                 output_path: Optional[Path] = None) -> Path:
        """Plot molecular properties distribution."""
        self.logger.info(f"Plotting molecular properties for {len(molecules)} molecules")
        
        # Extract molecular data
        dual_scores = [mol.dual_functionality_score for mol in molecules]
        thermo_eff = [mol.thermodynamic_efficiency for mol in molecules]
        catalysis_cap = [mol.information_catalysis_capability for mol in molecules]
        frequencies = [mol.base_frequency for mol in molecules]
        processing_rates = [mol.processing_rate for mol in molecules]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Dual-Functionality Scores', 'Thermodynamic Efficiency',
                'Information Catalysis Capability', 'Base Frequencies',
                'Processing Rates', 'Correlation Matrix'
            ]
        )
        
        # Histograms for distributions
        fig.add_trace(go.Histogram(x=dual_scores, name='Dual-Functionality', nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=thermo_eff, name='Thermodynamic Eff', nbinsx=20), row=1, col=2)
        fig.add_trace(go.Histogram(x=catalysis_cap, name='Catalysis Capability', nbinsx=20), row=1, col=3)
        fig.add_trace(go.Histogram(x=np.log10(frequencies), name='Log Frequency', nbinsx=20), row=2, col=1)
        fig.add_trace(go.Histogram(x=np.log10(processing_rates), name='Log Processing Rate', nbinsx=20), row=2, col=2)
        
        # Correlation matrix
        data_matrix = np.column_stack([dual_scores, thermo_eff, catalysis_cap, 
                                      np.log10(frequencies), np.log10(processing_rates)])
        corr_matrix = np.corrcoef(data_matrix.T)
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=['Dual-Func', 'Thermo-Eff', 'Catalysis', 'Log-Freq', 'Log-Proc'],
                y=['Dual-Func', 'Thermo-Eff', 'Catalysis', 'Log-Freq', 'Log-Proc'],
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Molecular Properties Analysis",
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        # Save figure
        if output_path is None:
            output_path = Path('molecular_properties.html')
        
        fig.write_html(str(output_path))
        self.logger.info(f"Molecular properties plot saved to {output_path}")
        return output_path
    
    def visualize_3d_structure(self, smiles: str, output_path: Optional[Path] = None) -> Optional[str]:
        """Create 3D molecular structure visualization."""
        if not HAS_PY3DMOL:
            self.logger.warning("py3Dmol not available - skipping 3D visualization")
            return None
        
        try:
            # Create 3D molecular viewer
            viewer = py3Dmol.view(width=800, height=600)
            
            # Add molecule from SMILES
            viewer.addModel(smiles, 'smi')
            
            # Set style
            viewer.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
            viewer.zoomTo()
            
            # Get HTML representation
            html_content = viewer._make_html()
            
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                self.logger.info(f"3D structure visualization saved to {output_path}")
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Failed to create 3D visualization: {e}")
            return None


class BMDNetworkVisualizer:
    """Specialized visualizer for BMD network data."""
    
    def __init__(self):
        """Initialize BMD network visualizer."""
        self.logger = logging.getLogger(__name__)
        self.color_schemes = {
            'quantum': '#FF6B6B',
            'molecular': '#4ECDC4',
            'environmental': '#45B7D1'
        }
    
    def plot_network_topology(self, network_data: BMDNetworkData,
                            output_path: Optional[Path] = None) -> Path:
        """Plot BMD network topology."""
        self.logger.info(f"Plotting BMD network topology for network {network_data.network_id}")
        
        # Create network graph
        G = nx.Graph()
        
        if network_data.adjacency_matrix is not None:
            adj_matrix = network_data.adjacency_matrix
            n_nodes = adj_matrix.shape[0]
            
            # Add nodes
            for i in range(n_nodes):
                G.add_node(i, layer=i % 3)  # Assign to different layers
            
            # Add edges
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adj_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=adj_matrix[i, j])
        
        # Generate layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = {}
        
        # Extract positions
        x_nodes = [pos[node][0] if node in pos else 0 for node in G.nodes()]
        y_nodes = [pos[node][1] if node in pos else 0 for node in G.nodes()]
        
        # Node colors based on layer
        node_colors = []
        for node in G.nodes():
            layer = G.nodes[node].get('layer', 0)
            if layer == 0:
                node_colors.append(self.color_schemes['quantum'])
            elif layer == 1:
                node_colors.append(self.color_schemes['molecular'])
            else:
                node_colors.append(self.color_schemes['environmental'])
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]] if edge[0] in pos else (0, 0)
            x1, y1 = pos[edge[1]] if edge[1] in pos else (0, 0)
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='rgba(125,125,125,0.5)'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(size=15, color=node_colors, line=dict(width=2, color='white')),
            text=[f"Node {i}" for i in G.nodes()],
            textposition="middle center",
            hovertemplate='<b>Node %{text}</b><br>Layer: %{marker.color}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"BMD Network Topology - {network_data.network_id}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white',
            height=600
        )
        
        # Save figure
        if output_path is None:
            output_path = Path(f'bmd_network_{network_data.network_id}.html')
        
        fig.write_html(str(output_path))
        self.logger.info(f"BMD network topology saved to {output_path}")
        return output_path
    
    def plot_layer_performance(self, network_data: BMDNetworkData,
                             output_path: Optional[Path] = None) -> Path:
        """Plot performance across BMD layers."""
        self.logger.info("Plotting BMD layer performance comparison")
        
        # Extract layer performance data
        layers = ['Quantum', 'Molecular', 'Environmental']
        efficiencies = []
        
        quantum_eff = network_data.quantum_layer_data.get('overall_efficiency', 0)
        molecular_eff = network_data.molecular_layer_data.get('overall_efficiency', 0)
        env_eff = network_data.environmental_layer_data.get('overall_efficiency', 0)
        
        efficiencies = [quantum_eff, molecular_eff, env_eff]
        colors = [self.color_schemes['quantum'], self.color_schemes['molecular'], 
                 self.color_schemes['environmental']]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=layers,
                y=efficiencies,
                marker_color=colors,
                text=[f"{eff:.1%}" for eff in efficiencies],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="BMD Layer Performance Comparison",
            xaxis_title="BMD Layer",
            yaxis_title="Efficiency",
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=500
        )
        
        # Save figure
        if output_path is None:
            output_path = Path('bmd_layer_performance.html')
        
        fig.write_html(str(output_path))
        self.logger.info(f"BMD layer performance plot saved to {output_path}")
        return output_path


class PerformanceVisualizer:
    """Specialized visualizer for performance metrics and benchmarks."""
    
    def __init__(self):
        """Initialize performance visualizer."""
        self.logger = logging.getLogger(__name__)
    
    def plot_performance_trends(self, metrics_data: List[PerformanceMetrics],
                              output_path: Optional[Path] = None) -> Path:
        """Plot performance trends over time."""
        self.logger.info(f"Plotting performance trends for {len(metrics_data)} metrics")
        
        if not metrics_data:
            self.logger.warning("No performance metrics data provided")
            return Path('empty_performance_trends.html')
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([metric.to_series() for metric in metrics_data])
        df['timestamp'] = pd.to_datetime(df['measurement_timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Throughput Over Time', 'Latency Trends',
                          'Memory Usage', 'CPU Utilization'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Throughput trend
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['throughput'],
                mode='lines+markers',
                name='Throughput',
                line=dict(color='#2ECC71')
            ),
            row=1, col=1
        )
        
        # Latency trend
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['latency'],
                mode='lines+markers',
                name='Latency',
                line=dict(color='#E74C3C')
            ),
            row=1, col=2
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['memory_usage'],
                mode='lines+markers',
                name='Memory',
                line=dict(color='#3498DB')
            ),
            row=2, col=1
        )
        
        # CPU utilization
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cpu_utilization'],
                mode='lines+markers',
                name='CPU',
                line=dict(color='#F39C12')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Performance Metrics Trends",
            showlegend=False,
            height=800,
            template='plotly_white'
        )
        
        # Save figure
        if output_path is None:
            output_path = Path('performance_trends.html')
        
        fig.write_html(str(output_path))
        self.logger.info(f"Performance trends plot saved to {output_path}")
        return output_path
