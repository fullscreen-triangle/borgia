"""
Simple Visualizer for Borgia Validation Results
===============================================

Minimal dependency visualizer using only matplotlib.
Creates simple, clear visualizations of validation results.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class SimpleVisualizer:
    """
    Simple visualization tool for Borgia validation results.
    """
    
    def __init__(self, output_dir: str = "validation_plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
    def plot_validation_summary(self, results: Dict[str, Any], 
                              save_path: Optional[str] = None) -> str:
        """
        Plot validation results summary.
        
        Args:
            results: Validation results dictionary
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Pass/Fail Summary
        test_results = results.get("results", {})
        test_names = []
        test_scores = []
        test_status = []
        
        for name, result in test_results.items():
            test_names.append(name.replace('_', ' ').title())
            test_scores.append(result.get('score', 0))
            test_status.append('Pass' if result.get('success', False) else 'Fail')
        
        # Bar chart of scores
        colors = ['green' if status == 'Pass' else 'red' for status in test_status]
        bars = ax1.bar(range(len(test_names)), test_scores, color=colors, alpha=0.7)
        
        ax1.set_xlabel('Test Components')
        ax1.set_ylabel('Score (0.0 - 1.0)')
        ax1.set_title('Validation Test Scores')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, test_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 2: Overall Summary Pie Chart
        passed = results.get("passed_tests", 0)
        total = results.get("total_tests", 1)
        failed = total - passed
        
        sizes = [passed, failed]
        labels = [f'Passed ({passed})', f'Failed ({failed})']
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        
        ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
                autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Overall Results\n({results.get("pass_rate", 0):.1%} Pass Rate)')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "validation_summary.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_performance_metrics(self, results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """
        Plot performance metrics from validation results.
        
        Args:
            results: Validation results dictionary
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        test_results = results.get("results", {})
        
        # Extract performance metrics
        test_names = []
        execution_times = []
        amplification_factors = []
        efficiency_scores = []
        precision_values = []
        
        for name, result in test_results.items():
            metrics = result.get('metrics', {})
            test_names.append(name.replace('_', ' ').title())
            
            # Execution time
            exec_time = metrics.get('execution_time', metrics.get('generation_time', 
                       metrics.get('coordination_time', metrics.get('catalysis_time', 
                       metrics.get('integration_time', metrics.get('validation_time', 0))))))
            execution_times.append(exec_time)
            
            # Amplification factor
            amp_factor = metrics.get('amplification_factor', 1.0)
            amplification_factors.append(amp_factor)
            
            # Efficiency (various names)
            efficiency = metrics.get('efficiency', metrics.get('dual_functional_rate',
                        metrics.get('quantum_efficiency', metrics.get('catalysis_efficiency', 0.5))))
            efficiency_scores.append(efficiency * 100 if efficiency <= 1.0 else efficiency)
            
            # Precision 
            precision = metrics.get('average_clock_precision', metrics.get('clock_precision', 1e-30))
            if isinstance(precision, (int, float)) and precision > 0:
                precision_values.append(abs(np.log10(precision)))
            else:
                precision_values.append(30)  # Default for 1e-30
        
        # Plot 1: Execution Times
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(test_names)), execution_times, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Test Components')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Test Execution Times')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Amplification Factors
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(test_names)), amplification_factors, color='orange', alpha=0.7)
        ax2.set_xlabel('Test Components')
        ax2.set_ylabel('Amplification Factor')
        ax2.set_title('Thermodynamic Amplification Factors')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='Target: 1000×')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency Scores
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(test_names)), efficiency_scores, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Test Components')
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('System Efficiency Scores')
        ax3.set_xticks(range(len(test_names)))
        ax3.set_xticklabels(test_names, rotation=45, ha='right')
        ax3.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Target: 95%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Clock Precision (log scale)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(range(len(test_names)), precision_values, color='purple', alpha=0.7)
        ax4.set_xlabel('Test Components')
        ax4.set_ylabel('Clock Precision (-log₁₀ seconds)')
        ax4.set_title('Molecular Clock Precision')
        ax4.set_xticks(range(len(test_names)))
        ax4.set_xticklabels(test_names, rotation=45, ha='right')
        ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Target: 10⁻³⁰s')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "performance_metrics.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_s_entropy_validation(self, results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Plot S-Entropy framework validation results.
        
        Args:
            results: S-Entropy validation results dictionary
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        test_results = results.get("results", {})
        
        # Plot 1: S-Entropy Claims Validation
        claim_names = []
        claim_scores = []
        claim_status = []
        
        for name, result in test_results.items():
            claim_names.append(name.replace('_', ' ').title())
            claim_scores.append(result.get('score', 0))
            claim_status.append('Validated' if result.get('success', False) else 'Failed')
        
        # Horizontal bar chart for better readability
        colors = ['darkgreen' if status == 'Validated' else 'darkred' for status in claim_status]
        bars = ax1.barh(range(len(claim_names)), claim_scores, color=colors, alpha=0.7)
        
        ax1.set_ylabel('S-Entropy Claims')
        ax1.set_xlabel('Validation Score (0.0 - 1.0)')
        ax1.set_title('S-Entropy Framework Validation')
        ax1.set_yticks(range(len(claim_names)))
        ax1.set_yticklabels(claim_names)
        ax1.set_xlim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add score labels on bars
        for bar, score in zip(bars, claim_scores):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center')
        
        # Plot 2: Theoretical Foundation Status
        core_validated = results.get("core_s_entropy_claims_validated", False)
        foundation_solid = results.get("theoretical_foundation_validated", False)
        
        categories = ['Core Claims', 'Theoretical Foundation', 'Overall Framework']
        statuses = [
            1.0 if core_validated else 0.0,
            1.0 if foundation_solid else 0.0,
            1.0 if (core_validated and foundation_solid) else 0.0
        ]
        
        colors = ['green' if status > 0.5 else 'red' for status in statuses]
        bars2 = ax2.bar(categories, statuses, color=colors, alpha=0.7)
        
        ax2.set_ylabel('Validation Status')
        ax2.set_title('S-Entropy Theoretical Foundation')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add status labels
        for bar, status in zip(bars2, statuses):
            height = bar.get_height()
            label = 'VALIDATED' if status > 0.5 else 'FAILED'
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    label, ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "s_entropy_validation.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comprehensive_report_plot(self, validation_results: Dict[str, Any],
                                       s_entropy_results: Dict[str, Any],
                                       save_path: Optional[str] = None) -> str:
        """
        Create comprehensive report visualization.
        
        Args:
            validation_results: Core validation results
            s_entropy_results: S-Entropy validation results
            save_path: Optional save path
            
        Returns:
            Path to saved plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Overall Summary (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        validation_pass_rate = validation_results.get("pass_rate", 0)
        s_entropy_pass_rate = s_entropy_results.get("pass_rate", 0)
        overall_pass_rate = (validation_pass_rate + s_entropy_pass_rate) / 2
        
        ax1.pie([overall_pass_rate, 1-overall_pass_rate], 
               labels=['Validated', 'Failed'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%')
        ax1.set_title(f'Overall Validation\n{overall_pass_rate:.1%} Success Rate')
        
        # Core BMD Capabilities (top middle-right)
        ax2 = fig.add_subplot(gs[0, 1:3])
        core_tests = validation_results.get("results", {})
        test_names = [name.replace('_', ' ').title() for name in core_tests.keys()]
        test_scores = [result.get('score', 0) for result in core_tests.values()]
        
        bars = ax2.bar(range(len(test_names)), test_scores, 
                      color=['green' if score > 0.5 else 'red' for score in test_scores],
                      alpha=0.7)
        ax2.set_title('Core BMD Capabilities')
        ax2.set_ylabel('Validation Score')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # S-Entropy Claims (top right)
        ax3 = fig.add_subplot(gs[0, 3])
        s_entropy_tests = s_entropy_results.get("results", {})
        s_entropy_scores = [result.get('score', 0) for result in s_entropy_tests.values()]
        avg_s_entropy_score = np.mean(s_entropy_scores) if s_entropy_scores else 0
        
        ax3.bar(['S-Entropy\nFramework'], [avg_s_entropy_score], 
               color='purple', alpha=0.7, width=0.6)
        ax3.set_title('S-Entropy Framework')
        ax3.set_ylabel('Average Score')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Performance Metrics (middle row)
        ax4 = fig.add_subplot(gs[1, :])
        
        # Collect key performance metrics
        all_results = {**core_tests, **s_entropy_tests}
        metrics_data = {
            'Amplification Factors': [],
            'Execution Times': [],
            'Efficiency Scores': [],
            'Test Names': []
        }
        
        for name, result in all_results.items():
            metrics = result.get('metrics', {})
            metrics_data['Test Names'].append(name.replace('_', ' ').title())
            
            # Amplification
            amp = metrics.get('amplification_factor', 1.0)
            metrics_data['Amplification Factors'].append(amp)
            
            # Execution time
            exec_time = metrics.get('execution_time', metrics.get('generation_time', 0))
            metrics_data['Execution Times'].append(exec_time)
            
            # Efficiency
            eff = metrics.get('efficiency', metrics.get('dual_functional_rate', 0.5))
            if eff <= 1.0:
                eff *= 100
            metrics_data['Efficiency Scores'].append(eff)
        
        # Performance heatmap-style visualization
        x_pos = np.arange(len(metrics_data['Test Names']))
        width = 0.25
        
        bars1 = ax4.bar(x_pos - width, [a/1000 for a in metrics_data['Amplification Factors']], 
                       width, label='Amplification (÷1000)', alpha=0.7, color='orange')
        bars2 = ax4.bar(x_pos, metrics_data['Execution Times'], 
                       width, label='Execution Time (s)', alpha=0.7, color='blue')
        bars3 = ax4.bar(x_pos + width, [e/100 for e in metrics_data['Efficiency Scores']], 
                       width, label='Efficiency (÷100)', alpha=0.7, color='green')
        
        ax4.set_title('Performance Metrics Overview')
        ax4.set_xlabel('Test Components')
        ax4.set_ylabel('Normalized Values')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(metrics_data['Test Names'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # System Status (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        
        status_categories = [
            'Dual-Functionality\nMolecules',
            'BMD Network\nCoordination', 
            'Information\nCatalysis',
            'Hardware\nIntegration',
            'S-Entropy\nFramework'
        ]
        
        status_values = [
            1.0 if core_tests.get('dual_functionality', {}).get('success', False) else 0.0,
            1.0 if core_tests.get('bmd_networks', {}).get('success', False) else 0.0,
            1.0 if core_tests.get('information_catalysis', {}).get('success', False) else 0.0,
            1.0 if core_tests.get('hardware_integration', {}).get('success', False) else 0.0,
            avg_s_entropy_score
        ]
        
        colors = ['darkgreen' if val > 0.75 else 'orange' if val > 0.5 else 'darkred' 
                 for val in status_values]
        
        bars = ax5.bar(status_categories, status_values, color=colors, alpha=0.8)
        ax5.set_title('System Component Status')
        ax5.set_ylabel('Operational Status')
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3)
        
        # Add status labels
        for bar, val in zip(bars, status_values):
            height = bar.get_height()
            if val > 0.75:
                label = 'OPERATIONAL'
            elif val > 0.5:
                label = 'PARTIAL'
            else:
                label = 'FAILED'
            ax5.text(bar.get_x() + bar.get_width()/2., height/2,
                    label, ha='center', va='center', fontweight='bold', 
                    color='white' if val > 0.5 else 'black')
        
        # Add main title
        fig.suptitle('Borgia BMD Framework - Comprehensive Validation Report', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "comprehensive_report.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_text_summary(self, validation_results: Dict[str, Any],
                            s_entropy_results: Dict[str, Any],
                            save_path: Optional[str] = None) -> str:
        """
        Generate text summary report.
        
        Args:
            validation_results: Core validation results
            s_entropy_results: S-Entropy validation results
            save_path: Optional save path
            
        Returns:
            Path to saved text file
        """
        summary = f"""
BORGIA BMD FRAMEWORK - VALIDATION SUMMARY REPORT
={'='*50}

OVERALL RESULTS:
• Core BMD Validation: {validation_results.get('passed_tests', 0)}/{validation_results.get('total_tests', 0)} tests passed ({validation_results.get('pass_rate', 0):.1%})
• S-Entropy Framework: {s_entropy_results.get('passed_tests', 0)}/{s_entropy_results.get('total_tests', 0)} claims validated ({s_entropy_results.get('pass_rate', 0):.1%})
• System Status: {'🚀 READY' if validation_results.get('system_ready', False) else '⚠️  NEEDS ATTENTION'}
• Theoretical Foundation: {'✅ SOLID' if s_entropy_results.get('theoretical_foundation_validated', False) else '❌ QUESTIONABLE'}

CORE BMD CAPABILITIES:
"""
        
        core_tests = validation_results.get("results", {})
        for name, result in core_tests.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            score = result.get('score', 0)
            summary += f"• {name.replace('_', ' ').title()}: {status} (Score: {score:.3f})\n"
        
        summary += f"""
S-ENTROPY FRAMEWORK CLAIMS:
"""
        
        s_entropy_tests = s_entropy_results.get("results", {})
        for name, result in s_entropy_tests.items():
            status = "✅ VALIDATED" if result.get('success', False) else "❌ FAILED"
            score = result.get('score', 0)
            summary += f"• {name.replace('_', ' ').title()}: {status} (Score: {score:.3f})\n"
        
        summary += f"""
KEY PERFORMANCE METRICS:
• Execution Time: {validation_results.get('execution_time', 0) + s_entropy_results.get('execution_time', 0):.1f} seconds
• Overall Score: {(validation_results.get('overall_score', 0) + s_entropy_results.get('overall_score', 0)) / 2:.3f}
• Critical Claims: {'✅ VALIDATED' if validation_results.get('critical_claims_validated', False) else '❌ FAILED'}

CONCLUSION:
The Borgia BMD Framework demonstrates {'excellent' if validation_results.get('pass_rate', 0) > 0.9 else 'good' if validation_results.get('pass_rate', 0) > 0.7 else 'limited'} validation of core capabilities.
The S-Entropy theoretical framework shows {'strong' if s_entropy_results.get('pass_rate', 0) > 0.8 else 'partial' if s_entropy_results.get('pass_rate', 0) > 0.5 else 'weak'} mathematical foundation validation.

{'🎉 System ready for production deployment and downstream integration.' if validation_results.get('system_ready', False) and s_entropy_results.get('theoretical_foundation_validated', False) else '⚠️ Issues require resolution before production deployment.'}

Report generated: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save text summary
        if save_path is None:
            save_path = self.output_dir / "validation_summary.txt"
        
        with open(save_path, 'w') as f:
            f.write(summary)
        
        return str(save_path)


def create_validation_plots(validation_results: Dict[str, Any], 
                          s_entropy_results: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Create all validation plots.
    
    Args:
        validation_results: Core validation results
        s_entropy_results: Optional S-Entropy results
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    visualizer = SimpleVisualizer()
    plots = {}
    
    try:
        # Core validation plots
        plots['summary'] = visualizer.plot_validation_summary(validation_results)
        plots['performance'] = visualizer.plot_performance_metrics(validation_results)
        
        # S-Entropy plots if available
        if s_entropy_results:
            plots['s_entropy'] = visualizer.plot_s_entropy_validation(s_entropy_results)
            plots['comprehensive'] = visualizer.create_comprehensive_report_plot(
                validation_results, s_entropy_results
            )
            plots['text_summary'] = visualizer.generate_text_summary(
                validation_results, s_entropy_results
            )
        
        print(f"\n📊 Generated {len(plots)} visualization files:")
        for plot_name, path in plots.items():
            print(f"   • {plot_name}: {path}")
            
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")
        print("   Continuing without plots...")
    
    return plots
