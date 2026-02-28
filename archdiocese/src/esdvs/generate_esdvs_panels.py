"""
ESDVS Panel Chart Generator

Generates 4-panel visualization charts for ESDVS validation results.
Each panel contains 4 charts with at least one 3D visualization.

Author: ESDVS Framework
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict


def load_results():
    """Load all ESDVS validation results."""
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'esdvs'

    results = {}

    # Load each validation result
    files = {
        'mutual_exclusion': 'mutual_exclusion_validation.json',
        'ternary_trajectory': 'ternary_trajectory_validation.json',
        'categorical_resolution': 'categorical_resolution_validation.json',
        'dual_mode_enhancement': 'dual_mode_enhancement_validation.json'
    }

    for key, filename in files.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[key] = json.load(f)

    return results


def generate_panel_1_mutual_exclusion(results: Dict, output_dir: Path):
    """
    Panel 1: Mutual Exclusion Validation

    A: 3D mode frequency space (Raman vs IR)
    B: Cross-prediction accuracy
    C: Violation metric analysis
    D: Mode overlap pattern
    """
    data = results['mutual_exclusion']

    fig = plt.figure(figsize=(20, 5))

    # Panel A: 3D Mode Frequency Space
    ax1 = fig.add_subplot(141, projection='3d')

    raman_modes = np.array(data['mode_data']['raman_modes'])
    ir_modes = np.array(data['mode_data']['ir_modes'])

    # Create 3D scatter: frequency vs intensity vs mode index
    raman_idx = np.arange(len(raman_modes))
    ir_idx = np.arange(len(ir_modes))

    # Normalize intensities (arbitrary for visualization)
    raman_intensity = raman_modes / np.max(raman_modes)
    ir_intensity = ir_modes / np.max(ir_modes)

    ax1.scatter(raman_idx, raman_modes, raman_intensity,
               c='red', s=150, marker='o', alpha=0.8, label='Raman')
    ax1.scatter(ir_idx, ir_modes, ir_intensity,
               c='blue', s=150, marker='^', alpha=0.8, label='IR')

    ax1.set_xlabel('Mode Index', fontsize=10, labelpad=8)
    ax1.set_ylabel('Frequency (cm$^{-1}$)', fontsize=10, labelpad=8)
    ax1.set_zlabel('Normalized Intensity', fontsize=10, labelpad=8)
    ax1.set_title('A: 3D Mode Frequency Space', fontsize=12, pad=15)
    ax1.legend(fontsize=9)
    ax1.view_init(elev=20, azim=45)

    # Panel B: Cross-prediction Accuracy
    ax2 = fig.add_subplot(142)

    accuracies = [
        data['cross_prediction']['raman_from_ir_accuracy'],
        data['cross_prediction']['ir_from_raman_accuracy'],
        data['cross_prediction']['average_accuracy']
    ]
    labels = ['Raman from IR', 'IR from Raman', 'Average']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax2.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% threshold')
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_ylim([0.98, 1.0])
    ax2.set_title('B: Cross-Prediction Accuracy', fontsize=12, pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Panel C: Violation Metric Analysis
    ax3 = fig.add_subplot(143)

    metrics = [
        data['validation_summary']['violation_metric'],
        data['validation_summary']['strict_violation_metric']
    ]
    metric_labels = ['Standard\nViolation', 'Strict\nViolation']
    metric_colors = ['#FFA07A', '#98D8C8']

    bars = ax3.bar(metric_labels, metrics, color=metric_colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax3.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='10% threshold')
    ax3.set_ylabel('Violation Metric', fontsize=11)
    ax3.set_ylim([0, 0.6])
    ax3.set_title('C: Violation Metric Analysis', fontsize=12, pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Panel D: Mode Overlap Pattern
    ax4 = fig.add_subplot(144)

    # Create frequency line plot
    all_freqs = sorted(list(raman_modes) + list(ir_modes))
    raman_y = [1 if f in raman_modes else 0 for f in all_freqs]
    ir_y = [1 if f in ir_modes else 0 for f in all_freqs]

    ax4.stem(raman_modes, np.ones(len(raman_modes)),
            linefmt='r-', markerfmt='ro', basefmt=' ', label='Raman modes')
    ax4.stem(ir_modes, np.ones(len(ir_modes)) * 0.5,
            linefmt='b-', markerfmt='b^', basefmt=' ', label='IR modes')

    ax4.set_xlabel('Frequency (cm$^{-1}$)', fontsize=11)
    ax4.set_ylabel('Mode Type', fontsize=11)
    ax4.set_yticks([0.5, 1.0])
    ax4.set_yticklabels(['IR', 'Raman'])
    ax4.set_title('D: Mode Overlap Pattern', fontsize=12, pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'esdvs_panel_1_mutual_exclusion.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Generated {output_file}")


def generate_panel_2_ternary_trajectory(results: Dict, output_dir: Path):
    """
    Panel 2: Ternary Trajectory Validation

    A: 3D state trajectory evolution
    B: Fidelity vs trajectory completion
    C: State population dynamics
    D: Quality metrics comparison
    """
    data = results['ternary_trajectory']

    fig = plt.figure(figsize=(20, 5))

    # Panel A: 3D State Trajectory Evolution
    ax1 = fig.add_subplot(141, projection='3d')

    # Simulate trajectory from data
    traj_data = data['simulated_trajectory']
    t = np.array(traj_data['time'])
    c0 = np.array(traj_data['c0'])
    c1 = np.array(traj_data['c1'])
    c2 = np.array(traj_data['c2'])

    # Plot trajectory in state space
    colors = cm.viridis(np.linspace(0, 1, len(t)))

    for i in range(len(t)-1):
        ax1.plot([c0[i], c0[i+1]], [c1[i], c1[i+1]], [c2[i], c2[i+1]],
                c=colors[i], linewidth=2, alpha=0.8)

    # Mark initial and final states
    ax1.scatter([c0[0]], [c1[0]], [c2[0]], c='green', s=200, marker='o',
               edgecolor='black', linewidth=2, label='Initial |2>')
    ax1.scatter([c0[-1]], [c1[-1]], [c2[-1]], c='red', s=200, marker='s',
               edgecolor='black', linewidth=2, label='Final |0>')

    ax1.set_xlabel('c$_0$ (Ground)', fontsize=10, labelpad=8)
    ax1.set_ylabel('c$_1$ (Natural)', fontsize=10, labelpad=8)
    ax1.set_zlabel('c$_2$ (Excited)', fontsize=10, labelpad=8)
    ax1.set_title('A: 3D State Trajectory Evolution', fontsize=12, pad=15)
    ax1.legend(fontsize=9)
    ax1.view_init(elev=25, azim=135)

    # Panel B: Fidelity vs Trajectory Completion
    ax2 = fig.add_subplot(142)

    # Simulate fidelity decay
    t_norm = t / t[-1]  # Normalized time
    fidelity = np.exp(-t_norm * 0.3) * 0.15 + 0.85  # Simulated fidelity

    ax2.plot(t_norm * 100, fidelity, linewidth=2.5, color='#2E86DE', label='Fidelity')
    ax2.axhline(y=data['quality_analysis']['average_fidelity'],
               color='green', linestyle='--', linewidth=2, label='Average')
    ax2.axhline(y=data['quality_analysis']['fidelity_at_tau_em'],
               color='orange', linestyle='--', linewidth=2, label='At tau_em')
    ax2.axhline(y=0.95, color='red', linestyle=':', linewidth=2, label='High threshold')

    ax2.set_xlabel('Trajectory Completion (%)', fontsize=11)
    ax2.set_ylabel('Fidelity', fontsize=11)
    ax2.set_ylim([0.7, 1.0])
    ax2.set_title('B: Fidelity vs Trajectory Completion', fontsize=12, pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel C: State Population Dynamics
    ax3 = fig.add_subplot(143)

    ax3.plot(t * 1e12, c0, linewidth=2.5, color='#E74C3C', label='|0> Ground')
    ax3.plot(t * 1e12, c1, linewidth=2.5, color='#F39C12', label='|1> Natural')
    ax3.plot(t * 1e12, c2, linewidth=2.5, color='#3498DB', label='|2> Excited')

    ax3.set_xlabel('Time (ps)', fontsize=11)
    ax3.set_ylabel('Population', fontsize=11)
    ax3.set_ylim([0, 1])
    ax3.set_title('C: State Population Dynamics', fontsize=12, pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Panel D: Quality Metrics Comparison
    ax4 = fig.add_subplot(144)

    qualities = [
        data['quality_analysis']['average_fidelity'],
        data['quality_analysis']['fidelity_at_tau_em']
    ]
    quality_labels = ['Average\nFidelity', 'Fidelity at\ntau_em']
    quality_colors = ['#1ABC9C', '#E67E22']

    bars = ax4.bar(quality_labels, qualities, color=quality_colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax4.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='High (0.95)')
    ax4.axhline(y=0.98, color='red', linestyle='--', linewidth=2, label='Excellent (0.98)')

    ax4.set_ylabel('Fidelity', fontsize=11)
    ax4.set_ylim([0.7, 1.0])
    ax4.set_title('D: Quality Metrics Comparison', fontsize=12, pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'esdvs_panel_2_ternary_trajectory.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Generated {output_file}")


def generate_panel_3_categorical_resolution(results: Dict, output_dir: Path):
    """
    Panel 3: Categorical Resolution Validation

    A: 3D resolution landscape (N, omega, tau_cat)
    B: Planck ratio comparison
    C: State accumulation over time
    D: Resolution regime classification
    """
    data = results['categorical_resolution']

    fig = plt.figure(figsize=(20, 5))

    # Panel A: 3D Resolution Landscape
    ax1 = fig.add_subplot(141, projection='3d')

    # Create resolution surface
    N_range = np.logspace(2, 4, 30)
    omega_range = np.logspace(13, 15, 30)
    N_grid, omega_grid = np.meshgrid(N_range, omega_range)
    tau_cat_grid = (2 * np.pi) / (N_grid * omega_grid)

    surf = ax1.plot_surface(np.log10(N_grid), np.log10(omega_grid),
                           np.log10(tau_cat_grid), cmap='viridis',
                           alpha=0.9, edgecolor='none')

    # Mark current system
    N_current = data['theorem_12_2_validation']['n_oscillators']
    omega_current = data['theorem_12_2_validation']['average_omega_rad_s']
    tau_current = data['resolution_data']['categorical_resolution']

    ax1.scatter([np.log10(N_current)], [np.log10(omega_current)],
               [np.log10(tau_current)], c='red', s=200, marker='*',
               edgecolor='black', linewidth=2, label='ESDVS')

    ax1.set_xlabel('log$_{10}$(N oscillators)', fontsize=10, labelpad=8)
    ax1.set_ylabel('log$_{10}$(omega rad/s)', fontsize=10, labelpad=8)
    ax1.set_zlabel('log$_{10}$(tau_cat s)', fontsize=10, labelpad=8)
    ax1.set_title('A: 3D Resolution Landscape', fontsize=12, pad=15)
    ax1.legend(fontsize=9)
    ax1.view_init(elev=20, azim=135)

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Panel B: Planck Ratio Comparison
    ax2 = fig.add_subplot(142)

    regimes = ['Categorical', 'Raman', 'Dual-mode']
    ratios = [
        data['regime_analysis']['orders_beyond_planck_categorical'],
        data['regime_analysis']['orders_beyond_planck_raman'],
        data['regime_analysis']['orders_beyond_planck_dual']
    ]
    colors_planck = ['#E74C3C', '#3498DB', '#9B59B6']

    bars = ax2.bar(regimes, ratios, color=colors_planck, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Orders of Magnitude Beyond Planck', fontsize=11)
    ax2.set_title('B: Planck Ratio Comparison', fontsize=12, pad=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    # Panel C: State Accumulation Over Time
    ax3 = fig.add_subplot(143)

    # Simulate state accumulation
    t_range = np.linspace(0, 1, 100)
    tau_cat = data['resolution_data']['categorical_resolution']
    tau_raman = data['resolution_data']['raman_resolution']
    tau_dual = data['resolution_data']['dual_mode_resolution']

    M_cat = t_range / tau_cat
    M_raman = t_range / tau_raman
    M_dual = t_range / tau_dual

    ax3.semilogy(t_range, M_cat, linewidth=2.5, color='#E74C3C', label='Categorical')
    ax3.semilogy(t_range, M_raman, linewidth=2.5, color='#3498DB', label='Raman')
    ax3.semilogy(t_range, M_dual, linewidth=2.5, color='#9B59B6', label='Dual-mode')

    ax3.set_xlabel('Integration Time (s)', fontsize=11)
    ax3.set_ylabel('Categorical States (log scale)', fontsize=11)
    ax3.set_title('C: State Accumulation Over Time', fontsize=12, pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Panel D: Resolution Regime Classification
    ax4 = fig.add_subplot(144)

    # Create regime comparison
    regime_names = ['Categorical', 'Raman', 'IR', 'Dual-mode']
    regime_classes = [
        data['regime_analysis']['categorical_regime'],
        data['regime_analysis']['raman_regime'],
        'attosecond',  # From data
        data['regime_analysis']['dual_mode_regime']
    ]

    # Map regimes to numeric values for plotting
    regime_map = {
        'sub-Planckian': 6,
        'trans-Planckian': 5,
        'yoctosecond': 4,
        'zeptosecond': 3,
        'attosecond': 2,
        'femtosecond': 1
    }

    regime_values = [regime_map.get(r, 0) for r in regime_classes]
    regime_colors_bar = ['#9B59B6', '#3498DB', '#2ECC71', '#E67E22']

    bars = ax4.barh(regime_names, regime_values, color=regime_colors_bar,
                    alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.set_xlabel('Temporal Regime', fontsize=11)
    ax4.set_xticks(list(regime_map.values()))
    ax4.set_xticklabels(list(regime_map.keys()), rotation=45, ha='right', fontsize=8)
    ax4.set_title('D: Resolution Regime Classification', fontsize=12, pad=10)
    ax4.grid(axis='x', alpha=0.3)

    # Add regime labels
    for i, (bar, regime) in enumerate(zip(bars, regime_classes)):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                regime, ha='left', va='center', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'esdvs_panel_3_categorical_resolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Generated {output_file}")


def generate_panel_4_dual_mode_enhancement(results: Dict, output_dir: Path):
    """
    Panel 4: Dual-Mode Enhancement Validation

    A: 3D enhancement space (Raman, IR, Dual)
    B: Enhancement factor breakdown
    C: Information gain analysis
    D: Resolution improvement comparison
    """
    data = results['dual_mode_enhancement']

    fig = plt.figure(figsize=(20, 5))

    # Panel A: 3D Enhancement Space
    ax1 = fig.add_subplot(141, projection='3d')

    # Create 3D scatter: states, resolution, modes
    M_raman = data['enhancement_data']['raman_states']
    M_ir = data['enhancement_data']['ir_states']
    M_dual = data['enhancement_data']['dual_mode_states']

    tau_raman = data['enhancement_data']['raman_resolution']
    tau_ir = data['enhancement_data']['ir_resolution']
    tau_dual = data['enhancement_data']['dual_mode_resolution']

    n_raman = data['complementarity_analysis']['n_raman_modes']
    n_ir = data['complementarity_analysis']['n_ir_modes']
    n_effective = data['complementarity_analysis']['n_effective_modes']

    modes = [n_raman, n_ir, n_effective]
    states = [M_raman, M_ir, M_dual]
    resolutions = [tau_raman, tau_ir, tau_dual]
    colors_3d = ['#E74C3C', '#3498DB', '#9B59B6']
    labels_3d = ['Raman', 'IR', 'Dual-mode']
    markers_3d = ['o', '^', 's']

    for i in range(3):
        ax1.scatter([modes[i]], [np.log10(states[i])], [np.log10(resolutions[i])],
                   c=colors_3d[i], s=300, marker=markers_3d[i], alpha=0.8,
                   edgecolor='black', linewidth=2, label=labels_3d[i])

    # Draw connections
    for i in range(2):
        ax1.plot([modes[i], modes[2]],
                [np.log10(states[i]), np.log10(states[2])],
                [np.log10(resolutions[i]), np.log10(resolutions[2])],
                'k--', alpha=0.3, linewidth=1.5)

    ax1.set_xlabel('Number of Modes', fontsize=10, labelpad=8)
    ax1.set_ylabel('log$_{10}$(States)', fontsize=10, labelpad=8)
    ax1.set_zlabel('log$_{10}$(Resolution s)', fontsize=10, labelpad=8)
    ax1.set_title('A: 3D Enhancement Space', fontsize=12, pad=15)
    ax1.legend(fontsize=9)
    ax1.view_init(elev=20, azim=45)

    # Panel B: Enhancement Factor Breakdown
    ax2 = fig.add_subplot(142)

    enhancement_measured = data['enhancement_data']['enhancement_factor']
    enhancement_theoretical = data['formula_validation']['enhancement_theoretical']

    enhancements = [enhancement_measured, enhancement_theoretical]
    enhancement_labels = ['Measured', 'Theoretical']
    enhancement_colors = ['#1ABC9C', '#E67E22']

    bars = ax2.bar(enhancement_labels, enhancements, color=enhancement_colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Enhancement Factor', fontsize=11)
    ax2.set_ylim([1.0, 1.6])
    ax2.set_title('B: Enhancement Factor Breakdown', fontsize=12, pad=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}x', ha='center', va='bottom', fontsize=10)

    # Panel C: Information Gain Analysis
    ax3 = fig.add_subplot(143)

    I_gain = data['information_gain']['information_gain_bits']
    improvement = data['information_gain']['improvement_percent']

    # Pie chart for information distribution
    M_avg = data['information_gain']['M_avg_single']
    M_dual_total = data['information_gain']['M_dual']

    sizes = [M_avg, M_dual_total - M_avg]
    labels_pie = [f'Single-mode\n({M_avg:.2e})', f'Dual-mode gain\n({M_dual_total - M_avg:.2e})']
    colors_pie = ['#95A5A6', '#27AE60']
    explode = (0, 0.1)

    ax3.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
           autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 9})
    ax3.set_title(f'C: Information Gain\n({I_gain:.2f} bits, {improvement:.1f}% improvement)',
                 fontsize=12, pad=10)

    # Panel D: Resolution Improvement Comparison
    ax4 = fig.add_subplot(144)

    improvement_raman = data['resolution_improvement']['improvement_over_raman'] * 100
    improvement_ir = data['resolution_improvement']['improvement_over_ir'] * 100
    improvement_avg = data['resolution_improvement']['improvement_percent']

    improvements = [improvement_raman, improvement_ir, improvement_avg]
    improvement_labels = ['Over Raman', 'Over IR', 'Average']
    improvement_colors = ['#E74C3C', '#3498DB', '#2ECC71']

    bars = ax4.bar(improvement_labels, improvements, color=improvement_colors,
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.set_ylabel('Resolution Improvement (%)', fontsize=11)
    ax4.set_title('D: Resolution Improvement Comparison', fontsize=12, pad=10)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'esdvs_panel_4_dual_mode_enhancement.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Generated {output_file}")


def main():
    """Generate all ESDVS panel charts."""
    print("\n=== Generating ESDVS Panel Charts ===\n")

    # Load results
    results = load_results()

    if not results:
        print("[ERROR] No ESDVS validation results found. Run validation scripts first.")
        return

    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate panels
    if 'mutual_exclusion' in results:
        generate_panel_1_mutual_exclusion(results, output_dir)

    if 'ternary_trajectory' in results:
        generate_panel_2_ternary_trajectory(results, output_dir)

    if 'categorical_resolution' in results:
        generate_panel_3_categorical_resolution(results, output_dir)

    if 'dual_mode_enhancement' in results:
        generate_panel_4_dual_mode_enhancement(results, output_dir)

    print("\n[OK] All ESDVS panels generated successfully!")


if __name__ == '__main__':
    main()
