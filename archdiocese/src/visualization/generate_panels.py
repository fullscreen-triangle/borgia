"""
Visualization Panel Generator

Creates 4-panel charts for each validation result file.
Each panel includes at least one 3D visualization.

Author: Trajectory Completion Framework
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PanelGenerator:
    """Generates 4-panel visualization charts."""

    def __init__(self):
        """Initialize with results directory."""
        self.results_dir = Path(__file__).parent.parent.parent / 'results'
        self.figures_dir = Path(__file__).parent.parent.parent / 'figures'
        self.figures_dir.mkdir(exist_ok=True, parents=True)

    def create_panel(self, nrows=1, ncols=4, figsize=(20, 5)):
        """Create figure with 4 subplots."""
        fig = plt.figure(figsize=figsize)
        return fig

    def panel_1_oscillator_mapping(self):
        """
        Panel 1: Oscillator Mapping Results

        Charts:
        1. 3D S-space scatter plot
        2. Frequency distribution
        3. Categorical resolution comparison
        4. Coordinate heatmap
        """
        # Load data
        with open(self.results_dir / 'oscillator_mapping_results.json', 'r') as f:
            data = json.load(f)

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D S-space scatter
        ax1 = fig.add_subplot(141, projection='3d')

        coords = np.array([[d['coordinates']['S_k'],
                           d['coordinates']['S_t'],
                           d['coordinates']['S_e']] for d in data])

        colors = ['red' if d['type'] == 'uv_vis' else 'blue' for d in data]
        sizes = [d['n_oscillators'] * 10 for d in data]

        ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors, s=sizes, alpha=0.6, edgecolors='black')
        ax1.set_xlabel('$S_k$', fontsize=12)
        ax1.set_ylabel('$S_t$', fontsize=12)
        ax1.set_zlabel('$S_e$', fontsize=12)
        ax1.set_title('S-Space Coordinates', fontsize=14, fontweight='bold')

        # Chart 2: Frequency distribution
        ax2 = fig.add_subplot(142)

        all_freqs = []
        for d in data:
            all_freqs.extend(d['oscillators']['frequencies'])

        ax2.hist(np.log10(all_freqs), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_xlabel('log$_{10}$(Frequency [Hz])', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Oscillator Frequencies', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Chart 3: Categorical resolution
        ax3 = fig.add_subplot(143)

        names = [d['file'] for d in data]
        resolutions = [np.log10(d['categorical_resolution']) for d in data]
        colors_res = ['coral' if d['type'] == 'uv_vis' else 'lightblue' for d in data]

        bars = ax3.barh(range(len(names)), resolutions, color=colors_res, edgecolor='black')
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels([n.replace('.csv', '') for n in names], fontsize=10)
        ax3.set_xlabel('log$_{10}$($\\tau_{cat}$ [s])', fontsize=12)
        ax3.set_title('Categorical Resolution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # Chart 4: Coordinate heatmap
        ax4 = fig.add_subplot(144)

        coord_matrix = np.array([[d['coordinates']['S_k'],
                                 d['coordinates']['S_t'],
                                 d['coordinates']['S_e']] for d in data])

        im = ax4.imshow(coord_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['$S_k$', '$S_t$', '$S_e$'], fontsize=12)
        ax4.set_xticks(range(len(data)))
        ax4.set_xticklabels([d['file'].replace('.csv', '')[:4] for d in data], fontsize=9, rotation=45)
        ax4.set_title('Coordinate Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Value')

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_1_oscillator_mapping.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 1 saved: {output_file}")
        return output_file

    def panel_2_fixed_points(self):
        """
        Panel 2: Fixed Point Uniqueness

        Charts:
        1. 3D molecular distribution in S-space
        2. Pairwise distance distribution
        3. Depth vs coordinate scatter
        4. Density contour plot
        """
        # Load data
        with open(self.results_dir / 'fixed_point_uniqueness.json', 'r') as f:
            data = json.load(f)

        molecules = data['molecules']

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D molecular distribution
        ax1 = fig.add_subplot(141, projection='3d')

        coords = np.array([[m['coordinates']['S_k'],
                           m['coordinates']['S_t'],
                           m['coordinates']['S_e']] for m in molecules])

        depths = np.array([m['trit_string_length'] for m in molecules])

        scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            c=depths, cmap='plasma', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('$S_k$', fontsize=12)
        ax1.set_ylabel('$S_t$', fontsize=12)
        ax1.set_zlabel('$S_e$', fontsize=12)
        ax1.set_title('Molecular Fixed Points', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label('Depth', fontsize=10)

        # Chart 2: Pairwise distance distribution
        ax2 = fig.add_subplot(142)

        distances = data['validation']['pairwise_distances']
        distances_nonzero = [d for d in distances if d > 1e-6]

        ax2.hist(distances_nonzero, bins=50, alpha=0.7, color='teal', edgecolor='black')
        ax2.set_xlabel('Distance', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Pairwise Distances', fontsize=14, fontweight='bold')
        ax2.axvline(data['validation']['mean_distance'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Chart 3: Depth vs coordinates
        ax3 = fig.add_subplot(143)

        ax3.scatter(coords[:, 0], depths, alpha=0.5, s=30, c='blue', label='$S_k$')
        ax3.scatter(coords[:, 1], depths, alpha=0.5, s=30, c='orange', label='$S_t$')
        ax3.scatter(coords[:, 2], depths, alpha=0.5, s=30, c='green', label='$S_e$')
        ax3.set_xlabel('Coordinate Value', fontsize=12)
        ax3.set_ylabel('Trit Depth', fontsize=12)
        ax3.set_title('Depth vs Coordinates', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Chart 4: 2D density contour
        ax4 = fig.add_subplot(144)

        from scipy.stats import gaussian_kde

        xy = np.vstack([coords[:, 0], coords[:, 1]])
        z = gaussian_kde(xy)(xy)

        scatter = ax4.scatter(coords[:, 0], coords[:, 1], c=z, s=20, cmap='hot', alpha=0.6)
        ax4.set_xlabel('$S_k$', fontsize=12)
        ax4.set_ylabel('$S_t$', fontsize=12)
        ax4.set_title('Density (S$_k$-S$_t$ Projection)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Density')

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_2_fixed_points.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 2 saved: {output_file}")
        return output_file

    def panel_3_penultimate_states(self):
        """
        Panel 3: Penultimate State Validation

        Charts:
        1. 3D trajectory simulation
        2. Gateway trit distribution
        3. Method comparison
        4. Accuracy vs noise
        """
        # Load data
        with open(self.results_dir / 'penultimate_validation.json', 'r') as f:
            data = json.load(f)

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D simulated trajectory
        ax1 = fig.add_subplot(141, projection='3d')

        # Simulate a trajectory
        s_start = np.array([0.2, 0.3, 0.1])
        s_end = np.array([0.8, 0.7, 0.9])
        n_steps = 50

        t = np.linspace(0, 1, n_steps)
        traj = s_start[None, :] * (1 - t[:, None]) + s_end[None, :] * t[:, None]

        # Add noise
        noise = np.random.normal(0, 0.02, traj.shape)
        traj_noisy = np.clip(traj + noise, 0, 1)

        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, alpha=0.5, label='Clean')
        ax1.scatter(traj_noisy[:, 0], traj_noisy[:, 1], traj_noisy[:, 2],
                   c=t, cmap='coolwarm', s=20, alpha=0.7)
        ax1.scatter(*s_end, color='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Target')
        ax1.set_xlabel('$S_k$', fontsize=12)
        ax1.set_ylabel('$S_t$', fontsize=12)
        ax1.set_zlabel('$S_e$', fontsize=12)
        ax1.set_title('Trajectory to Fixed Point', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)

        # Chart 2: Gateway trit distribution
        ax2 = fig.add_subplot(142)

        gateway_trits = [p['gateway_trit'] for p in data['penultimate_structures'] if p.get('gateway_trit') is not None]

        unique, counts = np.unique(gateway_trits, return_counts=True)
        colors_trit = ['#FF6B6B', '#4ECDC4', '#95E1D3']

        bars = ax2.bar(unique, counts, color=[colors_trit[int(u)] for u in unique],
                      edgecolor='black', linewidth=1.5)
        ax2.set_xticks([0, 1, 2])
        ax2.set_xticklabels(['0', '1', '2'], fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Gateway Trit Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Chart 3: Method comparison
        ax3 = fig.add_subplot(143)

        comparisons = data['method_comparison']['comparisons']
        noise_levels = [c['noise_level'] for c in comparisons]
        penu_acc = [c['penultimate_accuracy'] * 100 for c in comparisons]
        fixed_acc = [c['fixed_point_accuracy'] * 100 for c in comparisons]

        x = np.arange(len(noise_levels))
        width = 0.35

        bars1 = ax3.bar(x - width/2, penu_acc, width, label='Penultimate',
                       color='steelblue', edgecolor='black')
        bars2 = ax3.bar(x + width/2, fixed_acc, width, label='Fixed Point',
                       color='coral', edgecolor='black')

        ax3.set_ylabel('Accuracy (%)', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{n:.3f}' for n in noise_levels], fontsize=10)
        ax3.set_title('Method Comparison', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Accuracy vs noise curve
        ax4 = fig.add_subplot(144)

        ax4.plot(noise_levels, penu_acc, 'o-', linewidth=2, markersize=8,
                color='steelblue', label='Penultimate', markeredgecolor='black')
        ax4.plot(noise_levels, fixed_acc, 's-', linewidth=2, markersize=8,
                color='coral', label='Fixed Point', markeredgecolor='black')
        ax4.set_xlabel('Noise Level', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_title('Robustness Analysis', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_3_penultimate_states.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 3 saved: {output_file}")
        return output_file

    def panel_4_trans_planckian(self):
        """
        Panel 4: Trans-Planckian Resolution

        Charts:
        1. 3D resolution landscape
        2. Resolution vs oscillators
        3. Isotope discrimination
        4. Planck comparison
        """
        # Load data
        with open(self.results_dir / 'trans_planckian_validation.json', 'r') as f:
            data = json.load(f)

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D resolution landscape
        ax1 = fig.add_subplot(141, projection='3d')

        # Create mesh for resolution surface
        N = np.logspace(0, 3, 30)
        omega = np.logspace(12, 15, 30)
        N_mesh, omega_mesh = np.meshgrid(N, omega)
        tau_cat = (2 * np.pi) / (N_mesh * omega_mesh)

        surf = ax1.plot_surface(np.log10(N_mesh), np.log10(omega_mesh), np.log10(tau_cat),
                               cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('log$_{10}$(N)', fontsize=12)
        ax1.set_ylabel('log$_{10}$($\\omega$)', fontsize=12)
        ax1.set_zlabel('log$_{10}$($\\tau_{cat}$)', fontsize=12)
        ax1.set_title('Resolution Landscape', fontsize=14, fontweight='bold')
        plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

        # Chart 2: Resolution vs N oscillators
        ax2 = fig.add_subplot(142)

        N_vals = np.logspace(0, 4, 100)
        omega_fixed = 1e13
        tau_vals = (2 * np.pi) / (N_vals * omega_fixed)

        t_planck = 5.39e-44

        ax2.loglog(N_vals, tau_vals, linewidth=3, color='steelblue', label='$\\tau_{cat}$')
        ax2.axhline(t_planck, color='red', linestyle='--', linewidth=2, label='Planck time')
        ax2.fill_between(N_vals, t_planck, tau_vals, where=(tau_vals < t_planck),
                        alpha=0.3, color='green', label='Trans-Planckian')
        ax2.set_xlabel('N oscillators', fontsize=12)
        ax2.set_ylabel('Resolution (s)', fontsize=12)
        ax2.set_title('N vs Resolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        # Chart 3: Isotope discrimination
        ax3 = fig.add_subplot(143)

        isotopes = data['isotope_discrimination']['isotopomers']
        names = list(isotopes.keys())
        freqs = list(isotopes.values())

        colors_iso = plt.cm.plasma(np.linspace(0, 1, len(names)))
        bars = ax3.bar(range(len(names)), freqs, color=colors_iso, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, fontsize=11)
        ax3.set_ylabel('Frequency (Hz)', fontsize=12)
        ax3.set_title('Isotopomer Frequencies', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Planck ratio comparison
        ax4 = fig.add_subplot(144)

        spectral_keys = [k for k in data.keys() if k.startswith('spectral_')]
        if spectral_keys:
            labels = []
            ratios = []

            for key in spectral_keys:
                labels.append(key.replace('spectral_', 'Spec '))
                ratios.append(np.log10(data[key]['ratio_to_planck']))

            colors_spec = ['green' if r < 0 else 'red' for r in ratios]
            bars = ax4.barh(range(len(labels)), ratios, color=colors_spec,
                           edgecolor='black', linewidth=1.5)
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels, fontsize=10)
            ax4.set_xlabel('log$_{10}$($\\tau_{cat}$ / $t_P$)', fontsize=12)
            ax4.axvline(0, color='black', linestyle='-', linewidth=2)
            ax4.set_title('Planck Ratio', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_4_trans_planckian.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 4 saved: {output_file}")
        return output_file

    def panel_5_state_counting(self):
        """
        Panel 5: State Counting

        Charts:
        1. 3D state evolution
        2. Entropy growth
        3. Time-state identity
        4. Ensemble comparison
        """
        # Load data
        with open(self.results_dir / 'state_counting.json', 'r') as f:
            data = json.load(f)

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D state evolution
        ax1 = fig.add_subplot(141, projection='3d')

        # Simulate state evolution
        n_steps = 100
        t = np.linspace(0, 1, n_steps)
        M = np.arange(n_steps)
        S_cat = 1.38e-23 * np.log(M + 1)

        ax1.plot(t, M, S_cat * 1e23, linewidth=3, color='purple')
        ax1.scatter(t[::10], M[::10], S_cat[::10] * 1e23, s=50, c=t[::10],
                   cmap='viridis', edgecolors='black', linewidth=1)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('States (M)', fontsize=12)
        ax1.set_zlabel('Entropy ($10^{-23}$ J/K)', fontsize=12)
        ax1.set_title('State Evolution', fontsize=14, fontweight='bold')

        # Chart 2: Entropy growth
        ax2 = fig.add_subplot(142)

        M_vals = np.logspace(0, 6, 100)
        S_vals = 1.38e-23 * np.log(M_vals)

        ax2.loglog(M_vals, S_vals * 1e23, linewidth=3, color='darkgreen')
        ax2.set_xlabel('Categorical States (M)', fontsize=12)
        ax2.set_ylabel('Entropy ($10^{-23}$ J/K)', fontsize=12)
        ax2.set_title('Entropy Growth: S = k$_B$ ln(M)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')

        # Chart 3: Time-state identity
        ax3 = fig.add_subplot(143)

        single = data['single_oscillator']
        ensemble = data['ensemble']

        # Plot verification
        categories = ['Single\nOscillator', 'Ensemble']
        durations = [single['duration_s'], ensemble['duration_s']]
        computed = [single['time_computed_s'], ensemble['time_computed_s']]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax3.bar(x - width/2, durations, width, label='Actual',
                       color='skyblue', edgecolor='black')
        bars2 = ax3.bar(x + width/2, computed, width, label='Computed',
                       color='lightcoral', edgecolor='black')

        ax3.set_ylabel('Time (s)', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories, fontsize=11)
        ax3.set_title('Time-State Identity', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Ensemble state counting
        ax4 = fig.add_subplot(144)

        N_ensemble = [1, 10, 100, 1000]
        states_per_N = [single['total_categorical_states'] * n for n in N_ensemble]

        bars = ax4.bar(range(len(N_ensemble)), states_per_N,
                      color=plt.cm.plasma(np.linspace(0.2, 0.9, len(N_ensemble))),
                      edgecolor='black', linewidth=1.5)
        ax4.set_xticks(range(len(N_ensemble)))
        ax4.set_xticklabels([f'N={n}' for n in N_ensemble], fontsize=11)
        ax4.set_ylabel('Total States', fontsize=12)
        ax4.set_yscale('log')
        ax4.set_title('Ensemble Scaling', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_5_state_counting.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 5 saved: {output_file}")
        return output_file

    def panel_6_atlas(self):
        """
        Panel 6: Fixed Point Atlas

        Charts:
        1. 3D full atlas (molecular + spectral)
        2. Coverage analysis
        3. Density map
        4. Type distribution
        """
        # Load data
        with open(self.results_dir / 'fixed_point_atlas.json', 'r') as f:
            atlas_data = json.load(f)

        entries = atlas_data['entries']

        fig = plt.figure(figsize=(20, 5))

        # Chart 1: 3D full atlas
        ax1 = fig.add_subplot(141, projection='3d')

        coords = np.array([[e['fixed_point']['S_k'],
                           e['fixed_point']['S_t'],
                           e['fixed_point']['S_e']] for e in entries])

        # Color by type
        types = ['molecular' if 'molecule_name' in e else 'spectral' for e in entries]
        colors_type = ['blue' if t == 'molecular' else 'red' for t in types]

        ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=colors_type, s=15, alpha=0.5, edgecolors='black', linewidth=0.3)
        ax1.set_xlabel('$S_k$', fontsize=12)
        ax1.set_ylabel('$S_t$', fontsize=12)
        ax1.set_zlabel('$S_e$', fontsize=12)
        ax1.set_title('Complete Atlas', fontsize=14, fontweight='bold')

        # Add legend
        blue_patch = mpatches.Patch(color='blue', label='Molecular')
        red_patch = mpatches.Patch(color='red', label='Spectral')
        ax1.legend(handles=[blue_patch, red_patch], fontsize=10)

        # Chart 2: Coverage analysis
        ax2 = fig.add_subplot(142)

        coverage_data = [
            ['$S_k$', coords[:, 0].min(), coords[:, 0].max()],
            ['$S_t$', coords[:, 1].min(), coords[:, 1].max()],
            ['$S_e$', coords[:, 2].min(), coords[:, 2].max()]
        ]

        y_pos = np.arange(len(coverage_data))
        mins = [d[1] for d in coverage_data]
        maxs = [d[2] for d in coverage_data]
        ranges = [m - n for m, n in zip(maxs, mins)]

        bars = ax2.barh(y_pos, ranges, left=mins,
                       color=['#FF6B6B', '#4ECDC4', '#95E1D3'],
                       edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([d[0] for d in coverage_data], fontsize=12)
        ax2.set_xlabel('Coordinate Range', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_title('S-Space Coverage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Chart 3: 2D density heatmap
        ax3 = fig.add_subplot(143)

        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=30)

        im = ax3.imshow(H.T, origin='lower', cmap='hot', aspect='auto',
                       extent=[0, 1, 0, 1], interpolation='bilinear')
        ax3.set_xlabel('$S_k$', fontsize=12)
        ax3.set_ylabel('$S_t$', fontsize=12)
        ax3.set_title('Density Map', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Count')

        # Chart 4: Type distribution pie
        ax4 = fig.add_subplot(144)

        type_counts = {'Molecular': types.count('molecular'),
                      'Spectral': types.count('spectral')}

        colors_pie = ['#3498db', '#e74c3c']
        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax4.pie(type_counts.values(),
                                           labels=type_counts.keys(),
                                           colors=colors_pie,
                                           autopct='%1.1f%%',
                                           explode=explode,
                                           startangle=90,
                                           textprops={'fontsize': 12, 'weight': 'bold'})

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
            autotext.set_weight('bold')

        ax4.set_title('Atlas Composition', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_file = self.figures_dir / 'panel_6_atlas.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[OK] Panel 6 saved: {output_file}")
        return output_file

    def generate_all_panels(self):
        """Generate all visualization panels."""
        print("\n=== Generating Visualization Panels ===\n")

        panels = []

        panels.append(self.panel_1_oscillator_mapping())
        panels.append(self.panel_2_fixed_points())
        panels.append(self.panel_3_penultimate_states())
        panels.append(self.panel_4_trans_planckian())
        panels.append(self.panel_5_state_counting())
        panels.append(self.panel_6_atlas())

        print(f"\n[OK] Generated {len(panels)} visualization panels")
        print(f"[OK] Saved to: {self.figures_dir}")

        return panels


def main():
    """Generate all panels."""
    generator = PanelGenerator()
    panels = generator.generate_all_panels()
    return panels


if __name__ == '__main__':
    main()
