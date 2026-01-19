"""
Instrument Visualization Suite
==============================

Comprehensive visualizations for partition-coordinate spectroscopic instruments.
Each instrument type gets specialized charts reflecting its physical characteristics.

Instrument Types:
- Depth (n) → XPS / X-ray spectroscopy
- Complexity (l) → UV-Vis / Optical spectroscopy
- Orientation (m) → Zeeman / Microwave spectroscopy
- Chirality (s) → NMR / Radio spectroscopy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.special import sph_harm
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os


# ============================================================================
# PANEL 1: DEPTH COORDINATE (n) - XPS / X-RAY SPECTROSCOPY
# ============================================================================

def create_xps_panel(save_dir: str):
    """
    XPS / X-ray spectroscopy visualization panel for depth coordinate (n).

    Includes:
    - 3D core electron binding energy surface
    - Radial probability distributions
    - Photoelectron kinetic energy diagram
    - Circular bar chart of shell capacities
    - Auger transition heatmap
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('DEPTH COORDINATE (n) — X-ray Photoelectron Spectroscopy',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. 3D Binding Energy Surface ---
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')

    n_vals = np.arange(1, 8)
    l_vals = np.arange(0, 7)
    N, L = np.meshgrid(n_vals, l_vals)

    # Binding energy: E ∝ Z²/n² (simplified)
    Z = 26  # Iron as example
    # Mask invalid l >= n
    valid = L < N
    E_binding = np.where(valid, -13.6 * Z**2 / N**2, np.nan)

    surf = ax1.plot_surface(N, L, E_binding, cmap='plasma', alpha=0.8,
                            edgecolor='black', linewidth=0.3)
    ax1.set_xlabel('Principal (n)')
    ax1.set_ylabel('Angular (l)')
    ax1.set_zlabel('Binding Energy (eV)')
    ax1.set_title('Core-Level Binding Energy Surface')
    ax1.view_init(elev=25, azim=45)

    # --- 2. Radial Probability Distribution ---
    ax2 = fig.add_subplot(gs[0, 2:4])

    r = np.linspace(0, 20, 500)
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for n in range(1, 6):
        # Simplified radial probability (hydrogen-like)
        a0 = 1  # Bohr radius units
        rho = 2 * r / (n * a0)
        # R_n0 for s-orbitals
        if n == 1:
            R = 2 * np.exp(-rho/2)
        elif n == 2:
            R = (1/np.sqrt(8)) * (2 - rho) * np.exp(-rho/2)
        elif n == 3:
            R = (2/81/np.sqrt(3)) * (27 - 18*rho + 2*rho**2) * np.exp(-rho/2)
        elif n == 4:
            R = (1/768) * (192 - 144*rho + 24*rho**2 - rho**3) * np.exp(-rho/2)
        else:
            R = np.exp(-rho/2) * (1 - rho/n)

        P = (r**2) * (R**2)
        P = P / np.max(P)  # Normalize
        ax2.fill_between(r, P * n, n - 1, alpha=0.6, color=colors[n-1])
        ax2.plot(r, P * n + (n - 1), color=colors[n-1], linewidth=2,
                 label=f'n={n}')

    ax2.set_xlabel('Radial Distance (a₀)')
    ax2.set_ylabel('Probability (stacked)')
    ax2.set_title('Radial Probability Distributions')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 15)

    # --- 3. Circular Bar Chart: Shell Capacities (2n²) ---
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')

    n_shells = 7
    capacities = [2 * n**2 for n in range(1, n_shells + 1)]
    cumulative = np.cumsum(capacities)

    theta = np.linspace(0, 2*np.pi, n_shells, endpoint=False)
    width = 2*np.pi / n_shells * 0.8

    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, n_shells))
    bars = ax3.bar(theta, capacities, width=width, color=colors,
                   edgecolor='black', alpha=0.8)

    # Add capacity labels
    for angle, cap, bar in zip(theta, capacities, bars):
        ax3.text(angle, cap + 5, f'{cap}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax3.set_title('Shell Capacity (2n²)', pad=20)
    ax3.set_xticks(theta)
    ax3.set_xticklabels([f'n={i}' for i in range(1, n_shells + 1)])

    # --- 4. Photoelectron Energy Diagram ---
    ax4 = fig.add_subplot(gs[1, 1])

    # Energy level diagram
    photon_energy = 1500  # eV (Al Kα)
    binding_energies = {'1s': 710, '2s': 85, '2p': 56, '3s': 8, '3p': 4}

    y_positions = np.arange(len(binding_energies))
    kinetic_energies = [photon_energy - be for be in binding_energies.values()]

    # Draw energy levels
    for i, (orbital, be) in enumerate(binding_energies.items()):
        ke = photon_energy - be
        ax4.barh(i, ke, height=0.6, color=plt.cm.coolwarm(be/800),
                 edgecolor='black', alpha=0.8)
        ax4.text(ke + 20, i, f'KE={ke:.0f} eV', va='center', fontsize=9)

    ax4.set_yticks(y_positions)
    ax4.set_yticklabels(binding_energies.keys())
    ax4.set_xlabel('Kinetic Energy (eV)')
    ax4.set_title(f'XPS Kinetic Energies\n(hν = {photon_energy} eV)')
    ax4.axvline(photon_energy, color='red', linestyle='--', alpha=0.5,
                label=f'hν = {photon_energy} eV')

    # --- 5. Auger Transition Heatmap ---
    ax5 = fig.add_subplot(gs[1, 2:4])

    shells = ['K', 'L₁', 'L₂', 'L₃', 'M₁', 'M₂', 'M₃']
    n_shells_aug = len(shells)

    # Auger transition probabilities (simplified)
    auger_matrix = np.zeros((n_shells_aug, n_shells_aug))
    for i in range(n_shells_aug):
        for j in range(i+1, n_shells_aug):
            # Higher probability for adjacent shells
            auger_matrix[i, j] = np.exp(-(j - i) / 2) * (1 - i/n_shells_aug)

    im = ax5.imshow(auger_matrix, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(range(n_shells_aug))
    ax5.set_yticks(range(n_shells_aug))
    ax5.set_xticklabels(shells)
    ax5.set_yticklabels(shells)
    ax5.set_xlabel('Final State Shell')
    ax5.set_ylabel('Initial Vacancy Shell')
    ax5.set_title('Auger Transition Probability Matrix')
    plt.colorbar(im, ax=ax5, label='Probability')

    # --- 6. 3D Electron Density Isosurface ---
    ax6 = fig.add_subplot(gs[2, 0:2], projection='3d')

    # Create spherical coordinates
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 50)
    PHI, THETA = np.meshgrid(phi, theta)

    # Multiple orbital shells
    for n in [1, 2, 3]:
        r = n * 2  # Shell radius
        X = r * np.sin(THETA) * np.cos(PHI)
        Y = r * np.sin(THETA) * np.sin(PHI)
        Z = r * np.cos(THETA)

        alpha = 0.3 / n
        ax6.plot_surface(X, Y, Z, color=plt.cm.viridis(n/4),
                        alpha=alpha, edgecolor='none')

    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.set_title('Electron Shell Isosurfaces (n=1,2,3)')
    ax6.set_box_aspect([1,1,1])

    # --- 7. XPS Spectrum (Peak Areas) ---
    ax7 = fig.add_subplot(gs[2, 2])

    binding_range = np.linspace(0, 800, 1000)
    spectrum = np.zeros_like(binding_range)

    peaks = [
        (710, 50, 0.8, '1s'),
        (720, 30, 0.6, '1s shake-up'),
        (85, 15, 0.3, '2s'),
        (56, 20, 0.5, '2p₃/₂'),
        (54, 18, 0.4, '2p₁/₂'),
    ]

    for be, width, intensity, label in peaks:
        peak = intensity * np.exp(-0.5 * ((binding_range - be) / (width/2.355))**2)
        spectrum += peak

    ax7.fill_between(binding_range, spectrum, alpha=0.6, color='steelblue')
    ax7.plot(binding_range, spectrum, 'k-', linewidth=1)

    for be, _, intensity, label in peaks:
        if intensity > 0.3:
            ax7.annotate(label, (be, intensity),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8)

    ax7.set_xlabel('Binding Energy (eV)')
    ax7.set_ylabel('Intensity')
    ax7.set_title('XPS Survey Spectrum (Fe)')
    ax7.invert_xaxis()
    ax7.set_xlim(800, 0)

    # --- 8. Cross-section scaling (n⁻³) ---
    ax8 = fig.add_subplot(gs[2, 3])

    n_range = np.linspace(1, 7, 100)
    cross_section = n_range ** -3

    ax8.semilogy(n_range, cross_section, 'b-', linewidth=2)
    ax8.fill_between(n_range, cross_section, alpha=0.3)

    # Mark integer n values
    for n in range(1, 8):
        ax8.scatter(n, n**-3, s=100, c='red', zorder=5, edgecolor='black')
        ax8.annotate(f'n={n}', (n, n**-3), textcoords="offset points",
                    xytext=(10, 0), fontsize=9)

    ax8.set_xlabel('Principal Quantum Number (n)')
    ax8.set_ylabel('Relative Cross-Section')
    ax8.set_title('Photoionization Cross-Section (∝ n⁻³)')
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panel_xps_depth_coordinate.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  [XPS Panel] Depth coordinate (n) visualization saved")


# ============================================================================
# PANEL 2: COMPLEXITY COORDINATE (l) - UV-VIS / OPTICAL SPECTROSCOPY
# ============================================================================

def create_uvvis_panel(save_dir: str):
    """
    UV-Vis / Optical spectroscopy visualization panel for complexity coordinate (l).

    Includes:
    - Spherical harmonics 3D plots
    - Angular momentum coupling diagram
    - Selection rule matrix
    - Absorption spectrum with vibronic structure
    - Molecular orbital energy diagram
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('COMPLEXITY COORDINATE (l) — UV-Vis / Optical Spectroscopy',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Spherical Harmonics Surface Plot ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    THETA, PHI = np.meshgrid(theta, phi)

    l, m = 2, 0  # d-orbital
    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y.real)

    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    colors = Y.real
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    ax1.plot_surface(X, Y_coord, Z, facecolors=cm.RdBu(colors),
                     alpha=0.8, rstride=2, cstride=2)
    ax1.set_title(f'Y₂⁰ (d-orbital)')
    ax1.set_box_aspect([1,1,1])
    ax1.axis('off')

    # --- 2. Multiple Spherical Harmonics ---
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')

    l, m = 3, 2  # f-orbital
    Y = sph_harm(m, l, PHI, THETA)
    R = np.abs(Y.real) * 0.8 + 0.2

    X = R * np.sin(THETA) * np.cos(PHI)
    Y_coord = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)

    colors = Y.real
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    ax2.plot_surface(X, Y_coord, Z, facecolors=cm.coolwarm(colors),
                     alpha=0.8, rstride=2, cstride=2)
    ax2.set_title(f'Y₃² (f-orbital)')
    ax2.set_box_aspect([1,1,1])
    ax2.axis('off')

    # --- 3. Selection Rules Matrix (Δl = ±1) ---
    ax3 = fig.add_subplot(gs[0, 2:4])

    l_max = 6
    selection_matrix = np.zeros((l_max, l_max))

    for l1 in range(l_max):
        for l2 in range(l_max):
            if abs(l2 - l1) == 1:
                selection_matrix[l1, l2] = 1

    im = ax3.imshow(selection_matrix, cmap='Greens', aspect='equal')

    # Add arrows for allowed transitions
    for l1 in range(l_max):
        for l2 in range(l_max):
            if selection_matrix[l1, l2] == 1:
                ax3.annotate('', xy=(l2, l1), xytext=(l1, l1),
                            arrowprops=dict(arrowstyle='->', color='darkgreen',
                                          lw=2, mutation_scale=15))

    ax3.set_xticks(range(l_max))
    ax3.set_yticks(range(l_max))
    ax3.set_xticklabels(['s', 'p', 'd', 'f', 'g', 'h'])
    ax3.set_yticklabels(['s', 'p', 'd', 'f', 'g', 'h'])
    ax3.set_xlabel('Final State (l\')')
    ax3.set_ylabel('Initial State (l)')
    ax3.set_title('Selection Rules: Δl = ±1 (Allowed Transitions)')

    # --- 4. Radar Chart: Orbital Characteristics ---
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')

    categories = ['Nodes', 'Angular\nMomentum', 'Radial\nExtent',
                  'Shielding', 'Energy', 'Degeneracy']
    N = len(categories)

    # Values for different l
    orbitals = {
        's (l=0)': [0, 0, 3, 5, 5, 1],
        'p (l=1)': [1, 2, 4, 4, 4, 3],
        'd (l=2)': [2, 4, 5, 3, 3, 5],
        'f (l=3)': [3, 5, 6, 2, 2, 7]
    }

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['blue', 'green', 'orange', 'red']
    for i, (name, values) in enumerate(orbitals.items()):
        values = values + values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=name,
                color=colors[i], markersize=6)
        ax4.fill(angles, values, alpha=0.15, color=colors[i])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, size=8)
    ax4.set_title('Orbital Characteristics', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # --- 5. UV-Vis Absorption Spectrum with Vibronic Structure ---
    ax5 = fig.add_subplot(gs[1, 1:3])

    wavelength = np.linspace(200, 800, 1000)

    # Electronic transitions with vibronic progressions
    def vibronic_band(wl, center, width, intensity, vib_spacing=15, n_vib=5):
        band = np.zeros_like(wl)
        for v in range(n_vib):
            peak_center = center - v * vib_spacing
            peak_int = intensity * np.exp(-0.5 * v)  # Franck-Condon
            band += peak_int * np.exp(-0.5 * ((wl - peak_center) / (width * (1 + 0.1*v)))**2)
        return band

    # Multiple electronic transitions
    spectrum = np.zeros_like(wavelength)
    spectrum += vibronic_band(wavelength, 450, 20, 0.8, 12, 6)  # S0→S1
    spectrum += vibronic_band(wavelength, 320, 15, 0.5, 10, 4)  # S0→S2
    spectrum += vibronic_band(wavelength, 260, 10, 1.0, 8, 3)   # S0→S3

    # Color the spectrum by wavelength
    for i in range(len(wavelength)-1):
        wl = wavelength[i]
        if 380 <= wl <= 750:
            color = wavelength_to_rgb(wl)
        else:
            color = 'gray'
        ax5.fill_between(wavelength[i:i+2], spectrum[i:i+2],
                        color=color, alpha=0.8)

    ax5.plot(wavelength, spectrum, 'k-', linewidth=0.5)
    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_ylabel('Absorbance')
    ax5.set_title('UV-Vis Absorption with Vibronic Structure')
    ax5.set_xlim(200, 800)

    # --- 6. Jablonski Diagram ---
    ax6 = fig.add_subplot(gs[1, 3])

    # Energy levels
    levels = {
        'S₀': (0.2, [0, 0.05, 0.09, 0.12]),
        'S₁': (0.5, [0.4, 0.45, 0.49, 0.52]),
        'S₂': (0.5, [0.7, 0.74, 0.77]),
        'T₁': (0.8, [0.3, 0.34, 0.37])
    }

    for state, (x, energies) in levels.items():
        for i, e in enumerate(energies):
            lw = 3 if i == 0 else 1
            ax6.hlines(e, x-0.1, x+0.1, colors='black', linewidth=lw)
        ax6.text(x, energies[0] - 0.05, state, ha='center', fontsize=10,
                fontweight='bold')

    # Transitions
    # Absorption
    ax6.annotate('', xy=(0.5, 0.4), xytext=(0.2, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax6.text(0.35, 0.2, 'Abs', color='blue', fontsize=9, rotation=60)

    # Fluorescence
    ax6.annotate('', xy=(0.2, 0), xytext=(0.5, 0.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax6.text(0.38, 0.15, 'Fl', color='green', fontsize=9, rotation=60)

    # ISC
    ax6.annotate('', xy=(0.8, 0.3), xytext=(0.5, 0.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                              linestyle='dashed'))
    ax6.text(0.65, 0.38, 'ISC', color='gray', fontsize=9)

    # Phosphorescence
    ax6.annotate('', xy=(0.2, 0), xytext=(0.8, 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax6.text(0.5, 0.1, 'Phos', color='red', fontsize=9)

    ax6.set_xlim(0, 1)
    ax6.set_ylim(-0.1, 0.9)
    ax6.set_ylabel('Energy')
    ax6.set_title('Jablonski Diagram')
    ax6.axis('off')

    # --- 7. Frequency Scaling ω_l ∝ l(l+1) ---
    ax7 = fig.add_subplot(gs[2, 0])

    l_range = np.arange(0, 7)
    omega_l = l_range * (l_range + 1)

    # Area chart
    ax7.fill_between(l_range, omega_l, alpha=0.5, color='purple')
    ax7.plot(l_range, omega_l, 'ko-', markersize=10, linewidth=2)

    for l, w in zip(l_range, omega_l):
        ax7.annotate(f'{w}', (l, w), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10)

    ax7.set_xlabel('Angular Momentum (l)')
    ax7.set_ylabel('ω_l / ω₀β')
    ax7.set_title('Frequency Scaling: ω_l ∝ l(l+1)')
    ax7.set_xticks(l_range)
    ax7.set_xticklabels(['s', 'p', 'd', 'f', 'g', 'h', 'i'])
    ax7.grid(True, alpha=0.3)

    # --- 8. Transition Dipole Moments ---
    ax8 = fig.add_subplot(gs[2, 1], projection='3d')

    # Electric dipole vectors for transitions
    transitions = [
        ('s→p', [0, 0, 0], [1, 0, 0], 'red'),
        ('p→d', [0, 0, 0], [0, 1, 0], 'green'),
        ('d→f', [0, 0, 0], [0, 0, 1], 'blue'),
    ]

    for name, start, end, color in transitions:
        ax8.quiver(start[0], start[1], start[2],
                   end[0], end[1], end[2],
                   color=color, arrow_length_ratio=0.2, linewidth=3,
                   label=name)

    # Add coordinate axes
    ax8.quiver(0, 0, 0, 1.5, 0, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1)
    ax8.quiver(0, 0, 0, 0, 1.5, 0, color='gray', alpha=0.3, arrow_length_ratio=0.1)
    ax8.quiver(0, 0, 0, 0, 0, 1.5, color='gray', alpha=0.3, arrow_length_ratio=0.1)

    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    ax8.set_zlabel('z')
    ax8.set_title('Transition Dipole Moments')
    ax8.legend()
    ax8.set_box_aspect([1,1,1])

    # --- 9. Oscillator Strength Distribution ---
    ax9 = fig.add_subplot(gs[2, 2])

    transitions_data = ['1s→2p', '2s→3p', '2p→3d', '3s→4p', '3p→4d', '3d→4f']
    f_values = [0.416, 0.103, 0.637, 0.048, 0.122, 0.876]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(transitions_data)))
    bars = ax9.barh(transitions_data, f_values, color=colors, edgecolor='black')

    ax9.set_xlabel('Oscillator Strength (f)')
    ax9.set_title('Oscillator Strengths')
    ax9.set_xlim(0, 1)

    for bar, f in zip(bars, f_values):
        ax9.text(f + 0.02, bar.get_y() + bar.get_height()/2,
                f'{f:.3f}', va='center', fontsize=9)

    # --- 10. Degeneracy Pattern ---
    ax10 = fig.add_subplot(gs[2, 3])

    l_vals = range(6)
    degeneracies = [2*l + 1 for l in l_vals]
    labels = ['s', 'p', 'd', 'f', 'g', 'h']

    # Stacked representation
    bottom = 0
    for l, deg, label in zip(l_vals, degeneracies, labels):
        color = plt.cm.Set3(l / 6)
        ax10.bar(0.5, deg, bottom=bottom, width=0.6, color=color,
                edgecolor='black', label=f'{label} ({deg})')
        ax10.text(0.5, bottom + deg/2, f'{label}\n(2×{l}+1={deg})',
                 ha='center', va='center', fontsize=9, fontweight='bold')
        bottom += deg

    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, sum(degeneracies) + 2)
    ax10.set_ylabel('Cumulative States')
    ax10.set_title('Degeneracy: 2l+1')
    ax10.set_xticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panel_uvvis_complexity_coordinate.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  [UV-Vis Panel] Complexity coordinate (l) visualization saved")


def wavelength_to_rgb(wavelength):
    """Convert wavelength (nm) to RGB color."""
    if wavelength < 380:
        return (0.5, 0, 0.5)
    elif wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0
        b = 1
    elif wavelength < 490:
        r = 0
        g = (wavelength - 440) / (490 - 440)
        b = 1
    elif wavelength < 510:
        r = 0
        g = 1
        b = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1
        b = 0
    elif wavelength < 645:
        r = 1
        g = -(wavelength - 645) / (645 - 580)
        b = 0
    elif wavelength <= 750:
        r = 1
        g = 0
        b = 0
    else:
        return (0.5, 0, 0)

    return (r, g, b)


# ============================================================================
# PANEL 3: ORIENTATION COORDINATE (m) - ZEEMAN / MICROWAVE SPECTROSCOPY
# ============================================================================

def create_zeeman_panel(save_dir: str):
    """
    Zeeman / Microwave spectroscopy visualization panel for orientation coordinate (m).

    Includes:
    - Magnetic field vector diagrams
    - Zeeman splitting patterns
    - Polar plots of angular distributions
    - Larmor precession visualization
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('ORIENTATION COORDINATE (m) — Zeeman / Microwave Spectroscopy',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Polar Bar Chart: m-state distribution ---
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')

    l = 3  # f-orbital as example
    m_values = list(range(-l, l+1))
    n_states = len(m_values)

    # Energy in magnetic field: E_m ∝ m
    theta = np.linspace(0, 2*np.pi, n_states, endpoint=False)
    energies = np.array(m_values) + l + 1  # Shift to positive

    colors = plt.cm.coolwarm(np.linspace(0, 1, n_states))
    width = 2*np.pi / n_states * 0.8

    bars = ax1.bar(theta, energies, width=width, color=colors,
                   edgecolor='black', alpha=0.8)

    ax1.set_xticks(theta)
    ax1.set_xticklabels([f'm={m}' for m in m_values])
    ax1.set_title('m-State Energy Distribution (l=3)', pad=20)

    # --- 2. Zeeman Splitting Diagram ---
    ax2 = fig.add_subplot(gs[0, 1:3])

    # No field (left) vs with field (right)
    x_no_field = 0.2
    x_field = 0.7

    # Original degenerate level
    ax2.hlines(0.5, x_no_field-0.1, x_no_field+0.1, colors='black', linewidth=3)
    ax2.text(x_no_field, 0.55, 'l=2 (5-fold degenerate)', ha='center', fontsize=10)

    # Split levels
    l = 2
    m_vals = range(-l, l+1)
    spacing = 0.08

    for i, m in enumerate(m_vals):
        y = 0.5 + m * spacing
        color = plt.cm.coolwarm((m + l) / (2*l))
        ax2.hlines(y, x_field-0.1, x_field+0.1, colors=color, linewidth=2)
        ax2.text(x_field+0.12, y, f'm={m:+d}', va='center', fontsize=9)

        # Connecting lines
        ax2.plot([x_no_field+0.1, x_field-0.1], [0.5, y],
                'k--', alpha=0.3, linewidth=0.5)

    # Labels
    ax2.text(x_no_field, 0.1, 'B = 0', ha='center', fontsize=12, fontweight='bold')
    ax2.text(x_field, 0.1, 'B > 0', ha='center', fontsize=12, fontweight='bold')

    # Magnetic field arrow
    ax2.annotate('', xy=(0.85, 0.8), xytext=(0.85, 0.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    ax2.text(0.88, 0.5, 'B', fontsize=14, fontweight='bold', color='blue')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Zeeman Splitting (Normal Zeeman Effect)')
    ax2.axis('off')

    # --- 3. Angular Momentum Precession (3D) ---
    ax3 = fig.add_subplot(gs[0, 3], projection='3d')

    # Larmor precession cone
    t = np.linspace(0, 4*np.pi, 200)
    l_mag = 2
    m = 1

    theta_L = np.arccos(m / np.sqrt(l_mag*(l_mag+1)))

    x = np.sin(theta_L) * np.cos(t)
    y = np.sin(theta_L) * np.sin(t)
    z = np.cos(theta_L) * np.ones_like(t)

    ax3.plot(x, y, z, 'b-', linewidth=2, label='L precession')

    # Angular momentum vector at different times
    for i in range(0, len(t), 50):
        ax3.quiver(0, 0, 0, x[i], y[i], z[i],
                   color='red', alpha=0.5, arrow_length_ratio=0.1)

    # B-field along z
    ax3.quiver(0, 0, 0, 0, 0, 1.5, color='blue', linewidth=3,
               arrow_length_ratio=0.1, label='B')

    ax3.set_xlabel('Lx')
    ax3.set_ylabel('Ly')
    ax3.set_zlabel('Lz')
    ax3.set_title('Larmor Precession')
    ax3.set_box_aspect([1,1,1])
    ax3.legend()

    # --- 4. Selection Rules for m (Δm = 0, ±1) ---
    ax4 = fig.add_subplot(gs[1, 0])

    # Transition diagram
    l = 2
    levels_initial = list(range(-l, l+1))
    levels_final = list(range(-(l-1), l))  # l-1 for Δl=-1

    y_init = 0.7
    y_final = 0.3

    # Draw levels
    for i, m in enumerate(levels_initial):
        x = 0.1 + 0.16 * i
        ax4.hlines(y_init, x-0.05, x+0.05, colors='blue', linewidth=2)
        ax4.text(x, y_init+0.05, f'{m:+d}', ha='center', fontsize=8, color='blue')

    for i, m in enumerate(levels_final):
        x = 0.18 + 0.16 * i
        ax4.hlines(y_final, x-0.05, x+0.05, colors='red', linewidth=2)
        ax4.text(x, y_final-0.08, f'{m:+d}', ha='center', fontsize=8, color='red')

    # Allowed transitions
    transition_colors = {'σ⁻': 'purple', 'π': 'green', 'σ⁺': 'orange'}

    for i, m_i in enumerate(levels_initial):
        for j, m_f in enumerate(levels_final):
            dm = m_f - m_i
            if dm in [-1, 0, 1]:
                x_i = 0.1 + 0.16 * i
                x_f = 0.18 + 0.16 * j

                if dm == -1:
                    color = 'purple'
                    ls = '-'
                elif dm == 0:
                    color = 'green'
                    ls = '--'
                else:
                    color = 'orange'
                    ls = '-.'

                ax4.plot([x_i, x_f], [y_init, y_final], color=color,
                        linestyle=ls, alpha=0.6, linewidth=1)

    ax4.text(0.5, 0.85, 'Initial (l=2)', ha='center', color='blue', fontweight='bold')
    ax4.text(0.5, 0.15, 'Final (l=1)', ha='center', color='red', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Δm Selection Rules')
    ax4.axis('off')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', label='σ⁻ (Δm=-1)'),
        Line2D([0], [0], color='green', linestyle='--', label='π (Δm=0)'),
        Line2D([0], [0], color='orange', linestyle='-.', label='σ⁺ (Δm=+1)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')

    # --- 5. Zeeman Spectrum (Triplet) ---
    ax5 = fig.add_subplot(gs[1, 1])

    freq = np.linspace(-3, 3, 1000)
    larmor = 1.0  # Larmor frequency
    width = 0.15

    # Three peaks
    spectrum = np.zeros_like(freq)
    for dm, label, color in [(-1, 'σ⁻', 'purple'), (0, 'π', 'green'), (1, 'σ⁺', 'orange')]:
        peak = 0.8 * np.exp(-0.5 * ((freq - dm*larmor) / width)**2)
        spectrum += peak
        ax5.fill_between(freq, peak, alpha=0.5, color=color, label=label)

    ax5.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Frequency shift (ω - ω₀) / ωL')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Normal Zeeman Triplet')
    ax5.legend()

    # --- 6. Magnetic Quantum Number Phase Diagram (Polar) ---
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')

    # Phase of exp(imφ) for different m
    phi = np.linspace(0, 2*np.pi, 100)

    for m in [-2, -1, 0, 1, 2]:
        # Real part of e^(imφ)
        wave = np.cos(m * phi)
        # Shift to positive for polar plot
        r = 1 + 0.3 * wave

        color = plt.cm.RdYlBu((m + 2) / 4)
        ax6.plot(phi, r, linewidth=2, label=f'm={m}', color=color)

    ax6.set_title('Phase Pattern: Re(eⁱᵐᵠ)', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # --- 7. Vector Model of Angular Momentum ---
    ax7 = fig.add_subplot(gs[1, 3], projection='3d')

    l = 2
    L_magnitude = np.sqrt(l * (l + 1))

    # Draw all possible orientations
    for m in range(-l, l+1):
        Lz = m
        Lxy = np.sqrt(L_magnitude**2 - Lz**2)

        # Draw the cone base
        phi = np.linspace(0, 2*np.pi, 50)
        x = Lxy * np.cos(phi)
        y = Lxy * np.sin(phi)
        z = Lz * np.ones_like(phi)

        color = plt.cm.coolwarm((m + l) / (2*l))
        ax7.plot(x, y, z, color=color, linewidth=2, alpha=0.7)

        # One representative vector
        ax7.quiver(0, 0, 0, Lxy, 0, Lz, color=color,
                   arrow_length_ratio=0.05, linewidth=2)

    # z-axis
    ax7.quiver(0, 0, -3, 0, 0, 6, color='black', arrow_length_ratio=0.02,
               linewidth=1, alpha=0.3)
    ax7.text(0, 0, 3.2, 'z (B)', fontsize=10)

    ax7.set_xlabel('Lx')
    ax7.set_ylabel('Ly')
    ax7.set_zlabel('Lz')
    ax7.set_title('Space Quantization (l=2)')
    ax7.set_box_aspect([1,1,1])

    # --- 8. Electric Field Oscillation (σ⁺, σ⁻, π) ---
    ax8 = fig.add_subplot(gs[2, 0])

    t = np.linspace(0, 4*np.pi, 200)

    # σ⁺ and σ⁻ are circularly polarized
    # π is linearly polarized along z

    ax8.plot(t, np.cos(t), 'b-', linewidth=2, label='Ex (π)')
    ax8.plot(t, np.sin(t), 'r--', linewidth=2, label='Ey (σ)')

    ax8.set_xlabel('Time (ωt)')
    ax8.set_ylabel('Electric Field')
    ax8.set_title('Light Polarization Components')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # --- 9. Circular Polarization 3D ---
    ax9 = fig.add_subplot(gs[2, 1], projection='3d')

    t = np.linspace(0, 4*np.pi, 200)

    # σ⁺: left-hand circular
    x_p = np.cos(t)
    y_p = np.sin(t)
    z_p = t / (2*np.pi)
    ax9.plot(x_p, y_p, z_p, 'r-', linewidth=2, label='σ⁺')

    # σ⁻: right-hand circular
    x_m = np.cos(-t)
    y_m = np.sin(-t)
    z_m = t / (2*np.pi)
    ax9.plot(x_m, y_m, z_m, 'b-', linewidth=2, label='σ⁻')

    ax9.set_xlabel('Ex')
    ax9.set_ylabel('Ey')
    ax9.set_zlabel('Propagation (z)')
    ax9.set_title('Circular Polarization Helices')
    ax9.legend()

    # --- 10. Microwave Cavity Mode Pattern ---
    ax10 = fig.add_subplot(gs[2, 2])

    x = np.linspace(0, 2*np.pi, 100)
    y = np.linspace(0, np.pi, 50)
    X, Y = np.meshgrid(x, y)

    # TE₁₁ mode
    m, n = 1, 1
    E = np.sin(m * X) * np.sin(n * Y)

    im = ax10.contourf(X, Y, E, levels=20, cmap='RdBu')
    ax10.set_xlabel('x (cavity)')
    ax10.set_ylabel('y (cavity)')
    ax10.set_title('Microwave Cavity Mode TE₁₁')
    plt.colorbar(im, ax=ax10, label='E-field')

    # --- 11. Frequency Scaling ω_m ∝ m ---
    ax11 = fig.add_subplot(gs[2, 3])

    m_range = np.arange(-3, 4)
    B_fields = [0.5, 1.0, 2.0]

    mu_B = 9.274e-24  # Bohr magneton
    hbar = 1.055e-34

    for B in B_fields:
        omega_m = mu_B * B * np.abs(m_range) / hbar / 1e9  # GHz
        ax11.plot(m_range, omega_m, 'o-', linewidth=2, markersize=8,
                 label=f'B = {B} T')

    ax11.set_xlabel('Magnetic Quantum Number (m)')
    ax11.set_ylabel('Frequency (GHz)')
    ax11.set_title('Zeeman Frequency ω_m ∝ |m|·B')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.axhline(0, color='black', linewidth=0.5)
    ax11.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panel_zeeman_orientation_coordinate.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  [Zeeman Panel] Orientation coordinate (m) visualization saved")


# ============================================================================
# PANEL 4: CHIRALITY COORDINATE (s) - NMR / RADIO SPECTROSCOPY
# ============================================================================

def create_nmr_panel(save_dir: str):
    """
    NMR / Radio spectroscopy visualization panel for chirality coordinate (s).

    Includes:
    - Spin precession dynamics
    - NMR spectrum
    - Bloch sphere representation
    - Relaxation curves
    - Pulse sequences
    """
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('CHIRALITY COORDINATE (s) — NMR / Radio Spectroscopy',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Bloch Sphere (3D) ---
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    # Draw sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x, y, z, alpha=0.1, color='lightblue')
    ax1.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.3)

    # Axes
    ax1.quiver(0, 0, 0, 1.3, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 1.3, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 0, 1.3, color='blue', arrow_length_ratio=0.1, linewidth=2)

    ax1.text(1.4, 0, 0, 'x', fontsize=12, color='red')
    ax1.text(0, 1.4, 0, 'y', fontsize=12, color='green')
    ax1.text(0, 0, 1.4, 'z (|↑⟩)', fontsize=12, color='blue')
    ax1.text(0, 0, -1.4, '|↓⟩', fontsize=12, color='blue')

    # Magnetization vector
    theta = np.pi/4
    phi = np.pi/6
    M = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ax1.quiver(0, 0, 0, M[0], M[1], M[2], color='purple',
               arrow_length_ratio=0.1, linewidth=3, label='M')

    ax1.set_box_aspect([1,1,1])
    ax1.set_title('Bloch Sphere')
    ax1.axis('off')

    # --- 2. Spin-1/2 Energy Levels ---
    ax2 = fig.add_subplot(gs[0, 1])

    B_range = np.linspace(0, 2, 100)
    gamma = 2.675e8  # gyromagnetic ratio for ¹H

    E_up = 0.5 * B_range  # Normalized
    E_down = -0.5 * B_range

    ax2.fill_between(B_range, E_up, E_down, alpha=0.2, color='purple')
    ax2.plot(B_range, E_up, 'r-', linewidth=2, label='|↑⟩ (s = +½)')
    ax2.plot(B_range, E_down, 'b-', linewidth=2, label='|↓⟩ (s = -½)')

    # Mark transition
    B_mark = 1.0
    ax2.annotate('', xy=(B_mark, 0.5*B_mark), xytext=(B_mark, -0.5*B_mark),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.text(B_mark + 0.1, 0, 'ΔE = γℏB', fontsize=10, color='green')

    ax2.set_xlabel('Magnetic Field B (T)')
    ax2.set_ylabel('Energy (a.u.)')
    ax2.set_title('Spin Energy Levels')
    ax2.legend()
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # --- 3. Polar: Spin Population Distribution ---
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')

    # Temperature-dependent populations
    temperatures = [100, 200, 300, 500, 1000]
    theta = np.array([0, np.pi])  # Up and down

    for T in temperatures:
        # Boltzmann distribution
        kT = 8.617e-5 * T  # eV
        dE = 2.5e-7  # NMR splitting in eV (typical)

        p_up = np.exp(-dE/(2*kT)) / (np.exp(-dE/(2*kT)) + np.exp(dE/(2*kT)))
        p_down = 1 - p_up

        r = np.array([p_up, p_down]) * 1.5

        ax3.bar(theta, r, width=0.4, alpha=0.5, label=f'T={T}K')

    ax3.set_xticks([0, np.pi])
    ax3.set_xticklabels(['|↑⟩', '|↓⟩'])
    ax3.set_title('Spin Population (Boltzmann)', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

    # --- 4. NMR Spectrum ---
    ax4 = fig.add_subplot(gs[0, 3])

    # Chemical shifts (ppm)
    ppm = np.linspace(0, 12, 1000)

    # Typical ¹H NMR peaks
    peaks = [
        (0.9, 0.1, 1.0, 'CH₃'),
        (1.3, 0.1, 0.8, 'CH₂'),
        (2.1, 0.08, 0.6, 'CH₃CO'),
        (3.8, 0.1, 0.4, 'OCH₃'),
        (7.2, 0.15, 0.9, 'Ar-H'),
        (9.8, 0.1, 0.3, 'CHO'),
    ]

    spectrum = np.zeros_like(ppm)
    for center, width, intensity, label in peaks:
        peak = intensity * np.exp(-0.5 * ((ppm - center) / width)**2)
        spectrum += peak

    ax4.fill_between(ppm, spectrum, alpha=0.6, color='steelblue')
    ax4.plot(ppm, spectrum, 'k-', linewidth=1)

    for center, _, intensity, label in peaks:
        ax4.annotate(label, (center, intensity),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, rotation=45)

    ax4.invert_xaxis()
    ax4.set_xlabel('Chemical Shift (ppm)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('¹H NMR Spectrum')

    # --- 5. T1 and T2 Relaxation ---
    ax5 = fig.add_subplot(gs[1, 0])

    t = np.linspace(0, 5, 200)
    T1 = 1.0
    T2 = 0.5

    # T1 (longitudinal) recovery
    Mz = 1 - 2*np.exp(-t/T1)

    # T2 (transverse) decay
    Mxy = np.exp(-t/T2)

    ax5.plot(t, Mz, 'b-', linewidth=2, label=f'Mz (T₁={T1}s)')
    ax5.plot(t, Mxy, 'r-', linewidth=2, label=f'Mxy (T₂={T2}s)')
    ax5.fill_between(t, Mxy, alpha=0.2, color='red')

    ax5.axhline(1, color='blue', linestyle=':', alpha=0.5)
    ax5.axhline(0, color='gray', linestyle='-', alpha=0.3)

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Magnetization')
    ax5.set_title('NMR Relaxation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 5)
    ax5.set_ylim(-1.1, 1.1)

    # --- 6. FID Signal ---
    ax6 = fig.add_subplot(gs[1, 1])

    t = np.linspace(0, 0.5, 1000)
    T2_star = 0.1
    omega = 50  # Hz offset

    # FID with multiple frequencies
    fid = np.zeros_like(t)
    freqs = [0, 30, 80, 120]

    for f in freqs:
        fid += np.exp(-t/T2_star) * np.cos(2*np.pi*f*t)

    ax6.plot(t*1000, fid, 'b-', linewidth=0.8)
    ax6.fill_between(t*1000, fid, alpha=0.3, color='blue')

    # Envelope
    envelope = len(freqs) * np.exp(-t/T2_star)
    ax6.plot(t*1000, envelope, 'r--', linewidth=2, label='Envelope')
    ax6.plot(t*1000, -envelope, 'r--', linewidth=2)

    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Signal')
    ax6.set_title('Free Induction Decay (FID)')
    ax6.legend()

    # --- 7. Pulse Sequence Diagram ---
    ax7 = fig.add_subplot(gs[1, 2])

    # Time axis
    t_points = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # RF pulses
    rf_times = [0.5, 2.5, 4.5]
    rf_heights = [1, 0.5, 1]
    rf_widths = [0.2, 0.1, 0.2]

    for t, h, w in zip(rf_times, rf_heights, rf_widths):
        rect = plt.Rectangle((t-w/2, 0), w, h, color='blue', alpha=0.7)
        ax7.add_patch(rect)

    ax7.text(0.5, 1.1, '90°', ha='center', fontsize=10)
    ax7.text(2.5, 0.6, '180°', ha='center', fontsize=10)
    ax7.text(4.5, 1.1, '90°', ha='center', fontsize=10)

    # Gradient pulses
    for t, sign in [(1.5, 1), (3.5, -1)]:
        rect = plt.Rectangle((t-0.15, 0), 0.3, sign*0.5, color='green', alpha=0.7)
        ax7.add_patch(rect)

    # Acquisition
    ax7.plot([5.5, 7], [0.3, 0.3], 'r-', linewidth=3)
    ax7.text(6.25, 0.4, 'ACQ', ha='center', fontsize=10, color='red')

    ax7.axhline(0, color='black', linewidth=0.5)
    ax7.set_xlim(0, 8)
    ax7.set_ylim(-0.7, 1.5)
    ax7.set_xlabel('Time')
    ax7.set_title('Spin Echo Pulse Sequence')

    # Labels
    ax7.text(-0.5, 0.75, 'RF', ha='center', fontsize=10, color='blue', rotation=90)
    ax7.text(-0.5, 0, 'Gx', ha='center', fontsize=10, color='green', rotation=90)

    ax7.set_yticks([])

    # --- 8. Radar Chart: Relaxation Properties ---
    ax8 = fig.add_subplot(gs[1, 3], projection='polar')

    categories = ['T₁', 'T₂', 'T₂*', 'Chemical\nShift',
                  'J-coupling', 'NOE']
    N = len(categories)

    # Different tissue types
    tissues = {
        'Water': [4, 3, 2, 1, 1, 1],
        'Fat': [2, 1, 0.5, 3, 2, 3],
        'Brain': [1, 0.5, 0.3, 2, 1.5, 2],
    }

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ['blue', 'yellow', 'pink']
    for i, (name, values) in enumerate(tissues.items()):
        values = values + values[:1]
        ax8.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax8.fill(angles, values, alpha=0.1, color=colors[i])

    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, size=8)
    ax8.set_title('NMR Properties by Tissue', pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # --- 9. 2D COSY-style Correlation ---
    ax9 = fig.add_subplot(gs[2, 0])

    # Create 2D correlation pattern
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Diagonal peaks
    Z = np.zeros_like(X)
    peaks = [(2, 2), (4, 4), (6, 6), (8, 8)]
    for px, py in peaks:
        Z += np.exp(-((X-px)**2 + (Y-py)**2)/0.2)

    # Cross peaks (coupling)
    cross_peaks = [(2, 4), (4, 2), (6, 8), (8, 6)]
    for px, py in cross_peaks:
        Z += 0.5 * np.exp(-((X-px)**2 + (Y-py)**2)/0.15)

    ax9.contourf(X, Y, Z, levels=15, cmap='Blues')
    ax9.contour(X, Y, Z, levels=5, colors='black', linewidths=0.5)

    ax9.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Diagonal')
    ax9.set_xlabel('F2 (ppm)')
    ax9.set_ylabel('F1 (ppm)')
    ax9.set_title('2D COSY Correlation Map')
    ax9.invert_xaxis()
    ax9.invert_yaxis()

    # --- 10. Spin-Spin Coupling (J-coupling) ---
    ax10 = fig.add_subplot(gs[2, 1])

    # Multiplet patterns
    ppm = np.linspace(0, 3, 500)

    # Singlet
    singlet = 0.4 * np.exp(-((ppm - 0.5)**2) / 0.002)

    # Doublet (J = 7 Hz ~ 0.02 ppm)
    J = 0.05
    doublet = 0.35 * (np.exp(-((ppm - 1.2 - J/2)**2) / 0.001) +
                      np.exp(-((ppm - 1.2 + J/2)**2) / 0.001))

    # Triplet
    triplet = 0.3 * (0.5 * np.exp(-((ppm - 2 - J)**2) / 0.001) +
                     1.0 * np.exp(-((ppm - 2)**2) / 0.001) +
                     0.5 * np.exp(-((ppm - 2 + J)**2) / 0.001))

    # Quartet
    quartet = 0.25 * (0.25 * np.exp(-((ppm - 2.7 - 1.5*J)**2) / 0.0008) +
                       0.75 * np.exp(-((ppm - 2.7 - 0.5*J)**2) / 0.0008) +
                       0.75 * np.exp(-((ppm - 2.7 + 0.5*J)**2) / 0.0008) +
                       0.25 * np.exp(-((ppm - 2.7 + 1.5*J)**2) / 0.0008))

    ax10.fill_between(ppm, singlet, alpha=0.6, label='Singlet')
    ax10.fill_between(ppm, doublet, alpha=0.6, label='Doublet')
    ax10.fill_between(ppm, triplet, alpha=0.6, label='Triplet')
    ax10.fill_between(ppm, quartet, alpha=0.6, label='Quartet')

    ax10.set_xlabel('Chemical Shift (ppm)')
    ax10.set_ylabel('Intensity')
    ax10.set_title('J-Coupling Multiplet Patterns')
    ax10.legend()
    ax10.invert_xaxis()

    # --- 11. Larmor Frequency vs Field ---
    ax11 = fig.add_subplot(gs[2, 2])

    B = np.linspace(0, 20, 100)

    # Different nuclei
    nuclei = {
        '¹H': 42.577,
        '¹³C': 10.705,
        '¹⁹F': 40.052,
        '³¹P': 17.235,
    }

    for name, gamma in nuclei.items():
        omega = gamma * B  # MHz
        ax11.plot(B, omega, linewidth=2, label=name)

    ax11.fill_between(B, 42.577 * B, alpha=0.1, color='blue')

    ax11.set_xlabel('Magnetic Field (T)')
    ax11.set_ylabel('Larmor Frequency (MHz)')
    ax11.set_title('ω = γB (Different Nuclei)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # Mark common field strengths
    for B_mark in [1.5, 3.0, 7.0, 11.7]:
        ax11.axvline(B_mark, color='gray', linestyle=':', alpha=0.5)

    # --- 12. Spin State Diagram ---
    ax12 = fig.add_subplot(gs[2, 3])

    # Two-spin system
    states = ['|↑↑⟩', '|↑↓⟩', '|↓↑⟩', '|↓↓⟩']
    energies = [1, 0.1, -0.1, -1]  # Simplified

    x_pos = [0.3, 0.5, 0.5, 0.7]

    for state, E, x in zip(states, energies, x_pos):
        ax12.hlines(E, x-0.1, x+0.1, colors='black', linewidth=2)
        ax12.text(x, E+0.15, state, ha='center', fontsize=10)

    # Transitions
    ax12.annotate('', xy=(0.5, 0.1), xytext=(0.3, 1),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax12.annotate('', xy=(0.5, -0.1), xytext=(0.3, 1),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax12.annotate('', xy=(0.7, -1), xytext=(0.5, 0.1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax12.annotate('', xy=(0.7, -1), xytext=(0.5, -0.1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax12.set_xlim(0, 1)
    ax12.set_ylim(-1.5, 1.5)
    ax12.set_ylabel('Energy')
    ax12.set_title('Two-Spin Energy Levels')
    ax12.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panel_nmr_chirality_coordinate.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  [NMR Panel] Chirality coordinate (s) visualization saved")


# ============================================================================
# PANEL 5: UNIFIED INSTRUMENT OVERVIEW
# ============================================================================

def create_unified_panel(save_dir: str):
    """
    Unified overview showing all four instrument types and their relationships.
    """
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle('UNIFIED SPECTROSCOPY — Four Partition Coordinates',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.35)

    # --- Row 1: Frequency Hierarchy ---
    ax_freq = fig.add_subplot(gs[0, :])

    freq_regimes = [
        ('s (Chirality)', 1e6, 1e8, 'NMR / Radio', 'red'),
        ('m (Orientation)', 1e10, 1e11, 'Zeeman / Microwave', 'orange'),
        ('l (Complexity)', 1e13, 1e15, 'UV-Vis / Optical', 'green'),
        ('n (Depth)', 1e16, 1e18, 'XPS / X-ray', 'blue'),
    ]

    for i, (label, f_min, f_max, instrument, color) in enumerate(freq_regimes):
        ax_freq.barh(i, np.log10(f_max) - np.log10(f_min),
                     left=np.log10(f_min), height=0.6,
                     color=color, alpha=0.7, edgecolor='black')
        ax_freq.text(np.log10(f_min) + 0.5, i, f'{label}\n{instrument}',
                    va='center', fontsize=10, fontweight='bold')

    ax_freq.set_xlabel('log₁₀(Frequency / Hz)')
    ax_freq.set_yticks(range(4))
    ax_freq.set_yticklabels(['Radio', 'Microwave', 'Optical', 'X-ray'])
    ax_freq.set_title('Frequency Regime Separation', fontsize=14)
    ax_freq.set_xlim(5, 19)
    ax_freq.grid(True, alpha=0.3, axis='x')

    # --- Row 2-4: Individual Instrument Summaries ---

    # Depth (n) - XPS
    ax_n1 = fig.add_subplot(gs[1, 0])
    n_vals = range(1, 8)
    capacities = [2*n**2 for n in n_vals]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(n_vals)))
    ax_n1.bar(n_vals, capacities, color=colors, edgecolor='black')
    ax_n1.set_xlabel('n')
    ax_n1.set_ylabel('Capacity (2n²)')
    ax_n1.set_title('Depth (n) — Shell Capacity')

    ax_n2 = fig.add_subplot(gs[1, 1], projection='3d')
    phi = np.linspace(0, 2*np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    PHI, THETA = np.meshgrid(phi, theta)
    for n in [1, 2, 3]:
        r = n * 1.5
        X = r * np.sin(THETA) * np.cos(PHI)
        Y = r * np.sin(THETA) * np.sin(PHI)
        Z = r * np.cos(THETA)
        ax_n2.plot_surface(X, Y, Z, alpha=0.2/n, color=plt.cm.Blues(n/4))
    ax_n2.set_title('Electron Shells')
    ax_n2.axis('off')

    # Complexity (l) - UV-Vis
    ax_l1 = fig.add_subplot(gs[2, 0], projection='polar')
    l_vals = range(5)
    theta = np.linspace(0, 2*np.pi, len(l_vals), endpoint=False)
    degeneracies = [2*l+1 for l in l_vals]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(l_vals)))
    ax_l1.bar(theta, degeneracies, width=1.0, color=colors, alpha=0.7)
    ax_l1.set_title('Complexity (l) — Degeneracy')

    ax_l2 = fig.add_subplot(gs[2, 1], projection='3d')
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)
    Y = sph_harm(0, 2, PHI, THETA)
    R = np.abs(Y.real) * 0.8 + 0.2
    X = R * np.sin(THETA) * np.cos(PHI)
    Y_c = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    ax_l2.plot_surface(X, Y_c, Z, cmap='RdYlGn', alpha=0.8)
    ax_l2.set_title('d-Orbital Shape')
    ax_l2.axis('off')

    # Orientation (m) - Zeeman
    ax_m1 = fig.add_subplot(gs[1, 2])
    m_vals = range(-3, 4)
    energies = [m * 0.5 for m in m_vals]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(m_vals)))
    for m, E, c in zip(m_vals, energies, colors):
        ax_m1.barh(m, 1, left=0, height=0.6, color=c, alpha=0.7)
        ax_m1.text(1.1, m, f'm={m:+d}', va='center')
    ax_m1.set_xlabel('E / μ_B·B')
    ax_m1.set_title('Orientation (m) — Zeeman Levels')

    ax_m2 = fig.add_subplot(gs[1, 3], projection='3d')
    t = np.linspace(0, 4*np.pi, 100)
    for m in [1, 2]:
        theta_L = np.arccos(m / np.sqrt(6))
        x = np.sin(theta_L) * np.cos(t)
        y = np.sin(theta_L) * np.sin(t)
        z = np.cos(theta_L) * np.ones_like(t)
        ax_m2.plot(x, y, z, linewidth=2)
    ax_m2.quiver(0, 0, 0, 0, 0, 1.2, color='blue', linewidth=2)
    ax_m2.set_title('Larmor Precession')
    ax_m2.axis('off')

    # Chirality (s) - NMR
    ax_s1 = fig.add_subplot(gs[2, 2])
    t = np.linspace(0, 1, 100)
    Mz = 1 - 2*np.exp(-t/0.3)
    Mxy = np.exp(-t/0.1)
    ax_s1.plot(t, Mz, 'b-', linewidth=2, label='T₁')
    ax_s1.plot(t, Mxy, 'r-', linewidth=2, label='T₂')
    ax_s1.set_xlabel('Time (s)')
    ax_s1.set_ylabel('M')
    ax_s1.set_title('Chirality (s) — Relaxation')
    ax_s1.legend()

    ax_s2 = fig.add_subplot(gs[2, 3], projection='3d')
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax_s2.plot_surface(x, y, z, alpha=0.15, color='lightblue')
    ax_s2.quiver(0, 0, 0, 0.5, 0.5, 0.7, color='red', linewidth=3)
    ax_s2.set_title('Bloch Sphere')
    ax_s2.axis('off')

    # --- Relationship Diagram ---
    ax_rel = fig.add_subplot(gs[1:3, 4])

    # Draw connection diagram
    positions = {
        'n': (0.5, 0.85),
        'l': (0.85, 0.5),
        'm': (0.5, 0.15),
        's': (0.15, 0.5),
    }

    colors = {'n': 'blue', 'l': 'green', 'm': 'orange', 's': 'red'}

    for coord, (x, y) in positions.items():
        circle = Circle((x, y), 0.12, color=colors[coord], alpha=0.7)
        ax_rel.add_patch(circle)
        ax_rel.text(x, y, coord, ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')

    # Connect with lines (selection rules)
    connections = [('n', 'l'), ('l', 'm'), ('m', 's')]
    for c1, c2 in connections:
        x1, y1 = positions[c1]
        x2, y2 = positions[c2]
        ax_rel.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5)

    # Labels
    labels = {
        'n': 'Core\n(XPS)',
        'l': 'Valence\n(UV-Vis)',
        'm': 'Orientation\n(Zeeman)',
        's': 'Spin\n(NMR)',
    }

    offsets = {'n': (0, 0.2), 'l': (0.2, 0), 'm': (0, -0.2), 's': (-0.2, 0)}

    for coord, (ox, oy) in offsets.items():
        x, y = positions[coord]
        ax_rel.text(x + ox, y + oy, labels[coord], ha='center', va='center',
                   fontsize=9)

    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_title('Coordinate Relationships')
    ax_rel.axis('off')
    ax_rel.set_aspect('equal')

    # --- Bottom: Summary Table ---
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')

    table_data = [
        ['Coordinate', 'Symbol', 'Frequency', 'Instrument', 'Physical Coupling'],
        ['Depth', 'n', 'ω_n ∝ n⁻³', 'XPS / X-ray', 'Core electron binding'],
        ['Complexity', 'l', 'ω_l ∝ l(l+1)', 'UV-Vis / Optical', 'Angular momentum'],
        ['Orientation', 'm', 'ω_m ∝ m·B', 'Zeeman / Microwave', 'Magnetic dipole'],
        ['Chirality', 's', 'ω_s ∝ s·B', 'NMR / Radio', 'Spin angular momentum'],
    ]

    table = ax_table.table(cellText=table_data, loc='center',
                           cellLoc='center',
                           colWidths=[0.15, 0.1, 0.2, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Color header row
    for j in range(5):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'panel_unified_spectroscopy.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  [Unified Panel] All instruments overview saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all instrument visualization panels."""
    print("=" * 70)
    print("INSTRUMENT VISUALIZATION SUITE")
    print("Generating spectroscopic instrument panels...")
    print("=" * 70)

    # Setup output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results', 'instrument_panels')
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nOutput directory: {results_dir}\n")

    # Generate panels
    print("[1/5] Creating XPS panel (Depth coordinate n)...")
    create_xps_panel(results_dir)

    print("[2/5] Creating UV-Vis panel (Complexity coordinate l)...")
    create_uvvis_panel(results_dir)

    print("[3/5] Creating Zeeman panel (Orientation coordinate m)...")
    create_zeeman_panel(results_dir)

    print("[4/5] Creating NMR panel (Chirality coordinate s)...")
    create_nmr_panel(results_dir)

    print("[5/5] Creating unified overview panel...")
    create_unified_panel(results_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated panels in: {results_dir}")
    print("  - panel_xps_depth_coordinate.png")
    print("  - panel_uvvis_complexity_coordinate.png")
    print("  - panel_zeeman_orientation_coordinate.png")
    print("  - panel_nmr_chirality_coordinate.png")
    print("  - panel_unified_spectroscopy.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
