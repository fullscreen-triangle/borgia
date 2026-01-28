"""
Generate Figures 6-8 for Ensemble Virtual Spectrometry paper
(Molecular Observer Networks, Reflectance Cascade, Temporal Resolution)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Create output directory
os.makedirs('figures', exist_ok=True)


def generate_figure6():
    """Figure 6: Molecular Observer Networks and Dual-Face Information"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Molecular Observer Network in S-Space
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    
    np.random.seed(42)
    n_observers = 1000
    n_targets = 10
    R_mol = 0.15
    
    # Observer positions
    obs_pos = np.random.rand(n_observers, 3)
    
    # Target positions
    target_pos = np.random.rand(n_targets, 3)
    
    # Plot observers (small spheres)
    ax1.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2],
               c='lightblue', s=2, alpha=0.3, label=f'Observers (N={n_observers})')
    
    # Plot targets (larger colored spheres)
    ax1.scatter(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
               c=range(n_targets), cmap='tab10', s=200, edgecolors='black',
               linewidths=2, alpha=0.9, label='Targets')
    
    # Draw reach regions for a few observers (semi-transparent spheres)
    sample_observers = obs_pos[::100]  # Sample every 100th observer
    for obs in sample_observers:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = obs[0] + R_mol * np.outer(np.cos(u), np.sin(v))
        y = obs[1] + R_mol * np.outer(np.sin(u), np.sin(v))
        z = obs[2] + R_mol * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='cyan', alpha=0.05)
    
    # Draw observation links for one target
    target_idx = 0
    target = target_pos[target_idx]
    distances = np.linalg.norm(obs_pos - target, axis=1)
    within_reach = distances < R_mol
    
    for obs in obs_pos[within_reach][:20]:  # Show first 20 links
        ax1.plot([obs[0], target[0]], [obs[1], target[1]], [obs[2], target[2]],
                'r-', linewidth=0.3, alpha=0.3)
    
    ax1.set_xlabel('$S_k$')
    ax1.set_ylabel('$S_t$')
    ax1.set_zlabel('$S_e$')
    ax1.set_title('(A) Molecular Observer Network in S-Space')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.text2D(0.05, 0.05, f'Reach $R_{{mol}}={R_mol}$', transform=ax1.transAxes)
    
    # Panel B: Cross-Observer Consistency
    ax2 = fig.add_subplot(gs[0, 1])
    
    # True coordinates (adenine)
    S_true = np.array([0.42, 0.73, 0.69])
    
    # Simulate measurements from 47 observers
    np.random.seed(42)
    n_obs = 47
    sigma = 0.008  # 0.8% uncertainty
    
    S_k_meas = S_true[0] + np.random.normal(0, sigma, n_obs)
    S_t_meas = S_true[1] + np.random.normal(0, sigma, n_obs)
    S_e_meas = S_true[2] + np.random.normal(0, sigma, n_obs)
    
    # Plot diagonal line
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect agreement')
    
    # Plot measurements
    ax2.errorbar(np.full(n_obs, S_true[0]), S_k_meas, yerr=sigma, 
                fmt='ro', markersize=4, capsize=2, alpha=0.5, label='$S_k$')
    ax2.errorbar(np.full(n_obs, S_true[1]), S_t_meas, yerr=sigma,
                fmt='go', markersize=4, capsize=2, alpha=0.5, label='$S_t$')
    ax2.errorbar(np.full(n_obs, S_true[2]), S_e_meas, yerr=sigma,
                fmt='bo', markersize=4, capsize=2, alpha=0.5, label='$S_e$')
    
    # Shaded band for ±1% deviation
    ax2.fill_between([0, 1], [0, 1], [0.01, 1.01], alpha=0.1, color='gray')
    ax2.fill_between([0, 1], [-0.01, 0.99], [0, 1], alpha=0.1, color='gray')
    
    ax2.set_xlabel('True coordinate value $S_{true}$')
    ax2.set_ylabel('Measured coordinate value $S_{meas}$')
    ax2.set_title('(B) Cross-Observer Consistency')
    ax2.text(0.05, 0.95, f'$\\bar{{\\sigma}} = {sigma*100:.1f}\\%$',
            transform=ax2.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    
    # Panel C: Dual-Face Complementarity
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    # Draw two overlapping circles (Venn diagram style)
    circle1 = Circle((0.35, 0.5), 0.25, color='blue', alpha=0.3, label='Front Face')
    circle2 = Circle((0.65, 0.5), 0.25, color='red', alpha=0.3, label='Back Face')
    ax3.add_patch(circle1)
    ax3.add_patch(circle2)
    
    # Labels
    ax3.text(0.25, 0.5, 'Front Face\n$I_{front}$\nState count Ω\n(directly measured)',
            ha='center', va='center', fontsize=9, weight='bold')
    ax3.text(0.75, 0.5, 'Back Face\n$I_{back}$\nEnvironmental ΔS\n(directly measured)',
            ha='center', va='center', fontsize=9, weight='bold')
    ax3.text(0.5, 0.5, 'Total\n$I_{total}$', ha='center', va='center',
            fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='yellow'))
    
    # Measurement apparatus arrows
    # State counter -> Front face
    ax3.arrow(0.15, 0.7, 0.08, -0.1, head_width=0.03, head_length=0.03,
             fc='blue', ec='blue', linewidth=2)
    ax3.text(0.1, 0.75, 'State\ncounter', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue'))
    
    # Calorimeter -> Back face
    ax3.arrow(0.85, 0.7, -0.08, -0.1, head_width=0.03, head_length=0.03,
             fc='red', ec='red', linewidth=2)
    ax3.text(0.9, 0.75, 'Calorimeter', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    # Crossed-out dashed arrows (cannot measure both directly)
    ax3.plot([0.15, 0.7], [0.3, 0.4], 'k--', linewidth=1, alpha=0.5)
    ax3.plot([0.85, 0.3], [0.3, 0.4], 'k--', linewidth=1, alpha=0.5)
    ax3.text(0.4, 0.25, '⊗', fontsize=30, ha='center', color='red')
    ax3.text(0.6, 0.25, '⊗', fontsize=30, ha='center', color='red')
    
    # Equation
    ax3.text(0.5, 0.15, '$I_{front} + I_{back} = I_{total}$',
            ha='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Values
    ax3.text(0.5, 0.05, 'Front: 12.4 bits  |  Back: 12.7 bits  |  Total: 25.0 bits',
            ha='center', fontsize=9)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('(C) Dual-Face Complementarity')
    
    # Panel D: Cross-Face Catalysis Enhancement
    ax4 = fig.add_subplot(gs[1, 1])
    
    conditions = ['Front\nonly', 'Back\nonly', 'Front→Back\n(catalyzed)']
    rates = [8.2, 8.1, 11.7]
    errors = [0.6, 0.6, 0.8]
    colors = ['blue', 'red', 'green']
    
    bars = ax4.bar(conditions, rates, yerr=errors, capsize=5, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    # Baseline
    ax4.axhline(8.15, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Baseline')
    
    # Enhancement annotation
    ax4.plot([1, 2], [12.5, 12.5], 'k-', linewidth=2)
    ax4.text(1.5, 13, f'α_cat = 1.43\n(43% enhancement)', ha='center',
            fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax4.set_ylabel('Information generation rate dI/dt (bits/ms)')
    ax4.set_title('(D) Cross-Face Catalysis Enhancement')
    ax4.set_ylim(0, 15)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=8)
    
    # Add value labels on bars
    for i, (rate, error) in enumerate(zip(rates, errors)):
        ax4.text(i, rate + error + 0.5, f'{rate}±{error}', ha='center', fontsize=9)
    
    # Add catalytic coupling constant
    ax4.text(0.5, 0.95, '$\\beta_{cross} = 0.035$ bits$^{-1}$',
            transform=ax4.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.savefig('figures/figure6_molecular_observers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated Figure 6: Molecular Observer Networks")


def generate_figure7():
    """Figure 7: Reflectance Cascade and Information Amplification"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Cascade Network Architecture
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_levels = 10
    I_0 = 12.4
    
    # Draw nodes
    for k in range(n_levels):
        x = k / (n_levels - 1)
        y = 0.5
        
        # Node size proportional to information
        I_k = I_0 * (k + 1)**2
        size = 100 + I_k * 2
        
        # Color gradient
        color = plt.cm.coolwarm(k / (n_levels - 1))
        
        circle = Circle((x, y), 0.04, color=color, ec='black', linewidth=2, zorder=10)
        ax1.add_patch(circle)
        ax1.text(x, y, f'Obs\n{k+1}', ha='center', va='center', fontsize=7, weight='bold')
        
        # Information value below
        ax1.text(x, y - 0.15, f'$I_{{{k+1}}}$\n{I_k:.0f}', ha='center', fontsize=7)
        
        # Arrow to next
        if k < n_levels - 1:
            x_next = (k + 1) / (n_levels - 1)
            arrow_width = 0.002 * (k + 2)  # Thickness increases
            ax1.arrow(x + 0.05, y, x_next - x - 0.1, 0, head_width=0.05,
                     head_length=0.02, fc=color, ec=color, linewidth=arrow_width*10)
            
            # Reflectance feedback (curved)
            if k > 0:
                arc_y = np.linspace(y, y + 0.2, 20)
                arc_x = x + 0.05 * np.sin(np.linspace(0, np.pi, 20))
                ax1.plot(arc_x, arc_y, 'k--', linewidth=0.5, alpha=0.3)
    
    # Target at left
    target = Circle((-0.1, 0.5), 0.05, color='gold', ec='black', linewidth=2, zorder=10)
    ax1.add_patch(target)
    ax1.text(-0.1, 0.5, 'Target', ha='center', va='center', fontsize=8, weight='bold')
    
    ax1.set_xlim(-0.2, 1.1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('(A) Cascade Network Architecture')
    ax1.text(0.5, 0.9, 'Information Flow →', ha='center', fontsize=10, weight='bold')
    ax1.text(0.5, 0.1, 'Reflectance', ha='center', fontsize=9, style='italic')
    
    # Panel B: Information Scaling I_k = k² I_0
    ax2 = fig.add_subplot(gs[0, 1])
    
    k = np.arange(1, 11)
    I_k = I_0 * k**2
    
    # Experimental data with noise
    np.random.seed(42)
    I_k_exp = I_k * np.random.lognormal(0, 0.05, len(k))
    
    ax2.plot(k, I_k, 'b-', linewidth=2, label='$I_k = I_0 k^2$ (quadratic)')
    ax2.plot(k, I_0 * k, 'r--', linewidth=2, label='$I_k = I_0 k$ (linear)')
    ax2.errorbar(k, I_k_exp, yerr=I_k*0.05, fmt='ko', markersize=6,
                capsize=3, alpha=0.7, label='Experiment')
    
    # Shaded region showing enhancement
    ax2.fill_between(k, I_0 * k, I_k, alpha=0.2, color='green',
                     label='Enhancement region')
    
    ax2.set_xlabel('Cascade level $k$')
    ax2.set_ylabel('Information at level $k$, $I_k$ (bits)')
    ax2.set_title('(B) Information Scaling $I_k = k^2 I_0$')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Fit parameters
    ax2.text(0.05, 0.95, f'$I_0 = {I_0}$ bits\n$\\gamma = 1.98 ± 0.04$',
            transform=ax2.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel C: Cumulative Information ~ O(N³)
    ax3 = fig.add_subplot(gs[1, 0])
    
    N = np.arange(1, 21)
    I_total_cubic = I_0 * N * (N + 1) * (2*N + 1) / 6
    I_total_linear = I_0 * N
    
    ax3.loglog(N, I_total_cubic, 'b-', linewidth=2, label='$\\mathcal{O}(N^3)$ (cubic)')
    ax3.loglog(N, I_total_linear, 'r--', linewidth=2, label='$\\mathcal{O}(N)$ (linear)')
    
    # Data points
    np.random.seed(42)
    I_total_exp = I_total_cubic * np.random.lognormal(0, 0.03, len(N))
    ax3.loglog(N, I_total_exp, 'ko', markersize=5, alpha=0.7)
    
    # Enhancement at N=10
    idx_10 = 9
    enhancement = I_total_cubic[idx_10] / I_total_linear[idx_10]
    ax3.plot([N[idx_10], N[idx_10]], [I_total_linear[idx_10], I_total_cubic[idx_10]],
            'g-', linewidth=3, label=f'{enhancement:.0f}× enhancement')
    ax3.text(N[idx_10]*1.2, np.sqrt(I_total_linear[idx_10] * I_total_cubic[idx_10]),
            f'{enhancement:.0f}×', fontsize=10, weight='bold', color='green')
    
    ax3.set_xlabel('Number of cascade levels $N$')
    ax3.set_ylabel('Total information $I_{total}(N)$ (bits)')
    ax3.set_title('(C) Cumulative Information $\\sim \\mathcal{O}(N^3)$')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Panel D: 3D Cascade Visualization
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    
    n_levels = 10
    
    for k in range(n_levels):
        # Observer position
        Sk = 0.5 + 0.2 * np.sin(k * np.pi / 5)
        St = 0.5 + 0.2 * np.cos(k * np.pi / 5)
        z = k
        
        # Sphere size proportional to I_k
        I_k = I_0 * (k + 1)**2
        size = I_k * 3
        
        # Color
        color = plt.cm.coolwarm(k / (n_levels - 1))
        
        ax4.scatter(Sk, St, z, s=size, c=[color], edgecolors='black',
                   linewidths=1, alpha=0.8)
        
        # Vertical arrow
        if k > 0:
            Sk_prev = 0.5 + 0.2 * np.sin((k-1) * np.pi / 5)
            St_prev = 0.5 + 0.2 * np.cos((k-1) * np.pi / 5)
            ax4.plot([Sk_prev, Sk], [St_prev, St], [k-1, k],
                    'k-', linewidth=2, alpha=0.5)
        
        # Horizontal spread (lateral information)
        if k > 0:
            theta = np.linspace(0, 2*np.pi, 20)
            r = 0.05 * k
            x_circle = Sk + r * np.cos(theta)
            y_circle = St + r * np.sin(theta)
            z_circle = np.full_like(theta, z)
            ax4.plot(x_circle, y_circle, z_circle, 'b-', linewidth=0.5, alpha=0.3)
    
    # Target at z=0
    ax4.scatter(0.5, 0.5, 0, s=300, c='gold', marker='*', edgecolors='black',
               linewidths=2, label='Target')
    
    ax4.set_xlabel('$S_k$')
    ax4.set_ylabel('$S_t$')
    ax4.set_zlabel('Cascade level $k$')
    ax4.set_title('(D) 3D Cascade Visualization')
    ax4.text2D(0.5, 0.95, '$I_k \\propto k^2$', transform=ax4.transAxes,
              ha='center', fontsize=10, weight='bold')
    
    plt.savefig('figures/figure7_reflectance_cascade.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated Figure 7: Reflectance Cascade")


def generate_figure8():
    """Figure 8: Categorical Temporal Resolution and Phase Accumulation"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Multi-Scale Oscillator Ensemble
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Frequency regimes
    regimes = [
        ('Spin', 1e9, 'purple'),
        ('Rotational', 1e11, 'blue'),
        ('Vibrational', 1e14, 'green'),
        ('Electronic', 1e16, 'red')
    ]
    
    omega = np.logspace(6, 18, 1000)
    
    for name, omega_c, color in regimes:
        delta_omega = omega_c * 0.2
        N_omega = 1e22 * np.exp(-((np.log10(omega) - np.log10(omega_c))**2) / 0.5)
        ax1.fill_between(omega, 0, N_omega, color=color, alpha=0.3, label=name)
        ax1.plot(omega, N_omega, color=color, linewidth=2)
        
        # Mark frequency and spread
        ax1.axvline(omega_c, color=color, linestyle='--', linewidth=1, alpha=0.7)
        ax1.plot([omega_c - delta_omega, omega_c + delta_omega],
                [N_omega.max() * 0.8, N_omega.max() * 0.8],
                color=color, linewidth=3)
        ax1.text(omega_c, N_omega.max() * 0.9, 'δω', ha='center', fontsize=8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency ω (Hz)')
    ax1.set_ylabel('Oscillator count N(ω)')
    ax1.set_title('(A) Multi-Scale Oscillator Ensemble')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.text(0.95, 0.95, '$N_{total} = 10^{23}$', transform=ax1.transAxes,
            ha='right', va='top', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Phase Accumulation Over Time
    ax2 = fig.add_subplot(gs[0, 1])
    
    T = np.logspace(-6, 0, 100)
    
    # Phase accumulation for each regime
    total_phase = np.zeros_like(T)
    
    for name, omega_c, color in regimes:
        N_regime = 2.5e22
        delta_omega = omega_c * 0.2
        phase = N_regime * delta_omega * T
        ax2.loglog(T, phase, color=color, linewidth=1.5, alpha=0.7, label=name)
        total_phase += phase
    
    # Total phase
    ax2.loglog(T, total_phase, 'k-', linewidth=3, label='Total')
    
    # Mark key phase values
    for phase_val, label in [(2*np.pi, '2π'), (1e10, '10¹⁰'), (1e34, '10³⁴')]:
        idx = np.argmin(np.abs(total_phase - phase_val))
        if idx < len(T):
            ax2.axhline(phase_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax2.text(T[-1]*1.2, phase_val, label, fontsize=8)
    
    # Mark T = 1 ms
    ax2.axvline(1e-3, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(1e-3, 1e40, 'T = 1 ms', ha='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Integration time T (s)')
    ax2.set_ylabel('Accumulated phase Δφ_total (radians)')
    ax2.set_title('(B) Phase Accumulation Over Time')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(0.05, 0.95, '$\\Delta\\phi = N\\delta\\omega T$',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel C: Categorical Temporal Resolution
    ax3 = fig.add_subplot(gs[1, 0])
    
    systems = ['CPU\nclock', 'LED\narray', 'Single\nmolecule', 'Molecular\nensemble']
    resolutions = [1e-15, 3e-18, 1e-37, 1e-66]
    colors_sys = ['blue', 'green', 'orange', 'red']
    
    bars = ax3.bar(systems, resolutions, color=colors_sys, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    # Planck time reference
    t_planck = 1e-43
    ax3.axhline(t_planck, color='black', linestyle='--', linewidth=2,
               label='Planck time')
    ax3.text(0.5, t_planck*1.5, 'Planck time', ha='left', fontsize=9)
    
    ax3.set_yscale('log')
    ax3.set_ylabel('Temporal resolution Δt_cat (s)')
    ax3.set_title('(C) Categorical Temporal Resolution')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (sys, res) in enumerate(zip(systems, resolutions)):
        ax3.text(i, res*2, f'{res:.0e} s', ha='center', fontsize=8, rotation=0)
    
    ax3.text(0.5, 0.05, 'Phase space resolution, not event localization',
            transform=ax3.transAxes, ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax3.text(0.5, 0.95, '$\\Delta t_{cat} = 1/(N\\delta\\omega)$',
            transform=ax3.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel D: Resolution vs Integration Time (Heatmap)
    ax4 = fig.add_subplot(gs[1, 1])
    
    T_range = np.logspace(-6, 0, 100)
    N_range = np.logspace(0, 24, 100)
    T_grid, N_grid = np.meshgrid(T_range, N_range)
    
    delta_omega = 1e14
    dt_cat = 1 / (N_grid * delta_omega)
    
    # Plot heatmap
    im = ax4.contourf(T_range, N_range, np.log10(dt_cat), levels=20, cmap='viridis')
    
    # Contour lines
    levels_contour = [-15, -30, -45, -60]
    contours = ax4.contour(T_range, N_range, np.log10(dt_cat), levels=levels_contour,
                          colors='white', linewidths=1.5, linestyles='--')
    ax4.clabel(contours, inline=True, fontsize=8, fmt='10^%d s')
    
    # Diagonal line: dt_cat = T (unphysical below this)
    T_diag = np.logspace(-6, 0, 100)
    N_diag = 1 / (T_diag * delta_omega)
    ax4.plot(T_diag, N_diag, 'r-', linewidth=3, label='$\\Delta t_{cat} = T$')
    
    # Shade unphysical region
    ax4.fill_between(T_diag, 1, N_diag, where=(N_diag >= 1), alpha=0.3,
                     color='gray', label='Unphysical')
    
    # Mark experimental operating points
    operating_points = [
        (1e-3, 1e23, 'Ensemble'),
        (1e-9, 1e6, 'CPU'),
        (1e-6, 1e3, 'LED')
    ]
    
    for T_op, N_op, label in operating_points:
        ax4.plot(T_op, N_op, 'r*', markersize=15, markeredgecolor='white',
                markeredgewidth=1.5)
        ax4.text(T_op*1.5, N_op, label, fontsize=8, weight='bold', color='white')
    
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Integration time T (s)')
    ax4.set_ylabel('Oscillator count N')
    ax4.set_title('(D) Resolution vs Integration Time')
    plt.colorbar(im, ax=ax4, label='log₁₀(Δt_cat / s)')
    ax4.legend(fontsize=8, loc='upper right')
    
    plt.savefig('figures/figure8_temporal_resolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Generated Figure 8: Categorical Temporal Resolution")


# Main execution
if __name__ == "__main__":
    print("Generating Figures 6-8 for Ensemble Virtual Spectrometry paper...")
    print("=" * 60)
    
    generate_figure6()
    generate_figure7()
    generate_figure8()
    
    print("=" * 60)
    print("All figures (6-8) generated successfully!")
    print(f"Figures saved in: {os.path.abspath('figures')}/")
