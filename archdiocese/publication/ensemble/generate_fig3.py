"""Generate Figure 3: Ensemble Measurement Architecture"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Hardware Oscillator Ensemble
ax1 = fig.add_subplot(gs[0, 0])

# Oscillator frequencies (log scale)
N_osc = 20
frequencies = np.logspace(6, 15, N_osc)
phases = np.random.uniform(0, 2*np.pi, N_osc)

# Plot oscillators as points
colors = plt.cm.viridis(np.linspace(0, 1, N_osc))
for i, (f, ph, c) in enumerate(zip(frequencies, phases, colors)):
    ax1.scatter(f, ph, s=150, c=[c], edgecolors='black', linewidths=1.5, zorder=10)
    
# Highlight partition coordinate regimes
regimes = [(1e6, 1e7, 's'), (1e9, 1e10, 'm'), (1e12, 1e13, 'l'), (1e15, 1e16, 'n')]
for f_min, f_max, label in regimes:
    ax1.axvspan(f_min, f_max, alpha=0.15, label=f'coord {label}')

ax1.set_xscale('log')
ax1.set_xlabel('Oscillator frequency (Hz)', fontsize=11)
ax1.set_ylabel('Phase φ (rad)', fontsize=11)
ax1.set_ylim(0, 2*np.pi)
ax1.set_title('(A) Hardware Oscillator Ensemble', fontsize=11, weight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel B: Temporal Resolution vs Ensemble Size
ax2 = fig.add_subplot(gs[0, 1])

N = np.logspace(0, 4, 100)
delta_t = 1e-15 / np.sqrt(N)  # Standard quantum limit
C = 1 - np.exp(-N/1000)  # Coverage

ax2_right = ax2.twinx()

line1, = ax2.loglog(N, delta_t, 'b-', linewidth=2, label='Δt ∝ N^(-1/2)')
line2, = ax2_right.semilogx(N, C, 'r-', linewidth=2, label='C → 1')

# Crossover point
idx_cross = np.argmin(np.abs(delta_t/delta_t[0] - C))
line3, = ax2.plot(N[idx_cross], delta_t[idx_cross], 'ko', markersize=10, 
        label=f'Optimal N ≈ {N[idx_cross]:.0f}')

ax2.set_xlabel('Ensemble size N', fontsize=11)
ax2.set_ylabel('Temporal resolution Δt (s)', color='b', fontsize=11)
ax2_right.set_ylabel('Spatial coverage C', color='r', fontsize=11)
ax2.set_title('(B) Temporal Resolution vs Ensemble Size', fontsize=11, weight='bold')
ax2.tick_params(axis='y', labelcolor='b')
ax2_right.tick_params(axis='y', labelcolor='r')
ax2.grid(True, alpha=0.3)

lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=8)

# Panel C: Phase Accumulation
ax3 = fig.add_subplot(gs[1, 0])

t = np.linspace(0, 10, 1000)
omega_1 = 1.0
omega_2 = 1.1

phase_1 = omega_1 * t
phase_2 = omega_2 * t
delta_phase = (omega_2 - omega_1) * t

ax3.plot(t, np.mod(phase_1, 2*np.pi), 'b-', linewidth=2, label='ω₁', alpha=0.7)
ax3.plot(t, np.mod(phase_2, 2*np.pi), 'r-', linewidth=2, label='ω₂', alpha=0.7)
ax3.plot(t, np.mod(delta_phase, 2*np.pi), 'k-', linewidth=2.5, 
        label='Δφ = (ω₂-ω₁)t', alpha=0.9)

ax3.set_xlabel('Time t', fontsize=11)
ax3.set_ylabel('Phase (rad)', fontsize=11)
ax3.set_title('(C) Phase Accumulation', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 2*np.pi)

# Panel D: Categorical Temporal Resolution
ax4 = fig.add_subplot(gs[1, 1])

# Different ensemble sizes
N_ensemble = [1, 10, 100, 1000]
colors_ens = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

omega_range = np.linspace(0.9, 1.1, 1000)
T_int = 100  # Integration time

for N, color in zip(N_ensemble, colors_ens):
    # Phase sensitivity
    delta_phi = np.sqrt(N) * (omega_range - 1.0) * T_int
    sensitivity = np.sinc(delta_phi / np.pi)**2
    
    ax4.plot(omega_range, sensitivity, color=color, linewidth=2, 
            label=f'N = {N}', alpha=0.8)

ax4.set_xlabel('Frequency ω/ω₀', fontsize=11)
ax4.set_ylabel('Detection sensitivity', fontsize=11)
ax4.set_title('(D) Categorical Temporal Resolution', fontsize=11, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.savefig('figures/figure3_ensemble_measurement.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 3: Ensemble Measurement")
print(f"Saved to: {os.path.abspath('figures/figure3_ensemble_measurement.png')}")
