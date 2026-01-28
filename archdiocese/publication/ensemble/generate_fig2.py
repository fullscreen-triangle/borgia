"""Generate Figure 2: Frequency-Selective Coupling"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Partition Coordinate Frequencies
ax1 = fig.add_subplot(gs[0, :])

# Quantum number regimes
n_vals = np.arange(1, 8)
l_vals = np.arange(0, 4)
m_vals = np.arange(-2, 3)
s_vals = np.array([-0.5, 0.5])

# Frequency scales (in Hz, log scale)
freq_n = 1e15 * (13.6 / (n_vals**2))  # Electronic transitions
freq_l = 1e12 * (l_vals + 1)  # Vibrational
freq_m = 1e9 * (m_vals + 3)  # Rotational
freq_s = 1e6 * np.ones_like(s_vals)  # Hyperfine

# Plot frequency regimes
positions = [1, 2, 3, 4]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
labels = ['n (electronic)', 'l (vibrational)', 'm (rotational)', 's (hyperfine)']

for pos, freq_set, color, label in zip(positions, 
                                         [freq_n, freq_l, freq_m, freq_s],
                                         colors, labels):
    y_pos = np.full_like(freq_set, pos, dtype=float)
    ax1.scatter(freq_set, y_pos, s=200, c=color, alpha=0.7, 
               edgecolors='black', linewidths=1.5, label=label, zorder=10)
    
    # Draw frequency range
    if len(freq_set) > 1:
        ax1.plot([freq_set.min(), freq_set.max()], [pos, pos], 
                color=color, linewidth=4, alpha=0.3, zorder=1)

ax1.set_xscale('log')
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_yticks(positions)
ax1.set_yticklabels(labels)
ax1.set_ylim(0.5, 4.5)
ax1.set_title('(A) Partition Coordinate Frequency Regimes', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.legend(loc='upper right', fontsize=9)

# Panel B: Resonance Condition
ax2 = fig.add_subplot(gs[1, 0])

omega = np.linspace(0, 10, 1000)
omega_0 = 5.0
gamma = 0.5

# Lorentzian resonance
resonance = gamma**2 / ((omega - omega_0)**2 + gamma**2)

ax2.plot(omega, resonance, 'b-', linewidth=2.5)
ax2.axvline(omega_0, color='red', linestyle='--', linewidth=2, 
           label=f'ω₀ = {omega_0}')
ax2.fill_between(omega, 0, resonance, where=(np.abs(omega - omega_0) < gamma),
                 alpha=0.3, color='red', label=f'Δω = {2*gamma}')

ax2.set_xlabel('Oscillator frequency ω', fontsize=11)
ax2.set_ylabel('Coupling strength', fontsize=11)
ax2.set_title('(B) Resonance Condition', fontsize=11, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel C: Multi-Modal Frequency Matching
ax3 = fig.add_subplot(gs[1, 1])

# Multiple resonances
omega_resonances = [2, 4, 6, 8]
colors_res = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
omega_range = np.linspace(0, 10, 1000)

total_response = np.zeros_like(omega_range)
for omega_r, color in zip(omega_resonances, colors_res):
    response = 0.3 / ((omega_range - omega_r)**2 + 0.1)
    ax3.plot(omega_range, response, color=color, linewidth=2, alpha=0.7)
    total_response += response

ax3.plot(omega_range, total_response, 'k-', linewidth=3, 
        label='Total response', alpha=0.8)
ax3.set_xlabel('Frequency ω', fontsize=11)
ax3.set_ylabel('Detection response', fontsize=11)
ax3.set_title('(C) Multi-Modal Frequency Matching', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel D: Frequency Resolution vs Integration Time
ax4 = fig.add_subplot(gs[1, 2])

T_int = np.logspace(-6, 2, 100)  # Integration time (seconds)
delta_omega = 2 * np.pi / T_int  # Frequency resolution

ax4.loglog(T_int, delta_omega, 'b-', linewidth=2.5, label='Δω = 2π/T')

# Mark key timescales
timescales = [1e-6, 1e-3, 1, 100]
labels_time = ['1 μs', '1 ms', '1 s', '100 s']
for t, lbl in zip(timescales, labels_time):
    dw = 2 * np.pi / t
    ax4.plot(t, dw, 'ro', markersize=10)
    ax4.text(t, dw*2, lbl, ha='center', fontsize=8)

ax4.set_xlabel('Integration time T (s)', fontsize=11)
ax4.set_ylabel('Frequency resolution Δω (rad/s)', fontsize=11)
ax4.set_title('(D) Frequency Resolution vs Time', fontsize=11, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, which='both')

plt.savefig('figures/figure2_frequency_coupling.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 2: Frequency-Selective Coupling")
print(f"Saved to: {os.path.abspath('figures/figure2_frequency_coupling.png')}")
