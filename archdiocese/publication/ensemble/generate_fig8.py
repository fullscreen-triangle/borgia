"""Generate Figure 8: Categorical Temporal Resolution"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Phase Accumulation in Oscillator Ensemble
ax1 = fig.add_subplot(gs[0, 0])

t = np.linspace(0, 10, 1000)
N_oscillators = [1, 10, 100, 1000]
colors_osc = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

delta_omega = 0.1  # Frequency difference
for N, color in zip(N_oscillators, colors_osc):
    # Phase accumulation
    delta_phi = np.sqrt(N) * delta_omega * t
    # Wrap to [0, 2π]
    delta_phi_wrapped = np.mod(delta_phi, 2*np.pi)
    
    ax1.plot(t, delta_phi_wrapped, color=color, linewidth=2, 
            label=f'N = {N}', alpha=0.8)

ax1.set_xlabel('Integration time t', fontsize=11)
ax1.set_ylabel('Phase difference Δφ (rad)', fontsize=11)
ax1.set_title('(A) Phase Accumulation', fontsize=11, weight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 2*np.pi)

# Panel B: Temporal Resolution vs Ensemble Size
ax2 = fig.add_subplot(gs[0, 1])

N = np.logspace(0, 15, 100)
omega_osc = 1e9  # 1 GHz oscillator
T_int = 1  # 1 second integration

delta_t_cat = 1 / (N * omega_osc * T_int)

# Fundamental limits
planck_time = 5.4e-44
quantum_limit = 1 / (np.sqrt(N) * omega_osc)

ax2.loglog(N, delta_t_cat, 'r-', linewidth=3, label='Categorical (∝ 1/N)')
ax2.loglog(N, quantum_limit, 'b--', linewidth=2, label='Quantum limit (∝ 1/√N)')
ax2.axhline(planck_time, color='gray', linestyle=':', linewidth=2, 
           label='Planck time')

# Highlight achievable region
ax2.fill_between(N, delta_t_cat, 1e-70, alpha=0.2, color='red',
                label='Categorical regime')

# Mark specific resolutions
N_mark = 1e12
dt_mark = 1 / (N_mark * omega_osc * T_int)
ax2.plot(N_mark, dt_mark, 'ko', markersize=12)
ax2.text(N_mark*2, dt_mark, f'~10^-66 s\nat N=10^12', fontsize=8)

ax2.set_xlabel('Ensemble size N', fontsize=11)
ax2.set_ylabel('Temporal resolution Δt (s)', fontsize=11)
ax2.set_title('(B) Temporal Resolution', fontsize=11, weight='bold')
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3, which='both')

# Panel C: Time-Spectroscopy Duality
ax3 = fig.add_subplot(gs[0, 2])

# Two measurement scenarios
omega_range = np.linspace(0.5, 1.5, 1000)
omega_ref = 1.0

# Scenario 1: Fixed oscillator, varying sample (spectroscopy)
response_spectro = np.sinc((omega_range - omega_ref) * 10)**2

# Scenario 2: Fixed sample, varying oscillator (temporal)
response_temporal = np.sinc((omega_range - omega_ref) * 10)**2

ax3.plot(omega_range, response_spectro, 'b-', linewidth=2.5, 
        label='Spectroscopy mode', alpha=0.8)
ax3.plot(omega_range, response_temporal, 'r--', linewidth=2.5, 
        label='Temporal mode', alpha=0.8)

ax3.axvline(omega_ref, color='green', linestyle=':', linewidth=2,
           label='Reference ω₀')

ax3.set_xlabel('Frequency ω/ω₀', fontsize=11)
ax3.set_ylabel('Detection response', fontsize=11)
ax3.set_title('(C) Time-Spectroscopy Duality', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel D: Multi-Scale Temporal Structure
ax4 = fig.add_subplot(gs[1, 0])

# Different timescales
timescales = ['Planck\n(10^-44 s)', 'Categorical\n(10^-66 s)', 'Attosecond\n(10^-18 s)', 
              'Femtosecond\n(10^-15 s)', 'Nanosecond\n(10^-9 s)', 'Second\n(10^0 s)']
time_values = [1e-44, 1e-66, 1e-18, 1e-15, 1e-9, 1]
colors_time = ['gray', 'red', 'blue', 'green', 'orange', 'purple']
accessible = [False, True, True, True, True, True]

y_pos = np.arange(len(timescales))
bars = ax4.barh(y_pos, np.log10(time_values), 
               color=[c if a else 'lightgray' for c, a in zip(colors_time, accessible)],
               edgecolor='black', linewidth=1.5)

# Highlight accessible regimes
for i, (bar, acc) in enumerate(zip(bars, accessible)):
    if acc:
        bar.set_alpha(0.8)
    else:
        bar.set_alpha(0.3)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(timescales, fontsize=9)
ax4.set_xlabel('log₁₀(time) [s]', fontsize=11)
ax4.set_title('(D) Multi-Scale Temporal Structure', fontsize=11, weight='bold')
ax4.grid(True, alpha=0.3, axis='x')
ax4.axvline(np.log10(1e-44), color='red', linestyle='--', linewidth=2, alpha=0.5)

# Panel E: Categorical State Counting
ax5 = fig.add_subplot(gs[1, 1], projection='3d')

# Phase space trajectory
n_states = 50
t_states = np.linspace(0, 4*np.pi, n_states)

# Oscillatory trajectory
x_traj = np.cos(t_states) * np.exp(-t_states/(4*np.pi))
y_traj = np.sin(t_states) * np.exp(-t_states/(4*np.pi))
z_traj = t_states / (4*np.pi)

# Color by categorical state
colors_states = plt.cm.viridis(np.linspace(0, 1, n_states))

# Draw trajectory
for i in range(n_states-1):
    ax5.plot3D([x_traj[i], x_traj[i+1]], [y_traj[i], y_traj[i+1]], 
              [z_traj[i], z_traj[i+1]], color=colors_states[i], 
              linewidth=2, alpha=0.7)

# Mark categorical states
ax5.scatter(x_traj, y_traj, z_traj, c=colors_states, s=100, 
           edgecolors='black', linewidths=1, alpha=0.9)

ax5.set_xlabel('Phase₁', fontsize=10)
ax5.set_ylabel('Phase₂', fontsize=10)
ax5.set_zlabel('Time', fontsize=10)
ax5.set_title('(E) Categorical State Counting', fontsize=11, weight='bold')

# Panel F: Resolution vs Integration Time
ax6 = fig.add_subplot(gs[1, 2])

T_int_range = np.logspace(-9, 3, 100)
N_ens = 1e9  # 1 billion oscillators
omega = 1e9  # 1 GHz

delta_t_res = 1 / (N_ens * omega * T_int_range)

ax6.loglog(T_int_range, delta_t_res, 'r-', linewidth=3)

# Mark practical integration times
T_practical = [1e-6, 1e-3, 1, 100]
labels_prac = ['1 μs', '1 ms', '1 s', '100 s']

for T, lbl in zip(T_practical, labels_prac):
    dt = 1 / (N_ens * omega * T)
    ax6.plot(T, dt, 'bo', markersize=10)
    ax6.text(T, dt*0.3, lbl, ha='center', fontsize=8)

ax6.set_xlabel('Integration time T (s)', fontsize=11)
ax6.set_ylabel('Categorical resolution Δt_cat (s)', fontsize=11)
ax6.set_title('(F) Resolution vs Integration', fontsize=11, weight='bold')
ax6.grid(True, alpha=0.3, which='both')

# Add diagonal line showing relationship
ax6.plot([1e-9, 1e3], [1e-9, 1e3], 'k--', alpha=0.3, linewidth=1)

plt.savefig('figures/figure8_temporal_resolution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 8: Categorical Temporal Resolution")
print(f"Saved to: {os.path.abspath('figures/figure8_temporal_resolution.png')}")
