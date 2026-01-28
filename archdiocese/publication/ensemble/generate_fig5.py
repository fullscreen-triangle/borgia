"""Generate Figure 5: Information Catalysis Mechanism"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Categorical Burden Accumulation
ax1 = fig.add_subplot(gs[0, 0])

time = np.linspace(0, 10, 200)
burden_linear = time
burden_quadratic = 0.1 * time**2
burden_cubic = 0.01 * time**3

ax1.plot(time, burden_linear, 'b-', linewidth=2.5, label='Linear (no catalysis)')
ax1.plot(time, burden_quadratic, 'r-', linewidth=2.5, label='Quadratic (2-body)')
ax1.plot(time, burden_cubic, 'g-', linewidth=2.5, label='Cubic (3-body)')

ax1.set_xlabel('Measurement time', fontsize=11)
ax1.set_ylabel('Categorical burden B', fontsize=11)
ax1.set_title('(A) Categorical Burden Accumulation', fontsize=11, weight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel B: Information Generation Rate
ax2 = fig.add_subplot(gs[0, 1])

burden = np.linspace(0, 100, 200)
dI_linear = np.ones_like(burden)
dI_catalytic = 1 + 0.05*burden

ax2.plot(burden, dI_linear, 'b-', linewidth=2.5, label='Standard')
ax2.plot(burden, dI_catalytic, 'r-', linewidth=2.5, label='Catalytic')

# Fill area between curves
ax2.fill_between(burden, dI_linear, dI_catalytic, alpha=0.2, color='green',
                label='Catalytic gain')

ax2.set_xlabel('Categorical burden B', fontsize=11)
ax2.set_ylabel('Information generation rate dI/dB', fontsize=11)
ax2.set_title('(B) Information Generation Rate', fontsize=11, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel C: Aperture vs Demon Comparison
ax3 = fig.add_subplot(gs[0, 2])

categories = ['Energy\ncost', 'Entropy\nproduction', 'Information\ngain', 'Reversibility']
aperture = [0, 0, 1, 1]
demon = [1, 1, 1, 0]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, aperture, width, label='Categorical Aperture',
               color='#2ecc71', edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, demon, width, label='Maxwell Demon',
               color='#e74c3c', edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Normalized value', fontsize=11)
ax3.set_title('(C) Aperture vs Demon', fontsize=11, weight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=9)
ax3.legend(fontsize=9)
ax3.set_ylim(0, 1.2)
ax3.grid(True, alpha=0.3, axis='y')

# Panel D: Resonant Partition Coupling
ax4 = fig.add_subplot(gs[1, 0])

# Energy levels
n_levels = 5
E_levels = np.array([1/(n**2) for n in range(1, n_levels+1)])

# Plot energy levels
for i, E in enumerate(E_levels):
    ax4.hlines(E, 0, 1, colors='black', linewidth=2)
    ax4.text(-0.1, E, f'n={i+1}', ha='right', va='center', fontsize=9)

# Resonant transitions
transitions = [(0, 1), (1, 2), (0, 2)]
colors_trans = ['red', 'blue', 'green']
for (i, j), color in zip(transitions, colors_trans):
    ax4.annotate('', xy=(0.5, E_levels[j]), xytext=(0.5, E_levels[i]),
                arrowprops=dict(arrowstyle='<->', color=color, lw=2.5))
    delta_E = E_levels[i] - E_levels[j]
    ax4.text(0.6, (E_levels[i] + E_levels[j])/2, 
            f'ΔE={delta_E:.2f}', fontsize=8, color=color)

ax4.set_xlim(-0.2, 1)
ax4.set_ylim(0, 1.2)
ax4.set_ylabel('Energy', fontsize=11)
ax4.set_title('(D) Resonant Partition Coupling', fontsize=11, weight='bold')
ax4.set_xticks([])
ax4.grid(True, alpha=0.3, axis='y')

# Panel E: Multi-Modal Information Synthesis
ax5 = fig.add_subplot(gs[1, 1], projection='3d')

# Reference ion array
n_refs = 8
theta = np.linspace(0, 2*np.pi, n_refs, endpoint=False)
r = 0.8

x_refs = r * np.cos(theta)
y_refs = r * np.sin(theta)
z_refs = np.zeros(n_refs)

# Unknown ion at center
ax5.scatter([0], [0], [0], s=400, c='red', marker='o', 
           edgecolors='black', linewidths=2, label='Unknown', zorder=10)

# Reference ions
ax5.scatter(x_refs, y_refs, z_refs, s=200, c='blue', marker='s',
           edgecolors='black', linewidths=1.5, label='References', zorder=9)

# Measurement connections
for i in range(n_refs):
    ax5.plot([0, x_refs[i]], [0, y_refs[i]], [0, z_refs[i]], 
            'gray', alpha=0.4, linewidth=1.5)

# Information flow (spiral upward)
t_spiral = np.linspace(0, 4*np.pi, 100)
r_spiral = 0.3 * np.exp(-t_spiral/(4*np.pi))
x_spiral = r_spiral * np.cos(t_spiral)
y_spiral = r_spiral * np.sin(t_spiral)
z_spiral = t_spiral / (4*np.pi)

ax5.plot(x_spiral, y_spiral, z_spiral, 'green', linewidth=3, 
        alpha=0.7, label='Info synthesis')

ax5.set_xlabel('$S_k$', fontsize=10)
ax5.set_ylabel('$S_t$', fontsize=10)
ax5.set_zlabel('$S_e$', fontsize=10)
ax5.set_title('(E) Multi-Modal Synthesis', fontsize=11, weight='bold')
ax5.legend(fontsize=8)

# Panel F: Thermodynamic Cost Analysis
ax6 = fig.add_subplot(gs[1, 2])

operations = ['Categorical\ndistinction', 'Partition\ncompletion', 'Information\ngeneration', 'Memory\nwrite']
cost_landauer = [0, 0, 0, 1]
cost_measured = [0, 0, 0, 0.95]

x = np.arange(len(operations))
width = 0.35

bars1 = ax6.bar(x - width/2, cost_landauer, width, label='Landauer bound',
               color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x + width/2, cost_measured, width, label='Measured',
               color='#e74c3c', edgecolor='black', linewidth=1.5)

ax6.set_ylabel('Energy cost (k_B T ln 2)', fontsize=11)
ax6.set_title('(F) Thermodynamic Cost', fontsize=11, weight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(operations, fontsize=9)
ax6.legend(fontsize=9)
ax6.set_ylim(0, 1.2)
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('figures/figure5_information_catalysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 5: Information Catalysis")
print(f"Saved to: {os.path.abspath('figures/figure5_information_catalysis.png')}")
