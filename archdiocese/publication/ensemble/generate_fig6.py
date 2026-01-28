"""Generate Figure 6: Molecular Observer Networks"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Finite Observer Reach
ax1 = fig.add_subplot(gs[0, 0], projection='3d')

# S-entropy space cube
r = [0, 1]
for i in [0, 1]:
    for j in [0, 1]:
        ax1.plot3D([r[0], r[1]], [r[i], r[i]], [r[j], r[j]], 'gray', alpha=0.2, linewidth=0.5)
        ax1.plot3D([r[i], r[i]], [r[0], r[1]], [r[j], r[j]], 'gray', alpha=0.2, linewidth=0.5)
        ax1.plot3D([r[i], r[i]], [r[j], r[j]], [r[0], r[1]], 'gray', alpha=0.2, linewidth=0.5)

# Observer position
obs_pos = np.array([0.3, 0.4, 0.5])
ax1.scatter([obs_pos[0]], [obs_pos[1]], [obs_pos[2]], s=300, c='red', 
           marker='o', edgecolors='black', linewidths=2, label='Observer', zorder=10)

# Observable region (sphere)
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
reach = 0.3
x_sphere = obs_pos[0] + reach * np.outer(np.cos(u), np.sin(v))
y_sphere = obs_pos[1] + reach * np.outer(np.sin(u), np.sin(v))
z_sphere = obs_pos[2] + reach * np.outer(np.ones(np.size(u)), np.cos(v))

ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='blue')

# Sample observable points
n_points = 15
theta_pts = np.random.uniform(0, 2*np.pi, n_points)
phi_pts = np.random.uniform(0, np.pi, n_points)
r_pts = reach * np.random.uniform(0.5, 1, n_points)

x_pts = obs_pos[0] + r_pts * np.sin(phi_pts) * np.cos(theta_pts)
y_pts = obs_pos[1] + r_pts * np.sin(phi_pts) * np.sin(theta_pts)
z_pts = obs_pos[2] + r_pts * np.cos(phi_pts)

ax1.scatter(x_pts, y_pts, z_pts, s=50, c='blue', alpha=0.6, edgecolors='black', linewidths=0.5)

ax1.set_xlabel('$S_k$', fontsize=10)
ax1.set_ylabel('$S_t$', fontsize=10)
ax1.set_zlabel('$S_e$', fontsize=10)
ax1.set_title('(A) Finite Observer Reach', fontsize=11, weight='bold')
ax1.legend(fontsize=8)

# Panel B: Overlapping Observer Network
ax2 = fig.add_subplot(gs[0, 1])

# Create network graph
G = nx.Graph()
n_observers = 8

# Observer positions in 2D projection
np.random.seed(42)
positions = {}
for i in range(n_observers):
    angle = 2 * np.pi * i / n_observers
    r_pos = 0.5 + 0.3 * np.random.randn()
    positions[i] = (r_pos * np.cos(angle), r_pos * np.sin(angle))
    G.add_node(i)

# Add edges based on overlap
reach_2d = 0.6
for i in range(n_observers):
    for j in range(i+1, n_observers):
        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
        if dist < 2 * reach_2d:
            G.add_edge(i, j)

# Draw network
nx.draw_networkx_nodes(G, positions, node_size=500, node_color='#3498db',
                      edgecolors='black', linewidths=2, ax=ax2)
nx.draw_networkx_edges(G, positions, width=2, alpha=0.5, edge_color='gray', ax=ax2)
nx.draw_networkx_labels(G, positions, font_size=10, font_color='white',
                       font_weight='bold', ax=ax2)

# Draw reach circles
for i, pos in positions.items():
    circle = plt.Circle(pos, reach_2d, fill=False, edgecolor='blue', 
                       linestyle='--', linewidth=1.5, alpha=0.3)
    ax2.add_patch(circle)

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(B) Overlapping Observer Network', fontsize=11, weight='bold')

# Panel C: Cross-Observer Consistency
ax3 = fig.add_subplot(gs[0, 2])

# Measurement consistency matrix
n_obs = 8
consistency = np.eye(n_obs)
for i in range(n_obs):
    for j in range(i+1, n_obs):
        if G.has_edge(i, j):
            # High consistency for connected observers
            val = 0.85 + 0.1 * np.random.randn()
            consistency[i, j] = consistency[j, i] = np.clip(val, 0, 1)
        else:
            # Lower consistency for non-overlapping
            val = 0.3 + 0.1 * np.random.randn()
            consistency[i, j] = consistency[j, i] = np.clip(val, 0, 1)

im = ax3.imshow(consistency, cmap='RdYlGn', vmin=0, vmax=1)
ax3.set_xticks(range(n_obs))
ax3.set_yticks(range(n_obs))
ax3.set_xlabel('Observer j', fontsize=11)
ax3.set_ylabel('Observer i', fontsize=11)
ax3.set_title('(C) Cross-Observer Consistency', fontsize=11, weight='bold')
plt.colorbar(im, ax=ax3, label='Consistency', fraction=0.046)

# Panel D: Dual-Face Information Structure
ax4 = fig.add_subplot(gs[1, 0])

# Front face and back face
categories = np.arange(1, 11)
I_front = np.log2(categories)
I_back = I_front + 0.2 * np.random.randn(len(categories))

ax4.plot(categories, I_front, 'bo-', linewidth=2.5, markersize=8, 
        label='Front face (direct)', alpha=0.8)
ax4.plot(categories, I_back, 'rs-', linewidth=2.5, markersize=8, 
        label='Back face (derived)', alpha=0.8)

# Complementarity region
ax4.fill_between(categories, I_front, I_back, alpha=0.2, color='purple',
                label='Complementarity gap')

ax4.set_xlabel('Categorical distinctions', fontsize=11)
ax4.set_ylabel('Information (bits)', fontsize=11)
ax4.set_title('(D) Dual-Face Information', fontsize=11, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Panel E: Face Complementarity Test
ax5 = fig.add_subplot(gs[1, 1])

# Measurement scenarios
scenarios = ['Direct\nfront', 'Direct\nback', 'Both\n(impossible)', 'Sequential']
front_measured = [1, 0, 0.5, 1]
back_measured = [0, 1, 0.5, 1]
uncertainty = [0.1, 0.1, 0.8, 0.2]

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax5.bar(x - width/2, front_measured, width, label='Front face',
               color='#3498db', edgecolor='black', linewidth=1.5,
               yerr=uncertainty, capsize=5)
bars2 = ax5.bar(x + width/2, back_measured, width, label='Back face',
               color='#e74c3c', edgecolor='black', linewidth=1.5,
               yerr=uncertainty, capsize=5)

ax5.set_ylabel('Measurement fidelity', fontsize=11)
ax5.set_title('(E) Face Complementarity Test', fontsize=11, weight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(scenarios, fontsize=9)
ax5.legend(fontsize=9)
ax5.set_ylim(0, 1.3)
ax5.grid(True, alpha=0.3, axis='y')

# Highlight impossible scenario
ax5.axvspan(1.5, 2.5, alpha=0.2, color='red')

# Panel F: Cross-Face Catalysis
ax6 = fig.add_subplot(gs[1, 2])

# Catalytic enhancement
burden = np.linspace(0, 100, 200)
I_no_catalysis = burden
I_front_only = burden + 0.3 * burden
I_cross_face = burden + 0.3 * burden + 0.1 * burden**1.2

ax6.plot(burden, I_no_catalysis, 'k--', linewidth=2, label='No catalysis', alpha=0.6)
ax6.plot(burden, I_front_only, 'b-', linewidth=2.5, label='Front-face only')
ax6.plot(burden, I_cross_face, 'r-', linewidth=2.5, label='Cross-face catalysis')

# Fill enhancement regions
ax6.fill_between(burden, I_no_catalysis, I_front_only, alpha=0.2, 
                color='blue', label='Front gain')
ax6.fill_between(burden, I_front_only, I_cross_face, alpha=0.2, 
                color='red', label='Cross-face gain')

ax6.set_xlabel('Categorical burden B', fontsize=11)
ax6.set_ylabel('Total information I', fontsize=11)
ax6.set_title('(F) Cross-Face Catalysis', fontsize=11, weight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.savefig('figures/figure6_molecular_observers.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 6: Molecular Observer Networks")
print(f"Saved to: {os.path.abspath('figures/figure6_molecular_observers.png')}")
