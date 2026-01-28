"""Generate Figure 7: Reflectance Cascade and Information Amplification"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Reflectance Cascade Network
ax1 = fig.add_subplot(gs[0, :2])

# Create hierarchical cascade network
G = nx.DiGraph()
levels = 4
nodes_per_level = [1, 3, 9, 27]

positions = {}
node_id = 0
level_nodes = {}

for level in range(levels):
    level_nodes[level] = []
    n_nodes = nodes_per_level[level]
    y = 1 - level / (levels - 1)
    
    for i in range(n_nodes):
        x = (i + 1) / (n_nodes + 1)
        positions[node_id] = (x, y)
        G.add_node(node_id, level=level)
        level_nodes[level].append(node_id)
        node_id += 1

# Add edges (cascade connections)
for level in range(levels - 1):
    for parent in level_nodes[level]:
        # Each node connects to 3 children
        parent_x = positions[parent][0]
        children = level_nodes[level + 1]
        
        # Find 3 nearest children
        distances = [(child, abs(positions[child][0] - parent_x)) for child in children]
        distances.sort(key=lambda x: x[1])
        
        for child, _ in distances[:3]:
            G.add_edge(parent, child)

# Color nodes by level
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
node_colors = [colors[G.nodes[node]['level']] for node in G.nodes()]

# Draw network
nx.draw_networkx_nodes(G, positions, node_size=300, node_color=node_colors,
                      edgecolors='black', linewidths=1.5, ax=ax1)
nx.draw_networkx_edges(G, positions, width=1.5, alpha=0.4, 
                      edge_color='gray', arrows=True, 
                      arrowsize=15, ax=ax1)

# Add level labels
for level in range(levels):
    ax1.text(-0.05, 1 - level/(levels-1), f'Level {level}', 
            ha='right', va='center', fontsize=10, weight='bold')

ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.axis('off')
ax1.set_title('(A) Reflectance Cascade Network', fontsize=12, weight='bold')

# Panel B: Information Scaling
ax2 = fig.add_subplot(gs[0, 2])

N_cascade = np.arange(1, 11)
I_linear = N_cascade
I_quadratic = N_cascade**2
I_cubic = N_cascade**3

ax2.semilogy(N_cascade, I_linear, 'b-', linewidth=2.5, marker='o', 
            markersize=8, label='Linear (N)', alpha=0.7)
ax2.semilogy(N_cascade, I_quadratic, 'g-', linewidth=2.5, marker='s', 
            markersize=8, label='Quadratic (N²)', alpha=0.7)
ax2.semilogy(N_cascade, I_cubic, 'r-', linewidth=2.5, marker='^', 
            markersize=8, label='Cubic (N³)', alpha=0.7)

ax2.set_xlabel('Cascade levels N', fontsize=11)
ax2.set_ylabel('Total information I', fontsize=11)
ax2.set_title('(B) Information Scaling', fontsize=11, weight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which='both')

# Panel C: Cascade Depth vs Information Gain
ax3 = fig.add_subplot(gs[1, 0])

depth = np.arange(0, 8)
branching = 3
I_total = branching**depth

ax3.semilogy(depth, I_total, 'ro-', linewidth=2.5, markersize=10)

# Highlight practical limit
practical_limit = 5
ax3.axvline(practical_limit, color='green', linestyle='--', linewidth=2,
           label=f'Practical limit (N={practical_limit})')
ax3.fill_betweenx([1, 1e6], 0, practical_limit, alpha=0.2, color='green')

ax3.set_xlabel('Cascade depth', fontsize=11)
ax3.set_ylabel('Information gain', fontsize=11)
ax3.set_title('(C) Cascade Depth vs Gain', fontsize=11, weight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, which='both')

# Panel D: Reflectance Efficiency
ax4 = fig.add_subplot(gs[1, 1])

# Efficiency vs cascade level
levels_eff = np.arange(1, 11)
efficiency_ideal = np.ones_like(levels_eff, dtype=float)
efficiency_real = np.exp(-0.1 * levels_eff)

ax4.plot(levels_eff, efficiency_ideal, 'k--', linewidth=2, 
        label='Ideal', alpha=0.6)
ax4.plot(levels_eff, efficiency_real, 'r-', linewidth=2.5, marker='o',
        markersize=8, label='Real (with loss)')

# Fill loss region
ax4.fill_between(levels_eff, efficiency_real, efficiency_ideal, 
                alpha=0.2, color='red', label='Loss')

ax4.set_xlabel('Cascade level', fontsize=11)
ax4.set_ylabel('Reflectance efficiency', fontsize=11)
ax4.set_title('(D) Reflectance Efficiency', fontsize=11, weight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.2)

# Panel E: Harmonic Coincidence Network
ax5 = fig.add_subplot(gs[1, 2])

# Create harmonic network
G_harmonic = nx.Graph()
n_nodes = 12

# Arrange nodes in circle
theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
pos_harmonic = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(n_nodes)}

# Add nodes
for i in range(n_nodes):
    G_harmonic.add_node(i)

# Add edges based on harmonic relationships
for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        # Connect if harmonic (divisor relationship)
        if (j - i) in [1, 2, 3, 4, 6]:  # Harmonics
            G_harmonic.add_edge(i, j)

# Draw network
node_colors_harm = plt.cm.hsv(np.linspace(0, 1, n_nodes))
nx.draw_networkx_nodes(G_harmonic, pos_harmonic, node_size=400, 
                      node_color=node_colors_harm,
                      edgecolors='black', linewidths=1.5, ax=ax5)
nx.draw_networkx_edges(G_harmonic, pos_harmonic, width=1, alpha=0.3, 
                      edge_color='gray', ax=ax5)
nx.draw_networkx_labels(G_harmonic, pos_harmonic, font_size=9, 
                       font_color='white', font_weight='bold', ax=ax5)

ax5.set_xlim(-1.3, 1.3)
ax5.set_ylim(-1.3, 1.3)
ax5.set_aspect('equal')
ax5.axis('off')
ax5.set_title('(E) Harmonic Coincidence Network', fontsize=11, weight='bold')

plt.savefig('figures/figure7_reflectance_cascade.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 7: Reflectance Cascade")
print(f"Saved to: {os.path.abspath('figures/figure7_reflectance_cascade.png')}")
