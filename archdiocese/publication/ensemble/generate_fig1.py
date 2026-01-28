"""Generate Figure 1 only"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os

os.makedirs('figures', exist_ok=True)
plt.style.use('seaborn-v0_8-paper')

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Panel A: 3D Entropy Coordinate Space
ax1 = fig.add_subplot(gs[0, 0], projection='3d')

# Draw unit cube wireframe
r = [0, 1]
for i in [0, 1]:
    for j in [0, 1]:
        ax1.plot3D([r[0], r[1]], [r[i], r[i]], [r[j], r[j]], 'gray', alpha=0.3, linewidth=0.5)
        ax1.plot3D([r[i], r[i]], [r[0], r[1]], [r[j], r[j]], 'gray', alpha=0.3, linewidth=0.5)
        ax1.plot3D([r[i], r[i]], [r[j], r[j]], [r[0], r[1]], 'gray', alpha=0.3, linewidth=0.5)

# Sample trajectory
t = np.linspace(0, 4*np.pi, 200)
Sk_traj = 0.5 + 0.3*np.sin(t) * np.exp(-t/20)
St_traj = 0.5 + 0.3*np.cos(t) * np.exp(-t/20)
Se_traj = 0.3 + 0.4*t/(4*np.pi)

# Color gradient
colors = plt.cm.coolwarm(np.linspace(0, 1, len(t)))
for i in range(len(t)-1):
    ax1.plot3D(Sk_traj[i:i+2], St_traj[i:i+2], Se_traj[i:i+2], 
               color=colors[i], linewidth=2, alpha=0.8)

# Categorical states as spheres
n_states = 10
state_idx = np.linspace(0, len(t)-1, n_states, dtype=int)
ax1.scatter(Sk_traj[state_idx], St_traj[state_idx], Se_traj[state_idx],
            c=np.linspace(0, 1, n_states), cmap='coolwarm', s=100, 
            edgecolors='black', linewidths=1, alpha=0.9)

# Partition boundaries (k=3, showing 27 cells)
for i in [1/3, 2/3]:
    xx, yy = np.meshgrid([i, i], [0, 1])
    zz = np.array([[0, 0], [1, 1]])
    ax1.plot_surface(xx, yy, zz, alpha=0.1, color='blue')
    ax1.plot_surface(yy, xx, zz, alpha=0.1, color='green')
    ax1.plot_surface(yy, zz, xx, alpha=0.1, color='red')

ax1.set_xlabel('$S_k$')
ax1.set_ylabel('$S_t$')
ax1.set_zlabel('$S_e$')
ax1.set_title('(A) 3D Entropy Coordinate Space')

# Panel B: Hierarchical Partition Refinement
ax2 = fig.add_subplot(gs[0, 1])

k_values = [1, 2, 3, 4]
for idx, k in enumerate(k_values):
    n_cells = 3**k
    grid_size = int(np.sqrt(n_cells))
    
    # Create mini subplot
    mini_ax = plt.axes([0.55 + (idx % 2)*0.2, 0.7 - (idx // 2)*0.2, 0.15, 0.15])
    
    # Generate ternary addresses
    grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            grid[i, j] = (i * grid_size + j) % 27
    
    mini_ax.imshow(grid, cmap='tab20', interpolation='nearest')
    mini_ax.set_xticks([])
    mini_ax.set_yticks([])
    mini_ax.set_title(f'k={k} (3^{3*k}={3**(3*k):,})', fontsize=8)
    
    # Highlight one cell
    if k <= 3:
        rect_size = grid_size // 3
        rect = plt.Rectangle((rect_size, rect_size), rect_size, rect_size,
                            fill=False, edgecolor='red', linewidth=2)
        mini_ax.add_patch(rect)

ax2.axis('off')
ax2.set_title('(B) Hierarchical Partition Refinement')

# Panel C: Ternary Address Encoding
ax3 = fig.add_subplot(gs[1, 0])

def draw_ternary_tree(ax, x, y, level, max_level, address=""):
    if level > max_level:
        return
    
    # Draw node
    node_size = 300 / (level + 1)
    color_val = int(address, 3) / (3**max_level) if address else 0
    ax.scatter(x, y, s=node_size, c=[color_val], cmap='viridis', 
              vmin=0, vmax=1, edgecolors='black', linewidths=1, zorder=10)
    
    if level == max_level:
        if address == "0210":
            ax.scatter(x, y, s=node_size*1.5, facecolors='none', 
                      edgecolors='red', linewidths=3, zorder=11)
            ax.text(x, y-0.15, f'{address}_3', ha='center', fontsize=9, 
                   color='red', weight='bold')
        return
    
    # Draw branches
    dx = 0.8 / (3**(level+1))
    dy = 0.2
    
    for i, digit in enumerate(['0', '1', '2']):
        child_x = x + (i - 1) * dx
        child_y = y - dy
        
        # Draw edge
        line_color = 'red' if address + digit == "0210"[:level+2] else 'gray'
        line_width = 2 if address + digit == "0210"[:level+2] else 0.5
        ax.plot([x, child_x], [y, child_y], color=line_color, 
               linewidth=line_width, alpha=0.6, zorder=1)
        
        # Draw child
        draw_ternary_tree(ax, child_x, child_y, level+1, max_level, address + digit)

draw_ternary_tree(ax3, 0.5, 0.9, 0, 4)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.set_title('(C) Ternary Address Encoding')

# Panel D: Convergence to Continuum
ax4 = fig.add_subplot(gs[1, 1])

k = np.arange(1, 21)
V = 3.0**(-3*k)

ax4.semilogy(k, V, 'o-', linewidth=2, markersize=6, label='V(k) = 3^(-3k)')
ax4.axhline(1e-16, color='red', linestyle='--', linewidth=1.5, 
            label='Machine precision')
ax4.fill_between(k, 1e-18, 1e-16, alpha=0.2, color='gray', 
                 label='Continuum limit')

ax4.set_xlabel('Number of trits (k)')
ax4.set_ylabel('Cell volume')
ax4.set_title('(D) Convergence to Continuum')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.savefig('figures/figure1_ternary_encoding.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Generated Figure 1: Ternary Encoding")
print(f"Saved to: {os.path.abspath('figures/figure1_ternary_encoding.png')}")
