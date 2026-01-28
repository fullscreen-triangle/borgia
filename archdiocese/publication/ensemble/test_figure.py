"""Quick test to generate one figure"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title('Test Figure')
plt.savefig('figures/test.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Test figure generated successfully!")
print(f"Saved to: {os.path.abspath('figures/test.png')}")
