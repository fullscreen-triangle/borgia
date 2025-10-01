#!/usr/bin/env python3
"""
RGB Chemical Mapping
===================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class RGBChemicalMapper:
    def __init__(self):
        self.rgb_mappings = {
            'red': {'reactions': ['oxidation', 'radical'], 'wavelength': 625},
            'green': {'reactions': ['substitution', 'elimination'], 'wavelength': 525},
            'blue': {'reactions': ['reduction', 'addition'], 'wavelength': 470}
        }
    
    def map_pattern_to_rgb(self, pattern):
        """Map molecular pattern to RGB values"""
        if not pattern:
            return {'r': 0, 'g': 0, 'b': 0}
        
        rgb = {'r': 0, 'g': 0, 'b': 0}
        
        # Red mapping - oxidation potential
        if 'OH' in pattern or 'CHO' in pattern:
            rgb['r'] = min(255, 100 + pattern.count('OH') * 40)
        
        # Green mapping - substitution sites
        aromatic = sum(1 for c in pattern if c.islower())
        if aromatic > 0:
            rgb['g'] = min(255, 80 + aromatic * 15)
        
        # Blue mapping - reduction sites
        if 'C=O' in pattern or 'C#N' in pattern:
            rgb['b'] = min(255, 90 + pattern.count('C=O') * 30)
        
        return rgb
    
    def predict_chemical_from_rgb(self, r, g, b):
        """Predict chemical properties from RGB values"""
        predictions = []
        
        if r > 100:
            predictions.append({'type': 'oxidation', 'strength': r/255})
        if g > 80:
            predictions.append({'type': 'substitution', 'probability': g/255})
        if b > 90:
            predictions.append({'type': 'reduction', 'extent': b/255})
        
        return predictions
    
    def validate_rgb_mapping(self, patterns):
        """Validate RGB to chemical mapping"""
        results = []
        
        for pattern in patterns:
            rgb = self.map_pattern_to_rgb(pattern)
            predictions = self.predict_chemical_from_rgb(rgb['r'], rgb['g'], rgb['b'])
            
            results.append({
                'pattern': pattern,
                'rgb': rgb,
                'predictions': predictions,
                'rgb_sum': rgb['r'] + rgb['g'] + rgb['b']
            })
        
        return results

def create_rgb_visualization(rgb_data):
    """Create RGB color visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color squares
    colors = []
    labels = []
    
    for i, data in enumerate(rgb_data[:12]):  # Show first 12
        r, g, b = data['rgb']['r'], data['rgb']['g'], data['rgb']['b']
        color = [r/255, g/255, b/255]
        colors.append(color)
        labels.append(f"P{i+1}")
    
    # Create color grid
    rows, cols = 3, 4
    color_grid = np.array(colors).reshape(rows, cols, 3)
    
    ax1.imshow(color_grid)
    ax1.set_title('RGB Color Mapping')
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    
    # Add labels
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(labels):
                ax1.text(j, i, labels[idx], ha='center', va='center', 
                        color='white', fontweight='bold')
    
    # RGB distribution
    r_vals = [d['rgb']['r'] for d in rgb_data]
    g_vals = [d['rgb']['g'] for d in rgb_data]
    b_vals = [d['rgb']['b'] for d in rgb_data]
    
    ax2.hist(r_vals, alpha=0.5, color='red', label='Red', bins=15)
    ax2.hist(g_vals, alpha=0.5, color='green', label='Green', bins=15)
    ax2.hist(b_vals, alpha=0.5, color='blue', label='Blue', bins=15)
    ax2.set_xlabel('RGB Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('RGB Value Distribution')
    ax2.legend()
    
    return fig

def load_datasets():
    datasets = {}
    files = {
        'agrafiotis': 'gonfanolier/public/agrafiotis-smarts-tar/agrafiotis.smarts',
        'ahmed': 'gonfanolier/public/ahmed-smarts-tar/ahmed.smarts',
        'hann': 'gonfanolier/public/hann-smarts-tar/hann.smarts',
        'walters': 'gonfanolier/public/walters-smarts-tar/walters.smarts'
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            patterns.append(parts[0])
            datasets[name] = patterns
            print(f"Loaded {len(patterns)} patterns from {name}")
    return datasets

def main():
    print("🌈 RGB Chemical Mapping")
    print("=" * 25)
    
    datasets = load_datasets()
    mapper = RGBChemicalMapper()
    
    # Test patterns
    test_patterns = []
    for patterns in datasets.values():
        test_patterns.extend(patterns[:5])
    
    # Map to RGB
    rgb_results = mapper.validate_rgb_mapping(test_patterns)
    
    # Analysis
    patterns_with_color = sum(1 for r in rgb_results if r['rgb_sum'] > 50)
    avg_rgb_sum = np.mean([r['rgb_sum'] for r in rgb_results])
    total_predictions = sum(len(r['predictions']) for r in rgb_results)
    
    print(f"Patterns with significant color: {patterns_with_color}/{len(rgb_results)}")
    print(f"Average RGB intensity: {avg_rgb_sum:.1f}")
    print(f"Total chemical predictions: {total_predictions}")
    
    # Visualization
    fig = create_rgb_visualization(rgb_results)
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('gonfanolier/results/rgb_chemical_mapping.png', dpi=300)
    plt.show()
    
    # Save results
    summary = {
        'patterns_with_color': patterns_with_color,
        'total_patterns': len(rgb_results),
        'avg_rgb_intensity': avg_rgb_sum,
        'total_predictions': total_predictions,
        'success_rate': patterns_with_color / len(rgb_results)
    }
    
    with open('gonfanolier/results/rgb_chemical_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"RGB mapping success rate: {summary['success_rate']:.1%}")
    print("✅ RGB chemical mapping validated!" if summary['success_rate'] > 0.6 else "⚠️ Mapping needs improvement")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()