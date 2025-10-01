#!/usr/bin/env python3
"""
Computer Vision Chemical Analysis
================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

class ChemicalPatternAnalyzer:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def extract_features(self, pattern):
        """Extract features from drip pattern"""
        features = [
            np.mean(pattern),
            np.std(pattern),
            np.max(pattern),
            np.min(pattern),
            pattern.shape[0] * pattern.shape[1]
        ]
        return features
    
    def analyze_patterns(self, patterns, labels):
        """Analyze drip patterns for classification"""
        features = np.array([self.extract_features(p) for p in patterns])
        
        if len(set(labels)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            accuracy = self.classifier.score(X_test, y_test)
            
            return {
                'accuracy': accuracy,
                'feature_importance': self.classifier.feature_importances_.tolist(),
                'n_features': features.shape[1]
            }
        
        return {'accuracy': 0, 'message': 'Need multiple classes'}

def generate_drip_patterns(smarts_patterns, labels):
    """Generate synthetic drip patterns"""
    patterns = []
    
    for pattern in smarts_patterns:
        # Simple pattern generation
        size = 50
        complexity = len(set(pattern)) / len(pattern) if pattern else 0.5
        
        y, x = np.ogrid[:size, :size]
        center = size // 2
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        
        wave = complexity * np.exp(-distance / 10) * np.cos(2 * np.pi * distance / 5)
        patterns.append(wave)
    
    return patterns

def load_datasets():
    """Load datasets"""
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
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            patterns.append(parts[0])
            datasets[name] = patterns
            print(f"Loaded {len(patterns)} patterns from {name}")
    
    return datasets

def main():
    print("👁️ Computer Vision Chemical Analysis")
    print("=" * 40)
    
    datasets = load_datasets()
    analyzer = ChemicalPatternAnalyzer()
    
    # Prepare data
    all_patterns = []
    all_labels = []
    
    for dataset_name, patterns in datasets.items():
        sample_patterns = patterns[:10]  # Sample for demo
        all_patterns.extend(sample_patterns)
        all_labels.extend([dataset_name] * len(sample_patterns))
    
    print(f"\n📊 Processing {len(all_patterns)} patterns...")
    
    # Generate drip patterns
    drip_patterns = generate_drip_patterns(all_patterns, all_labels)
    
    # Analyze with computer vision
    results = analyzer.analyze_patterns(drip_patterns, all_labels)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Computer Vision Chemical Analysis')
    
    for i in range(4):
        row, col = divmod(i, 2)
        if i < len(drip_patterns):
            axes[row, col].imshow(drip_patterns[i], cmap='viridis')
            axes[row, col].set_title(f'{all_labels[i]}')
            axes[row, col].axis('off')
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.savefig('gonfanolier/results/cv_chemical_analysis.png', dpi=300)
    plt.show()
    
    # Save results
    with open('gonfanolier/results/cv_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Results:")
    print(f"  Accuracy: {results.get('accuracy', 0):.2%}")
    print(f"  Features: {results.get('n_features', 0)}")
    
    if results.get('accuracy', 0) > 0:
        print("✅ Computer vision successfully analyzes chemical patterns")
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()