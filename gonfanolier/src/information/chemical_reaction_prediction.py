#!/usr/bin/env python3
"""
Chemical Reaction Prediction Using Fuzzy SMARTS
==============================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

class ReactionPatternMatcher:
    def count_exact_pattern_matches(self, patterns, reactions):
        matches = sum(1 for p in patterns for r in reactions if p in r)
        return matches
    
    def count_fuzzy_pattern_matches(self, patterns, reactions, tolerance=0.2):
        fuzzy_matches = 0
        for pattern in patterns:
            # Create fuzzy variants
            variants = [pattern]
            if '=' in pattern:
                variants.append(pattern.replace('=', '-'))
            if 'C' in pattern:
                variants.append(pattern.replace('C', '[C,N]'))
            
            for variant in variants:
                fuzzy_matches += sum(1 for r in reactions if variant in r)
        
        return fuzzy_matches

class ReactionMechanismPredictor:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def extract_features(self, patterns):
        features = []
        for pattern in patterns:
            feature_vec = [
                len(pattern),
                pattern.count('='),
                pattern.count('C'),
                pattern.count('N'),
                pattern.count('O'),
                pattern.count('OH'),
                pattern.count('C=O')
            ]
            features.append(feature_vec)
        return np.array(features)
    
    def predict_mechanisms(self, patterns, labels):
        features = self.extract_features(patterns)
        
        # Create mechanism labels
        mechanism_labels = []
        for pattern in patterns:
            if 'C=O' in pattern:
                mechanism_labels.append('carbonyl_reaction')
            elif 'OH' in pattern:
                mechanism_labels.append('hydroxyl_substitution')
            else:
                mechanism_labels.append('general_reaction')
        
        if len(set(mechanism_labels)) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                features, mechanism_labels, test_size=0.3, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            accuracy = self.classifier.score(X_test, y_test)
            
            return {
                'accuracy': accuracy,
                'mechanisms_identified': len(set(mechanism_labels))
            }
        
        return {'accuracy': 0, 'mechanisms_identified': len(set(mechanism_labels))}

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
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            patterns.append(parts[0])
            datasets[name] = patterns
            print(f"Loaded {len(patterns)} patterns from {name}")
    return datasets

def create_reaction_database(patterns):
    """Create synthetic reaction patterns"""
    reactions = []
    for pattern in patterns:
        if 'C=O' in pattern:
            reactions.append(f"{pattern}>>C-OH")
        elif 'OH' in pattern:
            reactions.append(f"{pattern}>>Cl")
        else:
            reactions.append(f"{pattern}>>{pattern}[mod]")
    return reactions

def main():
    print("🧪 Chemical Reaction Prediction Analysis")
    print("=" * 45)
    
    datasets = load_datasets()
    matcher = ReactionPatternMatcher()
    predictor = ReactionMechanismPredictor()
    
    # Combine all patterns
    all_patterns = []
    all_labels = []
    
    for name, patterns in datasets.items():
        all_patterns.extend(patterns)
        all_labels.extend([name] * len(patterns))
    
    # Create reaction database
    reaction_db = create_reaction_database(all_patterns)
    
    # Test pattern matching
    print(f"\n🔍 Testing pattern matching on {len(all_patterns)} patterns...")
    rigid_matches = matcher.count_exact_pattern_matches(all_patterns, reaction_db)
    fuzzy_matches = matcher.count_fuzzy_pattern_matches(all_patterns, reaction_db)
    
    print(f"Rigid SMARTS matches: {rigid_matches}")
    print(f"Fuzzy SMARTS matches: {fuzzy_matches}")
    print(f"Additional matches: {fuzzy_matches - rigid_matches}")
    
    # Test mechanism prediction
    print(f"\n🔬 Testing mechanism prediction...")
    mechanism_results = predictor.predict_mechanisms(all_patterns, all_labels)
    
    print(f"Prediction accuracy: {mechanism_results['accuracy']:.2%}")
    print(f"Mechanisms identified: {mechanism_results['mechanisms_identified']}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pattern matching
    categories = ['Rigid', 'Fuzzy']
    matches = [rigid_matches, fuzzy_matches]
    ax1.bar(categories, matches, color=['blue', 'orange'])
    ax1.set_title('Pattern Matching Performance')
    ax1.set_ylabel('Number of Matches')
    
    if rigid_matches > 0:
        improvement = (fuzzy_matches - rigid_matches) / rigid_matches * 100
        ax1.text(1, fuzzy_matches + 2, f'+{improvement:.1f}%', ha='center')
    
    # Dataset contributions
    dataset_counts = [len(patterns) for patterns in datasets.values()]
    ax2.pie(dataset_counts, labels=datasets.keys(), autopct='%1.1f%%')
    ax2.set_title('Dataset Composition')
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('gonfanolier/results/reaction_prediction.png', dpi=300)
    plt.show()
    
    # Save results
    results = {
        'pattern_matching': {
            'rigid_matches': rigid_matches,
            'fuzzy_matches': fuzzy_matches,
            'additional_matches': fuzzy_matches - rigid_matches,
            'improvement_ratio': fuzzy_matches / rigid_matches if rigid_matches > 0 else 0
        },
        'mechanism_prediction': mechanism_results,
        'dataset_info': {k: len(v) for k, v in datasets.items()}
    }
    
    with open('gonfanolier/results/reaction_prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Results:")
    print(f"✅ Fuzzy SMARTS found {fuzzy_matches - rigid_matches} additional reaction patterns")
    print(f"✅ Mechanism prediction accuracy: {mechanism_results['accuracy']:.1%}")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()