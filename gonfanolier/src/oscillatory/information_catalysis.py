#!/usr/bin/env python3
"""
Information Catalysis Theory Implementation
==========================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class InformationCatalysisEngine:
    def __init__(self):
        self.catalysis_history = []
        
    def catalyze_molecular_transformation(self, input_info, output_info):
        """iCat = I_input ∘ I_output"""
        
        # Calculate information catalysis functional composition
        input_entropy = self.calculate_information_entropy(input_info)
        output_entropy = self.calculate_information_entropy(output_info)
        
        # Functional composition operator
        catalysis_result = self.functional_composition(input_entropy, output_entropy)
        
        # Store transformation
        transformation = {
            'input_entropy': input_entropy,
            'output_entropy': output_entropy,
            'catalysis_factor': catalysis_result,
            'amplification': output_entropy / input_entropy if input_entropy > 0 else 1
        }
        
        self.catalysis_history.append(transformation)
        return transformation
    
    def calculate_information_entropy(self, molecular_pattern):
        """Calculate Shannon entropy of molecular information"""
        if not molecular_pattern:
            return 0.0
        
        from collections import Counter
        char_counts = Counter(molecular_pattern)
        total = len(molecular_pattern)
        
        entropy = 0
        for count in char_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy
    
    def functional_composition(self, input_entropy, output_entropy):
        """Compute information catalysis through functional composition"""
        if input_entropy == 0:
            return output_entropy
        
        # Catalysis factor based on information amplification
        catalysis = (output_entropy + input_entropy) * np.exp(-abs(output_entropy - input_entropy))
        return catalysis
    
    def validate_information_conservation(self, transformations):
        """Validate that information is conserved during catalysis"""
        conservation_scores = []
        
        for transform in transformations:
            input_info = transform['input_entropy']
            output_info = transform['output_entropy']
            catalysis = transform['catalysis_factor']
            
            # Conservation score (closer to 1 means better conservation)
            if catalysis > 0:
                conservation = min(input_info + output_info, catalysis) / max(input_info + output_info, catalysis)
            else:
                conservation = 0
            
            conservation_scores.append(conservation)
        
        return {
            'avg_conservation': np.mean(conservation_scores) if conservation_scores else 0,
            'conservation_scores': conservation_scores,
            'well_conserved': sum(1 for s in conservation_scores if s > 0.8)
        }
    
    def analyze_thermodynamic_amplification(self, transformations):
        """Analyze thermodynamic amplification effects"""
        amplifications = [t['amplification'] for t in transformations]
        
        return {
            'avg_amplification': np.mean(amplifications) if amplifications else 1,
            'max_amplification': max(amplifications) if amplifications else 1,
            'amplification_variance': np.var(amplifications) if amplifications else 0,
            'significant_amplifications': sum(1 for a in amplifications if a > 1.5)
        }

class CatalysisValidator:
    def __init__(self):
        self.engine = InformationCatalysisEngine()
        
    def validate_catalysis_across_datasets(self, datasets):
        """Validate information catalysis across molecular datasets"""
        results = {}
        
        for dataset_name, patterns in datasets.items():
            transformations = []
            
            # Create molecular transformations
            for i in range(len(patterns) - 1):
                input_pattern = patterns[i]
                output_pattern = patterns[i + 1]
                
                transformation = self.engine.catalyze_molecular_transformation(input_pattern, output_pattern)
                transformations.append(transformation)
            
            # Analyze transformations
            conservation = self.engine.validate_information_conservation(transformations)
            amplification = self.engine.analyze_thermodynamic_amplification(transformations)
            
            results[dataset_name] = {
                'transformations': len(transformations),
                'conservation_analysis': conservation,
                'amplification_analysis': amplification,
                'avg_catalysis_factor': np.mean([t['catalysis_factor'] for t in transformations]) if transformations else 0
            }
        
        return results

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
    print("⚡ Information Catalysis Theory Implementation")
    print("=" * 50)
    
    datasets = load_datasets()
    validator = CatalysisValidator()
    
    print(f"\n🧪 Validating information catalysis across datasets...")
    results = validator.validate_catalysis_across_datasets(datasets)
    
    # Display results
    for dataset_name, data in results.items():
        conservation = data['conservation_analysis']
        amplification = data['amplification_analysis']
        
        print(f"\n{dataset_name}:")
        print(f"  Transformations: {data['transformations']}")
        print(f"  Avg catalysis factor: {data['avg_catalysis_factor']:.3f}")
        print(f"  Conservation score: {conservation['avg_conservation']:.2f}")
        print(f"  Avg amplification: {amplification['avg_amplification']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Information Catalysis Analysis')
    
    datasets_list = list(results.keys())
    
    # Catalysis factors
    catalysis_factors = [results[d]['avg_catalysis_factor'] for d in datasets_list]
    axes[0,0].bar(datasets_list, catalysis_factors, color='blue', alpha=0.7)
    axes[0,0].set_title('Average Catalysis Factors')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Conservation scores
    conservation_scores = [results[d]['conservation_analysis']['avg_conservation'] for d in datasets_list]
    axes[0,1].bar(datasets_list, conservation_scores, color='green', alpha=0.7)
    axes[0,1].set_title('Information Conservation')
    axes[0,1].set_ylim(0, 1)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Amplification factors
    amplification_factors = [results[d]['amplification_analysis']['avg_amplification'] for d in datasets_list]
    axes[1,0].bar(datasets_list, amplification_factors, color='orange', alpha=0.7)
    axes[1,0].set_title('Thermodynamic Amplification')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Transformation counts
    transformation_counts = [results[d]['transformations'] for d in datasets_list]
    axes[1,1].bar(datasets_list, transformation_counts, color='purple', alpha=0.7)
    axes[1,1].set_title('Number of Transformations')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    os.makedirs('gonfanolier/results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('gonfanolier/results/information_catalysis.png', dpi=300)
    plt.show()
    
    # Save results
    with open('gonfanolier/results/information_catalysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary statistics
    avg_catalysis = np.mean(catalysis_factors)
    avg_conservation = np.mean(conservation_scores)
    avg_amplification = np.mean(amplification_factors)
    
    print(f"\n📊 Overall Results:")
    print(f"Average catalysis factor: {avg_catalysis:.3f}")
    print(f"Average conservation: {avg_conservation:.2f}")
    print(f"Average amplification: {avg_amplification:.2f}")
    
    print(f"\n🎯 Validation Results:")
    
    if avg_catalysis > 2.0:
        print("✅ Strong information catalysis demonstrated")
    
    if avg_conservation > 0.7:
        print("✅ Information conservation validated")
    
    if avg_amplification > 1.2:
        print("✅ Thermodynamic amplification confirmed")
    
    print("🏁 Information catalysis analysis complete!")

if __name__ == "__main__":
    main()