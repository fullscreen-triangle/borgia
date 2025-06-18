//! Borgia - Revolutionary Probabilistic Cheminformatics Engine
//! 
//! Main executable demonstrating the core functionality.

use borgia::{
    BorgiaError, Result,
    ProbabilisticMolecule,
    SimilarityEngine, SimilarityAlgorithm,
    ProbabilisticValue,
};

fn main() -> Result<()> {
    println!("🧬 Borgia - Revolutionary Probabilistic Cheminformatics Engine");
    println!("================================================================");
    
    // Demonstrate basic functionality
    demo_molecular_representation()?;
    demo_similarity_calculation()?;
    demo_uncertainty_quantification()?;
    
    println!("\n✅ All demonstrations completed successfully!");
    Ok(())
}

/// Demonstrate enhanced molecular representation
fn demo_molecular_representation() -> Result<()> {
    println!("\n📋 1. Enhanced Molecular Representation");
    println!("---------------------------------------");
    
    // Create probabilistic molecules
    let ethanol = ProbabilisticMolecule::from_smiles("CCO")?;
    let methanol = ProbabilisticMolecule::from_smiles("CO")?;
    let benzene = ProbabilisticMolecule::from_smiles("c1ccccc1")?;
    
    println!("Ethanol:  {}", ethanol);
    println!("Methanol: {}", methanol);
    println!("Benzene:  {}", benzene);
    
    // Show fuzzy features
    println!("\n🔍 Fuzzy Features:");
    println!("Ethanol aromaticity: {:.3}±{:.3}", 
        ethanol.fuzzy_aromaticity.overall_aromaticity.mean,
        ethanol.fuzzy_aromaticity.overall_aromaticity.std_dev);
    println!("Benzene aromaticity: {:.3}±{:.3}", 
        benzene.fuzzy_aromaticity.overall_aromaticity.mean,
        benzene.fuzzy_aromaticity.overall_aromaticity.std_dev);
    
    // Show hydrogen bonding capacity
    println!("Ethanol H-bonding: {:.3}±{:.3}", 
        ethanol.fuzzy_groups.h_bonding.mean,
        ethanol.fuzzy_groups.h_bonding.std_dev);
    
    Ok(())
}

/// Demonstrate similarity calculations
fn demo_similarity_calculation() -> Result<()> {
    println!("\n🔬 2. Probabilistic Similarity Calculation");
    println!("------------------------------------------");
    
    let mol1 = ProbabilisticMolecule::from_smiles("CCO")?;  // Ethanol
    let mol2 = ProbabilisticMolecule::from_smiles("CCC")?; // Propane
    let mol3 = ProbabilisticMolecule::from_smiles("CCO")?; // Ethanol (identical)
    
    let engine = SimilarityEngine::new();
    
    // Calculate different types of similarities
    let sim_different = engine.calculate_similarity(
        &mol1, &mol2, SimilarityAlgorithm::Tanimoto, "structural_similarity"
    )?;
    
    let sim_identical = engine.calculate_similarity(
        &mol1, &mol3, SimilarityAlgorithm::Tanimoto, "structural_similarity"
    )?;
    
    let sim_probabilistic = engine.calculate_similarity(
        &mol1, &mol2, SimilarityAlgorithm::ProbabilisticTanimoto, "drug_metabolism"
    )?;
    
    println!("Ethanol vs Propane (Tanimoto): {:.3}±{:.3} [{}]",
        sim_different.similarity.mean,
        sim_different.similarity.std_dev,
        sim_different.distribution.most_likely());
    
    println!("Ethanol vs Ethanol (Tanimoto): {:.3}±{:.3} [{}]",
        sim_identical.similarity.mean,
        sim_identical.similarity.std_dev,
        sim_identical.distribution.most_likely());
    
    println!("Ethanol vs Propane (Probabilistic): {:.3}±{:.3} [{}]",
        sim_probabilistic.similarity.mean,
        sim_probabilistic.similarity.std_dev,
        sim_probabilistic.distribution.most_likely());
    
    println!("Similarity distribution: {}", sim_different.distribution);
    
    Ok(())
}

/// Demonstrate uncertainty quantification
fn demo_uncertainty_quantification() -> Result<()> {
    println!("\n📊 3. Uncertainty Quantification");
    println!("--------------------------------");
    
    // Create a probabilistic value and demonstrate operations
    let molecular_weight = ProbabilisticValue::new_normal(46.07, 0.5, 0.95); // Ethanol MW
    let logp = ProbabilisticValue::new_normal(-0.31, 0.2, 0.90); // Ethanol LogP
    
    println!("Molecular Weight: {:.2}±{:.2}", molecular_weight.mean, molecular_weight.std_dev);
    println!("LogP: {:.3}±{:.3}", logp.mean, logp.std_dev);
    
    // Show confidence intervals
    let mw_ci = molecular_weight.confidence_interval();
    let logp_ci = logp.confidence_interval();
    
    println!("MW 95% CI: [{:.2}, {:.2}]", mw_ci.lower_bound, mw_ci.upper_bound);
    println!("LogP 90% CI: [{:.3}, {:.3}]", logp_ci.lower_bound, logp_ci.upper_bound);
    
    // Demonstrate uncertainty propagation
    let combined = molecular_weight.combine_with(&logp, |mw, lp| mw * lp.abs(), 1000)?;
    println!("Combined property: {:.2}±{:.2}", combined.mean, combined.std_dev);
    
    // Show probability calculations
    let prob_heavy = molecular_weight.probability_greater_than(50.0);
    let prob_hydrophilic = logp.probability_greater_than(0.0);
    
    println!("P(MW > 50): {:.3}", prob_heavy);
    println!("P(LogP > 0): {:.3}", prob_hydrophilic);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_functionality() {
        // Test that main functions don't panic
        assert!(demo_molecular_representation().is_ok());
        assert!(demo_similarity_calculation().is_ok());
        assert!(demo_uncertainty_quantification().is_ok());
    }
} 