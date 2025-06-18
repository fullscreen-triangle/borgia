//! Borgia - Revolutionary Probabilistic Cheminformatics Engine
//! 
//! Main executable demonstrating the core functionality.

use borgia::{
    BorgiaError, Result,
    ProbabilisticMolecule, EnhancedFingerprint,
    SimilarityEngine, SimilarityAlgorithm,
    ProbabilisticValue, BayesianInference,
    EvidenceProcessor, EvidenceContext, EvidenceType, UpstreamSystem,
    IntegrationManager, HegelIntegration, LavoisierIntegration,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧬 Borgia - Revolutionary Probabilistic Cheminformatics Engine");
    println!("================================================================");
    
    // Demonstrate basic functionality
    demo_molecular_representation()?;
    demo_similarity_calculation()?;
    demo_uncertainty_quantification()?;
    demo_enhanced_fingerprints()?;
    demo_bayesian_inference()?;
    demo_evidence_processing()?;
    demo_integration_systems().await?;
    
    println!("\n✅ All demonstrations completed successfully!");
    println!("🎯 Borgia is ready for revolutionary probabilistic cheminformatics!");
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

/// Demonstrate enhanced molecular fingerprints
fn demo_enhanced_fingerprints() -> Result<()> {
    println!("\n🔍 4. Enhanced Molecular Fingerprints");
    println!("-------------------------------------");
    
    let fp1 = EnhancedFingerprint::from_smiles("c1ccccc1CCO")?; // Benzyl alcohol
    let fp2 = EnhancedFingerprint::from_smiles("c1ccccc1CCC")?; // Propylbenzene
    let fp3 = EnhancedFingerprint::from_smiles("CCO")?; // Ethanol
    
    println!("Fingerprint dimensions: {}", fp1.dimension());
    println!("Benzyl alcohol density: {:.4}", fp1.density());
    println!("Propylbenzene density: {:.4}", fp2.density());
    println!("Ethanol density: {:.4}", fp3.density());
    
    let sim_aromatic = fp1.tanimoto_similarity(&fp2);
    let sim_different = fp1.tanimoto_similarity(&fp3);
    
    println!("Aromatic compounds similarity: {:.3}", sim_aromatic);
    println!("Aromatic vs aliphatic similarity: {:.3}", sim_different);
    
    Ok(())
}

/// Demonstrate Bayesian inference
fn demo_bayesian_inference() -> Result<()> {
    println!("\n🎯 5. Bayesian Evidence Integration");
    println!("-----------------------------------");
    
    let mut bayesian = BayesianInference::new();
    
    // Simulate evidence from different sources
    println!("Adding evidence from multiple sources:");
    bayesian.update_evidence(0.85, 0.9)?; // High confidence evidence
    println!("  - Source 1: 0.85 (confidence: 0.9)");
    
    bayesian.update_evidence(0.75, 0.7)?; // Medium confidence evidence
    println!("  - Source 2: 0.75 (confidence: 0.7)");
    
    bayesian.update_evidence(0.92, 0.95)?; // Very high confidence evidence
    println!("  - Source 3: 0.92 (confidence: 0.95)");
    
    let posterior = bayesian.calculate_posterior()?;
    println!("Posterior probability: {:.3} ± {:.3}", 
        posterior.mean, posterior.std_dev);
    
    let prediction = bayesian.predict_outcome(0.8)?;
    println!("Prediction for threshold 0.8: {:.3}", prediction);
    
    Ok(())
}

/// Demonstrate evidence processing
fn demo_evidence_processing() -> Result<()> {
    println!("\n📋 6. Evidence Processing System");
    println!("--------------------------------");
    
    let evidence_processor = EvidenceProcessor::new();
    let evidence_context = EvidenceContext::new(
        EvidenceType::StructuralSimilarity,
        UpstreamSystem::Hegel,
        "drug_discovery_pipeline".to_string(),
    );
    
    let mol1 = ProbabilisticMolecule::from_smiles("CCO")?;
    let mol2 = ProbabilisticMolecule::from_smiles("CCN")?;
    let molecules = vec![mol1, mol2];
    
    let evidence_strength = evidence_processor.assess_evidence_strength(
        &evidence_context,
        &molecules,
    )?;
    
    println!("Evidence Assessment:");
    println!("  Raw strength: {:.3}", evidence_strength.raw_strength);
    println!("  Adjusted strength: {:.3}", evidence_strength.adjusted_strength);
    println!("  Confidence: {:.3}", evidence_strength.confidence);
    println!("  Reliability: {:.3}", evidence_strength.reliability);
    println!("  Supporting factors: {:?}", evidence_strength.supporting_factors);
    
    if !evidence_strength.limiting_factors.is_empty() {
        println!("  Limiting factors: {:?}", evidence_strength.limiting_factors);
    }
    
    Ok(())
}

/// Demonstrate integration with upstream systems
async fn demo_integration_systems() -> Result<()> {
    println!("\n🔗 7. Upstream System Integration");
    println!("---------------------------------");
    
    // Set up integration systems
    let hegel = HegelIntegration::new("http://hegel-system:8080".to_string())
        .with_timeout(30);
    let lavoisier = LavoisierIntegration::new("http://lavoisier-system:8080".to_string())
        .with_timeout(45);
    
    let mut integration_manager = IntegrationManager::new()
        .with_hegel(hegel)
        .with_lavoisier(lavoisier);
    
    // Create test molecules
    let molecules = vec![
        ProbabilisticMolecule::from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")?, // Ibuprofen
        ProbabilisticMolecule::from_smiles("COC1=CC=C(C=C1)CCN")?, // 4-Methoxyphenethylamine
    ];
    
    println!("Requesting comprehensive analysis for {} molecules", molecules.len());
    
    // Request analysis from all available systems
    let responses = integration_manager
        .request_comprehensive_analysis(&molecules, "drug_discovery")
        .await?;
    
    println!("Received {} integration responses:", responses.len());
    for (i, response) in responses.iter().enumerate() {
        println!("  Response {}: {}", i + 1, response.request_id);
        println!("    Evidence strength: {:.3}", response.evidence_strength);
        println!("    Confidence: {:.3}", response.confidence);
        println!("    Processing time: {}ms", response.processing_time_ms);
        println!("    Recommendations: {:?}", response.recommendations);
        
        if !response.supporting_data.is_empty() {
            println!("    Supporting data:");
            for (key, value) in &response.supporting_data {
                println!("      {}: {}", key, value);
            }
        }
    }
    
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
        assert!(demo_enhanced_fingerprints().is_ok());
        assert!(demo_bayesian_inference().is_ok());
        assert!(demo_evidence_processing().is_ok());
    }
} 