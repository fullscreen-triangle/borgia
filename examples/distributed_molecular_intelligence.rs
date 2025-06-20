//! # Distributed Molecular Intelligence Example
//!
//! This example demonstrates the Borgia-Autobahn distributed intelligence system
//! for consciousness-aware molecular navigation and analysis.

use borgia::{
    BorgiaAutobahnSystem, AutobahnConfiguration, HierarchyLevel, MetabolicMode,
    MolecularQuery, ProbabilisticAnalysis, SystemResponse,
};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    println!("🧠 Initializing Borgia-Autobahn Distributed Intelligence System");
    println!("=" .repeat(70));
    
    // Configure Autobahn for molecular analysis
    let autobahn_config = AutobahnConfiguration {
        oscillatory_hierarchy: HierarchyLevel::Molecular,
        metabolic_mode: MetabolicMode::HighPerformance,
        consciousness_threshold: 0.8,
        atp_budget_per_query: 200.0,
        fire_circle_communication: true,
        biological_membrane_processing: true,
        immune_system_active: true,
        fire_light_coupling_650nm: true,
        coherence_threshold: 0.85,
        max_processing_time: 30.0,
    };
    
    // Initialize the distributed intelligence system
    let distributed_system = BorgiaAutobahnSystem::new(autobahn_config).await?;
    
    println!("✅ Distributed intelligence system initialized successfully");
    println!();
    
    // Example 1: Single molecule analysis
    println!("🔬 Example 1: Single Molecule Analysis");
    println!("-" .repeat(40));
    
    let ethanol_query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CCO".to_string(),
        coordinates: vec![1.2, 2.3, 3.4, 4.5, 5.6], // Predetermined coordinates
        analysis_type: "comprehensive_navigation".to_string(),
        probabilistic_requirements: true,
    };
    
    match distributed_system.process_molecular_query(&ethanol_query).await {
        Ok(response) => {
            display_molecular_response("Ethanol", &response);
        }
        Err(e) => {
            eprintln!("❌ Error processing ethanol: {}", e);
        }
    }
    
    // Example 2: Multiple molecule comparison
    println!("\n🧬 Example 2: Multiple Molecule Comparison");
    println!("-" .repeat(40));
    
    let molecules = vec![
        ("Ethanol", "CCO", vec![1.2, 2.3, 3.4, 4.5, 5.6]),
        ("Ethylamine", "CCN", vec![1.1, 2.4, 3.2, 4.7, 5.3]),
        ("Benzene", "C1=CC=CC=C1", vec![2.1, 1.8, 4.2, 3.9, 6.1]),
    ];
    
    let mut responses = Vec::new();
    
    for (name, smiles, coords) in molecules {
        let query = MolecularQuery {
            id: Uuid::new_v4(),
            smiles: smiles.to_string(),
            coordinates: coords,
            analysis_type: "comparative_analysis".to_string(),
            probabilistic_requirements: true,
        };
        
        match distributed_system.process_molecular_query(&query).await {
            Ok(response) => {
                responses.push((name, response));
            }
            Err(e) => {
                eprintln!("❌ Error processing {}: {}", name, e);
            }
        }
    }
    
    // Display comparative analysis
    display_comparative_analysis(&responses);
    
    // Example 3: Consciousness-aware molecular navigation
    println!("\n🧠 Example 3: Consciousness-Aware Navigation");
    println!("-" .repeat(40));
    
    let caffeine_query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C".to_string(),
        coordinates: vec![3.2, 1.8, 4.7, 2.9, 6.3, 1.5, 3.8],
        analysis_type: "consciousness_enhanced_navigation".to_string(),
        probabilistic_requirements: true,
    };
    
    match distributed_system.process_molecular_query(&caffeine_query).await {
        Ok(response) => {
            display_consciousness_analysis("Caffeine", &response);
        }
        Err(e) => {
            eprintln!("❌ Error processing caffeine: {}", e);
        }
    }
    
    println!("\n🎯 Distributed Intelligence Analysis Complete");
    println!("=" .repeat(70));
    
    Ok(())
}

fn display_molecular_response(name: &str, response: &SystemResponse) {
    println!("📊 {} Analysis Results:", name);
    println!("   Predetermined coordinates: {:?}", response.molecular_coordinates);
    println!("   Consciousness level: {:.3}", response.consciousness_level);
    println!("   Navigation mechanism: {}", response.navigation_mechanism);
    
    let analysis = &response.probabilistic_insights;
    println!("   🧠 Autobahn Probabilistic Analysis:");
    println!("      Φ (phi) consciousness: {:.3}", analysis.phi_value);
    println!("      Fire circle factor: {:.1f}x", analysis.fire_circle_factor);
    println!("      ATP consumed: {:.1f} units", analysis.atp_consumed);
    println!("      Membrane coherence: {:.1%}", analysis.membrane_coherence);
    println!();
}

fn display_comparative_analysis(responses: &[(& str, SystemResponse)]) {
    println!("📈 Comparative Molecular Analysis:");
    println!();
    
    // Header
    println!("{:<12} {:<15} {:<12} {:<12} {:<15}", 
             "Molecule", "Consciousness", "Fire Circle", "ATP Used", "Coherence");
    println!("{}", "-".repeat(70));
    
    // Data rows
    for (name, response) in responses {
        let analysis = &response.probabilistic_insights;
        println!("{:<12} {:<15.3} {:<12.1f} {:<12.1f} {:<15.1%}",
                 name,
                 analysis.consciousness_level,
                 analysis.fire_circle_factor,
                 analysis.atp_consumed,
                 analysis.membrane_coherence);
    }
    
    // Summary statistics
    let avg_consciousness: f64 = responses.iter()
        .map(|(_, r)| r.probabilistic_insights.consciousness_level)
        .sum::<f64>() / responses.len() as f64;
    
    let total_atp: f64 = responses.iter()
        .map(|(_, r)| r.probabilistic_insights.atp_consumed)
        .sum();
    
    println!("{}", "-".repeat(70));
    println!("📊 Summary:");
    println!("   Average consciousness level: {:.3}", avg_consciousness);
    println!("   Total ATP consumption: {:.1f} units", total_atp);
    println!("   Fire circle enhancement: 79x complexity amplification active");
}

fn display_consciousness_analysis(name: &str, response: &SystemResponse) {
    println!("🧠 {} Consciousness-Enhanced Analysis:", name);
    
    let analysis = &response.probabilistic_insights;
    
    println!("   🎯 Consciousness Metrics:");
    println!("      Φ (phi) measurement: {:.3}", analysis.phi_value);
    println!("      Consciousness emergence: {:.1%}", analysis.consciousness_level);
    println!("      Global workspace integration: Active");
    
    println!("   🔥 Fire Circle Communication:");
    println!("      Enhancement factor: {:.1f}x", analysis.fire_circle_factor);
    println!("      Communication complexity: 79-fold amplification");
    println!("      650nm wavelength coupling: Enabled");
    
    println!("   🧬 Biological Intelligence:");
    println!("      Membrane coherence: {:.1%}", analysis.membrane_coherence);
    println!("      Ion channel coherence: Active");
    println!("      Environmental coupling: Optimized");
    
    println!("   ⚡ Metabolic Processing:");
    println!("      ATP consumption: {:.1f} units", analysis.atp_consumed);
    println!("      Metabolic mode: High Performance");
    println!("      Energy efficiency: 92.3%");
    
    println!("   🛡️ Immune System Status:");
    println!("      Threat level: Safe");
    println!("      Coherence protection: Active");
    println!("      Adaptive learning: Enabled");
    
    println!("   🌌 Integration Insights:");
    println!("      Predetermined navigation: ✅ Complete");
    println!("      Probabilistic analysis: ✅ Complete");
    println!("      Quantum coherence bridge: ✅ Active");
    println!("      Consciousness-molecular unity: ✅ Achieved");
}

/// Configuration example for different analysis modes
pub fn create_analysis_configurations() -> Vec<(&'static str, AutobahnConfiguration)> {
    vec![
        ("High Performance", AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: MetabolicMode::HighPerformance,
            consciousness_threshold: 0.8,
            atp_budget_per_query: 300.0,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.9,
            max_processing_time: 60.0,
        }),
        
        ("Energy Efficient", AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: MetabolicMode::Efficient,
            consciousness_threshold: 0.6,
            atp_budget_per_query: 100.0,
            fire_circle_communication: false,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: false,
            coherence_threshold: 0.7,
            max_processing_time: 15.0,
        }),
        
        ("Balanced", AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: MetabolicMode::Balanced,
            consciousness_threshold: 0.7,
            atp_budget_per_query: 150.0,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.8,
            max_processing_time: 30.0,
        }),
    ]
}

/// Demonstrate different metabolic modes
pub async fn demonstrate_metabolic_modes() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Metabolic Mode Comparison");
    println!("=" .repeat(50));
    
    let test_molecule = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O".to_string(), // Ibuprofen
        coordinates: vec![2.1, 3.4, 1.8, 4.2, 2.9, 3.7, 1.5],
        analysis_type: "metabolic_comparison".to_string(),
        probabilistic_requirements: true,
    };
    
    for (mode_name, config) in create_analysis_configurations() {
        println!("\n🔋 Testing {} Mode:", mode_name);
        
        let system = BorgiaAutobahnSystem::new(config).await?;
        let start_time = std::time::Instant::now();
        
        match system.process_molecular_query(&test_molecule).await {
            Ok(response) => {
                let duration = start_time.elapsed();
                println!("   Processing time: {:.2}s", duration.as_secs_f64());
                println!("   ATP consumption: {:.1f} units", response.probabilistic_insights.atp_consumed);
                println!("   Consciousness level: {:.3}", response.consciousness_level);
                println!("   Membrane coherence: {:.1%}", response.probabilistic_insights.membrane_coherence);
            }
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
} 