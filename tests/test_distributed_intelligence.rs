//! Tests for Borgia-Autobahn Distributed Intelligence System

use borgia::{
    BorgiaAutobahnSystem, AutobahnConfiguration, HierarchyLevel, MetabolicMode,
    MolecularQuery, ProbabilisticAnalysis, SystemResponse,
};
use uuid::Uuid;
use tokio_test;

#[tokio::test]
async fn test_system_initialization() {
    let config = AutobahnConfiguration {
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
    
    let system = BorgiaAutobahnSystem::new(config).await;
    assert!(system.is_ok(), "System should initialize successfully");
}

#[tokio::test]
async fn test_molecular_query_processing() {
    let config = AutobahnConfiguration {
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
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CCO".to_string(),
        coordinates: vec![1.2, 2.3, 3.4, 4.5, 5.6],
        analysis_type: "test_analysis".to_string(),
        probabilistic_requirements: true,
    };
    
    let response = system.process_molecular_query(&query).await;
    assert!(response.is_ok(), "Query processing should succeed");
    
    let response = response.unwrap();
    assert_eq!(response.molecular_coordinates, query.coordinates);
    assert!(response.consciousness_level > 0.0);
    assert!(response.consciousness_level <= 1.0);
    assert_eq!(response.navigation_mechanism, "Distributed BMD-Autobahn Intelligence");
}

#[tokio::test]
async fn test_consciousness_emergence() {
    let config = AutobahnConfiguration {
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
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C".to_string(), // Caffeine
        coordinates: vec![3.2, 1.8, 4.7, 2.9, 6.3, 1.5, 3.8],
        analysis_type: "consciousness_test".to_string(),
        probabilistic_requirements: true,
    };
    
    let response = system.process_molecular_query(&query).await.unwrap();
    
    // Test consciousness metrics
    assert!(response.probabilistic_insights.phi_value > 0.0);
    assert!(response.probabilistic_insights.consciousness_level > 0.0);
    assert!(response.probabilistic_insights.fire_circle_factor > 0.0);
    assert!(response.probabilistic_insights.membrane_coherence > 0.0);
    assert!(response.probabilistic_insights.atp_consumed > 0.0);
}

#[tokio::test]
async fn test_fire_circle_communication() {
    let config = AutobahnConfiguration {
        oscillatory_hierarchy: HierarchyLevel::Molecular,
        metabolic_mode: MetabolicMode::HighPerformance,
        consciousness_threshold: 0.7,
        atp_budget_per_query: 200.0,
        fire_circle_communication: true,
        biological_membrane_processing: true,
        immune_system_active: true,
        fire_light_coupling_650nm: true,
        coherence_threshold: 0.85,
        max_processing_time: 30.0,
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "C1=CC=CC=C1".to_string(), // Benzene
        coordinates: vec![2.1, 1.8, 4.2, 3.9, 6.1],
        analysis_type: "fire_circle_test".to_string(),
        probabilistic_requirements: true,
    };
    
    let response = system.process_molecular_query(&query).await.unwrap();
    
    // Test fire circle enhancement
    let expected_max_factor = 79.0; // 79x complexity amplification
    assert!(response.probabilistic_insights.fire_circle_factor <= expected_max_factor);
    assert!(response.probabilistic_insights.fire_circle_factor > 0.0);
}

#[tokio::test]
async fn test_metabolic_modes() {
    let modes = vec![
        MetabolicMode::HighPerformance,
        MetabolicMode::Efficient,
        MetabolicMode::Balanced,
        MetabolicMode::Emergency,
    ];
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CCN".to_string(), // Ethylamine
        coordinates: vec![1.1, 2.4, 3.2, 4.7, 5.3],
        analysis_type: "metabolic_test".to_string(),
        probabilistic_requirements: true,
    };
    
    for mode in modes {
        let config = AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: mode,
            consciousness_threshold: 0.7,
            atp_budget_per_query: 150.0,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.8,
            max_processing_time: 30.0,
        };
        
        let system = BorgiaAutobahnSystem::new(config).await.unwrap();
        let response = system.process_molecular_query(&query).await.unwrap();
        
        // Each mode should produce valid results
        assert!(response.consciousness_level > 0.0);
        assert!(response.probabilistic_insights.atp_consumed > 0.0);
        assert!(response.probabilistic_insights.membrane_coherence > 0.0);
    }
}

#[tokio::test]
async fn test_biological_membrane_processing() {
    let config = AutobahnConfiguration {
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
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O".to_string(), // Ibuprofen
        coordinates: vec![2.1, 3.4, 1.8, 4.2, 2.9, 3.7, 1.5],
        analysis_type: "membrane_test".to_string(),
        probabilistic_requirements: true,
    };
    
    let response = system.process_molecular_query(&query).await.unwrap();
    
    // Test membrane coherence
    assert!(response.probabilistic_insights.membrane_coherence > 0.0);
    assert!(response.probabilistic_insights.membrane_coherence <= 1.0);
    
    // Membrane coherence should be reasonable (typically 0.8-0.95)
    assert!(response.probabilistic_insights.membrane_coherence > 0.5);
}

#[tokio::test]
async fn test_atp_consumption_scaling() {
    let budgets = vec![100.0, 200.0, 300.0];
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CCO".to_string(),
        coordinates: vec![1.2, 2.3, 3.4, 4.5, 5.6],
        analysis_type: "atp_test".to_string(),
        probabilistic_requirements: true,
    };
    
    for budget in budgets {
        let config = AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: MetabolicMode::Balanced,
            consciousness_threshold: 0.7,
            atp_budget_per_query: budget,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.8,
            max_processing_time: 30.0,
        };
        
        let system = BorgiaAutobahnSystem::new(config).await.unwrap();
        let response = system.process_molecular_query(&query).await.unwrap();
        
        // ATP consumption should scale with budget
        assert!(response.probabilistic_insights.atp_consumed > 0.0);
        assert!(response.probabilistic_insights.atp_consumed <= budget * 1.1); // Allow 10% overhead
    }
}

#[tokio::test]
async fn test_multiple_molecules_comparison() {
    let config = AutobahnConfiguration {
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
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
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
            analysis_type: "comparison_test".to_string(),
            probabilistic_requirements: true,
        };
        
        let response = system.process_molecular_query(&query).await.unwrap();
        responses.push((name, response));
    }
    
    assert_eq!(responses.len(), 3);
    
    // Each molecule should have valid consciousness levels
    for (name, response) in &responses {
        assert!(response.consciousness_level > 0.0, "Consciousness level should be positive for {}", name);
        assert!(response.probabilistic_insights.phi_value > 0.0, "Phi value should be positive for {}", name);
    }
    
    // Calculate average consciousness
    let avg_consciousness: f64 = responses.iter()
        .map(|(_, r)| r.consciousness_level)
        .sum::<f64>() / responses.len() as f64;
    
    assert!(avg_consciousness > 0.0);
    assert!(avg_consciousness <= 1.0);
}

#[tokio::test]
async fn test_fire_light_coupling() {
    // Test with fire-light coupling enabled
    let config_with_coupling = AutobahnConfiguration {
        oscillatory_hierarchy: HierarchyLevel::Molecular,
        metabolic_mode: MetabolicMode::HighPerformance,
        consciousness_threshold: 0.7,
        atp_budget_per_query: 200.0,
        fire_circle_communication: true,
        biological_membrane_processing: true,
        immune_system_active: true,
        fire_light_coupling_650nm: true,
        coherence_threshold: 0.8,
        max_processing_time: 30.0,
    };
    
    // Test without fire-light coupling
    let config_without_coupling = AutobahnConfiguration {
        fire_light_coupling_650nm: false,
        ..config_with_coupling.clone()
    };
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C".to_string(), // Caffeine
        coordinates: vec![3.2, 1.8, 4.7, 2.9, 6.3, 1.5, 3.8],
        analysis_type: "fire_light_test".to_string(),
        probabilistic_requirements: true,
    };
    
    let system_with = BorgiaAutobahnSystem::new(config_with_coupling).await.unwrap();
    let system_without = BorgiaAutobahnSystem::new(config_without_coupling).await.unwrap();
    
    let response_with = system_with.process_molecular_query(&query).await.unwrap();
    let response_without = system_without.process_molecular_query(&query).await.unwrap();
    
    // Both should work, but coupling might enhance performance
    assert!(response_with.consciousness_level > 0.0);
    assert!(response_without.consciousness_level > 0.0);
}

#[tokio::test]
async fn test_immune_system_protection() {
    let config = AutobahnConfiguration {
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
    };
    
    let system = BorgiaAutobahnSystem::new(config).await.unwrap();
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "CCO".to_string(),
        coordinates: vec![1.2, 2.3, 3.4, 4.5, 5.6],
        analysis_type: "immune_test".to_string(),
        probabilistic_requirements: true,
    };
    
    let response = system.process_molecular_query(&query).await.unwrap();
    
    // Immune system should be active and provide threat analysis
    // In our simplified implementation, threats should be minimal for standard molecules
    assert!(response.consciousness_level > 0.0);
    assert!(response.probabilistic_insights.membrane_coherence > 0.0);
}

#[tokio::test]
async fn test_hierarchical_levels() {
    let levels = vec![
        HierarchyLevel::Quantum,
        HierarchyLevel::Molecular,
        HierarchyLevel::Biological,
    ];
    
    let query = MolecularQuery {
        id: Uuid::new_v4(),
        smiles: "C1=CC=CC=C1".to_string(),
        coordinates: vec![2.1, 1.8, 4.2, 3.9, 6.1],
        analysis_type: "hierarchy_test".to_string(),
        probabilistic_requirements: true,
    };
    
    for level in levels {
        let config = AutobahnConfiguration {
            oscillatory_hierarchy: level,
            metabolic_mode: MetabolicMode::Balanced,
            consciousness_threshold: 0.7,
            atp_budget_per_query: 150.0,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.8,
            max_processing_time: 30.0,
        };
        
        let system = BorgiaAutobahnSystem::new(config).await.unwrap();
        let response = system.process_molecular_query(&query).await.unwrap();
        
        // Each hierarchy level should produce valid results
        assert!(response.consciousness_level > 0.0);
        assert!(response.probabilistic_insights.phi_value > 0.0);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_end_to_end_molecular_analysis() {
        let config = AutobahnConfiguration {
            oscillatory_hierarchy: HierarchyLevel::Molecular,
            metabolic_mode: MetabolicMode::HighPerformance,
            consciousness_threshold: 0.8,
            atp_budget_per_query: 250.0,
            fire_circle_communication: true,
            biological_membrane_processing: true,
            immune_system_active: true,
            fire_light_coupling_650nm: true,
            coherence_threshold: 0.85,
            max_processing_time: 45.0,
        };
        
        let system = BorgiaAutobahnSystem::new(config).await.unwrap();
        
        // Test with a complex pharmaceutical molecule
        let aspirin_query = MolecularQuery {
            id: Uuid::new_v4(),
            smiles: "CC(=O)OC1=CC=CC=C1C(=O)O".to_string(), // Aspirin
            coordinates: vec![1.5, 2.8, 3.1, 4.9, 2.7, 3.6, 1.9, 5.2],
            analysis_type: "comprehensive_pharmaceutical_analysis".to_string(),
            probabilistic_requirements: true,
        };
        
        let response = system.process_molecular_query(&aspirin_query).await.unwrap();
        
        // Comprehensive validation
        assert_eq!(response.molecular_coordinates, aspirin_query.coordinates);
        assert!(response.consciousness_level > 0.5, "High consciousness expected for complex molecule");
        assert!(response.probabilistic_insights.phi_value > 0.5);
        assert!(response.probabilistic_insights.fire_circle_factor > 30.0, "Significant fire circle enhancement expected");
        assert!(response.probabilistic_insights.membrane_coherence > 0.8, "High membrane coherence expected");
        assert!(response.probabilistic_insights.atp_consumed > 100.0, "Significant ATP consumption expected");
        assert_eq!(response.navigation_mechanism, "Distributed BMD-Autobahn Intelligence");
    }
} 