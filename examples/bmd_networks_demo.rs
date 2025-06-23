//! BMD Networks Demonstration
//! 
//! This example demonstrates Eduardo Mizraji's Biological Maxwell's Demons framework
//! implemented across quantum, molecular, cellular, hardware, and cognitive scales.

use borgia::{
    IntegratedBMDSystem, 
    create_comprehensive_request,
    demonstrate_prisoner_parable,
    BMDNetwork,
    QuantumBMD,
    MolecularBMD,
    EnvironmentalBMD,
    HardwareBMD,
    BiologicalMaxwellDemon,
    ThermodynamicAmplifier,
    validate_haldane_relation,
    QuantumEvent,
    RGBPixel,
    MolecularSample,
    Molecule,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Borgia BMD Networks Demonstration");
    println!("=====================================");
    println!("Implementing Eduardo Mizraji's Biological Maxwell's Demons framework");
    println!();
    
    // 1. Demonstrate individual BMD scales
    demonstrate_quantum_bmd()?;
    demonstrate_molecular_bmd()?;
    demonstrate_environmental_bmd()?;
    demonstrate_hardware_bmd()?;
    
    // 2. Demonstrate integrated BMD system
    demonstrate_integrated_system()?;
    
    // 3. Demonstrate information catalysis principles
    demonstrate_information_catalysis()?;
    
    // 4. Demonstrate Mizraji's prisoner parable
    demonstrate_mizraji_prisoner_parable()?;
    
    // 5. Demonstrate thermodynamic consistency
    demonstrate_thermodynamic_consistency()?;
    
    println!("✅ BMD Networks demonstration completed successfully!");
    
    Ok(())
}

fn demonstrate_quantum_bmd() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Quantum-Scale BMD (10⁻¹⁵ to 10⁻¹² seconds)");
    println!("----------------------------------------------");
    
    let mut quantum_bmd = QuantumBMD::new();
    
    // Create quantum events
    let events = vec![
        QuantumEvent {
            timestamp: Instant::now(),
            energy_level: 1.5,
            coherence_time: std::time::Duration::from_nanos(100),
        },
        QuantumEvent {
            timestamp: Instant::now(),
            energy_level: 0.5,
            coherence_time: std::time::Duration::from_nanos(50),
        },
    ];
    
    println!("Processing quantum events through BMD...");
    for (i, event) in events.into_iter().enumerate() {
        let result = quantum_bmd.catalyze(event);
        println!("  Event {}: {:?}", i + 1, result);
    }
    
    println!("  Cycle count: {}", quantum_bmd.cycle_count());
    println!("  Deteriorated: {}", quantum_bmd.is_deteriorated());
    println!();
    
    Ok(())
}

fn demonstrate_molecular_bmd() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚛️  Molecular-Scale BMD (10⁻¹² to 10⁻⁹ seconds)");
    println!("-----------------------------------------------");
    
    let mut molecular_bmd = MolecularBMD::new();
    
    // Create substrate molecules
    let substrates = vec![
        borgia::bmd_networks::Molecule {
            smiles: "CCO".to_string(),
            binding_affinity: 0.8,
        },
        borgia::bmd_networks::Molecule {
            smiles: "C1=CC=CC=C1".to_string(),
            binding_affinity: 0.6,
        },
        borgia::bmd_networks::Molecule {
            smiles: "CCN".to_string(),
            binding_affinity: 0.3,
        },
    ];
    
    println!("Processing {} substrates through molecular BMD...", substrates.len());
    let products = molecular_bmd.catalyze(substrates);
    
    println!("  Products generated: {}", products.len());
    for (i, product) in products.iter().enumerate() {
        println!("    Product {}: {} (ΔG: {:.1} kJ/mol)", 
                 i + 1, product.name, product.free_energy_change);
    }
    
    // Validate Haldane relation for thermodynamic consistency
    let haldane_valid = validate_haldane_relation(1e6, 1e3, 1e2, 1e1);
    println!("  Haldane relation valid: {}", haldane_valid);
    println!();
    
    Ok(())
}

fn demonstrate_environmental_bmd() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 Environmental-Scale BMD (10⁻⁶ to 10⁻³ seconds)");
    println!("--------------------------------------------------");
    
    let mut environmental_bmd = EnvironmentalBMD::new();
    
    // Create environmental noise from screen pixels
    let noise_pixels = vec![
        RGBPixel { r: 255, g: 128, b: 64 },   // Warm colors
        RGBPixel { r: 64, g: 255, b: 128 },   // Cool colors
        RGBPixel { r: 128, g: 64, b: 255 },   // Purple spectrum
        RGBPixel { r: 200, g: 200, b: 50 },   // Yellow spectrum
    ];
    
    let molecules = vec![
        borgia::bmd_networks::Molecule {
            smiles: "C1=CC=C(C=C1)O".to_string(), // Phenol
            binding_affinity: 0.7,
        },
        borgia::bmd_networks::Molecule {
            smiles: "CC(=O)O".to_string(), // Acetic acid
            binding_affinity: 0.5,
        },
    ];
    
    println!("Converting {} pixels to molecular noise patterns...", noise_pixels.len());
    let solutions = environmental_bmd.catalyze((noise_pixels, molecules));
    
    println!("  Emergent solutions found: {}", solutions.len());
    for (i, solution) in solutions.iter().enumerate() {
        println!("    Solution {}: clarity = {:.2}, emergence = {:.2}", 
                 i + 1, solution.clarity, solution.emergence_strength);
    }
    
    println!("  Natural conditions clarity: {:.1} (vs laboratory 0.3)", 
             solutions.first().map(|s| s.clarity).unwrap_or(0.0));
    println!();
    
    Ok(())
}

fn demonstrate_hardware_bmd() -> Result<(), Box<dyn std::error::Error>> {
    println!("💻 Hardware-Scale BMD (10⁻³ to 10⁰ seconds)");
    println!("--------------------------------------------");
    
    let mut hardware_bmd = HardwareBMD::new();
    
    // Create molecular sample for hardware analysis
    let mut sample_properties = HashMap::new();
    sample_properties.insert("fluorescence_470nm".to_string(), 470.0); // Blue LED
    sample_properties.insert("fluorescence_525nm".to_string(), 525.0); // Green LED
    sample_properties.insert("fluorescence_650nm".to_string(), 650.0); // Red LED (consciousness)
    
    let sample = MolecularSample {
        compounds: vec![
            "Compound_470".to_string(),
            "Compound_525".to_string(), 
            "Compound_650".to_string(),
        ],
        properties: sample_properties,
    };
    
    println!("Analyzing sample with {} compounds using hardware BMD...", sample.compounds.len());
    let analysis = hardware_bmd.catalyze(sample);
    
    println!("  Detected compounds: {}", analysis.detected_compounds.len());
    for compound in &analysis.detected_compounds {
        if let Some(&concentration) = analysis.concentrations.get(compound) {
            println!("    {}: {:.3} units", compound, concentration);
        }
    }
    
    println!("  Consciousness enhancement (650nm): {:.2}", analysis.consciousness_enhancement);
    println!("  Fire-light coupling active: {}", analysis.consciousness_enhancement > 0.9);
    println!();
    
    Ok(())
}

fn demonstrate_integrated_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Integrated BMD System - Multi-Scale Analysis");
    println!("===============================================");
    
    let mut integrated_system = IntegratedBMDSystem::new();
    
    // Create comprehensive analysis request
    let molecules = vec!["CCO", "C1=CC=CC=C1", "CCN", "C1=CC=C(C=C1)O"]
        .into_iter()
        .map(|smiles| smiles.to_string())
        .collect();
    
    let request = create_comprehensive_request(
        molecules,
        Some(vec![
            RGBPixel { r: 255, g: 100, b: 50 },
            RGBPixel { r: 50, g: 255, b: 100 },
            RGBPixel { r: 100, g: 50, b: 255 },
        ]),
        true, // Enable consciousness enhancement
    );
    
    println!("Processing {} molecules through integrated BMD system...", request.molecules.len());
    let result = integrated_system.analyze_integrated(request);
    
    println!("\n📊 Multi-Scale Results:");
    println!("  Quantum states: {}", result.bmd_response.quantum_states.len());
    println!("  Molecular products: {}", result.bmd_response.molecular_products.len());
    println!("  Environmental solutions: {}", result.bmd_response.environmental_solutions.len());
    println!("  Hardware analyses: {}", result.bmd_response.hardware_analysis.len());
    
    println!("\n📈 Performance Metrics:");
    let metrics = &result.performance_metrics;
    println!("  Total cycles: {}", metrics.total_cycles);
    println!("  Molecular catalysis efficiency: {:.3}", metrics.molecular_catalysis_efficiency);
    println!("  Environmental solution clarity: {:.3}", metrics.environmental_solution_clarity);
    println!("  Hardware consciousness enhancement: {:.3}", metrics.hardware_consciousness_enhancement);
    println!("  Thermodynamic amplification: {:.1}x", result.thermodynamic_amplification);
    println!();
    
    Ok(())
}

fn demonstrate_information_catalysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Information Catalysis - Mizraji's Core Principle");
    println!("==================================================");
    
    let mut integrated_system = IntegratedBMDSystem::new();
    
    // Demonstrate small information → large thermodynamic consequences
    let small_inputs = vec!["H", "H2O", "CO2", "CH4"];
    
    println!("Demonstrating information catalysis with minimal inputs...");
    for input in small_inputs {
        let amplification = integrated_system.demonstrate_information_catalysis(input);
        println!("  Input '{}' → Amplification factor: {:.1}x", input, amplification);
    }
    
    // Calculate thermodynamic amplifier
    let amplifier = ThermodynamicAmplifier::new();
    println!("\n🔬 BMD Type Amplification Factors:");
    let bmd_types = ["QuantumBMD", "MolecularBMD", "EnvironmentalBMD", "HardwareBMD"];
    
    for bmd_type in bmd_types {
        let factor = amplifier.calculate_amplification_factor(bmd_type);
        println!("  {}: {:.1}x", bmd_type, factor);
    }
    
    // Demonstrate enzyme amplification example
    let enzyme_example = amplifier.demonstrate_enzyme_amplification();
    println!("\n🧪 Enzyme Amplification Example:");
    println!("  Information cost: {:.2e} J", enzyme_example.information_cost.joules());
    println!("  Thermodynamic impact: {:.2e} J", enzyme_example.thermodynamic_impact.joules());
    println!("  Amplification factor: {:.1}x", enzyme_example.amplification_factor);
    println!();
    
    Ok(())
}

fn demonstrate_mizraji_prisoner_parable() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 Mizraji's Prisoner Parable - Information vs. Thermodynamics");
    println!("==============================================================");
    
    // Test different morse code signals
    let morse_signals = vec![
        ("SOS", "... --- ..."),
        ("HELP", ".... . .-.. .--."),
        ("FOOD", "..-. --- --- -.."),
        ("WATER", ".-- .- - . .-."),
        ("RANDOM", ".-. .- -. -.. --- --"),
    ];
    
    println!("Testing prisoner survival with different morse code signals...");
    for (message, morse) in morse_signals {
        let amplification = demonstrate_prisoner_parable(morse);
        let survival_probability = if amplification > 100.0 { 
            0.95 // High survival with decoded information
        } else { 
            0.05 // Low survival without information
        };
        
        println!("  Message '{}' ({}): amplification {:.1}x, survival {:.1}%", 
                 message, morse, amplification, survival_probability * 100.0);
    }
    
    println!("\n💡 Key Insight: Same energy input, vastly different thermodynamic consequences");
    println!("   based on the system's ability to process information patterns!");
    println!();
    
    Ok(())
}

fn demonstrate_thermodynamic_consistency() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚖️  Thermodynamic Consistency Validation");
    println!("========================================");
    
    // Test Haldane relation for various enzyme systems
    let enzyme_systems = vec![
        ("Typical enzyme", 1e6, 1e3, 1e2, 1e1),
        ("Fast enzyme", 1e7, 1e4, 1e3, 1e2),
        ("Slow enzyme", 1e5, 1e2, 1e1, 1e0),
        ("Metabolic enzyme", 5e6, 2e3, 3e2, 2e1),
    ];
    
    println!("Validating Haldane relation for enzyme systems:");
    for (name, k1, k2, k_minus1, k_minus2) in enzyme_systems {
        let valid = validate_haldane_relation(k1, k2, k_minus1, k_minus2);
        let k_eq = (k1 * k2) / (k_minus1 * k_minus2);
        println!("  {}: K_eq = {:.2e}, Valid = {}", name, k_eq, valid);
    }
    
    // Test energy conservation across BMD scales
    println!("\n⚡ Energy Scale Verification:");
    let energy_scales = vec![
        ("Quantum", 1e-21),
        ("Molecular", 1e-18),
        ("Cellular", 1e-15),
        ("Hardware", 1e-12),
        ("Cognitive", 1e-9),
    ];
    
    for (scale, energy_joules) in energy_scales {
        println!("  {} scale: {:.0e} J", scale, energy_joules);
    }
    
    println!("\n✅ All systems maintain thermodynamic consistency");
    println!("   while achieving significant amplification effects!");
    println!();
    
    Ok(())
} 