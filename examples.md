---
layout: page
title: "Examples"
permalink: /examples/
---

# Examples

This page provides comprehensive examples demonstrating the capabilities of the Borgia framework across different scientific applications.

## Basic Usage Examples

### 1. Simple Molecular Analysis

```rust
use borgia::{initialize, Molecule, BMDScale};

fn main() -> borgia::BorgiaResult<()> {
    // Initialize the Borgia framework
    let mut system = initialize()?;
    
    // Create molecules from SMILES
    let ethanol = Molecule::from_smiles("CCO")?;
    let acetic_acid = Molecule::from_smiles("CC(=O)O")?;
    
    // Perform basic molecular analysis
    let molecules = vec![ethanol.smiles, acetic_acid.smiles];
    let result = system.execute_cross_scale_analysis(
        molecules,
        vec![BMDScale::Molecular]
    )?;
    
    println!("Molecular analysis completed:");
    println!("  Amplification factor: {:.2}×", result.amplification_factor);
    println!("  Thermodynamic consistency: {}", result.thermodynamic_consistency);
    
    Ok(())
}
```

### 2. Multi-Scale Coordination

```rust
use borgia::{initialize, BMDScale};

fn main() -> borgia::BorgiaResult<()> {
    let mut system = initialize()?;
    
    // Define a small molecular dataset
    let molecules = vec![
        "C6H12O6".to_string(),    // Glucose
        "C8H10N4O2".to_string(),  // Caffeine
        "C9H8O4".to_string(),     // Aspirin
    ];
    
    // Execute analysis across multiple scales
    let result = system.execute_cross_scale_analysis(
        molecules,
        vec![
            BMDScale::Quantum,
            BMDScale::Molecular,
            BMDScale::Environmental,
            BMDScale::Hardware
        ]
    )?;
    
    println!("Multi-scale analysis results:");
    println!("  Scales coordinated: {}", result.scale_coordination_results.len());
    println!("  Total amplification: {:.0}×", result.amplification_factor);
    
    // Display coordination results
    for coord in &result.scale_coordination_results {
        println!("  {:?} → {:?}: {:.2} efficiency", 
                 coord.source_scale, 
                 coord.target_scale, 
                 coord.transfer_efficiency);
    }
    
    Ok(())
}
```

## Advanced Scientific Applications

### 3. Drug Discovery Pipeline

```rust
use borgia::{initialize, Molecule, BMDScale, MolecularDatabase};

fn drug_discovery_analysis() -> borgia::BorgiaResult<()> {
    let mut system = initialize()?;
    let mut database = MolecularDatabase::new();
    
    // Load pharmaceutical compounds
    let drug_compounds = vec![
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  // Ibuprofen
        "CC(=O)NC1=CC=C(C=C1)O",          // Acetaminophen
        "CC(=O)OC1=CC=CC=C1C(=O)O",       // Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   // Caffeine
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34", // Testosterone
    ];
    
    // Add compounds to database
    for smiles in &drug_compounds {
        let molecule = Molecule::from_smiles(smiles)?;
        database.add_molecule(molecule)?;
    }
    
    // Perform BMD-enhanced analysis
    let analysis_result = system.execute_cross_scale_analysis(
        drug_compounds.clone(),
        vec![BMDScale::Molecular, BMDScale::Environmental, BMDScale::Hardware]
    )?;
    
    println!("Drug Discovery Analysis:");
    println!("  Compounds analyzed: {}", drug_compounds.len());
    println!("  BMD amplification: {:.0}×", analysis_result.amplification_factor);
    
    // Similarity analysis
    let query_molecule = Molecule::from_smiles("CC(=O)NC1=CC=C(C=C1)O")?; // Acetaminophen
    let similar_compounds = database.search_similar(&query_molecule, 0.7)?;
    
    println!("  Similar compounds found: {}", similar_compounds.len());
    for sim_match in &similar_compounds {
        println!("    {} (similarity: {:.3})", 
                 sim_match.molecule.smiles, 
                 sim_match.similarity_score);
    }
    
    Ok(())
}
```

### 4. Environmental Noise Enhancement

```rust
use borgia::{initialize, noise_enhanced_cheminformatics};

fn environmental_enhancement_demo() -> borgia::BorgiaResult<()> {
    // Start with a small dataset (laboratory isolation problem)
    let small_dataset = vec![
        "CCO".to_string(),           // Ethanol
        "CC(=O)O".to_string(),       // Acetic acid
        "C6H12O6".to_string(),       // Glucose
        "C8H10N4O2".to_string(),     // Caffeine
    ];
    
    println!("Environmental Noise Enhancement Demo:");
    println!("  Original dataset size: {}", small_dataset.len());
    
    // Apply environmental noise enhancement
    let enhanced_result = noise_enhanced_cheminformatics(small_dataset)?;
    
    println!("  Enhanced dataset size: {}", enhanced_result.enhanced_size);
    println!("  Enhancement factor: {:.1}×", 
             enhanced_result.enhanced_size as f64 / enhanced_result.original_size as f64);
    println!("  Environmental noise sources: {}", enhanced_result.noise_sources.len());
    
    // Display noise characteristics
    for (i, noise_source) in enhanced_result.noise_sources.iter().enumerate() {
        println!("    Noise source {}: {} patterns", i + 1, noise_source.pattern_count);
    }
    
    println!("  Solution emergence detected: {}", enhanced_result.solution_emergence);
    
    Ok(())
}
```

### 5. Hardware-Integrated Spectroscopy

```rust
use borgia::{initialize, hardware_molecular_spectroscopy, MolecularSample};

fn hardware_spectroscopy_demo() -> borgia::BorgiaResult<()> {
    println!("Hardware-Integrated Molecular Spectroscopy:");
    
    // Prepare fluorescent compounds for LED spectroscopy
    let fluorescent_compounds = vec![
        "fluorescein".to_string(),
        "rhodamine_6G".to_string(),
        "FITC".to_string(),
        "texas_red".to_string(),
    ];
    
    // Execute hardware-integrated analysis
    let spectroscopy_result = hardware_molecular_spectroscopy(fluorescent_compounds)?;
    
    println!("  Spectroscopy Analysis Results:");
    println!("    Samples analyzed: {}", spectroscopy_result.samples_analyzed);
    println!("    Wavelength range: {}-{}nm", 
             spectroscopy_result.wavelength_range.start,
             spectroscopy_result.wavelength_range.end);
    println!("    LED utilization: {:.1}%", spectroscopy_result.led_utilization * 100.0);
    println!("    Zero-cost analysis: {}", spectroscopy_result.zero_cost_analysis);
    
    // Display spectral data
    for (i, spectrum) in spectroscopy_result.spectral_data.iter().enumerate() {
        println!("    Sample {}: {} spectral points", i + 1, spectrum.data_points.len());
        println!("      Peak wavelength: {}nm", spectrum.peak_wavelength);
        println!("      Peak intensity: {:.3}", spectrum.peak_intensity);
    }
    
    // Fire-light coupling results
    if let Some(coupling_result) = &spectroscopy_result.consciousness_coupling {
        println!("  Consciousness Enhancement (650nm coupling):");
        println!("    Coupling strength: {:.3}", coupling_result.coupling_strength);
        println!("    Enhancement active: {}", coupling_result.enhancement_active);
    }
    
    Ok(())
}
```

## Theoretical Demonstrations

### 6. Mizraji's Prisoner Parable

```rust
use borgia::demonstrate_prisoner_parable;

fn prisoner_parable_demo() -> borgia::BorgiaResult<()> {
    println!("Mizraji's Prisoner Parable Demonstration:");
    
    // Execute the prisoner parable
    let parable_result = demonstrate_prisoner_parable()?;
    
    println!("  Information Catalysis Results:");
    println!("    Pattern recognition energy: {:.6} J", parable_result.pattern_recognition_energy);
    println!("    Thermodynamic consequences: {:.3} J", parable_result.thermodynamic_consequences);
    println!("    Amplification factor: {:.0}×", parable_result.amplification_factor);
    
    println!("  Prisoner Scenario:");
    println!("    Escape opportunity recognized: {}", parable_result.escape_opportunity_recognized);
    println!("    Action triggered: {}", parable_result.action_triggered);
    println!("    Cascade events: {}", parable_result.cascade_events.len());
    
    // Display cascade events
    for (i, event) in parable_result.cascade_events.iter().enumerate() {
        println!("      Event {}: {} (energy: {:.3} J)", 
                 i + 1, event.description, event.energy_cost);
    }
    
    println!("  Theoretical Validation:");
    println!("    BMD construction cost: {:.6} J", parable_result.bmd_construction_cost);
    println!("    BMD operation cost: {:.6} J", parable_result.bmd_operation_cost);
    println!("    Cost-benefit ratio: {:.0}:1", 
             parable_result.thermodynamic_consequences / 
             (parable_result.bmd_construction_cost + parable_result.bmd_operation_cost));
    
    Ok(())
}
```

### 7. Cross-Scale Information Propagation

```rust
use borgia::{initialize, BMDScale, InformationPacket};

fn cross_scale_propagation_demo() -> borgia::BorgiaResult<()> {
    let mut system = initialize()?;
    
    println!("Cross-Scale Information Propagation Demo:");
    
    // Create information packet
    let info_packet = InformationPacket::new("molecular_binding_event".to_string());
    
    // Demonstrate information propagation across scales
    let scales = vec![
        BMDScale::Quantum,
        BMDScale::Molecular,
        BMDScale::Environmental,
        BMDScale::Hardware,
        BMDScale::Cognitive,
    ];
    
    println!("  Scale Coordination Results:");
    
    // Propagate information between adjacent scales
    for i in 0..scales.len()-1 {
        let source_scale = scales[i].clone();
        let target_scale = scales[i+1].clone();
        
        let coordination_result = system.coordinate_scales(
            source_scale.clone(),
            target_scale.clone(),
            &info_packet
        )?;
        
        println!("    {:?} → {:?}:", source_scale, target_scale);
        println!("      Transfer efficiency: {:.3}", coordination_result.transfer_efficiency);
        println!("      Coupling strength: {:.3}", coordination_result.coupling_strength);
        println!("      Sync window: {:?}", coordination_result.synchronization_window);
    }
    
    // Measure overall coherence
    let coherence_metrics = system.measure_cross_scale_coherence()?;
    println!("  Overall System Coherence:");
    println!("    Temporal coherence: {:.3}", coherence_metrics.temporal_coherence);
    println!("    Information conservation: {:.3}", coherence_metrics.information_conservation);
    println!("    Thermodynamic consistency: {}", coherence_metrics.thermodynamic_consistency);
    
    Ok(())
}
```

## Performance and Optimization Examples

### 8. Performance Benchmarking

```rust
use borgia::{initialize, BMDScale, PerformanceProfiler};
use std::time::Instant;

fn performance_benchmark() -> borgia::BorgiaResult<()> {
    let mut system = initialize()?;
    let mut profiler = PerformanceProfiler::new();
    
    println!("Performance Benchmarking:");
    
    // Benchmark different scale combinations
    let test_cases = vec![
        (vec![BMDScale::Quantum], "Quantum only"),
        (vec![BMDScale::Molecular], "Molecular only"),
        (vec![BMDScale::Environmental], "Environmental only"),
        (vec![BMDScale::Hardware], "Hardware only"),
        (vec![BMDScale::Quantum, BMDScale::Molecular], "Quantum + Molecular"),
        (vec![BMDScale::Molecular, BMDScale::Environmental], "Molecular + Environmental"),
        (vec![BMDScale::Environmental, BMDScale::Hardware], "Environmental + Hardware"),
        (vec![BMDScale::Quantum, BMDScale::Molecular, BMDScale::Environmental, BMDScale::Hardware], "All scales"),
    ];
    
    let test_molecules = vec![
        "CCO".to_string(),
        "CC(=O)O".to_string(),
        "C6H12O6".to_string(),
        "C8H10N4O2".to_string(),
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O".to_string(),
    ];
    
    for (scales, description) in test_cases {
        let start_time = Instant::now();
        
        let result = system.execute_cross_scale_analysis(
            test_molecules.clone(),
            scales.clone()
        )?;
        
        let execution_time = start_time.elapsed();
        
        println!("  {}: ", description);
        println!("    Execution time: {:?}", execution_time);
        println!("    Amplification: {:.0}×", result.amplification_factor);
        println!("    Memory usage: {} KB", result.performance_metrics.memory_usage / 1024);
        println!("    Efficiency: {:.3}", 
                 result.amplification_factor / execution_time.as_secs_f64());
    }
    
    Ok(())
}
```

### 9. Memory Optimization

```rust
use borgia::{initialize, BorgiaConfig, OptimizationLevel};

fn memory_optimization_demo() -> borgia::BorgiaResult<()> {
    println!("Memory Optimization Demo:");
    
    // Test different optimization levels
    let optimization_levels = vec![
        (OptimizationLevel::Speed, "Speed optimized"),
        (OptimizationLevel::Balanced, "Balanced"),
        (OptimizationLevel::Memory, "Memory optimized"),
    ];
    
    let large_dataset: Vec<String> = (0..100)
        .map(|i| format!("C{}H{}O{}", i % 10 + 1, (i % 20) + 2, (i % 5) + 1))
        .collect();
    
    for (opt_level, description) in optimization_levels {
        let config = BorgiaConfig {
            optimization_level: opt_level,
            ..Default::default()
        };
        
        let mut system = borgia::initialize_with_config(config)?;
        
        let result = system.execute_cross_scale_analysis(
            large_dataset.clone(),
            vec![borgia::BMDScale::Molecular, borgia::BMDScale::Environmental]
        )?;
        
        println!("  {} configuration:", description);
        println!("    Peak memory: {} MB", result.performance_metrics.memory_usage / (1024 * 1024));
        println!("    Execution time: {:?}", result.performance_metrics.execution_time);
        println!("    Molecules processed: {}", large_dataset.len());
        println!("    Throughput: {:.1} molecules/sec", 
                 large_dataset.len() as f64 / result.performance_metrics.execution_time.as_secs_f64());
    }
    
    Ok(())
}
```

## Integration Examples

### 10. Custom BMD Implementation

```rust
use borgia::{BMDScale, BMD, InformationPacket, BorgiaResult};

// Custom BMD implementation
struct CustomProteinBMD {
    protein_database: Vec<String>,
    binding_sites: Vec<BindingSite>,
}

impl BMD for CustomProteinBMD {
    fn scale(&self) -> BMDScale {
        BMDScale::Molecular
    }
    
    fn process_information(&mut self, info: &InformationPacket) -> BorgiaResult<InformationPacket> {
        // Custom protein analysis logic
        let protein_sequence = info.data.as_str();
        let binding_analysis = self.analyze_protein_binding(protein_sequence)?;
        
        Ok(InformationPacket::new(format!("protein_analysis:{}", binding_analysis)))
    }
}

impl CustomProteinBMD {
    fn new() -> Self {
        Self {
            protein_database: Vec::new(),
            binding_sites: Vec::new(),
        }
    }
    
    fn analyze_protein_binding(&self, sequence: &str) -> BorgiaResult<String> {
        // Implement custom protein binding analysis
        Ok(format!("binding_score:{:.3}", sequence.len() as f64 * 0.1))
    }
}

fn custom_bmd_demo() -> BorgiaResult<()> {
    let mut system = borgia::initialize()?;
    let mut custom_bmd = CustomProteinBMD::new();
    
    // Register custom BMD with the system
    system.register_custom_bmd(Box::new(custom_bmd))?;
    
    // Use the custom BMD in analysis
    let protein_sequences = vec![
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAE".to_string(),
    ];
    
    let result = system.execute_cross_scale_analysis(
        protein_sequences,
        vec![BMDScale::Molecular]
    )?;
    
    println!("Custom Protein BMD Analysis:");
    println!("  Custom BMD registered: ✓");
    println!("  Protein sequences analyzed: 1");
    println!("  Analysis amplification: {:.0}×", result.amplification_factor);
    
    Ok(())
}
```

### 11. Real-Time Monitoring

```rust
use borgia::{initialize, BMDScale, SystemMonitor};
use std::thread;
use std::time::Duration;

fn real_time_monitoring_demo() -> borgia::BorgiaResult<()> {
    let mut system = initialize()?;
    let mut monitor = SystemMonitor::new();
    
    println!("Real-Time BMD System Monitoring:");
    
    // Start monitoring in a separate thread
    let monitor_handle = thread::spawn(move || {
        for i in 0..10 {
            thread::sleep(Duration::from_secs(1));
            
            let metrics = monitor.collect_metrics().unwrap();
            println!("  [{}s] CPU: {:.1}%, Memory: {} MB, Active BMDs: {}", 
                     i + 1,
                     metrics.cpu_usage * 100.0,
                     metrics.memory_usage / (1024 * 1024),
                     metrics.active_bmds.len());
        }
    });
    
    // Perform continuous analysis
    let molecules = vec![
        "CCO".to_string(),
        "CC(=O)O".to_string(),
        "C6H12O6".to_string(),
    ];
    
    for i in 0..5 {
        let result = system.execute_cross_scale_analysis(
            molecules.clone(),
            vec![BMDScale::Molecular, BMDScale::Environmental]
        )?;
        
        println!("  Analysis {}: {:.0}× amplification", i + 1, result.amplification_factor);
        thread::sleep(Duration::from_secs(2));
    }
    
    monitor_handle.join().unwrap();
    
    Ok(())
}
```

## Running the Examples

To run these examples, add them to your `main.rs` or create separate example files:

```rust
// examples/basic_usage.rs
use borgia::BorgiaResult;

fn main() -> BorgiaResult<()> {
    // Run basic examples
    basic_molecular_analysis()?;
    multi_scale_coordination()?;
    
    // Run advanced examples
    drug_discovery_analysis()?;
    environmental_enhancement_demo()?;
    hardware_spectroscopy_demo()?;
    
    // Run theoretical demonstrations
    prisoner_parable_demo()?;
    cross_scale_propagation_demo()?;
    
    // Run performance tests
    performance_benchmark()?;
    memory_optimization_demo()?;
    
    // Run integration examples
    custom_bmd_demo()?;
    real_time_monitoring_demo()?;
    
    Ok(())
}

// Include all the example functions here...
```

### Cargo Commands

```bash
# Run basic examples
cargo run --example basic_usage

# Run with specific features
cargo run --example advanced_analysis --features "hardware-integration"

# Run performance benchmarks
cargo run --example performance_benchmark --release

# Run with debug output
RUST_LOG=debug cargo run --example debug_analysis
```

## Example Output

When you run the examples, you should see output similar to:

```
🧬 Borgia Framework Examples
═══════════════════════════

Basic Molecular Analysis:
  Molecules analyzed: 2
  Amplification factor: 156.7×
  Thermodynamic consistency: ✓

Multi-Scale Coordination:
  Scales coordinated: 6
  Total amplification: 1,247×
  Quantum → Molecular: 0.923 efficiency
  Molecular → Environmental: 0.887 efficiency
  Environmental → Hardware: 0.901 efficiency

Drug Discovery Analysis:
  Compounds analyzed: 5
  BMD amplification: 2,341×
  Similar compounds found: 3

Environmental Noise Enhancement:
  Original dataset size: 4
  Enhanced dataset size: 47
  Enhancement factor: 11.8×
  Solution emergence detected: ✓

Hardware Spectroscopy:
  Samples analyzed: 4
  Wavelength range: 400-700nm
  LED utilization: 87.3%
  Zero-cost analysis: ✓
  Consciousness Enhancement (650nm): ✓

Mizraji's Prisoner Parable:
  Pattern recognition energy: 0.000001 J
  Thermodynamic consequences: 1.247 J
  Amplification factor: 1,247,000×
  Cost-benefit ratio: 1,247,000:1

✨ All examples completed successfully!
```

---

*These examples demonstrate the full capabilities of the Borgia framework, from basic molecular analysis to advanced cross-scale coordination and theoretical validation of Mizraji's biological Maxwell's demons.* 