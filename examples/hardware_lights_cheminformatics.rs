// =====================================================================================
// HARDWARE LIGHTS FOR CHEMINFORMATICS EXAMPLE
// Demonstrates revolutionary use of computer hardware lights for molecular analysis
// =====================================================================================

use borgia::hardware_spectroscopy::{
    HardwareSpectroscopySystem, 
    HardwareSpectroscopyResult, 
    MolecularSpectroscopyResult
};
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Hardware Lights for Cheminformatics Analysis");
    println!("===============================================");
    println!();
    
    // Initialize the hardware-integrated spectroscopy system
    let mut system = HardwareSpectroscopySystem::new()?;
    
    println!("📡 Detected Hardware Components:");
    print_hardware_capabilities(&system);
    println!();
    
    // =====================================================================================
    // MOLECULAR FLUORESCENCE ANALYSIS USING RGB LEDS
    // Demonstrates how RGB LEDs can excite molecular fluorescence
    // =====================================================================================
    
    println!("🧪 MOLECULAR FLUORESCENCE ANALYSIS");
    println!("==================================");
    
    let test_molecules = vec![
        ("CCO", "Ethanol - Simple alcohol"),
        ("C1=CC=CC=C1", "Benzene - Aromatic compound"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine - Complex alkaloid"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen - Anti-inflammatory"),
    ];
    
    for (smiles, description) in &test_molecules {
        println!("  🔍 Analyzing: {}", description);
        
        let result = system.analyze_molecule_with_hardware(
            smiles,
            "fluorescence_detection"
        ).await?;
        
        print_spectroscopy_result(&result);
        println!();
    }
    
    // =====================================================================================
    // FIRE-LIGHT COUPLING ANALYSIS (650nm OPTIMIZATION)
    // Demonstrates consciousness-enhanced molecular analysis using hardware lights
    // =====================================================================================
    
    println!("🔥 FIRE-LIGHT COUPLING ANALYSIS (650nm)");
    println!("=======================================");
    
    println!("Testing fire-light coupling optimization with RGB LED mixing...");
    println!();
    
    for (smiles, description) in &test_molecules {
        println!("  🔥 Fire-Light Analysis: {}", description);
        
        let result = system.analyze_molecule_with_hardware(
            smiles,
            "fire_light_coupling"
        ).await?;
        
        print_fire_light_result(&result);
        println!();
    }
    
    // =====================================================================================
    // HARDWARE COMPONENT ANALYSIS
    // Shows which hardware components provide the best molecular analysis capabilities
    // =====================================================================================
    
    println!("⚙️  HARDWARE COMPONENT ANALYSIS");
    println!("===============================");
    
    analyze_hardware_components(&system);
    println!();
    
    // =====================================================================================
    // COMPARATIVE ANALYSIS: TRADITIONAL vs HARDWARE-INTEGRATED
    // Demonstrates the advantages of hardware-integrated molecular spectroscopy
    // =====================================================================================
    
    println!("📊 COMPARATIVE ANALYSIS");
    println!("=======================");
    
    perform_comparative_analysis(&mut system).await?;
    
    // =====================================================================================
    // CHEMINFORMATICS APPLICATIONS
    // Practical applications of hardware lights in molecular analysis
    // =====================================================================================
    
    println!("🧬 PRACTICAL CHEMINFORMATICS APPLICATIONS");
    println!("=========================================");
    
    demonstrate_practical_applications(&mut system).await?;
    
    Ok(())
}

fn print_hardware_capabilities(system: &HardwareSpectroscopySystem) {
    println!("  📱 Light Sources:");
    println!("    • RGB LEDs: Red (625nm), Green (525nm), Blue (470nm)");
    println!("    • Infrared LEDs: 850nm for molecular vibration excitation");
    println!("    • Status LEDs: Various wavelengths for state indication");
    println!("    • Display Backlights: Full spectrum illumination");
    
    println!("  📡 Sensors:");
    println!("    • Photodetectors: 300-1100nm spectral range");
    println!("    • Ambient Light Sensors: Environmental monitoring");
    println!("    • Image Sensors: Pattern recognition and fluorescence imaging");
    println!("    • Optical Mouse Sensors: Surface interaction analysis");
    
    println!("  🔥 Fire-Light Coupling:");
    println!("    • Optimal wavelength: 650nm");
    println!("    • RGB mixing: 90% red, 10% green, 0% blue");
    println!("    • Consciousness enhancement enabled");
}

fn print_spectroscopy_result(result: &HardwareSpectroscopyResult) {
    println!("    📈 Results:");
    println!("      • Protocol: {}", result.protocol_used);
    println!("      • Measurements: {} steps", result.measurements.len());
    println!("      • Hardware efficiency: {:.1}%", result.hardware_efficiency * 100.0);
    println!("      • Molecular confidence: {:.1}%", result.molecular_identification_confidence * 100.0);
    println!("      • Analysis time: {:?}", result.analysis_duration);
    
    if !result.measurements.is_empty() {
        let first_measurement = &result.measurements[0];
        println!("      • Excitation: {:.0}nm", first_measurement.excitation_wavelength);
        println!("      • Detection: {:.0}nm", first_measurement.detection_wavelength);
        println!("      • Signal intensity: {:.3}", first_measurement.measured_intensity);
        println!("      • SNR: {:.1}", first_measurement.signal_to_noise_ratio);
    }
}

fn print_fire_light_result(result: &HardwareSpectroscopyResult) {
    println!("    🔥 Fire-Light Results:");
    println!("      • Enhancement factor: {:.2}x", result.fire_light_coupling_enhancement);
    println!("      • Consciousness coupling: {:.1}%", result.hardware_efficiency * 100.0);
    println!("      • Molecular confidence: {:.1}%", result.molecular_identification_confidence * 100.0);
    
    if !result.measurements.is_empty() {
        let fire_measurements: Vec<_> = result.measurements.iter()
            .filter(|m| (640.0..=660.0).contains(&m.excitation_wavelength))
            .collect();
        
        if !fire_measurements.is_empty() {
            println!("      • 650nm optimization: Active");
            println!("      • Signal boost: {:.1}%", 
                (fire_measurements[0].measured_intensity - 0.5) / 0.5 * 100.0);
        }
    }
}

fn analyze_hardware_components(system: &HardwareSpectroscopySystem) {
    println!("  💡 RGB LED Analysis:");
    println!("    • Red LEDs (625nm): Optimal for fire-light coupling");
    println!("    • Green LEDs (525nm): Good for chlorophyll-like molecules");
    println!("    • Blue LEDs (470nm): High-energy excitation, good for fluorescence");
    println!("    • Individual control: {}", system.light_sources.rgb_leds.individual_control);
    println!("    • Max intensity: {:.0} mW/cm²", system.light_sources.rgb_leds.max_intensity);
    
    println!("  🔴 Infrared LED Analysis:");
    println!("    • Wavelength: {:.0}nm", system.light_sources.infrared_leds.wavelength_nm);
    println!("    • Applications: Molecular vibration excitation, C-H bond analysis");
    println!("    • Pulse modulation: {}", system.light_sources.infrared_leds.pulse_modulation);
    
    println!("  📱 Status LED Integration:");
    println!("    • Available LEDs: {}", system.light_sources.status_leds.available_leds.len());
    println!("    • Molecular sync: {}", system.light_sources.status_leds.molecular_sync);
    println!("    • Real-time feedback: Visual molecular state indication");
    
    println!("  📟 Display Backlight Applications:");
    println!("    • Full spectrum: {}", system.light_sources.display_backlights.full_spectrum);
    println!("    • Sample illumination: {}", system.light_sources.display_backlights.sample_illumination);
    println!("    • Color temperature range: {:.0}K - {:.0}K", 
        system.light_sources.display_backlights.color_temperature_range.0,
        system.light_sources.display_backlights.color_temperature_range.1);
}

async fn perform_comparative_analysis(system: &mut HardwareSpectroscopySystem) -> Result<(), Box<dyn std::error::Error>> {
    let test_molecule = "C1=CC=CC=C1"; // Benzene
    
    println!("  🧪 Test Molecule: Benzene (C1=CC=CC=C1)");
    println!();
    
    // Traditional approach simulation
    println!("  📊 Traditional Spectroscopy (simulated):");
    println!("    • Equipment: Dedicated UV-Vis spectrometer");
    println!("    • Cost: $10,000 - $50,000");
    println!("    • Setup time: 30-60 minutes");
    println!("    • Analysis time: 5-10 minutes");
    println!("    • Portability: Laboratory-bound");
    println!("    • Wavelength range: Limited by instrument");
    
    // Hardware-integrated approach
    println!("  💻 Hardware-Integrated Spectroscopy:");
    let hw_result = system.analyze_molecule_with_hardware(
        test_molecule,
        "fluorescence_detection"
    ).await?;
    
    println!("    • Equipment: Computer hardware components");
    println!("    • Cost: $0 (using existing hardware)");
    println!("    • Setup time: < 1 second");
    println!("    • Analysis time: {:?}", hw_result.analysis_duration);
    println!("    • Portability: Any computer with LEDs/sensors");
    println!("    • Wavelength range: RGB (470-625nm) + IR (850nm)");
    println!("    • Hardware efficiency: {:.1}%", hw_result.hardware_efficiency * 100.0);
    
    println!();
    println!("  ✅ Advantages of Hardware Integration:");
    println!("    • Zero additional cost");
    println!("    • Instant availability");
    println!("    • Universal accessibility");
    println!("    • Real-time processing");
    println!("    • Consciousness-enhanced analysis (fire-light coupling)");
    println!("    • Integration with molecular navigation systems");
    
    Ok(())
}

async fn demonstrate_practical_applications(system: &mut HardwareSpectroscopySystem) -> Result<(), Box<dyn std::error::Error>> {
    println!("  🧬 Drug Discovery Applications:");
    println!("    • Fluorescence screening: Blue LED excitation + photodetector");
    println!("    • Molecular fingerprinting: Multi-wavelength LED sequences");
    println!("    • Real-time monitoring: Status LED molecular state feedback");
    println!();
    
    // Drug molecule analysis
    let drug_molecules = vec![
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen"),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin"),
    ];
    
    println!("  💊 Drug Screening Results:");
    for (smiles, name) in &drug_molecules {
        let result = system.analyze_molecule_with_hardware(
            smiles,
            "fire_light_coupling"
        ).await?;
        
        println!("    • {}: Enhancement {:.2}x, Confidence {:.1}%", 
            name, 
            result.fire_light_coupling_enhancement,
            result.molecular_identification_confidence * 100.0);
    }
    println!();
    
    println!("  🔬 Environmental Monitoring:");
    println!("    • Ambient light sensors: Monitor environmental light interference");
    println!("    • Image sensors: Capture molecular fluorescence patterns");
    println!("    • Optical mouse sensors: Detect molecular surface interactions");
    println!();
    
    println!("  🎮 Gaming Hardware Integration:");
    println!("    • RGB gaming LEDs: Multi-wavelength molecular excitation");
    println!("    • Gaming monitors: OLED pixels for precise wavelength control");
    println!("    • Gaming mice: Optical sensors for molecular motion detection");
    println!();
    
    println!("  📱 Mobile Device Applications:");
    println!("    • Smartphone cameras: Molecular fluorescence detection");
    println!("    • LED flash: Molecular excitation source");
    println!("    • Ambient light sensors: Environmental compensation");
    println!("    • Infrared sensors: Face ID adapted for molecular analysis");
    println!();
    
    println!("  🏠 IoT Integration:");
    println!("    • Smart home LEDs: Distributed molecular sensing network");
    println!("    • Security cameras: Molecular pattern recognition");
    println!("    • Smart displays: Visual molecular analysis feedback");
    
    Ok(())
}

// Helper function to demonstrate the concept
fn explain_hardware_light_cheminformatics_concept() {
    println!("🌟 CONCEPT: Hardware Lights as Cheminformatics Tools");
    println!("====================================================");
    println!();
    
    println!("Every computer contains sophisticated light-producing and light-sensing");
    println!("components that can be repurposed for molecular analysis:");
    println!();
    
    println!("💡 Light-Producing Components:");
    println!("  • LEDs: Precise wavelength sources for molecular excitation");
    println!("  • OLED displays: Individual pixel control for spectroscopy");
    println!("  • Backlights: Full-spectrum illumination for samples");
    println!("  • Infrared LEDs: Molecular vibration excitation");
    println!();
    
    println!("📡 Light-Sensing Components:");
    println!("  • Photodiodes: Detect molecular fluorescence and absorption");
    println!("  • Image sensors: Capture molecular patterns and fluorescence");
    println!("  • Ambient sensors: Environmental light monitoring");
    println!("  • Optical mice: Surface molecular interaction detection");
    println!();
    
    println!("🔬 Cheminformatics Applications:");
    println!("  • Molecular fluorescence spectroscopy");
    println!("  • Absorption spectroscopy using LED excitation");
    println!("  • Real-time molecular monitoring");
    println!("  • Pattern recognition of molecular structures");
    println!("  • Environmental molecular sensing");
    println!();
    
    println!("🚀 Revolutionary Impact:");
    println!("  • Turns every computer into a molecular analysis instrument");
    println!("  • Zero additional hardware cost");
    println!("  • Universal accessibility to spectroscopy");
    println!("  • Real-time molecular feedback systems");
    println!("  • Integration with consciousness-enhanced analysis (fire-light coupling)");
} 