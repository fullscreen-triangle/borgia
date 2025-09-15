# Borgia BMD Framework - Validation Demo

Simplified Python validation framework for the Borgia Rust implementation. Validates core S-Entropy framework claims and BMD capabilities with minimal dependencies.

## Quick Start

```bash
cd demos
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -e .
python validate_borgia.py --quick --visualize
```

## What This Validates

### Core BMD Capabilities
- **Dual-Functionality Molecules**: Every molecule functions as both a precision clock AND computational processor
- **BMD Network Coordination**: Multi-scale coordination across quantum (10⁻¹⁵s), molecular (10⁻⁹s), and environmental (10²s) timescales
- **Information Catalysis**: >1000× thermodynamic amplification with >95% efficiency
- **Hardware Integration**: Zero-cost LED spectroscopy (470nm, 525nm, 625nm) with performance gains

### S-Entropy Framework Claims
- **Universal Predetermined Solutions**: Every problem has a unique optimal solution existing before computation
- **S-Distance Minimization**: Optimal problem-solving through minimizing observer-process separation
- **Reality Solutions Theorem**: Since reality happens, every problem ≤ reality complexity has solutions
- **Consciousness-Computation Equivalence**: BMD frame selection ≡ S-entropy navigation processes
- **Universal Oscillation Equation**: S = k log α, where solutions correspond to amplitude configurations

## Usage

### Basic Validation
```bash
python validate_borgia.py
```
Validates core BMD capabilities using the Rust implementation.

### Quick Mode
```bash
python validate_borgia.py --quick
```
Faster validation with fewer test molecules (20 vs 50).

### Include S-Entropy Framework
```bash
python validate_borgia.py --s-entropy
```
Validates theoretical S-Entropy claims in addition to BMD capabilities.

### Generate Visualizations
```bash
python validate_borgia.py --visualize
```
Creates matplotlib plots showing validation results and performance metrics.

### Complete Validation
```bash
python validate_borgia.py --s-entropy --visualize
```
Full validation with theoretical framework verification and visual reports.

## Output Files

Results are saved to `borgia_validation_results/` (customizable with `--output-dir`):

- `validation_results_<timestamp>.json` - Detailed validation data
- `validation_summary.png` - Test results overview
- `performance_metrics.png` - Performance characteristics
- `s_entropy_validation.png` - S-Entropy framework validation
- `comprehensive_report.png` - Complete system status
- `validation_summary.txt` - Text summary report

## Interface Modes

### Rust Integration (Preferred)
When `Cargo.toml` is found in the parent directory, the validator attempts to call:
```bash
cargo run -- <command> [args]
```

Expected Rust commands:
- `generate-molecules` - Generate dual-functionality molecules
- `test-bmd-coordination` - Test multi-scale BMD networks
- `validate-catalysis` - Validate information catalysis
- `test-hardware` - Test hardware integration
- `validate-s-entropy` - Validate S-Entropy framework
- `comprehensive-validate` - Run all validations
- `info` - Get system information

### Mock Mode (Fallback)
If Rust implementation is not available, uses mock interface with simulated results for testing the validation framework itself.

## Dependencies

**Minimal Required:**
- `numpy>=1.21.0` - Numerical computations
- `matplotlib>=3.5.0` - Basic plotting
- `pytest>=7.0.0` - Testing framework

**Optional:**
- `scipy>=1.9.0` - Advanced statistics (Python <3.13 only)

## Integration with Rust

The validator interfaces with your Rust Borgia implementation through JSON-based command line interface:

```python
# Example: Generate molecules
result = borgia.run_borgia_command("generate-molecules", input_data={
    "count": 50,
    "precision_target": 1e-30,
    "dual_functionality": True
})
```

Expected JSON response format:
```json
{
  "success": true,
  "data": {
    "molecules": [...],
    "generation_time": 0.5,
    "success_rate": 1.0
  },
  "execution_time": 0.3
}
```

## Success Criteria

### Critical Requirements (Zero Tolerance)
- Dual-functionality rate: 100% (every molecule must be both clock AND processor)
- Information catalysis efficiency: ≥95%
- Thermodynamic amplification: ≥1000×
- Hardware integration: All components operational with zero cost

### Performance Targets
- Clock precision: ≤10⁻³⁰ seconds
- BMD network efficiency: ≥90% across all timescales
- S-Entropy complexity reduction: ≥100× factor

## Theoretical Foundation

Based on:
- Eduardo Mizraji's Biological Maxwell Demons (BMD) theory
- S-Entropy framework as described in st-stellas papers
- Universal Predetermined Solutions theorem
- Reality-based complexity bounds
- Consciousness-computation mathematical equivalence

The validation confirms these theoretical claims through direct testing of the Rust implementation.

## Architecture

```
demos/
├── borgia_validation/           # Python validation package
│   ├── rust_interface.py       # Interface to Rust Borgia
│   ├── core_validator.py       # BMD capabilities validation
│   ├── s_entropy_validator.py  # S-Entropy framework validation
│   └── simple_visualizer.py    # Minimal dependency plotting
├── validate_borgia.py          # Main validation script
├── setup.py                    # Package installation
└── requirements.txt            # Minimal dependencies
```

## Troubleshooting

**"Could not find Cargo.toml"**: Make sure you're running from the `demos/` directory inside a Rust project.

**"Mock interface active"**: Rust implementation not found. Validator will use simulated results for framework testing.

**Module import errors**: Run `pip install -e .` to install the validation package.

**Visualization errors**: Install matplotlib: `pip install matplotlib>=3.5.0`

## Expected Output

```
🧬 BORGIA BMD VALIDATION
   S-Entropy Framework & BMD Capabilities Test

🔧 Initializing Borgia interface...
   ✓ Interface: rust (version 1.0.0)

🧪 RUNNING CORE BMD VALIDATION

🧪 Validating Dual-Functionality Molecules...
   ✓ Generated: 50 molecules
   ✓ Clock functional: 100.0%
   ✓ Processor functional: 100.0%
   ✓ Dual functional: 100.0%
   ✅ PASS: Zero tolerance requirement

🔬 Validating BMD Network Coordination...
   ✓ Quantum efficiency: 95.2%
   ✓ Molecular efficiency: 97.8%
   ✓ Environmental efficiency: 91.4%
   ✓ Amplification factor: 1247.3×
   ✅ PASS: BMD network coordination

🏁 VALIDATION COMPLETE
✓ Core BMD validation: 100.0% pass rate
✓ Overall success rate: 100.0%

🎉 BORGIA BMD FRAMEWORK VALIDATED
   • All critical systems operational
   • Ready for production deployment
```

This simplified framework focuses on validating the actual claims through the Rust implementation rather than recreating complex simulations in Python.