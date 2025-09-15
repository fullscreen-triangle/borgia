"""
Core Borgia Validator
====================

Simplified validator for core Borgia BMD framework claims.
Focuses on validating the essential capabilities through Rust interface.
"""

import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from .rust_interface import get_borgia_interface


class ValidationResult:
    """Simple validation result container."""
    
    def __init__(self, test_name: str, success: bool, score: float, 
                 metrics: Dict[str, Any], errors: List[str] = None):
        self.test_name = test_name
        self.success = success
        self.score = score  # 0.0 to 1.0
        self.metrics = metrics
        self.errors = errors or []
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "success": self.success,
            "score": self.score,
            "metrics": self.metrics,
            "errors": self.errors,
            "timestamp": self.timestamp
        }


class BorgiaValidator:
    """
    Core validator for Borgia BMD framework capabilities.
    """
    
    def __init__(self, output_dir: str = "validation_results", quiet_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            output_dir: Directory to save validation results
            quiet_mode: If True, minimize console output and save all data as JSON
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quiet_mode = quiet_mode
        
        # Get Borgia interface
        self.borgia = get_borgia_interface()
        
        # Validation results and progress data
        self.results = {}
        self.progress_data = []
        
    def _log_progress(self, test_name: str, stage: str, data: Dict[str, Any]):
        """Log progress data to JSON file instead of console."""
        progress_entry = {
            "timestamp": time.time(),
            "test_name": test_name,
            "stage": stage,
            "data": data
        }
        self.progress_data.append(progress_entry)
        
        # Save progress to intermediate file
        progress_file = self.output_dir / f"progress_{test_name}_{int(time.time())}.json"
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_entry, f, indent=2, default=str)
        except Exception:
            pass  # Don't fail validation if progress logging fails
            
    def _print_or_log(self, message: str, test_name: str = "", stage: str = "info", data: Dict[str, Any] = None):
        """Print to console or log to JSON based on quiet mode."""
        if self.quiet_mode:
            if data:
                self._log_progress(test_name, stage, data)
        else:
            print(message)
            
    def validate_dual_functionality_molecules(self, count: int = 50) -> ValidationResult:
        """
        Validate dual-functionality molecule generation.
        
        Core claim: Every molecule functions as both precision clock AND processor.
        """
        self._print_or_log("\n🧪 Validating Dual-Functionality Molecules...", 
                          "dual_functionality", "start", {"molecule_count": count})
        
        try:
            # Generate molecules using Rust implementation
            result = self.borgia.generate_dual_functionality_molecules(
                count=count, 
                precision_target=1e-30
            )
            
            if not result["success"]:
                return ValidationResult(
                    "dual_functionality_molecules",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Generation failed")]
                )
            
            data = result["data"]
            molecules = data.get("molecules", [])
            
            # Validate each molecule has dual functionality
            clock_functional = 0
            processor_functional = 0
            dual_functional = 0
            
            precision_sum = 0.0
            processing_sum = 0.0
            
            for mol in molecules:
                has_clock = "clock_precision" in mol and mol["clock_precision"] <= 1e-29
                has_processor = "processing_capacity" in mol and mol["processing_capacity"] >= 1e5
                is_dual = mol.get("dual_functionality", False)
                
                if has_clock:
                    clock_functional += 1
                    precision_sum += mol["clock_precision"]
                
                if has_processor:
                    processor_functional += 1
                    processing_sum += mol["processing_capacity"]
                
                if has_clock and has_processor and is_dual:
                    dual_functional += 1
            
            # Calculate metrics
            total_molecules = len(molecules)
            clock_rate = clock_functional / total_molecules if total_molecules > 0 else 0
            processor_rate = processor_functional / total_molecules if total_molecules > 0 else 0
            dual_rate = dual_functional / total_molecules if total_molecules > 0 else 0
            
            avg_precision = precision_sum / clock_functional if clock_functional > 0 else 0
            avg_processing = processing_sum / processor_functional if processor_functional > 0 else 0
            
            # Success criteria: 100% dual functionality (zero tolerance)
            success = dual_rate == 1.0
            score = dual_rate
            
            metrics = {
                "total_molecules": total_molecules,
                "clock_functional_rate": clock_rate,
                "processor_functional_rate": processor_rate,
                "dual_functional_rate": dual_rate,
                "average_clock_precision": avg_precision,
                "average_processing_capacity": avg_processing,
                "generation_time": result.get("execution_time", 0),
                "requirement_met": success
            }
            
            errors = [] if success else [f"Dual functionality rate {dual_rate:.1%} < 100% required"]
            
            # Log intermediate results
            intermediate_data = {
                "total_molecules": total_molecules,
                "clock_functional_rate": clock_rate,
                "processor_functional_rate": processor_rate,
                "dual_functional_rate": dual_rate,
                "average_precision": avg_precision,
                "success": success
            }
            
            self._print_or_log(f"   ✓ Generated: {total_molecules} molecules", 
                              "dual_functionality", "progress", intermediate_data)
            self._print_or_log(f"   ✓ Clock functional: {clock_rate:.1%}")
            self._print_or_log(f"   ✓ Processor functional: {processor_rate:.1%}")
            self._print_or_log(f"   ✓ Dual functional: {dual_rate:.1%}")
            self._print_or_log(f"   ✓ Average precision: {avg_precision:.2e} seconds")
            self._print_or_log(f"   {'✅ PASS' if success else '❌ FAIL'}: Zero tolerance requirement", 
                              "dual_functionality", "result", {"success": success, "metrics": metrics})
            
            return ValidationResult("dual_functionality_molecules", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "dual_functionality_molecules", 
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_bmd_network_coordination(self) -> ValidationResult:
        """
        Validate multi-scale BMD network coordination.
        
        Core claim: >1000× amplification through quantum/molecular/environmental coordination.
        """
        self._print_or_log("\n🔬 Validating BMD Network Coordination...", "bmd_network", "start", {})
        
        try:
            # Test with sample molecules
            test_molecules = [f"test_mol_{i}" for i in range(20)]
            
            result = self.borgia.test_bmd_network_coordination(
                molecules=test_molecules,
                timescales=["quantum", "molecular", "environmental"]
            )
            
            if not result["success"]:
                return ValidationResult(
                    "bmd_network_coordination",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Coordination test failed")]
                )
            
            data = result["data"]
            
            # Extract key metrics
            quantum_eff = data.get("quantum_efficiency", 0.0)
            molecular_eff = data.get("molecular_efficiency", 0.0)
            environmental_eff = data.get("environmental_efficiency", 0.0)
            amplification = data.get("amplification_factor", 0.0)
            sync_quality = data.get("synchronization_quality", 0.0)
            
            # Success criteria: >1000× amplification, >90% efficiency
            amplification_ok = amplification >= 1000.0
            efficiency_ok = all(eff >= 0.9 for eff in [quantum_eff, molecular_eff, environmental_eff])
            sync_ok = sync_quality >= 0.9
            
            success = amplification_ok and efficiency_ok and sync_ok
            
            # Score based on amplification factor and efficiency
            amp_score = min(amplification / 1000.0, 1.0)
            eff_score = (quantum_eff + molecular_eff + environmental_eff) / 3.0
            score = (amp_score * 0.6 + eff_score * 0.4)
            
            metrics = {
                "quantum_efficiency": quantum_eff,
                "molecular_efficiency": molecular_eff,
                "environmental_efficiency": environmental_eff,
                "amplification_factor": amplification,
                "synchronization_quality": sync_quality,
                "amplification_requirement_met": amplification_ok,
                "efficiency_requirement_met": efficiency_ok,
                "coordination_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not amplification_ok:
                errors.append(f"Amplification {amplification:.1f}× < 1000× required")
            if not efficiency_ok:
                errors.append("Efficiency requirements not met")
            if not sync_ok:
                errors.append(f"Synchronization quality {sync_quality:.1%} < 90% required")
            
            # Log intermediate results
            intermediate_data = {
                "quantum_efficiency": quantum_eff,
                "molecular_efficiency": molecular_eff,
                "environmental_efficiency": environmental_eff,
                "amplification_factor": amplification,
                "synchronization_quality": sync_quality,
                "success": success
            }
            
            self._print_or_log(f"   ✓ Quantum efficiency: {quantum_eff:.1%}", 
                              "bmd_network", "progress", intermediate_data)
            self._print_or_log(f"   ✓ Molecular efficiency: {molecular_eff:.1%}")
            self._print_or_log(f"   ✓ Environmental efficiency: {environmental_eff:.1%}")
            self._print_or_log(f"   ✓ Amplification factor: {amplification:.1f}×")
            self._print_or_log(f"   ✓ Synchronization quality: {sync_quality:.1%}")
            self._print_or_log(f"   {'✅ PASS' if success else '❌ FAIL'}: BMD network coordination", 
                              "bmd_network", "result", {"success": success, "metrics": metrics})
            
            return ValidationResult("bmd_network_coordination", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "bmd_network_coordination",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_information_catalysis(self) -> ValidationResult:
        """
        Validate information catalysis and thermodynamic amplification.
        
        Core claim: >95% efficiency in information catalysis with >1000× amplification.
        """
        self._print_or_log("\n⚡ Validating Information Catalysis...", "information_catalysis", "start", {})
        
        try:
            result = self.borgia.validate_information_catalysis(
                input_entropy=10.0,
                target_entropy=1.0
            )
            
            if not result["success"]:
                return ValidationResult(
                    "information_catalysis",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Catalysis validation failed")]
                )
            
            data = result["data"]
            
            # Extract key metrics
            efficiency = data.get("efficiency", 0.0)
            amplification = data.get("amplification_factor", 0.0)
            entropy_reduction = data.get("entropy_reduction", 0.0)
            info_preservation = data.get("information_preservation", 0.0)
            
            # Success criteria: >95% efficiency, >1000× amplification, >98% info preservation
            efficiency_ok = efficiency >= 0.95
            amplification_ok = amplification >= 1000.0
            preservation_ok = info_preservation >= 0.98
            
            success = efficiency_ok and amplification_ok and preservation_ok
            
            # Score based on all factors
            eff_score = efficiency
            amp_score = min(amplification / 1000.0, 1.0)
            pres_score = info_preservation
            score = (eff_score * 0.4 + amp_score * 0.4 + pres_score * 0.2)
            
            metrics = {
                "efficiency": efficiency,
                "amplification_factor": amplification,
                "entropy_reduction": entropy_reduction,
                "information_preservation": info_preservation,
                "efficiency_requirement_met": efficiency_ok,
                "amplification_requirement_met": amplification_ok,
                "preservation_requirement_met": preservation_ok,
                "catalysis_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not efficiency_ok:
                errors.append(f"Efficiency {efficiency:.1%} < 95% required")
            if not amplification_ok:
                errors.append(f"Amplification {amplification:.1f}× < 1000× required")
            if not preservation_ok:
                errors.append(f"Information preservation {info_preservation:.1%} < 98% required")
            
            # Log intermediate results
            intermediate_data = {
                "efficiency": efficiency,
                "amplification_factor": amplification,
                "entropy_reduction": entropy_reduction,
                "information_preservation": info_preservation,
                "success": success
            }
            
            self._print_or_log(f"   ✓ Catalysis efficiency: {efficiency:.1%}", 
                              "information_catalysis", "progress", intermediate_data)
            self._print_or_log(f"   ✓ Amplification factor: {amplification:.1f}×")
            self._print_or_log(f"   ✓ Entropy reduction: {entropy_reduction:.2f} units")
            self._print_or_log(f"   ✓ Information preservation: {info_preservation:.1%}")
            self._print_or_log(f"   {'✅ PASS' if success else '❌ FAIL'}: Information catalysis", 
                              "information_catalysis", "result", {"success": success, "metrics": metrics})
            
            return ValidationResult("information_catalysis", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "information_catalysis",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_hardware_integration(self) -> ValidationResult:
        """
        Validate hardware integration capabilities.
        
        Core claim: Zero-cost LED spectroscopy with significant performance gains.
        """
        self._print_or_log("\n💻 Validating Hardware Integration...", "hardware_integration", "start", {})
        
        try:
            result = self.borgia.test_hardware_integration(
                led_wavelengths=[470, 525, 625]
            )
            
            if not result["success"]:
                return ValidationResult(
                    "hardware_integration",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Hardware integration test failed")]
                )
            
            data = result["data"]
            
            # Extract key metrics
            led_success = data.get("led_spectroscopy_success", False)
            cpu_success = data.get("cpu_timing_success", False)
            noise_success = data.get("noise_enhancement_success", False)
            performance_gain = data.get("performance_improvement", 0.0)
            memory_reduction = data.get("memory_reduction", 0.0)
            zero_cost = data.get("zero_cost_confirmed", False)
            
            # Success criteria: All components working, zero cost, performance gain
            components_ok = led_success and cpu_success and noise_success
            performance_ok = performance_gain >= 2.0
            zero_cost_ok = zero_cost
            
            success = components_ok and performance_ok and zero_cost_ok
            
            # Score based on performance and functionality
            component_score = (led_success + cpu_success + noise_success) / 3.0
            perf_score = min(performance_gain / 3.0, 1.0)
            cost_score = 1.0 if zero_cost else 0.0
            score = (component_score * 0.4 + perf_score * 0.4 + cost_score * 0.2)
            
            metrics = {
                "led_spectroscopy_success": led_success,
                "cpu_timing_success": cpu_success,
                "noise_enhancement_success": noise_success,
                "performance_improvement": performance_gain,
                "memory_reduction": memory_reduction,
                "zero_cost_confirmed": zero_cost,
                "components_requirement_met": components_ok,
                "performance_requirement_met": performance_ok,
                "integration_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not components_ok:
                errors.append("Not all hardware components functioning")
            if not performance_ok:
                errors.append(f"Performance gain {performance_gain:.1f}× < 2× required")
            if not zero_cost_ok:
                errors.append("Zero-cost operation not confirmed")
            
            # Log intermediate results
            intermediate_data = {
                "led_spectroscopy_success": led_success,
                "cpu_timing_success": cpu_success,
                "noise_enhancement_success": noise_success,
                "performance_improvement": performance_gain,
                "memory_reduction": memory_reduction,
                "zero_cost_confirmed": zero_cost,
                "success": success
            }
            
            self._print_or_log(f"   ✓ LED spectroscopy: {'✅' if led_success else '❌'}", 
                              "hardware_integration", "progress", intermediate_data)
            self._print_or_log(f"   ✓ CPU timing: {'✅' if cpu_success else '❌'}")
            self._print_or_log(f"   ✓ Noise enhancement: {'✅' if noise_success else '❌'}")
            self._print_or_log(f"   ✓ Performance improvement: {performance_gain:.1f}×")
            self._print_or_log(f"   ✓ Memory reduction: {memory_reduction:.1f}×")
            self._print_or_log(f"   ✓ Zero-cost confirmed: {'✅' if zero_cost else '❌'}")
            self._print_or_log(f"   {'✅ PASS' if success else '❌ FAIL'}: Hardware integration", 
                              "hardware_integration", "result", {"success": success, "metrics": metrics})
            
            return ValidationResult("hardware_integration", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "hardware_integration",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def run_comprehensive_validation(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive validation of all core claims.
        
        Args:
            quick_mode: Whether to run in quick mode
            
        Returns:
            Dictionary containing all validation results
        """
        if not self.quiet_mode:
            print("\n" + "="*60)
            print("🧬 BORGIA BMD FRAMEWORK VALIDATION")
            print("   Validating Core S-Entropy Claims")
            print("="*60)
        
        self._log_progress("comprehensive", "start", {"quick_mode": quick_mode})
        start_time = time.time()
        
        # Get system info
        try:
            sys_info = self.borgia.get_system_info()
            if sys_info["success"]:
                data = sys_info["data"]
                sys_data = {
                    "version": data.get('version', 'Unknown'),
                    "mode": data.get('mode', 'Unknown'),
                    "capabilities": data.get('capabilities', [])
                }
                self._log_progress("comprehensive", "system_info", sys_data)
                
                if not self.quiet_mode:
                    print(f"\n📋 System Info:")
                    print(f"   Version: {data.get('version', 'Unknown')}")
                    print(f"   Mode: {data.get('mode', 'Unknown')}")
                    if "capabilities" in data:
                        print(f"   Capabilities: {', '.join(data['capabilities'])}")
        except Exception:
            self._log_progress("comprehensive", "system_info", {"error": "not_available"})
            if not self.quiet_mode:
                print("\n📋 System Info: Not available")
        
        # Run core validations
        molecule_count = 20 if quick_mode else 50
        
        validations = [
            ("dual_functionality", lambda: self.validate_dual_functionality_molecules(molecule_count)),
            ("bmd_networks", self.validate_bmd_network_coordination),
            ("information_catalysis", self.validate_information_catalysis),
            ("hardware_integration", self.validate_hardware_integration)
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(validations)
        
        for test_name, test_func in validations:
            try:
                result = test_func()
                results[test_name] = result
                self.results[test_name] = result
                
                if result.success:
                    passed_tests += 1
                    
            except Exception as e:
                self._log_progress(test_name, "error", {"error": str(e)})
                if not self.quiet_mode:
                    print(f"\n❌ Validation {test_name} failed: {e}")
                error_result = ValidationResult(test_name, False, 0.0, {"error": str(e)}, [str(e)])
                results[test_name] = error_result
                self.results[test_name] = error_result
        
        # Calculate overall results
        execution_time = time.time() - start_time
        pass_rate = passed_tests / total_tests
        overall_score = sum(r.score for r in results.values()) / len(results) if results else 0
        
        # Critical claims validation
        critical_claims = ["dual_functionality", "information_catalysis"]
        critical_passed = all(results.get(claim, ValidationResult("", False, 0, {})).success 
                            for claim in critical_claims)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "overall_score": overall_score,
            "critical_claims_validated": critical_passed,
            "execution_time": execution_time,
            "results": {name: result.to_dict() for name, result in results.items()},
            "system_ready": critical_passed and pass_rate >= 0.75
        }
        
        # Log summary
        summary_data = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "overall_score": overall_score,
            "critical_claims_validated": critical_passed,
            "system_ready": summary['system_ready'],
            "execution_time": execution_time
        }
        self._log_progress("comprehensive", "summary", summary_data)
        
        # Print summary (if not quiet)
        if not self.quiet_mode:
            print("\n" + "="*60)
            print("📊 VALIDATION SUMMARY")
            print("="*60)
            print(f"✓ Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1%})")
            print(f"✓ Overall score: {overall_score:.3f}")
            print(f"✓ Critical claims: {'✅ VALIDATED' if critical_passed else '❌ FAILED'}")
            print(f"✓ System status: {'🚀 READY' if summary['system_ready'] else '⚠️  NEEDS ATTENTION'}")
            print(f"✓ Execution time: {execution_time:.1f} seconds")
        
        if not critical_passed:
            critical_issues = []
            for claim in critical_claims:
                if not results.get(claim, ValidationResult("", False, 0, {})).success:
                    critical_issues.append(claim.replace('_', ' ').title())
            
            self._log_progress("comprehensive", "critical_issues", {"failed_claims": critical_issues})
            
            if not self.quiet_mode:
                print(f"\n⚠️  Critical issues detected:")
                for claim in critical_issues:
                    print(f"   - {claim} validation failed")
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save validation results and progress data to files."""
        timestamp = int(time.time())
        
        try:
            # Save main results
            results_file = self.output_dir / f"validation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save detailed progress data
            progress_file = self.output_dir / f"progress_data_{timestamp}.json"
            with open(progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2, default=str)
            
            # Save combined data for easy access
            combined_data = {
                "results": results,
                "progress": self.progress_data,
                "timestamp": timestamp
            }
            combined_file = self.output_dir / f"complete_validation_{timestamp}.json"
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f, indent=2, default=str)
            
            if not self.quiet_mode:
                print(f"\n💾 Results saved to: {results_file}")
                print(f"💾 Progress saved to: {progress_file}")
                print(f"💾 Combined data saved to: {combined_file}")
            else:
                print(f"Results saved: {results_file.name}, {progress_file.name}, {combined_file.name}")
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"\n⚠️  Could not save results: {e}")
            else:
                print(f"ERROR: Could not save results: {e}")


def run_quick_validation() -> Dict[str, Any]:
    """Run a quick validation of core Borgia claims."""
    validator = BorgiaValidator()
    return validator.run_comprehensive_validation(quick_mode=True)
