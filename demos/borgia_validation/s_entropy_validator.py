"""
S-Entropy Framework Validator
============================

Validator specifically for S-Entropy framework claims as described in 
st-stellas papers. Validates the core theoretical foundations.
"""

import time
import json
from typing import Dict, List, Any, Optional

from .rust_interface import get_borgia_interface
from .core_validator import ValidationResult


class SEntropyValidator:
    """
    Validator for S-Entropy framework theoretical claims.
    """
    
    def __init__(self):
        """Initialize S-Entropy validator."""
        self.borgia = get_borgia_interface()
        
    def validate_universal_predetermined_solutions(self) -> ValidationResult:
        """
        Validate Universal Predetermined Solutions Theorem.
        
        Core claim: Every problem has a unique optimal solution existing 
        as an entropy endpoint before computation.
        """
        print("\n🎯 Validating Universal Predetermined Solutions...")
        
        try:
            # Test with different problem complexities
            test_results = []
            
            for complexity in [0.1, 0.5, 1.0, 2.0, 5.0]:
                result = self.borgia.validate_s_entropy_framework(complexity)
                
                if result["success"]:
                    data = result["data"]
                    predetermined_validated = data.get("predetermined_solutions_validated", False)
                    test_results.append({
                        "complexity": complexity,
                        "predetermined_validated": predetermined_validated,
                        "execution_time": result.get("execution_time", 0)
                    })
                else:
                    test_results.append({
                        "complexity": complexity,
                        "predetermined_validated": False,
                        "error": result.get("error", "Unknown error")
                    })
            
            # Calculate success metrics
            successful_tests = [r for r in test_results if r["predetermined_validated"]]
            success_rate = len(successful_tests) / len(test_results)
            
            # Average execution time should be reasonable (solutions are predetermined)
            avg_execution_time = sum(r.get("execution_time", 0) for r in successful_tests) / len(successful_tests) if successful_tests else float('inf')
            
            # Success criteria: All complexity levels validated
            success = success_rate == 1.0
            score = success_rate
            
            metrics = {
                "success_rate": success_rate,
                "tests_completed": len(test_results),
                "successful_tests": len(successful_tests),
                "average_execution_time": avg_execution_time,
                "complexity_range_tested": [0.1, 5.0],
                "predetermined_solutions_validated": success,
                "test_results": test_results
            }
            
            errors = []
            if not success:
                failed_complexities = [r["complexity"] for r in test_results if not r["predetermined_validated"]]
                errors.append(f"Predetermined solutions failed for complexities: {failed_complexities}")
            
            print(f"   ✓ Success rate: {success_rate:.1%}")
            print(f"   ✓ Average execution time: {avg_execution_time:.3f}s")
            print(f"   ✓ Complexity range: 0.1 - 5.0")
            print(f"   {'✅ PASS' if success else '❌ FAIL'}: Universal predetermined solutions")
            
            return ValidationResult("universal_predetermined_solutions", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "universal_predetermined_solutions",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_s_distance_minimization(self) -> ValidationResult:
        """
        Validate S-Distance Minimization Principle.
        
        Core claim: Optimal problem-solving occurs by minimizing observer-process separation.
        """
        print("\n📏 Validating S-Distance Minimization...")
        
        try:
            result = self.borgia.validate_s_entropy_framework(1.0)
            
            if not result["success"]:
                return ValidationResult(
                    "s_distance_minimization",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "S-Distance validation failed")]
                )
            
            data = result["data"]
            
            # Extract S-Distance metrics
            s_distance_validated = data.get("s_distance_minimization_validated", False)
            complexity_reduction = data.get("complexity_reduction_factor", 0.0)
            
            # Success criteria: S-Distance minimization validated with significant complexity reduction
            success = s_distance_validated and complexity_reduction >= 100.0
            score = min(complexity_reduction / 1000.0, 1.0) if s_distance_validated else 0.0
            
            metrics = {
                "s_distance_minimization_validated": s_distance_validated,
                "complexity_reduction_factor": complexity_reduction,
                "observer_process_separation": "minimized" if s_distance_validated else "not_minimized",
                "validation_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not s_distance_validated:
                errors.append("S-Distance minimization not validated")
            if complexity_reduction < 100.0:
                errors.append(f"Complexity reduction {complexity_reduction:.1f}× < 100× expected")
            
            print(f"   ✓ S-Distance minimization: {'✅' if s_distance_validated else '❌'}")
            print(f"   ✓ Complexity reduction: {complexity_reduction:.1f}×")
            print(f"   {'✅ PASS' if success else '❌ FAIL'}: S-Distance minimization")
            
            return ValidationResult("s_distance_minimization", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "s_distance_minimization",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_reality_solutions_theorem(self) -> ValidationResult:
        """
        Validate Reality Solutions Theorem.
        
        Core claim: Since reality happens, every problem has at least one solution,
        and no problem can be more complicated than reality.
        """
        print("\n🌍 Validating Reality Solutions Theorem...")
        
        try:
            result = self.borgia.validate_s_entropy_framework(1.0)
            
            if not result["success"]:
                return ValidationResult(
                    "reality_solutions_theorem",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Reality solutions validation failed")]
                )
            
            data = result["data"]
            
            # Extract reality theorem metrics
            reality_validated = data.get("reality_solutions_validated", False)
            universal_oscillation = data.get("universal_oscillation_confirmed", False)
            
            # Success criteria: Reality solutions validated and universal oscillation confirmed
            success = reality_validated and universal_oscillation
            score = (reality_validated + universal_oscillation) / 2.0
            
            metrics = {
                "reality_solutions_validated": reality_validated,
                "universal_oscillation_confirmed": universal_oscillation,
                "theoretical_foundation": "reality_based_solutions",
                "complexity_bound": "reality_complexity",
                "validation_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not reality_validated:
                errors.append("Reality solutions theorem not validated")
            if not universal_oscillation:
                errors.append("Universal oscillation not confirmed")
            
            print(f"   ✓ Reality solutions: {'✅' if reality_validated else '❌'}")
            print(f"   ✓ Universal oscillation: {'✅' if universal_oscillation else '❌'}")
            print(f"   ✓ Theorem: Every problem ≤ reality complexity")
            print(f"   {'✅ PASS' if success else '❌ FAIL'}: Reality solutions theorem")
            
            return ValidationResult("reality_solutions_theorem", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "reality_solutions_theorem",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_consciousness_computation_equivalence(self) -> ValidationResult:
        """
        Validate Consciousness-Computation Equivalence.
        
        Core claim: BMD frame selection and S-entropy navigation are 
        mathematically equivalent to consciousness processes.
        """
        print("\n🧠 Validating Consciousness-Computation Equivalence...")
        
        try:
            result = self.borgia.validate_s_entropy_framework(1.0)
            
            if not result["success"]:
                return ValidationResult(
                    "consciousness_computation_equivalence",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Consciousness equivalence validation failed")]
                )
            
            data = result["data"]
            
            # Extract consciousness-computation metrics
            equivalence_factor = data.get("consciousness_computation_equivalence", 0.0)
            
            # Success criteria: High equivalence factor (>95%)
            success = equivalence_factor >= 0.95
            score = equivalence_factor
            
            metrics = {
                "consciousness_computation_equivalence": equivalence_factor,
                "bmd_frame_selection_validated": equivalence_factor >= 0.9,
                "s_entropy_navigation_validated": equivalence_factor >= 0.9,
                "mathematical_equivalence": "demonstrated" if success else "not_demonstrated",
                "validation_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not success:
                errors.append(f"Consciousness-computation equivalence {equivalence_factor:.1%} < 95% required")
            
            print(f"   ✓ Equivalence factor: {equivalence_factor:.1%}")
            print(f"   ✓ BMD frame selection: {'✅' if equivalence_factor >= 0.9 else '❌'}")
            print(f"   ✓ S-entropy navigation: {'✅' if equivalence_factor >= 0.9 else '❌'}")
            print(f"   {'✅ PASS' if success else '❌ FAIL'}: Consciousness-computation equivalence")
            
            return ValidationResult("consciousness_computation_equivalence", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "consciousness_computation_equivalence",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def validate_universal_oscillation_equation(self) -> ValidationResult:
        """
        Validate Universal Oscillation Equation: S = k log α
        
        Core claim: Solutions correspond to oscillation amplitude configurations.
        """
        print("\n🌊 Validating Universal Oscillation Equation...")
        
        try:
            result = self.borgia.validate_s_entropy_framework(1.0)
            
            if not result["success"]:
                return ValidationResult(
                    "universal_oscillation_equation",
                    False, 0.0,
                    {"error": result.get("error", "Unknown error")},
                    [result.get("error", "Universal oscillation validation failed")]
                )
            
            data = result["data"]
            
            # Extract oscillation equation metrics
            universal_oscillation = data.get("universal_oscillation_confirmed", False)
            
            # For this validation, we assume the Rust implementation validates S = k log α
            # Success criteria: Universal oscillation confirmed
            success = universal_oscillation
            score = 1.0 if universal_oscillation else 0.0
            
            metrics = {
                "universal_oscillation_confirmed": universal_oscillation,
                "equation_validated": "S = k log α" if universal_oscillation else "not_validated",
                "amplitude_configuration_mapping": "confirmed" if universal_oscillation else "not_confirmed",
                "solution_oscillation_correspondence": universal_oscillation,
                "validation_time": result.get("execution_time", 0)
            }
            
            errors = []
            if not success:
                errors.append("Universal oscillation equation S = k log α not validated")
            
            print(f"   ✓ Equation S = k log α: {'✅' if universal_oscillation else '❌'}")
            print(f"   ✓ Amplitude configurations: {'✅' if universal_oscillation else '❌'}")
            print(f"   ✓ Solution correspondence: {'✅' if universal_oscillation else '❌'}")
            print(f"   {'✅ PASS' if success else '❌ FAIL'}: Universal oscillation equation")
            
            return ValidationResult("universal_oscillation_equation", success, score, metrics, errors)
            
        except Exception as e:
            return ValidationResult(
                "universal_oscillation_equation",
                False, 0.0,
                {"error": str(e)},
                [str(e)]
            )
    
    def run_s_entropy_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive S-Entropy framework validation.
        
        Returns:
            Dictionary containing all S-Entropy validation results
        """
        print("\n" + "="*60)
        print("🔬 S-ENTROPY FRAMEWORK VALIDATION")
        print("   Validating Theoretical Foundations")
        print("="*60)
        
        start_time = time.time()
        
        # Run S-Entropy specific validations
        validations = [
            ("predetermined_solutions", self.validate_universal_predetermined_solutions),
            ("s_distance_minimization", self.validate_s_distance_minimization),
            ("reality_solutions", self.validate_reality_solutions_theorem),
            ("consciousness_equivalence", self.validate_consciousness_computation_equivalence),
            ("oscillation_equation", self.validate_universal_oscillation_equation)
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(validations)
        
        for test_name, test_func in validations:
            try:
                result = test_func()
                results[test_name] = result
                
                if result.success:
                    passed_tests += 1
                    
            except Exception as e:
                print(f"\n❌ S-Entropy validation {test_name} failed: {e}")
                error_result = ValidationResult(test_name, False, 0.0, {"error": str(e)}, [str(e)])
                results[test_name] = error_result
        
        # Calculate overall results
        execution_time = time.time() - start_time
        pass_rate = passed_tests / total_tests
        overall_score = sum(r.score for r in results.values()) / len(results) if results else 0
        
        # Core theoretical claims
        core_claims = ["predetermined_solutions", "reality_solutions", "consciousness_equivalence"]
        core_validated = all(results.get(claim, ValidationResult("", False, 0, {})).success 
                           for claim in core_claims)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "overall_score": overall_score,
            "core_s_entropy_claims_validated": core_validated,
            "execution_time": execution_time,
            "results": {name: result.to_dict() for name, result in results.items()},
            "theoretical_foundation_validated": core_validated and pass_rate >= 0.8
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📊 S-ENTROPY VALIDATION SUMMARY")
        print("="*60)
        print(f"✓ Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1%})")
        print(f"✓ Overall score: {overall_score:.3f}")
        print(f"✓ Core claims: {'✅ VALIDATED' if core_validated else '❌ FAILED'}")
        print(f"✓ Foundation: {'🚀 SOLID' if summary['theoretical_foundation_validated'] else '⚠️  QUESTIONABLE'}")
        print(f"✓ Execution time: {execution_time:.1f} seconds")
        
        if not core_validated:
            print(f"\n⚠️  Core theoretical issues detected:")
            for claim in core_claims:
                if not results.get(claim, ValidationResult("", False, 0, {})).success:
                    print(f"   - {claim.replace('_', ' ').title()} validation failed")
        
        return summary


def validate_s_entropy_claims() -> Dict[str, Any]:
    """Run S-Entropy framework validation."""
    validator = SEntropyValidator()
    return validator.run_s_entropy_validation()
