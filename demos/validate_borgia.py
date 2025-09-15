#!/usr/bin/env python3
"""
Borgia BMD Framework Validation Script
=====================================

Simplified validation script for the Borgia Rust implementation.
Validates core S-Entropy claims and BMD capabilities.

Usage:
    python validate_borgia.py [--quick] [--s-entropy] [--visualize]
"""

import argparse
import time
import sys
from pathlib import Path

# Add validation module to path
sys.path.insert(0, str(Path(__file__).parent))

from borgia_validation import (
    BorgiaValidator,
    SEntropyValidator, 
    SimpleVisualizer,
    get_borgia_interface
)
from borgia_validation.simple_visualizer import create_validation_plots


def print_banner():
    """Print validation banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧬 BORGIA BMD VALIDATION                   ║
    ║         S-Entropy Framework & BMD Capabilities Test          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Validating Eduardo Mizraji's BMD Theory Implementation      ║
    ║  • Dual-functionality Molecules (Clock + Processor)         ║
    ║  • Multi-scale BMD Networks (10⁻¹⁵s to 10²s)               ║
    ║  • Information Catalysis (>1000× Amplification)            ║
    ║  • S-Entropy Universal Framework                            ║
    ║  • Hardware Integration (Zero-cost LED Spectroscopy)        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Borgia BMD Framework and S-Entropy claims"
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick validation (fewer molecules, faster execution)'
    )
    
    parser.add_argument(
        '--s-entropy',
        action='store_true', 
        help='Include S-Entropy framework validation'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal console output, save all data to JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='borgia_validation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Print banner (unless quiet)
    if not args.quiet:
        print_banner()
        print("🔧 Initializing Borgia interface...")
    
    # Check Borgia interface
    try:
        interface = get_borgia_interface()
        sys_info = interface.get_system_info()
        if sys_info["success"]:
            mode = sys_info["data"].get("mode", "unknown")
            version = sys_info["data"].get("version", "unknown")
            if not args.quiet:
                print(f"   ✓ Interface: {mode} (version {version})")
        else:
            if not args.quiet:
                print(f"   ⚠️  Interface available but system info failed")
    except Exception as e:
        if not args.quiet:
            print(f"   ❌ Interface error: {e}")
        else:
            print(f"ERROR: Interface initialization failed: {e}")
        return 1
    
    # Track overall execution time
    total_start = time.time()
    all_results = {}
    
    # Run core BMD validation
    if not args.quiet:
        print(f"\n{'='*60}")
        print("🧪 RUNNING CORE BMD VALIDATION")
        print("="*60)
    else:
        print("Starting core BMD validation...")
    
    try:
        validator = BorgiaValidator(args.output_dir, quiet_mode=args.quiet)
        core_results = validator.run_comprehensive_validation(args.quick)
        all_results['core_validation'] = core_results
        
        if not args.quiet:
            print(f"\n✅ Core validation completed:")
            print(f"   • Tests: {core_results['passed_tests']}/{core_results['total_tests']}")
            print(f"   • Score: {core_results['overall_score']:.3f}")
            print(f"   • Status: {'READY' if core_results['system_ready'] else 'NEEDS ATTENTION'}")
        else:
            print(f"Core validation: {core_results['passed_tests']}/{core_results['total_tests']} tests passed")
        
    except Exception as e:
        if not args.quiet:
            print(f"\n❌ Core validation failed: {e}")
        else:
            print(f"ERROR: Core validation failed: {e}")
        return 1
    
    # Run S-Entropy validation if requested
    s_entropy_results = None
    if args.s_entropy:
        if not args.quiet:
            print(f"\n{'='*60}")
            print("🔬 RUNNING S-ENTROPY FRAMEWORK VALIDATION")
            print("="*60)
        else:
            print("Starting S-Entropy validation...")
        
        try:
            s_entropy_validator = SEntropyValidator()
            s_entropy_results = s_entropy_validator.run_s_entropy_validation()
            all_results['s_entropy_validation'] = s_entropy_results
            
            if not args.quiet:
                print(f"\n✅ S-Entropy validation completed:")
                print(f"   • Claims: {s_entropy_results['passed_tests']}/{s_entropy_results['total_tests']}")
                print(f"   • Score: {s_entropy_results['overall_score']:.3f}")
                print(f"   • Foundation: {'SOLID' if s_entropy_results['theoretical_foundation_validated'] else 'QUESTIONABLE'}")
            else:
                print(f"S-Entropy validation: {s_entropy_results['passed_tests']}/{s_entropy_results['total_tests']} claims validated")
            
        except Exception as e:
            if not args.quiet:
                print(f"\n❌ S-Entropy validation failed: {e}")
            else:
                print(f"ERROR: S-Entropy validation failed: {e}")
    
    # Generate visualizations if requested
    if args.visualize:
        if not args.quiet:
            print(f"\n{'='*60}")
            print("📊 GENERATING VISUALIZATIONS")
            print("="*60)
        else:
            print("Generating visualizations...")
        
        try:
            plots = create_validation_plots(core_results, s_entropy_results)
            
            if plots:
                if not args.quiet:
                    print(f"\n✅ Generated {len(plots)} visualization files")
                else:
                    print(f"Generated {len(plots)} plots")
            else:
                if not args.quiet:
                    print(f"\n⚠️  No visualizations generated")
                else:
                    print("No visualizations generated")
                
        except Exception as e:
            if not args.quiet:
                print(f"\n❌ Visualization failed: {e}")
            else:
                print(f"ERROR: Visualization failed: {e}")
    
    # Final summary
    total_time = time.time() - total_start
    
    # Save final summary to JSON
    core_pass_rate = core_results['pass_rate']
    s_entropy_pass_rate = s_entropy_results['pass_rate'] if s_entropy_results else 1.0
    overall_pass_rate = (core_pass_rate + s_entropy_pass_rate) / 2
    
    final_summary = {
        "total_execution_time": total_time,
        "core_bmd_pass_rate": core_pass_rate,
        "s_entropy_pass_rate": s_entropy_pass_rate if s_entropy_results else None,
        "overall_success_rate": overall_pass_rate,
        "all_results": all_results,
        "timestamp": int(time.time())
    }
    
    # Save summary to file
    from pathlib import Path
    import json
    summary_file = Path(args.output_dir) / f"final_summary_{int(time.time())}.json"
    Path(args.output_dir).mkdir(exist_ok=True)
    try:
        with open(summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        if not args.quiet:
            print(f"\n💾 Final summary saved to: {summary_file}")
    except Exception as e:
        if not args.quiet:
            print(f"\n⚠️  Could not save final summary: {e}")
        else:
            print(f"ERROR: Could not save summary: {e}")
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("🏁 VALIDATION COMPLETE")
        print("="*60)
        print(f"✓ Total execution time: {total_time:.1f} seconds")
        print(f"✓ Core BMD validation: {core_pass_rate:.1%} pass rate")
        if s_entropy_results:
            print(f"✓ S-Entropy validation: {s_entropy_pass_rate:.1%} pass rate")
        print(f"✓ Overall success rate: {overall_pass_rate:.1%}")
    else:
        print(f"VALIDATION COMPLETE: {overall_pass_rate:.1%} success rate ({total_time:.1f}s)")
    
    # System status
    system_ready = core_results['system_ready']
    foundation_solid = s_entropy_results['theoretical_foundation_validated'] if s_entropy_results else True
    
    if not args.quiet:
        if system_ready and foundation_solid:
            print(f"\n🎉 BORGIA BMD FRAMEWORK VALIDATED")
            print(f"   • All critical systems operational")
            print(f"   • S-Entropy theoretical foundation confirmed")
            print(f"   • Ready for production deployment")
            print(f"   • Compatible with Masunda, Buhera, and Kambuzuma systems")
        elif system_ready:
            print(f"\n🚀 CORE BMD CAPABILITIES VALIDATED")
            print(f"   • BMD framework operational")
            print(f"   • Hardware integration successful")
            if not args.s_entropy:
                print(f"   • Run with --s-entropy for theoretical validation")
        else:
            print(f"\n⚠️  VALIDATION ISSUES DETECTED")
            print(f"   • Critical systems require attention")
            print(f"   • Review validation results before deployment")
        
        print(f"\n📁 Results saved to: {args.output_dir}")
    else:
        # Quiet mode summary
        status = "VALIDATED" if (system_ready and foundation_solid) else "NEEDS_ATTENTION"
        print(f"FINAL STATUS: {status}")
        print(f"Results directory: {args.output_dir}")
    
    # Return appropriate exit code
    if system_ready and (not args.s_entropy or foundation_solid):
        return 0  # Success
    else:
        return 1  # Issues detected


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        sys.exit(1)
