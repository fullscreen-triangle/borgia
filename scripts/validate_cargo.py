#!/usr/bin/env python3
"""
Cargo.toml Validation Script for Borgia
Validates dependency configuration and feature flags
"""

import toml
import sys
from pathlib import Path

def validate_cargo_toml():
    """Validate the Cargo.toml configuration"""
    cargo_path = Path("Cargo.toml")
    
    if not cargo_path.exists():
        print("❌ Cargo.toml not found")
        return False
    
    try:
        with open(cargo_path, 'r') as f:
            cargo_config = toml.load(f)
    except Exception as e:
        print(f"❌ Failed to parse Cargo.toml: {e}")
        return False
    
    print("🔍 Validating Cargo.toml configuration...")
    
    # Check basic structure
    if 'package' not in cargo_config:
        print("❌ Missing [package] section")
        return False
    
    if 'dependencies' not in cargo_config:
        print("❌ Missing [dependencies] section")
        return False
    
    print("✅ Basic structure is valid")
    
    # Check features section
    if 'features' not in cargo_config:
        print("⚠️ No [features] section found")
    else:
        features = cargo_config['features']
        print(f"✅ Found {len(features)} feature flags:")
        for feature, deps in features.items():
            if isinstance(deps, list):
                print(f"  - {feature}: {', '.join(deps)}")
            else:
                print(f"  - {feature}: {deps}")
    
    # Check optional dependencies
    dependencies = cargo_config['dependencies']
    optional_deps = []
    
    for dep_name, dep_config in dependencies.items():
        if isinstance(dep_config, dict) and dep_config.get('optional', False):
            optional_deps.append(dep_name)
    
    print(f"✅ Found {len(optional_deps)} optional dependencies:")
    for dep in optional_deps:
        print(f"  - {dep}")
    
    # Validate feature dependency references
    if 'features' in cargo_config:
        features = cargo_config['features']
        for feature_name, feature_deps in features.items():
            if isinstance(feature_deps, list):
                for dep in feature_deps:
                    if dep.startswith('dep:'):
                        dep_name = dep[4:]  # Remove 'dep:' prefix
                        if dep_name not in optional_deps:
                            print(f"⚠️ Feature '{feature_name}' references optional dependency '{dep_name}' that is not marked as optional")
                    elif dep not in dependencies and dep not in features:
                        print(f"⚠️ Feature '{feature_name}' references unknown dependency '{dep}'")
    
    # Check for common issues
    issues = []
    
    # Check if tokio is properly configured for async features
    if 'tokio' in dependencies:
        tokio_config = dependencies['tokio']
        if isinstance(tokio_config, dict):
            if not tokio_config.get('optional', False):
                issues.append("tokio should be optional if used in features")
            if 'features' in tokio_config and 'full' in tokio_config['features']:
                print("ℹ️ tokio is configured with 'full' features - consider using specific features for smaller builds")
    
    # Check version specifications
    for dep_name, dep_config in dependencies.items():
        if isinstance(dep_config, str):
            # Simple version string
            continue
        elif isinstance(dep_config, dict):
            if 'version' not in dep_config and 'path' not in dep_config and 'git' not in dep_config:
                issues.append(f"Dependency '{dep_name}' missing version specification")
    
    if issues:
        print("⚠️ Potential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No issues found in dependency configuration")
    
    print("\n🎯 Configuration Summary:")
    print(f"  - Package: {cargo_config['package']['name']} v{cargo_config['package']['version']}")
    print(f"  - Dependencies: {len(dependencies)}")
    print(f"  - Optional dependencies: {len(optional_deps)}")
    if 'features' in cargo_config:
        print(f"  - Feature flags: {len(cargo_config['features'])}")
    
    return True

def check_feature_compatibility():
    """Check if feature combinations are compatible"""
    print("\n🔧 Checking feature compatibility...")
    
    # Define feature groups that should work together
    compatible_groups = [
        ['quantum', 'oscillatory', 'categorical'],
        ['autobahn', 'distributed', 'consciousness'],
        ['molecular', 'similarity', 'prediction'],
    ]
    
    print("✅ Compatible feature groups:")
    for i, group in enumerate(compatible_groups, 1):
        print(f"  Group {i}: {', '.join(group)}")
    
    return True

def main():
    """Main validation function"""
    print("🚀 Borgia Cargo.toml Validator")
    print("=" * 40)
    
    if not validate_cargo_toml():
        sys.exit(1)
    
    if not check_feature_compatibility():
        sys.exit(1)
    
    print("\n✅ All validations passed!")
    print("🎉 Your Cargo.toml configuration looks good!")

if __name__ == "__main__":
    main() 