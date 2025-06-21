# Borgia Scripts Directory

This directory contains utility scripts for development, validation, and maintenance of the Borgia quantum-oscillatory molecular analysis system.

## Scripts

### `validate_cargo.py`
Validates the Cargo.toml configuration file for:
- Proper dependency structure
- Optional dependency configuration
- Feature flag compatibility
- Version specifications
- Common configuration issues

**Usage:**
```bash
python3 scripts/validate_cargo.py
# or
make validate-cargo
```

**Features:**
- ✅ Validates basic Cargo.toml structure
- ✅ Checks optional dependencies are properly marked
- ✅ Validates feature flag references
- ✅ Identifies potential configuration issues
- ✅ Provides compatibility recommendations

## Requirements

The validation scripts require:
- Python 3.7+
- `toml` package (install with `pip install toml`)

## Integration

These scripts are integrated into the Makefile:
- `make validate-cargo` - Run Cargo.toml validation
- `make check` - Run all checks including validation

## Development

When adding new scripts:
1. Make them executable: `chmod +x scripts/your_script.py`
2. Add proper shebang: `#!/usr/bin/env python3`
3. Add documentation to this README
4. Integrate into Makefile if appropriate
5. Test thoroughly before committing

## Theoretical Framework Integration

These scripts support the Borgia quantum-oscillatory framework by ensuring:
- Proper dependency management for distributed intelligence
- Correct feature flag configuration for categorical predeterminism
- Optimal build configuration for membrane quantum computation
- Reliable development environment for oscillatory analysis

The validation tools embody the predetermined nature of the system by ensuring deterministic configuration states and preventing configuration drift through automated validation. 