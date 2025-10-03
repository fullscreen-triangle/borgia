# 🔧 Installation Notes

## RDKit Installation

**✅ FIXED**: We now use the standard `rdkit` package instead of `rdkit-pypi` to avoid installation headaches.

### Standard Installation:
```bash
# Install via conda (recommended for RDKit)
conda install -c conda-forge rdkit

# Or via pip (now using standard rdkit)
pip install rdkit

# Then install gonfanolier
pip install -e .
```

### Alternative Installation Methods:

#### Method 1: Conda Environment (Recommended)
```bash
# Create conda environment with RDKit
conda create -n gonfanolier python=3.9 rdkit -c conda-forge
conda activate gonfanolier

# Install gonfanolier
cd gonfanolier
pip install -e .
```

#### Method 2: Pip with Standard RDKit
```bash
# Create virtual environment
python -m venv gonfanolier_env
source gonfanolier_env/bin/activate  # On Windows: gonfanolier_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install gonfanolier
pip install -e .
```

#### Method 3: Full Installation with Optional Dependencies
```bash
# Install with all optional dependencies
pip install -e ".[full]"

# Or install specific optional groups
pip install -e ".[dev,viz,docs]"
```

## Why We Changed from rdkit-pypi

- **rdkit-pypi**: Often has dependency conflicts, installation issues, and version mismatches
- **rdkit**: Standard package, better maintained, fewer installation headaches
- **conda-forge rdkit**: Most reliable installation method for RDKit

## Troubleshooting

If you encounter RDKit installation issues:

1. **Use conda**: `conda install -c conda-forge rdkit`
2. **Check Python version**: RDKit requires Python 3.7+
3. **Update pip**: `pip install --upgrade pip`
4. **Clear cache**: `pip cache purge`

## Verification

Test your installation:
```python
# Test RDKit import
from rdkit import Chem
from rdkit.Chem import Descriptors

# Test gonfanolier import
import gonfanolier
print(f"Gonfanolier version: {gonfanolier.__version__}")

# Run validation test
gonfanolier-validate --test
```

## Dependencies Updated

- ✅ `pyproject.toml`: Changed `rdkit-pypi>=2022.3.1` → `rdkit>=2022.3.1`
- ✅ `requirements.txt`: Added `rdkit>=2022.3.1`
- ✅ `setup.py`: Automatically uses updated requirements.txt

This ensures consistent, reliable RDKit installation across all installation methods.
