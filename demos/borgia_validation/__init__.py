"""
Borgia Validation Framework
==========================

Simplified validation framework for the Borgia Rust BMD implementation.
Validates core S-Entropy framework claims through direct Rust interface.
"""

from .rust_interface import BorgiaRustInterface, MockBorgiaInterface, get_borgia_interface
from .core_validator import BorgiaValidator
from .s_entropy_validator import SEntropyValidator
from .simple_visualizer import SimpleVisualizer

__version__ = "1.0.0"
__all__ = [
    "BorgiaRustInterface",
    "MockBorgiaInterface", 
    "get_borgia_interface",
    "BorgiaValidator",
    "SEntropyValidator",
    "SimpleVisualizer"
]
