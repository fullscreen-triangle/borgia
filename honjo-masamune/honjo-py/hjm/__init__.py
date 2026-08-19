"""hjm --- Honjo Masamune for Python.

Two small languages sharing one semantic core:

  * **masamune** translates chemical structure records into contact
    graphs, tagging every element ``stated`` or ``supplied`` and
    returning a labelled verdict rather than a graph-or-nothing.
  * **honjo** computes on those graphs with a single primitive, the cut,
    carrying a floor and a provenance tag on every value.

Both produce JSON.

    >>> from hjm import run_honjo, run_masamune
    >>> out = run_honjo('floor 1.0\\nC := cut 6\\nobserve C')
    >>> out["bindings"][0]["symbol"]
    'C'
"""

from .core import (
    MEDIUM, Atom, Contact, ContactGraph, Label, Prov, Verdict,
)
from .honjo import Interpreter, Value, run_honjo
from .masamune import (
    capability, run_plan, translate, translate_smiles, translate_xyz,
)

__version__ = "0.1.0"

#: alias: running a masamune plan
run_masamune = run_plan

__all__ = [
    "MEDIUM", "Atom", "Contact", "ContactGraph", "Label", "Prov", "Verdict",
    "Interpreter", "Value", "run_honjo",
    "run_masamune", "run_plan", "translate", "translate_smiles",
    "translate_xyz", "capability",
    "__version__",
]
