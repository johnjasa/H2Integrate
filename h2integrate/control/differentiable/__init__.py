"""Differentiable optimization tools for H2Integrate.

This subpackage holds the H2I-owned adapter that differentiates parametric Pyomo optimization
problems through their Karush-Kuhn-Tucker (KKT) conditions, so that optimal dispatch or control
problems can provide exact derivatives to OpenMDAO. It requires the optional ``dol`` dependencies
(``pip install -e ".[dol]"`` followed by ``idaes get-extensions``).
"""

from h2integrate.control.differentiable.kkt_sensitivity import (
    KKTSolution,
    KKTSensitivity,
    KKTLinearization,
    kkt_tools_available,
)
