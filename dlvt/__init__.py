"""
Dynamic Leadership Vitality Theory (DLVT) — Python Package
===========================================================

A dynamical systems model of executive sustainability that integrates
organisational complexity dynamics with individual energy constraints.

Paper
-----
  "Dynamic Leadership Vitality Theory: A Formal Model of the Zombie-Leader Equilibrium"
  W. Bendinelli (2026)
  Manuscript submitted to The Leadership Quarterly
  JEL Codes: M12, D23, J24, L23

Core Concept
------------
Leadership sustainability depends on balancing two fundamental mechanisms:
  1. Recovery: Natural vitality renewal following logistic saturation kinetics
  2. Drain: Organisational complexity burden with nonlinear cost structure

Career capital drives endogenous complexity growth, which ultimately constrains available
vitality. The model predicts three distinct outcome regimes:
  - Sustainable: Leaders reach equilibrium with sufficient energy for strategic work
  - Zombie: Leaders maintain capital-building but with chronically insufficient vitality
  - Collapse-Prone: Unsustainable trajectories where burnout becomes inevitable

Mathematical Framework
---------------------
The DLVT system is governed by two coupled ODEs (Equations 3.3–3.4):

    dV/dt = R·(1 − V/V_max) − δ·O^γ · V/(V + ε)
    dC/dt = α·I − μ·C

Where:
    O(t) = O₀ + β·C(t)^η             (organisational complexity)
    I(t) = C·V / (1 + φ·O)            (energy-gated leadership impact)
    Γ(t) = δ·O^γ / R                  (depletion ratio; > 1 ⟹ net drain)

All 11 parameters are detailed in Table 1; default values: C₀=5.0.

Quick Start
-----------
    from dlvt import make_params, simulate, carrying_capacity
    from dlvt.analysis import find_interior_equilibria, classify_regime

    # Create parameter set
    p = make_params()

    # Simulate trajectory from initial state
    t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=120)

    # Find all equilibria and their stability
    equilibria = find_interior_equilibria(p)
    for eq in equilibria:
        status = "stable" if eq['stable'] else "unstable"
        print(f"V*={eq['V']:.2f}, C*={eq['C']:.2f}, {status}")

    # Regime classification and carrying capacity
    regime = classify_regime(p)
    C_max = carrying_capacity(p)

Main Modules
------------
model       : Core ODE system, parameters, numerical integration
analysis    : Equilibrium theory, stability analysis, bifurcations, regime classification
figures     : Publication figure generation (Figures 1–7 from paper)

Core Functions (Public API)
----------------------------
MODEL
  make_params(**overrides)            : Create parameter dictionary
  complexity(C, p)                    : Calculate organisational complexity
  impact(V, C, O, p)                  : Calculate energy-gated leadership impact
  simulate(p, V0=8, C0=5, T=120)      : Integrate ODE system over time

ANALYSIS
  find_interior_equilibria(p)         : Find all (V*, C*) fixed points with V*, C* > 0
  carrying_capacity(p)                : Maximum sustainable career capital C*_max
  jacobian_eigenvalues(V, C, p)       : Stability classification at equilibrium
  is_zombie(V_star, p)                : Check if V* < V_strategic (Definition 7)
  classify_regime(p)                  : Classify into 'sustainable', 'zombie', or 'collapse-prone'
  regime_map(beta_range, delta_range) : Generate (β, δ) regime classification grid

FIGURES
  fig1, fig2, ..., fig7               : Generate publication figures 1–7

Module Version
--------------
__version__ = "2.1.0"
__author__  = "W. Bendinelli"
"""

from .model import (
    DEFAULT_PARAMS, make_params, complexity, impact,
    dlvt_system, dlvt_exogenous, simulate
)
from .analysis import (
    carrying_capacity, trapping_capital_bound, find_interior_equilibria,
    jacobian_eigenvalues, is_zombie, classify_regime, regime_map
)

__version__ = "2.1.0"
__author__ = "W. Bendinelli"
__all__ = [
    # Model
    'DEFAULT_PARAMS', 'make_params', 'complexity', 'impact',
    'dlvt_system', 'dlvt_exogenous', 'simulate',
    # Analysis
    'carrying_capacity', 'trapping_capital_bound', 'find_interior_equilibria',
    'jacobian_eigenvalues', 'is_zombie', 'classify_regime', 'regime_map',
]
