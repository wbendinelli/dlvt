"""
dlvt.model
==========
Core ODE system and numerical integration for the Dynamic Leadership Vitality Theory (DLVT).

This module implements the fundamental dynamical system described in Equations 3.3–3.4
of "The Carrying Capacity of Leadership: A Dynamic Systems Model of Executive Sustainability"
(W. Bendinelli, 2026).

State Variables
---------------
V(t)  : Subjective vitality (energy available for leadership work)
        Range: [0, V_max], measured in arbitrary energy units
C(t)  : Career capital (accumulated human, social, relational capital)
        Range: [0, ∞), measured in capital units

Derived Quantities
------------------
O(t) = O₀ + β·C(t)^η
        Organisational complexity; increases with career capital (Eq. 3.1)
I(t) = C·V / (1 + φ·O)
        Energy-gated leadership impact; suppressed by complexity (Eq. 3.2)
Γ(t) = δ·O^γ / R
        Depletion ratio; > 1 indicates net vitality drain (Definition 5)

Dynamical System (Equations 3.3–3.4)
------------------------------------
The core model couples vitality recovery (logistic saturation) with complexity-driven drain:

    dV/dt = R·(1 − V/V_max) − δ·O^γ · V/(V + ε)
    dC/dt = α·I − μ·C

The smooth barrier V/(V + ε) ensures positive invariance of V for ε > 0.
Theoretical results use the ε → 0 limit; numerical integration uses ε = 0.1 for stability
(this causes < 2% shift in equilibrium values compared to ε → 0).

Default Parameters (Table 1)
----------------------------
All defaults are from Table 1 in the paper. C₀ = 5.0 throughout all figures.
  R=3.0       : Vitality recovery rate
  Vmax=10.0   : Maximum vitality capacity
  delta=0.02  : Energetic cost coefficient
  gamma=2.0   : Complexity exponent in drain term
  O0=1.0      : Baseline organisational complexity
  beta=0.25   : Capital-complexity coupling strength
  eta=1.0     : Capital scaling exponent in complexity
  alpha=0.1   : Capital accumulation rate
  phi=0.15    : Complexity-impact suppression coefficient
  mu=0.2      : Capital depreciation rate
  eps=0.1     : Smooth barrier regularisation (ε → 0 for analytics)

References
----------
  Bendinelli, W. (2026). The Carrying Capacity of Leadership: A Dynamic Systems Model
  of Executive Sustainability. SSRN Working Paper.
"""

from typing import Dict, List, Tuple, Union, Callable
import numpy as np
from scipy.integrate import solve_ivp

# ── Default parameter set ─────────────────────────────────────────────────────

DEFAULT_PARAMS = dict(
    R=3.0,       # vitality recovery rate
    Vmax=10.0,   # maximum vitality capacity
    delta=0.02,  # energetic cost coefficient
    gamma=2.0,   # complexity exponent in drain term
    O0=1.0,      # baseline (irreducible) complexity
    beta=0.25,   # capital-complexity coupling strength
    eta=1.0,     # capital scaling exponent in complexity
    alpha=0.1,   # capital accumulation rate
    phi=0.15,    # complexity-impact suppression coefficient
    mu=0.2,      # capital depreciation rate
    eps=0.1,     # smooth barrier regularisation (ε → 0 for analytics)
)


def make_params(**overrides: float) -> Dict[str, float]:
    """Return a copy of DEFAULT_PARAMS with keyword overrides applied.

    Parameters
    ----------
    **overrides : float
        Parameter overrides as keyword arguments.

    Returns
    -------
    Dict[str, float]
        Updated parameter dictionary.

    Example
    -------
    >>> p = make_params(beta=0.5, R=4.0)
    >>> p['beta']
    0.5
    """
    p = DEFAULT_PARAMS.copy()
    p.update(overrides)
    return p


# ── Core functions ────────────────────────────────────────────────────────────

def complexity(C: Union[float, np.ndarray],
               p: Dict[str, float]) -> Union[float, np.ndarray]:
    """Organisational complexity O = O₀ + β·C^η.

    Models how career capital drives organisational complexity growth.
    The smooth nonlinear relationship allows scaling from low-responsibility
    roles (low C, low O) to high-pressure executive positions.

    Parameters
    ----------
    C : float or ndarray
        Career capital, typically in [0, 100] range (non-negative).
    p : Dict[str, float]
        Parameter dictionary with keys: 'O0', 'beta', 'eta'.

    Returns
    -------
    float or ndarray
        Complexity value(s), shape and type matching input C.
        Guaranteed non-negative.

    Notes
    -----
    With default parameters (O0=1.0, beta=0.25, eta=1.0):
      - At C=0: O = 1.0 (baseline complexity)
      - At C=10: O = 3.5 (elevated complexity)
      - At C=50: O = 13.5 (high pressure environment)
    """
    return p['O0'] + p['beta'] * np.power(np.maximum(C, 0.0), p['eta'])


def impact(V: float, C: float, O: float,
           p: Dict[str, float]) -> float:
    """Energy-gated leadership impact I = C·V / (1 + φ·O).

    Models the reduction in effectiveness when complexity is high.
    Leadership impact depends on both available energy (V) and accumulated
    capital (C), but is suppressed by organisational complexity (O).

    Parameters
    ----------
    V : float
        Current vitality, typically in [0, Vmax] = [0, 10].
    C : float
        Career capital, typically in [0, 100].
    O : float
        Complexity (typically obtained from complexity() function).
    p : Dict[str, float]
        Parameter dictionary with keys: 'phi'.

    Returns
    -------
    float
        Leadership impact, non-negative.

    Notes
    -----
    The denominator (1 + φ·O) ensures impact → 0 as complexity → ∞.
    With phi=0.15: impact drops to ~50% of maximum at O ≈ 3.3.

    Examples
    --------
    >>> p = make_params()
    >>> O = complexity(10, p)
    >>> I = impact(8.0, 10, O, p)
    """
    return C * V / (1.0 + p['phi'] * O)


def dlvt_system(t: float, y: List[float],
                p: Dict[str, float]) -> List[float]:
    """Right-hand side of the DLVT ODE system.

    Implements Equations 3.3–3.4 of the paper:
      dV/dt = R·(1 − V/V_max) − δ·O^γ · V/(V + ε)
      dC/dt = α·I − μ·C

    This is the core dynamical system for solve_ivp integration.

    Parameters
    ----------
    t : float
        Time variable (not used explicitly; system is autonomous).
    y : List[float]
        State vector [V, C] where:
          V = current vitality
          C = current career capital
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    List[float]
        Time derivatives [dV/dt, dC/dt].

    Notes
    -----
    - Vitality V is clamped to [0, ∞) to maintain physical meaning.
    - Career capital C is clamped to [0, ∞) for the same reason.
    - The system is autonomous: the right-hand side does not depend on t.
    """
    V = max(y[0], 0.0)
    C = max(y[1], 0.0)
    O = complexity(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain    = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I        = impact(V, C, O, p)
    dVdt     = recovery - drain
    dCdt     = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]


def dlvt_exogenous(t: float, y: List[float], p: Dict[str, float],
                   C_func: Callable[[float], float]) -> List[float]:
    """DLVT system with *exogenous* career capital C(t).

    Use this variant to study vitality dynamics under an externally imposed
    capital trajectory (e.g., a promotion ramp or a sudden scope expansion).
    This is useful for scenario analysis where organisational decisions
    drive capital changes independent of impact feedback.

    Parameters
    ----------
    t : float
        Time variable.
    y : List[float]
        State vector [V] — only vitality is integrated.
    p : Dict[str, float]
        Parameter dictionary.
    C_func : Callable[[float], float]
        Exogenous career capital function C(t), must return float ≥ 0.

    Returns
    -------
    List[float]
        Time derivative [dV/dt].

    Examples
    --------
    >>> def promotion_ramp(t): return 5.0 + 0.1 * t  # gradual promotion
    >>> from scipy.integrate import solve_ivp
    >>> p = make_params()
    >>> sol = solve_ivp(dlvt_exogenous, [0, 120], [8.0],
    ...                 args=(p, promotion_ramp), method='RK45')
    """
    V = max(y[0], 0.0)
    C = C_func(t)
    O = complexity(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain    = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    return [recovery - drain]


# ── Simulation wrapper ────────────────────────────────────────────────────────

def simulate(p: Dict[str, float], V0: float = 8.0, C0: float = 0.5,
             T: float = 120.0, max_step: float = 0.05
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the DLVT system from (V0, C0) over [0, T].

    Solves the coupled ODE system using scipy.integrate.solve_ivp with
    the RK45 method (5th-order Runge–Kutta).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary (use make_params() or DEFAULT_PARAMS).
    V0 : float, optional
        Initial vitality, default 8.0 (within range [0, Vmax]).
    C0 : float, optional
        Initial career capital, default 0.5.
    T : float, optional
        Time horizon for integration, default 120.0.
    max_step : float, optional
        Maximum RK45 step size, default 0.05.
        Smaller values give higher resolution but slower computation.

    Returns
    -------
    t : ndarray
        Time grid at solution points.
    V : ndarray
        Vitality trajectory V(t).
    C : ndarray
        Career capital trajectory C(t).
    O : ndarray
        Complexity O(t), derived from C.
    I : ndarray
        Leadership impact I(t), derived from V, C, O.
    G : ndarray
        Depletion ratio Γ(t) = δ·O^γ / R.
        Values > 1 indicate net vitality drain; < 1 indicate net recovery.

    Examples
    --------
    >>> p = make_params(beta=0.25)
    >>> t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=120)
    >>> print(f'Final vitality: V(120) = {V[-1]:.2f}')
    """
    sol = solve_ivp(
        dlvt_system, [0.0, T], [V0, C0],
        args=(p,), method='RK45', max_step=max_step, dense_output=True
    )
    t = sol.t
    V = np.maximum(sol.y[0], 0.0)
    C = np.maximum(sol.y[1], 0.0)
    O = complexity(C, p)
    I = impact(V, C, O, p)
    G = p['delta'] * O**p['gamma'] / p['R']   # depletion ratio Γ
    return t, V, C, O, I, G
