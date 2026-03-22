"""
dlvt.analysis
=============
Equilibrium analysis, stability classification, and bifurcation utilities for the DLVT model.

This module implements the analytical and numerical results from "The Carrying Capacity of
Leadership: A Dynamic Systems Model of Executive Sustainability" (W. Bendinelli, 2026),
including Theorems 1–2 and Propositions 1–3.

Theoretical Foundations
-----------------------
THEOREM 1 (Existence): For any parameter vector (R, δ, γ, ...) in ℝ^11_>0, the DLVT system
  possesses at least one interior fixed point (V*, C*) with V*, C* > 0.

THEOREM 2 (Stability): Interior fixed points are classified as stable or saddle by examining
  the sign of Re(λ) for eigenvalues of the Jacobian matrix at the equilibrium. All equilibria
  obtained via find_interior_equilibria() include stability classification.

PROPOSITION 1 (V-Nullcline): The vitality recovery curve (dV/dt = 0) is strictly decreasing
  in C. This ensures unique intersection with the C-nullcline under normal parameter ranges.

PROPOSITION 2 (Bifurcation): Varying β generically produces saddle-node bifurcations. For
  certain parameter regions, bistability and hysteresis emerge (see Fig 8).

PROPOSITION 3 (Carrying Capacity): The maximum sustainable career capital is given by
    C*_max = ( ((R/δ)^(1/γ) − O₀) / β )^(1/η)
  This is the critical threshold beyond which sustainable equilibria cease to exist (Eq. 4.3).

Key Functions
-----------
carrying_capacity()         : Proposition 3 — maximum sustainable capital C*_max
find_interior_equilibria()  : Theorem 2 — find all (V*, C*) with V*, C* > 0
jacobian_eigenvalues()      : Theorem 2 — compute eigenvalues, classify stability
is_zombie()                 : Definition 7 — check if V* < V_strategic
classify_regime()           : Regime classification (sustainable, zombie, collapse-prone)
regime_map()                : Figure 7 — parameter-space regime classification in (β, δ)

Module Constants
----------------
V_STRATEGIC_FRACTION = 0.5  : Strategic vitality threshold = 0.5 * V_max
                              Below this, executives are classified as "zombie" (Definition 7)

References
----------
  Bendinelli, W. (2026). The Carrying Capacity of Leadership: A Dynamic Systems Model
  of Executive Sustainability. SSRN Working Paper.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import brentq

from .model import complexity, impact, DEFAULT_PARAMS

# Strategic vitality threshold (Definition in paper: V_strategic = 0.5 * V_max)
V_STRATEGIC_FRACTION = 0.5


# ── Analytical results ────────────────────────────────────────────────────────

def carrying_capacity(p: Dict[str, float]) -> float:
    """Maximum sustainable career capital C*_max (Proposition 3).

    C*_max is the value of C at which the depletion ratio Γ = δ·O^γ / R = 1
    evaluated at full vitality (V = V_max), i.e., the tipping point beyond
    which energy drain permanently exceeds recovery.

    This is the critical boundary between sustainable and collapse-prone regimes.

    Formula (ε → 0 limit, η = 1):
      C*_max = ( (R/δ)^{1/γ} − O₀ ) / β

    Generalised for η ≠ 1 (Proposition 3 extended):
      C*_max = ( ((R/δ)^{1/γ} − O₀) / β )^{1/η}

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary with keys: 'R', 'delta', 'gamma', 'O0', 'beta', 'eta'.

    Returns
    -------
    float
        C*_max, the maximum sustainable career capital.
        Returns 0.0 if no sustainable capacity exists (Omax ≤ O0).

    Notes
    -----
    C*_max = 0 signals an "unsustainable" regime where even zero capital
    leads to persistent vitality drain (complete burnout is inevitable).

    Examples
    --------
    >>> from dlvt import make_params
    >>> p = make_params()
    >>> cc = carrying_capacity(p)
    >>> print(f'Maximum sustainable capital: C* = {cc:.2f}')
    """
    Omax = (p['R'] / p['delta']) ** (1.0 / p['gamma'])
    if Omax <= p['O0']:
        return 0.0
    return max(0.0, ((Omax - p['O0']) / p['beta']) ** (1.0 / p['eta']))


def find_interior_equilibria(p: Dict[str, float], C_max: float = 120.0,
                            n_scan: int = 8000
                            ) -> List[Dict[str, any]]:
    """Find all interior equilibria (V*, C*) with V*, C* > 0 (Theorem 2).

    Solves the system:
      dV/dt = 0: recovery (V*) = drain (V*, C*)
      dC/dt = 0: impact (V*, C*) = depreciation

    Algorithm: substitute V* from the dC/dt = 0 nullcline into the
    dV/dt = 0 condition and scan for sign changes using Brent's method.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C_max : float, optional
        Upper bound for capital scan, default 120.0.
    n_scan : int, optional
        Number of scan points (more = fewer missed roots), default 8000.

    Returns
    -------
    List[Dict[str, any]]
        List of equilibrium points. Each dict contains:
          - 'C': equilibrium capital C*
          - 'V': equilibrium vitality V*
          - 'O': equilibrium complexity O*
          - 'I': equilibrium impact I*
          - 'stable': bool, True if locally asymptotically stable
          - 'eigenvalues': ndarray of Jacobian eigenvalues
          - 'zombie': bool, True if V* < V_strategic

    Notes
    -----
    The algorithm performs a coarse scan over [0.01, C_max] and refines
    at each sign change. This robustly finds multiple equilibria for
    bistability/hysteresis regimes.
    """
    def V_from_C(Cs):
        """V* from dC/dt = 0: α·I* = μ·C*  ⟹  V* = μ·(1 + φ·O) / α"""
        Os = p['O0'] + p['beta'] * Cs**p['eta']
        return p['mu'] * (1.0 + p['phi'] * Os) / p['alpha']

    def residual(Cs):
        if Cs <= 0:
            return 1e10
        Vs = V_from_C(Cs)
        if Vs <= 0 or Vs >= p['Vmax']:
            return 1e10
        Os = p['O0'] + p['beta'] * Cs**p['eta']
        rec = p['R'] * (1.0 - Vs / p['Vmax'])
        drn = p['delta'] * Os**p['gamma'] * Vs / (Vs + p['eps'])
        return rec - drn

    C_scan = np.linspace(0.01, C_max, n_scan)
    res    = np.array([residual(c) for c in C_scan])

    equilibria = []
    for i in range(len(res) - 1):
        if res[i] * res[i + 1] < 0:
            try:
                Cs = brentq(residual, C_scan[i], C_scan[i + 1])
                Vs = V_from_C(Cs)
                Os = complexity(Cs, p)
                if 0 < Vs < 0.999 * p['Vmax'] and Cs > 0:
                    eigvals, stable = jacobian_eigenvalues(Vs, Cs, p)
                    equilibria.append(dict(
                        C=Cs, V=Vs, O=Os,
                        I=impact(Vs, Cs, Os, p),
                        stable=stable,
                        eigenvalues=eigvals,
                        zombie=is_zombie(Vs, p),
                    ))
            except Exception:
                pass

    return equilibria


def jacobian_eigenvalues(V: float, C: float,
                        p: Dict[str, float]
                        ) -> Tuple[np.ndarray, bool]:
    """Compute eigenvalues of the Jacobian J at (V, C) and assess stability.

    Computes the 2×2 Jacobian matrix and its eigenvalues at an equilibrium.
    Linear stability is determined by the sign of the real parts.

    The Jacobian is:
      J[0,0] = ∂(dV/dt)/∂V = −R/V_max − δ·O^γ·ε/(V+ε)²
      J[0,1] = ∂(dV/dt)/∂C = −δ·γ·O^{γ−1}·(dO/dC)·V/(V+ε)
      J[1,0] = ∂(dC/dt)/∂V = α·C/(1+φ·O)
      J[1,1] = ∂(dC/dt)/∂C = α·V/(1+φ·O) − α·C·V·φ·(dO/dC)/(1+φ·O)² − μ

    Stability criterion: all eigenvalues satisfy Re(λ) < 0.

    Parameters
    ----------
    V : float
        Vitality at the equilibrium.
    C : float
        Career capital at the equilibrium.
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    eigvals : ndarray, shape (2,)
        Eigenvalues of the Jacobian (may be complex).
    stable : bool
        True if the equilibrium is locally asymptotically stable
        (all eigenvalues have negative real part).

    Notes
    -----
    Uses numpy.linalg.eigvals for robust computation.
    """
    O    = complexity(C, p)
    eps  = p['eps']
    dOdC = p['beta'] * p['eta'] * max(C, 1e-10)**(p['eta'] - 1)

    J = np.array([
        [
            -p['R'] / p['Vmax'] - p['delta'] * O**p['gamma'] * eps / (V + eps)**2,
            -p['delta'] * p['gamma'] * O**(p['gamma'] - 1) * dOdC * V / (V + eps)
        ],
        [
            p['alpha'] * C / (1.0 + p['phi'] * O),
            (p['alpha'] * V / (1.0 + p['phi'] * O)
             - p['alpha'] * C * V * p['phi'] * dOdC / (1.0 + p['phi'] * O)**2
             - p['mu'])
        ]
    ])
    eigvals = np.linalg.eigvals(J)
    stable  = bool(all(e.real < 0 for e in eigvals))
    return eigvals, stable


def is_zombie(V_star: float, p: Dict[str, float]) -> bool:
    """Return True if V* < V_strategic (Definition 7 of the paper).

    A "zombie" equilibrium is one where the leader maintains capital-building
    activity but with chronically insufficient energy. At these equilibria,
    the executive is perpetually exhausted and cannot perform strategic work.

    Parameters
    ----------
    V_star : float
        Equilibrium vitality.
    p : Dict[str, float]
        Parameter dictionary (must contain 'Vmax').

    Returns
    -------
    bool
        True if V_star < V_strategic, False otherwise.

    Notes
    -----
    V_strategic = V_STRATEGIC_FRACTION * V_max, where V_STRATEGIC_FRACTION=0.5.
    This threshold reflects the minimum vitality for effective strategic thinking.
    """
    return V_star < V_STRATEGIC_FRACTION * p['Vmax']


# ── Regime map ────────────────────────────────────────────────────────────────

def classify_regime(p: Dict[str, float], C_max: float = 300.0) -> str:
    """Classify a parameter combination into a leadership regime.

    Determines whether a set of parameter values leads to sustainable,
    zombie, or collapse-prone leadership outcomes.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C_max : float, optional
        Maximum capital for equilibrium search, default 300.0.

    Returns
    -------
    str
        One of:
          - 'sustainable': stable equilibrium with V* ≥ V_strategic
          - 'zombie': stable equilibrium with V* < V_strategic
          - 'collapse-prone': no stable interior equilibrium

    Notes
    -----
    Priority: uses the lowest-capital stable equilibrium if multiple exist.
    This reflects the "dominant" organizational outcome.
    """
    cc = carrying_capacity(p)
    if cc <= 0:
        return 'collapse-prone'
    eqs = find_interior_equilibria(p, C_max=C_max)
    stable_eqs = [eq for eq in eqs if eq['stable']]
    if not stable_eqs:
        return 'collapse-prone'
    # Use the first (lowest C*) stable equilibrium
    eq = stable_eqs[0]
    return 'zombie' if eq['zombie'] else 'sustainable'


def regime_map(beta_range: np.ndarray, delta_range: np.ndarray,
               base_params: Optional[Dict[str, float]] = None
               ) -> np.ndarray:
    """Compute the leadership regime map over a (β, δ) grid (Figure 7).

    Scans a 2-D parameter space and classifies each point into a regime.
    This generates the phase diagram showing how leadership sustainability
    depends on capital-complexity coupling (β) and energetic cost (δ).

    Parameters
    ----------
    beta_range : ndarray
        Values of β (capital-complexity coupling), typically [0.01, 1.0].
    delta_range : ndarray
        Values of δ (energetic cost coefficient), typically [0.001, 0.1].
    base_params : Optional[Dict[str, float]]
        Base parameters (defaults to DEFAULT_PARAMS).
        β and δ are overridden by the grid values.

    Returns
    -------
    regimes : ndarray, shape (len(delta_range), len(beta_range)), dtype=object
        Grid of regime classifications.
        regimes[i, j] = classify_regime(p) for p with β=beta_range[j], δ=delta_range[i].
        Values: 'sustainable', 'zombie', or 'collapse-prone'.

    Notes
    -----
    This function is O(len(beta_range) * len(delta_range)) and may be slow
    for large grids. Typical grid sizes: ~50 × 50 for fast preview, ~200 × 200
    for publication quality.

    Examples
    --------
    >>> import numpy as np
    >>> from dlvt import regime_map
    >>> betas = np.linspace(0.01, 1.0, 50)
    >>> deltas = np.linspace(0.001, 0.1, 50)
    >>> regimes = regime_map(betas, deltas)
    """
    if base_params is None:
        base_params = DEFAULT_PARAMS.copy()

    regimes = np.empty((len(delta_range), len(beta_range)), dtype=object)
    for i, dv in enumerate(delta_range):
        for j, bv in enumerate(beta_range):
            p = {**base_params, 'beta': bv, 'delta': dv}
            regimes[i, j] = classify_regime(p)
    return regimes
