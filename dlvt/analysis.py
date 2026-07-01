"""
dlvt.analysis
=============
Equilibrium analysis, stability classification, and bifurcation utilities for the DLVT model.

This module implements the analytical and numerical results from "Dynamic Leadership
Vitality Theory: A Formal Model of the Zombie-Leader Equilibrium" (W. Bendinelli, 2026),
targeting *The Leadership Quarterly*. Includes Theorems 1–2, Lemma 2, and Propositions 1–3.

Theoretical Foundations (corrected statements, v2.1)
----------------------------------------------------
THEOREM 1 (Existence): Whenever αVmax/μ > 1 + φO₀, the DLVT system possesses an interior
  fixed point (V*, C*) with V*, C* > 0. (When the inequality fails, the C-nullcline
  V_c(C) = μ(1+φO)/α sits at or above Vmax for all C ≥ 0, dC/dt < 0 everywhere, and no
  interior equilibrium exists.)

THEOREM 2a (Global Uniqueness): the interior equilibrium is unique in the ENTIRE positive
  orthant, for ALL positive parameter vectors — not just the baseline. Proof: along the
  C-nullcline V_c(O) = (μ/α)(1+φO), define
      Φ(O) = R(1 − V_c/Vmax) − δ·O^γ·V_c/(V_c+ε).
  Every term of dΦ/dO is strictly negative:
      −(R/Vmax)·V_c′ < 0,
      −δγO^{γ−1}·V_c/(V_c+ε) < 0,
      −δO^γ·ε·V_c′/(V_c+ε)² < 0,
  so Φ is strictly decreasing in O; since O(C) is strictly increasing, Φ has at most one
  root in C. Hence: at most one interior equilibrium anywhere, and no fold/saddle-node
  from root multiplicity is possible. See Appendix A10.

THEOREM 2b (No Hopf / no cycles): at any interior equilibrium, dC/dt = 0 forces
  αV*/(1+φO*) = μ, hence
      J[1,1] = −α·C*·V*·φ·(dO/dC)/(1+φO*)² < 0,
  and J[0,0] < 0 identically, so trace(J) < 0 at EVERY interior equilibrium for all
  positive parameters — no Hopf bifurcation exists anywhere in parameter space. This is
  consistent with (and independent of) the Bendixson–Dulac certificate B(V,C) = 1/C, which
  rules out closed orbits globally on {C > 0}. No elementary global Lyapunov function
  exists for this system (weighted quadratics fail), so Dulac + Poincaré–Bendixson is the
  appropriate proof route.

THEOREM 2c (Global Asymptotic Stability): the unique interior equilibrium is globally
  asymptotically stable on the OPEN set {C > 0} ∩ Ω. The line C = 0 is invariant and
  carries a saddle equilibrium at (V ≈ 9.934, C = 0) (see find_regularization_branch), so
  the closed quadrant is NOT the basin — trajectories started exactly on C = 0 converge to
  the axis saddle instead. The trapping rectangle is
      Ω = [0, Vmax] × [0, C_trap],  C_trap^η = ((αVmax/μ − 1)/φ − O₀)/β
  (baseline C_trap = 102.67; see trapping_capital_bound). CAUTION: earlier drafts used the
  carrying capacity C*_max = 44.99 as the rectangle ceiling; that rectangle LEAKS (at
  C = 44.99, V = Vmax, dC/dt > 0) — C_trap and C*_max are distinct constants. Proof:
  Dulac certificate (no closed orbits) + uniqueness (2a) + Poincaré–Bendixson on Ω∩{C>0}.
  See Appendix A10.

COROLLARY (Scope Absorption; formerly "Lemma 2"): under the separable power-law drain
  kernel f(O)·g(V), the pair (V*, O*) solves a β-free system, so V* is invariant in β and
  β·C*^η is conserved. This is a reparameterization invariance of the kernel family, NOT a
  robust empirical law: under alternative kernels (exponential, Hill) the property fails.
  Managerial reading (intervention asymmetry): scope/coupling reduction alone cannot raise
  equilibrium vitality; recovery-side parameters (R↑, μ↓, α↑) can. ε-dependence of the
  invariant: β·C* = 8.008 at ε = 0.1, and exactly 7.933 in the ε → 0 closed form (O* solves
  δO² + (R/Vmax)(μφ/α)O + R((μ/α)/Vmax − 1) = 0 for γ=2, η=1). See §3.8 / §4.

PROPOSITION 1 (V-Nullcline): The vitality recovery curve (dV/dt = 0) is strictly decreasing
  in C. Together with Theorem 2a this gives the unique intersection with the C-nullcline.

PROPOSITION 2 (Bifurcation): Under non-power-law drain kernels, varying β can produce
  saddle-node bifurcations. Under the baseline power-law kernel, the Scope-Absorption
  Corollary forces V* to be invariant in β, so no classical V*-crossing bifurcation exists.
  See Appendix A8.

PROPOSITION 3 (Carrying Capacity — a FLUX THRESHOLD, not a bifurcation): the maximum
  sustainable career capital is
    C*_max = ( ((R/δ)^(1/γ) − O₀) / β )^(1/η)
  the capital level at which the depletion ratio Γ = δO^γ/R reaches 1 at FULL vitality
  (V = Vmax). Since det J > 0 everywhere, no saddle-node/fold occurs anywhere: equilibria
  do not "cease to exist" past C*_max. The constant separates the region where recovery at
  full vitality can offset drain from the region where it cannot; it is NOT the trapping
  ceiling (see trapping_capital_bound) and NOT a critical point of the flow.

CALIBRATION CAVEAT (regime label): the interior equilibrium satisfies
  V* = (μ/α)(1 + φO*), so whether it falls below the stipulated threshold 0.5·Vmax is
  governed by the ratio μ/α. At baseline μ/α = 2.0 the flip value is (μ/α)_crit ≈ 2.163
  (8% away); across a ±2× log-uniform parameter hypercube, P(zombie | stable eq) ≈ 0.49.
  The baseline numbers (V* ≈ 4.70, C* ≈ 32.03) are ILLUSTRATIVE, not empirical estimates.
  See dlvt.nondimensional for the effective-parameter analysis (~6 dimensionless groups).

Key Functions
-----------
carrying_capacity()             : Proposition 3 — flux threshold C*_max (not a bifurcation)
trapping_capital_bound()        : Theorem 2c — trapping-rectangle ceiling C_trap
find_interior_equilibria()      : Theorem 2 — find all (V*, C*) with V*, C* > 0
jacobian_eigenvalues()          : Theorem 2 — compute eigenvalues, classify stability
is_zombie()                     : Definition 7 — check if V* < V_strategic
classify_regime()               : Regime classification (sustainable, zombie, collapse-prone)
regime_map()                    : Figure 7 — parameter-space regime classification in (β, δ)
bendixson_dulac_certificate()   : Theorem 2 proof — Dulac divergence grid + trapping rectangle
basin_of_attraction_sweep()     : Theorem 2 corroboration — 64 IC convergence test
find_regularization_branch()    : Appendix A9 — characterize ε-regularization saddle
estimate_bifurcation_interval() : Appendix A8 — β_crit sensitivity to ε and rtol

Module Constants
----------------
V_STRATEGIC_FRACTION = 0.5  : Strategic vitality threshold = 0.5 * V_max
                              Below this, executives are classified as "zombie" (Definition 7)

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  of the Zombie-Leader Equilibrium. Manuscript submitted to The Leadership Quarterly.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import brentq

from .model import complexity, impact, DEFAULT_PARAMS

# Strategic vitality threshold (Definition in paper: V_strategic = 0.5 * V_max)
# NOTE: this threshold is *stipulated*, not derived from data. At the baseline
# calibration the equilibrium V* = 4.7025 sits only 5.9% below it; any threshold
# fraction below 0.47 reclassifies the baseline as 'sustainable'. See §5.
V_STRATEGIC_FRACTION = 0.5


# ── Analytical results ────────────────────────────────────────────────────────

def trapping_capital_bound(p: Dict[str, float]) -> float:
    """Capital ceiling C_trap of the forward-invariant trapping rectangle.

    C_trap is the unique capital level at which the C-nullcline
    V_c(C) = μ·(1 + φ·O(C))/α reaches V_max:

        C_trap^η = ( (α·V_max/μ − 1)/φ − O₀ ) / β

    For any C > C_trap and any V ∈ [0, V_max],
    dC/dt = C·(α·V/(1+φ·O) − μ) < 0, so the rectangle
    Ω = [0, V_max] × [0, C_trap] is forward invariant (Theorem 2).

    Two distinct constants — do not conflate them:

    - ``C_trap`` (this function; baseline **102.67**) bounds the *trapping
      rectangle* used in the global-stability proof. It also bounds every
      interior equilibrium, since dC/dt = 0 with C* > 0 forces
      V* = μ(1+φO*)/α < V_max ⟹ C* < C_trap.
    - ``carrying_capacity`` (Proposition 3; baseline **44.99**) is the flux
      threshold where the depletion ratio Γ = 1 at V = V_max. Earlier drafts
      wrongly used it as the rectangle ceiling; that rectangle *leaks*
      (at C = 44.99, V = V_max, dC/dt > 0 — a trajectory started at
      (V, C) = (10, 40) overshoots to C ≈ 46.4 before settling).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    float
        C_trap. Raises ValueError when α·V_max/μ ≤ 1 + φ·O₀, in which
        case the C-nullcline never reaches V_max: dC/dt < 0 for all C > 0,
        no interior equilibrium exists, and no trapping ceiling is needed.
    """
    rhs = (p['alpha'] * p['Vmax'] / p['mu'] - 1.0) / p['phi'] - p['O0']
    if rhs <= 0:
        raise ValueError(
            "Parameter regime has αVmax/μ ≤ 1+φO0; the C-nullcline never "
            "reaches Vmax, so no interior equilibrium exists and the "
            "trapping ceiling is undefined (capital decays for all C > 0)."
        )
    return (rhs / p['beta']) ** (1.0 / p['eta'])

def carrying_capacity(p: Dict[str, float]) -> float:
    """Maximum sustainable career capital C*_max (Proposition 3) — a flux threshold.

    C*_max is the value of C at which the depletion ratio Γ = δ·O^γ / R = 1
    evaluated at full vitality (V = V_max): below it, recovery at full
    vitality can offset the drain; above it, it cannot.

    IMPORTANT (corrected interpretation): C*_max is *not* a bifurcation
    point. det J > 0 holds everywhere, so no saddle-node/fold exists
    anywhere in parameter or state space — equilibria do not "cease to
    exist" past C*_max (Theorem 2a guarantees a unique interior equilibrium
    whenever αVmax/μ > 1+φO₀). C*_max is also *not* the ceiling of the
    trapping rectangle used in the global-stability proof; that constant is
    C_trap (see :func:`trapping_capital_bound`; baseline 102.67 vs 44.99).

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


def find_interior_equilibria(p: Dict[str, float], C_max: Optional[float] = None,
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
        Upper bound for capital scan. Default ``None`` derives the bound
        analytically: any interior equilibrium satisfies
        V* = μ(1+φO*)/α < V_max, hence C* < C_trap
        (see :func:`trapping_capital_bound`), so the scan uses
        ``1.05 · C_trap``. This bound scales as 1/β, exactly like
        C*(β) = (β·C*)/β itself, so no equilibrium can fall outside the
        window at any β. A *fixed* default (120.0 in earlier releases)
        silently missed the equilibrium for β < 0.066 — C*(β) ≈ 8.008/β
        exceeds 120 there — causing :func:`classify_regime` to mislabel
        those regimes as 'collapse-prone'. Pass an explicit value only if
        you deliberately want a restricted window.
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
    at each sign change. Uniqueness is guaranteed analytically (the
    residual Φ is strictly decreasing along the C-nullcline — see the
    module header, Theorem 2a), so at most one root exists; the scan is
    retained as a defensive check that would surface any violation.
    """
    if C_max is None:
        try:
            C_max = 1.05 * trapping_capital_bound(p)
        except ValueError:
            # αVmax/μ ≤ 1+φO0: the C-nullcline sits at/above Vmax for all
            # C ≥ 0, so dC/dt < 0 everywhere and no interior equilibrium
            # exists. Return the empty list directly.
            return []
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

def classify_regime(p: Dict[str, float], C_max: Optional[float] = None) -> str:
    """Classify a parameter combination into a leadership regime.

    Determines whether a set of parameter values leads to sustainable,
    zombie, or collapse-prone leadership outcomes.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.
    C_max : float, optional
        Maximum capital for equilibrium search. Default ``None`` uses the
        analytical bound (see :func:`find_interior_equilibria`), which is
        valid for every β. Fixed values silently mislabel small-β regimes
        as 'collapse-prone' when C*(β) ∝ 1/β exceeds the window.

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


# ── Bifurcation-interval diagnostics (R7-6) ───────────────────────────────────

def estimate_bifurcation_interval(
    base_params: Dict[str, float],
    eps_grid: Optional[List[float]] = None,
    n_scan_grid: Optional[List[int]] = None,
    beta_range: Tuple[float, float] = (0.005, 2.0),
    n_beta: int = 400,
    V_strategic: Optional[float] = None,
    C_max: Optional[float] = None,
) -> Dict[str, any]:
    """Estimate the critical-beta interval, exposing regularization sensitivity.

    This function replaces the false-precision ``beta_crit ≈ 0.1015`` claim in
    earlier drafts of Appendix A8. It answers two honest questions instead of
    one fake-precise one:

    (1) *Does* the equilibrium vitality :math:`V^*(\\beta)` cross the
        strategic threshold :math:`V_{\\mathrm{strategic}}` anywhere in the
        sweep? If Lemma 2 (scope-absorption) applies — i.e.\\ the drain
        kernel is multiplicative power-law in :math:`O = O_0 + \\beta C^\\eta`
        and :math:`V^*` depends on :math:`\\beta` only through :math:`\\beta
        C^{*\\eta}` — then the answer for the baseline calibration is *no*:
        :math:`V^*` is constant across :math:`\\beta`, and the "critical"
        value reported in earlier drafts was a scan-window artifact, not a
        structural property.

    (2) If the crossing *does* exist (non-baseline calibration), what is the
        range of numerical estimates of :math:`\\beta_{\\mathrm{crit}}`
        produced by varying the regularization :math:`\\varepsilon` and the
        scan resolution? The returned interval is the appropriate
        methods-honest reporting granularity.

    Parameters
    ----------
    base_params : Dict[str, float]
        Parameter dictionary (will be shallow-copied; ``eps`` is overwritten).
    eps_grid : List[float], optional
        Values of the vitality-barrier regularization :math:`\\varepsilon`
        to sweep. Default: ``[0.01, 0.05, 0.1, 0.2]``.
    n_scan_grid : List[int], optional
        Values of ``n_scan`` (scan resolution for ``find_interior_equilibria``)
        to sweep. A proxy for integration tolerance. Default:
        ``[8000, 16000]``.
    beta_range : Tuple[float, float], optional
        ``(beta_min, beta_max)`` for the sweep. Default: ``(0.005, 2.0)``.
    n_beta : int, optional
        Number of beta sample points in each sweep. Default: ``400``.
    V_strategic : float, optional
        Strategic threshold. Defaults to ``V_STRATEGIC_FRACTION * Vmax``.
    C_max : float, optional
        Upper bound for the equilibrium scan. Default ``None`` uses the
        per-β analytical bound C_trap ∝ 1/β (see
        :func:`find_interior_equilibria`), which catches the low-β
        equilibria that the legacy fixed C_max=80 missed — the source of
        the original 0.1015 artifact.

    Returns
    -------
    result : Dict[str, any]
        A diagnostic dictionary with the keys:

        - ``'v_star_invariant'`` : Optional[float]
            If :math:`V^*(\\beta)` is numerically constant across the sweep
            (to tolerance 1e-4), the invariant value; otherwise ``None``.
        - ``'crosses_threshold'`` : bool
            True iff there exists a :math:`\\beta` in the sweep at which a
            stable interior equilibrium has
            :math:`V^* \\geq V_{\\mathrm{strategic}}` and another at which
            :math:`V^* < V_{\\mathrm{strategic}}`.
        - ``'beta_crit_interval'`` : Optional[Tuple[float, float]]
            ``(low, high)`` range of the numerical estimate across the
            ``(eps, n_scan)`` grid, if the crossing exists; otherwise
            ``None``.
        - ``'beta_crit_per_eps'`` : Dict[float, Optional[float]]
            Per-epsilon estimate (median across n_scan values); ``None``
            entries mark no crossing.
        - ``'diagnostic'`` : str
            Plain-English explanation of the finding, including the Lemma 2
            pointer when applicable.
        - ``'baseline_beta_C_product'`` : Optional[float]
            If ``v_star_invariant`` is not None, the invariant product
            :math:`\\beta C^*` (which Lemma 2 shows must be constant).

    Notes
    -----
    The function is deliberately *not* a one-liner. It exists to make the
    scope-absorption phenomenon explicit in the code, not hide it. The
    estimate intentionally uses a large C_max so that low-β equilibria are
    not missed --- this is how we detect and report the original scan
    artifact.

    Examples
    --------
    >>> from dlvt import make_params
    >>> from dlvt.analysis import estimate_bifurcation_interval
    >>> result = estimate_bifurcation_interval(make_params())
    >>> result['crosses_threshold']     # baseline does not cross
    False
    >>> round(result['v_star_invariant'], 4)
    4.7025
    """
    if eps_grid is None:
        eps_grid = [0.01, 0.05, 0.1, 0.2]
    if n_scan_grid is None:
        n_scan_grid = [8000, 16000]
    if V_strategic is None:
        V_strategic = V_STRATEGIC_FRACTION * base_params['Vmax']

    beta_min, beta_max = beta_range
    betas = np.linspace(beta_min, beta_max, n_beta)

    def _sweep(p_local: Dict[str, float], n_scan: int):
        """Return list of (beta, V_star) for stable interior equilibria."""
        curve = []
        for b in betas:
            p_try = dict(p_local)
            p_try['beta'] = b
            eqs = find_interior_equilibria(p_try, C_max=C_max, n_scan=n_scan)
            stable = [e for e in eqs if e['stable']]
            if stable:
                # Take the equilibrium on the dominant (high-C) branch, which
                # is the sustainable/zombie branch of Figure 8.
                stable.sort(key=lambda e: e['C'])
                curve.append((b, stable[-1]['V'], stable[-1]['C']))
        return curve

    def _beta_crit_from_curve(curve: List[Tuple[float, float, float]]
                              ) -> Optional[float]:
        """Find beta where V_star(beta) crosses V_strategic, or None."""
        if len(curve) < 2:
            return None
        crossings = []
        for i in range(len(curve) - 1):
            v1, v2 = curve[i][1], curve[i + 1][1]
            if (v1 - V_strategic) * (v2 - V_strategic) < 0:
                # linear interpolation
                b1, b2 = curve[i][0], curve[i + 1][0]
                frac = (V_strategic - v1) / (v2 - v1)
                crossings.append(b1 + frac * (b2 - b1))
        if not crossings:
            return None
        return float(np.median(crossings))

    per_eps: Dict[float, Optional[float]] = {}
    per_eps_v_star: Dict[float, Optional[float]] = {}
    per_eps_bC: Dict[float, Optional[float]] = {}
    any_crosses = False
    all_crits: List[float] = []

    for eps in eps_grid:
        p_eps = dict(base_params)
        p_eps['eps'] = eps
        eps_crits: List[float] = []
        eps_v_stars: List[float] = []
        eps_bC: List[float] = []
        for ns in n_scan_grid:
            curve = _sweep(p_eps, ns)
            if curve:
                eps_v_stars.extend(v for (_, v, _) in curve)
                eps_bC.extend(b * c for (b, _, c) in curve)
            bc = _beta_crit_from_curve(curve)
            if bc is not None:
                eps_crits.append(bc)
                any_crosses = True
        per_eps[eps] = float(np.median(eps_crits)) if eps_crits else None
        per_eps_v_star[eps] = float(np.median(eps_v_stars)) if eps_v_stars else None
        per_eps_bC[eps] = float(np.median(eps_bC)) if eps_bC else None
        all_crits.extend(eps_crits)

    # scope-absorption diagnostic: at each eps, check whether V*(β) is
    # numerically constant across the sweep. If it is at *every* eps, the
    # Lemma 2 invariant holds (V* depends on eps but not on β).
    v_star_invariant: Optional[float] = None
    baseline_bC: Optional[float] = None
    per_eps_constant = True
    for eps in eps_grid:
        p_eps = dict(base_params); p_eps['eps'] = eps
        eps_vs: List[float] = []
        for ns in n_scan_grid:
            curve = _sweep(p_eps, ns)
            eps_vs.extend(v for (_, v, _) in curve)
        if len(eps_vs) >= 2:
            if float(np.max(eps_vs) - np.min(eps_vs)) > 1e-4:
                per_eps_constant = False
                break
        else:
            per_eps_constant = False
            break
    if per_eps_constant and per_eps_v_star:
        # report the value at the middle eps (or the user's calibration eps if present)
        mid_eps = eps_grid[len(eps_grid) // 2]
        v_star_invariant = per_eps_v_star[mid_eps]
        baseline_bC = per_eps_bC[mid_eps]

    if any_crosses:
        beta_crit_interval: Optional[Tuple[float, float]] = (
            float(min(all_crits)),
            float(max(all_crits)),
        )
        diagnostic = (
            f"V*(β) crosses V_strategic={V_strategic:.3f}. Numerical estimates "
            f"of β_crit across the (eps, n_scan) grid span "
            f"[{beta_crit_interval[0]:.4f}, {beta_crit_interval[1]:.4f}]; "
            f"report this interval, not a point."
        )
    else:
        beta_crit_interval = None
        if v_star_invariant is not None:
            diagnostic = (
                f"V*(β) is numerically constant at V*={v_star_invariant:.4f} "
                f"across the sweep (scope-absorption / Lemma 2: β·C* is "
                f"invariant at {baseline_bC:.4f}). "
                f"{'V* < V_strategic' if v_star_invariant < V_strategic else 'V* ≥ V_strategic'} "
                f"for all β, so β_crit as a V*-crossing does not exist. Any "
                f"previously reported 'critical β' for this calibration was a "
                f"scan-window artifact of a too-small C_max (the equilibrium "
                f"C*(β) = {baseline_bC:.3f}/β exceeds legacy C_max=80 for β < "
                f"{baseline_bC/80:.4f})."
            )
        else:
            diagnostic = (
                f"V*(β) does not cross V_strategic={V_strategic:.3f} in "
                f"β ∈ [{beta_min}, {beta_max}], but is not numerically "
                f"constant either. Verify parameter regime before interpreting."
            )

    return {
        'v_star_invariant': v_star_invariant,
        'crosses_threshold': any_crosses,
        'beta_crit_interval': beta_crit_interval,
        'beta_crit_per_eps': per_eps,
        'diagnostic': diagnostic,
        'baseline_beta_C_product': baseline_bC,
    }


# ── ε-regularization branch audit (R7 issue 3) ────────────────────────────────

def find_regularization_branch(
    p: Dict[str, float],
    near_zero_threshold: Optional[float] = None,
) -> Dict[str, any]:
    """Enumerate *all* equilibria and look for an ε-regularization ``near-zero'' branch.

    The smooth barrier $V/(V+\\varepsilon)$ used in the vitality ODE is a
    positive-invariance trick: without it, strong drain regimes would push
    $V$ across zero into the unphysical negative half-plane. A careful
    reviewer may reasonably worry that the regularization introduces a
    *spurious* second equilibrium branch at small positive $V$, of the form
    $V^* \\approx R\\varepsilon/(\\delta O^\\gamma)$, that has no correlate in
    the $\\varepsilon\\to 0$ limit. This function audits that concern by
    explicitly enumerating all equilibria of the full 2D system and
    reporting any with $V^* < $ ``near_zero_threshold''.

    Structure of the enumeration:

    1. **Axis equilibria.** At $C = 0$, $\\dot C \\equiv 0$ automatically, and
       $\\dot V = 0$ reduces to a quadratic in $V$:
           :math:`\\frac{R}{V_{\\max}} V^2 - (R - R\\varepsilon/V_{\\max} - \\delta O_0^\\gamma) V - R\\varepsilon = 0`.
       The product of the roots is $-\\varepsilon V_{\\max} < 0$, so there
       is exactly *one* positive root. We solve the quadratic in closed form.

    2. **Interior equilibria.** Delegated to
       :func:`find_interior_equilibria`, which parametrizes along the
       $\\dot C = 0$ nullcline $V_c(C) = \\mu(1+\\varphi O)/\\alpha$. Because
       $V_c \\geq \\mu/\\alpha$ identically, the interior branch is *bounded
       away from zero* by construction: no interior equilibrium can have
       $V^* < \\mu/\\alpha$ (at baseline, $V^* \\geq 2$).

    3. **Near-zero filter.** Any equilibrium with $V^* <$
       ``near_zero_threshold`` is flagged as ``near-zero''. By default the
       threshold is $\\min(\\mu/\\alpha, 1.0)$, which is a safe lower bound
       below the minimum possible interior $V^*$.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary; must contain the standard DLVT keys.
    near_zero_threshold : Optional[float]
        Threshold below which an equilibrium is flagged ``near zero''.
        Defaults to ``min(mu/alpha, 1.0)``.

    Returns
    -------
    Dict[str, any]
        Audit report with keys:

        - ``axis_equilibrium`` : dict or None
            The $C=0$ axis equilibrium, classified by Jacobian eigenvalues.
            Keys: ``V``, ``C``, ``eigenvalues``, ``stable``, ``classification``
            (``saddle``, ``stable node/focus``, ``unstable node/focus``).
        - ``interior_equilibria`` : List[Dict]
            All interior equilibria from :func:`find_interior_equilibria`.
        - ``near_zero_branch`` : Optional[Dict]
            The near-zero equilibrium if one exists, else ``None``.
            At baseline DLVT parameters this is ``None``: the axis root
            lives near $V_{\\max}$ (not near 0), and the interior branch
            is pinned at $V^* = \\mu(1+\\varphi O^*)/\\alpha \\geq \\mu/\\alpha$.
        - ``interior_V_lower_bound`` : float
            The analytical lower bound $\\mu/\\alpha$ on any interior $V^*$.
        - ``quadratic_positive_root_count`` : int
            Number of positive roots of the axis quadratic (always 1 by
            the product-of-roots argument; included as a numerical check).
        - ``diagnostic`` : str
            Human-readable summary of the audit.

    Notes
    -----
    The appendix section `A7` in the main manuscript derives these facts
    formally. The function exists as defensive infrastructure: it makes
    the ``no spurious branch'' claim *executable*, so any future parameter
    change that would reintroduce a near-zero equilibrium will be caught by
    the pinning test in ``code/tests/test_analysis.py``.
    """
    if near_zero_threshold is None:
        near_zero_threshold = min(p['mu'] / p['alpha'], 1.0)

    # -- 1. Axis equilibrium ---------------------------------------------------
    O0_eff = p['O0']  # complexity at C=0
    a = p['R'] / p['Vmax']
    b = -(p['R'] - p['R'] * p['eps'] / p['Vmax'] - p['delta'] * O0_eff**p['gamma'])
    c = -p['R'] * p['eps']
    disc = b * b - 4.0 * a * c
    pos_roots: List[float] = []
    if disc >= 0 and a != 0:
        sqrt_disc = float(np.sqrt(disc))
        for V_root in ((-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)):
            if V_root > 0:
                pos_roots.append(V_root)

    axis_eq: Optional[Dict[str, any]] = None
    if pos_roots:
        V_axis = pos_roots[0]  # exactly one by product-of-roots argument
        eigvals, stable = jacobian_eigenvalues(V_axis, 0.0, p)
        re_parts = np.real(eigvals)
        if np.all(re_parts < 0):
            classification = 'stable'
        elif np.all(re_parts > 0):
            classification = 'unstable node/focus'
        else:
            classification = 'saddle'
        axis_eq = dict(
            V=float(V_axis),
            C=0.0,
            eigenvalues=eigvals,
            stable=bool(stable),
            classification=classification,
        )

    # -- 2. Interior equilibria ------------------------------------------------
    interior = find_interior_equilibria(p, n_scan=12000)

    # -- 3. Near-zero filter ---------------------------------------------------
    candidates: List[Dict[str, any]] = []
    if axis_eq is not None and axis_eq['V'] < near_zero_threshold:
        candidates.append({**axis_eq, 'source': 'axis'})
    for eq in interior:
        if eq['V'] < near_zero_threshold:
            candidates.append({**eq, 'source': 'interior'})

    near_zero_branch = candidates[0] if candidates else None

    interior_V_lower_bound = p['mu'] / p['alpha']

    if near_zero_branch is None:
        axis_v_str = f"{axis_eq['V']:.3f}" if axis_eq is not None else "n/a"
        axis_cls_str = axis_eq['classification'] if axis_eq is not None else 'absent'
        diagnostic = (
            f"No ε-regularization near-zero branch detected. "
            f"Interior V* is bounded below by μ/α = {interior_V_lower_bound:.3f} "
            f"(by the C-isocline V_c(C) = μ(1+φO)/α). "
            f"The C=0 axis equilibrium sits at V ≈ {axis_v_str} "
            f"and is classified as '{axis_cls_str}'. "
            f"The axis V-isocline quadratic has exactly "
            f"{len(pos_roots)} positive root(s); its product-of-roots is "
            f"-ε·Vmax = {-p['eps'] * p['Vmax']:.3f}, guaranteeing a single "
            f"positive root for any ε > 0. See Appendix A7."
        )
    else:
        diagnostic = (
            f"Near-zero equilibrium candidate detected at V ≈ "
            f"{near_zero_branch['V']:.4f}, C ≈ {near_zero_branch['C']:.4f} "
            f"(source: {near_zero_branch['source']}). This violates the "
            f"expected structure documented in Appendix A7 and must be "
            f"investigated — verify ε, δ, γ, and the V-isocline analysis."
        )

    return {
        'axis_equilibrium': axis_eq,
        'interior_equilibria': interior,
        'near_zero_branch': near_zero_branch,
        'interior_V_lower_bound': float(interior_V_lower_bound),
        'quadratic_positive_root_count': len(pos_roots),
        'diagnostic': diagnostic,
    }


# ── Global stability audit (R7 issue 2) ───────────────────────────────────────

def bendixson_dulac_certificate(
    p: Dict[str, float],
    V_grid_n: int = 60,
    C_grid_n: int = 60,
    C_trap_safety: float = 1.2,
) -> Dict[str, any]:
    """Verify the Bendixson–Dulac no-closed-orbit certificate on the trapping set.

    The DLVT system admits the Dulac function $B(V, C) = 1/C$ on the
    positive quadrant $\\{C > 0\\}$. Computing $\\nabla \\cdot (B \\mathbf{F})$
    where $\\mathbf{F} = (\\dot V, \\dot C)$ yields:

    .. math::
        \\frac{\\partial (Bf)}{\\partial V}
        = \\frac{1}{C}\\left[ -\\frac{R}{V_{\\max}}
        - \\frac{\\delta O^\\gamma \\varepsilon}{(V+\\varepsilon)^2} \\right]
        < 0,

    .. math::
        \\frac{\\partial (Bg)}{\\partial C}
        = -\\frac{\\alpha V \\varphi \\,(\\beta \\eta C^{\\eta-1})}
               {(1+\\varphi O)^2}
        < 0

    for every $(V, C)$ with $V > 0$ and $C > 0$. By the Bendixson–Dulac
    theorem this rules out closed orbits in the positive quadrant. This
    function verifies the claim *numerically* on a dense grid inside the
    analytical trapping rectangle.

    Analytical trapping rectangle. The rectangle
    $\\Omega = [0, V_{\\max}] \\times [0, C_{\\text{trap}}]$ is forward
    invariant with

    .. math::
        C_{\\text{trap}}^\\eta =
        \\frac{(\\alpha V_{\\max}/\\mu - 1)/\\varphi - O_0}{\\beta},

    which is the unique $C$ at which $\\mu(1 + \\varphi O(C))/\\alpha =
    V_{\\max}$. For any $C > C_{\\text{trap}}$ and any $V \\in [0, V_{\\max}]$,
    we have $\\dot C = C(\\alpha V/(1+\\varphi O) - \\mu) < 0$, so the
    rectangle is a trap.

    Parameters
    ----------
    p : Dict[str, float]
        DLVT parameters.
    V_grid_n, C_grid_n : int
        Grid resolution for the numerical divergence check.
    C_trap_safety : float
        Multiplicative safety factor on the analytical $C_{\\text{trap}}$;
        the grid actually spans $[0, C_{\\text{trap\\_safety}} \\cdot C_{\\text{trap}}]$
        so that any small analytical-to-numerical mismatch is captured.

    Returns
    -------
    Dict[str, any]
        Keys:
        - ``c_trap`` : float, analytical C_{trap}
        - ``max_divergence`` : float, supremum of $\\nabla \\cdot (B\\mathbf{F})$
          over the grid; negative confirms the certificate.
        - ``divergence_is_strictly_negative`` : bool
        - ``dc_dt_above_c_trap_is_negative`` : bool, sanity-check the trap.
        - ``diagnostic`` : str
    """
    # Analytical C_trap from the α V/(1+φO) = μ threshold at V = Vmax.
    # With general η: C_trap^η · β = (αVmax/μ - 1)/φ - O0.
    c_trap = trapping_capital_bound(p)

    C_hi = C_trap_safety * c_trap
    # Skip V=0 and C=0 edges — the Dulac function has a removable singularity
    # at C=0 and the divergence expression below has a 1/C factor.
    Vs = np.linspace(1e-4 * p['Vmax'], p['Vmax'], V_grid_n)
    Cs = np.linspace(1e-3 * C_hi, C_hi, C_grid_n)

    max_div = -np.inf
    for V in Vs:
        for C in Cs:
            O = p['O0'] + p['beta'] * C ** p['eta']
            dOdC = p['beta'] * p['eta'] * C ** (p['eta'] - 1.0)
            d1 = (1.0 / C) * (
                -p['R'] / p['Vmax']
                - p['delta'] * O ** p['gamma'] * p['eps'] / (V + p['eps']) ** 2
            )
            d2 = -(p['alpha'] * V * p['phi'] * dOdC) / (1.0 + p['phi'] * O) ** 2
            div = d1 + d2
            if div > max_div:
                max_div = div

    # Sanity check: above C_trap, dC/dt < 0 for every V ∈ [0, Vmax].
    dc_ok = True
    for V in np.linspace(0.0, p['Vmax'], 20):
        C_test = 1.1 * c_trap
        O_test = p['O0'] + p['beta'] * C_test ** p['eta']
        dC = C_test * (p['alpha'] * V / (1.0 + p['phi'] * O_test) - p['mu'])
        if dC >= 0:
            dc_ok = False
            break

    strictly_negative = bool(max_div < 0.0)
    diagnostic = (
        f"Bendixson–Dulac certificate with B(V,C) = 1/C: max divergence "
        f"over grid = {max_div:.4e}, C_trap = {c_trap:.4f}. "
        f"{'Strictly negative everywhere — no closed orbits in the trapping rectangle.' if strictly_negative else 'NOT strictly negative — certificate fails.'}"
    )
    return {
        'c_trap': float(c_trap),
        'max_divergence': float(max_div),
        'divergence_is_strictly_negative': strictly_negative,
        'dc_dt_above_c_trap_is_negative': dc_ok,
        'diagnostic': diagnostic,
    }


def basin_of_attraction_sweep(
    p: Dict[str, float],
    V0_grid: Optional[List[float]] = None,
    C0_grid: Optional[List[float]] = None,
    T: float = 600.0,
    tol: float = 1e-2,
) -> Dict[str, any]:
    """Integrate the DLVT system from a grid of initial conditions and
    verify convergence to the unique interior equilibrium (Theorem 2).

    This function provides *numerical corroboration* of the global
    asymptotic stability statement proved analytically via
    Bendixson–Dulac + Poincaré–Bendixson in Appendix A10. It is
    intentionally redundant with the theorem: the theorem rules out
    closed orbits and forces every bounded trajectory in
    $\\Omega \\cap \\{C > 0\\}$ to converge to the interior zombie
    equilibrium; this function confirms that no numerical artifacts
    (stiff-step rejections, basin boundaries, etc.) defeat the prediction.

    Parameters
    ----------
    p : Dict[str, float]
        DLVT parameters.
    V0_grid, C0_grid : Optional[List[float]]
        Initial-condition grids. Default: 8 points each along
        $V \\in [0.1, V_{\\max}]$ and $C \\in [0.5, \\text{carrying\\_capacity}(p) \\cdot 2]$.
    T : float
        Integration horizon. Default 600 time units — several
        e-folding times at baseline.
    tol : float
        Tolerance for declaring ``converged to the interior zombie'';
        the trajectory must land within ``tol`` of the target in both
        V and C.

    Returns
    -------
    Dict[str, any]
        Keys:
        - ``zombie_target`` : (V*, C*) tuple
        - ``n_total`` : int
        - ``n_converged`` : int
        - ``max_final_error`` : float, max component-wise error at t=T
        - ``non_converged`` : list of (V0, C0) tuples that failed
        - ``diagnostic`` : str
    """
    # Import here to avoid circular/partial imports at module load.
    from scipy.integrate import solve_ivp
    from .model import dlvt_system

    interior = find_interior_equilibria(p, n_scan=12000)
    if not interior:
        raise ValueError("No interior equilibria found; basin sweep not applicable.")
    eq = interior[0]
    target = (eq['V'], eq['C'])

    if V0_grid is None:
        V0_grid = list(np.linspace(0.1, p['Vmax'], 8))
    if C0_grid is None:
        C_upper = max(1.0, 2.0 * carrying_capacity(p))
        C0_grid = list(np.linspace(0.5, C_upper, 8))

    n_total = 0
    n_conv = 0
    max_err = 0.0
    non_conv: List[Tuple[float, float]] = []
    for V0 in V0_grid:
        for C0 in C0_grid:
            sol = solve_ivp(
                dlvt_system, [0.0, T], [V0, C0], args=(p,),
                method='RK45', rtol=1e-8, atol=1e-10,
            )
            V_final = float(sol.y[0, -1])
            C_final = float(sol.y[1, -1])
            err = max(abs(V_final - target[0]), abs(C_final - target[1]))
            if err < tol:
                n_conv += 1
            else:
                non_conv.append((float(V0), float(C0)))
            if err > max_err:
                max_err = err
            n_total += 1

    diagnostic = (
        f"Basin sweep: {n_conv}/{n_total} initial conditions converged to "
        f"the interior zombie (V*, C*) ≈ ({target[0]:.4f}, {target[1]:.4f}) "
        f"within tol={tol}. Max final error: {max_err:.3e}. "
        f"{'All trajectories converge (numerical corroboration of Theorem 2).' if n_conv == n_total else 'NON-CONVERGENT trajectories detected — investigate.'}"
    )
    return {
        'zombie_target': target,
        'n_total': n_total,
        'n_converged': n_conv,
        'max_final_error': max_err,
        'non_converged': non_conv,
        'diagnostic': diagnostic,
    }
