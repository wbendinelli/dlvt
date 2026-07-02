"""
dlvt.nondimensional
===================
Nondimensionalization and global-sensitivity utilities for the DLVT model.

This module derives and verifies the reduced (dimensionless) form of the DLVT
system, computes exact structural invariances of the interior equilibrium, and
provides global-sensitivity screening tools (elasticities, regime-boundary
maps, and Latin-hypercube sampling with Spearman rank correlations).

Dimensional system (Equations 3.3-3.4 of the paper)
---------------------------------------------------
    dV/dt = R*(1 - V/Vmax) - delta*O^gamma * V/(V + eps)
    dC/dt = alpha*I - mu*C
with
    O = O0 + beta*C^eta          (organisational complexity)
    I = C*V / (1 + phi*O)        (energy-gated impact)

Derivation of the nondimensional (reduced) form
-----------------------------------------------
Introduce the dimensionless state and time

    v   = V / Vmax               (vitality as a fraction of capacity)
    tau = mu * t                 (time in units of the capital-depreciation
                                  time 1/mu)
    w   = (O - O0) / O0          (excess complexity relative to baseline)
        = beta * C^eta / O0

**Vitality equation.**  Since dV/dt = Vmax * mu * dv/dtau and
O = O0*(1 + w),

    Vmax*mu*dv/dtau = R*(1 - v)
                      - delta*O0^gamma*(1 + w)^gamma * v/(v + eps/Vmax),

so, dividing by mu*Vmax,

    dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e),          (N1)

with rho = R/(mu*Vmax), kappa = delta*O0^gamma/(mu*Vmax), e = eps/Vmax.
Equation (N1) is exact for every eta.

**Complexity equation (eta = 1 assumed).**  The clean reduced form below
requires eta = 1 (the baseline calibration); for eta != 1 the w-equation
picks up an extra factor eta*(w*O0/beta)^((eta-1)/eta)*(beta/O0) and the
system is no longer autonomous in the six groups alone.  With eta = 1,
w = beta*C/O0 so dw/dt = (beta/O0)*dC/dt, and since
dC/dt = C*(alpha*V/(1 + phi*O) - mu),

    dw/dt = w*(alpha*Vmax*v / (1 + phi*O0*(1 + w)) - mu).

Dividing by mu (dw/dtau = (1/mu) dw/dt):

    dw/dtau = a * v * w / (1 + f*(1 + w)) - w,                        (N2)

with a = alpha*Vmax/mu and f = phi*O0.

**Reduced system (eta = 1).**

    dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e)
    dw/dtau = a*v*w / (1 + f*(1 + w)) - w

Exactly SIX independent dimensionless groups remain out of the 11 raw
parameters:

    rho   = R / (mu*Vmax)              relative recovery rate      (base: 1.5)
    kappa = delta*O0^gamma / (mu*Vmax) relative baseline drain     (base: 0.01)
    a     = alpha*Vmax / mu            relative capital gain       (base: 5.0)
    f     = phi*O0                     baseline impact suppression (base: 0.15)
    e     = eps / Vmax                 relative regularisation     (base: 0.01)
    gamma                              drain nonlinearity          (base: 2.0)

Bookkeeping: 11 raw parameters, minus eta (fixed at 1 by assumption), minus
three scale choices (t ~ 1/mu, V ~ Vmax, C ~ O0/beta) gives 7; the count
drops to 6 because O0 enters the vector field only through the products
delta*O0^gamma (inside kappa) and phi*O0 (inside f) once w is measured
relative to O0.  beta is absorbed *entirely* into the C-scale: it appears
nowhere in (N1)-(N2) and only re-enters through the map back to dimensional
capital, C = O0*w/beta.  This is the nondimensional statement of Lemma 2
(scope absorption).

Exact structural consequences for the interior equilibrium
----------------------------------------------------------
At an interior equilibrium the pair (V*, O*) solves the CLOSED system

    V* = (mu/alpha) * (1 + phi*O*)                       (from dC/dt = 0)
    R*(1 - V*/Vmax) = delta*O*^gamma * V*/(V* + eps)     (from dV/dt = 0)

which involves neither O0, beta, nor eta.  Hence, exactly (not merely to
numerical tolerance):

* V* and O* have ZERO elasticity with respect to beta, eta, and O0
  (only C* = ((O* - O0)/beta)^(1/eta) depends on them);
* V* depends on (mu, alpha) only through the ratio mu/alpha;
* V* depends on (R, delta) only through the ratio R/delta, and the joint
  scaling (R, delta) -> (s*R, s*delta) leaves (V*, C*, O*) unchanged;
* in the eps -> 0 limit with gamma = 2 (and any eta), O* solves the quadratic

      delta*O*^2 + (R/Vmax)*(mu*phi/alpha)*O* + R*((mu/alpha)/Vmax - 1) = 0,

  which at baseline gives O* = 8.93313 and V* = 4.67994 (the eps = 0.1
  regularised values are O* = 9.00843, V* = 4.70253).

The critical depreciation-to-accumulation ratio (mu/alpha)_crit at which V*
crosses the strategic threshold 0.5*Vmax is ~2.163 at the baseline phi
(baseline mu/alpha = 2 sits ~8% below the flip).

Note that although the interior equilibrium (V*, O*) is independent of O0,
the transient DYNAMICS are not: O0 enters the reduced vector field through
kappa and f.

Global-sensitivity tools
------------------------
v_star_elasticities()    : central-difference elasticities d ln V*/d ln p_i
mu_alpha_critical()      : bisection for the zombie/sustainable flip in mu/alpha
zombie_boundary_map()    : regime grid over (mu/alpha, phi)
zombie_boundary_map_beta(): regime grid over (mu/alpha, beta) - demonstrates
                           that the boundary is invariant in beta
lhs_zombie_fraction()    : log-uniform Latin-hypercube screening with Spearman
                           rank correlations.  This is a RANK-CORRELATION
                           SCREENING (a global-sensitivity proxy), not a full
                           variance-based Sobol decomposition.

References
----------
  Bendinelli, W. (2026). Dynamic Leadership Vitality Theory: A Formal Model
  of the Zombie-Leader Equilibrium. Manuscript submitted to The Leadership
  Quarterly.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc, spearmanr

from .analysis import V_STRATEGIC_FRACTION, find_interior_equilibria

# The 11 raw model parameters, in canonical order.
PARAM_NAMES: List[str] = [
    'R', 'Vmax', 'delta', 'gamma', 'O0', 'beta', 'eta', 'alpha', 'phi', 'mu',
    'eps',
]


# -- Reduced (nondimensional) form --------------------------------------------

def reduced_groups(p: Dict[str, float]) -> Dict[str, float]:
    """Map a dimensional parameter dict to the six dimensionless groups.

    The reduced system (see module docstring) is

        dv/dtau = rho*(1 - v) - kappa*(1 + w)^gamma * v/(v + e)
        dw/dtau = a*v*w / (1 + f*(1 + w)) - w

    and is valid for eta = 1 only.

    Parameters
    ----------
    p : Dict[str, float]
        Dimensional parameter dictionary (11 raw parameters).
        Must have p['eta'] == 1; otherwise the reduced w-equation does not
        close in these groups and a ValueError is raised.

    Returns
    -------
    Dict[str, float]
        Keys: 'rho', 'kappa', 'a', 'f', 'e', 'gamma'.
        Baseline values: rho=1.5, kappa=0.01, a=5.0, f=0.15, e=0.01, gamma=2.

    Raises
    ------
    ValueError
        If p['eta'] != 1.
    """
    if abs(p['eta'] - 1.0) > 1e-12:
        raise ValueError(
            f"reduced_groups requires eta = 1 (got eta = {p['eta']}); the "
            "reduced form does not close in six groups for eta != 1."
        )
    return {
        'rho':   p['R'] / (p['mu'] * p['Vmax']),
        'kappa': p['delta'] * p['O0'] ** p['gamma'] / (p['mu'] * p['Vmax']),
        'a':     p['alpha'] * p['Vmax'] / p['mu'],
        'f':     p['phi'] * p['O0'],
        'e':     p['eps'] / p['Vmax'],
        'gamma': p['gamma'],
    }


def reduced_rhs(tau: float, y: List[float],
                g: Dict[str, float]) -> List[float]:
    """Right-hand side of the reduced (nondimensional) DLVT system.

    Parameters
    ----------
    tau : float
        Dimensionless time (tau = mu*t); the system is autonomous.
    y : List[float]
        Reduced state [v, w] with v = V/Vmax and w = (O - O0)/O0.
    g : Dict[str, float]
        Dimensionless groups from :func:`reduced_groups`.

    Returns
    -------
    List[float]
        [dv/dtau, dw/dtau].
    """
    v = max(y[0], 0.0)
    w = max(y[1], 0.0)
    dv = g['rho'] * (1.0 - v) \
        - g['kappa'] * (1.0 + w) ** g['gamma'] * v / (v + g['e'])
    dw = g['a'] * v * w / (1.0 + g['f'] * (1.0 + w)) - w
    return [dv, dw]


def from_dimensional(V: float, C: float,
                     p: Dict[str, float]) -> Tuple[float, float]:
    """Map dimensional state (V, C) to reduced state (v, w).  Requires eta=1.

    Parameters
    ----------
    V, C : float
        Dimensional vitality and career capital.
    p : Dict[str, float]
        Dimensional parameter dictionary.

    Returns
    -------
    Tuple[float, float]
        (v, w) = (V/Vmax, beta*C/O0).
    """
    return V / p['Vmax'], p['beta'] * C / p['O0']


def to_dimensional(v: np.ndarray, w: np.ndarray,
                   p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Map reduced state (v, w) back to dimensional (V, C).  Requires eta=1.

    Note that beta re-enters ONLY here (C = O0*w/beta); it does not appear
    in the reduced dynamics — the nondimensional face of Lemma 2.

    Parameters
    ----------
    v, w : float or ndarray
        Reduced vitality and excess complexity.
    p : Dict[str, float]
        Dimensional parameter dictionary.

    Returns
    -------
    Tuple[ndarray, ndarray]
        (V, C) = (Vmax*v, O0*w/beta).
    """
    return p['Vmax'] * np.asarray(v), p['O0'] * np.asarray(w) / p['beta']


def simulate_reduced(p: Dict[str, float], V0: float = 8.0, C0: float = 0.5,
                     T: float = 120.0, t_eval: Optional[np.ndarray] = None,
                     max_step: float = 0.05,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the reduced system and map the result back to (t, V, C).

    This is the correctness oracle for the nondimensionalization: for eta = 1
    the returned trajectory must coincide with :func:`dlvt.model.simulate`
    up to integration error (see tests/test_nondimensional.py).

    Parameters
    ----------
    p : Dict[str, float]
        Dimensional parameter dictionary (eta must be 1).
    V0, C0 : float, optional
        Dimensional initial conditions (defaults match dlvt.model.simulate).
    T : float, optional
        Dimensional time horizon; the reduced system is integrated over
        tau in [0, mu*T].
    t_eval : ndarray, optional
        Dimensional times at which to evaluate the solution.  Defaults to
        200 evenly spaced points in [0, T].
    max_step : float, optional
        Maximum solver step in DIMENSIONAL time units (internally scaled by
        mu), so accuracy is comparable to dlvt.model.simulate.

    Returns
    -------
    t : ndarray
        Dimensional time grid.
    V : ndarray
        Vitality trajectory mapped back from v (V = Vmax*v).
    C : ndarray
        Capital trajectory mapped back from w (C = O0*w/beta).
    """
    g = reduced_groups(p)
    v0, w0 = from_dimensional(V0, C0, p)
    if t_eval is None:
        t_eval = np.linspace(0.0, T, 200)
    tau_eval = p['mu'] * np.asarray(t_eval)
    sol = solve_ivp(
        reduced_rhs, [0.0, p['mu'] * T], [v0, w0], args=(g,),
        method='RK45', max_step=p['mu'] * max_step, t_eval=tau_eval,
        rtol=1e-10, atol=1e-12,
    )
    V, C = to_dimensional(sol.y[0], sol.y[1], p)
    return np.asarray(t_eval), V, C


# -- Equilibrium helpers -------------------------------------------------------

def _generous_c_max(p: Dict[str, float]) -> float:
    """Scan window for find_interior_equilibria that avoids the small-beta bug.

    The equilibrium capital scales as C* = (O* - O0)/beta ~ 1/beta, so a
    fixed window silently loses the equilibrium at small beta (the source of
    the historical beta_crit = 0.1015 artifact; see
    dlvt.analysis.estimate_bifurcation_interval).
    """
    return max(300.0, 20.0 / p['beta'])


def stable_equilibrium(p: Dict[str, float]) -> Optional[Dict[str, object]]:
    """Return the lowest-C stable interior equilibrium, or None.

    Uses :func:`dlvt.analysis.find_interior_equilibria` with a generous,
    beta-aware C_max (see :func:`_generous_c_max`).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    Optional[Dict]
        The equilibrium dict (keys 'V', 'C', 'O', 'I', 'stable',
        'eigenvalues', 'zombie') for the lowest-C stable equilibrium, or
        None if no stable interior equilibrium exists.
    """
    eqs = find_interior_equilibria(p, C_max=_generous_c_max(p))
    stable = [eq for eq in eqs if eq['stable']]
    if not stable:
        return None
    stable.sort(key=lambda eq: eq['C'])
    return stable[0]


def _v_star(p: Dict[str, float]) -> float:
    """Equilibrium V* at the lowest-C stable equilibrium; raises if none."""
    eq = stable_equilibrium(p)
    if eq is None:
        raise ValueError("No stable interior equilibrium for these parameters.")
    return float(eq['V'])


# -- Local elasticities of V* --------------------------------------------------

def v_star_elasticities(p: Dict[str, float],
                        rel: float = 1e-4) -> Dict[str, float]:
    """Central-difference elasticities d ln V* / d ln p_i for all 11 parameters.

    Each parameter is perturbed multiplicatively to p_i*(1 +/- rel) and the
    elasticity is the log-log central difference

        E_i = [ln V*(p_i*(1+rel)) - ln V*(p_i*(1-rel))]
              / [ln(1+rel) - ln(1-rel)].

    Structural expectations at baseline (see module docstring):
    E_beta = E_eta = E_O0 = 0 exactly; E_R = -E_delta (R/delta ratio);
    E_mu = -E_alpha (mu/alpha ratio).

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary; a stable interior equilibrium must exist at p
        and at each perturbed point.
    rel : float, optional
        Relative perturbation size, default 1e-4.

    Returns
    -------
    Dict[str, float]
        Elasticity for each of the 11 parameters in PARAM_NAMES.
    """
    denom = np.log(1.0 + rel) - np.log(1.0 - rel)
    out: Dict[str, float] = {}
    for name in PARAM_NAMES:
        p_hi = dict(p)
        p_lo = dict(p)
        p_hi[name] = p[name] * (1.0 + rel)
        p_lo[name] = p[name] * (1.0 - rel)
        v_hi = _v_star(p_hi)
        v_lo = _v_star(p_lo)
        out[name] = float((np.log(v_hi) - np.log(v_lo)) / denom)
    return out


# -- Critical mu/alpha ratio ----------------------------------------------------

def mu_alpha_critical(p: Dict[str, float], lo: float = 0.5, hi: float = 10.0,
                      tol: float = 1e-6, max_iter: int = 100) -> float:
    """Bisection for the ratio r = mu/alpha at which V* crosses 0.5*Vmax.

    Holds alpha fixed at p['alpha'] and varies mu = r*alpha (by the exact
    mu/alpha degeneracy of V*, only the ratio matters).  V*(r) is increasing
    in r, so the zombie/sustainable flip is a single crossing.  If no stable
    interior equilibrium exists at some trial r (which happens for large r,
    where V* would exceed Vmax), that trial is treated as being above the
    threshold.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    lo, hi : float, optional
        Initial bracket for r = mu/alpha.  V*(lo) must be below the
        strategic threshold and V*(hi) above it (or infeasible).
    tol : float, optional
        Absolute tolerance on r, default 1e-6.
    max_iter : int, optional
        Maximum bisection iterations, default 100.

    Returns
    -------
    float
        The critical ratio (mu/alpha)_crit.  Baseline: ~2.163.

    Raises
    ------
    ValueError
        If the initial bracket does not straddle the threshold.
    """
    target = V_STRATEGIC_FRACTION * p['Vmax']

    def excess(r: float) -> Optional[float]:
        """V*(r) - target, or None if no stable interior equilibrium."""
        p_trial = dict(p)
        p_trial['mu'] = r * p['alpha']
        try:
            return _v_star(p_trial) - target
        except ValueError:
            return None

    g_lo = excess(lo)
    if g_lo is None or g_lo >= 0.0:
        raise ValueError(
            f"mu_alpha_critical: V*(lo={lo}) is not below the threshold; "
            "widen the bracket downward."
        )
    g_hi = excess(hi)
    if g_hi is not None and g_hi <= 0.0:
        raise ValueError(
            f"mu_alpha_critical: V*(hi={hi}) is not above the threshold; "
            "widen the bracket upward."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = excess(mid)
        if g_mid is not None and g_mid < 0.0:
            lo = mid
        else:
            # Above threshold, or infeasible (equilibrium lost at high r).
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# -- Regime boundary maps --------------------------------------------------------

def classify_point(p: Dict[str, float]) -> str:
    """Classify a parameter point as 'zombie', 'sustainable', or 'none'.

    'none' means no stable interior equilibrium exists (collapse-prone /
    infeasible).  Otherwise the lowest-C stable equilibrium is classified by
    the strategic threshold V_strategic = 0.5*Vmax.

    Parameters
    ----------
    p : Dict[str, float]
        Parameter dictionary.

    Returns
    -------
    str
        One of 'zombie', 'sustainable', 'none'.
    """
    eq = stable_equilibrium(p)
    if eq is None:
        return 'none'
    return 'zombie' if eq['zombie'] else 'sustainable'


def zombie_boundary_map(p: Dict[str, float],
                        r_range: Tuple[float, float] = (1.0, 4.0),
                        phi_range: Tuple[float, float] = (0.02, 0.40),
                        n: int = 41) -> Dict[str, object]:
    """Regime map over the (mu/alpha, phi) plane.

    For each grid point, mu is set to r*alpha (alpha held fixed) and phi is
    set to the grid value; the point is classified with
    :func:`classify_point`.  The zombie/sustainable boundary in this plane is
    the curve (mu/alpha)_crit(phi); at baseline phi = 0.15 it sits at
    r ~ 2.163, about 8% above the baseline ratio mu/alpha = 2.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    r_range : Tuple[float, float], optional
        (min, max) for r = mu/alpha, default (1.0, 4.0).
    phi_range : Tuple[float, float], optional
        (min, max) for phi, default (0.02, 0.40).
    n : int, optional
        Grid points per axis, default 41.

    Returns
    -------
    Dict
        Keys:
        - 'r_values'   : ndarray (n,), the mu/alpha axis
        - 'phi_values' : ndarray (n,), the phi axis
        - 'regimes'    : object ndarray (n, n); regimes[i, j] is the class at
                         phi=phi_values[i], r=r_values[j]
        - 'baseline'   : (mu/alpha, phi) of the input p
        - 'axes'       : ('mu/alpha', 'phi')
    """
    r_values = np.linspace(r_range[0], r_range[1], n)
    phi_values = np.linspace(phi_range[0], phi_range[1], n)
    regimes = np.empty((n, n), dtype=object)
    for i, phi in enumerate(phi_values):
        for j, r in enumerate(r_values):
            p_trial = dict(p)
            p_trial['mu'] = r * p['alpha']
            p_trial['phi'] = phi
            regimes[i, j] = classify_point(p_trial)
    return {
        'r_values': r_values,
        'phi_values': phi_values,
        'regimes': regimes,
        'baseline': (p['mu'] / p['alpha'], p['phi']),
        'axes': ('mu/alpha', 'phi'),
    }


def zombie_boundary_map_beta(p: Dict[str, float],
                             r_range: Tuple[float, float] = (1.0, 4.0),
                             beta_range: Tuple[float, float] = (0.05, 1.0),
                             n: int = 41) -> Dict[str, object]:
    """Regime map over the (mu/alpha, beta) plane — beta-invariance check.

    Same shape as :func:`zombie_boundary_map`, but the second axis is beta.
    Because V* has exactly zero elasticity in beta (Lemma 2 / the reduced
    form), every column of the returned grid (fixed r, varying beta) must be
    constant: the zombie boundary is a vertical line, invariant in beta.
    The boolean 'boundary_invariant_in_beta' reports this check.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary.
    r_range : Tuple[float, float], optional
        (min, max) for r = mu/alpha, default (1.0, 4.0).
    beta_range : Tuple[float, float], optional
        (min, max) for beta, default (0.05, 1.0).
    n : int, optional
        Grid points per axis, default 41.

    Returns
    -------
    Dict
        Keys: 'r_values', 'beta_values', 'regimes' (object ndarray (n, n),
        rows indexed by beta), 'baseline', 'axes',
        'boundary_invariant_in_beta' (bool).
    """
    r_values = np.linspace(r_range[0], r_range[1], n)
    beta_values = np.linspace(beta_range[0], beta_range[1], n)
    regimes = np.empty((n, n), dtype=object)
    for i, beta in enumerate(beta_values):
        for j, r in enumerate(r_values):
            p_trial = dict(p)
            p_trial['mu'] = r * p['alpha']
            p_trial['beta'] = beta
            regimes[i, j] = classify_point(p_trial)
    invariant = all(
        len({regimes[i, j] for i in range(n)}) == 1 for j in range(n)
    )
    return {
        'r_values': r_values,
        'beta_values': beta_values,
        'regimes': regimes,
        'baseline': (p['mu'] / p['alpha'], p['beta']),
        'axes': ('mu/alpha', 'beta'),
        'boundary_invariant_in_beta': invariant,
    }


# -- Latin-hypercube global screening ---------------------------------------------

def lhs_zombie_fraction(p: Dict[str, float], n_samples: int = 600,
                        factor: float = 2.0, seed: int = 1
                        ) -> Dict[str, object]:
    """Log-uniform Latin-hypercube screening of regime outcomes and V*.

    Draws a seeded Latin hypercube (scipy.stats.qmc.LatinHypercube) over all
    11 raw parameters, each log-uniform on [p_i/factor, p_i*factor], and for
    each draw records whether a stable interior equilibrium exists and, if
    so, whether it is a zombie (V* < 0.5*Vmax).  Spearman rank correlations
    between each log-parameter and V* (over the stable draws) are returned as
    a global-sensitivity proxy.

    HONEST LABELLING: this is a rank-correlation SCREENING, not a full
    variance-based (Sobol) decomposition.  Spearman coefficients capture
    monotone marginal effects; they do not decompose interactions.

    Parameters
    ----------
    p : Dict[str, float]
        Baseline parameter dictionary (centre of the hypercube).
    n_samples : int, optional
        Number of LHS draws, default 600.
    factor : float, optional
        Half-width of the log-uniform range (each parameter spans
        [p_i/factor, p_i*factor]), default 2.0.
    seed : int, optional
        RNG seed for the Latin hypercube, default 1 (deterministic).

    Returns
    -------
    Dict
        Keys:
        - 'n_samples', 'factor', 'seed'  : the inputs
        - 'n_stable'                     : draws with a stable interior eq.
        - 'n_zombie'                     : stable draws classified zombie
        - 'frac_stable'                  : n_stable / n_samples
        - 'zombie_fraction_given_stable' : n_zombie / n_stable
        - 'zombie_fraction_overall'      : n_zombie / n_samples
        - 'v_stars'                      : ndarray of V* over stable draws
        - 'spearman'                     : dict param -> Spearman rho between
                                           log(param) and V* (stable draws)
        - 'spearman_pvalues'             : dict param -> two-sided p-value
        - 'method'                       : honest-labelling note

    Notes
    -----
    At baseline with factor=2.0 the expected zombie share among stable draws
    is ~0.49 +/- 0.1.  Equilibria are searched with the generous beta-aware
    C_max window (see :func:`_generous_c_max`).
    """
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    U = sampler.random(n=n_samples)

    log_x = np.empty_like(U)
    stable_mask = np.zeros(n_samples, dtype=bool)
    zombie_mask = np.zeros(n_samples, dtype=bool)
    v_stars = np.full(n_samples, np.nan)

    for i in range(n_samples):
        p_trial = dict(p)
        for j, name in enumerate(PARAM_NAMES):
            # log-uniform on [p/factor, p*factor]
            p_trial[name] = p[name] * factor ** (2.0 * U[i, j] - 1.0)
            log_x[i, j] = np.log(p_trial[name])
        eq = stable_equilibrium(p_trial)
        if eq is not None:
            stable_mask[i] = True
            v_stars[i] = eq['V']
            zombie_mask[i] = bool(
                eq['V'] < V_STRATEGIC_FRACTION * p_trial['Vmax']
            )

    n_stable = int(stable_mask.sum())
    n_zombie = int(zombie_mask.sum())
    v_stable = v_stars[stable_mask]

    spearman: Dict[str, float] = {}
    pvalues: Dict[str, float] = {}
    for j, name in enumerate(PARAM_NAMES):
        rho_s, pv = spearmanr(log_x[stable_mask, j], v_stable)
        spearman[name] = float(rho_s)
        pvalues[name] = float(pv)

    return {
        'n_samples': n_samples,
        'factor': factor,
        'seed': seed,
        'n_stable': n_stable,
        'n_zombie': n_zombie,
        'frac_stable': n_stable / n_samples,
        'zombie_fraction_given_stable':
            (n_zombie / n_stable) if n_stable else float('nan'),
        'zombie_fraction_overall': n_zombie / n_samples,
        'v_stars': v_stable,
        'spearman': spearman,
        'spearman_pvalues': pvalues,
        'method': (
            'Log-uniform Latin-hypercube screening with Spearman rank '
            'correlations between log-parameters and V* (stable draws). '
            'This is a rank-correlation screening / global-sensitivity '
            'proxy, NOT a full variance-based Sobol decomposition.'
        ),
    }
