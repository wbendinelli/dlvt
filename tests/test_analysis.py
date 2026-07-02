"""
Tests for dlvt.analysis: carrying capacity, equilibria, stability, regimes.

Pins analytical results to the final paper (§3.6–3.9 and Appendix A4–A6).
"""

import numpy as np
import pytest

from dlvt.model import make_params
from dlvt.analysis import (
    V_STRATEGIC_FRACTION,
    basin_of_attraction_sweep,
    bendixson_dulac_certificate,
    carrying_capacity,
    estimate_bifurcation_interval,
    find_interior_equilibria,
    find_regularization_branch,
)


# ────────────────────────────────────────────────────────────────────────────
# Carrying capacity — Proposition 3 / Eq. (3.14)
# ────────────────────────────────────────────────────────────────────────────


def test_carrying_capacity_baseline_equals_45():
    """At baseline params, C*_max = 45.0 (Table 1 of the paper)."""
    p = make_params()
    assert carrying_capacity(p) == pytest.approx(45.0, abs=0.1)


def test_carrying_capacity_formula_matches_definition():
    """C*_max = ((R/delta)^(1/gamma) - O0) / beta for eta=1."""
    p = make_params()
    Omax = (p["R"] / p["delta"]) ** (1.0 / p["gamma"])
    expected = (Omax - p["O0"]) / p["beta"]
    assert carrying_capacity(p) == pytest.approx(expected, rel=1e-12)


def test_carrying_capacity_eta2_paper_value():
    """Table 1: C*_max drops from 45.0 at eta=1 to ~6.7 at eta=2."""
    p = make_params(eta=2.0)
    assert carrying_capacity(p) == pytest.approx(6.7, abs=0.1)


def test_carrying_capacity_eta1_5_paper_value():
    """Table 1: C*_max ≈ 12.7 at eta=1.5."""
    p = make_params(eta=1.5)
    assert carrying_capacity(p) == pytest.approx(12.7, abs=0.1)


def test_carrying_capacity_monotone_in_beta():
    """dC*_max/dbeta < 0 (Eq. 3.15)."""
    betas = [0.10, 0.15, 0.25, 0.40, 0.60]
    values = [carrying_capacity(make_params(beta=b)) for b in betas]
    # strictly decreasing
    for a, b in zip(values, values[1:]):
        assert a > b, f"C*_max not monotone in beta: {values}"


def test_carrying_capacity_monotone_in_R():
    """dC*_max/dR > 0: more recovery raises the ceiling."""
    Rs = [1.0, 2.0, 3.0, 4.5, 6.0]
    values = [carrying_capacity(make_params(R=r)) for r in Rs]
    for a, b in zip(values, values[1:]):
        assert a < b, f"C*_max not monotone in R: {values}"


def test_carrying_capacity_zero_when_O0_too_large():
    """If baseline complexity already exceeds the drain ceiling, C*_max = 0."""
    # Force (R/delta)^{1/gamma} < O0
    p = make_params(O0=100.0)
    assert carrying_capacity(p) == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Interior equilibria and stability — Theorem 2
# ────────────────────────────────────────────────────────────────────────────


def test_baseline_equilibrium_is_zombie_and_stable():
    """Baseline calibration converges to stable zombie equilibrium.

    The paper reports V* ≈ 4.70, C* ≈ 32 at the baseline. The Jacobian
    should have complex eigenvalues with negative real parts (damped spiral).
    """
    p = make_params()
    eqs = find_interior_equilibria(p)
    assert len(eqs) >= 1
    zombies = [e for e in eqs if e.get("zombie") and e.get("stable")]
    assert len(zombies) == 1
    eq = zombies[0]
    assert eq["V"] == pytest.approx(4.70, abs=0.05)
    assert eq["C"] == pytest.approx(32.0, abs=0.3)
    # Damped oscillatory convergence: complex conjugate eigenvalues
    lam = np.asarray(eq["eigenvalues"])
    assert np.all(np.real(lam) < 0.0), f"unstable: eigs={lam}"
    assert np.any(np.imag(lam) != 0.0), (
        f"expected complex eigenvalues (spiral), got {lam}"
    )


def test_sustainable_regime_not_zombie():
    """Low-coupling, low-delta regime yields non-zombie stable equilibrium."""
    p = make_params(delta=0.008, beta=0.15)
    eqs = find_interior_equilibria(p)
    assert any(not e.get("zombie") and e.get("stable") for e in eqs)


def test_high_coupling_regime_is_zombie():
    """Doubling beta while holding everything else fixed should still yield
    a zombie equilibrium (lower V*, lower C*)."""
    p = make_params(beta=0.5)
    eqs = find_interior_equilibria(p)
    zombies = [e for e in eqs if e.get("zombie") and e.get("stable")]
    assert len(zombies) >= 1
    eq = zombies[0]
    assert eq["V"] < V_STRATEGIC_FRACTION * p["Vmax"]


# ────────────────────────────────────────────────────────────────────────────
# Bifurcation interval diagnostic — R7-6 / Appendix A8
# ────────────────────────────────────────────────────────────────────────────

# A single session-scoped estimate shared by all four R7-6 tests. The grid is
# deliberately small (3 eps × 1 n_scan × 60 beta points) so the whole group
# finishes in well under a minute; the appendix uses a wider grid for reporting.

@pytest.fixture(scope="module")
def bifurcation_result():
    p = make_params()
    return estimate_bifurcation_interval(
        p,
        eps_grid=[0.05, 0.1, 0.2],
        n_scan_grid=[4000],
        n_beta=60,
    )


def test_bifurcation_interval_baseline_does_not_cross_strategic(bifurcation_result):
    """R7-6: under baseline parameters, V*(β) does not cross V_strategic.

    This is the scope-absorption property of Lemma 2 made mechanical. The
    earlier claim β_crit ≈ 0.1015 was a scan-window artifact; the honest
    report is that no V*-crossing exists at this calibration.
    """
    assert bifurcation_result["crosses_threshold"] is False
    assert bifurcation_result["beta_crit_interval"] is None


def test_bifurcation_interval_baseline_v_star_is_invariant(bifurcation_result):
    """R7-6: V*(β) is numerically constant across β at the baseline eps.

    Pins Lemma 2: V* depends on β only through β·C*^η, which the equilibrium
    conditions force to be invariant. The reference value 4.7025 is the
    scope-absorption invariant at eps=0.1.
    """
    assert bifurcation_result["v_star_invariant"] is not None
    assert abs(bifurcation_result["v_star_invariant"] - 4.7025) < 5e-3


def test_bifurcation_interval_baseline_beta_C_product_is_pinned(bifurcation_result):
    """R7-6: the invariant product β·C* is pinned to its analytical value.

    From the equilibrium conditions with γ=2, η=1 and baseline parameters,
    β·C* = 8.0083 (derived from V* = μ(1+φO*)/α = 4.7025 and the definition
    of O*). This product is the structural invariant that Lemma 2 names.
    """
    assert bifurcation_result["baseline_beta_C_product"] is not None
    assert abs(bifurcation_result["baseline_beta_C_product"] - 8.0083) < 5e-3


def test_bifurcation_interval_diagnostic_names_the_artifact(bifurcation_result):
    """R7-6: the diagnostic string explicitly identifies the scan-window artifact.

    The earlier 'β_crit ≈ 0.1015' figure was a function of the legacy C_max=80
    in find_interior_equilibria — specifically, it was the smallest β at which
    C*(β) = 8.008/β fell below 80. The diagnostic must surface this so a
    reader cannot accidentally reproduce the false-precision claim.
    """
    diag = bifurcation_result["diagnostic"].lower()
    assert "scan" in diag or "c_max" in diag
    assert "artifact" in diag or "invariant" in diag


# ────────────────────────────────────────────────────────────────────────────
# ε-regularization branch audit — R7 issue 3 / Appendix A7
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def regularization_report():
    """Cache the expensive enumeration across R7-3 tests."""
    return find_regularization_branch(make_params())


def test_regularization_branch_no_near_zero_at_baseline(regularization_report):
    """No ε-spurious near-zero equilibrium exists at baseline parameters.

    Pins the core claim of Appendix A7: the smooth barrier V/(V+ε) does
    *not* introduce a second equilibrium branch at small positive V. The
    V-isocline quadratic has product-of-roots -ε·Vmax = -1.0 < 0, so
    exactly one positive root per fixed C; the interior C-isocline bounds
    V* ≥ μ/α = 2.0. Therefore no interior or axis equilibrium can sit
    below the near-zero threshold.
    """
    assert regularization_report["near_zero_branch"] is None
    assert regularization_report["interior_V_lower_bound"] == pytest.approx(2.0, abs=1e-9)
    assert regularization_report["quadratic_positive_root_count"] == 1


def test_regularization_branch_axis_equilibrium_is_saddle(regularization_report):
    """The C=0 axis equilibrium is a saddle at baseline.

    The axis V-isocline quadratic at baseline gives V ≈ 9.934, and the
    Jacobian there is upper-triangular (J[1,0] = αC/(1+φO) = 0) with
    diagonal entries of opposite sign: one negative eigenvalue (vitality
    relaxation toward V*) and one positive eigenvalue (αV/(1+φO) - μ > 0).
    This classifies the axis root as a saddle, not a stable node, so it
    is *not* an attractor for any interior trajectory.
    """
    axis = regularization_report["axis_equilibrium"]
    assert axis is not None
    assert axis["classification"] == "saddle"
    assert not axis["stable"]
    # Pin the specific baseline location (Chapter 3, Appendix A7).
    assert axis["V"] == pytest.approx(9.934, abs=1e-2)
    assert axis["C"] == 0.0
    eigvals = np.real(axis["eigenvalues"])
    assert np.min(eigvals) < 0 < np.max(eigvals)


def test_regularization_branch_interior_is_unique_zombie(regularization_report):
    """The interior branch at baseline contains exactly one equilibrium (the zombie)."""
    interior = regularization_report["interior_equilibria"]
    assert len(interior) == 1
    eq = interior[0]
    assert eq["V"] == pytest.approx(4.7025, abs=5e-3)
    assert eq["C"] == pytest.approx(32.0337, abs=5e-3)
    assert eq["stable"] is True
    assert eq["zombie"] is True


def test_regularization_branch_diagnostic_cites_appendix(regularization_report):
    """The diagnostic string explicitly names Appendix A7 and the analytical bound."""
    diag = regularization_report["diagnostic"].lower()
    assert "appendix a7" in diag
    assert "μ/α" in diag or "mu/alpha" in diag or "2.000" in diag
    assert "no" in diag and ("near-zero" in diag or "near zero" in diag)


# ────────────────────────────────────────────────────────────────────────────
# Global asymptotic stability — R7 issue 2 / Appendix A10
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bendixson_report():
    """Cache the Bendixson-Dulac certificate computation."""
    return bendixson_dulac_certificate(make_params())


@pytest.fixture(scope="module")
def basin_report():
    """Cache the basin-of-attraction sweep; integrates 64 ODEs."""
    return basin_of_attraction_sweep(make_params())


def test_bendixson_dulac_divergence_strictly_negative(bendixson_report):
    """Pin the Dulac certificate: max divergence must be < 0.

    This is the numerical witness of the analytical Bendixson-Dulac result
    with B(V,C) = 1/C. Both ∂(Bf)/∂V and ∂(Bg)/∂C are strictly negative
    on {V>0, C>0}, so their sum is bounded away from zero — confirming
    that no closed orbits exist in the trapping rectangle, which, combined
    with uniqueness of the interior equilibrium and Poincaré-Bendixson,
    gives global asymptotic stability (Appendix A10).
    """
    assert bendixson_report["divergence_is_strictly_negative"]
    assert bendixson_report["max_divergence"] < -1e-4


def test_bendixson_dulac_trapping_rectangle_is_valid(bendixson_report):
    """Pin the analytical C_trap and its forward-invariance check."""
    # Baseline C_trap = ((α·Vmax/μ - 1)/φ - O0)/β = ((5 - 1)/0.15 - 1)/0.25
    #                 = (26.667 - 1)/0.25 = 102.667.
    assert bendixson_report["c_trap"] == pytest.approx(102.667, abs=1e-2)
    assert bendixson_report["dc_dt_above_c_trap_is_negative"]


def test_basin_sweep_all_trajectories_converge_to_zombie(basin_report):
    """Numerical corroboration of Theorem 2: every IC on the default grid
    converges to the unique interior zombie equilibrium."""
    assert basin_report["n_converged"] == basin_report["n_total"]
    assert basin_report["n_total"] >= 64  # 8 × 8 default grid
    assert basin_report["max_final_error"] < 1e-2
    assert basin_report["non_converged"] == []


def test_basin_sweep_target_matches_zombie_equilibrium(basin_report):
    """The target used by the basin sweep is the actual interior zombie."""
    V_target, C_target = basin_report["zombie_target"]
    assert V_target == pytest.approx(4.7025, abs=5e-3)
    assert C_target == pytest.approx(32.0337, abs=5e-3)


def test_linear_drain_gamma1_yields_sustainable_equilibrium():
    """Robustness check (§3.10, corrected): at gamma=1 the equilibrium is SUSTAINABLE.

    Earlier drafts said the equilibrium "disappears" at gamma=1. That was wrong:
    a unique stable interior equilibrium persists, but with V* ≈ 8.56 — well above
    the strategic threshold. Linear drain *converts the low-vitality (zombie)
    equilibrium into a sustainable one*; nonlinear complexity scaling (gamma > 1)
    is necessary for the low-vitality regime, not for the existence of an attractor.
    """
    p = make_params(gamma=1.0)
    eqs = find_interior_equilibria(p, n_scan=12000)
    assert len(eqs) == 1, f"expected unique interior equilibrium, got {len(eqs)}"
    eq = eqs[0]
    assert eq["stable"] is True
    assert eq["zombie"] is False, "gamma=1 equilibrium should be sustainable"
    assert eq["V"] == pytest.approx(8.559, abs=0.02)
    assert eq["V"] > V_STRATEGIC_FRACTION * p["Vmax"]


# ────────────────────────────────────────────────────────────────────────────
# Independent verification checks (not pinned to the solver's own output)
#
# Each test below checks the code against a source of truth *external* to
# find_interior_equilibria itself: a closed-form root, a second integrator,
# a hand-derived boundary, or an analytic sign condition evaluated directly.
# ────────────────────────────────────────────────────────────────────────────


def test_equilibrium_matches_eps_to_zero_closed_form_oracle():
    """External oracle: the ε → 0 equilibrium has a closed form (γ=2, η=1).

    Substituting the C-nullcline V = (μ/α)(1+φO) into the ε → 0 V-nullcline
    R(1 − V/Vmax) = δO² gives the quadratic

        δ·O² + (R/Vmax)(μφ/α)·O + R((μ/α)/Vmax − 1) = 0,

    whose positive root at baseline is O* = 8.93313, hence V* = 4.67994 and
    C* = 31.7325 (β·C* = 7.93313). The numerical solver at ε = 1e-6 must
    reproduce this closed form — a check fully independent of the scan/brentq
    pipeline that produced the paper's reference numbers.
    """
    p = make_params(eps=1e-6)
    a = p["delta"]
    b = (p["R"] / p["Vmax"]) * (p["mu"] * p["phi"] / p["alpha"])
    c = p["R"] * ((p["mu"] / p["alpha"]) / p["Vmax"] - 1.0)
    O_star = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    V_star = (p["mu"] / p["alpha"]) * (1.0 + p["phi"] * O_star)
    C_star = (O_star - p["O0"]) / p["beta"]

    assert O_star == pytest.approx(8.93313, abs=1e-4)
    assert V_star == pytest.approx(4.67994, abs=1e-4)

    eqs = find_interior_equilibria(p, n_scan=16000)
    assert len(eqs) == 1
    assert eqs[0]["V"] == pytest.approx(V_star, abs=1e-3)
    assert eqs[0]["C"] == pytest.approx(C_star, abs=1e-2)


def test_equilibrium_confirmed_by_independent_stiff_integrator():
    """External oracle: an implicit (Radau) integration lands on the equilibrium.

    The package integrates with RK45 everywhere; this check uses a different
    method family (implicit Runge–Kutta, Radau IIA) so that solver-specific
    artifacts cannot silently pin both the equilibrium finder and the tests.
    """
    from scipy.integrate import solve_ivp
    from dlvt.model import dlvt_system

    p = make_params()
    sol = solve_ivp(
        dlvt_system, [0.0, 400.0], [8.0, 5.0], args=(p,),
        method="Radau", rtol=1e-9, atol=1e-11,
    )
    assert sol.success
    assert sol.y[0, -1] == pytest.approx(4.7025, abs=1e-2)
    assert sol.y[1, -1] == pytest.approx(32.034, abs=1e-1)


def draw_f(rng, base):
    """Log-uniform ±2× draw around a base value."""
    return float(base * np.exp(rng.uniform(np.log(0.5), np.log(2.0))))


def test_uniqueness_no_multiple_equilibria_across_random_parameters():
    """Theorem 2a witness: at most one interior equilibrium, all regimes.

    dΦ/dO < 0 term-by-term along the C-nullcline implies at most one interior
    equilibrium in the whole positive orthant for every positive parameter
    vector. Sweep 100 random log-uniform parameter draws (±2×) and assert no
    draw ever produces two interior equilibria.
    """
    rng = np.random.default_rng(20260701)

    for _ in range(100):
        p = make_params(
            R=draw_f(rng, 3.0), delta=draw_f(rng, 0.02), gamma=draw_f(rng, 2.0),
            O0=draw_f(rng, 1.0), beta=draw_f(rng, 0.25), eta=draw_f(rng, 1.0),
            alpha=draw_f(rng, 0.1), phi=draw_f(rng, 0.15), mu=draw_f(rng, 0.2),
        )
        eqs = find_interior_equilibria(p, n_scan=4000)
        assert len(eqs) <= 1, (
            f"multiple interior equilibria found — contradicts Theorem 2a: "
            f"params={p}, eqs={[(e['V'], e['C']) for e in eqs]}"
        )


def test_no_hopf_trace_negative_at_every_interior_equilibrium():
    """Theorem 2b witness: trace(J) < 0 at every interior equilibrium.

    At any interior equilibrium dC/dt = 0 forces αV/(1+φO) = μ, which makes
    J[1,1] = −αCVφ(dO/dC)/(1+φO)² < 0, while J[0,0] < 0 identically — so no
    Hopf bifurcation exists anywhere in parameter space. Evaluate the trace
    directly (bypassing jacobian_eigenvalues) on random draws.
    """
    rng = np.random.default_rng(8008)
    checked = 0
    for _ in range(100):
        p = make_params(
            R=draw_f(rng, 3.0), delta=draw_f(rng, 0.02), gamma=draw_f(rng, 2.0),
            O0=draw_f(rng, 1.0), beta=draw_f(rng, 0.25), eta=draw_f(rng, 1.0),
            alpha=draw_f(rng, 0.1), phi=draw_f(rng, 0.15), mu=draw_f(rng, 0.2),
        )
        for eq in find_interior_equilibria(p, n_scan=4000):
            V, C = eq["V"], eq["C"]
            O = p["O0"] + p["beta"] * C ** p["eta"]
            dOdC = p["beta"] * p["eta"] * C ** (p["eta"] - 1.0)
            J00 = (-p["R"] / p["Vmax"]
                   - p["delta"] * O ** p["gamma"] * p["eps"] / (V + p["eps"]) ** 2)
            J11 = (p["alpha"] * V / (1.0 + p["phi"] * O)
                   - p["alpha"] * C * V * p["phi"] * dOdC / (1.0 + p["phi"] * O) ** 2
                   - p["mu"])
            assert J00 + J11 < 0.0, (
                f"trace ≥ 0 at interior equilibrium — Hopf candidate: params={p}"
            )
            checked += 1
    assert checked >= 50, f"too few interior equilibria sampled ({checked})"


def test_mu_alpha_critical_boundary_at_baseline_phi():
    """Hand-derived regime boundary: (μ/α)_crit ≈ 2.163 at baseline φ.

    The regime label is governed by μ/α: V* = (μ/α)(1+φO*). Holding α = 0.1,
    μ = 0.2163 puts V* at the threshold (5.0006), μ = 0.20 (baseline) is
    'zombie', and μ = 0.25 is 'sustainable' with V* ≈ 5.58. This pins the
    calibration-dependence of the headline classification: the baseline sits
    ~8% from the flip.
    """
    from dlvt.analysis import classify_regime

    assert classify_regime(make_params(mu=0.20)) == "zombie"
    assert classify_regime(make_params(mu=0.25)) == "sustainable"

    eq_at_crit = find_interior_equilibria(make_params(mu=0.2163))[0]
    assert eq_at_crit["V"] == pytest.approx(5.0, abs=0.01)

    eq_sust = find_interior_equilibria(make_params(mu=0.25))[0]
    assert eq_sust["V"] == pytest.approx(5.58, abs=0.01)


def test_small_beta_equilibrium_found_with_default_window():
    """Regression for the fixed-window bug (M6): β < 0.066 must not be mislabeled.

    With the legacy fixed default C_max = 120, C*(β) ≈ 8.008/β exceeded the
    window for β < 0.066, find_interior_equilibria returned [], and
    classify_regime mislabeled the regime as 'collapse-prone'. The corrected
    default derives the window from C_trap ∝ 1/β, so the equilibrium is found
    at every β and scope absorption holds: V* = 4.7025 with β·C* conserved.
    """
    from dlvt.analysis import classify_regime

    for beta in (0.05, 0.02, 0.01):
        p = make_params(beta=beta)
        eqs = find_interior_equilibria(p)
        assert len(eqs) == 1, f"equilibrium missed at beta={beta}"
        eq = eqs[0]
        assert eq["V"] == pytest.approx(4.7025, abs=5e-3)
        assert beta * eq["C"] == pytest.approx(8.008, abs=5e-3)
        assert classify_regime(p) == "zombie", (
            f"beta={beta} mislabeled as {classify_regime(p)!r} — "
            f"fixed-window bug regression"
        )


def test_trapping_capital_bound_and_carrying_capacity_are_distinct():
    """M2 regression: C_trap (102.67) ≠ C*_max (44.99), and only C_trap traps.

    The rectangle [0, Vmax] × [0, C*_max] used in earlier drafts LEAKS: at
    C = C*_max and V = Vmax, dC/dt > 0. The corrected ceiling C_trap satisfies
    dC/dt < 0 for all V ∈ [0, Vmax] at any C > C_trap. Both facts are checked
    directly on the RHS, independent of the certificate function.
    """
    from dlvt.model import dlvt_system
    from dlvt.analysis import trapping_capital_bound

    p = make_params()
    c_trap = trapping_capital_bound(p)
    cc = carrying_capacity(p)

    assert c_trap == pytest.approx(102.667, abs=1e-2)
    assert cc == pytest.approx(44.99, abs=0.05)
    assert c_trap > cc

    # The old rectangle leaks at its top edge:
    _, dC_leak = dlvt_system(0.0, [p["Vmax"], cc], p)
    assert dC_leak > 0.0, "old C*_max ceiling should leak (dC/dt > 0)"

    # The corrected ceiling traps for every V in [0, Vmax]:
    for V in np.linspace(0.0, p["Vmax"], 21):
        _, dC = dlvt_system(0.0, [V, 1.001 * c_trap], p)
        assert dC < 0.0, f"C_trap ceiling fails to trap at V={V}"
