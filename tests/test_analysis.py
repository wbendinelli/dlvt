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


def test_linear_drain_gamma1_escapes_zombie_regime():
    """Robustness check (§3.10): at gamma=1 the zombie equilibrium disappears.

    This is the 'linear drain' alternative specification that establishes
    nonlinear complexity scaling as a *necessary* condition for the theory.
    """
    p = make_params(gamma=1.0)
    eqs = find_interior_equilibria(p, C_max=1000.0, n_scan=12000)
    stable_zombies = [
        e for e in eqs if e.get("zombie") and e.get("stable")
    ]
    assert len(stable_zombies) == 0, (
        "Linear drain should eliminate stable zombie equilibria; "
        "got {}".format(stable_zombies)
    )
