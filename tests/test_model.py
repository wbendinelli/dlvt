"""
Tests for dlvt.model: core ODE, parameters, simulation invariants.

These tests pin the Python implementation to the final theoretical
formulation in `paper/chapters/03_mathematical_formulation.tex`.
They are meant to fail loudly if the model or its parameters drift
away from the paper's equations.

References
----------
Eq. (3.2)  Impact:              I = C * V / (1 + phi * O)
Eq. (3.5)  Vitality dynamics:   dV/dt = R (1 - V/Vmax)
                                       - delta * O^gamma * V / (V + eps)
Eq. (3.6)  Capital dynamics:    dC/dt = alpha * I - mu * C
Eq. (3.7)  Complexity coupling: O     = O0 + beta * C^eta
"""

import math

import numpy as np
import pytest

from dlvt.model import (
    DEFAULT_PARAMS,
    complexity,
    dlvt_system,
    impact,
    make_params,
    simulate,
)


# ────────────────────────────────────────────────────────────────────────────
# Parameter block — pinned to paper Table 1 (baseline calibration)
# ────────────────────────────────────────────────────────────────────────────


PAPER_BASELINE = dict(
    R=3.0,
    Vmax=10.0,
    delta=0.02,
    gamma=2.0,
    O0=1.0,
    beta=0.25,
    eta=1.0,
    alpha=0.1,
    phi=0.15,
    mu=0.2,
    eps=0.1,
)


def test_default_params_match_paper_baseline():
    """DEFAULT_PARAMS must match the baseline calibration in §3.7 / Table 1."""
    for key, value in PAPER_BASELINE.items():
        assert key in DEFAULT_PARAMS, f"missing parameter {key}"
        assert DEFAULT_PARAMS[key] == pytest.approx(value), (
            f"{key}: expected {value}, got {DEFAULT_PARAMS[key]}"
        )


def test_make_params_overrides():
    """make_params must return a copy and apply overrides."""
    p = make_params(beta=0.5, R=4.0)
    assert p["beta"] == 0.5
    assert p["R"] == 4.0
    # original dict untouched
    assert DEFAULT_PARAMS["beta"] == 0.25
    assert DEFAULT_PARAMS["R"] == 3.0


# ────────────────────────────────────────────────────────────────────────────
# Functional-form identities — Eq. (3.2), Eq. (3.7)
# ────────────────────────────────────────────────────────────────────────────


def test_complexity_formula():
    """O(C) = O0 + beta * C^eta."""
    p = make_params()
    for C in [0.0, 1.0, 5.0, 25.0, 100.0]:
        expected = p["O0"] + p["beta"] * (C ** p["eta"])
        assert complexity(C, p) == pytest.approx(expected)


def test_complexity_vectorised():
    """complexity() must accept numpy arrays."""
    p = make_params()
    C = np.array([0.0, 5.0, 25.0, 100.0])
    out = complexity(C, p)
    assert out.shape == C.shape
    np.testing.assert_allclose(
        out, p["O0"] + p["beta"] * C ** p["eta"]
    )


def test_complexity_nonnegative_clamp():
    """Negative C should be clamped to 0 inside complexity()."""
    p = make_params()
    assert complexity(-5.0, p) == pytest.approx(p["O0"])


def test_impact_formula_matches_paper():
    """I = C * V / (1 + phi * O), Eq. (3.2)."""
    p = make_params()
    V, C = 7.0, 20.0
    O = complexity(C, p)
    expected = C * V / (1.0 + p["phi"] * O)
    assert impact(V, C, O, p) == pytest.approx(expected)


def test_impact_vanishes_at_zero_vitality():
    """When V = 0, I must be exactly 0 (energy-gating)."""
    p = make_params()
    assert impact(0.0, 50.0, 10.0, p) == 0.0


def test_impact_vanishes_in_infinite_complexity_limit():
    """As O → ∞, I → 0 (complexity-friction limit)."""
    p = make_params()
    assert impact(10.0, 20.0, 1e9, p) < 1e-5


# ────────────────────────────────────────────────────────────────────────────
# ODE right-hand side — Eq. (3.5), Eq. (3.6)
# ────────────────────────────────────────────────────────────────────────────


def test_dlvt_system_matches_equations():
    """Spot-check RHS against a hand-computed value."""
    p = make_params()
    V, C = 8.0, 10.0
    O = complexity(C, p)
    recovery = p["R"] * (1.0 - V / p["Vmax"])
    drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
    I = impact(V, C, O, p)
    expected_dV = recovery - drain
    expected_dC = p["alpha"] * I - p["mu"] * C
    dV, dC = dlvt_system(0.0, [V, C], p)
    assert dV == pytest.approx(expected_dV, rel=1e-12)
    assert dC == pytest.approx(expected_dC, rel=1e-12)


def test_dv_dt_positive_at_V_zero():
    """Positive invariance (Appendix A1).

    At V = 0 the smooth barrier V/(V+eps) vanishes and dV/dt = R > 0.
    This proves V(t) ≥ 0 for all t ≥ 0.
    """
    p = make_params()
    for C in [0.0, 5.0, 50.0]:
        dV, _ = dlvt_system(0.0, [0.0, C], p)
        assert dV == pytest.approx(p["R"], rel=1e-12), (
            f"dV/dt at V=0, C={C} should equal R={p['R']}, got {dV}"
        )


def test_dc_dt_zero_at_C_zero():
    """At C = 0, dC/dt = 0 (capital cannot become negative)."""
    p = make_params()
    _, dC = dlvt_system(0.0, [8.0, 0.0], p)
    assert dC == 0.0


# ────────────────────────────────────────────────────────────────────────────
# Simulation invariants
# ────────────────────────────────────────────────────────────────────────────


def test_simulate_returns_nonnegative_states():
    """V(t) ≥ 0 and C(t) ≥ 0 for all integration points."""
    p = make_params()
    t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=200.0)
    assert np.all(V >= 0.0)
    assert np.all(C >= 0.0)
    assert np.all(O >= p["O0"] - 1e-12)
    assert np.all(I >= 0.0)


def test_simulate_vitality_stays_below_ceiling():
    """V(t) cannot exceed V_max under the logistic recovery law."""
    p = make_params()
    _, V, *_ = simulate(p, V0=p["Vmax"], C0=1.0, T=50.0)
    # Small numerical tolerance only.
    assert V.max() <= p["Vmax"] + 1e-6


def test_simulate_baseline_reproduces_paper_zombie_numerics():
    """At T = 400 with baseline params, V* ≈ 4.70 and C* ≈ 32.0.

    These are the numerics reported in Figure 3 (phase portrait) and the
    reference values for the zombie equilibrium in §3.8. If these drift,
    the code and the paper are no longer in sync.
    """
    p = make_params()
    _, V, C, *_ = simulate(p, V0=8.0, C0=5.0, T=400.0, max_step=0.1)
    assert V[-1] == pytest.approx(4.70, abs=0.05), (
        f"V* drift: expected ≈4.70, got {V[-1]:.3f}"
    )
    assert C[-1] == pytest.approx(32.0, abs=0.3), (
        f"C* drift: expected ≈32.0, got {C[-1]:.3f}"
    )


def test_simulate_sustainable_regime_above_strategic_threshold():
    """Under low-coupling / high-recovery parameters, V* stays above V_strat.

    This corresponds to the 'Sustainable leadership' panel in Figure 2
    (delta=0.008, beta=0.15).
    """
    p = make_params(delta=0.008, beta=0.15)
    _, V, *_ = simulate(p, V0=8.0, C0=5.0, T=400.0, max_step=0.1)
    assert V[-1] > 5.0, (
        f"Sustainable regime should yield V* > V_strategic=5.0, got {V[-1]:.3f}"
    )
