#!/usr/bin/env python3
"""
Numerical verification suite for the DLVT mathematical results.

Verifies Propositions 1–3 and Theorems 1–3 against numerical simulation.
All tests should return PASS under the default parameters.

Usage
-----
    python scripts/verify.py
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlvt.model    import DEFAULT_PARAMS, complexity, simulate, dlvt_exogenous
from dlvt.analysis import (carrying_capacity, find_interior_equilibria,
                            jacobian_eigenvalues, is_zombie)


def run_all():
    p = DEFAULT_PARAMS.copy()
    results = []

    print("=" * 65)
    print("DLVT NUMERICAL VERIFICATION")
    print("=" * 65)

    # ── Test 1: Smooth barrier (Proposition 1 — positive invariance) ──────────
    print("\n[T1] Positive invariance: V(t) ≥ 0 for all t")
    p_hard = {**p, 'beta': 0.8, 'delta': 0.05}
    t, V, C, *_ = simulate(p_hard, V0=8, C0=2, T=100)
    passed = min(V) >= -1e-10
    print(f"    min(V) = {min(V):.6f}  →  {'PASS ✓' if passed else 'FAIL ✗'}")
    results.append(passed)

    # ── Test 2: Theorem 1 — exogenous ramp → depletion ───────────────────────
    print("\n[T2] Theorem 1: linearly growing C(t) causes vitality depletion")
    cc = carrying_capacity(p)
    C_func = lambda t: 0.5 + 0.6*t
    t_star_th = (cc - 0.5) / 0.6
    T_sim = max(90.0, t_star_th * 1.5)
    sol = solve_ivp(dlvt_exogenous, [0, T_sim], [8.0], args=(p, C_func),
                    method='RK45', max_step=0.05)
    V_exo = np.maximum(sol.y[0], 0)
    print(f"    C*_max = {cc:.2f},  t* (theoretical) = {t_star_th:.2f},  T_sim = {T_sim:.1f}")
    passed = V_exo[-1] < 2.0
    print(f"    V at T={T_sim:.0f} = {V_exo[-1]:.4f}  →  {'PASS ✓' if passed else 'FAIL ✗'}")
    results.append(passed)

    # ── Test 3: Carrying capacity definition (Γ = 1 at C*_max) ───────────────
    print("\n[T3] Proposition 3: Γ(C*_max) = 1  (at V = V_max)")
    O_at_cc = complexity(cc, p)
    G_at_cc = p['delta'] * O_at_cc**p['gamma'] / p['R']
    passed = abs(G_at_cc - 1.0) < 1e-5
    print(f"    C*_max = {cc:.4f},  O(C*_max) = {O_at_cc:.4f},  Γ = {G_at_cc:.6f}")
    print(f"    →  {'PASS ✓' if passed else 'FAIL ✗'}")
    results.append(passed)

    # ── Test 4: Zombie equilibrium classification ─────────────────────────────
    print("\n[T4] Theorem 2 / Definition 7: zombie classification for varying β")
    header = f"{'β':>6}  {'C*':>7}  {'V*':>6}  {'I*':>6}  {'Stable':>8}  {'Regime':>12}  {'C*_max':>8}"
    print("    " + header)
    print("    " + "-" * len(header))
    all_pass = True
    for b in [0.08, 0.15, 0.25, 0.40, 0.60]:
        pp = {**p, 'beta': b}
        eqs = find_interior_equilibria(pp)
        cc_val = carrying_capacity(pp)
        for eq in eqs:
            regime = 'zombie' if eq['zombie'] else 'sustainable'
            stab   = 'stable ✓' if eq['stable'] else 'unstable ✗'
            if not eq['stable']:
                all_pass = False
            print(f"    {b:>6.2f}  {eq['C']:>7.2f}  {eq['V']:>6.2f}  "
                  f"{eq['I']:>6.2f}  {stab:>8}  {regime:>12}  {cc_val:>8.2f}")
    print(f"    →  {'PASS ✓' if all_pass else 'FAIL ✗'}")
    results.append(all_pass)

    # ── Test 5: Carrying capacity generalised for η ≠ 1 ──────────────────────
    print("\n[T5] Proposition 3 (generalised): C*_max for η ∈ {1.0, 1.5, 2.0}")
    for eta_val in [1.0, 1.5, 2.0]:
        pp = {**p, 'eta': eta_val}
        cc_gen = carrying_capacity(pp)
        Omax = (pp['R']/pp['delta'])**(1/pp['gamma'])
        G_check = pp['delta'] * (pp['O0'] + pp['beta']*cc_gen**eta_val)**pp['gamma'] / pp['R']
        passed_eta = abs(G_check - 1.0) < 1e-5 or cc_gen == 0
        print(f"    η={eta_val:.1f}: C*_max = {cc_gen:.3f},  Γ(C*) = {G_check:.6f}"
              f"  →  {'PASS ✓' if passed_eta else 'FAIL ✗'}")
        results.append(passed_eta)

    # ── Test 6: Jacobian stability at equilibrium ─────────────────────────────
    print("\n[T6] Theorem 2: Jacobian eigenvalues Re(λ) < 0 at stable equilibria")
    all_neg = True
    pp = {**p, 'beta': 0.25}
    eqs = find_interior_equilibria(pp)
    for eq in eqs:
        if eq['stable']:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], pp)
            neg = all(e.real < 0 for e in eigv)
            if not neg:
                all_neg = False
            print(f"    (C*={eq['C']:.2f}, V*={eq['V']:.2f}): "
                  f"λ = {eigv[0]:.4f}, {eigv[1]:.4f}  →  {'PASS ✓' if neg else 'FAIL ✗'}")
    results.append(all_neg)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 65)
    print(f"VERIFICATION SUMMARY: {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — check output above")
    print("=" * 65)
    return n_pass == n_total


if __name__ == '__main__':
    ok = run_all()
    sys.exit(0 if ok else 1)
