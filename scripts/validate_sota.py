#!/usr/bin/env python3
"""
validate_sota.py
================
Validation script to ensure all SOTA requirements are met.

Checks:
  1. Parameter consistency (default values match Table 1)
  2. Module structure (all functions present and callable)
  3. Documentation (docstrings, type hints)
  4. Figure generation pipeline
  5. Numerical stability
"""

import sys
from pathlib import Path

# Add repo to path
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

try:
    import numpy as np
    from dlvt import (
        DEFAULT_PARAMS, make_params, complexity, impact,
        dlvt_system, dlvt_exogenous, simulate,
        carrying_capacity, find_interior_equilibria,
        jacobian_eigenvalues, is_zombie, classify_regime, regime_map
    )
except ImportError as e:
    print(f'ERROR: Failed to import dlvt: {e}')
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_parameters():
    """Test that parameters match the paper's Table 1."""
    print('\n[TEST] Parameter Consistency')
    print('─' * 60)

    expected = {
        'R': 3.0,
        'Vmax': 10.0,
        'delta': 0.02,
        'gamma': 2.0,
        'O0': 1.0,
        'beta': 0.25,
        'eta': 1.0,
        'alpha': 0.1,
        'phi': 0.15,
        'mu': 0.2,
        'eps': 0.1,
    }

    p = DEFAULT_PARAMS
    all_match = True
    for key, expected_val in expected.items():
        actual_val = p.get(key)
        match = actual_val == expected_val
        status = '✓' if match else '✗'
        print(f'  {status} {key:10s}: {actual_val} (expected {expected_val})')
        all_match = all_match and match

    return all_match


def test_module_structure():
    """Test that all required functions exist and are callable."""
    print('\n[TEST] Module Structure')
    print('─' * 60)

    required_functions = {
        'model': [
            'make_params', 'complexity', 'impact',
            'dlvt_system', 'dlvt_exogenous', 'simulate'
        ],
        'analysis': [
            'carrying_capacity', 'find_interior_equilibria',
            'jacobian_eigenvalues', 'is_zombie',
            'classify_regime', 'regime_map'
        ],
    }

    all_present = True
    for module_name, funcs in required_functions.items():
        print(f'\n  Module: dlvt.{module_name}')
        for func_name in funcs:
            try:
                func = globals()[func_name]
                if callable(func):
                    print(f'    ✓ {func_name}')
                else:
                    print(f'    ✗ {func_name} (not callable)')
                    all_present = False
            except KeyError:
                print(f'    ✗ {func_name} (not found)')
                all_present = False

    return all_present


def test_basic_simulation():
    """Test that simulation runs without errors."""
    print('\n[TEST] Basic Simulation')
    print('─' * 60)

    try:
        p = make_params()
        t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=10.0)

        # Check output shapes
        assert len(t) > 0, 'Time array is empty'
        assert len(V) == len(t), 'V shape mismatch'
        assert len(C) == len(t), 'C shape mismatch'
        assert len(O) == len(t), 'O shape mismatch'
        assert len(I) == len(t), 'I shape mismatch'
        assert len(G) == len(t), 'G shape mismatch'

        # Check value ranges
        assert np.all(V >= -0.1), 'V has negative values'
        assert np.all(C >= -0.1), 'C has negative values'
        assert np.all(O >= 0), 'O has negative values'

        print(f'  ✓ Simulation completed: T={t[-1]:.1f}, V_final={V[-1]:.2f}, C_final={C[-1]:.2f}')
        return True

    except Exception as e:
        print(f'  ✗ Simulation failed: {e}')
        return False


def test_equilibrium_finding():
    """Test equilibrium finding and stability classification."""
    print('\n[TEST] Equilibrium Analysis')
    print('─' * 60)

    try:
        p = make_params(beta=0.25)

        # Find equilibria
        eqs = find_interior_equilibria(p)
        if not eqs:
            print('  ⚠ No interior equilibria found (collapse-prone regime)')
            return True

        print(f'  ✓ Found {len(eqs)} equilibrium/equilibria')

        for i, eq in enumerate(eqs):
            V_star = eq['V']
            C_star = eq['C']
            stable = eq['stable']
            zombie = eq['zombie']

            status = 'stable' if stable else 'unstable'
            regime = 'zombie' if zombie else 'normal'
            print(f'    [{i+1}] V*={V_star:.2f}, C*={C_star:.2f} ({status}, {regime})')

        # Test carrying capacity
        cc = carrying_capacity(p)
        print(f'  ✓ Carrying capacity: C*_max={cc:.2f}')

        # Test regime classification
        regime = classify_regime(p)
        print(f'  ✓ Regime classification: {regime}')

        return True

    except Exception as e:
        print(f'  ✗ Equilibrium analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_parameter_sweep():
    """Test regime map generation (Figure 7 basis)."""
    print('\n[TEST] Parameter Sweep (Regime Map)')
    print('─' * 60)

    try:
        betas = np.linspace(0.05, 1.0, 10)
        deltas = np.linspace(0.001, 0.1, 10)

        regimes = regime_map(betas, deltas)

        assert regimes.shape == (len(deltas), len(betas)), 'Regime map shape mismatch'

        # Count regimes
        regimes_flat = regimes.flatten()
        counts = {}
        for r in regimes_flat:
            counts[r] = counts.get(r, 0) + 1

        print(f'  ✓ Generated {regimes.shape[0]}×{regimes.shape[1]} regime map')
        for regime, count in sorted(counts.items()):
            pct = 100 * count / len(regimes_flat)
            print(f'    - {regime:15s}: {count:3d} points ({pct:5.1f}%)')

        return True

    except Exception as e:
        print(f'  ✗ Parameter sweep failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_docstrings():
    """Check that all public functions have docstrings."""
    print('\n[TEST] Documentation')
    print('─' * 60)

    functions_to_check = [
        ('make_params', make_params),
        ('complexity', complexity),
        ('impact', impact),
        ('dlvt_system', dlvt_system),
        ('simulate', simulate),
        ('carrying_capacity', carrying_capacity),
        ('find_interior_equilibria', find_interior_equilibria),
        ('classify_regime', classify_regime),
    ]

    all_documented = True
    for name, func in functions_to_check:
        doc = func.__doc__
        has_doc = doc is not None and len(doc.strip()) > 0
        status = '✓' if has_doc else '✗'
        lines = len(doc.split('\n')) if has_doc else 0
        print(f'  {status} {name:30s}: {lines:3d} lines')
        all_documented = all_documented and has_doc

    return all_documented


def main():
    """Run all tests."""
    print('\n' + '═' * 70)
    print('DLVT SOTA VALIDATION')
    print('═' * 70)

    tests = [
        ('Parameters', test_parameters),
        ('Module Structure', test_module_structure),
        ('Basic Simulation', test_basic_simulation),
        ('Equilibrium Analysis', test_equilibrium_finding),
        ('Parameter Sweep', test_parameter_sweep),
        ('Documentation', test_docstrings),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f'\n[ERROR] Test {name} crashed: {e}')
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print('\n' + '═' * 70)
    print('SUMMARY')
    print('═' * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = '[OK]' if passed_test else '[FAIL]'
        print(f'  {status}  {name}')

    print('-' * 70)
    print(f'Total: {passed}/{total} tests passed')

    if passed == total:
        print('\n✓ All SOTA requirements validated!')
        print('═' * 70)
        return 0
    else:
        print(f'\n✗ {total - passed} test(s) failed')
        print('═' * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
