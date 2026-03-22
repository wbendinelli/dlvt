import pytest
import numpy as np
from dlvt.model import make_params, complexity, impact, dlvt_system

@pytest.fixture
def base_params():
    return make_params()

def test_complexity_calculation(base_params):
    # O = O_0 + beta * C^eta
    # Default: O_0=1.0, beta=0.25, eta=1.0
    C = 10.0
    expected = 1.0 + 0.25 * (10.0 ** 1.0)
    assert complexity(C, base_params) == expected

def test_impact_calculation(base_params):
    # I = (C * V) / (1 + phi * O)
    # Default phi=0.15
    V = 8.0
    C = 5.0
    O_val = complexity(C, base_params) # 1.0 + 0.25*5 = 2.25
    expected = (5.0 * 8.0) / (1.0 + 0.15 * 2.25)
    assert np.isclose(impact(V, C, O_val, base_params), expected)

def test_dlvt_system_derivatives(base_params):
    # Check that the system computes the correct finite derivatives
    # sys(t, [V, C], p)
    V, C = 8.0, 5.0
    # Expected dV
    O_val = complexity(C, base_params)
    drain = base_params['delta'] * (O_val**base_params['gamma']) * V / (V + base_params['eps'])
    expected_dV = base_params['R'] * (1 - V/base_params['Vmax']) - drain
    
    # Expected dC
    imp = impact(V, C, O_val, base_params)
    expected_dC = base_params['alpha'] * imp - base_params['mu'] * C
    
    dV, dC = dlvt_system(0, [V, C], base_params)
    assert np.isclose(dV, expected_dV)
    assert np.isclose(dC, expected_dC)
