import pytest
import numpy as np
from dlvt.model import make_params
from dlvt.analysis import carrying_capacity, find_interior_equilibria

@pytest.fixture
def base_params():
    return make_params()

def test_carrying_capacity(base_params):
    # C_max = ((R/delta)^(1/gamma) - O_0) / beta ) ^ (1/eta)
    p = base_params
    O_max_gamma = p['R'] / p['delta']
    O_max = O_max_gamma ** (1/p['gamma'])
    
    expected = ((O_max - p['O0']) / p['beta']) ** (1/p['eta'])
    assert np.isclose(carrying_capacity(p), expected)

def test_carrying_capacity_zero():
    # If O_max <= O_0, carrying capacity is 0
    p = make_params(delta=5.0) # very high delta makes O_max small
    assert carrying_capacity(p) == 0.0

def test_find_interior_equilibria(base_params):
    # Find equilibria for sustainable parameters
    p_sust = make_params(delta=0.008, beta=0.15)
    eqs = find_interior_equilibria(p_sust)
    
    assert len(eqs) > 0
    
    for eq in eqs:
        C = eq['C']
        V = eq['V']
        
        # Check numerical V_nullcline
        v_diff = p_sust['R']*(1 - V/p_sust['Vmax']) - p_sust['delta'] * (p_sust['O0'] + p_sust['beta']*C**p_sust['eta'])**p_sust['gamma'] * V / (V + p_sust['eps'])
        assert np.isclose(v_diff, 0.0, atol=1e-3)
