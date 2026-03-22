# Quick Start Guide — DLVT

Get started with the Dynamic Leadership Vitality Theory model in 5 minutes.

## Installation

```bash
pip install -e .
```

Or just install dependencies:

```bash
pip install numpy scipy matplotlib
```

## 1. Run a Simulation

```python
from dlvt import make_params, simulate
import matplotlib.pyplot as plt

# Create parameters (defaults from Table 1 of paper)
p = make_params()

# Simulate a leader's trajectory over 120 time units
t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=120)

# Plot vitality and capital
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(t, V, 'b-', label='Vitality V(t)')
ax1.axhline(p['Vmax']/2, color='r', linestyle='--', alpha=0.5, label='Strategic threshold')
ax1.set_ylabel('V(t)')
ax1.legend()

ax2.plot(t, C, 'g-', label='Career Capital C(t)')
ax2.set_xlabel('Time')
ax2.set_ylabel('C(t)')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 2. Find Equilibrium Points

```python
from dlvt import make_params
from dlvt.analysis import find_interior_equilibria, carrying_capacity

p = make_params()

# Find all equilibria
equilibria = find_interior_equilibria(p)
print(f"Found {len(equilibria)} equilibrium point(s):")
for i, eq in enumerate(equilibria):
    print(f"  Eq {i+1}: V* = {eq['V']:.3f}, C* = {eq['C']:.3f}, stable = {eq['stable']}")

# Maximum sustainable capital
C_max = carrying_capacity(p)
print(f"\nCarrying capacity: C*_max = {C_max:.2f}")
```

## 3. Compare Parameters

```python
from dlvt import make_params, simulate

# Default parameters
p_default = make_params()

# Higher capital-complexity coupling (more burden per unit capital)
p_high_beta = make_params(beta=0.5)

# Simulate both
t1, V1, C1, *_ = simulate(p_default, T=120)
t2, V2, C2, *_ = simulate(p_high_beta, T=120)

# Compare final vitality
print(f"Beta=0.25: V(120) = {V1[-1]:.2f}")
print(f"Beta=0.50: V(120) = {V2[-1]:.2f}")
```

## 4. Generate All Figures

```bash
python3 scripts/run_all_figures.py
```

Figures 1–10 are saved to `figures/` in both PNG and PDF formats.

Selective figures:

```bash
# Core results only (Figures 1–7)
python3 scripts/run_all_figures.py --fig 1-7

# Extended analysis (Figures 8–10)
python3 scripts/run_all_figures.py --fig 8-10

# Specific figures
python3 scripts/run_all_figures.py --fig 1,3,5
```

## 5. Classify Leadership Regime

```python
from dlvt import make_params
from dlvt.analysis import classify_regime, is_zombie

p = make_params()
regime = classify_regime(p)
print(f"Regime: {regime}")  # 'sustainable', 'zombie', or 'collapse'

# Check if equilibrium is "zombie" (exhausted)
p_zombie = make_params(beta=0.8)
regime2 = classify_regime(p_zombie)
print(f"High-complexity regime: {regime2}")
```

## Key Concepts

| Symbol | Meaning |
|--------|---------|
| **V(t)** | Subjective vitality (energy available) — range [0, V_max] |
| **C(t)** | Career capital (skills, status, relational capital) |
| **O(t)** | Organisational complexity driven by capital growth |
| **I(t)** | Leadership impact (effectiveness) — suppressed by complexity |
| **Γ(t)** | Depletion ratio (Γ > 1 means net drain; Γ < 1 means recovery) |

## Parameter Defaults (Table 1, Paper)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| R | 3.0 | Vitality recovery rate |
| V_max | 10.0 | Maximum vitality capacity |
| δ | 0.02 | Energetic cost of complexity |
| γ | 2.0 | Complexity exponent in drain |
| O₀ | 1.0 | Baseline irreducible complexity |
| β | 0.25 | Capital-complexity coupling |
| η | 1.0 | Capital scaling exponent |
| α | 0.1 | Capital accumulation rate |
| φ | 0.15 | Impact suppression by complexity |
| μ | 0.2 | Capital depreciation rate |
| ε | 0.1 | Regularization (ε → 0 in theory) |

## The Two-Equation System

The DLVT model integrates two dynamics:

```
Recovery:   dV/dt = R·(1 − V/Vmax)
Drain:      dV/dt − δ·O^γ · V/(V + ε)

Capital:    dC/dt = α·I − μ·C
            where I = C·V / (1 + φ·O)
```

**Recovery** is logistic (saturation). **Drain** is nonlinear in complexity. Capital accumulates through impact (which depends on energy) and decays naturally.

## Three Outcome Regimes

1. **Sustainable**: Equilibrium with high vitality and manageable capital
2. **Zombie**: Chronic exhaustion at low vitality; capital still accumulates but impact is suppressed
3. **Collapse**: No equilibrium; vitality crashes to zero

## Next Steps

- Read `README.md` for full API reference
- Explore `notebooks/` for interactive examples
- See `CONTRIBUTING.md` for how to extend the model
- Consult the paper for theoretical details

## Questions?

- Check existing GitHub issues
- Review docstrings: `help(dlvt.simulate)`
- Open a new issue with your question
