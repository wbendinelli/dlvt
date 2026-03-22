# The Carrying Capacity of Leadership: A Dynamic Systems Model of Executive Sustainability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SSRN](https://img.shields.io/badge/SSRN-Working%20Paper-blue.svg)](https://ssrn.com)

**Author:** William Bendinelli
**Networks:** Leadership, Organizational Behavior, Strategic Management
**JEL Codes:** M12, D23, J24, L23
**Year:** 2026

---

## Overview

This repository contains the Python implementation of the **Dynamic Leadership Vitality Theory (DLVT)** model—a dynamical systems framework for studying executive sustainability under endogenous organizational complexity growth.

The DLVT model integrates two critical mechanisms:

1. **Vitality Recovery**: Leaders experience natural energy renewal, following logistic saturation kinetics.
2. **Complexity-Driven Drain**: Organizational complexity imposes a nonlinear energetic cost that increases with career capital.

As executives accumulate career capital, organizational complexity grows endogenously, ultimately constraining available vitality. The model predicts three distinct outcome regimes:

- **Sustainable**: Leaders reach a stable equilibrium with sufficient energy for strategic work
- **Zombie**: Leaders maintain capital-building activity but with chronically insufficient vitality
- **Collapse-Prone**: Unsustainable trajectories where burnout is inevitable

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dlvt.git
cd dlvt/code

# Install the package
pip install -e .

# Or install dependencies directly
pip install numpy scipy matplotlib
```

**Requirements:**
- Python ≥ 3.8
- numpy ≥ 1.20
- scipy ≥ 1.5
- matplotlib ≥ 3.3

### Basic Usage

```python
from dlvt import make_params, simulate
from dlvt.analysis import find_interior_equilibria, carrying_capacity

# Create parameter set with defaults (C₀=5.0 throughout paper)
p = make_params()

# Simulate from initial state (V₀=8.0, C₀=5.0) over T=120 time units
t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=120)

# V(t)  — subjective vitality (energy available for leadership work)
# C(t)  — career capital (accumulated human/social/relational capital)
# O(t)  — organizational complexity
# I(t)  — leadership impact (energy-gated effectiveness)
# G(t)  — depletion ratio Γ(t) = δ·O^γ / R (Γ > 1 ⟹ net drain)

print(f"Final vitality: V(T) = {V[-1]:.2f}")
print(f"Final capital: C(T) = {C[-1]:.2f}")

# Find all equilibria and classify stability
equilibria = find_interior_equilibria(p)
for eq in equilibria:
    status = "stable" if eq['stable'] else "unstable"
    print(f"Equilibrium: V*={eq['V']:.2f}, C*={eq['C']:.2f}, {status}")

# Maximum sustainable career capital
C_max = carrying_capacity(p)
print(f"Carrying capacity: C*_max = {C_max:.2f}")
```

## The Dynamical System

The DLVT model is governed by two coupled ODEs (Equations 3.3–3.4 in the paper):

$$\frac{dV}{dt} = R \left(1 - \frac{V}{V_{\max}}\right) - \delta \cdot O^{\gamma} \cdot \frac{V}{V + \varepsilon}$$

$$\frac{dC}{dt} = \alpha \cdot I - \mu \cdot C$$

Where the derived quantities are:

- **Organizational Complexity**: $O(t) = O_0 + \beta \cdot C(t)^{\eta}$
- **Leadership Impact**: $I(t) = \frac{C \cdot V}{1 + \phi \cdot O}$ (energy-gated effectiveness)
- **Depletion Ratio**: $\Gamma(t) = \frac{\delta \cdot O^{\gamma}}{R}$ (net drain indicator)

**Interpretation:**
- The first equation balances vitality **recovery** (logistic saturation with capacity $V_{\max}$) against **drain** (complexity burden with smooth barrier $\varepsilon$).
- The second equation models career capital accumulation through impact feedback minus depreciation.
- The smooth barrier $\frac{V}{V + \varepsilon}$ ensures positive invariance; numerics use $\varepsilon = 0.1$ (~2% shift from the $\varepsilon \to 0$ analytical limit).

## Parameter Table

All parameters from **Table 1** of the paper. Default values use $C_0 = 5.0$ throughout.

| Symbol | Python Key | Default | Units | Description |
|:------:|:----------:|:-------:|:-----:|:------------|
| $R$ | `R` | 3.0 | [1/time] | Vitality recovery rate |
| $V_{\max}$ | `Vmax` | 10.0 | [energy] | Maximum vitality capacity |
| $\delta$ | `delta` | 0.02 | [cost/complexity^γ] | Energetic cost coefficient |
| $\gamma$ | `gamma` | 2.0 | [—] | Complexity exponent in drain term |
| $O_0$ | `O0` | 1.0 | [complexity] | Baseline (irreducible) organizational complexity |
| $\beta$ | `beta` | 0.25 | [complexity/capital^η] | Capital-complexity coupling strength |
| $\eta$ | `eta` | 1.0 | [—] | Capital scaling exponent in complexity |
| $\alpha$ | `alpha` | 0.1 | [capital/(impact·time)] | Capital accumulation rate |
| $\phi$ | `phi` | 0.15 | [1/complexity] | Complexity-impact suppression coefficient |
| $\mu$ | `mu` | 0.2 | [1/time] | Capital depreciation rate |
| $\varepsilon$ | `eps` | 0.1 | [energy] | Smooth barrier regularisation parameter |

## API Reference

### `dlvt.model` — Core ODE System

```python
from dlvt import make_params, simulate, complexity, impact

# Create parameter dictionary
p = make_params(beta=0.5, R=4.0)  # Override specific parameters

# Core functions
O = complexity(C, p)              # Organizational complexity
I = impact(V, C, O, p)            # Leadership impact
t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=120)
```

**Functions:**
- `make_params(**overrides)` — Create parameter dict with keyword overrides
- `complexity(C, p)` — Calculate $O = O_0 + \beta \cdot C^{\eta}$
- `impact(V, C, O, p)` — Calculate $I = \frac{C \cdot V}{1 + \phi \cdot O}$
- `dlvt_system(t, y, p)` — ODE right-hand side for `solve_ivp`
- `dlvt_exogenous(t, y, p, C_func)` — Variant with exogenous career capital
- `simulate(p, V0=8.0, C0=5.0, T=120.0, max_step=0.05)` — Integrate ODE system

### `dlvt.analysis` — Equilibrium Theory & Stability

```python
from dlvt.analysis import (
    find_interior_equilibria, carrying_capacity,
    jacobian_eigenvalues, is_zombie, classify_regime, regime_map
)

# Analytical results
C_max = carrying_capacity(p)            # Proposition 3

# Find all equilibria
equilibria = find_interior_equilibria(p)
for eq in equilibria:
    print(f"V*={eq['V']:.2f}, C*={eq['C']:.2f}, stable={eq['stable']}")

# Regime classification
regime = classify_regime(p)              # 'sustainable', 'zombie', 'collapse-prone'
is_zomb = is_zombie(5.0, p)              # V* < V_strategic?

# Phase space map
betas = np.linspace(0.01, 1.0, 50)
deltas = np.linspace(0.001, 0.1, 50)
regimes = regime_map(betas, deltas)      # Fig 7 data
```

**Functions:**
- `find_interior_equilibria(p, C_max=120.0, n_scan=8000)` — Find all $(V^*, C^*) > 0$ (Theorem 2)
- `carrying_capacity(p)` — Maximum sustainable capital $C^*_{\max}$ (Proposition 3)
- `jacobian_eigenvalues(V, C, p)` — Eigenvalues & stability classification
- `is_zombie(V, p)` — Check if $V^* < V_{\text{strategic}}$ (Definition 7)
- `classify_regime(p)` — Classify as 'sustainable', 'zombie', or 'collapse-prone'
- `regime_map(beta_range, delta_range, base_params=None)` — Generate $(β, δ)$ regime grid

### `dlvt.figures` — Publication Figure Generation

```python
from dlvt.figures import fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Generate individual figures
fig1(output_dir='figures/')   # Temporal evolution
fig2(output_dir='figures/')   # Three outcome scenarios
fig3(output_dir='figures/')   # Phase portrait
fig4(output_dir='figures/')   # Bifurcation diagrams
fig5(output_dir='figures/')   # Impact comparison (DLVT vs Becker)
fig6(output_dir='figures/')   # Carrying capacity heatmap
fig7(output_dir='figures/')   # Regime map
```

**Figures 1–7:**
- **Fig 1:** Temporal evolution of $V(t)$, $C(t)$, $O(t)$, $\Gamma(t)$ over time
- **Fig 2:** Three outcome scenarios (sustainable, zombie, collapse)
- **Fig 3:** Phase portrait with V-nullcline, C-nullcline, trajectories, and equilibria
- **Fig 4:** Bifurcation diagrams: $C^*$ vs $\beta$; $C^*$ vs $R$; $V^*$ vs $\beta$
- **Fig 5:** Leadership impact comparison: DLVT vs Human Capital Theory (Becker)
- **Fig 6:** Carrying capacity heatmap: $C^*_{\max}(\beta, R)$ parameter space
- **Fig 7:** Leadership regime map in $(\beta, \delta)$ parameter space

## Reproducing All Paper Figures

All 10 publication figures can be regenerated with a single command:

```bash
python3 scripts/run_all_figures.py
```

**Output:** All figures saved to `figures/` in both PDF (vector) and PNG (300 dpi raster) formats.

### Selective Figure Generation

```bash
# Core results only (Figures 1–7)
python3 scripts/run_all_figures.py --fig 1-7

# Extended analysis only (Figures 8–10)
python3 scripts/run_all_figures.py --fig 8-10

# Specific figures
python3 scripts/run_all_figures.py --fig 1,3,5,7
```

### Extended Figures (8–10)

**Figure 8: Bifurcation Analysis with Hysteresis Detection**
```bash
python3 scripts/fig8_bifurcation_hysteresis.py
```
Parameter scan over $\beta$ with hysteresis detection for bistability windows.

**Figure 9: Structural Robustness Analysis**
```bash
python3 scripts/fig9_robustness.py
```
Sensitivity analysis: equilibrium variation under alternative $\gamma$ and $\eta$ values.

**Figure 10: Intervention Comparison**
```bash
python3 scripts/fig10_intervention_comparison.py
```
Recovery policies: comparing $\beta$ reduction (complexity reduction) vs $R$ increase (support) interventions.

## Project Structure

```
code/
├── README.md                              # This file
├── LICENSE                                # MIT license
├── setup.py                               # Package configuration
├── requirements.txt                       # Dependencies
├── references.bib                         # Full bibliography
│
├── dlvt/                                  # Main package
│   ├── __init__.py                       # Public API
│   ├── model.py                          # ODE system & numerical integration
│   ├── analysis.py                       # Equilibrium theory & stability
│   └── figures.py                        # Publication figure generation
│
├── scripts/                               # Figure generation scripts
│   ├── run_all_figures.py                # Master orchestration script
│   ├── fig8_bifurcation_hysteresis.py    # Figure 8: hysteresis
│   ├── fig9_robustness.py                # Figure 9: robustness
│   └── fig10_intervention_comparison.py  # Figure 10: interventions
│
├── figures/                               # Generated output (after running scripts/)
│   ├── fig1_temporal_evolution.{pdf,png}
│   ├── fig2_three_outcomes.{pdf,png}
│   ├── ... (and so on)
│
└── notebooks/                             # Jupyter notebooks for exploration (optional)
    └── example_analysis.ipynb
```

## Design Principles

1. **Reproducibility:** All figures are fully reproducible from code. Exact numerical results match the paper.
2. **Clarity:** Extensive docstrings follow NumPy documentation style; all equations referenced to paper sections.
3. **Modularity:** Core model (`model.py`) is independent of analysis and visualization layers.
4. **Stability:** Equilibrium finding uses robust bracketing and multi-pass scanning to catch all fixed points.
5. **Accessibility:** All default parameters match Table 1; easy parameter overrides for scenario analysis.

## Theoretical Results

The paper establishes the following key results (implemented in this code):

**Theorem 1 (Existence):** For any parameter set $(R, \delta, \gamma, \ldots) \in \mathbb{R}^{11}_{>0}$, the DLVT system possesses at least one interior fixed point.

**Theorem 2 (Stability):** All interior fixed points can be classified as stable or saddle using the Jacobian eigenvalue test at $(V^*, C^*)$.

**Proposition 1 (V-Nullcline):** The vitality recovery curve is strictly decreasing in $C$.

**Proposition 2 (Codimension-1 Bifurcation):** Varying $\beta$ generically produces saddle-node bifurcations; hysteresis can arise for certain parameter regions.

**Proposition 3 (Carrying Capacity):** The maximum sustainable capital is
$$C^*_{\max} = \left( \left( \frac{R}{\delta} \right)^{1/\gamma} - O_0 \right)^{1/\eta} / \beta$$

## Citation

If you use this code or data in your research, please cite:

```bibtex
@techreport{Bendinelli2026DLVT,
  title={The Carrying Capacity of Leadership: A Dynamic Systems Model of Executive Sustainability},
  author={Bendinelli, William},
  institution={SSRN},
  type={Working Paper},
  year={2026},
  url={https://ssrn.com/}
}
```

## References

The full bibliography is in `references.bib`, containing foundational works on:
- Human capital theory (Becker)
- Organizational burnout (Maslach, Schaufeli)
- Complexity theory (Simon, Nelson & Winter)
- Leadership and decision-making
- Dynamical systems and bifurcation theory

See the SSRN preprint for the complete paper with full citations.

## License

MIT License — see `LICENSE` file for details.

Copyright © 2026 William Bendinelli

## Contributing

Issues, feature requests, and pull requests are welcome. Please ensure:
- Code follows NumPy/SciPy conventions
- All new functions include docstrings with parameter descriptions and examples
- Equations are referenced to paper section numbers
- New figures are generated reproducibly with `scripts/run_all_figures.py`

## Questions & Support

For questions about the model, math, or code:
- Open an issue on GitHub
- Check the docstrings: `help(dlvt.analyze.find_interior_equilibria)` etc.
- See `scripts/run_all_figures.py` for working examples

---

**Paper Source:** The complete paper is available on SSRN.
