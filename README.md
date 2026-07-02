# Dynamic Leadership Vitality Theory (DLVT) — Python Package

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/dlvt.svg)](https://pypi.org/project/dlvt/)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen.svg)](#running-the-tests)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Author:** William Bendinelli
**Paper:** *Dynamic Leadership Vitality Theory: A Closed-Form Capability-Trap Model of Leader Vitality and Career Capital* (working paper, 2026 — manuscript in the canonical [dynamic-leadership-vitality-theory](https://github.com/wbendinelli/dynamic-leadership-vitality-theory) repository; this package is code-only)
**Target venue:** *Computational and Mathematical Organization Theory* or *Nonlinear Dynamics, Psychology, and Life Sciences*

---

## Overview

This package implements the **Dynamic Leadership Vitality Theory (DLVT)** model — a two-dimensional autonomous dynamical system in which a leader's accumulated career capital endogenously generates the organizational complexity that drains the vitality needed to deploy it. It is a closed-form, individual-level member of the *capability-trap* family (Repenning & Sterman 2002; Rahmandad & Repenning 2016), micro-founded on effort-recovery theory (Meijman & Mulder 1998; Sonnentag) rather than ego depletion.

**The two headline results:**

1. **A single, globally stable low-vitality/high-status equilibrium.** Under the baseline calibration the system has exactly one interior attractor, at $(V^{*} \approx 4.7,\; C^{*} \approx 32)$ — informally, a "zombie-leader" state: authority and reputation retained, energetic bandwidth for strategic work chronically depleted. Unlike the bistable/tipping-point structure of capability-erosion models, here the trap is the *default destination*: uniqueness and absence of cycles are proved **globally** (all positive parameters), and the equilibrium is globally asymptotically stable on $\{C>0\}$ (Bendixson–Dulac + Poincaré–Bendixson on a trapping rectangle).

2. **Intervention asymmetry (scope absorption).** Under the separable power-law drain kernel, equilibrium vitality $V^{*}$ is *invariant* to the capital–complexity coupling $\beta$: scope reduction alone cannot raise it — capital re-expands until $\beta C^{*}$ is restored. Only recovery-side parameters ($R\uparrow$, $\mu\downarrow$, $\alpha\uparrow$) move $V^{*}$. This invariance is a property of the kernel family (it fails under exponential/Hill kernels) — an honest corollary, not a law.

**Calibration caveat (read this before citing the numbers).** The baseline values are *illustrative*, not empirical estimates. The regime label is governed by the ratio $\mu/\alpha$: the equilibrium satisfies $V^{*} = (\mu/\alpha)(1 + \phi O^{*})$, the baseline $\mu/\alpha = 2.0$ sits only ~8% below the flip value $(\mu/\alpha)_{\mathrm{crit}} \approx 2.163$, and across a ±2× log-uniform parameter hypercube roughly half the stable equilibria are "zombie". The threshold $V_{\mathrm{strategic}} = 0.5\,V_{\max}$ is stipulated, and $V^*$ sits just 5.9% below it. See `dlvt.nondimensional` and Figure 11 for the effective-parameter and sensitivity analysis.

**Three outcome regimes** partition the parameter space:

- **Sustainable** — stable equilibrium with $V^{*} > V_{\mathrm{strategic}}$
- **Zombie** (low-vitality/high-status) — stable equilibrium with $V^{*} < V_{\mathrm{strategic}}$; the label is kept in the code API for continuity
- **Collapse-Prone** — no sustainable interior equilibrium

## Quick Start

### Installation

```bash
pip install dlvt
```

**Requirements:** Python ≥ 3.8, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7

### Basic Usage

```python
from dlvt import make_params, simulate
from dlvt.analysis import find_interior_equilibria, carrying_capacity

# Create parameter set (defaults match Table 1 of the paper)
p = make_params()

# Simulate from initial state (V₀=8.0, C₀=5.0) over T=120 time units
t, V, C, O, I, G = simulate(p, V0=8.0, C0=5.0, T=120)
# V — vitality, C — career capital, O — complexity, I — impact, G — depletion ratio

# Find equilibria and classify stability
equilibria = find_interior_equilibria(p)
for eq in equilibria:
    status = "stable" if eq['stable'] else "unstable"
    print(f"V*={eq['V']:.2f}, C*={eq['C']:.2f} ({status})")

# Maximum sustainable career capital (Proposition 3)
C_max = carrying_capacity(p)
print(f"Carrying capacity: C*_max = {C_max:.2f}")
```

### Scenario Analysis

```python
from dlvt import make_params, simulate
from dlvt.analysis import classify_regime

# What happens if we increase recovery support?
p_high_R = make_params(R=5.0)
regime = classify_regime(p_high_R)
print(f"Regime with R=5.0: {regime}")

# What if complexity coupling is halved?
p_low_beta = make_params(beta=0.125)
t, V, C, O, I, G = simulate(p_low_beta, V0=8.0, C0=5.0, T=200)
print(f"Final V={V[-1]:.2f}, C={C[-1]:.2f}")
```

## The Dynamical System

The DLVT model is governed by two coupled ODEs:

$$\frac{dV}{dt} = R \left(1 - \frac{V}{V_{\max}}\right) - \delta \cdot O^{\gamma} \cdot \frac{V}{V + \varepsilon}$$

$$\frac{dC}{dt} = \alpha \cdot I - \mu \cdot C$$

with derived quantities:

- **Organisational Complexity:** $O(t) = O_0 + \beta \cdot C(t)^{\eta}$
- **Leadership Impact:** $I(t) = C \cdot V \,/\, (1 + \phi \cdot O)$ — energy-gated effectiveness
- **Depletion Ratio:** $\Gamma(t) = \delta \cdot O^{\gamma} \,/\, R$ — net drain when $\Gamma > 1$

The first equation balances bounded vitality recovery against complexity-driven drain. The second models career capital accumulation through impact feedback minus depreciation. The smooth barrier $V/(V+\varepsilon)$ ensures positive invariance of the state space.

## Parameter Table

All parameters from **Table 1** of the paper (baseline calibration, $C_0 = 5.0$). The calibration is **illustrative** — no parameter is estimated from data. Only ~6 dimensionless groups are structurally independent: $V^{*}$ depends on $\mu, \alpha$ only through $\mu/\alpha$, on $R, \delta$ only through $R/\delta$, and has exactly **zero** elasticity to $\beta$, $\eta$ and $O_0$ (see `dlvt.nondimensional`).

| Symbol | Key | Default | Description |
|:------:|:---:|:-------:|:------------|
| $R$ | `R` | 3.0 | Vitality recovery rate |
| $V_{\max}$ | `Vmax` | 10.0 | Maximum vitality capacity |
| $\delta$ | `delta` | 0.02 | Energetic cost coefficient |
| $\gamma$ | `gamma` | 2.0 | Complexity exponent in drain |
| $O_0$ | `O0` | 1.0 | Baseline organisational complexity |
| $\beta$ | `beta` | 0.25 | Capital–complexity coupling |
| $\eta$ | `eta` | 1.0 | Capital scaling exponent |
| $\alpha$ | `alpha` | 0.1 | Capital accumulation rate |
| $\phi$ | `phi` | 0.15 | Complexity–impact suppression |
| $\mu$ | `mu` | 0.2 | Capital depreciation rate |
| $\varepsilon$ | `eps` | 0.1 | Smooth barrier regularisation |

## Key Theoretical Results (corrected statements, v2.1)

**Theorem 1 (Finite-Time Band Entry & Positive Quasi-Equilibrium).** Vitality never reaches zero: $dV/dt|_{V=0} = R > 0$, so $V=0$ is repelling. The correct depletion statements are: *(a)* under exogenous overload $O(t) \geq \Omega$ with $\Omega^{\gamma} > 2R/\delta$, vitality enters the band $\{V < \varepsilon\}$ in finite time $T^{*} = t_0 + (V_0 - \varepsilon)/(\delta\Omega^{\gamma}/2 - R)$; *(b)* otherwise $V$ converges exponentially to a **positive** quasi-equilibrium $V_{qe}(O)$, the positive root of $(R/V_{\max})V^2 - (R - R\varepsilon/V_{\max} - \delta O^{\gamma})V - R\varepsilon = 0$. *(Earlier drafts claimed finite-time depletion to zero under any persistent deficit $\Gamma > 1$; that claim was false and has been retracted.)*

**Theorem 2 (Global Uniqueness, No Cycles, Global Asymptotic Stability).** *(a)* The interior equilibrium is unique in the entire positive orthant for **all** positive parameters: along the C-nullcline $V_c(O) = (\mu/\alpha)(1+\phi O)$, the residual $\Phi(O)$ is strictly decreasing term-by-term. *(b)* $\mathrm{tr}\,J < 0$ at every interior equilibrium (since $dC/dt=0$ forces $\alpha V^{*}/(1+\phi O^{*})=\mu$), so no Hopf bifurcation exists anywhere; the Dulac function $B = 1/C$ independently rules out closed orbits. *(c)* The equilibrium is globally asymptotically stable on the **open** set $\{C > 0\} \cap \Omega$ — the line $C=0$ is invariant and carries a saddle at $(V \approx 9.93, 0)$. The trapping rectangle is $\Omega = [0, V_{\max}] \times [0, C_{\mathrm{trap}}]$ with $C_{\mathrm{trap}}^{\eta} = ((\alpha V_{\max}/\mu - 1)/\phi - O_0)/\beta$ (baseline $C_{\mathrm{trap}} = 102.67$). *(Earlier drafts used the carrying capacity $C^{*}_{\max} = 44.99$ as the rectangle ceiling; that rectangle leaks — $C_{\mathrm{trap}} \neq C^{*}_{\max}$.)*

**Corollary (Scope Absorption / Intervention Asymmetry).** Under the separable power-law kernel, $(V^{*}, O^{*})$ solve a $\beta$-free system, so $V^{*}(\beta) \equiv V^{*}_0$ and $\beta C^{*\eta}$ is conserved ($\beta C^{*} = 8.008$ at $\varepsilon = 0.1$; $7.933$ in the $\varepsilon \to 0$ closed form). Scope reduction alone does not raise equilibrium vitality; recovery-side interventions do. The invariance is kernel-dependent (fails under exponential/Hill drains) — see Appendix A8 and `estimate_bifurcation_interval`.

**Proposition 3 (Carrying Capacity — flux threshold, not a bifurcation):** $C^{*}_{\max} = \left[\left(\left(R/\delta\right)^{1/\gamma} - O_0\right) / \beta\right]^{1/\eta}$ is the capital level at which the depletion ratio $\Gamma = 1$ at full vitality. Since $\det J > 0$ everywhere, **no saddle-node/fold exists**: equilibria do not cease to exist past $C^{*}_{\max}$; the constant separates flux-balance regions only.

**Robustness note ($\gamma = 1$):** with linear drain the unique stable equilibrium persists but is *sustainable* ($V^{*} \approx 8.56$): nonlinear complexity scaling is necessary for the low-vitality regime, not for the existence of an attractor.

## API Reference

### `dlvt.model` — Core ODE System

```python
from dlvt import make_params, simulate, complexity, impact
```

| Function | Description |
|:---------|:------------|
| `make_params(**overrides)` | Create parameter dict with keyword overrides |
| `complexity(C, p)` | $O = O_0 + \beta C^{\eta}$ |
| `impact(V, C, O, p)` | $I = CV/(1 + \phi O)$ |
| `dlvt_system(t, y, p)` | ODE right-hand side for `scipy.integrate.solve_ivp` |
| `dlvt_exogenous(t, y, p, C_func)` | Variant with exogenous career capital path |
| `simulate(p, V0, C0, T, max_step)` | Integrate ODE; returns `(t, V, C, O, I, G)` |

### `dlvt.analysis` — Equilibrium Theory & Stability

```python
from dlvt.analysis import (
    find_interior_equilibria, carrying_capacity,
    jacobian_eigenvalues, is_zombie, classify_regime, regime_map
)
```

| Function | Description |
|:---------|:------------|
| `find_interior_equilibria(p)` | Find all $(V^{*}, C^{*}) > 0$ fixed points (analytical scan window by default) |
| `carrying_capacity(p)` | Flux threshold $C^{*}_{\max}$ (Proposition 3 — not a bifurcation) |
| `trapping_capital_bound(p)` | Trapping-rectangle ceiling $C_{\mathrm{trap}}$ (Theorem 2c) |
| `jacobian_eigenvalues(V, C, p)` | Eigenvalue structure and stability class |
| `is_zombie(V, p)` | Check $V^{*} < V_{\mathrm{strategic}}$ (stipulated threshold) |
| `classify_regime(p)` | Returns `'sustainable'`, `'zombie'`, or `'collapse-prone'` |
| `regime_map(betas, deltas)` | $(\beta, \delta)$ regime classification grid |

### `dlvt.nondimensional` — Effective Parameters & Global Sensitivity

| Function | Description |
|:---------|:------------|
| `reduced_groups(p)` | The ~6 independent dimensionless groups |
| `v_star_elasticities(p)` | $\partial \ln V^{*} / \partial \ln \theta$ for all 11 parameters |
| `mu_alpha_critical(p)` | Regime flip value $(\mu/\alpha)_{\mathrm{crit}}$ |
| `zombie_boundary_map(p, ...)` | Regime frontier in $(\mu/\alpha, \phi)$ and $(\mu/\alpha, \beta)$ |
| `lhs_zombie_fraction(p, ...)` | Latin-hypercube genericity + rank-correlation screening |

### `dlvt.figures` — Publication Figure Generation

```python
from dlvt.figures import fig1, fig2, fig3, fig4, fig5, fig6, fig7

fig3(output_dir='figures/')  # Phase portrait
```

| Figure | Content |
|:-------|:--------|
| Fig 1 | Temporal evolution of $V(t)$, $C(t)$, $O(t)$, $\Gamma(t)$ |
| Fig 2 | Three outcome scenarios (sustainable, zombie, collapse) |
| Fig 3 | Phase portrait with nullclines and equilibria |
| Fig 4 | Bifurcation diagrams: $C^{*}$ and $V^{*}$ vs $\beta$, $R$ |
| Fig 5 | Impact comparison: DLVT's energetic-deployment constraint vs the frictionless human-capital benchmark (Becker makes no vitality claim; DLVT *adds* the energy channel) |
| Fig 6 | Carrying capacity heatmap $C^{*}_{\max}(\beta, R)$ |
| Fig 7 | Regime map in $(\beta, \delta)$ space |

## Reproducing All Paper Figures

```bash
python3 scripts/run_all_figures.py             # All 10 figures (PDF + PNG)
python3 scripts/run_all_figures.py --fig 1-7   # Core only
python3 scripts/run_all_figures.py --fig 8-10  # Extended analysis only
```

Extended figures (standalone scripts):

```bash
python3 scripts/fig8_bifurcation_hysteresis.py    # Scan-window artifact vs flat V*(β) — no hysteresis exists
python3 scripts/fig9_robustness.py                # Structural robustness
python3 scripts/fig10_intervention_comparison.py  # Recovery vs redesign (intervention asymmetry)
python3 scripts/fig11_sensitivity_global.py       # Elasticities, regime frontier, LHS genericity
python3 scripts/fig12_stochastic_robustness.py    # SDE attractor persistence + identifiability ridges
python3 scripts/fig13_sobol_indices.py            # Formal Sobol indices (Jansen/Saltelli)
python3 scripts/fig14_fastslow_basin.py           # Slow manifold, basin portrait, reduction error
```

## Running the Tests

```bash
pytest tests/ -q          # 38 tests, all must pass
pytest tests/ -v --tb=short  # verbose with tracebacks
```

The suite combines *pinning tests* (equilibrium values, carrying capacity, scope-absorption invariant, eigenvalue signs, basin-of-attraction convergence — so any code change that breaks paper–code consistency is caught immediately) with *independent verification checks* that use external oracles: the $\varepsilon \to 0$ closed-form equilibrium, a second integrator family (implicit Radau vs RK45), the hand-derived $(\mu/\alpha)_{\mathrm{crit}}$ boundary, direct sign checks of the trace condition and of the trapping/leaking rectangles, and random-parameter sweeps witnessing global uniqueness.

## Project Structure

```
dlvt/
├── dlvt/                     # Core package
│   ├── __init__.py           # Public API exports
│   ├── model.py              # ODE system, parameters, integration
│   ├── analysis.py           # Equilibria, stability, bifurcation, regimes
│   ├── nondimensional.py     # Effective parameters, elasticities, sensitivity
│   └── figures.py            # Publication figures 1–7
├── tests/                    # Pinning + independent verification tests
│   ├── test_model.py         # Model numerics, simulation, positive invariance
│   ├── test_analysis.py      # Equilibria, stability, regimes, external oracles
│   └── test_nondimensional.py# Reduced form, degeneracies, genericity
├── scripts/                  # Figure generation (8–11) and robustness grid
├── figures/                  # Generated output (PDF + PNG)
├── pyproject.toml            # Package metadata and dependencies
├── setup.py                  # Legacy installer
├── CITATION.cff              # Citation metadata
└── LICENSE                   # MIT
```

## Citation

```bibtex
@unpublished{bendinelli2026dlvt,
  title  = {Dynamic Leadership Vitality Theory: A Closed-Form Capability-Trap
            Model of Leader Vitality and Career Capital},
  author = {Bendinelli, William},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright 2026 William Bendinelli

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues, feature requests, and pull requests are welcome.
