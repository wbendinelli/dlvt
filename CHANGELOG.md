# Changelog

All notable changes to **dlvt** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-04-11 — R7-Complete Public Release

### Added
- **`dlvt.analysis` module** — Full equilibrium theory suite:
  - `find_interior_equilibria` — numerical root-finding for all $(V^*, C^*)$ fixed points
  - `carrying_capacity` — closed-form $C^*_{\max}$ from Proposition 3
  - `jacobian_eigenvalues` — eigenvalue structure and stability classification
  - `is_zombie` — threshold check $V^* < V_{\text{strategic}}$
  - `classify_regime` — returns `'sustainable'`, `'zombie'`, or `'collapse-prone'`
  - `regime_map` — $(\beta, \delta)$ classification grid
- **Extended figure scripts** (`scripts/`):
  - `fig8_bifurcation_hysteresis.py` — hysteresis detection across parameter sweeps
  - `fig9_robustness.py` — structural robustness analysis
  - `fig10_intervention_comparison.py` — recovery vs. redesign intervention comparison
- **38-test pinning suite** (`tests/`): every numerical claim in the paper is pinned
  - `test_model.py` — ODE numerics, simulation invariants, positive invariance
  - `test_analysis.py` — equilibria, carrying capacity, regime classification
- **`dlvt_exogenous`** variant supporting exogenous career capital path $C(t)$
- **`CONTRIBUTING.md`** with branching strategy and PR guidelines
- **`QUICKSTART.md`** for immediate usage without reading the full README
- **Scope-absorption invariant** verification: $\beta C^* \approx 8.008$ preserved

### Changed
- Refactored `dlvt.model` to cleanly separate ODE right-hand side, integration, and derived quantities
- `simulate()` now returns full tuple `(t, V, C, O, I, G)` including depletion ratio $\Gamma(t)$
- README expanded with full API reference, parameter table, and theoretical results

### Fixed
- Positive invariance of state space ensured via smooth barrier $V/(V+\varepsilon)$
- Numerical precision for equilibrium detection at boundary cases

---

## [1.0.0] — 2026-04-01 — Initial Public Release

### Added
- **Core ODE system** (`dlvt.model`):
  - `make_params(**overrides)` — parameter dict with Table 1 defaults
  - `dlvt_system(t, y, p)` — right-hand side for `scipy.integrate.solve_ivp`
  - `simulate(p, V0, C0, T)` — full trajectory integration
  - `complexity(C, p)` and `impact(V, C, O, p)` — derived quantities
- **Publication figures 1–7** (`dlvt.figures`):
  - Fig 1: temporal evolution of $V(t)$, $C(t)$, $O(t)$, $\Gamma(t)$
  - Fig 2: three outcome scenarios
  - Fig 3: phase portrait with nullclines
  - Fig 4: bifurcation diagrams
  - Fig 5: DLVT vs Human Capital Theory comparison
  - Fig 6: carrying capacity heatmap
  - Fig 7: regime map
- **`scripts/run_all_figures.py`** — reproducible figure generation (PDF + PNG)
- **`pyproject.toml`** + `setup.py` — installable as `pip install -e .`
- **`requirements.txt`** — Python ≥ 3.8, NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7
- **MIT License** and **`references.bib`** (BibTeX citation)
- Preprint available on SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6455001

---

[2.0.0]: https://github.com/wbendinelli/dlvt/releases/tag/v2.0.0
[1.0.0]: https://github.com/wbendinelli/dlvt/releases/tag/v1.0.0
