# Changelog

All notable changes to **dlvt** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] — 2026-07-01 — Referee-Panel Corrections Release

Corrections resulting from an adversarial multi-referee review of the theory
and the code. **Every substantive change below fixes an error in a stated
result or documents a calibration dependence** — the ODE system itself is
unchanged.

### Fixed (theory statements)
- **Theorem 1 (finite-time depletion) retracted and replaced.** $V=0$ is
  repelling ($dV/dt|_{V=0}=R>0$); vitality never reaches zero. Corrected
  statements: finite-time entry into the band $\{V<\varepsilon\}$ with an
  explicit bound $T^*$ under exogenous overload, and exponential convergence
  to a *positive* quasi-equilibrium $V_{qe}(O)$ otherwise (`dlvt.model`
  docstrings, README).
- **Theorem 2 trapping rectangle corrected.** Earlier text used the carrying
  capacity $C^*_{\max}=44.99$ as the rectangle ceiling; that rectangle leaks
  (at $C=C^*_{\max}$, $V=V_{\max}$, $dC/dt>0$). The correct ceiling is
  $C_{\mathrm{trap}}=102.67$, now exposed as `trapping_capital_bound()`.
- **Basin re-scoped to the open set $\{C>0\}$** — the $C=0$ axis is invariant
  and carries a saddle; the closed quadrant is not the basin.
- **Proposition 3 re-stated as a flux threshold, not a bifurcation** —
  $\det J>0$ everywhere; no saddle-node exists anywhere.
- **$\gamma=1$ description corrected**: linear drain yields a *sustainable*
  stable equilibrium ($V^*\approx 8.56$); the equilibrium does not "disappear".
- **`references.bib`: `repenning2002capability` mis-citation fixed** — the key
  carried the title/venue of Repenning (2002, *Organization Science*); it now
  correctly cites Repenning & Sterman (2002, *ASQ*, capability traps), with the
  Org Sci paper added under `repenning2002simulation`.

### Fixed (code)
- **`find_interior_equilibria` fixed-window bug**: the fixed default
  `C_max=120` silently missed the equilibrium for $\beta<0.066$ (since
  $C^*(\beta)\approx 8.008/\beta$), causing `classify_regime` to mislabel those
  regimes as `'collapse-prone'`. The default now derives the scan window from
  the analytical bound $C_{\mathrm{trap}}\propto 1/\beta$, valid at every β.
- **`scripts/fig8_bifurcation_hysteresis.py` rewritten**: the old script
  re-implemented the model with a fixed `C_max=80` window and reported a
  spurious "β_crit ≈ 0.1015" with hysteresis. It now *illustrates* the
  scan-window artifact (panel a) against the corrected flat $V^*(\beta)$
  (panel b), importing everything from the `dlvt` package.
  `find_critical_beta`/`detect_hysteresis` removed.
- **`scripts/audit_refs.py`**: hardcoded absolute path replaced by a
  repo-relative default; added `--bib` and `--offline` (structure-only) modes.

### Added
- **Global uniqueness and no-Hopf proofs** (Theorem 2a/2b in the
  `dlvt.analysis` header): $d\Phi/dO<0$ term-by-term ⟹ at most one interior
  equilibrium in the whole positive orthant; $\mathrm{tr}\,J<0$ at every
  interior equilibrium ⟹ no Hopf bifurcation anywhere — both for *all*
  positive parameters, strengthening the former baseline-only claims.
- **`dlvt.nondimensional`**: reduced dimensionless form (~6 effective groups),
  elasticities of $V^*$, the regime flip value $(\mu/\alpha)_{\mathrm{crit}}
  \approx 2.163$, regime frontier maps, and Latin-hypercube genericity
  analysis; plus `scripts/fig11_sensitivity_global.py`.
- **Calibration-dependence disclosure** (README, module headers): baseline
  numbers are illustrative; the regime label is governed by $\mu/\alpha$
  (8% from the flip at baseline); $P(\text{zombie}\,|\,\text{stable})\approx
  0.49$ over a ±2× hypercube; the ε-dependence of analytical constants
  ($\beta C^*=8.008$ at $\varepsilon=0.1$ vs $7.933$ as $\varepsilon\to0$).
- **Independent verification tests**: ε→0 closed-form oracle, second
  integrator family (Radau), hand-derived $(\mu/\alpha)_{\mathrm{crit}}$
  boundary, direct trapping/leaking checks, random-parameter uniqueness and
  trace-sign sweeps, small-β regression for the fixed-window bug.
- **Manuscript** (LaTeX): the paper skeleton introduced in v2.1.0 now lives in the canonical [dynamic-leadership-vitality-theory](https://github.com/wbendinelli/dynamic-leadership-vitality-theory) repository; this package is code-only.
  with the corrected and strengthened statements — resolves the previously
  dangling §/appendix references in docstrings and tests.
- **CI test workflow** (`.github/workflows/tests.yml`): pytest on Python
  3.9–3.12 (previously CI only published to PyPI and never ran the suite).
- **`CITATION.cff`**; version unified at 2.1.0 across `setup.py`,
  `pyproject.toml` and `dlvt.__version__` (was inconsistently 2.0.0/2.0.1).

### Changed
- **Contribution reframing** (README, paper): "zombie" retired as a defined
  construct (kept as informal label and in the code API); headline is now the
  *globally* stable low-vitality/high-status attractor (capability-trap
  family: Repenning & Sterman 2002; Rahmandad & Repenning 2016) plus the
  *intervention asymmetry* (scope absorption as an honest, kernel-dependent
  corollary); micro-foundation moved from ego depletion to effort-recovery;
  Becker comparison corrected (DLVT *adds* an energy channel; human-capital
  theory makes no vitality claim).
- Build artifacts (`dlvt.egg-info/`, `__pycache__/`) untracked from git.

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

[2.1.0]: https://github.com/wbendinelli/dlvt/releases/tag/v2.1.0
[2.0.0]: https://github.com/wbendinelli/dlvt/releases/tag/v2.0.0
[1.0.0]: https://github.com/wbendinelli/dlvt/releases/tag/v1.0.0
