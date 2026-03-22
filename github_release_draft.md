# DLVT GitHub Publication Texts

## 1. Description para o "About" do GitHub (Canto superior direito)
**Title:** dlvt
**Description:** Official Python implementation of the Dynamic Leadership Vitality Theory (DLVT) — a dynamical systems model of executive sustainability, endogenous complexity, and the Zombie Equilibrium.
**Website:** https://ssrn.com/ (ou cole o link real do seu SSRN)
**Topics:** `dynamical-systems` `leadership` `organizational-behavior` `complexity` `burnout` `python`

---

## 2. Release v1.0.0 (Release Notes)
**Tag version:** `v1.0.0`
**Release title:** DLVT v1.0.0: Initial Scientific Release (SSRN Working Paper)

**Descrição Completa (Copie e cole na caixa de texto do Release):**

```markdown
### 🚀 Official Code Release for the Dynamic Leadership Vitality Theory (DLVT)

This is the inaugural open-source release of the Python codebase underpinning the paper:  
**"The Carrying Capacity of Leadership: A Dynamic Systems Model of Executive Sustainability"** *(SSRN Working Paper, 2026)*.

This repository provides full computational reproducibility for all mathematical propositions, equilibrium analysis, and phase-space simulations discussed in the paper. We have engineered this codebase to meet the highest State-of-the-Art (SOTA) standards for open-source scientific software.

### 🧬 Scientific Implementation
* **ODEs & Stability:** Full implementation of the two-dimensional dynamical system tracking Vitality ($V$) and Career Capital ($C$).
* **Endogenous Drain:** Algorithmic representation of the "Zombie Equilibrium" through capital-complexity coupling ($\beta$).
* **Analytical Solvers:** Matrix calculations for Jacobian eigenvalues to evaluate the stability of equilibrium points.

### 🛠️ SOTA Engineering Standards
* **Test Suite:** Comprehensive unit testing via `pytest` to guarantee mathematical fidelity across systems.
* **Continuous Integration:** Fully automated CI pipeline via GitHub Actions to maintain reproducibility.
* **Reproducibility script:** A single entry point (`scripts/dlvt_figures.py`) automatically generates all 7 publication-ready vector figures exactly as they appear in the preprint.

### 📚 Quick Start
```bash
git clone https://github.com/yourusername/dlvt.git
cd dlvt/code
pip install -e .
python scripts/dlvt_figures.py
```

*See the `README.md` for full API documentation and instructions on extending the model.*
```
