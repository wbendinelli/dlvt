# DLVT Manuscript — "Dynamic Leadership Vitality Theory: A Closed-Form Capability-Trap Model of Leader Vitality and Career Capital"

**Status:** working-paper skeleton with corrected, reviewed statements. Earlier drafts' claims (finite-time depletion to zero, beta_crit ~ 0.1015, C*_max as trapping bound/bifurcation, "zombie" as a construct) are retracted and replaced; see Sections 3.5, 3.9 and Appendices A6, A8, A10. Appendices A2–A3 were retracted (numbering A1, A4–A10 is kept for continuity with the code).

**Code mapping:** Eq. (3.2)/(3.5)/(3.6)/(3.7) ↔ `dlvt/model.py` (`impact`, `dlvt_system`, `complexity`); Table 1 (Section 3.7) ↔ `DEFAULT_PARAMS` and `tests/test_model.py::PAPER_BASELINE`; Theorem 1, Theorem 2, Lemma 1, Corollary 1 (legacy "Lemma 2"), Proposition 3 / Eqs. (3.14)–(3.15) ↔ `dlvt/analysis.py` (`find_interior_equilibria`, `bendixson_dulac_certificate`, `jacobian_eigenvalues`, `estimate_bifurcation_interval`, `carrying_capacity`); Definitions 5 and 7 ↔ `simulate` (depletion ratio G) and `is_zombie`.

Appendix ↔ code: A1 `test_model.py` (invariance tests); A4–A6 `test_analysis.py` (equilibria, Jacobian, carrying capacity); A7 `find_regularization_branch`; A8 `estimate_bifurcation_interval`; A9 epsilon ladder; A10 `bendixson_dulac_certificate` + `basin_of_attraction_sweep`.

Figures 1–7 are read from `../figures/` (generate via `python3 scripts/run_all_figures.py --fig 1-7` at the repo root).

Every numerical value quoted in the manuscript is pinned by `pytest tests/ -q` (38 tests).

**Build** (from this directory; bibliography lives at `../references.bib`):

    pdflatex main && bibtex main && pdflatex main && pdflatex main
