# Pre-Registration (DRAFT / TEMPLATE)

> **STATUS: DRAFT / TEMPLATE — NOT YET REGISTERED.**
> This document is an OSF-style pre-registration draft for the DLVT "killer
> test" (paper §6.5; Propositions P3 and P6). Fields marked
> `[TO BE FINALIZED]` must be completed by the author — after the pilot
> study and the mandatory parameter-recovery simulation — **before** the
> document is frozen on OSF and before the first participant is randomized.

---

## 1. Title

Scope Reduction versus Recovery Support: A Pre-Registered 2×2 Field
Experiment Testing the Intervention-Asymmetry Prediction of Dynamic
Leadership Vitality Theory (DLVT).

## 2. Background and theoretical predictions

DLVT (Bendinelli, working paper; companion software: the `dlvt` Python
package in this repository) models leader vitality `V` and career capital
`C` as a coupled dynamical system in which capital endogenously generates
the complexity load `O = O0 + β·C^η` that drains vitality. Two verified
model results drive this experiment:

- **Scope absorption (P3).** Within the power-law drain family, lowering
  the scope coupling β produces only a *transient* vitality bump: capital
  re-expands until the product β·C^η is restored, and equilibrium vitality
  `V*` is unchanged.
- **Intervention asymmetry (P6).** Recovery-side interventions (raising the
  recovery rate R) durably raise `V*`; scope-side interventions (lowering
  β) do not.

The relevant verified timescales at the illustrative calibration:
eigenvalues −0.205 ± 0.331i (relaxation time ≈ 4.9 model time units;
oscillation period ≈ 19 units), capital relaxation time 1/μ = 5 units,
fast vitality timescale 1/R ≈ 0.33 units. Under the illustrative weekly
calibration the re-expansion dynamics complete within months; the endpoint
below allows for real-world timescales several times slower.

## 3. Hypotheses

- **H1 (Recovery durability).** At the primary endpoint (12–18 months
  post-randomization), the recovery-support arms (R↑ and R↑+β↓) show a
  durable increase in latent vitality relative to control, with the effect
  exceeding the declared SESOI (§7).
- **H2 (Scope transience).** The scope-reduction-only arm (β↓) shows
  (a) a positive vitality difference from control at ~3 months
  (the transient bump), and (b) a vitality difference from control at
  12–18 months that lies **within** the declared equivalence bounds
  (±SESOI) — i.e., the bump decays as β·C^η re-asserts itself.
- **H3 (Conservation).** In the β↓-only arm, the product β·C^η is conserved
  at the endpoint: measured complexity `O` returns to its pre-treatment
  level while capital indicators re-expand (predicted re-expansion factor
  β_pre/β_post for η = 1). Operationally: the endpoint-vs-baseline change
  in structural complexity indicators is within equivalence bounds
  `[TO BE FINALIZED]`, while capital indicators show a positive change.

## 4. Design

- **Type.** Randomized controlled field experiment, 2×2 factorial between
  subjects, with longitudinal follow-up (≥ 8 quarterly measurement waves
  plus embedded daily ESM bursts; identical instruments to paper §6.1).
- **Arms.**
  1. **Control** (waitlist; receives the recovery program after the
     primary endpoint).
  2. **Scope reduction (β↓):** structural delegation of units, interface
     pruning, span reduction. Protocol: `[TO BE FINALIZED]`.
  3. **Recovery support (R↑):** protected detachment blocks, recovery
     training, sabbatical scheduling. Protocol: `[TO BE FINALIZED]`.
  4. **Both (β↓ + R↑).**
- **Population.** Executives/senior leaders in comparable roles;
  participating organizations: `[TO BE FINALIZED]`.
- **Primary endpoint.** 12–18 months post-randomization (exact month
  `[TO BE FINALIZED]` from pilot timescale estimates) — chosen to exceed
  2–3 capital relaxation times so that capital re-expansion can occur;
  a shorter endpoint would sample the transient and mistake it for a
  durable effect.
- **Interim assessment.** ~3 months post-randomization (transient bump,
  H2a).

## 5. Randomization

- Stratified block randomization within organization and role-tenure
  cohort (strata: `[TO BE FINALIZED]`), allocation ratio 1:1:1:1.
- Allocation concealed from the research team performing measurement;
  blinding of participants to hypotheses (participant materials use
  neutral language; see §11).
- Randomization seed and code deposited with the registration.

## 6. Outcomes and instruments

Identical to the measurement model of paper §6.1 (Table 6.1):

- **Primary outcome.** Latent state vitality `V`: Subjective Vitality
  Scale (state version; Ryan & Frederick 1997) from daily ESM bursts,
  aggregated by the state-space measurement model.
- **Secondary outcomes.**
  - Recovery experiences (Recovery Experience Questionnaire; Sonnentag &
    Fritz 2007) — manipulation check for R↑.
  - Emotional exhaustion (MBI subscale; Maslach & Jackson 1981) —
    discriminant marker.
  - Career capital `C`: two-factor status/competence composite (tenure,
    span, budget authority, network centrality, external reputation;
    assessed competence) — H3.
  - Complexity `O`: executive job demands (Hambrick, Finkelstein & Mooney
    2005), WDQ complexity/information-processing subscales (Morgeson &
    Humphrey 2006), structural load metrics — manipulation check for β↓
    and H3.
  - Impact `I`: multi-source (360) effectiveness ratings and objective
    unit performance.
- **Manipulation checks.** β↓ arm must show reduced structural complexity
  at 3 months; R↑ arm must show increased recovery experiences at
  3 months. Failure thresholds: `[TO BE FINALIZED]`.

## 7. Smallest Effect Size of Interest (SESOI)

- **Declared SESOI for ΔV:** `[TO BE FINALIZED]` (placeholder;
  candidate: d = 0.3 on latent V, pending justification).
- **Justification method (choose and document one before freezing):**
  (a) anchor-based — the latent-V change corresponding to a minimally
  noticeable change on an external anchor (e.g., one response category on
  a global energy item); (b) cost-effectiveness — the smallest ΔV that
  would justify the intervention's organizational cost; or (c) benchmark —
  the smallest ΔV distinguishable from the median decaying respite effect
  in the recovery literature. The SESOI defines both the superiority
  margin (H1) and the equivalence bounds (H2b, H3).

## 8. Analysis plan

1. **Confirmatory models.**
   - Mixed-effects models (leader-level random effects; organization and
     cohort strata as fixed effects) for arm × time effects on latent
     outcome scores.
   - **Equivalence tests** (TOST against ±SESOI) for H2b and H3; H1 tested
     as superiority against the SESOI margin.
   - **Continuous-time state-space estimation (ctsem or Stan)** with the
     exact DLVT drift, targeting the reduced dimensionless groups (never
     the raw 11 parameters), to estimate arm-specific shifts in R and β
     and to test conservation of β·C^η structurally.
2. **Sequencing gates (pre-committed).** Longitudinal measurement
   invariance (configural/metric/scalar) must be established before latent
   change is interpreted; the bifactor rival CFA for C is estimated first,
   and if scalar-C measurement is rejected, H3 is evaluated on the status
   sub-factor with the deviation reported.
3. **Attrition.** Joint competing-risks / shared-parameter survival
   sub-model with hazard depending on latent V; exits analyzed, not
   dropped. (Secondary, exploratory DLVT test: hazard concentration where
   the depletion ratio exceeds 1.)
4. **Inference thresholds.** α = 0.05 two-sided for superiority; 90% CI
   within bounds for equivalence; Bayesian model comparisons reported with
   pre-declared priors `[TO BE FINALIZED]`.
5. **Deviations** from this plan will be reported as exploratory.

## 9. Power and sample size (simulation-based)

- Power is established **only** by simulation from the state-space model
  (companion parameter-recovery module `dlvt/recovery.py` extended to the
  trial design): synthetic trials generated under (i) the DLVT drift and
  (ii) a rival durable-scope-effect drift, at the proposed cadence,
  reliabilities, and attrition.
- **Working sketch:** N = 60–100 leaders per arm (240–400 total), ≥ 8
  quarterly waves post-randomization with embedded 10-working-day daily
  ESM bursts. Required: ≥ 80% power for H1, ≥ 80% probability of
  affirming equivalence for H2b under the DLVT drift, and ≥ 80% power for
  the 3-month transient (H2a). Final N per arm: `[TO BE FINALIZED]` from
  the simulation output, which will be deposited with this registration.

## 10. Exclusion criteria

- Enrollment: < 6 months expected role continuity; concurrent participation
  in another workload/recovery intervention; roles where complexity load is
  predominantly exogenous to own scope (violates the model's scope
  condition, Assumption 1 of the paper).
- Post-randomization: manipulation-check failure handled by
  intention-to-treat as primary, per-protocol as sensitivity; waves with
  < `[TO BE FINALIZED]` % ESM compliance flagged, not excluded.
- Data: pre-declared careless-responding screens `[TO BE FINALIZED]`.

## 11. Falsification statement

**The theory dies if:** at the 12–18-month endpoint, the
scope-reduction-only arm shows a durable vitality gain exceeding the SESOI
**and** the product β·C^η is not conserved (capital does not re-expand;
complexity remains durably lowered). That outcome falsifies the
re-expansion mechanism itself (P3 and P6 jointly), not a calibration
choice, and no kernel-family qualification (P4) rescues it. Conversely:
a durable scope effect *with* β·C^η conserved indicates kernel
mis-specification (P4); a transient bump that decays with conservation is
the theory's predicted fingerprint.

## 12. Data and ethics governance

- **Research–HR firewall:** individual-level vitality, exhaustion, and
  survival-model outputs never reach the employer and never enter
  personnel decisions; only pre-agreed aggregates are reported out.
- **Consent:** opt-in, with per-wave re-consent and separate consent for
  passive metadata (calendar/network indicators); data minimization and
  on-site aggregation for metadata.
- **Language:** all participant-facing materials use neutral terms
  (workload, recovery, sustainable performance); theoretical labels —
  including any legacy informal labels from earlier drafts — never appear
  in participant materials.
- **Waitlist control:** control participants receive the recovery program
  after the primary endpoint.
- **Approvals:** IRB and (where applicable) works-council approval precede
  enrollment. IRB protocol number: `[TO BE FINALIZED]`.
- **Data sharing:** de-identified data and analysis code deposited at
  `[TO BE FINALIZED]` under a controlled-access agreement.

---

*Author: William Bendinelli. Draft prepared alongside paper §6
(`paper/chapters/06_empirical_program.tex`). Freeze on OSF only after all
`[TO BE FINALIZED]` fields are completed and the parameter-recovery and
power simulations are deposited.*
