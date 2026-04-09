"""
robustness_grid.py
==================
Programmatic construction of the DLVT structural robustness table (R7-7).

For each perturbation of the baseline model, compute three yes/no/conditional
answers:

  1.  Does an interior equilibrium (V*, C*) with V* > 0 and C* > 0 exist?
  2.  Is V* < V_strategic = 0.5 * V_max (i.e. the interior equilibrium is a
      zombie)?
  3.  Does the bifurcation narrative survive --- does V*(beta) respond
      non-trivially to beta, so that a beta_crit can in principle exist?
      Under the baseline power-law kernel scope absorption (Lemma 2) gives
      dV*/dbeta = 0, so this column is "no" for the baseline and "yes" or
      "conditional" for non-power-law kernels.

The script integrates a small grid of perturbed ODEs to long time and reads
the converged state as the equilibrium. Alternative drain kernels are
supplied as RHS overrides rather than through dlvt.model (which bakes in the
power-law kernel).

Usage
-----
    python3 code/scripts/robustness_grid.py
    python3 code/scripts/robustness_grid.py --csv out.csv --tex out.tex

Outputs CSV and LaTeX booktabs table suitable for \\input into the paper.

Runtime: < 30 s on a laptop.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# Make the dlvt package importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from dlvt.model import DEFAULT_PARAMS, make_params  # noqa: E402


V_STRATEGIC_FRACTION = 0.5
T_FINAL = 1200.0
RTOL = 1e-8
ATOL = 1e-10

# Three-state cell values.
YES = "yes"
NO = "no"
COND = "conditional"


# ── Drain kernel factories ────────────────────────────────────────────────────

def power_law_rhs(p: Dict[str, float]) -> Callable[[float, List[float]], List[float]]:
    """Baseline DLVT drain kernel: delta * O^gamma * V/(V + eps)."""
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def exponential_rhs(p: Dict[str, float], kappa: float = 0.08) -> Callable:
    """Exponential drain kernel with explicit C dependence:
        drain = delta * exp(kappa * C) * V / (V + eps).

    Note that scope absorption (Lemma 2) holds for ANY drain of the form
    f(O) * g(V) because the C-isocline forces O to be a function of V alone,
    reducing the V-isocline to a pure V equation. An exponential in
    (beta * C) does not break this, because beta * C is also determined by V
    alone when eta = 1. The genuine counterexample is a drain that depends on
    C directly, not only through the composite O or beta * C --- hence
    exp(kappa * C), which introduces an un-absorbed C term into the
    V-isocline and lets V*(beta) respond to beta.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * np.exp(kappa * C) * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def hill_rhs(p: Dict[str, float], K: float = 4.0) -> Callable:
    """Hill (saturating) drain kernel: delta * O^gamma / (K^gamma + O^gamma) *
    V/(V + eps). The drain saturates at delta as complexity grows, so a
    high-complexity leader is NOT unbounded-drain, which breaks the zombie
    narrative in strong-saturation regimes. Like the power-law kernel this
    is of the form f(O)*g(V), so Lemma 2 still applies and the V-isocline
    collapses to a pure V equation: the zombie equilibrium disappears
    because saturation pushes V* above V_strategic, not because the
    bifurcation structure changes.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        hill = O ** p["gamma"] / (K ** p["gamma"] + O ** p["gamma"])
        drain = p["delta"] * hill * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def explicit_C_drain_rhs(p: Dict[str, float], theta: float = 0.001) -> Callable:
    """Drain with an explicit C-dependent multiplier not mediated by O.

    drain = delta * O^gamma * (1 + theta * C^2) * V/(V + eps).

    Scope absorption (Lemma 2) holds for any drain of the form f(O)*g(V)
    because the C-isocline forces O to be a function of V alone, collapsing
    the V-isocline to a pure V equation. The only way to break scope
    absorption while keeping the C-isocline intact is to add an explicit C
    term to the drain that is NOT absorbed into O. This kernel does exactly
    that: the (1 + theta*C^2) factor introduces a beta-dependent term into
    the V-isocline through C = (A(V)/beta)^(1/eta), so V*(beta) now responds
    to beta.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C ** p["eta"]
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * (1.0 + theta * C * C) * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


def saturating_complexity_rhs(p: Dict[str, float], K: float = 50.0) -> Callable:
    """Baseline kernel, but with saturating complexity O = O0 + beta*C/(1 + C/K).
    Complexity plateaus, so drain cannot grow without bound.
    """
    def rhs(t, y):
        V = max(y[0], 0.0)
        C = max(y[1], 0.0)
        O = p["O0"] + p["beta"] * C / (1.0 + C / K)
        recovery = p["R"] * (1.0 - V / p["Vmax"])
        drain = p["delta"] * O ** p["gamma"] * V / (V + p["eps"])
        I = C * V / (1.0 + p["phi"] * O)
        return [recovery - drain, p["alpha"] * I - p["mu"] * C]
    return rhs


# ── Equilibrium computation ───────────────────────────────────────────────────

def integrate_to_equilibrium(rhs: Callable,
                             V0: float = 8.0,
                             C0: float = 5.0,
                             t_final: float = T_FINAL) -> Tuple[float, float, bool]:
    """Integrate the rhs to t_final. Return (V*, C*, converged?)."""
    sol = solve_ivp(rhs, (0.0, t_final), [V0, C0],
                    method="RK45", rtol=RTOL, atol=ATOL,
                    t_eval=np.linspace(t_final * 0.9, t_final, 50))
    V_tail = sol.y[0]
    C_tail = sol.y[1]
    V_star = float(V_tail[-1])
    C_star = float(C_tail[-1])
    # "Converged" = tail variation small relative to level.
    tail_spread_V = float(V_tail.max() - V_tail.min())
    tail_spread_C = float(C_tail.max() - C_tail.min())
    converged = (tail_spread_V < max(1e-3, 1e-3 * abs(V_star)) and
                 tail_spread_C < max(1e-3, 1e-3 * abs(C_star)))
    return V_star, C_star, converged


def has_interior_equilibrium(V_star: float, C_star: float,
                             V_max: float, tol: float = 1e-3) -> bool:
    """Interior means both V* > 0 and C* > 0 (strictly)."""
    return V_star > tol and C_star > tol


def is_zombie(V_star: float, V_max: float) -> bool:
    return V_star < V_STRATEGIC_FRACTION * V_max


def bifurcation_survives(rhs_factory: Callable[[Dict[str, float]], Callable],
                         p_base: Dict[str, float],
                         beta_low: float = 0.15,
                         beta_high: float = 0.40) -> str:
    """Check whether V*(beta) depends on beta.

    Returns YES if |dV*/dbeta| is "large" (relative shift > 1%),
    COND if the shift is 0.1% - 1% (borderline),
    NO  if the shift is < 0.1% (scope-absorbed).
    """
    p_low = dict(p_base); p_low["beta"] = beta_low
    p_high = dict(p_base); p_high["beta"] = beta_high
    V_low, C_low, ok_low = integrate_to_equilibrium(rhs_factory(p_low))
    V_high, C_high, ok_high = integrate_to_equilibrium(rhs_factory(p_high))
    if not (ok_low and ok_high):
        return COND
    if not (has_interior_equilibrium(V_low, C_low, p_low["Vmax"]) and
            has_interior_equilibrium(V_high, C_high, p_high["Vmax"])):
        return COND
    rel_shift = abs(V_high - V_low) / max(1e-6, 0.5 * (V_low + V_high))
    if rel_shift > 1e-2:
        return YES
    if rel_shift > 1e-3:
        return COND
    return NO


# ── Perturbation grid ─────────────────────────────────────────────────────────

def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    V_max = DEFAULT_PARAMS["Vmax"]

    def evaluate(label: str, rhs_factory: Callable, p: Dict[str, float]) -> Dict[str, str]:
        V_star, C_star, conv = integrate_to_equilibrium(rhs_factory(p))
        interior = has_interior_equilibrium(V_star, C_star, V_max)
        if not interior:
            exists = NO
            zombie = NO
        else:
            exists = YES
            zombie = YES if is_zombie(V_star, V_max) else NO
        bifurc = bifurcation_survives(rhs_factory, p)
        return {
            "perturbation": label,
            "V_star": f"{V_star:.3f}",
            "C_star": f"{C_star:.3f}",
            "exists": exists,
            "zombie": zombie,
            "bifurcation": bifurc,
        }

    # Baseline (anchor row).
    rows.append(evaluate(
        r"Baseline (power-law, $\gamma=2$)",
        power_law_rhs,
        make_params(),
    ))

    # Gamma perturbations.
    for g in (1.0, 3.0):
        rows.append(evaluate(
            rf"$\gamma = {int(g)}$ (power-law drain)",
            power_law_rhs,
            make_params(gamma=g),
        ))

    # Drain-kernel perturbations.
    rows.append(evaluate(
        "Exponential drain kernel",
        lambda p: exponential_rhs(p, kappa=0.35),
        make_params(),
    ))
    rows.append(evaluate(
        "Hill (saturating) drain kernel",
        lambda p: hill_rhs(p, K=4.0),
        make_params(),
    ))
    rows.append(evaluate(
        "Saturating complexity $O(C)$",
        lambda p: saturating_complexity_rhs(p, K=50.0),
        make_params(),
    ))
    rows.append(evaluate(
        "Drain with explicit $C$ term",
        lambda p: explicit_C_drain_rhs(p, theta=0.001),
        make_params(),
    ))

    # phi perturbations (impact sensitivity to complexity).
    for phi in (0.075, 0.30):
        rows.append(evaluate(
            rf"$\phi = {phi}$",
            power_law_rhs,
            make_params(phi=phi),
        ))

    # mu perturbations (capital decay rate).
    for mu in (0.10, 0.40):
        rows.append(evaluate(
            rf"$\mu = {mu}$",
            power_law_rhs,
            make_params(mu=mu),
        ))

    # eps perturbations (regularization).
    for eps in (0.01, 1.0):
        rows.append(evaluate(
            rf"$\varepsilon = {eps}$",
            power_law_rhs,
            make_params(eps=eps),
        ))

    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    fieldnames = ["perturbation", "V_star", "C_star", "exists", "zombie", "bifurcation"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: List[Dict[str, str]], path: Path) -> None:
    header = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Structural robustness of DLVT predictions. Rows perturb one "
        r"assumption of the baseline model (Table~\ref{tab:parameters}) at a time; "
        r"columns ask whether each core feature of the theory survives. "
        r"``Interior $(V^*, C^*)$ exists'' asks whether an interior stable "
        r"equilibrium is reached from $(V_0, C_0)=(8, 5)$ under long-time integration. "
        r"``Zombie ($V^* < V_{\text{strategic}}$)'' asks whether that equilibrium "
        r"lies below the strategic threshold. ``Bifurcation survives'' asks whether "
        r"$V^*(\beta)$ responds non-trivially to $\beta$; scope absorption "
        r"(Lemma~\ref{lem:scope_absorption}) forces ``no'' for any power-law drain "
        r"kernel at the baseline calibration. All rows are generated by "
        r"\texttt{code/scripts/robustness\_grid.py}.}",
        r"\label{tab:robustness}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Perturbation & Interior $(V^*, C^*)$ exists & "
        r"Zombie ($V^* < V_{\text{strategic}}$) & Bifurcation survives \\",
        r"\midrule",
    ]
    body = []
    for row in rows:
        body.append(
            f"{row['perturbation']} & {row['exists']} & {row['zombie']} & {row['bifurcation']} \\\\"
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(header + body + footer) + "\n")


def print_ascii(rows: List[Dict[str, str]]) -> None:
    col_widths = [40, 10, 10, 10, 15]
    headers = ["Perturbation", "V*", "C*", "zombie", "bifurcation"]
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        # Strip LaTeX for readable console output.
        label = (row["perturbation"]
                 .replace("$", "").replace("\\gamma", "gamma")
                 .replace("\\phi", "phi").replace("\\mu", "mu")
                 .replace("\\varepsilon", "eps").replace("O(C)", "O(C)"))
        cells = [label[: col_widths[0]],
                 row["V_star"],
                 row["C_star"],
                 row["zombie"] if row["exists"] == YES else "n/a",
                 row["bifurcation"]]
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths)))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("robustness_grid.csv"))
    ap.add_argument("--tex", type=Path, default=Path("robustness_grid.tex"))
    args = ap.parse_args()

    rows = build_rows()
    print_ascii(rows)
    write_csv(rows, args.csv)
    write_latex(rows, args.tex)
    print(f"\nWrote {args.csv} and {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
