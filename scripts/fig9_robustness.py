#!/usr/bin/env python3
"""
DLVT — Figure 9: Structural Robustness / Sensitivity Analysis (R1.6, corrected).

Tests alternative functional specifications against the baseline. The baseline
model (parameters, complexity, impact, RHS, Jacobian) is imported from the
`dlvt` package — nothing baseline is re-implemented here. Only the
*alternative-kernel* right-hand sides live in this script, because they are
the experiment.

Corrections in this revision (v2.1):
- The equilibrium scan window was a fixed ``C_max=80`` — the same class of
  scan-window artifact retracted in Appendix A8. The saturating-complexity
  kernel's interior equilibrium sits at C* ≈ 89.1, *outside* the old window,
  so the script reported "no equilibrium" while its own simulation settled at
  (V, C) ≈ (4.70, 89.15). The default window is now 2000.
- ``find_equilibria_generic`` previously hardcoded the *baseline* C-nullcline
  ``V = μ(1+φO)/α`` for every spec, which is wrong for non-linear-in-C impact
  kernels: the "equilibrium" it reported for the diminishing-impact spec
  (V*=4.70, C*=32.03) was the baseline's, while the true equilibrium of that
  spec is (V* ≈ 8.25, C* ≈ 16.6) — sustainable, not zombie. The nullcline is
  now built from the actual impact kernel (valid for any V-linear impact).
- Stability was classified with the baseline Jacobian for every spec; it is
  now computed by central finite differences on the actual RHS of each spec.

The corrected table is more honest AND more interesting: the low-vitality
equilibrium survives the saturating-complexity kernel (with a much larger
C*), but *fails* under linear drain (γ=1) and under diminishing impact
returns — exactly the kernel-dependence the Scope-Absorption Corollary
declares.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from dlvt.model import (                      # noqa: E402
    DEFAULT_PARAMS, complexity as complexity_baseline,
    impact as impact_baseline, dlvt_system as dlvt_baseline,
)

OUTPUT_DIR = REPO_DIR / 'figures'

rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE SPECIFICATIONS (the experiment — deliberately local)
# ══════════════════════════════════════════════════════════════════════════

def dlvt_gamma1(t, y, p):
    """γ=1: linear complexity drain (handled by the package via p['gamma'])."""
    return dlvt_baseline(t, y, {**p, 'gamma': 1.0})


def dlvt_gamma3(t, y, p):
    """γ=3: cubic complexity drain."""
    return dlvt_baseline(t, y, {**p, 'gamma': 3.0})


def complexity_saturating(C, p, K=50.0):
    """O = O0 + β·C/(1 + C/K) — complexity saturates at O0 + βK."""
    Cc = np.maximum(C, 0)
    return p['O0'] + p['beta'] * Cc / (1.0 + Cc / K)


def dlvt_saturating_complexity(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_saturating(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact_baseline(V, C, O, p)
    return [recovery - drain, p['alpha'] * I - p['mu'] * C]


def impact_diminishing(V, C, O, p):
    """I = C^0.7 · V/(1+φO) — diminishing returns to capital."""
    return (np.power(np.maximum(C, 0), 0.7) * V) / (1.0 + p['phi'] * O)


def dlvt_diminishing_impact(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_baseline(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact_diminishing(V, C, O, p)
    return [recovery - drain, p['alpha'] * I - p['mu'] * C]


# ══════════════════════════════════════════════════════════════════════════
# GENERIC EQUILIBRIUM ANALYSIS (kernel-aware — see module docstring)
# ══════════════════════════════════════════════════════════════════════════

def find_equilibria_generic(p, complexity_func, impact_func,
                            gamma=None, C_max=2000.0, n_scan=16000):
    """Interior equilibria for a spec with V-linear impact kernel.

    dC/dt = 0 with I = g(C, O)·V gives V_c(C) = μ·C / (α·g(C, O)); the
    baseline g = C/(1+φO) recovers V_c = μ(1+φO)/α. Substituting into
    dV/dt = 0 and scanning for sign changes. ``C_max=2000`` avoids the
    fixed-window artifact (Appendix A8): the saturating kernel's equilibrium
    sits at C* ≈ 89, outside the legacy window of 80.
    """
    g = gamma if gamma is not None else p['gamma']

    def V_from_C(Cs):
        Os = complexity_func(Cs, p)
        gain = impact_func(1.0, Cs, Os, p)  # I is linear in V: I = gain·V
        if gain <= 0:
            return np.inf
        return p['mu'] * Cs / (p['alpha'] * gain)

    def residual(Cs):
        if Cs <= 0:
            return 1e10
        Vs = V_from_C(Cs)
        if Vs <= 0 or Vs >= p['Vmax']:
            return 1e10
        Os = complexity_func(Cs, p)
        rec = p['R'] * (1.0 - Vs / p['Vmax'])
        drn = p['delta'] * Os**g * Vs / (Vs + p['eps'])
        return rec - drn

    C_scan = np.linspace(0.01, C_max, n_scan)
    res = np.array([residual(c) for c in C_scan])
    eqs = []
    for i in range(len(res) - 1):
        # Skip brackets touching the 1e10 out-of-domain sentinel: brentq on
        # such a bracket converges to the V=Vmax discontinuity, not a root.
        if abs(res[i]) >= 1e9 or abs(res[i + 1]) >= 1e9:
            continue
        if res[i] * res[i + 1] < 0:
            try:
                Cs = brentq(residual, C_scan[i], C_scan[i + 1])
                Vs = V_from_C(Cs)
                Os = complexity_func(Cs, p)
                if 0 < Vs < p['Vmax'] and Cs > 0 and abs(residual(Cs)) < 1e-6:
                    eqs.append(dict(C=Cs, V=Vs, O=Os))
            except Exception:
                pass
    return eqs


def numerical_jacobian(system_func, V, C, p, h=1e-6):
    """Central-difference Jacobian of the actual spec RHS at (V, C).

    Replaces the earlier (incorrect) use of the baseline analytical Jacobian
    for every alternative kernel.
    """
    J = np.empty((2, 2))
    for j, (dV, dC) in enumerate(((h, 0.0), (0.0, h))):
        f_plus = np.array(system_func(0.0, [V + dV, C + dC], p))
        f_minus = np.array(system_func(0.0, [V - dV, C - dC], p))
        J[:, j] = (f_plus - f_minus) / (2.0 * h)
    eigvals = np.linalg.eigvals(J)
    stable = bool(all(e.real < 0 for e in eigvals))
    return eigvals, 'stable' if stable else 'unstable'


def find_carrying_capacity_generic(p, complexity_func):
    """Carrying capacity: C where Γ = 1 at V = Vmax, per complexity kernel."""
    try:
        Omax = (p['R'] / p['delta']) ** (1.0 / p['gamma'])
        if Omax <= p['O0']:
            return 0.0
        if complexity_func is complexity_baseline:
            return max(0.0, (Omax - p['O0']) / p['beta'])
        if complexity_func is complexity_saturating:
            target = Omax - p['O0']
            if target <= 0:
                return 0.0
            K = 50.0
            if abs(p['beta'] - target / K) < 1e-10:
                return float('inf')
            val = target / (p['beta'] - target / K)
            # Negative root means complexity saturates below Omax: drain can
            # never exceed recovery at Vmax — capacity is unbounded.
            return val if val > 0 else float('inf')
        return 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════

def simulate_generic(system_func, p, V0=8.0, C0=0.5, T=200.0, ms=0.05):
    sol = solve_ivp(system_func, [0, T], [V0, C0], args=(p,),
                    method='RK45', max_step=ms, dense_output=True)
    return sol.t, np.maximum(sol.y[0], 0), np.maximum(sol.y[1], 0)


# ══════════════════════════════════════════════════════════════════════════
# ROBUSTNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def analyze_robustness():
    """Test 5 functional specifications with kernel-aware equilibria."""
    specs = [
        dict(name='Baseline', desc='γ=2, η=1, linear recovery, linear impact',
             system=dlvt_baseline, complexity=complexity_baseline,
             impact=impact_baseline, params=DEFAULT_PARAMS.copy(), gamma=None),
        dict(name='γ=1 (Linear Drain)', desc='Linear complexity drain (vs quadratic)',
             system=dlvt_gamma1, complexity=complexity_baseline,
             impact=impact_baseline,
             params={**DEFAULT_PARAMS, 'gamma': 1.0}, gamma=None),
        dict(name='γ=3 (Cubic Drain)', desc='Cubic complexity drain (vs quadratic)',
             system=dlvt_gamma3, complexity=complexity_baseline,
             impact=impact_baseline,
             params={**DEFAULT_PARAMS, 'gamma': 3.0}, gamma=None),
        dict(name='Saturating Complexity', desc='O = O₀ + βC/(1+C/K), K=50',
             system=dlvt_saturating_complexity, complexity=complexity_saturating,
             impact=impact_baseline, params=DEFAULT_PARAMS.copy(), gamma=None),
        dict(name='Diminishing Impact', desc='I = C^0.7 · V/(1+φO)',
             system=dlvt_diminishing_impact, complexity=complexity_baseline,
             impact=impact_diminishing, params=DEFAULT_PARAMS.copy(), gamma=None),
    ]

    results = []
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS: Alternative Functional Specifications")
    print("=" * 80)

    for spec in specs:
        print(f"\n[{spec['name']}]")
        print(f"  {spec['desc']}")
        p = spec['params']

        eqs = find_equilibria_generic(
            p, spec['complexity'], spec['impact'], gamma=spec['gamma'],
        )
        cc = find_carrying_capacity_generic(p, spec['complexity'])

        has_zombie = has_sustainable = False
        zombie_eq = sustainable_eq = None
        for eq in eqs:
            _, stab = numerical_jacobian(spec['system'], eq['V'], eq['C'], p)
            if stab == 'stable':
                if eq['V'] < 5.0:
                    has_zombie, zombie_eq = True, eq
                else:
                    has_sustainable, sustainable_eq = True, eq

        t, V, C = simulate_generic(spec['system'], p, V0=8.0, C0=0.5, T=200.0)

        print(f"  Carrying capacity: {cc:.2f}")
        print(f"  # Interior equilibria: {len(eqs)}")
        print(f"  Zombie exists: {has_zombie}")
        print(f"  Sustainable exists: {has_sustainable}")
        if zombie_eq:
            print(f"    Zombie eq: V*={zombie_eq['V']:.2f}, C*={zombie_eq['C']:.2f}")
        if sustainable_eq:
            print(f"    Sustainable eq: V*={sustainable_eq['V']:.2f}, "
                  f"C*={sustainable_eq['C']:.2f}")
        print(f"  Final state (t=200): V={V[-1]:.2f}, C={C[-1]:.2f}")

        # Consistency check that the old fixed-window version failed: the
        # simulation endpoint must agree with a found stable equilibrium.
        stable_eqs = [e for e in eqs
                      if numerical_jacobian(spec['system'], e['V'], e['C'], p)[1]
                      == 'stable']
        if stable_eqs:
            nearest = min(stable_eqs,
                          key=lambda e: abs(e['V'] - V[-1]) + abs(e['C'] - C[-1]))
            drift = max(abs(nearest['V'] - V[-1]), abs(nearest['C'] - C[-1]))
            if drift > 0.5:
                print(f"  [WARN] trajectory endpoint is {drift:.2f} away from "
                      f"the nearest stable equilibrium — inspect (slow "
                      f"transient or missed root).")

        results.append(dict(
            spec=spec['name'], description=spec['desc'], carrying_capacity=cc,
            num_equilibria=len(eqs), has_zombie=has_zombie,
            has_sustainable=has_sustainable, zombie_eq=zombie_eq,
            sustainable_eq=sustainable_eq, V_final=V[-1], C_final=C[-1],
            trajectory=(t, V, C), system=spec['system'], params=p,
        ))

    print("\n" + "=" * 80)
    return results


# ══════════════════════════════════════════════════════════════════════════
# OUTPUT TABLE
# ══════════════════════════════════════════════════════════════════════════

def write_robustness_table(results, filename):
    with open(filename, 'w') as f:
        f.write("=" * 110 + "\n")
        f.write("ROBUSTNESS ANALYSIS: Structural Sensitivity of DLVT Core Results\n")
        f.write("=" * 110 + "\n\n")
        f.write("Specification Tested               | C* CC | Zombie | Sustainable"
                " | V* Zombie | V* Sust | V_final | Status\n")
        f.write("-" * 110 + "\n")
        for res in results:
            spec_name = res['spec'][:30].ljust(30)
            cc = res['carrying_capacity']
            cc_str = ("   inf" if np.isinf(cc) else f"{cc:6.2f}").rjust(6)
            zombie_str = ("Yes" if res['has_zombie'] else "No").rjust(7)
            sust_str = ("Yes" if res['has_sustainable'] else "No").rjust(12)
            v_zombie = (f"{res['zombie_eq']['V']:.2f}" if res['zombie_eq']
                        else "---").rjust(9)
            v_sust = (f"{res['sustainable_eq']['V']:.2f}"
                      if res['sustainable_eq'] else "---").rjust(7)
            v_final = f"{res['V_final']:.2f}".rjust(7)
            status = ("OK" if res['has_zombie'] and res['carrying_capacity'] > 0
                      else "CHECK").rjust(6)
            f.write(f"{spec_name} | {cc_str} | {zombie_str} | {sust_str} | "
                    f"{v_zombie} | {v_sust} | {v_final} | {status}\n")

        f.write("\n" + "=" * 110 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("- C* CC: Carrying capacity (C*_max where Γ = 1 at V = Vmax); "
                "'inf' = complexity saturates below the drain ceiling\n")
        f.write("- Zombie: stable equilibrium with V* < 5.0 exists\n")
        f.write("- Sustainable: stable equilibrium with V* ≥ 5.0 exists\n")
        f.write("- V_final: vitality after T=200 simulation from V0=8, C0=0.5\n")
        f.write("- Status 'CHECK' marks specs where the low-vitality equilibrium\n")
        f.write("  does NOT survive — the kernel-dependence that the\n")
        f.write("  Scope-Absorption Corollary declares (γ=1 and diminishing\n")
        f.write("  impact yield sustainable equilibria instead).\n")
        f.write("=" * 110 + "\n")
    print(f"[OK] Robustness table written to {filename}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def fig9_robustness(results):
    """Generate 4-panel robustness figure (baseline + 3 alternatives)."""
    selected = results[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, res in zip(axes, selected):
        t, V, C = res['trajectory']
        ax.plot(t, V, 'b-', lw=2.5, label=r'$V(t)$ — Vitality')
        ax.plot(t, C, 'r--', lw=2, label=r'$C(t)$ — Career Capital')
        if res['zombie_eq']:
            ax.axhline(res['zombie_eq']['V'], color='blue', ls=':', alpha=0.3, lw=1)
        if res['sustainable_eq']:
            ax.axhline(res['sustainable_eq']['V'], color='green', ls=':',
                       alpha=0.4, lw=1)
        ax.axhline(5.0, color='orange', ls='-.', alpha=0.4, lw=1,
                   label='V_strategic=5')
        title = res['spec'] + (" [ZOMBIE]" if res['has_zombie']
                               else " [NO ZOMBIE]")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Level')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, max(max(V) * 1.1, max(C) * 1.1, 12))

    plt.suptitle(r'Robustness: Alternative Functional Specifications',
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'fig9_robustness.{ext}')
    plt.close(fig)
    print("[OK] Fig 9 saved to figures/")


if __name__ == '__main__':
    print("DLVT Structural Robustness Analysis (R1.6, corrected)\n")
    results = analyze_robustness()
    write_robustness_table(results, OUTPUT_DIR / 'robustness_table.txt')
    fig9_robustness(results)
    print("\nRobustness analysis complete.")
