#!/usr/bin/env python3
"""
DLVT — Structural Robustness / Sensitivity Analysis (R1.6)
Tests alternative functional specifications against baseline.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

DEFAULT_PARAMS = dict(
    R=3.0, Vmax=10.0, delta=0.02, gamma=2.0,
    O0=1.0, beta=0.25, eta=1.0,
    alpha=0.1, phi=0.15, mu=0.2,
    eps=0.1,
)

# ══════════════════════════════════════════════════════════════════════════
# BASELINE MODEL (γ=2, η=1)
# ══════════════════════════════════════════════════════════════════════════

def complexity_baseline(C, p):
    return p['O0'] + p['beta'] * np.power(np.maximum(C, 0), p['eta'])

def impact_baseline(V, C, O, p):
    return C * V / (1.0 + p['phi'] * O)

def dlvt_baseline(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_baseline(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact_baseline(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

# ══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE SPEC 1: γ=1 (Linear Drain)
# ══════════════════════════════════════════════════════════════════════════

def dlvt_gamma1(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_baseline(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O * V / (V + p['eps'])  # γ=1: linear
    I = impact_baseline(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

# ══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE SPEC 2: γ=3 (Cubic Drain)
# ══════════════════════════════════════════════════════════════════════════

def dlvt_gamma3(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_baseline(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**3 * V / (V + p['eps'])  # γ=3: cubic
    I = impact_baseline(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

# ══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE SPEC 3: Saturating Complexity
# ══════════════════════════════════════════════════════════════════════════

def complexity_saturating(C, p, K=50.0):
    """O = O0 + β * C / (1 + C/K)"""
    return p['O0'] + p['beta'] * np.maximum(C, 0) / (1.0 + np.maximum(C, 0) / K)

def dlvt_saturating_complexity(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_saturating(C, p, K=50.0)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact_baseline(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

# ══════════════════════════════════════════════════════════════════════════
# ALTERNATIVE SPEC 4: Diminishing Returns in Impact
# ══════════════════════════════════════════════════════════════════════════

def impact_diminishing(V, C, O, p):
    """I = C^0.7 * V / (1 + φO)"""
    return (np.power(np.maximum(C, 0), 0.7) * V) / (1.0 + p['phi'] * O)

def dlvt_diminishing_impact(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity_baseline(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact_diminishing(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

# ══════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def find_equilibria_generic(p, system_func, complexity_func, C_max=80.0):
    """
    Find equilibria for generic system.
    system_func: ODE function
    complexity_func: function that takes (C, p) and returns O
    """
    def V_from_C(Cs):
        Os = complexity_func(Cs, p)
        return p['mu'] * (1.0 + p['phi'] * Os) / p['alpha']

    def residual(Cs):
        if Cs <= 0:
            return 1e10
        Vs = V_from_C(Cs)
        if Vs <= 0 or Vs >= p['Vmax']:
            return 1e10
        Os = complexity_func(Cs, p)
        rec = p['R'] * (1.0 - Vs / p['Vmax'])
        drn = p['delta'] * Os**p['gamma'] * Vs / (Vs + p['eps'])
        return rec - drn

    C_scan = np.linspace(0.01, C_max, 8000)
    res = np.array([residual(c) for c in C_scan])
    eqs = []

    for i in range(len(res) - 1):
        if res[i] * res[i+1] < 0:
            try:
                Cs = brentq(residual, C_scan[i], C_scan[i+1])
                Vs = V_from_C(Cs)
                Os = complexity_func(Cs, p)
                if 0 < Vs < p['Vmax'] and Cs > 0:
                    eqs.append(dict(C=Cs, V=Vs, O=Os))
            except:
                pass

    return eqs

def find_carrying_capacity_generic(p, complexity_func):
    """Find carrying capacity: C* where Γ(C*) = 1."""
    try:
        Omax = (p['R'] / p['delta'])**(1.0 / p['gamma'])
        if Omax <= p['O0']:
            return 0.0

        # For baseline: (O0 + β*C) = Omax
        if complexity_func == complexity_baseline:
            return max(0, (Omax - p['O0']) / p['beta'])

        # For saturating: O0 + β*C/(1+C/K) = Omax
        if complexity_func == complexity_saturating:
            # Solve: β*C/(1+C/K) = Omax - O0
            target = Omax - p['O0']
            if target <= 0:
                return 0.0
            K = 50.0
            # β*C/(1+C/K) = target
            # β*C = target*(1+C/K) = target + target*C/K
            # β*C - target*C/K = target
            # C*(β - target/K) = target
            if abs(p['beta'] - target/K) < 1e-10:
                return 0.0
            return target / (p['beta'] - target/K)

        return 0.0
    except:
        return 0.0

def jacobian_eigenvalues_baseline(V, C, p):
    """Compute eigenvalues for baseline model."""
    O = complexity_baseline(C, p)
    eps = p['eps']
    dOdC = p['beta'] * p['eta'] * max(C, 1e-10)**(p['eta']-1)
    J = np.array([
        [-p['R']/p['Vmax'] - p['delta']*O**p['gamma']*eps/(V+eps)**2,
         -p['delta']*p['gamma']*O**(p['gamma']-1)*dOdC*V/(V+eps)],
        [p['alpha']*C/(1+p['phi']*O),
         p['alpha']*V/(1+p['phi']*O) - p['alpha']*C*V*p['phi']*dOdC/(1+p['phi']*O)**2 - p['mu']]
    ])
    eigvals = np.linalg.eigvals(J)
    stable = all(e.real < 0 for e in eigvals)
    return eigvals, 'stable' if stable else 'unstable'

# ══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════

def simulate_generic(system_func, p, V0=8.0, C0=0.5, T=200.0, ms=0.05):
    """Simulate generic system."""
    sol = solve_ivp(system_func, [0, T], [V0, C0], args=(p,),
                    method='RK45', max_step=ms, dense_output=True)
    t = sol.t
    V = np.maximum(sol.y[0], 0)
    C = np.maximum(sol.y[1], 0)
    return t, V, C

# ══════════════════════════════════════════════════════════════════════════
# ROBUSTNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def analyze_robustness():
    """Test 5 functional specifications."""

    specs = [
        {
            'name': 'Baseline',
            'desc': 'γ=2, η=1, linear recovery, linear impact',
            'system': dlvt_baseline,
            'complexity': complexity_baseline,
            'params': DEFAULT_PARAMS.copy(),
        },
        {
            'name': 'γ=1 (Linear Drain)',
            'desc': 'Linear complexity drain (vs quadratic)',
            'system': dlvt_gamma1,
            'complexity': complexity_baseline,
            'params': {**DEFAULT_PARAMS, 'gamma': 1.0},
        },
        {
            'name': 'γ=3 (Cubic Drain)',
            'desc': 'Cubic complexity drain (vs quadratic)',
            'system': dlvt_gamma3,
            'complexity': complexity_baseline,
            'params': {**DEFAULT_PARAMS, 'gamma': 3.0},
        },
        {
            'name': 'Saturating Complexity',
            'desc': 'O = O₀ + βC/(1+C/K), K=50',
            'system': dlvt_saturating_complexity,
            'complexity': complexity_saturating,
            'params': DEFAULT_PARAMS.copy(),
        },
        {
            'name': 'Diminishing Impact',
            'desc': 'I = C^0.7 · V/(1+φO)',
            'system': dlvt_diminishing_impact,
            'complexity': complexity_baseline,
            'params': DEFAULT_PARAMS.copy(),
        },
    ]

    results = []

    print("="*80)
    print("ROBUSTNESS ANALYSIS: Alternative Functional Specifications")
    print("="*80)

    for spec in specs:
        print(f"\n[{spec['name']}]")
        print(f"  {spec['desc']}")

        p = spec['params']

        # Find equilibria
        eqs = find_equilibria_generic(p, spec['system'], spec['complexity'])
        cc = find_carrying_capacity_generic(p, spec['complexity'])

        # Classify equilibria
        has_zombie = False
        zombie_eq = None
        has_sustainable = False
        sustainable_eq = None

        for eq in eqs:
            eigv, stab = jacobian_eigenvalues_baseline(eq['V'], eq['C'], p)
            if stab == 'stable':
                if eq['V'] < 5.0:
                    has_zombie = True
                    zombie_eq = eq
                else:
                    has_sustainable = True
                    sustainable_eq = eq

        # Simulate
        t, V, C = simulate_generic(spec['system'], p, V0=8.0, C0=0.5, T=200.0)
        V_final = V[-1]
        C_final = C[-1]

        print(f"  Carrying capacity: {cc:.2f}")
        print(f"  # Interior equilibria: {len(eqs)}")
        print(f"  Zombie exists: {has_zombie}")
        print(f"  Sustainable exists: {has_sustainable}")

        if zombie_eq:
            print(f"    Zombie eq: V*={zombie_eq['V']:.2f}, C*={zombie_eq['C']:.2f}")
        if sustainable_eq:
            print(f"    Sustainable eq: V*={sustainable_eq['V']:.2f}, C*={sustainable_eq['C']:.2f}")

        print(f"  Final state (t=200): V={V_final:.2f}, C={C_final:.2f}")

        results.append({
            'spec': spec['name'],
            'description': spec['desc'],
            'carrying_capacity': cc,
            'num_equilibria': len(eqs),
            'has_zombie': has_zombie,
            'has_sustainable': has_sustainable,
            'zombie_eq': zombie_eq,
            'sustainable_eq': sustainable_eq,
            'V_final': V_final,
            'C_final': C_final,
            'trajectory': (t, V, C),
            'system': spec['system'],
            'params': p,
        })

    print("\n" + "="*80)
    return results

# ══════════════════════════════════════════════════════════════════════════
# OUTPUT TABLE
# ══════════════════════════════════════════════════════════════════════════

def write_robustness_table(results, filename):
    """Write summary table."""

    with open(filename, 'w') as f:
        f.write("="*110 + "\n")
        f.write("ROBUSTNESS ANALYSIS: Structural Sensitivity of DLVT Core Results\n")
        f.write("="*110 + "\n\n")

        f.write("Specification Tested               | C* CC | Zombie | Sustainable | V* Zombie | V* Sust | V_final | Status\n")
        f.write("-"*110 + "\n")

        for res in results:
            spec_name = res['spec'][:30].ljust(30)
            cc_str = f"{res['carrying_capacity']:6.2f}".rjust(6)
            zombie_str = "Yes" if res['has_zombie'] else "No"
            zombie_str = zombie_str.rjust(7)
            sust_str = "Yes" if res['has_sustainable'] else "No"
            sust_str = sust_str.rjust(12)

            v_zombie = f"{res['zombie_eq']['V']:.2f}" if res['zombie_eq'] else "---"
            v_zombie = v_zombie.rjust(9)

            v_sust = f"{res['sustainable_eq']['V']:.2f}" if res['sustainable_eq'] else "---"
            v_sust = v_sust.rjust(7)

            v_final = f"{res['V_final']:.2f}".rjust(7)

            status = "OK" if res['has_zombie'] and res['carrying_capacity'] > 0 else "CHECK"
            status = status.rjust(6)

            line = f"{spec_name} | {cc_str} | {zombie_str} | {sust_str} | {v_zombie} | {v_sust} | {v_final} | {status}\n"
            f.write(line)

        f.write("\n" + "="*110 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("- C* CC: Carrying capacity (C*_max where Γ = 1)\n")
        f.write("- Zombie: Whether zombie equilibrium (V* < 5.0) exists and is stable\n")
        f.write("- Sustainable: Whether sustainable equilibrium (V* ≥ 5.0) exists and is stable\n")
        f.write("- V* Zombie: Vitality at zombie equilibrium\n")
        f.write("- V* Sust: Vitality at sustainable equilibrium\n")
        f.write("- V_final: Vitality after T=200 simulation from V0=8, C0=0.5\n")
        f.write("- Status: 'OK' if zombie exists and C* > 0 (core results hold); 'CHECK' otherwise\n")
        f.write("="*110 + "\n")

    print(f"[OK] Robustness table written to {filename}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def fig9_robustness(results):
    """Generate 4-panel robustness figure (baseline + 3 alternatives)."""

    # Select first 4 specs for figure
    selected = results[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (ax, res) in enumerate(zip(axes, selected)):
        t, V, C = res['trajectory']

        ax.plot(t, V, 'b-', lw=2.5, label=r'$V(t)$ — Vitality')
        ax.plot(t, C, 'r--', lw=2, label=r'$C(t)$ — Career Capital')

        # Mark equilibrium if it exists
        if res['zombie_eq']:
            ax.axhline(res['zombie_eq']['V'], color='blue', ls=':', alpha=0.3, lw=1)

        # Threshold
        ax.axhline(5.0, color='orange', ls='-.', alpha=0.4, lw=1, label='V_strategic=5')

        title = res['spec']
        if res['has_zombie']:
            title += " [ZOMBIE]"
        else:
            title += " [NO ZOMBIE]"

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Level')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, max(max(V)*1.1, max(C)*1.1, 12))

    plt.suptitle(r'Robustness: Alternative Functional Specifications', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('/tmp/dlvt-work/manuscript/figures/fig9_robustness.pdf')
    plt.savefig('/tmp/dlvt-work/manuscript/figures/fig9_robustness.png')
    plt.close()

    print("[OK] Fig 9 saved to manuscript/figures/")


if __name__ == '__main__':
    import os
    os.makedirs('/tmp/dlvt-work/manuscript/figures', exist_ok=True)

    print("DLVT Structural Robustness Analysis (R1.6)\n")

    results = analyze_robustness()

    # Write table
    write_robustness_table(results, '/tmp/dlvt-work/manuscript/figures/robustness_table.txt')

    # Generate figure
    fig9_robustness(results)

    print("\n✓ Robustness analysis complete!")
