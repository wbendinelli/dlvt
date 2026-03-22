#!/usr/bin/env python3
"""
DLVT — Bifurcation Analysis with Hysteresis (R1.5)
Scans β parameter, finds equilibria, classifies stability, detects hysteresis.
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
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def complexity(C, p):
    return p['O0'] + p['beta'] * np.power(np.maximum(C, 0), p['eta'])

def impact(V, C, O, p):
    return C * V / (1.0 + p['phi'] * O)

def dlvt_system(t, y, p):
    V, C = max(y[0], 0.0), max(y[1], 0.0)
    O = complexity(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    I = impact(V, C, O, p)
    dVdt = recovery - drain
    dCdt = p['alpha'] * I - p['mu'] * C
    return [dVdt, dCdt]

def simulate(p, V0=8.0, C0=0.5, T=80.0, ms=0.05):
    sol = solve_ivp(dlvt_system, [0, T], [V0, C0], args=(p,),
                    method='RK45', max_step=ms, dense_output=True)
    t = sol.t; V = np.maximum(sol.y[0], 0); C = np.maximum(sol.y[1], 0)
    O = complexity(C, p); I = impact(V, C, O, p)
    return t, V, C, O, I

def find_interior_equilibria(p, C_max=80.0):
    """Find all interior equilibria (V*, C*)."""
    def V_from_C(Cs):
        Os = p['O0'] + p['beta'] * Cs**p['eta']
        return p['mu'] * (1.0 + p['phi'] * Os) / p['alpha']

    def residual(Cs):
        if Cs <= 0: return 1e10
        Vs = V_from_C(Cs)
        if Vs <= 0 or Vs >= p['Vmax']: return 1e10
        Os = p['O0'] + p['beta'] * Cs**p['eta']
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
                Vs = V_from_C(Cs); Os = complexity(Cs, p)
                if 0 < Vs < p['Vmax'] and Cs > 0:
                    eqs.append(dict(C=Cs, V=Vs, O=Os, I=impact(Vs, Cs, Os, p)))
            except:
                pass
    return eqs

def jacobian_eigenvalues(V, C, p):
    """Compute eigenvalues and stability."""
    O = complexity(C, p); eps = p['eps']
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
# BIFURCATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def bifurcation_scan(beta_min=0.02, beta_max=1.5, n_points=1000):
    """Scan β parameter and find all equilibria with stability classification."""
    betas = np.linspace(beta_min, beta_max, n_points)

    stable_branch = {'beta': [], 'V': [], 'C': []}
    unstable_branch = {'beta': [], 'V': [], 'C': []}

    print(f"[Bifurcation] Scanning β from {beta_min} to {beta_max} ({n_points} points)...")

    for i, b in enumerate(betas):
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{n_points}...")

        p = {**DEFAULT_PARAMS, 'beta': b}
        eqs = find_interior_equilibria(p)

        for eq in eqs:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable':
                stable_branch['beta'].append(b)
                stable_branch['V'].append(eq['V'])
                stable_branch['C'].append(eq['C'])
            else:
                unstable_branch['beta'].append(b)
                unstable_branch['V'].append(eq['V'])
                unstable_branch['C'].append(eq['C'])

    return betas, stable_branch, unstable_branch

def find_critical_beta(V_strategic=5.0):
    """Find β_crit where V* crosses V_strategic."""
    betas = np.linspace(0.02, 1.5, 1000)

    for b in betas:
        p = {**DEFAULT_PARAMS, 'beta': b}
        eqs = find_interior_equilibria(p)

        for eq in eqs:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable' and eq['V'] < V_strategic:
                return b

    return None

def detect_hysteresis():
    """
    Detect hysteresis by comparing forward and backward sweeps.
    Forward: start sustainable, gradually increase β until jump to zombie.
    Backward: start zombie, gradually decrease β until jump back to sustainable.
    """

    print("[Hysteresis] Detecting hysteresis via forward-backward sweep...")

    # Forward sweep: β increasing
    print("  Forward sweep (β increasing from low to high)...")
    betas_fwd = np.linspace(0.02, 1.5, 500)
    beta_forward_jump = None
    prev_state = 'sustainable'

    for b in betas_fwd:
        p = {**DEFAULT_PARAMS, 'beta': b}
        eqs = find_interior_equilibria(p)

        # Check if system jumps from sustainable (V* > 5) to zombie (V* < 5)
        has_sustainable = False
        has_zombie = False

        for eq in eqs:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable':
                if eq['V'] >= 5.0:
                    has_sustainable = True
                if eq['V'] < 5.0:
                    has_zombie = True

        # Detect jump in forward direction
        if prev_state == 'sustainable' and has_zombie and not has_sustainable:
            beta_forward_jump = b
            print(f"    → Forward jump detected at β_fwd ≈ {b:.4f}")
            prev_state = 'zombie'
        elif has_sustainable:
            prev_state = 'sustainable'
        elif has_zombie:
            prev_state = 'zombie'

    # Backward sweep: β decreasing
    print("  Backward sweep (β decreasing from high to low)...")
    betas_bwd = np.linspace(1.5, 0.02, 500)
    beta_backward_jump = None
    prev_state = 'zombie'

    for b in betas_bwd:
        p = {**DEFAULT_PARAMS, 'beta': b}
        eqs = find_interior_equilibria(p)

        has_sustainable = False
        has_zombie = False

        for eq in eqs:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable':
                if eq['V'] >= 5.0:
                    has_sustainable = True
                if eq['V'] < 5.0:
                    has_zombie = True

        # Detect jump in backward direction
        if prev_state == 'zombie' and has_sustainable and not has_zombie:
            beta_backward_jump = b
            print(f"    → Backward jump detected at β_bwd ≈ {b:.4f}")
            prev_state = 'sustainable'
        elif has_zombie:
            prev_state = 'zombie'
        elif has_sustainable:
            prev_state = 'sustainable'

    hysteresis_width = None
    if beta_forward_jump is not None and beta_backward_jump is not None:
        hysteresis_width = beta_forward_jump - beta_backward_jump
        if hysteresis_width > 0:
            print(f"  ✓ Hysteresis detected: h = {hysteresis_width:.4f}")
        else:
            print(f"  Note: Forward jump at {beta_forward_jump:.4f}, backward at {beta_backward_jump:.4f}")
    else:
        print(f"  Note: Incomplete hysteresis loop detected (fwd={beta_forward_jump}, bwd={beta_backward_jump})")

    return beta_forward_jump, beta_backward_jump, hysteresis_width

# ══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════

def fig8_bifurcation_hysteresis():
    """Generate bifurcation diagram with hysteresis."""

    # Perform bifurcation scan
    betas, stable, unstable = bifurcation_scan(beta_min=0.02, beta_max=1.5, n_points=1000)

    # Find critical beta and hysteresis
    beta_crit = find_critical_beta(V_strategic=5.0)
    beta_fwd, beta_bwd, h_width = detect_hysteresis()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot stable branch (solid)
    if stable['V']:
        ax.plot(stable['beta'], stable['V'], 'g-', lw=2.5, label='Stable equilibrium (V*)', zorder=5)

    # Plot unstable branch (dashed)
    if unstable['V']:
        ax.plot(unstable['beta'], unstable['V'], 'r--', lw=2, label='Unstable equilibrium (V*)', zorder=4)

    # Mark critical beta
    if beta_crit is not None:
        ax.axvline(beta_crit, color='purple', ls=':', lw=2, alpha=0.6, label=f'β_crit ≈ {beta_crit:.3f}')
        ax.plot([beta_crit]*2, [0, 10.5], 'purple', ls=':', lw=1, alpha=0.3)

    # Strategic threshold
    V_strategic = 5.0
    ax.axhline(V_strategic, color='orange', ls='-.', alpha=0.5, lw=1.5,
               label=f'V_strategic = {V_strategic}')

    # Hysteresis loop annotation
    if beta_fwd is not None and beta_bwd is not None:
        ax.axvline(beta_fwd, color='darkred', ls='--', lw=1.5, alpha=0.4)
        ax.axvline(beta_bwd, color='darkblue', ls='--', lw=1.5, alpha=0.4)

        # Add hysteresis region
        ax.axvspan(beta_bwd, beta_fwd, alpha=0.1, color='yellow', label='Hysteresis region')

        # Annotate hysteresis width
        mid_beta = (beta_fwd + beta_bwd) / 2
        ax.text(mid_beta, 1.0, f'h={h_width:.4f}', fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Labeling
    ax.set_xlabel(r'Capital-Complexity Coupling $\beta$', fontsize=13)
    ax.set_ylabel(r'Vitality at Equilibrium $V^*$', fontsize=13)
    ax.set_title(r'Bifurcation Analysis: $V^*(\beta)$ with Hysteresis', fontsize=14, fontweight='bold')

    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

    # Add zones
    ax.fill_between([0.0, 1.5], [V_strategic, V_strategic], [10.5, 10.5],
                     alpha=0.05, color='green', label='_nolegend_')
    ax.fill_between([0.0, 1.5], [0, 0], [V_strategic, V_strategic],
                     alpha=0.05, color='red', label='_nolegend_')

    ax.text(1.35, 8.0, 'Sustainable\nLeadership', fontsize=11, ha='right', color='darkgreen', alpha=0.7)
    ax.text(1.35, 2.0, 'Zombie\nLeadership', fontsize=11, ha='right', color='darkred', alpha=0.7)

    plt.tight_layout()
    plt.savefig('/tmp/dlvt-work/manuscript/figures/fig8_bifurcation_hysteresis.pdf')
    plt.savefig('/tmp/dlvt-work/manuscript/figures/fig8_bifurcation_hysteresis.png')
    plt.close()

    print("[OK] Fig 8 saved to manuscript/figures/")

    # Print summary
    print("\n" + "="*70)
    print("BIFURCATION ANALYSIS SUMMARY")
    print("="*70)
    print(f"β range: 0.02 to 1.5")
    print(f"Critical β (V* crosses {V_strategic}): {beta_crit:.4f}" if beta_crit else "Not found")
    print(f"Forward jump at β_fwd: {beta_fwd:.4f}" if beta_fwd else "Not detected")
    print(f"Backward jump at β_bwd: {beta_bwd:.4f}" if beta_bwd else "Not detected")
    print(f"Hysteresis width h: {h_width:.4f}" if h_width else "No hysteresis detected")
    print("="*70 + "\n")

    return beta_crit, beta_fwd, beta_bwd, h_width


if __name__ == '__main__':
    import os
    os.makedirs('/tmp/dlvt-work/manuscript/figures', exist_ok=True)

    print("DLVT Bifurcation Analysis with Hysteresis (R1.5)\n")

    beta_crit, beta_fwd, beta_bwd, h_width = fig8_bifurcation_hysteresis()

    print(f"✓ Bifurcation diagram with hysteresis analysis complete!")
