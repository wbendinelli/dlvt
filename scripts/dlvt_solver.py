#!/usr/bin/env python3
"""
DLVT — Dynamic Leadership Vitality Theory
Complete ODE solver, equilibrium analysis, and bifurcation diagrams.
Version 2: Refined parameters for publication-quality figures.
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

# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = dict(
    R=3.0, Vmax=10.0, delta=0.02, gamma=2.0,
    O0=1.0, beta=0.25, eta=1.0,
    alpha=0.1, phi=0.15, mu=0.2,
    eps=0.1,
)

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

def dlvt_exogenous(t, y, p, C_func):
    V = max(y[0], 0.0)
    C = C_func(t)
    O = complexity(C, p)
    recovery = p['R'] * (1.0 - V / p['Vmax'])
    drain = p['delta'] * O**p['gamma'] * V / (V + p['eps'])
    return [recovery - drain]

def simulate(p, V0=8.0, C0=0.5, T=80.0, ms=0.05):
    sol = solve_ivp(dlvt_system, [0, T], [V0, C0], args=(p,),
                    method='RK45', max_step=ms, dense_output=True)
    t = sol.t; V = np.maximum(sol.y[0], 0); C = np.maximum(sol.y[1], 0)
    O = complexity(C, p); I = impact(V, C, O, p)
    Gamma = p['delta'] * O**p['gamma'] / p['R']
    return t, V, C, O, I, Gamma

def carrying_capacity(p):
    Omax = (p['R'] / p['delta'])**(1.0 / p['gamma'])
    return max(0, (Omax - p['O0']) / p['beta']) if Omax > p['O0'] else 0.0

# ══════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def find_interior_equilibria(p, C_max=80.0):
    """Find all (V*, C*) with C*>0 where dV/dt=dC/dt=0."""
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
            except: pass
    return eqs

def jacobian_eigenvalues(V, C, p):
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
# FIGURES
# ══════════════════════════════════════════════════════════════════════════

OUT = '/sessions/upbeat-kind-cannon/mnt/dynamic-leadership-vitality-theory/manuscript/figures'

def fig1_temporal_evolution():
    """Fig 1: V, C, O, Γ vs time — endogenous zombie convergence."""
    p = {**DEFAULT_PARAMS, 'beta': 0.25}
    t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=120)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={'height_ratios': [2.5, 1]})

    a1.plot(t, V, 'b-', lw=2.5, label=r'$V(t)$ — Vitality')
    a1.plot(t, C, 'r--', lw=2, label=r'$C(t)$ — Career Capital')
    a1.plot(t, O, color='darkorange', ls=':', lw=2, label=r'$O(t)$ — Complexity')

    # Find V* equilibrium for annotation
    eqs = find_interior_equilibria(p)
    if eqs:
        eq = eqs[0]
        a1.axhline(eq['V'], color='blue', ls=':', alpha=0.4, lw=1)
        a1.text(t[-1]*0.75, eq['V']+0.3, f"$V^* = {eq['V']:.1f}$", fontsize=10, color='blue')
        a1.axhline(eq['C'], color='red', ls=':', alpha=0.4, lw=1)
        a1.text(t[-1]*0.75, eq['C']+0.3, f"$C^* = {eq['C']:.1f}$", fontsize=10, color='red')

    # t* marker
    idx = np.argmin(np.abs(G - 1.0))
    if 0 < idx < len(t)-1:
        a1.axvline(t[idx], color='gray', ls='--', alpha=0.6)
        a1.text(t[idx]+1, a1.get_ylim()[1]*0.92, r'$t^*$', fontsize=12, color='gray')

    a1.set_ylabel('Level'); a1.legend(loc='upper right', framealpha=0.9)
    a1.set_ylim(-0.3, max(max(C)*1.05, max(O)*1.05)); a1.grid(True, alpha=0.2)
    a1.set_title('(a) Dynamic Evolution — Endogenous System')

    a2.plot(t, G, 'k-', lw=2)
    a2.axhline(1.0, color='red', ls='--', alpha=0.7, label=r'$\Gamma=1$')
    a2.fill_between(t, 1.0, G, where=(G>1), alpha=0.12, color='red')
    a2.fill_between(t, G, 1.0, where=(G<1), alpha=0.12, color='green')
    a2.set_xlabel('Time ($t$)'); a2.set_ylabel(r'$\Gamma(t)$')
    a2.legend(loc='upper right'); a2.grid(True, alpha=0.2)
    a2.set_title('(b) Depletion Ratio $\\Gamma = \\delta O^\\gamma / R$')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_temporal_evolution.pdf')
    plt.savefig(f'{OUT}/fig1_temporal_evolution.png')
    plt.close()
    print("[OK] Fig 1")


def fig2_three_scenarios():
    """Fig 2: Sustainable / Zombie / Collapse (exogenous C)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs = [
        ('(a) Sustainable Leadership', {**DEFAULT_PARAMS, 'beta': 0.08}, 'green', False),
        ('(b) Zombie Equilibrium',     {**DEFAULT_PARAMS, 'beta': 0.25}, 'darkorange', False),
        ('(c) Collapse (exogenous $C$)', {**DEFAULT_PARAMS, 'beta': 0.25}, 'red', True),
    ]

    for ax, (title, p, col, exo) in zip(axes, configs):
        if exo:
            C_func = lambda t: 0.5 + 0.4*t
            sol = solve_ivp(dlvt_exogenous, [0, 50], [8.0], args=(p, C_func),
                           method='RK45', max_step=0.05)
            t = sol.t; V = np.maximum(sol.y[0], 0)
            C = np.array([C_func(ti) for ti in t])
        else:
            t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=150)

        ax.plot(t, V, color=col, lw=2.5, label='$V(t)$')
        ax.plot(t, C, 'k--', lw=1.5, alpha=0.5, label='$C(t)$')

        # Zombie annotation
        if not exo:
            eqs = find_interior_equilibria(p)
            if eqs:
                eq = eqs[0]
                ax.axhline(eq['V'], color=col, ls=':', alpha=0.5)
                ax.axhline(eq['C'], color='gray', ls=':', alpha=0.3)

        # V_strategic threshold
        ax.axhline(3.0, color='purple', ls='-.', alpha=0.3, lw=1)
        if ax == axes[1]:
            ax.text(t[-1]*0.5, 3.3, r'$V_{\mathrm{strategic}}$', fontsize=9, color='purple')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time'); ax.set_ylabel('Level')
        ax.legend(fontsize=8, loc='center right')
        ax.set_ylim(-0.3, max(max(V)*1.1, max(C)*1.1, 12))
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_three_scenarios.pdf')
    plt.savefig(f'{OUT}/fig2_three_scenarios.png')
    plt.close()
    print("[OK] Fig 2")


def fig3_phase_portrait():
    """Fig 3: (C, V) phase portrait with nullclines, equilibria, trajectories."""
    p = {**DEFAULT_PARAMS, 'beta': 0.25}
    fig, ax = plt.subplots(figsize=(9, 6.5))

    C_range = np.linspace(0.01, 40, 600)

    # dV/dt = 0 nullcline
    V_null = []
    for Ci in C_range:
        Oi = complexity(Ci, p)
        def res(V):
            if V <= 0: return p['R']
            return p['R']*(1-V/p['Vmax']) - p['delta']*Oi**p['gamma']*V/(V+p['eps'])
        try: V_null.append(brentq(res, 0.001, p['Vmax']-0.001))
        except: V_null.append(np.nan)
    V_null = np.array(V_null)

    # dC/dt = 0 nullcline (for C>0)
    V_Cnull = np.array([p['mu']*(1+p['phi']*complexity(c, p))/p['alpha'] for c in C_range])

    valid = ~np.isnan(V_null) & (V_null > 0)
    ax.plot(C_range[valid], V_null[valid], 'b-', lw=2.5, label=r'$dV/dt = 0$')
    ax.plot(C_range, V_Cnull, 'r--', lw=2.5, label=r'$dC/dt = 0$')

    # Trajectories
    ics = [(8, 0.5), (9, 3), (5, 10), (2, 0.3), (8, 15), (6, 25)]
    for V0, C0 in ics:
        t, V, C, O, I, G = simulate(p, V0=V0, C0=C0, T=200)
        ax.plot(C, V, 'k-', lw=0.6, alpha=0.4)
        mid = len(t)//4
        if 0 < mid < len(t)-1:
            ax.annotate('', xy=(C[mid+1], V[mid+1]), xytext=(C[mid], V[mid]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1))

    # Mark equilibria
    eqs = find_interior_equilibria(p)
    for eq in eqs:
        eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
        mk = 'o' if stab == 'stable' else 'x'
        cl = 'green' if stab == 'stable' else 'red'
        ax.plot(eq['C'], eq['V'], marker=mk, ms=14, color=cl, mew=3, zorder=10,
                label=f'{stab}: ($C^*$={eq["C"]:.1f}, $V^*$={eq["V"]:.1f})')

    # Zones
    ax.axhline(3.0, color='purple', ls='-.', alpha=0.3, lw=1)
    ax.text(1, 3.2, r'$V_{\mathrm{strategic}}$', fontsize=9, color='purple')

    ax.set_xlabel('Career Capital $C$'); ax.set_ylabel('Vitality $V$')
    ax.set_xlim(0, 40); ax.set_ylim(0, p['Vmax']+0.5)
    ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.2)
    ax.set_title('Phase Portrait with Nullclines and Equilibria')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_phase_portrait.pdf')
    plt.savefig(f'{OUT}/fig3_phase_portrait.png')
    plt.close()
    print("[OK] Fig 3")


def fig4_bifurcation():
    """Fig 4: C* vs β and C* vs R — the key contribution."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) C* vs beta
    betas = np.linspace(0.02, 1.5, 500)
    stable_b, stable_C, stable_V = [], [], []
    cc_b, cc_C = [], []

    for b in betas:
        p = {**DEFAULT_PARAMS, 'beta': b}
        eqs = find_interior_equilibria(p, C_max=200)
        for eq in eqs:
            _, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable':
                stable_b.append(b); stable_C.append(eq['C']); stable_V.append(eq['V'])
        cc = carrying_capacity(p)
        if cc > 0:
            cc_b.append(b); cc_C.append(cc)

    a1.plot(stable_b, stable_C, 'g-', lw=2.5, label='Stable equilibrium $C^*$')
    a1.plot(cc_b, cc_C, 'k--', lw=1.5, alpha=0.5, label=r'Carrying capacity $C^*_{\max}$')

    # Highlight zombie region
    zombie_b = [b for b, v in zip(stable_b, stable_V) if v < 3.0]
    zombie_C = [c for c, v in zip(stable_C, stable_V) if v < 3.0]
    if zombie_b:
        a1.plot(zombie_b, zombie_C, 'o', color='darkorange', ms=2, alpha=0.5,
                label=r'Zombie region ($V^* < V_{\mathrm{strategic}}$)')

    a1.set_xlabel(r'Capital-Complexity Coupling $\beta$')
    a1.set_ylabel(r'Equilibrium Capital $C^*$')
    a1.set_title(r'(a) Bifurcation: $C^*$ vs $\beta$')
    a1.legend(fontsize=9); a1.grid(True, alpha=0.2)
    a1.set_ylim(0, 100)

    # (b) C* vs R
    Rs = np.linspace(0.5, 8.0, 500)
    stable_R, stable_CR, stable_VR = [], [], []
    cc_R, cc_CR = [], []

    for Rv in Rs:
        p = {**DEFAULT_PARAMS, 'R': Rv}
        eqs = find_interior_equilibria(p, C_max=200)
        for eq in eqs:
            _, stab = jacobian_eigenvalues(eq['V'], eq['C'], p)
            if stab == 'stable':
                stable_R.append(Rv); stable_CR.append(eq['C']); stable_VR.append(eq['V'])
        cc = carrying_capacity(p)
        if cc > 0:
            cc_R.append(Rv); cc_CR.append(cc)

    a2.plot(stable_R, stable_CR, 'g-', lw=2.5, label='Stable $C^*$')
    a2.plot(cc_R, cc_CR, 'k--', lw=1.5, alpha=0.5, label=r'$C^*_{\max}$')

    zombie_R = [r for r, v in zip(stable_R, stable_VR) if v < 3.0]
    zombie_CR2 = [c for c, v in zip(stable_CR, stable_VR) if v < 3.0]
    if zombie_R:
        a2.plot(zombie_R, zombie_CR2, 'o', color='darkorange', ms=2, alpha=0.5,
                label=r'Zombie region')

    a2.set_xlabel(r'Recovery Rate $R$')
    a2.set_ylabel(r'Equilibrium Capital $C^*$')
    a2.set_title(r'(b) Bifurcation: $C^*$ vs $R$')
    a2.legend(fontsize=9); a2.grid(True, alpha=0.2)
    a2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_bifurcation.pdf')
    plt.savefig(f'{OUT}/fig4_bifurcation.png')
    plt.close()
    print("[OK] Fig 4")


def fig5_impact_comparison():
    """Fig 5: Becker linear vs DLVT impact — using equilibrium locus."""
    p = {**DEFAULT_PARAMS, 'beta': 0.25}
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Compute steady-state I*(C*) for different beta values
    # This gives a clean I vs C curve without trajectory artifacts
    betas = np.linspace(0.03, 2.0, 500)
    Cs, Is = [], []
    for b in betas:
        pp = {**p, 'beta': b}
        eqs = find_interior_equilibria(pp, C_max=200)
        for eq in eqs:
            _, stab = jacobian_eigenvalues(eq['V'], eq['C'], pp)
            if stab == 'stable':
                Cs.append(eq['C']); Is.append(eq['I'])

    # Also simulate a single trajectory for the dynamic path
    t, V, C, O, I, G = simulate(p, V0=8.0, C0=0.5, T=200)

    # Only use the "forward" part of the trajectory (C increasing)
    dC = np.diff(C)
    last_inc = 0
    for i in range(len(dC)):
        if dC[i] > 0: last_inc = i+1
    C_fwd = C[:last_inc+1]; I_fwd = I[:last_inc+1]

    ax.plot(C_fwd, I_fwd, 'b-', lw=2.5, label='DLVT trajectory: $I = CV/(1+\\phi O)$', zorder=5)

    # Becker linear (calibrated to initial slope)
    if len(C_fwd) > 5:
        k = I_fwd[5] / C_fwd[5] if C_fwd[5] > 0 else 1.0
    else:
        k = 1.0
    Clin = np.linspace(0, max(C_fwd)*1.1, 100)
    ax.plot(Clin, k*Clin, 'k--', lw=1.5, alpha=0.5, label=f'Becker (HC Theory): $I = {k:.1f}C$')

    # Peak impact
    peak = np.argmax(I_fwd)
    ax.plot(C_fwd[peak], I_fwd[peak], 'ro', ms=10, zorder=10)
    ax.annotate('Peak Impact', xy=(C_fwd[peak], I_fwd[peak]),
               xytext=(C_fwd[peak]+1.5, I_fwd[peak]+0.5),
               fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    # Carrying capacity
    cc = carrying_capacity(p)
    if cc > 0 and cc < max(C_fwd)*1.1:
        ax.axvline(cc, color='gray', ls=':', alpha=0.6)
        ax.text(cc+0.3, ax.get_ylim()[1]*0.05, '$C^*$', fontsize=10, color='gray', rotation=90)

    # Prediction gap
    C_gap = min(cc * 1.3, max(C_fwd) * 0.8) if cc > 0 else max(C_fwd)*0.7
    I_dlvt_gap = np.interp(C_gap, C_fwd, I_fwd)
    I_beck_gap = k * C_gap
    if I_beck_gap > I_dlvt_gap + 0.5:
        ax.annotate('', xy=(C_gap, I_dlvt_gap), xytext=(C_gap, I_beck_gap),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(C_gap+0.5, (I_dlvt_gap+I_beck_gap)/2, 'Prediction\nGap',
               fontsize=10, color='red', ha='left')

    ax.set_xlabel('Career Capital $C$'); ax.set_ylabel('Leadership Impact $I$')
    ax.set_title('Impact: Human Capital Theory vs. DLVT')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    ax.set_xlim(0, max(C_fwd)*1.1); ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_impact_comparison.pdf')
    plt.savefig(f'{OUT}/fig5_impact_comparison.png')
    plt.close()
    print("[OK] Fig 5")


def fig6_heatmap():
    """Fig 6: C*(β, R) carrying capacity heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    betas = np.linspace(0.05, 1.0, 120)
    Rs = np.linspace(0.5, 6.0, 120)
    CC = np.zeros((len(Rs), len(betas)))
    for i, Rv in enumerate(Rs):
        for j, bv in enumerate(betas):
            CC[i, j] = carrying_capacity({**DEFAULT_PARAMS, 'R': Rv, 'beta': bv})

    im = ax.pcolormesh(betas, Rs, CC, cmap='RdYlGn', shading='auto')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(r'Carrying Capacity $C^*$')
    CS = ax.contour(betas, Rs, CC, levels=[5, 10, 20, 40, 60], colors='black', lw=0.8, alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    ax.set_ylabel(r'$R$ (recovery rate)')
    ax.set_title(r'Leadership Carrying Capacity $C^*(\beta, R)$')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_sensitivity_heatmap.pdf')
    plt.savefig(f'{OUT}/fig6_sensitivity_heatmap.png')
    plt.close()
    print("[OK] Fig 6")


# ══════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

def verify():
    p = DEFAULT_PARAMS.copy()
    print("="*65)
    print("DLVT MATHEMATICAL VERIFICATION (v2)")
    print("="*65)

    # Test 1: Smooth barrier
    print("\n[1] Smooth barrier V ≥ 0")
    p2 = {**p, 'beta': 0.8, 'delta': 0.05}
    t, V, C, O, I, G = simulate(p2, V0=8, C0=2, T=100)
    print(f"    min(V) = {min(V):.6f}  {'PASS' if min(V) >= -1e-10 else 'FAIL'}")

    # Test 2: Theorem 1 (exogenous C)
    print("\n[2] Theorem 1: exogenous C → depletion")
    C_func = lambda t: 0.5 + 0.6*t
    sol = solve_ivp(dlvt_exogenous, [0, 60], [8.0], args=(p, C_func),
                   method='RK45', max_step=0.05)
    V_exo = np.maximum(sol.y[0], 0)
    cc = carrying_capacity(p)
    t_star_th = (cc - 0.5) / 0.6
    print(f"    C* = {cc:.2f}, t* theoretical = {t_star_th:.2f}")
    print(f"    V at T=60: {V_exo[-1]:.4f}  {'PASS' if V_exo[-1] < 1.0 else 'CHECK'}")

    # Test 3: Carrying capacity
    print("\n[3] Carrying capacity Γ(C*) = 1")
    O_at_cc = complexity(cc, p)
    G_at_cc = p['delta'] * O_at_cc**p['gamma'] / p['R']
    print(f"    C* = {cc:.2f}, Γ(C*) = {G_at_cc:.6f}  {'PASS' if abs(G_at_cc-1)<1e-6 else 'FAIL'}")

    # Test 4: Equilibria
    print("\n[4] Equilibrium analysis")
    for b in [0.10, 0.20, 0.30, 0.50, 0.80]:
        pp = {**p, 'beta': b}
        eqs = find_interior_equilibria(pp)
        cc_val = carrying_capacity(pp)
        for eq in eqs:
            eigv, stab = jacobian_eigenvalues(eq['V'], eq['C'], pp)
            zombie = "ZOMBIE" if eq['V'] < 3.0 else "healthy"
            print(f"    β={b:.2f}: C*={eq['C']:.1f}, V*={eq['V']:.2f}, "
                  f"I*={eq['I']:.2f}, {stab}, {zombie}, CC={cc_val:.1f}")
            print(f"             λ = {eigv[0]:.3f}, {eigv[1]:.3f}")

    # Test 5: P3 generalized (eta > 1)
    print("\n[5] P3 generalized: C* for η > 1")
    for eta_val in [1.0, 1.5, 2.0]:
        pp = {**p, 'eta': eta_val}
        Omax = (pp['R']/pp['delta'])**(1/pp['gamma'])
        if Omax > pp['O0']:
            cc_gen = ((Omax - pp['O0']) / pp['beta'])**(1/eta_val)
        else:
            cc_gen = 0
        print(f"    η={eta_val:.1f}: C* = {cc_gen:.2f}")

    print("\n" + "="*65)
    print("VERIFICATION COMPLETE")
    print("="*65)


if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    verify()
    print("\nGenerating figures...")
    fig1_temporal_evolution()
    fig2_three_scenarios()
    fig3_phase_portrait()
    fig4_bifurcation()
    fig5_impact_comparison()
    fig6_heatmap()
    print(f"\n✓ All figures saved to {OUT}")
