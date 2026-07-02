"""
dlvt.figures
============
Publication figure generation for the DLVT paper (Figures 1–7).

All figures are saved as both PDF (vector, for LaTeX) and PNG (raster, 300 dpi).

Figure list
-----------
fig1 : Temporal evolution of V, C, O and depletion ratio Γ
fig2 : Three outcome scenarios — sustainable, low-vitality, collapse
fig3 : Phase portrait with nullclines, trajectories, and equilibria
fig4 : Bifurcation diagrams — C* and V* vs β, and C* vs R
fig5 : Leadership impact — DLVT vs Human Capital Theory (Becker)
fig6 : Carrying capacity heatmap C*(β, R)
fig7 : Leadership regime map in (β, δ) parameter space

Usage
-----
    from dlvt.figures import generate_all
    generate_all(output_dir='figures/')

    # Or individual figures:
    from dlvt.figures import fig1
    fig1(output_dir='figures/')
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .model    import make_params, complexity, impact, simulate, dlvt_system
from .analysis import (carrying_capacity, find_interior_equilibria,
                       jacobian_eigenvalues, V_STRATEGIC_FRACTION)

# ── Matplotlib style ──────────────────────────────────────────────────────────

rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def _save(fig, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, name)
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png')
    plt.close(fig)
    print(f'[OK] {name}  →  {output_dir}/')


def _v_strategic(p):
    return V_STRATEGIC_FRACTION * p['Vmax']


# ── Figure 1: Temporal Evolution ─────────────────────────────────────────────

def fig1(output_dir='figures/', p=None):
    """Temporal evolution of V, C, O and depletion ratio Γ."""
    if p is None:
        p = make_params(beta=0.25)
    t, V, C, Ov, Iv, G = simulate(p, T=150)
    Vstrat = _v_strategic(p)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                  gridspec_kw={'height_ratios': [2.5, 1]})

    a1.plot(t, V, 'b-',  lw=2.5, label=r'$V(t)$ — Vitality')
    a1.plot(t, C, 'r--', lw=2,   label=r'$C(t)$ — Career Capital')
    a1.plot(t, Ov, color='darkorange', ls=':', lw=2, label=r'$O(t)$ — Complexity')

    eqs = find_interior_equilibria(p)
    if eqs:
        Cs, Vs = eqs[0]['C'], eqs[0]['V']
        a1.axhline(Vs, color='blue', ls=':', alpha=0.4, lw=1)
        a1.annotate(f'$V^* = {Vs:.1f}$', xy=(t[-1]*0.7, Vs),
                    fontsize=10, color='blue', va='bottom')
        a1.axhline(Cs, color='red', ls=':', alpha=0.3, lw=1)
        a1.annotate(f'$C^* = {Cs:.1f}$', xy=(t[-1]*0.7, Cs),
                    fontsize=10, color='red', va='bottom')

    a1.axhline(Vstrat, color='purple', ls='-.', alpha=0.5, lw=1.5)
    a1.text(t[-1]*0.85, Vstrat + 0.3, r'$V_{\mathrm{strat}}$',
            fontsize=10, color='purple')

    idx = np.argmin(np.abs(G - 1.0))
    if 0 < idx < len(t) - 1:
        a1.axvline(t[idx], color='gray', ls='--', alpha=0.5)
        a1.text(t[idx] + 1, a1.get_ylim()[1] * 0.92,
                r'$t^*$', fontsize=12, color='gray')

    a1.set_ylabel('Level')
    a1.legend(loc='upper right', framealpha=0.9)
    a1.set_ylim(-0.3, max(max(C) * 1.05, max(Ov) * 1.05))
    a1.grid(True, alpha=0.15)
    a1.set_title('(a) Dynamic Evolution — Endogenous DLVT System')

    a2.plot(t, G, 'k-', lw=2)
    a2.axhline(1.0, color='red', ls='--', alpha=0.7, label=r'$\Gamma=1$ (critical)')
    a2.fill_between(t, 1, G, where=(G > 1), alpha=0.12, color='red')
    a2.fill_between(t, G, 1, where=(G < 1), alpha=0.12, color='green')
    a2.set_xlabel('Time ($t$)')
    a2.set_ylabel(r'$\Gamma(t)$')
    a2.legend(loc='upper right')
    a2.grid(True, alpha=0.15)
    a2.set_title(r'(b) Depletion Ratio $\Gamma = \delta O^\gamma / R$')
    a2.set_ylim(0, min(max(G) * 1.1, 8))

    plt.tight_layout()
    _save(fig, 'fig1_temporal_evolution', output_dir)


# ── Figure 2: Three Scenarios ─────────────────────────────────────────────────

def fig2(output_dir='figures/'):
    """Three outcome scenarios: sustainable, low-vitality, collapse."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # (a) Sustainable
    p_s = make_params(delta=0.008, beta=0.15)
    t1, V1, C1, *_ = simulate(p_s, T=200)
    eq_s = find_interior_equilibria(p_s)
    ax = axes[0]
    ax.plot(t1, V1, 'g-', lw=2.5, label='$V(t)$')
    ax.plot(t1, C1, 'k--', lw=1.5, alpha=0.5, label='$C(t)$')
    ax.axhline(_v_strategic(p_s), color='purple', ls='-.', alpha=0.4, lw=1)
    if eq_s:
        ax.axhline(eq_s[0]['V'], color='green', ls=':', alpha=0.5)
        ax.text(t1[-1]*0.55, eq_s[0]['V']+0.3,
                f"$V^*={eq_s[0]['V']:.1f}$", fontsize=9, color='green')
    ax.set_title('(a) Sustainable Leadership', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # (b) Low-vitality
    p_z = make_params(beta=0.25)
    t2, V2, C2, *_ = simulate(p_z, T=200)
    eq_z = find_interior_equilibria(p_z)
    Vstrat_z = _v_strategic(p_z)
    ax = axes[1]
    ax.plot(t2, V2, color='darkorange', lw=2.5, label='$V(t)$')
    ax.plot(t2, C2, 'k--', lw=1.5, alpha=0.5, label='$C(t)$')
    ax.axhline(Vstrat_z, color='purple', ls='-.', alpha=0.5, lw=1.5)
    ax.text(t2[-1]*0.55, Vstrat_z+0.3, r'$V_{\mathrm{strat}}$',
            fontsize=9, color='purple')
    ax.axhspan(0, Vstrat_z, alpha=0.06, color='red')
    if eq_z:
        ax.axhline(eq_z[0]['V'], color='darkorange', ls=':', alpha=0.5)
        ax.text(t2[-1]*0.55, eq_z[0]['V']-0.7,
                f"$V^*={eq_z[0]['V']:.1f}$ (low-vitality)", fontsize=9, color='darkorange')
    ax.set_title('(b) Low-Vitality Equilibrium', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # (c) Collapse (exogenous C)
    p_c = make_params(beta=0.25)
    C_func = lambda t: 0.5 + 0.5*t
    sol = solve_ivp(
        lambda t, y: [
            p_c['R']*(1-max(y[0],0)/p_c['Vmax'])
            - p_c['delta']*(p_c['O0']+p_c['beta']*C_func(t))**p_c['gamma']
              * max(y[0],0)/(max(y[0],0)+p_c['eps'])
        ],
        [0, 50], [8.0], method='RK45', max_step=0.05
    )
    t3 = sol.t; V3 = np.maximum(sol.y[0], 0)
    C3 = np.array([C_func(ti) for ti in t3])
    ax = axes[2]
    ax.plot(t3, V3, 'r-', lw=2.5, label='$V(t)$')
    ax.plot(t3, C3, 'k--', lw=1.5, alpha=0.5, label='$C(t)$ (exogenous)')
    ax.axhline(_v_strategic(p_c), color='purple', ls='-.', alpha=0.4, lw=1)
    ax.axhspan(0, _v_strategic(p_c), alpha=0.06, color='red')
    ax.set_title('(c) Collapse (exogenous $C$)', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    for ax, V_arr, C_arr in zip(axes, [V1,V2,V3], [C1,C2,C3]):
        ax.set_ylim(-0.3, max(max(V_arr), max(C_arr))*1.1)

    plt.tight_layout()
    _save(fig, 'fig2_three_scenarios', output_dir)


# ── Figure 3: Phase Portrait ──────────────────────────────────────────────────

def fig3(output_dir='figures/'):
    """Phase portrait with nullclines, sample trajectories, and equilibria."""
    p = make_params(beta=0.25)
    Vstrat = _v_strategic(p)
    C_range = np.linspace(0.01, 45, 600)

    # dV/dt = 0 nullcline
    Vnull = []
    for Ci in C_range:
        Oi = complexity(Ci, p)
        def r(V):
            if V <= 0: return p['R']
            return p['R']*(1-V/p['Vmax']) - p['delta']*Oi**p['gamma']*V/(V+p['eps'])
        try:
            Vnull.append(brentq(r, 0.001, p['Vmax']-0.001))
        except Exception:
            Vnull.append(np.nan)
    Vnull = np.array(Vnull)

    # dC/dt = 0 nullcline
    VCnull = np.array([
        p['mu'] * (1 + p['phi'] * (p['O0'] + p['beta']*c)) / p['alpha']
        for c in C_range
    ])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    valid = ~np.isnan(Vnull) & (Vnull > 0)
    ax.plot(C_range[valid], Vnull[valid], 'b-',  lw=2.5, label=r'$dV/dt = 0$', zorder=5)
    ax.plot(C_range, VCnull,              'r--', lw=2.5, label=r'$dC/dt = 0$', zorder=5)

    for V0, C0 in [(8,0.5),(6,3),(3,10),(9,5),(2,0.3),(8,20),(5,35)]:
        t, V, C, *_ = simulate(p, V0=V0, C0=C0, T=250)
        ax.plot(C, V, 'k-', lw=0.5, alpha=0.35)
        mid = len(t)//4
        if 0 < mid < len(t)-1:
            ax.annotate('', xy=(C[mid+1], V[mid+1]), xytext=(C[mid], V[mid]),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    eqs = find_interior_equilibria(p)
    for eq in eqs:
        mk = 'o' if eq['stable'] else 'x'
        cl = 'green' if eq['stable'] else 'red'
        ax.plot(eq['C'], eq['V'], marker=mk, ms=14, color=cl, mew=3, zorder=10,
                label=f"{'Stable' if eq['stable'] else 'Unstable'} "
                      f"($C^*$={eq['C']:.0f}, $V^*$={eq['V']:.1f})")

    ax.axhline(Vstrat, color='purple', ls='-.', alpha=0.4, lw=1.5)
    ax.axhspan(0, Vstrat, alpha=0.04, color='red')
    ax.text(1, Vstrat+0.2, r'$V_{\mathrm{strategic}}$', fontsize=10, color='purple')
    ax.text(1, 1.5, 'LOW-VITALITY ZONE', fontsize=9, color='red', alpha=0.5, weight='bold')

    ax.set_xlabel('Career Capital $C$')
    ax.set_ylabel('Vitality $V$')
    ax.set_xlim(0, 45); ax.set_ylim(0, p['Vmax']+0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_title('Phase Portrait with Nullclines and Equilibria')
    plt.tight_layout()
    _save(fig, 'fig3_phase_portrait', output_dir)


# ── Figure 4: Bifurcation ─────────────────────────────────────────────────────

def fig4(output_dir='figures/'):
    """Bifurcation diagrams: C* and V* vs β, and C* vs R."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) vs β
    betas = np.linspace(0.02, 1.5, 600)
    sb, sC, sV, cc_b, cc_C = [], [], [], [], []
    for b in betas:
        p = make_params(beta=b)
        for eq in find_interior_equilibria(p, C_max=500):
            if eq['stable']:
                sb.append(b); sC.append(eq['C']); sV.append(eq['V'])
        c = carrying_capacity(p)
        if c > 0:
            cc_b.append(b); cc_C.append(c)

    a1.plot(sb, sC, 'g-', lw=2, label='Equilibrium $C^*$')
    a1.plot(cc_b, cc_C, 'k--', lw=1.5, alpha=0.5,
            label=r'Carrying capacity $C^*_{\max}$')
    a1.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    a1.set_ylabel(r'Capital $C^*$', color='green')
    a1.set_ylim(0, 100); a1.grid(True, alpha=0.15)
    a1.set_title(r'(a) Bifurcation: $C^*$ and $V^*$ vs $\beta$')

    a1b = a1.twinx()
    a1b.plot(sb, sV, color='blue', ls='-.', lw=1.5, alpha=0.7, label='$V^*$')
    a1b.axhline(0.5*10.0, color='purple', ls=':', alpha=0.4)   # V_strategic
    a1b.set_ylabel(r'Vitality $V^*$', color='blue')
    a1b.set_ylim(0, 12)
    lines1, lbl1 = a1.get_legend_handles_labels()
    lines2, lbl2 = a1b.get_legend_handles_labels()
    a1.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc='upper right')

    # (b) vs R
    Rs = np.linspace(0.5, 8.0, 600)
    sr, sCr, sVr, ccr, ccCr = [], [], [], [], []
    for Rv in Rs:
        p = make_params(R=Rv)
        for eq in find_interior_equilibria(p, C_max=500):
            if eq['stable']:
                sr.append(Rv); sCr.append(eq['C']); sVr.append(eq['V'])
        c = carrying_capacity(p)
        if c > 0:
            ccr.append(Rv); ccCr.append(c)

    a2.plot(sr, sCr, 'g-', lw=2, label='Equilibrium $C^*$')
    a2.plot(ccr, ccCr, 'k--', lw=1.5, alpha=0.5, label=r'$C^*_{\max}$')
    a2.set_xlabel(r'Recovery Rate $R$')
    a2.set_ylabel(r'Capital $C^*$', color='green')
    a2.set_ylim(0, 100); a2.grid(True, alpha=0.15)
    a2.set_title(r'(b) Bifurcation: $C^*$ vs $R$')
    a2b = a2.twinx()
    a2b.plot(sr, sVr, color='blue', ls='-.', lw=1.5, alpha=0.7, label='$V^*$')
    a2b.axhline(0.5*10.0, color='purple', ls=':', alpha=0.4)
    a2b.set_ylabel(r'$V^*$', color='blue')
    a2b.set_ylim(0, 12)
    lines1, lbl1 = a2.get_legend_handles_labels()
    lines2, lbl2 = a2b.get_legend_handles_labels()
    a2.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc='center right')

    plt.tight_layout()
    _save(fig, 'fig4_bifurcation', output_dir)


# ── Figure 5: Impact Comparison ───────────────────────────────────────────────

def fig5(output_dir='figures/'):
    """Leadership impact: DLVT non-linear curve vs Becker linear prediction."""
    p = make_params(beta=0.30, mu=0.03)
    t, V, C, Ov, Iv, G = simulate(p, V0=8, C0=0.3, T=300)

    dC = np.diff(C)
    last_inc = np.max(np.where(dC > 0.001)[0]) if np.any(dC > 0.001) else len(C)-2
    Cf, If = C[:last_inc+1], Iv[:last_inc+1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Cf, If, 'b-', lw=2.5, label=r'DLVT: $I = CV/(1+\phi O)$', zorder=5)

    k = If[min(10, len(If)-1)] / Cf[min(10, len(Cf)-1)] if Cf[min(10, len(Cf)-1)] > 0 else 1
    Clin = np.linspace(0, max(Cf)*1.05, 100)
    ax.plot(Clin, k*Clin, 'k--', lw=1.5, alpha=0.5,
            label=f'Becker (HC Theory): $I = {k:.1f}C$')

    pk = np.argmax(If)
    ax.plot(Cf[pk], If[pk], 'ro', ms=10, zorder=10)
    ax.annotate('Peak Impact', xy=(Cf[pk], If[pk]),
                xytext=(Cf[pk]+2, If[pk]+5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))

    cc = carrying_capacity(p)
    if 0 < cc < max(Cf)*1.05:
        ax.axvline(cc, color='gray', ls=':', alpha=0.6)
        ax.text(cc+0.5, 3, r'$C^*_{\max}$', fontsize=10, color='gray', rotation=90)

    C_gap = min(cc * 1.2, max(Cf)*0.75) if cc > 0 else max(Cf)*0.7
    I_dlvt = np.interp(C_gap, Cf, If)
    I_beck = k * C_gap
    if I_beck > I_dlvt + 2:
        ax.annotate('', xy=(C_gap, I_dlvt), xytext=(C_gap, I_beck),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(C_gap+0.8, (I_dlvt+I_beck)/2, 'Prediction\nGap',
                fontsize=10, color='red', ha='left', va='center')

    ax.set_xlabel('Career Capital $C$'); ax.set_ylabel('Leadership Impact $I$')
    ax.set_title('Impact: Human Capital Theory vs. DLVT')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0, max(Cf)*1.05); ax.set_ylim(0, None)
    plt.tight_layout()
    _save(fig, 'fig5_impact_comparison', output_dir)


# ── Figure 6: Heatmap ─────────────────────────────────────────────────────────

def fig6(output_dir='figures/'):
    """Carrying capacity heatmap C*(β, R)."""
    betas = np.linspace(0.05, 1.0, 150)
    Rs    = np.linspace(0.5,  6.0, 150)
    Z = np.zeros((len(Rs), len(betas)))
    for i, Rv in enumerate(Rs):
        for j, bv in enumerate(betas):
            Z[i, j] = carrying_capacity(make_params(R=Rv, beta=bv))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(betas, Rs, Z, cmap='RdYlGn', shading='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Carrying Capacity $C^*_{\max}$')
    CS = ax.contour(betas, Rs, Z, levels=[5, 10, 20, 40, 60],
                    colors='black', linewidths=0.8, alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    ax.set_ylabel(r'$R$ (recovery rate)')
    ax.set_title(r'Leadership Carrying Capacity $C^*(\beta, R)$')
    plt.tight_layout()
    _save(fig, 'fig6_sensitivity_heatmap', output_dir)


# ── Figure 7: Regime Map ──────────────────────────────────────────────────────

def fig7(output_dir='figures/'):
    """Parameter-space regime map in (β, δ): sustainable / low-vitality / collapse-prone."""
    from .analysis import classify_regime

    betas  = np.linspace(0.02, 0.8,  100)
    deltas = np.linspace(0.005, 0.06, 100)
    REGIME_CODE = {'sustainable': 0, 'zombie': 1, 'collapse-prone': 2}

    Z = np.zeros((len(deltas), len(betas)))
    for i, dv in enumerate(deltas):
        for j, bv in enumerate(betas):
            Z[i, j] = REGIME_CODE[classify_regime(make_params(beta=bv, delta=dv))]

    cmap = ListedColormap(['#2ecc71', '#f39c12', '#e74c3c'])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.pcolormesh(betas, deltas, Z, cmap=cmap, shading='auto', vmin=0, vmax=2)
    ax.contour(betas, deltas, Z, levels=[0.5, 1.5],
               colors='white', linewidths=2, linestyles='-')

    for val, label in [(0, 'SUSTAINABLE'), (1, 'LOW-VITALITY'), (2, 'COLLAPSE-\nPRONE')]:
        ys, xs = np.where(Z == val)
        if len(xs):
            cx = betas[int(np.median(xs))]
            cy = deltas[int(np.median(ys))]
            ax.text(cx, cy, label, fontsize=13, fontweight='bold',
                    color='white', ha='center', va='center', alpha=0.9)

    Vstrat = V_STRATEGIC_FRACTION * 10.0  # default Vmax
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'Sustainable ($V^* \\geq {Vstrat:.0f}$)'),
        Patch(facecolor='#f39c12', label=f'Low-vitality ($0 < V^* < {Vstrat:.0f}$)'),
        Patch(facecolor='#e74c3c', label='Collapse-prone (no stable eq.)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    ax.set_ylabel(r'$\delta$ (energetic cost coefficient)')
    ax.set_title(r'Leadership Regime Map: $(\beta, \delta)$ Parameter Space')
    plt.tight_layout()
    _save(fig, 'fig7_regime_map', output_dir)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_all(output_dir='figures/'):
    """Generate all 7 publication figures and save to output_dir."""
    print(f'Generating all DLVT figures → {output_dir}')
    fig1(output_dir=output_dir)
    fig2(output_dir=output_dir)
    fig3(output_dir=output_dir)
    fig4(output_dir=output_dir)
    fig5(output_dir=output_dir)
    fig6(output_dir=output_dir)
    fig7(output_dir=output_dir)
    print(f'\n✓ All 7 figures saved to {output_dir}')
