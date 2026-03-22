#!/usr/bin/env python3
"""
DLVT Publication Figures — Version 3 (refined)
Generates all 7 figures for the SSRN Working Paper.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

V_STRATEGIC = 5.0  # Minimum V for generative strategic work

# ── Core model ────────────────────────────────────────────────────────────

def make_params(**overrides):
    p = dict(R=3.0, Vmax=10.0, delta=0.02, gamma=2.0, O0=1.0, beta=0.25,
             eta=1.0, alpha=0.1, phi=0.15, mu=0.2, eps=0.1)
    p.update(overrides)
    return p

def O(C, p): return p['O0'] + p['beta'] * np.power(np.maximum(C, 0), p['eta'])
def I(V, C, p): return C * V / (1 + p['phi'] * O(C, p))

def sys(t, y, p):
    V, C = max(y[0], 0), max(y[1], 0)
    Ov = O(C, p)
    dV = p['R']*(1-V/p['Vmax']) - p['delta']*Ov**p['gamma']*V/(V+p['eps'])
    dC = p['alpha']*I(V, C, p) - p['mu']*C
    return [dV, dC]

def sim(p, V0=8.0, C0=0.5, T=120, ms=0.05):
    sol = solve_ivp(sys, [0,T], [V0,C0], args=(p,), method='RK45', max_step=ms)
    t=sol.t; V=np.maximum(sol.y[0],0); C=np.maximum(sol.y[1],0)
    return t, V, C, O(C,p), I(V,C,p), p['delta']*O(C,p)**p['gamma']/p['R']

def find_eq(p, Cmax=120):
    def Vs(Cs):
        return p['mu']*(1+p['phi']*(p['O0']+p['beta']*Cs**p['eta']))/p['alpha']
    def res(Cs):
        v=Vs(Cs); o=p['O0']+p['beta']*Cs**p['eta']
        if v<=0 or v>=p['Vmax']: return 1e10
        return p['R']*(1-v/p['Vmax']) - p['delta']*o**p['gamma']*v/(v+p['eps'])
    Cscan=np.linspace(0.01,Cmax,10000); r=np.array([res(c) for c in Cscan])
    eqs=[]
    for i in range(len(r)-1):
        if r[i]*r[i+1]<0:
            try:
                cs=brentq(res,Cscan[i],Cscan[i+1]); vs=Vs(cs)
                if 0<vs<p['Vmax']: eqs.append((cs,vs))
            except: pass
    return eqs

def CC(p):
    Omax=(p['R']/p['delta'])**(1/p['gamma'])
    return max(0,((Omax-p['O0'])/p['beta'])**(1/p['eta'])) if Omax>p['O0'] else 0

# ── FIGURE 1: Temporal Evolution with Gamma ──────────────────────────────

def fig1():
    p = make_params(beta=0.25)
    t, V, C, Ov, Iv, G = sim(p, T=150)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                  gridspec_kw={'height_ratios': [2.5, 1]})

    a1.plot(t, V, 'b-', lw=2.5, label=r'$V(t)$ — Vitality')
    a1.plot(t, C, 'r--', lw=2, label=r'$C(t)$ — Career Capital')
    a1.plot(t, Ov, color='darkorange', ls=':', lw=2, label=r'$O(t)$ — Complexity')

    # Equilibrium annotations
    eqs = find_eq(p)
    if eqs:
        Cs, Vs = eqs[0]
        a1.axhline(Vs, color='blue', ls=':', alpha=0.4, lw=1)
        a1.annotate(f'$V^* = {Vs:.1f}$', xy=(t[-1]*0.7, Vs),
                   fontsize=10, color='blue', va='bottom')
        a1.axhline(Cs, color='red', ls=':', alpha=0.3, lw=1)
        a1.annotate(f'$C^* = {Cs:.1f}$', xy=(t[-1]*0.7, Cs),
                   fontsize=10, color='red', va='bottom')

    # V_strategic
    a1.axhline(V_STRATEGIC, color='purple', ls='-.', alpha=0.5, lw=1.5)
    a1.text(t[-1]*0.85, V_STRATEGIC+0.3, r'$V_{\mathrm{strat}}$',
           fontsize=10, color='purple')

    # t* marker
    idx = np.argmin(np.abs(G - 1.0))
    if 0 < idx < len(t)-1:
        a1.axvline(t[idx], color='gray', ls='--', alpha=0.5)
        a1.text(t[idx]+1, a1.get_ylim()[1]*0.92, r'$t^*$', fontsize=12, color='gray')

    a1.set_ylabel('Level'); a1.legend(loc='upper right', framealpha=0.9)
    a1.set_ylim(-0.3, max(max(C)*1.05, max(Ov)*1.05))
    a1.grid(True, alpha=0.15); a1.set_title('(a) Dynamic Evolution — Endogenous DLVT System')

    # Gamma panel
    a2.plot(t, G, 'k-', lw=2)
    a2.axhline(1.0, color='red', ls='--', alpha=0.7, label=r'$\Gamma=1$ (critical)')
    a2.fill_between(t, 1, G, where=(G>1), alpha=0.12, color='red')
    a2.fill_between(t, G, 1, where=(G<1), alpha=0.12, color='green')
    a2.set_xlabel('Time ($t$)'); a2.set_ylabel(r'$\Gamma(t)$')
    a2.legend(loc='upper right'); a2.grid(True, alpha=0.15)
    a2.set_title(r'(b) Depletion Ratio $\Gamma = \delta O^\gamma / R$')
    a2.set_ylim(0, min(max(G)*1.1, 8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_temporal_evolution.pdf')
    plt.savefig(f'{OUT}/fig1_temporal_evolution.png')
    plt.close(); print('[OK] Fig 1')


# ── FIGURE 2: Three Scenarios ────────────────────────────────────────────

def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # (a) Sustainable: low delta → less drain → higher V*
    p_sust = make_params(delta=0.008, beta=0.15)
    t1, V1, C1, O1, I1, G1 = sim(p_sust, T=200)
    eq_sust = find_eq(p_sust)

    ax = axes[0]
    ax.plot(t1, V1, 'g-', lw=2.5, label='$V(t)$')
    ax.plot(t1, C1, 'k--', lw=1.5, alpha=0.5, label='$C(t)$')
    ax.axhline(V_STRATEGIC, color='purple', ls='-.', alpha=0.4, lw=1)
    if eq_sust:
        ax.axhline(eq_sust[0][1], color='green', ls=':', alpha=0.5)
        ax.text(t1[-1]*0.55, eq_sust[0][1]+0.3, f'$V^*={eq_sust[0][1]:.1f}$',
               fontsize=9, color='green')
    ax.set_title('(a) Sustainable Leadership', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level'); ax.legend(fontsize=9)
    ax.set_ylim(-0.3, max(max(V1),max(C1))*1.1); ax.grid(True, alpha=0.15)

    # (b) Zombie: default params → V* < V_strategic
    p_zomb = make_params(beta=0.25)
    t2, V2, C2, O2, I2, G2 = sim(p_zomb, T=200)
    eq_zomb = find_eq(p_zomb)

    ax = axes[1]
    ax.plot(t2, V2, color='darkorange', lw=2.5, label='$V(t)$')
    ax.plot(t2, C2, 'k--', lw=1.5, alpha=0.5, label='$C(t)$')
    ax.axhline(V_STRATEGIC, color='purple', ls='-.', alpha=0.5, lw=1.5)
    ax.text(t2[-1]*0.55, V_STRATEGIC+0.3, r'$V_{\mathrm{strat}}$', fontsize=9, color='purple')
    if eq_zomb:
        ax.axhline(eq_zomb[0][1], color='darkorange', ls=':', alpha=0.5)
        ax.text(t2[-1]*0.55, eq_zomb[0][1]-0.7, f'$V^*={eq_zomb[0][1]:.1f}$ (zombie)',
               fontsize=9, color='darkorange')
    # Shade zombie region
    ax.axhspan(0, V_STRATEGIC, alpha=0.06, color='red')
    ax.set_title('(b) Zombie Equilibrium', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level'); ax.legend(fontsize=9)
    ax.set_ylim(-0.3, max(max(V2),max(C2))*1.1); ax.grid(True, alpha=0.15)

    # (c) Collapse: exogenous C
    p_coll = make_params(beta=0.25)
    C_func = lambda t: 0.5 + 0.5*t
    sol = solve_ivp(lambda t,y: [
        p_coll['R']*(1-max(y[0],0)/p_coll['Vmax'])
        - p_coll['delta']*(p_coll['O0']+p_coll['beta']*C_func(t))**p_coll['gamma']
        * max(y[0],0)/(max(y[0],0)+p_coll['eps'])
    ], [0, 50], [8.0], method='RK45', max_step=0.05)
    t3=sol.t; V3=np.maximum(sol.y[0],0); C3=np.array([C_func(ti) for ti in t3])

    ax = axes[2]
    ax.plot(t3, V3, 'r-', lw=2.5, label='$V(t)$')
    ax.plot(t3, C3, 'k--', lw=1.5, alpha=0.5, label='$C(t)$ (exogenous)')
    ax.axhline(V_STRATEGIC, color='purple', ls='-.', alpha=0.4, lw=1)
    ax.axhspan(0, V_STRATEGIC, alpha=0.06, color='red')
    ax.set_title('(c) Collapse (exogenous $C$)', fontsize=12)
    ax.set_xlabel('Time'); ax.set_ylabel('Level'); ax.legend(fontsize=9)
    ax.set_ylim(-0.3, max(max(V3),max(C3))*1.1); ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_three_scenarios.pdf')
    plt.savefig(f'{OUT}/fig2_three_scenarios.png')
    plt.close(); print('[OK] Fig 2')


# ── FIGURE 3: Phase Portrait ────────────────────────────────────────────

def fig3():
    p = make_params(beta=0.25)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    C_range = np.linspace(0.01, 45, 600)

    # dV/dt=0 nullcline
    Vnull = []
    for Ci in C_range:
        Oi = p['O0']+p['beta']*Ci
        def r(V):
            if V<=0: return p['R']
            return p['R']*(1-V/p['Vmax'])-p['delta']*Oi**p['gamma']*V/(V+p['eps'])
        try: Vnull.append(brentq(r, 0.001, p['Vmax']-0.001))
        except: Vnull.append(np.nan)
    Vnull = np.array(Vnull)

    # dC/dt=0 nullcline
    VCnull = np.array([p['mu']*(1+p['phi']*(p['O0']+p['beta']*c))/p['alpha'] for c in C_range])

    valid = ~np.isnan(Vnull) & (Vnull > 0)
    ax.plot(C_range[valid], Vnull[valid], 'b-', lw=2.5, label=r'$dV/dt = 0$', zorder=5)
    ax.plot(C_range, VCnull, 'r--', lw=2.5, label=r'$dC/dt = 0$', zorder=5)

    # Trajectories
    for V0, C0 in [(8,0.5),(6,3),(3,10),(9,5),(2,0.3),(8,20),(5,35)]:
        t,V,C,_,_,_ = sim(p, V0=V0, C0=C0, T=250)
        ax.plot(C, V, 'k-', lw=0.5, alpha=0.35)
        mid = len(t)//4
        if 0<mid<len(t)-1:
            ax.annotate('', xy=(C[mid+1],V[mid+1]), xytext=(C[mid],V[mid]),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # Equilibrium
    eqs = find_eq(p)
    for Cs, Vs in eqs:
        ax.plot(Cs, Vs, 'o', ms=14, color='green', mew=3, zorder=10,
                label=f'Equil. ($C^*$={Cs:.0f}, $V^*$={Vs:.1f})')

    # V_strategic zone
    ax.axhline(V_STRATEGIC, color='purple', ls='-.', alpha=0.4, lw=1.5)
    ax.axhspan(0, V_STRATEGIC, alpha=0.04, color='red')
    ax.text(1, V_STRATEGIC+0.2, r'$V_{\mathrm{strategic}}$', fontsize=10, color='purple')
    ax.text(1, 1.5, 'ZOMBIE ZONE', fontsize=9, color='red', alpha=0.5, weight='bold')

    ax.set_xlabel('Career Capital $C$'); ax.set_ylabel('Vitality $V$')
    ax.set_xlim(0, 45); ax.set_ylim(0, p['Vmax']+0.5)
    ax.legend(loc='upper right', fontsize=9); ax.grid(True, alpha=0.15)
    ax.set_title('Phase Portrait with Nullclines and Equilibria')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_phase_portrait.pdf')
    plt.savefig(f'{OUT}/fig3_phase_portrait.png')
    plt.close(); print('[OK] Fig 3')


# ── FIGURE 4: Bifurcation Diagrams ──────────────────────────────────────

def fig4():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) C* and V* vs beta
    betas = np.linspace(0.02, 1.5, 600)
    sb, sC, sV, cc_b, cc_C = [], [], [], [], []
    for b in betas:
        p = make_params(beta=b)
        eqs = find_eq(p, Cmax=500)
        for Cs, Vs in eqs:
            sb.append(b); sC.append(Cs); sV.append(Vs)
        c = CC(p)
        if c > 0: cc_b.append(b); cc_C.append(c)

    # Twin axis for V*
    a1.plot(sb, sC, 'g-', lw=2, label='Equilibrium $C^*$')
    a1.plot(cc_b, cc_C, 'k--', lw=1.5, alpha=0.5, label=r'Carrying capacity $C^*_{\max}$')
    a1.set_xlabel(r'Capital-Complexity Coupling $\beta$')
    a1.set_ylabel(r'Capital $C^*$', color='green')
    a1.set_ylim(0, 100); a1.grid(True, alpha=0.15)
    a1.set_title(r'(a) Bifurcation: $C^*$ and $V^*$ vs $\beta$')

    a1b = a1.twinx()
    a1b.plot(sb, sV, color='blue', ls='-.', lw=1.5, alpha=0.7, label='$V^*$')
    a1b.axhline(V_STRATEGIC, color='purple', ls=':', alpha=0.4)
    a1b.set_ylabel(r'Vitality $V^*$', color='blue')
    a1b.set_ylim(0, 12)

    # Combined legend
    lines1, labels1 = a1.get_legend_handles_labels()
    lines2, labels2 = a1b.get_legend_handles_labels()
    a1.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper right')

    # (b) C* vs R
    Rs = np.linspace(0.5, 8.0, 600)
    sr, sCr, sVr, ccr, ccCr = [], [], [], [], []
    for Rv in Rs:
        p = make_params(R=Rv)
        eqs = find_eq(p, Cmax=500)
        for Cs, Vs in eqs:
            sr.append(Rv); sCr.append(Cs); sVr.append(Vs)
        c = CC(p)
        if c > 0: ccr.append(Rv); ccCr.append(c)

    a2.plot(sr, sCr, 'g-', lw=2, label='Equilibrium $C^*$')
    a2.plot(ccr, ccCr, 'k--', lw=1.5, alpha=0.5, label=r'$C^*_{\max}$')
    a2.set_xlabel(r'Recovery Rate $R$'); a2.set_ylabel(r'Capital $C^*$', color='green')
    a2.set_ylim(0, 100); a2.grid(True, alpha=0.15)
    a2.set_title(r'(b) Bifurcation: $C^*$ vs $R$')

    a2b = a2.twinx()
    a2b.plot(sr, sVr, color='blue', ls='-.', lw=1.5, alpha=0.7, label='$V^*$')
    a2b.axhline(V_STRATEGIC, color='purple', ls=':', alpha=0.4)
    a2b.set_ylabel(r'$V^*$', color='blue'); a2b.set_ylim(0, 12)

    lines1, labels1 = a2.get_legend_handles_labels()
    lines2, labels2 = a2b.get_legend_handles_labels()
    a2.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='center right')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_bifurcation.pdf')
    plt.savefig(f'{OUT}/fig4_bifurcation.png')
    plt.close(); print('[OK] Fig 4')


# ── FIGURE 5: Impact Comparison ──────────────────────────────────────────

def fig5():
    p = make_params(beta=0.30, mu=0.03)  # slow depreciation to show full curve
    fig, ax = plt.subplots(figsize=(9, 5.5))

    t, V, C, Ov, Iv, G = sim(p, V0=8, C0=0.3, T=300)

    # Forward trajectory only (while C is increasing)
    dC = np.diff(C)
    last_inc = np.max(np.where(dC > 0.001)[0]) if np.any(dC > 0.001) else len(C)-2
    Cf, If = C[:last_inc+1], Iv[:last_inc+1]

    ax.plot(Cf, If, 'b-', lw=2.5, label=r'DLVT: $I = CV/(1+\phi O)$', zorder=5)

    # Becker linear
    k = If[min(10, len(If)-1)] / Cf[min(10, len(Cf)-1)] if Cf[min(10,len(Cf)-1)] > 0 else 1
    Clin = np.linspace(0, max(Cf)*1.05, 100)
    ax.plot(Clin, k*Clin, 'k--', lw=1.5, alpha=0.5, label=f'Becker: $I = {k:.1f}C$')

    # Peak
    pk = np.argmax(If)
    ax.plot(Cf[pk], If[pk], 'ro', ms=10, zorder=10)
    ax.annotate('Peak Impact', xy=(Cf[pk], If[pk]),
               xytext=(Cf[pk]+2, If[pk]+5), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red'))

    # C*
    cc = CC(p)
    if 0 < cc < max(Cf)*1.05:
        ax.axvline(cc, color='gray', ls=':', alpha=0.6)
        ax.text(cc+0.5, 3, r'$C^*_{\max}$', fontsize=10, color='gray', rotation=90)

    # Prediction gap
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
    ax.legend(fontsize=10, loc='upper left'); ax.grid(True, alpha=0.15)
    ax.set_xlim(0, max(Cf)*1.05); ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_impact_comparison.pdf')
    plt.savefig(f'{OUT}/fig5_impact_comparison.png')
    plt.close(); print('[OK] Fig 5')


# ── FIGURE 6: Heatmap ────────────────────────────────────────────────────

def fig6():
    fig, ax = plt.subplots(figsize=(8, 6))
    betas = np.linspace(0.05, 1.0, 150)
    Rs = np.linspace(0.5, 6.0, 150)
    Z = np.zeros((len(Rs), len(betas)))
    for i, Rv in enumerate(Rs):
        for j, bv in enumerate(betas):
            Z[i,j] = CC(make_params(R=Rv, beta=bv))

    im = ax.pcolormesh(betas, Rs, Z, cmap='RdYlGn', shading='auto')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(r'Carrying Capacity $C^*_{\max}$')
    CS = ax.contour(betas, Rs, Z, levels=[5, 10, 20, 40, 60],
                    colors='black', linewidths=0.8, alpha=0.6)
    ax.clabel(CS, inline=True, fontsize=8)
    ax.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    ax.set_ylabel(r'$R$ (recovery rate)')
    ax.set_title(r'Leadership Carrying Capacity $C^*(\beta, R)$')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_sensitivity_heatmap.pdf')
    plt.savefig(f'{OUT}/fig6_sensitivity_heatmap.png')
    plt.close(); print('[OK] Fig 6')


# ── FIGURE 7: Regime Map (NEW — advanced simulation) ─────────────────────

def fig7():
    """Parameter regime map: classify (beta, delta) into sustainable/zombie/collapse."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    betas = np.linspace(0.02, 0.8, 100)
    deltas = np.linspace(0.005, 0.06, 100)

    # 0=sustainable, 1=zombie, 2=no equilibrium (collapse-prone)
    regime = np.zeros((len(deltas), len(betas)))

    for i, dv in enumerate(deltas):
        for j, bv in enumerate(betas):
            p = make_params(beta=bv, delta=dv)
            eqs = find_eq(p, Cmax=300)
            cc = CC(p)
            if not eqs or cc <= 0:
                regime[i, j] = 2  # no sustainable equilibrium
            else:
                Cs, Vs = eqs[0]
                if Vs >= V_STRATEGIC:
                    regime[i, j] = 0  # sustainable
                else:
                    regime[i, j] = 1  # zombie

    # Custom colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#2ecc71', '#f39c12', '#e74c3c'])
    im = ax.pcolormesh(betas, deltas, regime, cmap=cmap, shading='auto', vmin=0, vmax=2)

    # Labels in each region
    # Find centroids
    for val, label, color in [(0, 'SUSTAINABLE', 'white'),
                               (1, 'ZOMBIE', 'white'),
                               (2, 'COLLAPSE-\nPRONE', 'white')]:
        ys, xs = np.where(regime == val)
        if len(xs) > 0:
            cx = betas[int(np.median(xs))]
            cy = deltas[int(np.median(ys))]
            ax.text(cx, cy, label, fontsize=13, fontweight='bold',
                   color=color, ha='center', va='center', alpha=0.9)

    # Contour boundaries
    ax.contour(betas, deltas, regime, levels=[0.5, 1.5], colors='white',
               linewidths=2, linestyles='-')

    ax.set_xlabel(r'$\beta$ (capital-complexity coupling)')
    ax.set_ylabel(r'$\delta$ (energetic cost coefficient)')
    ax.set_title(r'Leadership Regime Map: $(\beta, \delta)$ Parameter Space')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'Sustainable ($V^* \\geq {V_STRATEGIC:.0f}$)'),
        Patch(facecolor='#f39c12', label=f'Zombie ($0 < V^* < {V_STRATEGIC:.0f}$)'),
        Patch(facecolor='#e74c3c', label='Collapse-prone (no stable eq.)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_regime_map.pdf')
    plt.savefig(f'{OUT}/fig7_regime_map.png')
    plt.close(); print('[OK] Fig 7')


# ── MAIN ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print(f'\n✓ All 7 figures saved to {OUT}')
