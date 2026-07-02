#!/usr/bin/env python3
"""
DLVT — Figure 14: fast--slow structure and the colored basin portrait.

The two remaining virtual experiments requested by the methodology review,
in one figure (companion to Appendix A11):

  Panel (a): phase plane with the slow manifold V_qe(O(C)) — which IS the
             V-nullcline — the capital nullcline V_c = mu(1+phi*O)/alpha,
             their unique intersection (the interior equilibrium), three
             full trajectories spiraling in, and the axis saddle at
             (V ~ 9.934, C = 0) marked.
  Panel (b): colored basin portrait — pcolormesh of the time to enter and
             stay in a band around (V*, C*) over a 40x40 grid of initial
             conditions. Theorem 2c predicts every C0 > 0 converges: the
             portrait is a smooth gradient with no basin boundary
             ('basin = {C > 0}, no escape').
  Panel (c): C(t), full 2D system vs the reduced quasi-static 1D flow, from
             three initial conditions, with the MEASURED max relative error
             annotated. The reduced flow is monotone and cannot reproduce
             the spiral overshoot — where the ~15% peak error lives.

Honest separation summary (printed): the raw-rate ratio R/mu = 15 of
earlier drafts overstates the separation; the Jacobian-diagonal ratio at
the equilibrium is |J_VV/J_CC| ~ 3, and the eigenvalues are the complex
pair -0.205 +/- 0.331i, so the modes mix near the equilibrium — the
quasi-static reduction is an intuition-builder for the slow capital drift,
not a substitute for the 2D analysis.

All model math is imported from the `dlvt` package; the figure is
deterministic (no random numbers).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from dlvt.model import make_params, simulate, complexity            # noqa: E402
from dlvt.analysis import find_interior_equilibria                  # noqa: E402
from dlvt.fastslow import (                                         # noqa: E402
    basin_portrait_grid,
    reduction_error,
    slow_manifold,
)

OUTPUT_DIR = REPO_DIR / 'figures'

rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Fixed, CVD-distinct hues per initial condition (identity, never cycled).
IC_LIST = [(8.0, 5.0), (2.0, 60.0), (9.5, 80.0)]
IC_COLORS = ['#0072B2', '#D55E00', '#009E73']

V_AXIS_SADDLE = 9.934  # axis saddle (V, C) = (9.934, 0); Appendix A7


def fig14_fastslow_basin():
    p = make_params()
    eq = find_interior_equilibria(p)[0]
    V_star, C_star = eq['V'], eq['C']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): phase plane with the slow manifold ──────────────────────
    C_grid = np.linspace(0.0, 90.0, 400)
    V_man = slow_manifold(C_grid, p)
    O_grid = complexity(C_grid, p)
    V_cnull = p['mu'] * (1.0 + p['phi'] * O_grid) / p['alpha']

    ax1.plot(C_grid, V_man, color='black', lw=2.2,
             label=r'slow manifold $V_{\mathrm{qe}}(O(C))$ ($\equiv$ $V$-nullcline)')
    ax1.plot(C_grid, V_cnull, color='dimgray', lw=1.6, ls='--',
             label=r'$C$-nullcline $V_c=\mu(1+\varphi O)/\alpha$')
    for (V0, C0), col in zip(IC_LIST, IC_COLORS):
        _, V_tr, C_tr, *_ = simulate(p, V0=V0, C0=C0, T=120.0)
        ax1.plot(C_tr, V_tr, color=col, lw=1.1, alpha=0.9,
                 label=f'trajectory from $({V0:g},\\ {C0:g})$')
        ax1.plot(C0, V0, marker='o', ms=5, color=col, mec='white', mew=0.5)
    ax1.plot(C_star, V_star, marker='*', ms=16, color='crimson',
             mec='black', mew=0.6, zorder=5,
             label=f'nullcline intersection $(V^*,C^*)=({V_star:.2f},{C_star:.1f})$')
    ax1.plot(0.0, V_AXIS_SADDLE, marker='X', ms=10, color='goldenrod',
             mec='black', mew=0.6, zorder=5,
             label=f'axis saddle $({V_AXIS_SADDLE:.2f},\\ 0)$')
    ax1.set_xlabel('Career Capital $C$')
    ax1.set_ylabel('Vitality $V$')
    ax1.set_xlim(-2, 92)
    ax1.set_ylim(0, 10.5)
    ax1.set_title('(a) Slow manifold and spiral approach')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=7.5)

    # ── Panel (b): colored basin portrait (time-to-converge) ──────────────
    grid = basin_portrait_grid(p, n_V=40, n_C=40, V_range=None,
                               C_range=(0.2, 90.0), T=400.0, band=0.5)
    T_plot = np.ma.masked_invalid(grid['T_conv'])
    im = ax2.pcolormesh(grid['V0'], grid['C0'], T_plot,
                        cmap='viridis', shading='auto')
    cb = fig.colorbar(im, ax=ax2, pad=0.02)
    cb.set_label('time to enter and stay in band')
    ax2.plot(V_star, C_star, marker='*', ms=14, color='white',
             mec='black', mew=0.7)
    ax2.annotate(r'basin $=\{C>0\}$, no escape'
                 f"\n({grid['n_converged']}/{grid['n_total']} converge,"
                 f" {grid['n_exceptions']} exceptions)",
                 xy=(0.03, 0.97), xycoords='axes fraction',
                 ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.85))
    ax2.set_xlabel('initial vitality $V_0$')
    ax2.set_ylabel('initial capital $C_0$')
    ax2.set_title(f"(b) Basin portrait: time to converge\n"
                  f"(band $\\pm${grid['band']:g} around $(V^*,C^*)$, "
                  f"$T={grid['T']:g}$)")

    # ── Panel (c): C(t) — full vs reduced, with measured errors ────────────
    results = []
    for k, ((V0, C0), col) in enumerate(zip(IC_LIST, IC_COLORS)):
        res = reduction_error(p, V0=V0, C0=C0, T=120.0)
        results.append(((V0, C0), res))
        ax3.plot(res['t'], res['C_full'], color=col, lw=1.8,
                 label=f'full 2D, $({V0:g},\\ {C0:g})$')
        ax3.plot(res['t'], res['C_reduced'], color=col, lw=1.4, ls='--',
                 label='reduced 1D' if k == 0 else None)
    ax3.axhline(C_star, color='crimson', ls=':', lw=1.2,
                label=f'$C^* = {C_star:.2f}$')
    err_lines = '\n'.join(
        f"$({V0:g},{C0:g})$: max rel err "
        f"{res['max_rel_error_C'] * 100:.1f}%"
        for (V0, C0), res in results)
    sep = results[0][1]['eigen_separation']
    ax3.annotate('measured reduction error in $C(t)$:\n' + err_lines +
                 f"\n$|J_{{VV}}/J_{{CC}}| = {sep:.2f}$ (not 15)",
                 xy=(0.97, 0.05), xycoords='axes fraction',
                 ha='right', va='bottom', fontsize=8.5,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    ax3.set_xlabel('time')
    ax3.set_ylabel('career capital $C(t)$')
    ax3.set_title('(c) Full vs quasi-static reduction:\n'
                  'the 1D flow cannot overshoot')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.legend(loc='center right', framealpha=0.95, fontsize=8)

    fig.suptitle('Fast--slow structure: exact slow-manifold equilibrium, '
                 'honest ~3x separation, and the colored basin portrait',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'fig14_fastslow_basin.{ext}')
    plt.close(fig)

    # ── Summary ────────────────────────────────────────────────────────────
    res0 = results[0][1]
    Tc = grid['T_conv'][np.isfinite(grid['T_conv'])]
    print('[OK] Fig 14 saved to figures/')
    print('\n' + '=' * 70)
    print('FAST-SLOW / BASIN PORTRAIT SUMMARY (Appendix A11)')
    print('=' * 70)
    print(f"interior equilibrium    : (V*, C*) = ({V_star:.5f}, {C_star:.4f})")
    print(f"equilibrium mismatch    : {res0['equilibrium_mismatch']:.2e} "
          "(reduced == full, exact)")
    print(f"raw-rate ratio R/mu     : {res0['raw_rate_ratio']:.1f}  "
          "(the MISLEADING 15x claim of earlier drafts)")
    print(f"eigen-separation        : |J_VV/J_CC| = "
          f"{res0['eigen_separation']:.3f}  (measured: ~3x, not 15x)")
    print(f"modes mix (complex pair): {res0['modes_mix']}  "
          f"eigenvalues = {np.round(res0['eigenvalues'], 4)}")
    print('reduction error in C(t) over [0, 120]:')
    for (V0, C0), res in results:
        print(f"  IC ({V0:>4g}, {C0:>4g}): max rel = "
              f"{res['max_rel_error_C']:.4f}, mean rel = "
              f"{res['mean_rel_error_C']:.5f}")
    print('  (error peaks during the spiral overshoot; the reduction is')
    print('   good for the slow drift, degrades near the equilibrium)')
    print(f"basin portrait          : {grid['n_converged']}/{grid['n_total']} "
          f"converged, {grid['n_exceptions']} exceptions "
          "(Theorem 2c: basin = {C > 0})")
    print(f"time-to-converge        : min {Tc.min():.2f}, median "
          f"{np.median(Tc):.2f}, max {Tc.max():.2f} "
          "(median ~ one rotation period 2*pi/0.331 ~ 19)")
    print('=' * 70 + '\n')
    return grid, results


if __name__ == '__main__':
    print('DLVT Figure 14 — fast-slow reduction + colored basin portrait\n')
    grid, results = fig14_fastslow_basin()
    assert grid['n_exceptions'] == 0, 'Theorem 2c violated: escape detected'
    assert results[0][1]['modes_mix'], 'expected complex pair at equilibrium'
    print('Figure 14 complete.')
