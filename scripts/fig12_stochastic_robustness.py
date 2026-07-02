#!/usr/bin/env python3
"""
DLVT — Figure 12: Stochastic robustness and identifiability ridges.

Two of the virtual experiments requested by the methodology review, in one
figure:

  Panel (a): sample Euler–Maruyama V-paths under additive noise
             (sigma_V = 0.2) against the deterministic trajectory and the
             attractor level V* — the interior equilibrium survives noise as
             a stochastically-perturbed steady state, not a knife-edge.
  Panel (b): persistence vs noise amplitude — the fraction of post-burn-in
             time spent inside the (V*, C*) band and the fraction of paths
             that escape toward collapse, as functions of sigma_V.
  Panel (c): identifiability ridges made executable — on an
             equilibrium-dominated synthetic panel, the fitting loss is flat
             along the joint (mu, alpha) scaling (only the ratio mu/alpha is
             identified) but steep when mu moves alone; a transient-rich
             panel partially breaks the ridge, which is exactly why the
             empirical program (Chapter 6) relies on exogenous shocks and
             transients for identification.

All model math is imported from the `dlvt` package; seeds are fixed, so the
figure is bit-reproducible.
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

from dlvt.model import make_params, simulate                 # noqa: E402
from dlvt.analysis import find_interior_equilibria           # noqa: E402
from dlvt.stochastic import escape_curve, simulate_sde       # noqa: E402
from dlvt.recovery import ridge_profile, synthesize_panel    # noqa: E402

OUTPUT_DIR = REPO_DIR / 'figures'

rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def fig12_stochastic_robustness():
    p = make_params()
    eq = find_interior_equilibria(p)[0]
    V_star, C_star = eq['V'], eq['C']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): sample paths under noise ───────────────────────────────
    T_show = 120.0
    t_det, V_det, *_ = simulate(p, V0=8.0, C0=5.0, T=T_show)
    for k, seed in enumerate((1, 2, 3)):
        t_s, V_s, _ = simulate_sde(p, 8.0, 5.0, T=T_show, sigma_V=0.2,
                                   seed=seed)
        ax1.plot(t_s, V_s, lw=0.7, alpha=0.7,
                 label='SDE paths ($\\sigma_V=0.2$)' if k == 0 else None)
    ax1.plot(t_det, V_det, 'k-', lw=2.0, label='deterministic')
    ax1.axhline(V_star, color='crimson', ls='--', lw=1.5,
                label=f'$V^* = {V_star:.2f}$')
    ax1.set_xlabel('time')
    ax1.set_ylabel('vitality $V(t)$')
    ax1.set_title('(a) The attractor under noise')
    ax1.set_ylim(0, 10.5)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper right', framealpha=0.95)

    # ── Panel (b): persistence / escape vs sigma_V ─────────────────────────
    sigmas = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    curve = escape_curve(p, sigmas, n_paths=20, T=200.0, burn_in=100.0,
                         seed=1)
    ax2.plot(curve['sigmas'], curve['in_band_fraction'], 'o-', color='teal',
             lw=2, label='in-band fraction')
    ax2.plot(curve['sigmas'], curve['escape_fraction'], 's-',
             color='darkred', lw=2, label='escape fraction')
    ax2.set_xlabel('noise amplitude $\\sigma_V$')
    ax2.set_ylabel('fraction')
    ax2.set_title('(b) Persistence vs noise')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='center left', framealpha=0.95)

    # ── Panel (c): identifiability ridges ─────────────────────────────────
    panel_eq = synthesize_panel(
        p, n_leaders=8, n_waves=10, wave_dt=2.0, obs_noise=0.02, seed=7,
        sigma_V=0.02,
        V0_range=(V_star - 0.5, V_star + 0.5),
        C0_range=(C_star - 3.0, C_star + 3.0),
    )
    panel_tr = synthesize_panel(
        p, n_leaders=8, n_waves=10, wave_dt=2.0, obs_noise=0.02, seed=3,
        sigma_V=0.02,
    )
    prof_eq = ridge_profile(panel_eq, dict(p), pair=('mu', 'alpha'))
    prof_tr = ridge_profile(panel_tr, dict(p), pair=('mu', 'alpha'))

    floor = 1e-4  # display floor for log scale
    s = prof_eq['scales']
    ax3.semilogy(s, np.abs(prof_eq['excess_single']) + floor, 'o-',
                 color='darkred', lw=2,
                 label=r'$\mu$ alone (equilibrium panel)')
    ax3.semilogy(s, np.abs(prof_eq['excess_joint']) + floor, 'o-',
                 color='teal', lw=2,
                 label=r'joint $(\mu,\alpha)$ — flat ridge')
    ax3.semilogy(s, np.abs(prof_tr['excess_joint']) + floor, 's--',
                 color='slategray', lw=1.5,
                 label=r'joint $(\mu,\alpha)$, transient panel')
    ax3.axvline(1.0, color='k', lw=0.8, alpha=0.4)
    ax3.set_xlabel(r'scale $s$: $(\mu,\alpha)\to(s\mu,s\alpha)$ or $\mu\to s\mu$')
    ax3.set_ylabel('|excess loss| (log)')
    ax3.set_title('(c) The $\\mu/\\alpha$ ridge:\nflat at equilibrium, '
                  'broken by transients')
    ax3.grid(True, alpha=0.3, linestyle=':', which='both')
    ax3.legend(loc='lower right', framealpha=0.95)

    fig.suptitle('Stochastic robustness of the interior attractor and '
                 'executable identifiability ridges',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'fig12_stochastic_robustness.{ext}')
    plt.close(fig)

    print('[OK] Fig 12 saved to figures/')
    print('\n' + '=' * 70)
    print('STOCHASTIC ROBUSTNESS & IDENTIFIABILITY SUMMARY')
    print('=' * 70)
    print(f"target (V*, C*)        : ({V_star:.4f}, {C_star:.4f})")
    for sig, ib, esc in zip(curve['sigmas'], curve['in_band_fraction'],
                            curve['escape_fraction']):
        print(f"  sigma_V={sig:<5}: in-band={ib:.3f}  escape={esc:.3f}")
    print(f"ridge flatness (equilibrium panel): {prof_eq['ridge_flatness']:.4f}")
    print(f"ridge flatness (transient panel)  : {prof_tr['ridge_flatness']:.4f}")
    print('=' * 70 + '\n')
    return curve, prof_eq, prof_tr


if __name__ == '__main__':
    print('DLVT Figure 12 — stochastic robustness + identifiability ridges\n')
    curve, prof_eq, prof_tr = fig12_stochastic_robustness()
    assert prof_eq['ridge_flatness'] < 0.05, 'equilibrium ridge should be flat'
    print('Figure 12 complete.')
