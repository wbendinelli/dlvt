#!/usr/bin/env python3
"""
DLVT — Figure 10: Recovery Policy Comparison (intervention asymmetry).

Key structural insight (Scope-Absorption Corollary): equilibrium vitality V*
is invariant to the capital–complexity coupling β, so a β reduction produces
only a transient vitality bump — capital re-expands until β·C* is restored.
Only recovery-side interventions (R↑) shift V* durably; combining both is
fastest (β↓ buys transient relief while R↑ moves the destination).

All model math is imported from the `dlvt` package; the two-phase experiment
switches the parameter dictionary at the intervention time t = 20.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from dlvt.model import make_params, dlvt_system   # noqa: E402
from dlvt.analysis import V_STRATEGIC_FRACTION    # noqa: E402

OUTPUT_DIR = REPO_DIR / 'figures'


def solve_two_phase(p_before, p_after, y0=(3.0, 15.0), t_int=20.0, t_end=120.0):
    """Integrate with parameters p_before on [0, t_int], p_after afterwards."""
    kw = dict(method='RK45', dense_output=True, max_step=0.1,
              rtol=1e-8, atol=1e-10)
    sol1 = solve_ivp(dlvt_system, (0.0, t_int), list(y0), args=(p_before,), **kw)
    sol2 = solve_ivp(dlvt_system, (t_int, t_end), sol1.y[:, -1],
                     args=(p_after,), **kw)
    t = np.concatenate([sol1.t, sol2.t])
    V = np.concatenate([sol1.y[0], sol2.y[0]])
    C = np.concatenate([sol1.y[1], sol2.y[1]])
    return t, V, C


def main():
    p0 = make_params()
    V_strat = V_STRATEGIC_FRACTION * p0['Vmax']

    scenarios = [
        ('No intervention', p0, dict(color='k', ls='--', lw=2, alpha=0.6)),
        (r'Reduce $\beta$ by 60%', make_params(beta=0.10),
         dict(color='#2196F3', lw=2, alpha=0.85)),
        (r'Increase $R$ by 40%', make_params(R=4.2),
         dict(color='#4CAF50', lw=2, alpha=0.85)),
        ('Both simultaneously', make_params(R=4.2, beta=0.10),
         dict(color='#FF5722', lw=2.5, alpha=0.85)),
    ]

    runs = [(name, *solve_two_phase(p0, p_after), style)
            for name, p_after, style in scenarios]

    print('Intervention comparison (switch at t=20, from V0=3.0, C0=15.0):')
    for name, t, V, C, _ in runs:
        print(f'  {name:26s}: V(120)={V[-1]:.3f}  C(120)={C[-1]:.3f}  '
              f'escape={V[-1] > V_strat}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    for name, t, V, C, style in runs:
        ax1.plot(t, V, label=name, **style)
        ax2.plot(t, C, label=name, **style)

    ax1.axhline(V_strat, color='gray', ls=':', lw=1.2, alpha=0.5)
    ax1.axvline(20, color='gray', ls=':', lw=1.2, alpha=0.5)
    ax1.text(21, 2.2, 'Intervention', fontsize=9, color='gray', style='italic')
    ax1.text(85, V_strat + 0.15, r'$V_{\mathrm{strategic}}$', fontsize=10,
             color='gray')
    ax1.set_xlabel('Time (arbitrary units)', fontsize=11)
    ax1.set_ylabel(r'Vitality $V(t)$', fontsize=11)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(2, 7)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.92)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Vitality trajectories', fontsize=11, pad=10)

    ax2.axvline(20, color='gray', ls=':', lw=1.2, alpha=0.5)
    ax2.text(21, 5, 'Intervention', fontsize=9, color='gray', style='italic')
    ax2.set_xlabel('Time (arbitrary units)', fontsize=11)
    ax2.set_ylabel(r'Career Capital $C(t)$', fontsize=11)
    ax2.set_xlim(0, 120)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.92)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Career capital trajectories', fontsize=11, pad=10)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'fig10_intervention_comparison.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved: {OUTPUT_DIR}/fig10_intervention_comparison.pdf')

    with open(OUTPUT_DIR / 'fig10_summary.txt', 'w') as f:
        f.write('Fig 10: Intervention Comparison\n\n')
        f.write('KEY STRUCTURAL INSIGHT:\n')
        f.write('V* at equilibrium is beta-INDEPENDENT. C adjusts to '
                'restore the same O*.\n')
        f.write('Only R shifts V*. Beta reduction raises C*_max but not V*.\n\n')
        f.write('Scenarios (intervention at t=20, from V0=3.0, C0=15.0):\n')
        for name, t, V, C, _ in runs:
            f.write(f'  {name:26s}: V(120)={V[-1]:.3f}, C(120)={C[-1]:.1f}\n')
    print('Summary saved')


if __name__ == '__main__':
    main()
