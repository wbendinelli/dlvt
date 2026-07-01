#!/usr/bin/env python3
"""
DLVT — Figure 8: The scan-window artifact and scope absorption (Appendix A8).

HISTORY / PURPOSE. Earlier drafts of this script re-implemented the model,
scanned β with a *fixed* equilibrium-search window (C_max = 80), and reported
a "critical coupling" β_crit ≈ 0.1015 with an apparent hysteresis loop. Both
findings were artifacts of the fixed window: the equilibrium capital scales as
C*(β) = (β·C*)/β ≈ 8.008/β, so for β < 8.008/80 ≈ 0.10 the (perfectly
existing) equilibrium simply fell outside the scan window and the sweep
mislabeled the regime. There is no V*-crossing bifurcation and no hysteresis
in β at the baseline calibration: V*(β) is constant (scope absorption), and
det J > 0 everywhere rules out folds.

This script now *illustrates the artifact honestly* instead of reproducing it
as a finding:

  Panel (a): the legacy sweep with the fixed C_max = 80 window — the stable
             branch appears to terminate near β ≈ 0.10 ("β_crit"), which is
             exactly where C*(β) crosses the window edge (dotted curve).
  Panel (b): the corrected sweep with the analytical per-β window
             (C_max derived from C_trap ∝ 1/β): V*(β) is flat at 4.7025 for
             every β — the scope-absorption invariance, with β·C* conserved.

All model math is imported from the `dlvt` package; nothing is re-implemented
here. See `dlvt.analysis.estimate_bifurcation_interval` for the programmatic
version of this diagnostic, and Appendix A8 of the manuscript.
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

from dlvt.model import make_params                      # noqa: E402
from dlvt.analysis import (                              # noqa: E402
    V_STRATEGIC_FRACTION,
    estimate_bifurcation_interval,
    find_interior_equilibria,
)

OUTPUT_DIR = REPO_DIR / 'figures'

rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

LEGACY_C_MAX = 80.0  # the fixed window that produced the 0.1015 artifact


def sweep_v_star(betas, C_max=None):
    """Stable-branch V*(β) and C*(β) using find_interior_equilibria.

    C_max=None uses the corrected analytical per-β window; a fixed float
    reproduces the legacy (artifact-generating) behaviour.
    """
    out = []
    for b in betas:
        p = make_params(beta=b)
        eqs = find_interior_equilibria(p, C_max=C_max)
        stable = [e for e in eqs if e['stable']]
        if stable:
            stable.sort(key=lambda e: e['C'])
            out.append((b, stable[-1]['V'], stable[-1]['C']))
    return np.array(out) if out else np.empty((0, 3))


def fig8_scan_window_artifact():
    p = make_params()
    V_strategic = V_STRATEGIC_FRACTION * p['Vmax']
    betas = np.linspace(0.02, 1.5, 400)

    legacy = sweep_v_star(betas, C_max=LEGACY_C_MAX)
    correct = sweep_v_star(betas, C_max=None)

    # Programmatic diagnostic (small grid; the full grid is in Appendix A8).
    diag = estimate_bifurcation_interval(
        p, eps_grid=[0.1], n_scan_grid=[8000], n_beta=80,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # ── Panel (a): the artifact ────────────────────────────────────────────
    if legacy.size:
        ax1.plot(legacy[:, 0], legacy[:, 1], 'g-', lw=2.5,
                 label='Stable branch found (fixed $C_{max}=80$ window)')
        beta_edge = legacy[:, 0].min()
        ax1.axvline(beta_edge, color='purple', ls=':', lw=2, alpha=0.7,
                    label=f'apparent "$\\beta_{{crit}}$" ≈ {beta_edge:.3f} (artifact)')
    ax1.axhline(V_strategic, color='orange', ls='-.', alpha=0.6, lw=1.5,
                label=f'$V_{{strategic}}$ = {V_strategic:.1f}')
    # Where the equilibrium leaves the fixed window: C*(β)=bC*/β crosses 80.
    bC = diag['baseline_beta_C_product'] or 8.008
    ax1.axvspan(0.02, bC / LEGACY_C_MAX, alpha=0.12, color='red',
                label=f'equilibrium outside window ($C^*>{LEGACY_C_MAX:.0f}$)')
    ax1.set_title('(a) Legacy fixed scan window:\napparent branch termination is an artifact')
    ax1.set_xlabel(r'Capital–complexity coupling $\beta$')
    ax1.set_ylabel(r'Equilibrium vitality $V^*$')
    ax1.set_xlim(0.0, 1.5)
    ax1.set_ylim(0, 10.5)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax1.legend(loc='upper right', framealpha=0.95)

    # ── Panel (b): the corrected sweep ─────────────────────────────────────
    if correct.size:
        ax2.plot(correct[:, 0], correct[:, 1], 'g-', lw=2.5,
                 label=r'Stable branch, analytical per-$\beta$ window')
        v_inv = float(np.median(correct[:, 1]))
        ax2.annotate(
            f'scope absorption: $V^*(\\beta) \\equiv {v_inv:.4f}$\n'
            f'$\\beta \\cdot C^* = {bC:.3f}$ conserved (ε = 0.1)',
            xy=(0.75, v_inv), xytext=(0.55, 7.6),
            arrowprops=dict(arrowstyle='->', alpha=0.6), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        )
    ax2.axhline(V_strategic, color='orange', ls='-.', alpha=0.6, lw=1.5,
                label=f'$V_{{strategic}}$ = {V_strategic:.1f}')
    ax2.set_title('(b) Corrected window:\nno crossing, no hysteresis — $V^*(\\beta)$ is flat')
    ax2.set_xlabel(r'Capital–complexity coupling $\beta$')
    ax2.set_xlim(0.0, 1.5)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax2.legend(loc='upper right', framealpha=0.95)

    fig.suptitle('The scan-window artifact behind the retracted '
                 r'"$\beta_{crit} \approx 0.1015$" claim (Appendix A8)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUTPUT_DIR / f'fig8_bifurcation_hysteresis.{ext}')
    plt.close(fig)

    print('[OK] Fig 8 saved to figures/')
    print('\n' + '=' * 70)
    print('SCAN-WINDOW ARTIFACT DIAGNOSTIC (Appendix A8)')
    print('=' * 70)
    print(f"crosses_threshold : {diag['crosses_threshold']}")
    print(f"beta_crit_interval: {diag['beta_crit_interval']}")
    print(f"V* invariant      : {diag['v_star_invariant']}")
    print(f"beta*C* invariant : {diag['baseline_beta_C_product']}")
    print(f"diagnostic        : {diag['diagnostic']}")
    print('=' * 70 + '\n')
    return diag


if __name__ == '__main__':
    print('DLVT Figure 8 — scan-window artifact and scope absorption\n')
    result = fig8_scan_window_artifact()
    assert result['crosses_threshold'] is False, (
        'Unexpected V*-crossing at baseline: the scope-absorption invariance '
        'should make V*(beta) flat. Investigate before publishing figures.'
    )
    print('Figure 8 complete: artifact illustrated, no bifurcation at baseline.')
