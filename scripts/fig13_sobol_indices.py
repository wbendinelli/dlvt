#!/usr/bin/env python3
"""
DLVT — Figure 13: Variance-based (Sobol') global sensitivity of V*

Upgrades the LHS+Spearman screening (Figure 11) to a FORMAL variance
decomposition.  A single grouped horizontal bar chart shows, for each of the
11 raw parameters, the first-order (S1) and total-effect (ST) Sobol' indices
of the equilibrium vitality V*, sorted by ST.  All numbers come from the
Jansen (1999) estimator on a scrambled-Sobol' (Saltelli) design in
:func:`dlvt.sobol.sobol_indices` — nothing is re-implemented here.

The chart is annotated with the EXACT structural facts that serve as the
estimator's correctness oracle:
  * beta, eta, O0 have exactly zero effect on V* (they move only C*) -> ST ~ 0;
  * mu, alpha act on V* only through the ratio mu/alpha;
  * R, delta act on V* only through the ratio R/delta.

Deterministic (fixed seed).  Writes figures/fig13_sobol_indices.pdf/.png and
prints the index table to stdout.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

# The only DLVT dependency is the new formal-sensitivity module.
from dlvt.sobol import sobol_indices, make_params

rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 10,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Validated palette (dataviz reference instance, light mode)
BLUE    = '#2a78d6'   # categorical slot 1 -> total effect ST
AQUA    = '#1baf7a'   # categorical slot 2 -> first order  S1
INK     = '#0b0b0b'   # primary ink
INK2    = '#52514e'   # secondary ink
MUTED   = '#898781'   # axis / tick labels
GRID    = '#e1e0d9'   # hairline gridlines
AXISCOL = '#c3c2b7'   # baseline / axis
SURFACE = '#fcfcfb'   # chart surface

SEED = 1
N_BASE = 256
FACTOR = 2.0

GREEK = {
    'R': r'$R$', 'Vmax': r'$V_{\max}$', 'delta': r'$\delta$',
    'gamma': r'$\gamma$', 'O0': r'$O_0$', 'beta': r'$\beta$',
    'eta': r'$\eta$', 'alpha': r'$\alpha$', 'phi': r'$\varphi$',
    'mu': r'$\mu$', 'eps': r'$\varepsilon$',
}
# Structural annotations attached to the y-axis tick labels.
STRUCT_ZERO = ('beta', 'eta', 'O0')
RATIO_TAG = {
    'mu': r'$\mu/\alpha$', 'alpha': r'$\mu/\alpha$',
    'R': r'$R/\delta$', 'delta': r'$R/\delta$',
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXISCOL)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def _tick_label(name):
    """Greek symbol plus a structural tag (ratio pair or exact zero)."""
    if name in STRUCT_ZERO:
        return f'{GREEK[name]}'
    if name in RATIO_TAG:
        return f'{GREEK[name]}  ({RATIO_TAG[name]})'
    return GREEK[name]


def make_figure(res, out_pdf, out_png):
    params = res['params']
    S1, ST = res['S1'], res['ST']

    order = list(np.argsort(-ST))            # descending ST -> top of chart
    names = [params[i] for i in order]
    s1 = np.array([S1[i] for i in order])
    st = np.array([ST[i] for i in order])

    ypos = np.arange(len(names))[::-1]
    h = 0.38

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    fig.patch.set_facecolor(SURFACE)

    ax.barh(ypos + h / 2, st, height=h, color=BLUE, zorder=3, label='total effect $S_T$')
    ax.barh(ypos - h / 2, s1, height=h, color=AQUA, zorder=3, label='first order $S_1$')

    xmax = float(max(st.max(), s1.max()))
    for y, name, v in zip(ypos + h / 2, names, st):
        if name in STRUCT_ZERO:
            txt = r'$\approx 0$ (structural)'
        else:
            txt = f'{v:.3f}'
        ax.text(v + 0.006, y, txt, va='center', ha='left', fontsize=8,
                color=BLUE, zorder=4)
    for y, v in zip(ypos - h / 2, s1):
        ax.text(v + 0.006, y, f'{v:.3f}', va='center', ha='left', fontsize=8,
                color=INK2, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([_tick_label(n) for n in names], color=INK)
    ax.set_xlabel('Sobol’ sensitivity index of $V^{*}$ '
                  '(Jansen estimator, scrambled Sobol’ QMC)')
    ax.set_xlim(-0.01, xmax * 1.30)
    ax.axvline(0.0, color=AXISCOL, lw=1.0, zorder=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    # explanatory note: the structural oracle
    note = (
        r'$\beta,\ \eta,\ O_0$ have EXACTLY zero effect on $V^{*}$ '
        '(they move only $C^{*}$).'
        '\n'
        r'$\mu,\alpha$ act only via $\mu/\alpha$;  $R,\delta$ only via $R/\delta$.'
    )
    ax.text(0.985, 0.045, note, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color=INK2,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=SURFACE,
                      edgecolor=AXISCOL, alpha=0.95))

    ax.legend(loc='lower right', bbox_to_anchor=(0.985, 0.18), frameon=True,
              facecolor=SURFACE, edgecolor=AXISCOL, framealpha=0.95,
              labelcolor=INK2)

    ax.set_title(
        f'(fig 13) Variance-based Sobol’ indices of $V^{{*}}$ '
        f'(N={res["n_base"]}, log-uniform $\\times/\\div${FACTOR:g}, seed={SEED}; '
        f'retained {res["retained_fraction"]:.2f})',
        loc='left', color=INK)

    fig.tight_layout()
    fig.savefig(out_pdf, facecolor=SURFACE)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def print_table(res):
    params, S1, ST = res['params'], res['S1'], res['ST']
    order = list(np.argsort(-ST))
    print('=' * 68)
    print('Figure 13 — variance-based Sobol’ indices of V* '
          f'(seed={SEED}, N={res["n_base"]})')
    print('=' * 68)
    print(f'output            : {res["output"]}')
    print(f'total variance    : {res["var_total"]:.6f}')
    print(f'retained fraction : {res["retained_fraction"]:.4f}')
    print(f'model evaluations : {res["n_base"] * (2 + len(params))}')
    print('-' * 68)
    print(f'{"param":<8}{"S1":>10}{"ST":>10}   note')
    print('-' * 68)
    for i in order:
        name = params[i]
        if name in STRUCT_ZERO:
            note = 'structural 0 (moves only C*)'
        elif name in ('mu', 'alpha'):
            note = 'acts via mu/alpha'
        elif name in ('R', 'delta'):
            note = 'acts via R/delta'
        else:
            note = ''
        print(f'{name:<8}{S1[i]:>10.4f}{ST[i]:>10.4f}   {note}')
    print('-' * 68)


def main():
    fig_dir = REPO_ROOT / 'figures'
    fig_dir.mkdir(exist_ok=True)

    res = sobol_indices(make_params(), n_base=N_BASE, factor=FACTOR, seed=SEED,
                        output='V_star')

    out_pdf = fig_dir / 'fig13_sobol_indices.pdf'
    out_png = fig_dir / 'fig13_sobol_indices.png'
    make_figure(res, out_pdf, out_png)
    print_table(res)
    print(f'\nWrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
