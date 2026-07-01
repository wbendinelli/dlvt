#!/usr/bin/env python3
"""
DLVT — Figure 11: Nondimensionalization & Global Sensitivity (Task 9)

Three panels, all computed by dlvt.nondimensional (no re-implementation):
  (a) Tornado chart of |d ln V*/d ln p_i| for all 11 raw parameters,
      exposing the exact zero elasticities of beta, eta, O0.
  (b) Regime boundary map in the (mu/alpha, phi) plane with the baseline
      calibration marked; the baseline sits ~8% below the flip at
      (mu/alpha)_crit ~ 2.163.
  (c) LHS regime fractions (log-uniform Latin hypercube, factor 2, seed 1)
      plus Spearman rank correlations between log-parameters and V*
      (a rank-correlation screening, not a Sobol decomposition).

Deterministic (fixed seed).  Writes figures/fig11_sensitivity_global.pdf/.png
and prints a summary to stdout.
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dlvt.model import make_params
from dlvt.nondimensional import (
    PARAM_NAMES,
    lhs_zombie_fraction,
    mu_alpha_critical,
    reduced_groups,
    v_star_elasticities,
    zombie_boundary_map,
    zombie_boundary_map_beta,
)

rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'text.usetex': False, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Validated palette (dataviz reference instance, light mode)
BLUE    = '#2a78d6'   # categorical slot 1 / diverging cool pole
RED     = '#e34948'   # categorical slot 6 / diverging warm pole
INK     = '#0b0b0b'   # primary ink
INK2    = '#52514e'   # secondary ink
MUTED   = '#898781'   # axis / tick labels
GRID    = '#e1e0d9'   # hairline gridlines
AXISCOL = '#c3c2b7'   # baseline / axis
SURFACE = '#fcfcfb'   # chart surface

SEED = 1
GREEK = {
    'R': 'R', 'Vmax': r'$V_{\max}$', 'delta': r'$\delta$',
    'gamma': r'$\gamma$', 'O0': r'$O_0$', 'beta': r'$\beta$',
    'eta': r'$\eta$', 'alpha': r'$\alpha$', 'phi': r'$\varphi$',
    'mu': r'$\mu$', 'eps': r'$\varepsilon$',
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXISCOL)
    ax.tick_params(colors=MUTED, labelcolor=INK2)


def panel_a_tornado(ax, els):
    """Tornado chart of |elasticities|, exact zeros called out."""
    order = sorted(PARAM_NAMES, key=lambda k: abs(els[k]), reverse=True)
    mags = [abs(els[k]) for k in order]
    ypos = np.arange(len(order))[::-1]

    ax.barh(ypos, mags, height=0.62, color=BLUE, zorder=3)
    for y, k, m in zip(ypos, order, mags):
        signed = els[k]
        if k in ('beta', 'eta', 'O0'):
            label = '0 (exact)'
        else:
            label = f'{signed:+.3f}'
        ax.text(m + 0.02, y, label, va='center', ha='left',
                fontsize=8.5, color=INK2, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels([GREEK[k] for k in order], color=INK)
    ax.set_xlabel(r'$|\partial \ln V^{*} / \partial \ln p_i|$ (baseline)')
    ax.set_xlim(0, max(mags) * 1.28)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title('(a) Local elasticities of equilibrium vitality $V^{*}$',
                 loc='left', color=INK)


def panel_b_boundary(ax, bmap, r_crit_baseline, boundary_curve):
    """Regime map over (mu/alpha, phi) with baseline and flip marked."""
    codes = np.vectorize({'none': 0, 'zombie': 1, 'sustainable': 2}.get)(
        bmap['regimes']).astype(float)
    present = sorted(set(codes.ravel().astype(int)))
    cmap_full = {0: GRID, 1: RED, 2: BLUE}
    cmap = ListedColormap([cmap_full[c] for c in present])
    # remap codes to consecutive levels for the ListedColormap
    remap = {c: i for i, c in enumerate(present)}
    lv = np.vectorize(remap.get)(codes.astype(int))
    ax.pcolormesh(bmap['r_values'], bmap['phi_values'], lv, cmap=cmap,
                  shading='nearest', rasterized=True, alpha=0.85)

    # mesh extent (pcolormesh cells are centred on the grid values)
    dr = bmap['r_values'][1] - bmap['r_values'][0]
    dphi = bmap['phi_values'][1] - bmap['phi_values'][0]
    r_lo, r_hi = bmap['r_values'][0] - dr / 2, bmap['r_values'][-1] + dr / 2

    # exact flip curve (mu/alpha)_crit(phi), clipped to the map extent
    phis_c, rs_c = boundary_curve
    inside = (rs_c >= r_lo) & (rs_c <= r_hi)
    ax.plot(rs_c[inside], phis_c[inside], color=INK, lw=1.6,
            label=r'flip: $V^{*}=0.5\,V_{\max}$')
    ax.set_xlim(r_lo, r_hi)
    ax.set_ylim(bmap['phi_values'][0] - dphi / 2,
                bmap['phi_values'][-1] + dphi / 2)

    r0, phi0 = bmap['baseline']
    ax.plot([r0], [phi0], marker='o', ms=8, mfc=SURFACE, mec=INK, mew=1.6,
            ls='none', label=rf'baseline $(\mu/\alpha={r0:g},\ \varphi={phi0:g})$',
            zorder=5)
    ax.annotate(
        rf'flip at $\mu/\alpha \approx {r_crit_baseline:.3f}$'
        '\n' + rf'($+{100 * (r_crit_baseline / r0 - 1.0):.0f}\%$ from baseline)',
        xy=(r_crit_baseline, phi0), xytext=(r_crit_baseline + 0.45, phi0 + 0.075),
        fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))

    # direct region labels (secondary encoding on top of color)
    ax.text(1.25, 0.055, 'zombie', color=SURFACE, fontsize=10, weight='bold')
    ax.text(3.05, 0.19, 'sustainable', color=SURFACE, fontsize=10,
            weight='bold')

    ax.set_xlabel(r'depreciation-to-accumulation ratio $\mu/\alpha$')
    ax.set_ylabel(r'complexity-impact suppression $\varphi$')
    style_axes(ax)
    leg = ax.legend(loc='upper right', frameon=True, labelcolor=INK2,
                    facecolor=SURFACE, edgecolor=AXISCOL, framealpha=0.95)
    leg.set_zorder(6)
    ax.set_title(r'(b) Regime boundary in $(\mu/\alpha,\ \varphi)$'
                 ' — baseline sits ~8% from the flip',
                 loc='left', color=INK)


def panel_c_fractions(ax, lhs):
    """Bar chart of LHS regime fractions."""
    labels = ['stable\nequilibrium', 'zombie |\nstable', 'zombie\noverall']
    vals = [lhs['frac_stable'], lhs['zombie_fraction_given_stable'],
            lhs['zombie_fraction_overall']]
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.56, color=BLUE, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025, f'{v:.2f}', ha='center', va='bottom',
                fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('fraction of LHS draws')
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title(f'(c) LHS outcome fractions\n'
                 f'(n={lhs["n_samples"]}, log-uniform '
                 rf'$\times/\div${lhs["factor"]:g}, seed={lhs["seed"]})',
                 loc='left', color=INK)


def panel_c_spearman(ax, lhs, top=8):
    """Signed Spearman rank correlations (diverging blue/red)."""
    sp = lhs['spearman']
    order = sorted(PARAM_NAMES, key=lambda k: abs(sp[k]), reverse=True)[:top]
    vals = [sp[k] for k in order]
    ypos = np.arange(len(order))[::-1]
    colors = [BLUE if v >= 0 else RED for v in vals]
    ax.barh(ypos, vals, height=0.6, color=colors, zorder=3)
    for y, v in zip(ypos, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f'{v:+.2f}',
                va='center', ha='left' if v >= 0 else 'right',
                fontsize=8.5, color=INK2, zorder=4)
    ax.axvline(0.0, color=AXISCOL, lw=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([GREEK[k] for k in order], color=INK)
    lim = max(abs(v) for v in vals) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r'Spearman $\rho\,(\ln p_i,\ V^{*})$ over stable draws')
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    legend_handles = [Patch(color=BLUE, label=r'raises $V^{*}$'),
                      Patch(color=RED, label=r'lowers $V^{*}$')]
    ax.legend(handles=legend_handles, loc='lower right', frameon=False,
              labelcolor=INK2)
    ax.set_title('(c′) Rank-correlation screening of $V^{*}$\n'
                 '(global-sensitivity proxy — not Sobol)',
                 loc='left', color=INK)


def main():
    p = make_params()
    fig_dir = REPO_ROOT / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # -- computations (all from dlvt.nondimensional) --------------------------
    groups = reduced_groups(p)
    els = v_star_elasticities(p)
    r_crit = mu_alpha_critical(p)
    bmap = zombie_boundary_map(p, n=41)
    lhs = lhs_zombie_fraction(p, n_samples=600, factor=2.0, seed=SEED)

    # exact flip curve for panel (b)
    phis_c = np.linspace(bmap['phi_values'][0], bmap['phi_values'][-1], 15)
    rs_c = []
    for phi in phis_c:
        p_phi = dict(p)
        p_phi['phi'] = float(phi)
        rs_c.append(mu_alpha_critical(p_phi, lo=0.5, hi=10.0, tol=1e-4))
    boundary_curve = (phis_c, np.array(rs_c))

    # beta-invariance demonstration (printed; the boundary is vertical in beta)
    bmap_beta = zombie_boundary_map_beta(p, n=11)

    # -- figure ----------------------------------------------------------------
    fig = plt.figure(figsize=(12.5, 8.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 5, hspace=0.42, wspace=1.1)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    ax_c1 = fig.add_subplot(gs[1, :2])
    ax_c2 = fig.add_subplot(gs[1, 2:])

    panel_a_tornado(ax_a, els)
    panel_b_boundary(ax_b, bmap, r_crit, boundary_curve)
    panel_c_fractions(ax_c1, lhs)
    panel_c_spearman(ax_c2, lhs)

    fig.suptitle(
        'DLVT nondimensionalization & global sensitivity — '
        r'reduced groups $\rho, \kappa, a, f, e, \gamma$ '
        rf"($\rho$={groups['rho']:g}, $\kappa$={groups['kappa']:g}, "
        rf"$a$={groups['a']:g}, $f$={groups['f']:g}, "
        rf"$e$={groups['e']:g}, $\gamma$={groups['gamma']:g})",
        fontsize=11.5, color=INK, y=0.985)

    out_pdf = fig_dir / 'fig11_sensitivity_global.pdf'
    out_png = fig_dir / 'fig11_sensitivity_global.png'
    fig.savefig(out_pdf, facecolor=SURFACE)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)

    # -- stdout summary ----------------------------------------------------------
    print('=' * 72)
    print('Figure 11 — nondimensionalization & global sensitivity (seed=1)')
    print('=' * 72)
    print('Dimensionless groups (eta=1 reduced form, 6 of 11 raw params):')
    for k, v in groups.items():
        print(f'  {k:6s} = {v:g}')
    print('\nElasticities d ln V*/d ln p_i at baseline:')
    for k in sorted(PARAM_NAMES, key=lambda k: abs(els[k]), reverse=True):
        tag = '   <- exactly 0 (structural)' if k in ('beta', 'eta', 'O0') \
            else ''
        print(f'  {k:6s} {els[k]:+.6f}{tag}')
    print(f'\n(mu/alpha)_crit at baseline phi: {r_crit:.4f} '
          f'(baseline mu/alpha = {p["mu"] / p["alpha"]:g}; '
          f'{100 * (r_crit / (p["mu"] / p["alpha"]) - 1):.1f}% from flip)')
    print(f'\nBoundary map ({len(bmap["r_values"])}x{len(bmap["phi_values"])} '
          f'grid over (mu/alpha, phi)): '
          f'{np.sum(bmap["regimes"] == "zombie")} zombie / '
          f'{np.sum(bmap["regimes"] == "sustainable")} sustainable / '
          f'{np.sum(bmap["regimes"] == "none")} none cells')
    print(f'Beta-invariance check ((mu/alpha, beta) grid): boundary '
          f'invariant in beta = {bmap_beta["boundary_invariant_in_beta"]}')
    print(f'\nLHS screening (n={lhs["n_samples"]}, factor='
          f'{lhs["factor"]:g}, seed={lhs["seed"]}):')
    print(f'  stable-equilibrium fraction : {lhs["frac_stable"]:.3f} '
          f'({lhs["n_stable"]}/{lhs["n_samples"]})')
    print(f'  P(zombie | stable)          : '
          f'{lhs["zombie_fraction_given_stable"]:.3f} '
          f'({lhs["n_zombie"]}/{lhs["n_stable"]})')
    print(f'  P(zombie) overall           : '
          f'{lhs["zombie_fraction_overall"]:.3f}')
    print('  Top Spearman |rho| (log-param vs V*, stable draws; '
          'rank screening, not Sobol):')
    for k in sorted(PARAM_NAMES, key=lambda k: abs(lhs['spearman'][k]),
                    reverse=True)[:6]:
        print(f'    {k:6s} {lhs["spearman"][k]:+.3f}')
    print(f'\nWrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
