#!/usr/bin/env python3
"""
run_all_figures.py
==================
Master script to generate all publication figures for the DLVT paper.

This script orchestrates the generation of Figures 1–10:
  - Figures 1–7: Core results (from dlvt.figures)
  - Figures 8–10: Extended analysis (specialized scripts)

All output is saved to 'figures/' with both PDF and PNG formats.

Usage
-----
    python3 scripts/run_all_figures.py              # All figures
    python3 scripts/run_all_figures.py --fig 1-7   # Core only
    python3 scripts/run_all_figures.py --fig 8-10  # Extended only
    python3 scripts/run_all_figures.py --fig 1,3,5 # Selective
"""

import os
import sys
import argparse
import subprocess
import importlib.util
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_DIR))

FIGURES = {
    '1': {
        'name': 'Temporal Evolution',
        'description': 'Temporal evolution of V, C, O and depletion ratio Γ',
        'method': 'dlvt.figures.fig1',
        'output_name': 'fig1_temporal_evolution',
    },
    '2': {
        'name': 'Three Outcome Scenarios',
        'description': 'Sustainable, zombie, and collapse regimes',
        'method': 'dlvt.figures.fig2',
        'output_name': 'fig2_three_outcomes',
    },
    '3': {
        'name': 'Phase Portrait',
        'description': 'Phase portrait with nullclines and trajectories',
        'method': 'dlvt.figures.fig3',
        'output_name': 'fig3_phase_portrait',
    },
    '4': {
        'name': 'Bifurcation Diagrams',
        'description': 'C* and V* vs β; C* vs R',
        'method': 'dlvt.figures.fig4',
        'output_name': 'fig4_bifurcation',
    },
    '5': {
        'name': 'Leadership Impact',
        'description': 'DLVT vs Human Capital Theory (Becker)',
        'method': 'dlvt.figures.fig5',
        'output_name': 'fig5_impact_comparison',
    },
    '6': {
        'name': 'Carrying Capacity Heatmap',
        'description': 'C*(β, R) parameter space heatmap',
        'method': 'dlvt.figures.fig6',
        'output_name': 'fig6_carrying_capacity',
    },
    '7': {
        'name': 'Leadership Regime Map',
        'description': 'Regime classification in (β, δ) parameter space',
        'method': 'dlvt.figures.fig7',
        'output_name': 'fig7_regime_map',
    },
    '8': {
        'name': 'Bifurcation with Hysteresis',
        'description': 'β parameter scan, hysteresis detection',
        'method': 'scripts.fig8_bifurcation_hysteresis.main',
        'script': str(SCRIPT_DIR / 'fig8_bifurcation_hysteresis.py'),
    },
    '9': {
        'name': 'Structural Robustness',
        'description': 'Sensitivity analysis: γ and η alternatives',
        'method': 'scripts.fig9_robustness.main',
        'script': str(SCRIPT_DIR / 'fig9_robustness.py'),
    },
    '10': {
        'name': 'Intervention Comparison',
        'description': 'Recovery policies: β reduction vs R increase',
        'method': 'scripts.fig10_intervention_comparison.main',
        'script': str(SCRIPT_DIR / 'fig10_intervention_comparison.py'),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fig_from_dlvt(fig_num):
    """Generate Figure 1–7 from dlvt.figures module."""
    try:
        from dlvt import figures
        fig_func = getattr(figures, f'fig{fig_num}')
        fig_func(output_dir=str(OUTPUT_DIR))
        return True
    except Exception as e:
        print(f'  ERROR: {e}')
        return False


def generate_fig_from_script(script_path):
    """Generate Figure 8–10 by running standalone script."""
    try:
        result = subprocess.run(
            ['python3', script_path],
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f'  ERROR: Script returned code {result.returncode}')
            if result.stderr:
                print(f'  stderr: {result.stderr}')
            return False
        if result.stdout:
            print(f'  {result.stdout.strip()}')
        return True
    except subprocess.TimeoutExpired:
        print(f'  ERROR: Script timeout (>300s)')
        return False
    except Exception as e:
        print(f'  ERROR: {e}')
        return False


def generate_figure(fig_num):
    """Generate a single figure by number."""
    if fig_num not in FIGURES:
        print(f'Figure {fig_num} not found in FIGURES dict')
        return False

    fig_info = FIGURES[fig_num]
    name = fig_info['name']
    description = fig_info['description']

    print('\n' + '-' * 80)
    print(f'Figure {fig_num}: {name}')
    print(f'  {description}')
    print('-' * 80)

    # Route to appropriate generator
    if 'script' in fig_info:
        success = generate_fig_from_script(fig_info['script'])
    else:
        success = generate_fig_from_dlvt(int(fig_num))

    status = '[OK]' if success else '[FAIL]'
    print(f'{status} Figure {fig_num} generation complete')
    return success


def parse_figure_spec(spec):
    """Parse figure specification string (e.g., '1-7', '8-10', '1,3,5')."""
    figures_to_run = set()

    # Handle comma-separated list
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            figures_to_run.update(str(i) for i in range(start, end + 1))
        else:
            # Single figure
            figures_to_run.add(part)

    # Validate
    invalid = figures_to_run - set(FIGURES.keys())
    if invalid:
        print(f'Invalid figure numbers: {invalid}')
        return None

    return sorted(figures_to_run, key=int)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate all DLVT paper figures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 run_all_figures.py              # All figures (1–10)
  python3 run_all_figures.py --fig 1-7   # Core figures
  python3 run_all_figures.py --fig 8-10  # Extended figures
  python3 run_all_figures.py --fig 1,3,5 # Selective
        '''.strip(),
    )

    parser.add_argument(
        '--fig', '--figures',
        default='1-10',
        help='Figure range or list (default: 1-10)',
    )
    parser.add_argument(
        '--output-dir',
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output',
    )

    args = parser.parse_args()

    # Parse figure specification
    figures_to_run = parse_figure_spec(args.fig)
    if figures_to_run is None:
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Header
    print('\n' + '=' * 80)
    print('DLVT Figure Generation Pipeline')
    print('=' * 80)
    print(f'Output directory: {args.output_dir}')
    print(f'Figures to generate: {", ".join(figures_to_run)}')
    print('═' * 80)

    # Generate figures
    results = {}
    for fig_num in figures_to_run:
        results[fig_num] = generate_figure(fig_num)

    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for fig_num in sorted(results.keys(), key=int):
        status = '[OK]' if results[fig_num] else '[FAIL]'
        fig_name = FIGURES[fig_num]['name']
        print(f'  {status}  Figure {fig_num}: {fig_name}')

    print('-' * 80)
    print(f'Total: {passed}/{total} figures generated')

    if failed > 0:
        print(f'Failed: {failed} figure(s)')
        return 1

    print('All figures generated successfully!')
    print('=' * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
