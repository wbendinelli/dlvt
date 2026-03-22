#!/usr/bin/env python3
"""
Generate all publication figures for the DLVT paper.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --output figures/
    python scripts/generate_figures.py --fig 1 3 7   # selected figures only
"""

import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlvt.figures import fig1, fig2, fig3, fig4, fig5, fig6, fig7

FIGURE_FUNCS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}


def main():
    parser = argparse.ArgumentParser(
        description='Generate DLVT publication figures (PDF + PNG).')
    parser.add_argument('--output', '-o', default='figures/',
                        help='Output directory (default: figures/)')
    parser.add_argument('--fig', '-f', nargs='+', type=int,
                        choices=list(FIGURE_FUNCS.keys()),
                        help='Specific figure numbers to generate (default: all)')
    args = parser.parse_args()

    targets = args.fig if args.fig else list(FIGURE_FUNCS.keys())
    os.makedirs(args.output, exist_ok=True)

    print(f'Generating {len(targets)} figure(s) → {args.output}\n')
    for n in targets:
        FIGURE_FUNCS[n](output_dir=args.output)

    print(f'\n✓ Done. Figures saved to: {args.output}')


if __name__ == '__main__':
    main()
