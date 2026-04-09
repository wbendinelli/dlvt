# Contributing to DLVT

Thank you for your interest in contributing to the Dynamic Leadership Vitality Theory project!

## Code of Conduct

Be respectful and collaborative. We're building a scientific tool for everyone.

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/wbendinelli/dlvt.git
cd dlvt
pip install -e ".[dev]"
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting:
  ```bash
  black dlvt/ scripts/
  ```

### Documentation

- All functions must have docstrings (NumPy style)
- Include parameter descriptions, return types, and examples
- Link to relevant paper equations/theorems

### Testing

- Write unit tests for new functions
- Run tests before submitting:
  ```bash
  pytest
  ```

## Pull Request Process

1. **Create a feature branch** with a descriptive name
2. **Make your changes** with clear, atomic commits
3. **Update documentation** if API changes
4. **Test thoroughly** — especially on edge cases
5. **Submit a pull request** with a clear description

## Reporting Bugs

Create a GitHub issue with:
- **Title**: Clear, concise summary
- **Description**: What you observed vs. expected behavior
- **Steps to reproduce**: Minimal code example
- **Environment**: Python version, OS, installed versions

## Suggesting Features

Open an issue with:
- **Title**: Feature request — [your idea]
- **Description**: What problem does it solve?
- **Example usage**: How would users interact with it?
- **Related work**: Any similar tools or papers

## Paper References

When modifying model equations, always cite the relevant theorem/proposition:

```python
def my_function():
    """
    Description.

    References
    ----------
    Theorem 1 — [brief summary]
    Proposition 2 — [brief summary]
    """
```

## File Structure Guidelines

```
dlvt/
├── __init__.py      # Public API exports
├── model.py         # Core ODE system (NEVER modify parameter defaults without discussion)
├── analysis.py      # Equilibrium, bifurcation analysis
└── figures.py       # Publication figure generation

scripts/
├── run_all_figures.py                  # Master script
├── fig8_bifurcation_hysteresis.py     # Extended analysis
├── fig9_robustness.py
└── fig10_intervention_comparison.py
```

## Default Parameters

The parameter set in `dlvt.model.DEFAULT_PARAMS` is tied to the published paper results. **Do not modify without:**
1. Updating the paper citation
2. Regenerating all figures
3. Notifying the author

## Adding New Figures

1. Create a function `figN(output_dir='figures/')` in `dlvt/figures.py` (for Figures 1–7)
2. Or create a standalone script in `scripts/figN_*.py` (for Figures 8+)
3. Add entry to `FIGURES` dict in `scripts/run_all_figures.py`
4. Ensure output saved as `{output_dir}/figN_{name}.png` and `.pdf`

## Questions?

Open a discussion on GitHub or contact the author.

## License

All contributions are licensed under the MIT License (see LICENSE file).
