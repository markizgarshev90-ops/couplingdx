# Contributing

Bug reports and questions are welcome through GitHub issues.

- Run the tests before opening a pull request: `pip install pytest && python -m pytest couplingdx/tests`.
- The package has no runtime dependencies. Please keep it that way.
- Changes to the estimators or sizing formulas should come with a simulation check,
  as in `analysis/design_tables.py`.
