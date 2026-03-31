# Architecture

## Purpose

The `varateMC` package isolates model and inference logic from notebook plotting code.
This separation keeps computational routines reusable and unit-testable.

## Package Structure

- `varateMC/data.py`: data loading and derived summary quantities.
- `varateMC/priors.py`: analytic prior density curves for order statistics.
- `varateMC/constant_rate.py`: homogeneous Poisson process posterior utilities.
- `varateMC/change_point.py`: one-change-point model priors, likelihoods, and samplers.
- `varateMC/rjmcmc.py`: RJMCMC state decoding and model-averaged rate summaries.
- `varateMC/stats.py`: low-level prior density helper functions.

## Design Principles

- Prefer pure functions that accept arrays/scalars and return computed values.
- Keep plotting concerns in notebooks and figures scripts.
- Expose deterministic interfaces where possible.
- Place sampling wrappers in dedicated modules and keep pre/post-processing separate.

## Data Flow

1. Load event increment data and convert to cumulative event times.
2. Infer model parameters using constant-rate, one-change-point, or RJMCMC workflows.
3. Evaluate posterior summaries on common grids.
4. Plot posterior summaries in notebook cells.

## Error Handling Conventions

- Invalid support conditions in log-density functions return `-np.inf`.
- Invalid geometric/state configurations in helper PDFs return zero density.
- Invalid structural input (for example non-positive domain length) raises `ValueError`.
