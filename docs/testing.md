# Testing Guide

## Scope

The test suite validates the numerical and structural behavior of all public module
functions in `varateMC`.

Coverage includes:

- data loading and summary helpers,
- prior curve generation,
- constant-rate posterior kernels and evidence integration,
- one-change-point log-density utilities and transforms,
- RJMCMC state decoding and posterior band summaries,
- prior density helpers in `stats`.

## Running Tests

Install dependencies:

```bash
pip install -e .[test]
```

Execute all tests:

```bash
pytest
```

Execute one test module:

```bash
pytest tests/test_rjmcmc.py
```

## Test Design

- Tests avoid expensive long-chain sampling runs.
- Core numerical identities are validated with small deterministic inputs.
- Boundary behavior is explicitly checked (`-np.inf` support checks, shape checks,
  percentile outputs, and transform support).

## Extending Tests

When adding package functions:

1. Add a focused unit test in the corresponding `tests/test_<module>.py` file.
2. Prefer deterministic arrays over stochastic simulations.
3. Include one valid-path and one invalid/boundary-path assertion.
