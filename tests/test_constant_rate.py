import numpy as np

from varateMC.constant_rate import (
    compute_scaled_evidence,
    constant_rate_likelihood,
    evaluate_constant_rate_grid,
    log_constant_rate_likelihood,
    log_constant_rate_posterior,
)


def test_log_likelihood_matches_log_of_likelihood_kernel():
    h = 0.25
    total_events = 8
    total_period = 15.0

    ll = constant_rate_likelihood(h, total_events, total_period)
    log_ll = log_constant_rate_likelihood(h, total_events, total_period)
    assert np.isclose(log_ll, np.log(ll))


def test_log_posterior_vectorizes_over_h_values():
    h = np.array([0.1, 0.2, 0.3])
    out = log_constant_rate_posterior(h, alpha=1, beta=200, total_events=10, total_period=100.0)

    assert out.shape == h.shape


def test_evaluate_constant_rate_grid_returns_consistent_shapes():
    h_values, log_prior, log_post = evaluate_constant_rate_grid(
        total_events=10,
        total_period=100.0,
        n=120,
    )

    assert h_values.shape == (120,)
    assert log_prior.shape == (120,)
    assert log_post.shape == (120,)


def test_compute_scaled_evidence_returns_finite_result():
    integral, error = compute_scaled_evidence(total_events=5, total_period=20.0, lower=1e-4, upper=1.0)

    assert np.isfinite(integral)
    assert np.isfinite(error)
    assert integral >= 0
