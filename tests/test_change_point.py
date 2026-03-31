import numpy as np

from varateMC.change_point import (
    initialize_one_change_walkers,
    one_change_log_likelihood,
    one_change_log_posterior,
    one_change_log_prior,
    one_change_prior_transform,
)


def test_one_change_log_likelihood_rejects_invalid_support():
    event_times = np.array([1.0, 2.0, 3.0])
    val = one_change_log_likelihood((0.0, 1.0, 2.0), event_times, total_period=10.0)
    assert val == -np.inf


def test_one_change_log_prior_rejects_invalid_change_point():
    val = one_change_log_prior((0.2, 0.3, 0.0), total_period=10.0)
    assert val == -np.inf


def test_one_change_log_posterior_is_finite_for_valid_input():
    event_times = np.array([1.0, 2.0, 8.0])
    val = one_change_log_posterior((0.4, 0.2, 4.0), event_times, total_period=10.0)
    assert np.isfinite(val)


def test_initialize_walkers_shape_and_rate_positivity():
    walkers = initialize_one_change_walkers(nwalkers=24, total_period=10.0, seed=42)

    assert walkers.shape == (24, 3)
    assert np.all(walkers[:, 0] > 0)
    assert np.all(walkers[:, 1] > 0)
    assert np.all(walkers[:, 2] >= 0)
    assert np.all(walkers[:, 2] <= 10.0)


def test_prior_transform_maps_to_valid_parameter_domain():
    u = np.array([0.3, 0.4, 0.5])
    transformed = one_change_prior_transform(u, total_period=10.0, beta_param=200)

    assert transformed.shape == (3,)
    assert transformed[0] > 0
    assert transformed[1] > 0
    assert 0 < transformed[2] < 10.0
