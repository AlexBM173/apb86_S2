import numpy as np

from varateMC.rjmcmc import (
    evaluate_rates_across_chain,
    extract_model_sizes,
    map_model_size,
    post_burn_in_chain,
    summarize_rate_bands,
)


def test_extract_model_sizes_from_compact_state_encoding():
    # state format: [s1,...,sk, h0,...,hk, k]
    chains = [
        np.array([0.2, 0.1, 0.3, 1]),          # k=1
        np.array([0.2, 0.6, 0.1, 0.2, 0.3, 2]) # k=2
    ]

    assert extract_model_sizes(chains) == [1, 2]


def test_post_burn_in_chain_slices_correctly():
    chains = np.arange(20)
    out = post_burn_in_chain(chains, burn_in=5)

    np.testing.assert_array_equal(out, np.arange(5, 20))


def test_map_model_size_returns_mode():
    k_samples = [1, 2, 2, 3, 2, 1]
    assert map_model_size(k_samples) == 2


def test_evaluate_rates_across_chain_and_band_summary_shapes():
    chain = [
        np.array([3.0, 0.5, 0.2, 1]),
        np.array([5.0, 0.3, 0.1, 1]),
    ]

    t_eval, rate_eval = evaluate_rates_across_chain(chain, total_period=10.0, n_eval=50)

    assert t_eval.shape == (50,)
    assert rate_eval.shape == (2, 50)

    bands = summarize_rate_bands(rate_eval)
    expected = {"median_rate", "lower_90", "upper_90", "lower_50", "upper_50"}
    assert set(bands.keys()) == expected
    for key in expected:
        assert bands[key].shape == (50,)
