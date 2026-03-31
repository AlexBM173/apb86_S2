import numpy as np


def load_chains(path="mcmc_chains.npy"):
    """Load serialized RJMCMC chain states from disk.

    Parameters
    ----------
    path : str, optional
        Path to ``.npy`` file containing an object array of states.

    Returns
    -------
    np.ndarray
        Chain states as loaded by NumPy with pickle support enabled.
    """
    # Stored chain entries are variable-length objects, so allow pickle objects.
    return np.load(path, allow_pickle=True)


def extract_model_sizes(chains):
    """Extract number of change points ``k`` for each RJMCMC state.

    Each state is encoded as:
    ``[s_1, ..., s_k, h_0, ..., h_k, k]``
    so the number of change points can be inferred from vector length.

    Parameters
    ----------
    chains : sequence
        Iterable of RJMCMC state vectors.

    Returns
    -------
    list[int]
        Inferred model size per chain state.
    """
    # Reverse the compact state encoding to recover model dimensionality.
    return [(len(state) - 1) // 2 for state in chains]


def post_burn_in_chain(chains, burn_in=10000):
    """Discard initial RJMCMC samples as burn-in.

    Parameters
    ----------
    chains : np.ndarray
        Full Markov chain of states.
    burn_in : int, optional
        Number of initial states to drop.

    Returns
    -------
    np.ndarray
        Chain after burn-in removal.
    """
    # Keep only stationary-region samples for posterior summaries.
    return chains[burn_in:]


def map_model_size(k_samples):
    """Compute the modal (MAP-by-frequency) model size from samples.

    Parameters
    ----------
    k_samples : sequence[int]
        Sampled model sizes.

    Returns
    -------
    int
        Most frequently observed model size.
    """
    # Frequency mode is a simple discrete approximation to MAP over k.
    return max(set(k_samples), key=k_samples.count)


def evaluate_rates_across_chain(chain, total_period, n_eval=1000):
    """Evaluate piecewise-constant rate for each chain state on a fixed grid.

    Parameters
    ----------
    chain : sequence
        RJMCMC state vectors after burn-in.
    total_period : float
        Observation window upper bound.
    n_eval : int, optional
        Number of time-grid points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(t_eval, rate_evaluations)`` where ``rate_evaluations`` has shape
        ``(n_states, n_eval)``.
    """
    # Shared grid enables pointwise posterior summaries across variable-k states.
    t_eval = np.linspace(0, total_period, n_eval)
    rate_evaluations = np.zeros((len(chain), len(t_eval)))

    for i, state in enumerate(chain):
        # Decode each state into ordered change points and segment rates.
        k = (len(state) - 1) // 2
        change_points = state[0:k]
        rates = state[k : 2 * k + 1]

        # Find active segment for each time by counting passed change points.
        idx = np.searchsorted(change_points, t_eval, side="right")
        rate_evaluations[i, :] = rates[idx]

    return t_eval, rate_evaluations


def summarize_rate_bands(rate_evaluations):
    """Compute pointwise posterior credible bands for inferred rates.

    Parameters
    ----------
    rate_evaluations : np.ndarray
        Matrix of shape ``(n_samples, n_time_points)`` containing evaluated rates.

    Returns
    -------
    dict[str, np.ndarray]
        Median and central 50%/90% credible bands at each time point.
    """
    # Percentiles are taken across posterior samples for each fixed time index.
    return {
        "median_rate": np.percentile(rate_evaluations, 50, axis=0),
        "lower_90": np.percentile(rate_evaluations, 5, axis=0),
        "upper_90": np.percentile(rate_evaluations, 95, axis=0),
        "lower_50": np.percentile(rate_evaluations, 25, axis=0),
        "upper_50": np.percentile(rate_evaluations, 75, axis=0),
    }
