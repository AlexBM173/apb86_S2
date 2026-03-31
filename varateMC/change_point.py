import numpy as np
import emcee
from dynesty import NestedSampler
from scipy.stats import beta, expon


def one_change_log_likelihood(params, event_times, total_period):
    """Compute log-likelihood for a single change-point Poisson process.

    Parameters
    ----------
    params : sequence[float]
        ``(h_0, h_1, s_1)`` where ``h_0`` and ``h_1`` are rates before/after
        the change point ``s_1``.
    event_times : np.ndarray
        Absolute event-time coordinates.
    total_period : float
        Upper time bound for the observation window.

    Returns
    -------
    float
        Log-likelihood value, or ``-np.inf`` for invalid parameter regions.
    """
    h_0, h_1, s_1 = params
    # Reject non-physical rates and out-of-domain change points.
    if h_0 <= 0 or h_1 <= 0 or s_1 <= 0 or s_1 >= total_period:
        return -np.inf

    # Partition event counts by segment induced by the change point.
    n_0 = np.sum(event_times < s_1)
    n_1 = np.sum(event_times > s_1)

    # Segment durations contribute exposure terms in Poisson likelihood.
    t_0 = s_1
    t_1 = total_period - s_1

    # Sum log-likelihood contributions from both piecewise-constant segments.
    return n_0 * np.log(h_0) - h_0 * t_0 + n_1 * np.log(h_1) - h_1 * t_1


def one_change_log_prior(params, total_period, alpha=1, beta_param=200):
    """Compute log-prior for one-change-point model parameters.

    Rates are assigned independent Gamma priors (simplifying to Exponential for
    ``alpha=1`` in the notebook configuration), and the change point follows the
    ``Beta(2,2)``-induced polynomial form over ``[0, total_period]``.

    Parameters
    ----------
    params : sequence[float]
        ``(h_0, h_1, s_1)`` parameters.
    total_period : float
        Upper time bound.
    alpha : float, optional
        Gamma shape parameter for both rates.
    beta_param : float, optional
        Gamma rate (inverse scale) for both rates.

    Returns
    -------
    float
        Log-prior value, or ``-np.inf`` outside support.
    """
    h_0, h_1, s_1 = params
    # Enforce support constraints before evaluating logs.
    if h_0 <= 0 or h_1 <= 0 or s_1 <= 0 or s_1 >= total_period:
        return -np.inf

    # Analytical log-prior expression matching the notebook derivation.
    return (
        alpha * np.log(beta_param)
        - beta_param * h_0
        + alpha * np.log(beta_param)
        - beta_param * h_1
        + np.log(s_1)
        + np.log(total_period - s_1)
        - 3 * np.log(total_period)
    )


def one_change_log_posterior(params, event_times, total_period, alpha=1, beta_param=200):
    """Compute log-posterior kernel for the one-change-point model.

    Parameters
    ----------
    params : sequence[float]
        ``(h_0, h_1, s_1)`` model parameters.
    event_times : np.ndarray
        Event-time observations.
    total_period : float
        Observation window end time.
    alpha : float, optional
        Prior shape for rates.
    beta_param : float, optional
        Prior rate for rates.

    Returns
    -------
    float
        Log-posterior value up to normalization.
    """
    # Posterior in log-space is additive: log prior + log likelihood.
    return one_change_log_prior(
        params, total_period, alpha=alpha, beta_param=beta_param
    ) + one_change_log_likelihood(params, event_times, total_period)


def initialize_one_change_walkers(
    nwalkers,
    total_period,
    h0_mean=0.008,
    h0_std=0.001,
    h1_mean=0.002,
    h1_std=0.001,
    seed=None,
):
    """Initialize emcee walker positions for one-change-point inference.

    Parameters
    ----------
    nwalkers : int
        Number of MCMC walkers.
    total_period : float
        Observation window upper bound.
    h0_mean : float, optional
        Mean for initial normal draw of ``h_0``.
    h0_std : float, optional
        Standard deviation for ``h_0`` initialization.
    h1_mean : float, optional
        Mean for initial normal draw of ``h_1``.
    h1_std : float, optional
        Standard deviation for ``h_1`` initialization.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(nwalkers, 3)`` containing initial walker states.
    """
    rng = np.random.default_rng(seed)
    start_pos = np.zeros((nwalkers, 3))

    # Spread change points uniformly across the full domain.
    start_pos[:, 2] = rng.uniform(0, total_period, nwalkers)

    # Seed both rates near plausible values with Gaussian jitter.
    start_pos[:, 0] = rng.normal(h0_mean, h0_std, nwalkers)
    start_pos[:, 1] = rng.normal(h1_mean, h1_std, nwalkers)

    # Ensure positive rates before sampling begins.
    start_pos[:, :2] = np.abs(start_pos[:, :2])
    return start_pos


def run_one_change_emcee(
    event_times,
    total_period,
    nwalkers=100,
    nsteps=10000,
    alpha=1,
    beta_param=200,
    burnin_fraction=0.2,
    seed=None,
):
    """Run ensemble MCMC for the one-change-point model using emcee.

    Parameters
    ----------
    event_times : np.ndarray
        Observed event-time coordinates.
    total_period : float
        Observation window upper bound.
    nwalkers : int, optional
        Number of emcee walkers.
    nsteps : int, optional
        Number of MCMC steps to run.
    alpha : float, optional
        Prior shape parameter for rate priors.
    beta_param : float, optional
        Prior rate parameter for rate priors.
    burnin_fraction : float, optional
        Fraction of initial chain discarded as burn-in.
    seed : int | None, optional
        Seed controlling walker initialization.

    Returns
    -------
    dict
        Sampling artifacts including chain sampler, flattened posterior samples,
        autocorrelation diagnostics, thinning factor, and burn-in count.
    """
    ndim = 3
    nburnin = int(nsteps * burnin_fraction)

    # Draw starting points for each walker in parameter space.
    start_pos = initialize_one_change_walkers(
        nwalkers=nwalkers,
        total_period=total_period,
        seed=seed,
    )

    def _log_posterior(params):
        # Closure binds data and hyperparameters for emcee callback signature.
        return one_change_log_posterior(
            params,
            event_times=event_times,
            total_period=total_period,
            alpha=alpha,
            beta_param=beta_param,
        )

    # Build and execute the ensemble sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_posterior)
    sampler.run_mcmc(start_pos, nsteps, progress=True)

    # Estimate autocorrelation for principled thinning of highly correlated draws.
    autocorrtime = sampler.get_autocorr_time()
    thinning_factor = 2 * int(autocorrtime.max())

    # Return flattened posterior samples after burn-in and thinning.
    samples = sampler.get_chain(discard=nburnin, thin=thinning_factor, flat=True)

    return {
        "sampler": sampler,
        "samples": samples,
        "autocorrtime": autocorrtime,
        "thinning_factor": thinning_factor,
        "acceptance_fraction": sampler.acceptance_fraction,
        "nburnin": nburnin,
        "ndim": ndim,
    }


def one_change_prior_transform(u, total_period, beta_param=200):
    """Map unit-cube samples to physical one-change-point parameters.

    This function implements the prior transform required by nested sampling.

    Parameters
    ----------
    u : np.ndarray
        Unit-cube coordinates in ``[0, 1]^3``.
    total_period : float
        Observation window upper bound.
    beta_param : float, optional
        Exponential rate parameter used for both rate transforms.

    Returns
    -------
    np.ndarray
        Transformed parameter vector ``(h_0, h_1, s_1)``.
    """
    x = np.array(u)
    # Map first dimension to change point via Beta(2,2) on [0, total_period].
    x[2] = beta.ppf(u[0], 2, 2) * total_period

    # Map remaining dimensions to positive rates via exponential quantiles.
    x[0] = expon.ppf(u[1], scale=1 / beta_param)
    x[1] = expon.ppf(u[2], scale=1 / beta_param)
    return x


def run_one_change_nested(
    event_times,
    total_period,
    ndim=3,
    nlive=1000,
    sample="unif",
    bound="multi",
    beta_param=200,
):
    """Run nested sampling for the one-change-point model.

    Parameters
    ----------
    event_times : np.ndarray
        Event-time observations.
    total_period : float
        Observation window upper bound.
    ndim : int, optional
        Number of model parameters (fixed at 3 here).
    nlive : int, optional
        Number of live points.
    sample : str, optional
        Dynesty sampling strategy.
    bound : str, optional
        Dynesty bounding strategy.
    beta_param : float, optional
        Prior rate used in the prior transform.

    Returns
    -------
    tuple[NestedSampler, dynesty.results.Results]
        The configured sampler and its results object.
    """
    def _log_likelihood(params):
        # Dynesty calls this function with transformed physical parameters.
        return one_change_log_likelihood(params, event_times, total_period)

    def _prior_transform(u):
        # Convert unit-hypercube coordinates to model parameter space.
        return one_change_prior_transform(u, total_period, beta_param=beta_param)

    # Configure and execute nested sampler.
    sampler = NestedSampler(
        _log_likelihood,
        _prior_transform,
        ndim,
        nlive=nlive,
        sample=sample,
        bound=bound,
    )
    sampler.run_nested()
    return sampler, sampler.results
