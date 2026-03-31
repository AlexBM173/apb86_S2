import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist


def log_constant_rate_prior(h, alpha, beta):
    """Compute log-prior for a constant Poisson rate under Gamma prior.

    Parameters
    ----------
    h : float | np.ndarray
        Rate parameter(s) to evaluate.
    alpha : float
        Gamma prior shape.
    beta : float
        Gamma prior rate (inverse scale).

    Returns
    -------
    float | np.ndarray
        Log-prior values up to additive constants retained in this expression.
    """
    # Keep the algebraic form used in the original notebook derivation.
    return alpha * np.log(beta) - beta * h - gammaln(alpha)


def constant_rate_likelihood(h, total_events, total_period):
    """Evaluate the unnormalized Poisson-process likelihood.

    Parameters
    ----------
    h : float | np.ndarray
        Constant event rate.
    total_events : int
        Number of observed events in the interval.
    total_period : float
        Observation length.

    Returns
    -------
    float | np.ndarray
        Likelihood proportional to ``h**N * exp(-hT)``.
    """
    # Likelihood kernel for a homogeneous Poisson process over fixed window.
    return h**total_events * np.exp(-h * total_period)


def log_constant_rate_likelihood(h, total_events, total_period):
    """Evaluate log-likelihood for the constant-rate Poisson model.

    Parameters
    ----------
    h : float | np.ndarray
        Constant event rate.
    total_events : int
        Number of observed events.
    total_period : float
        Observation window length.

    Returns
    -------
    float | np.ndarray
        Log-likelihood values.
    """
    # Log form is numerically stable and easier to combine with priors.
    return total_events * np.log(h) - h * total_period


def log_constant_rate_posterior(h, alpha, beta, total_events, total_period):
    """Evaluate log-posterior for the constant-rate model.

    Parameters
    ----------
    h : float | np.ndarray
        Candidate rate value(s).
    alpha : float
        Gamma prior shape.
    beta : float
        Gamma prior rate.
    total_events : int
        Number of observed events.
    total_period : float
        Observation length.

    Returns
    -------
    float | np.ndarray
        Log-posterior values up to normalization.
    """
    # Posterior kernel equals prior kernel plus likelihood kernel in log-space.
    return log_constant_rate_prior(h, alpha, beta) + log_constant_rate_likelihood(
        h, total_events, total_period
    )


def evaluate_constant_rate_grid(total_events, total_period, alpha=1, beta=200, n=1000):
    """Create grid evaluations for prior and posterior diagnostics.

    Parameters
    ----------
    total_events : int
        Number of observed events.
    total_period : float
        Observation length.
    alpha : float, optional
        Gamma prior shape.
    beta : float, optional
        Gamma prior rate.
    n : int, optional
        Number of grid points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(h_values, log_prior_values, log_posterior_values)``.
    """
    # Start slightly above zero to avoid log(0) warnings in likelihood terms.
    h_values = np.linspace(np.finfo(float).eps, total_events * 2 / total_period, n)

    # Evaluate posterior kernel over the full grid in vectorized form.
    log_posterior_values = log_constant_rate_posterior(
        h_values, alpha, beta, total_events, total_period
    )

    # Reuse SciPy gamma implementation to compare with analytical prior form.
    log_prior_values = np.log(gamma_dist.pdf(h_values, a=alpha, scale=1 / beta))
    return h_values, log_prior_values, log_posterior_values


def exp_scaled_log_integrand(h, total_events, total_period):
    """Exponentiate a scaled log-integrand for evidence integration.

    The scaling follows the notebook derivation and is intended to keep the
    integrand in a numerically manageable range for quadrature.

    Parameters
    ----------
    h : float
        Rate value.
    total_events : int
        Number of observed events.
    total_period : float
        Observation duration.

    Returns
    -------
    float
        Scaled integrand value in linear space.
    """
    # Convert the stabilized log-expression back to linear space for quad().
    return np.exp(
        total_events * (np.log(h * (200 + total_period) / total_events) + 1)
        - h * (200 + total_period)
    )


def compute_scaled_evidence(total_events, total_period, lower=0.0, upper=1.0):
    """Numerically integrate the scaled evidence integrand.

    Parameters
    ----------
    total_events : int
        Number of observed events.
    total_period : float
        Observation duration.
    lower : float, optional
        Lower integration bound.
    upper : float, optional
        Upper integration bound.

    Returns
    -------
    tuple[float, float]
        ``(integral_estimate, absolute_error)`` from ``scipy.integrate.quad``.
    """
    # Bind model constants and integrate only over h.
    integrand = lambda h: exp_scaled_log_integrand(h, total_events, total_period)
    return quad(integrand, lower, upper)
