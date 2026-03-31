import numpy as np
from scipy.stats import expon, dirichlet, gamma as gamma_dist


def height_prior_pdf(h, alpha, beta):
    """Evaluate the prior density for a segment height (rate) parameter.

    Parameters
    ----------
    h : float
        Height/rate value at which to evaluate the prior.
    alpha : float
        Shape parameter of the Gamma prior.
    beta : float
        Rate parameter (inverse scale) of the Gamma prior.

    Returns
    -------
    float
        Prior density value at ``h``.

    Notes
    -----
    For ``alpha == 1``, the Gamma distribution reduces to Exponential.
    """
    # Special-case alpha==1 for exact equivalence with exponential prior form.
    if alpha == 1:
        return expon.pdf(h, scale=1 / beta)
    else:
        # Use SciPy's Gamma distribution implementation for general alpha.
        return gamma_dist.pdf(h, a=alpha, scale=1 / beta)


def change_point_prior_pdf(s, L):
    """Evaluate Dirichlet prior density over ordered change-point locations.

    Parameters
    ----------
    s : np.ndarray
        Sorted change-point locations on ``[0, L]``.
    L : float
        Upper boundary of the domain.

    Returns
    -------
    float
        Prior density evaluated using a Dirichlet model on normalized gaps.
    """
    if L <= 0:
        raise ValueError("L must be positive")

    s = np.asarray(s, dtype=float)
    if s.ndim != 1:
        raise ValueError("s must be a 1D array of ordered change points")
    if np.any(s <= 0) or np.any(s >= L):
        return 0.0
    if np.any(np.diff(s) <= 0):
        return 0.0

    # Normalize locations because scipy Dirichlet expects simplex components.
    s_normalized = s / L

    # Gap vector has k+1 components on the simplex for k change points.
    gaps = np.concatenate(([s_normalized[0]], np.diff(s_normalized), [1 - s_normalized[-1]]))
    shapes = np.ones(len(gaps)) * 2

    # Evaluate Dirichlet density on normalized segment lengths.
    return float(dirichlet.pdf(gaps, shapes))