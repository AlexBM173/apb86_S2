import numpy as np
from scipy.stats import beta, expon, dirichlet
from scipy.special import gamma

def height_prior_pdf(h, alpha, beta):
    """
    Returns the PDF of the prior on the heights evaluated at h for given parameters alpha and beta. The distribution is a
    gamma distribution, but simplified to an exponential distribution when alpha = 1.

    Arguments:
        h (float) - The point at which to evaluate the PDF.
        alpha (float) - The shape parameter of the gamma distribution.
        beta (float) - The inverse-scale parameter of the gamma distribution.

    Returns:
        prior_pdf - The PDF of the prior evaluated at h.
    """
    
    if alpha == 1:
        return expon.pdf(h, scale=1/beta)
    else:
        return gamma.pdf(h, a=alpha, scale=1/beta)
    
def change_point_prior_pdf(s, L):
    """
    Returns the PDF of the prior on the change point locations evaluated on an array of given s with length k. The
    distribution is a Dirichlet distribution on the gaps between the change points.

    Arguments:
        s (np.array) - The values of the change points.
        L (float) - The upper boundary of the domain on which the change points are defined.

     Returns:
        prior_pdf - The PDF of the prior evaluated for the array s.
    """
    
    k = len(s)
    s = s/L # Entries into the scipy implementation of the Dirichlet distribution must be between zero and one.
    shapes = np.ones(k) * 2
    
    gaps = np.array([s[0]])
    for i in range(1, k):
        np.append(gaps, s[i] - s[i-1])
    np.append(gaps, 1 - s[-1])

    prior_pdf = dirichlet.pdf(gaps, shapes)

    return prior_pdf