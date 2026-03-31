import numpy as np


def order_statistics_pdfs(x, total_period):
    """Evaluate order-statistics prior PDFs used for change-point locations.

    The notebook compares two constructions for four ordered points over the
    interval ``[0, total_period]``:
    - ``plain``: standard order-statistics marginals.
    - ``even``: a polynomial form favoring more even spacing.

    Parameters
    ----------
    x : np.ndarray
        Evaluation grid over time.
    total_period : float
        Upper boundary of the time domain.

    Returns
    -------
    tuple[dict, dict]
        ``(plain, even)`` dictionaries mapping component names to PDF values.
    """
    # Closed-form PDFs for the plain order-statistics construction.
    plain = {
        "u1": 4 * (total_period - x) ** 3 / (total_period ** 4),
        "u2": 12 * x * (total_period - x) ** 2 / (total_period ** 4),
        "u3": 12 * x ** 2 * (total_period - x) / (total_period ** 4),
        "u4": 4 * x ** 3 / (total_period ** 4),
    }

    # Precomputed normalization constant used by the symbolic polynomial form.
    norm = 8.168732867e35

    # Polynomial expansions for the even-spacing prior components.
    even = {
        "v1": -x
        * (
            x**7
            - 7 * x**6 * total_period
            + 21 * x**5 * total_period**2
            - 35 * x**4 * total_period**3
            + 35 * x**3 * total_period**4
            - 21 * x**2 * total_period**5
            + 7 * x * total_period**6
            - total_period**7
        )
        / (5040 * norm),
        "v2": -x**3
        * (
            x**5
            - 5 * x**4 * total_period
            + 10 * x**3 * total_period**2
            - 10 * x**2 * total_period**3
            + 5 * x * total_period**4
            - total_period**5
        )
        / (720 * norm),
        "v3": -x**5
        * (x**3 - 3 * x**2 * total_period + 3 * x * total_period**2 - total_period**3)
        / (720 * norm),
        "v4": -x**7 * (x - total_period) / (5040 * norm),
    }

    return plain, even
