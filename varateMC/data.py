import os

import numpy as np


def load_mining_data(data_dir="data", filename="coal_mining_accident_data.dat"):
    """Load and preprocess coal mining accident inter-arrival data.

    The source data file stores increments (typically days between accidents).
    This helper returns both the raw matrix form and several derived vectors used
    throughout the notebook analysis.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing the data file.
    filename : str, optional
        Name of the text file to load.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``mining_data``: raw 2D array from disk.
        - ``flattened_data``: 1D vector in Fortran order.
        - ``flattened_data_cumulative``: cumulative event-time locations.
        - ``number_of_accidents``: integer index per event.
    """
    # Construct a portable path so callers can override data location.
    file_path = os.path.join(data_dir, filename)
    mining_data = np.loadtxt(file_path)

    # Preserve historical notebook behavior: flatten by column-major ordering.
    flattened_data = np.ndarray.flatten(mining_data, order="F")

    # Convert increments to absolute event-time coordinates.
    flattened_data_cumulative = np.cumsum(flattened_data)

    # Build a simple event counter aligned to cumulative event times.
    number_of_accidents = np.arange(0, len(flattened_data_cumulative), 1)

    return {
        "mining_data": mining_data,
        "flattened_data": flattened_data,
        "flattened_data_cumulative": flattened_data_cumulative,
        "number_of_accidents": number_of_accidents,
    }


def mean_rates(total_events, total_period, number_of_accidents):
    """Compute average event rates in day and year units.

    Parameters
    ----------
    total_events : int
        Total number of accidents/events observed.
    total_period : float
        Observation window length in days.
    number_of_accidents : np.ndarray
        Event index array; the final entry corresponds to the last event count.

    Returns
    -------
    tuple[float, float]
        ``(mean_rate_per_day, mean_rate_per_year)``.
    """
    # Global rate estimate in native (day) units.
    mean_rate_per_day = total_events / total_period

    # Convert day-based count to annualized rate for interpretability.
    mean_rate_per_year = number_of_accidents[-1] / (total_period / 365)
    return mean_rate_per_day, mean_rate_per_year
