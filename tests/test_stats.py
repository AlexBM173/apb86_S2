import numpy as np

from varateMC.stats import change_point_prior_pdf, height_prior_pdf


def test_height_prior_pdf_positive_in_supported_region():
    val_exp = height_prior_pdf(0.5, alpha=1, beta=200)
    val_gamma = height_prior_pdf(0.5, alpha=2, beta=200)

    assert val_exp > 0
    assert val_gamma > 0


def test_change_point_prior_pdf_positive_for_valid_ordered_points():
    s = np.array([2.0, 5.0, 8.0])
    val = change_point_prior_pdf(s, L=10.0)

    assert val > 0


def test_change_point_prior_pdf_rejects_out_of_bounds_points():
    s = np.array([2.0, 11.0])
    val = change_point_prior_pdf(s, L=10.0)

    assert val == 0.0
