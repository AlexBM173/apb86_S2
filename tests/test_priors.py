import numpy as np

from varateMC.priors import order_statistics_pdfs


def test_order_statistics_pdfs_return_expected_components():
    x = np.linspace(0.0, 10.0, 50)
    plain, even = order_statistics_pdfs(x, total_period=10.0)

    assert set(plain.keys()) == {"u1", "u2", "u3", "u4"}
    assert set(even.keys()) == {"v1", "v2", "v3", "v4"}
    assert plain["u1"].shape == x.shape
    assert even["v4"].shape == x.shape


def test_plain_order_statistics_are_nonnegative_on_domain():
    x = np.linspace(0.0, 5.0, 20)
    plain, _ = order_statistics_pdfs(x, total_period=5.0)

    for arr in plain.values():
        assert np.all(arr >= 0)
