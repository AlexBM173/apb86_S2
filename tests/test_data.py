import numpy as np

from varateMC.data import load_mining_data, mean_rates


def test_load_mining_data_outputs_expected_keys_and_shapes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample = np.array([[1.0, 2.0], [3.0, 4.0]])
    file_path = data_dir / "coal_mining_accident_data.dat"
    np.savetxt(file_path, sample)

    loaded = load_mining_data(data_dir=str(data_dir), filename=file_path.name)

    assert set(loaded.keys()) == {
        "mining_data",
        "flattened_data",
        "flattened_data_cumulative",
        "number_of_accidents",
    }
    assert loaded["mining_data"].shape == (2, 2)
    np.testing.assert_allclose(loaded["flattened_data"], np.array([1.0, 3.0, 2.0, 4.0]))
    np.testing.assert_allclose(loaded["flattened_data_cumulative"], np.array([1.0, 4.0, 6.0, 10.0]))


def test_mean_rates_computes_day_and_year_units():
    number_of_accidents = np.arange(0, 6)
    mean_day, mean_year = mean_rates(total_events=6, total_period=365.0, number_of_accidents=number_of_accidents)

    assert np.isclose(mean_day, 6 / 365.0)
    assert np.isclose(mean_year, 5.0)
