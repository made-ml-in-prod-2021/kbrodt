import pandas as pd

from clfit.data import read_data


def test_read_data(tmp_path):
    path = tmp_path / "predicts.csv"
    pd.DataFrame([[1, 2], ["a", "b"]]).to_csv(path, index=False)
    data = read_data(path)

    assert data.shape == (2, 2)
