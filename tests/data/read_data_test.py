import tempfile

import pandas as pd

from ml_project.data import read_data


def test_read_data():
    f, path = tempfile.mkstemp()
    pd.DataFrame([[1, 2], ["a", "b"]]).to_csv(path, index=False)
    data = read_data(path)

    assert data.shape == (2, 2)