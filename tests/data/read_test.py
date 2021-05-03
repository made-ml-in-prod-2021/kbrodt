from ml_project.data import read_data

DATA_PATH = "data/raw/heart.csv"


def test_read_data():
    data = read_data(DATA_PATH)

    assert data.shape == (303, 14)
