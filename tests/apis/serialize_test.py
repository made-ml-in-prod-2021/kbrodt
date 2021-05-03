import pickle
import tempfile

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_project.apis import serialize_model


@pytest.fixture
def features():
    f = pd.DataFrame(
        [
            [0, 1, 0],
            [1, 2, 1],
        ]
    )

    return f


@pytest.fixture
def target():
    return pd.Series([0, 1])


@pytest.fixture
def model(features, target):
    model = LogisticRegression()
    model.fit(features, target)

    return model


def test_serialize_model(model):
    fd, path = tempfile.mkstemp()

    serialize_model(model, fd)
    with open(path, "rb") as fin:
        model_loaded = pickle.load(fin)

    assert model.coef_.tolist() == model_loaded.coef_.tolist()
    assert model.intercept_.tolist() == model_loaded.intercept_.tolist()
