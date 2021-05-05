import tempfile

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from clfit.apis import load_model, serialize_model


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


@pytest.fixture
def transformer():
    tr = ColumnTransformer(["all", "passthrough", [0, 1, 2]])

    return tr


def test_serialize_model(model, transformer):
    fd, path = tempfile.mkstemp()

    serialize_model(model, transformer, fd)
    pipeline = load_model(path)

    assert isinstance(pipeline, Pipeline)

    model_loaded = pipeline.steps[-1][-1]
    assert model.coef_.tolist() == model_loaded.coef_.tolist()
    assert model.intercept_.tolist() == model_loaded.intercept_.tolist()
