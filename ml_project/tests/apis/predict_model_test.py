import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_project.apis import predict_model


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


def test_train_model(model, features, target):
    predicts = predict_model(model, features, return_proba=False)

    assert predicts.shape == target.shape

    predicts = predict_model(model, features, return_proba=True)
    assert predicts.shape == (len(target), 2)
