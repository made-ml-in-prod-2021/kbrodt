import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ml_project.apis import train_model


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
def model():
    model = LogisticRegression()

    return model


def test_train_model(model, features, target):
    model = train_model(model, features, target)

    check_is_fitted(model)
