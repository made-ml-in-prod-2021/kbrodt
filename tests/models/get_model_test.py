import pytest

from ml_project.config.train_params import TrainingParams
from ml_project.models import get_model


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("LogisticRegression"),
        pytest.param("RandomForestClassifier"),
        pytest.param("KNeighborsClassifier"),
    ],
)
def test_get_model_logreg(model_type):
    params_raw = dict(
        model_type=model_type,
    )
    params = TrainingParams(**params_raw)

    model = get_model(params)

    assert model.__class__.__name__ == model_type
