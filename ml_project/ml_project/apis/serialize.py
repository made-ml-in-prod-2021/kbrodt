import logging
import pickle
from typing import Tuple, Union

from sklearn.compose import ColumnTransformer

from ml_project.models.zoo import BaseEstimator

Model = Tuple[BaseEstimator, ColumnTransformer]

logger = logging.getLogger(__name__)


def serialize_model(model: Model, output: Union[str, int]) -> None:
    logger.info("save model %s to %s", model.__class__.__name__, output)

    with open(output, "wb") as f:
        pickle.dump(model, f)


def load_model(output: Union[str, int]) -> Model:
    logger.info("load model from %s", output)

    with open(output, "rb") as f:
        model: Model = pickle.load(f)

    return model
