import logging
import pickle
from typing import Union

from ml_project.models.zoo import BaseEstimator

logger = logging.getLogger(__name__)


def serialize_model(model: BaseEstimator, output: Union[str, int]) -> None:
    logger.info("save model %s to %s", model.__class__.__name__, output)

    with open(output, "wb") as f:
        pickle.dump(model, f)


def load_model(output: Union[str, int]) -> BaseEstimator:
    logger.info("load model from %s", output)

    with open(output, "rb") as f:
        model = pickle.load(f)

    return model
