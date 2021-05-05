import logging
import pickle
from typing import Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from clfit.models.zoo import BaseEstimator

logger = logging.getLogger(__name__)


def serialize_model(
    model: BaseEstimator, transformer: ColumnTransformer, output: Union[str, int]
) -> None:
    logger.info("save model %s and transformer to %s", model.__class__.__name__, output)

    pipeline = Pipeline([("transformer", transformer), ("model", model)])

    with open(output, "wb") as f:
        pickle.dump(pipeline, f)


def load_model(output: Union[str, int]) -> Pipeline:
    logger.info("load model from %s", output)

    with open(output, "rb") as f:
        pipeline: Pipeline = pickle.load(f)

    return pipeline
