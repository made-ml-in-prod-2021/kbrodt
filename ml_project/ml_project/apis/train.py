import logging

import pandas as pd

from ml_project.models.zoo import BaseEstimator

logger = logging.getLogger(__name__)


def train_model(
    model: BaseEstimator, features: pd.DataFrame, target: pd.Series
) -> None:
    logger.info(
        "fit the model %s on dataset %s", model.__class__.__name__, features.shape
    )

    model.fit(features, target)
