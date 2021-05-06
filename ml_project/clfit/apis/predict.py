import logging

import numpy as np
import pandas as pd

from clfit.models.zoo import BaseEstimator

logger = logging.getLogger(__name__)


def predict_model(
    model: BaseEstimator, features: pd.DataFrame, return_proba: bool = True
) -> np.ndarray:
    logger.info(
        "model %s predict on dataset %s", model.__class__.__name__, features.shape
    )

    predicts: np.ndarray = (
        model.predict_proba(features)[:, 1] if return_proba else model.predict(features)
    )

    return predicts
