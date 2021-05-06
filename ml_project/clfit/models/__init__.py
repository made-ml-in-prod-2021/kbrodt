import logging

import clfit.models.zoo as MZ
from clfit.config.train_params import TrainingParams

logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> MZ.BaseEstimator:
    logger.debug("get model %s with params %s", params.model_type, params.params)

    model = getattr(MZ, params.model_type)(**params.params)

    return model


__all__ = [
    "get_model",
]
