import logging

import ml_project.models.zoo as MZ
from ml_project.config.train_params import TrainingParams

logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> MZ.BaseEstimator:
    logger.info("get model %s with params %s", params.model_type, params.params)
    model = getattr(MZ, params.model_type)(params.params)

    return model


__all__ = [
    "get_model",
]
