import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    logger.info(
        "evaluating model on dataset %s with threshold %s", predicts.shape, threshold
    )
    return {
        "acc": accuracy_score(target, predicts > threshold),
        "auc": roc_auc_score(target, predicts),
        "f1": f1_score(target, predicts > threshold),
    }
