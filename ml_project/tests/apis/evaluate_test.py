import numpy as np
import pandas as pd

from ml_project.apis import evaluate_model


def test_evaluate_model():
    predicts = np.array([0.1, 0.4])
    target = pd.Series([0, 1])

    etalon_res = {
        "acc": 0.5,
        "auc": 1,
        "f1": 0,
    }
    res = evaluate_model(predicts, target)

    assert etalon_res == res
