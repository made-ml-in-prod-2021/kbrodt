import numpy as np
import pandas as pd

from ml_project.features.transformers import Log1p


def test_log1p_transformer_np():
    X = np.array(
        [
            [1.0, 10, 20],
            [-10.0, -7, 4],
        ]
    )
    X_expected = X.copy()
    X_expected -= np.min(X_expected, axis=0, keepdims=True)
    X_expected = np.log1p(X_expected)

    log1p_transformer = Log1p()
    X_feat = log1p_transformer.fit_transform(X)

    assert X_feat.tolist() == X_expected.tolist()


def test_log1p_transformer_df():
    X = pd.DataFrame(
        [
            [1.0, 10, 20],
            [-10.0, -7, 4],
        ]
    )
    X_expected = X.copy()
    X_expected -= np.min(X_expected, axis=0)
    X_expected = np.log1p(X_expected)

    log1p_transformer = Log1p()
    X_feat = log1p_transformer.fit_transform(X)

    assert X_feat.tolist() == X_expected.values.tolist()
