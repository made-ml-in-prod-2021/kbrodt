from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import FLOAT_DTYPES

Arrayable = Union[np.ndarray, pd.DataFrame]

SimpleImputer = SimpleImputer
OneHotEncoder = OneHotEncoder


class Log1p(BaseEstimator, TransformerMixin):
    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    def fit(self, X: Arrayable, y: Optional[np.ndarray] = None) -> "Log1p":
        return self

    def transform(
        self, X: Arrayable, y: Optional[np.ndarray] = None, copy: Optional[bool] = None
    ) -> Arrayable:
        copy = copy or self.copy
        Xo = self._validate_data(
            X,
            copy=copy,
            reset=False,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        # Xo = X.copy()
        # if hasattr(Xo, "values"):
        # Xo = Xo.values

        Xo -= np.min(Xo, axis=0, keepdims=True)
        Xo = np.log1p(Xo)

        return Xo

    def fit_transform(self, X: Arrayable, y: Optional[np.ndarray] = None) -> Arrayable:
        return self.fit(X, y).transform(X, y)
