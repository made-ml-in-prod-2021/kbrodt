from typing import Union

from pydantic import BaseModel, conlist, validator

# This is done based on EDA during ml_project step
ID_COL = "Id"
FEATURES_MINMAX_VALS = [
    (ID_COL, None),
    ("age", None),
    ("sex", [0, 1]),
    ("cp", [0, 3]),
    ("trestbps", None),
    ("chol", None),
    ("fbs", [0, 1]),
    ("restecg", [0, 2]),
    ("thalach", None),
    ("exang", [0, 1]),
    ("oldpeak", None),
    ("slope", [0, 2]),
    ("ca", [0, 4]),
    ("thal", [0, 3]),
]
FEATURES = [f for f, _ in FEATURES_MINMAX_VALS]
N_FEATURES = len(FEATURES)

ENTRY_POINT_MSG = "It is heart desease predictor"
CAT_OUT_OF_RANGE_ERR = "Categorical value is out of range"
INCORRECT_FEATURES_ERR = "Incorrect features order/columns"


class Patient(BaseModel):
    data: conlist(
        conlist(Union[float, int], min_items=N_FEATURES, max_items=N_FEATURES),
        min_items=1,
    )
    features: conlist(str, min_items=N_FEATURES, max_items=N_FEATURES)

    @validator("data")
    def data_consistency(cls, data):
        for sample in data:
            for value, (_, min_max) in zip(sample, FEATURES_MINMAX_VALS):
                if min_max is None:
                    continue

                min_value, max_value = min_max
                if not (min_value <= value <= max_value):
                    raise ValueError(CAT_OUT_OF_RANGE_ERR)

        return data

    @validator("features")
    def feature_consistency(cls, features):
        if features != FEATURES:
            raise ValueError(INCORRECT_FEATURES_ERR)

        return features


class HeartDisease(BaseModel):
    id: int
    probability: float
