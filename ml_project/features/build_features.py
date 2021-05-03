from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import ml_project.features.transformers as T
from ml_project.config.feature_params import (
    ColumnTransformerParams,
    FeatureParams,
    PipelineParams,
)


def build_pipeline(params: List[PipelineParams]) -> Pipeline:
    pipeline = Pipeline(
        [(param.name, getattr(T, param.name)(**param.params)) for param in params]
    )

    return pipeline


def build_transformer(params: List[ColumnTransformerParams]) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                param.name,
                param.pipelines
                if isinstance(param.pipelines, str)
                else build_pipeline(param.pipelines),
                param.columns,
            )
            for param in params
        ]
    )

    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    features = transformer.transform(df)
    if hasattr(features, "toarray"):
        features = features.toarray()

    df = pd.DataFrame(features)

    return df


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]

    return target
