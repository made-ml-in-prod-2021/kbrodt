import logging
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

logger = logging.getLogger(__name__)


def build_pipeline(params: List[PipelineParams]) -> Pipeline:
    steps = []
    for param in params:
        logger.debug("create pipeline %s with parameters %s", param.name, param.params)

        steps.append((param.name, getattr(T, param.name)(**param.params)))

    pipeline = Pipeline(steps)

    return pipeline


def build_transformer(params: List[ColumnTransformerParams]) -> ColumnTransformer:
    logger.info("create column transformer")

    transformers = []
    for param in params:
        logger.debug("create transformer %s with columns %s", param.name, param.columns)

        transformers.append(
            (
                param.name,
                param.pipelines
                if isinstance(param.pipelines, str)
                else build_pipeline(param.pipelines),
                param.columns,
            )
        )

    transformer = ColumnTransformer(transformers)

    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("create features")

    features = transformer.transform(df)
    if hasattr(features, "toarray"):
        features = features.toarray()

    df = pd.DataFrame(features)

    return df


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    logger.info("extract target %s", params.target_col)

    target = df[params.target_col]

    return target
