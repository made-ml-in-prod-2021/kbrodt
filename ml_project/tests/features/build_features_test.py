import pandas as pd
import pytest

from ml_project.config.feature_params import (
    ColumnTransformerParams,
    FeatureParams,
    PipelineParams,
)
from ml_project.features import build_transformer, extract_target, make_features
from ml_project.features.build_features import build_pipeline


@pytest.fixture
def params_raw():
    params = [
        dict(
            name="SimpleImputer",
            params={"missing_values": "nan", "strategy": "most_frequent"},
        ),
    ]

    return params


def test_build_pipeline(params_raw):
    pipelines = [PipelineParams(**param) for param in params_raw]

    pipeline = build_pipeline(pipelines)
    for pr, pl in zip(params_raw, pipeline.steps):
        assert pr["name"] == pl[0]


@pytest.fixture
def pipeline(params_raw):
    pipelines = [PipelineParams(**param) for param in params_raw]

    pl = build_pipeline(pipelines)

    return pl


def test_pipeline(pipeline):
    X = [["nan", 1], [1, 0]]
    X_expected = [[1, 1], [1, 0]]
    X_feat = pipeline.fit_transform(X)
    assert X_feat.tolist() == X_expected


@pytest.fixture
def transformers_raw(params_raw):
    pipelines = [PipelineParams(**param) for param in params_raw]
    transformers = [
        dict(name="num", pipelines=pipelines, columns=[0, 1]),
        dict(name="bin", pipelines="passthrough", columns=[2]),
        dict(name="cat", pipelines="drop", columns=[3]),
    ]

    return transformers


def test_build_transform(transformers_raw):
    transformers = [ColumnTransformerParams(**tr) for tr in transformers_raw]

    transformer = build_transformer(transformers)
    for tr, t in zip(transformers_raw, transformer.transformers):
        assert tr["name"] == t[0]
        assert tr["columns"] == t[2]
        if isinstance(tr["pipelines"], str):
            assert tr["pipelines"] == t[1]
        else:
            for pr, pl in zip(tr["pipelines"], t[1].steps):
                assert pr.name == pl[0]


@pytest.fixture
def transformer(transformers_raw):
    transformers = [ColumnTransformerParams(**tr) for tr in transformers_raw]

    tf = build_transformer(transformers)

    return tf


def test_transform(transformer):
    X = [["nan", 1, "m", 0], [1, 0, "f", 0]]
    X_expected = [[1, 1, "m"], [1, 0, "f"]]
    X_feat = transformer.fit_transform(X)
    assert X_feat.tolist() == X_expected


def test_make_features(transformer):
    df = pd.DataFrame([["nan", 1, "m", 0], [1, 0, "f", 0]])
    transformer.fit(df)

    df_exptected = pd.DataFrame([[1, 1, "m"], [1, 0, "f"]])

    df_features = make_features(transformer, df)

    assert df_features.values.tolist() == df_exptected.values.tolist()


def test_exctract_target():
    df = pd.DataFrame(
        [["nan", 1, "m", 0], [1, 0, "f", 0]], columns=["1", "2", "3", "4"]
    )
    params = FeatureParams(feature_pipelines=[], target_col="4")

    target = extract_target(df, params)

    assert target.values.tolist() == df["4"].values.tolist()
