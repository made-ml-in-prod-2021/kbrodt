import json
from pathlib import Path
from urllib.parse import urlparse

import mlflow.sklearn
from clfit.apis import load_model, predict_model, serialize_model, train_model
from clfit.config.config import ConfigSchema, build_config
from clfit.data import read_data
from clfit.features import build_transformer, extract_target, make_features
from clfit.models import get_model
from constants import TRAIN_DATA_FNAME
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score


def train(
    config_path, input_data_dir, output_model_dir, model_name, stage="Production"
):
    config = build_config(config_path)

    exp_name = Path(config.experiment_path).name
    mlflow.set_experiment(exp_name)

    experiment = mlflow.get_experiment_by_name(exp_name)
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param("model type", config.train_params.model_type)
        mlflow.log_params(config.train_params.params)

        config_name = "config.json"
        with open(config_name, "w") as f:
            json.dump(ConfigSchema().dump(config), f)

        mlflow.log_artifact(config_name)

        input_data_path = Path(input_data_dir)
        train_df = read_data(str(input_data_path / TRAIN_DATA_FNAME))

        train_target = extract_target(train_df, config.feature_params)
        train_df.drop(config.feature_params.target_col, axis=1, inplace=True)

        transformer = build_transformer(config.feature_params.feature_pipelines)
        transformer.fit(train_df)

        train_features = make_features(transformer, train_df)

        model = get_model(config.train_params)
        train_model(model, train_features, train_target)

        predicts = predict_model(model, train_features, return_proba=False)
        metrics = {
            "accuracy_score_train": accuracy_score(train_target.values, predicts),
        }
        mlflow.log_metrics(metrics)

        output_model_path = Path(output_model_dir)
        output_model_path.mkdir(exist_ok=True, parents=True)

        output_model_path = str(output_model_path / config.output_model_fname)
        serialize_model(model, transformer, output_model_path)

        model = load_model(output_model_path)
        tracking_uri = mlflow.get_tracking_uri()
        tracking_url_type_store = urlparse(tracking_uri).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model,
                model_name,
                registered_model_name=model_name,
            )
        else:
            mlflow.sklearn.log_model(model, model_name)

    client = MlflowClient()
    version = len(list(client.search_model_versions(f"name='{model_name}'")))
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
    )
