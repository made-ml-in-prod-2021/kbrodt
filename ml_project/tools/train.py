import json
import logging
import logging.config
from pathlib import Path

import hydra
import yaml
from hydra.utils import to_absolute_path

from clfit.apis import evaluate_model, predict_model, serialize_model, train_model
from clfit.config import build_config
from clfit.data import read_data, split_data
from clfit.features import build_transformer, extract_target, make_features
from clfit.models import get_model


def setup_logging(logging_config_filepath):
    with open(logging_config_filepath) as config_in:
        config = yaml.safe_load(config_in)
        logging.config.dictConfig(config)


def train(cfg):
    config = build_config(cfg)

    df = read_data(to_absolute_path(config.input_data_path))
    train_df, dev_df = split_data(df, config.splitting_params)

    transformer = build_transformer(config.feature_params.feature_pipelines)
    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, config.feature_params)

    model = get_model(config.train_params)
    train_model(model, train_features, train_target)

    path_to_save = Path(to_absolute_path(config.experiment_path))
    path_to_save.mkdir(exist_ok=True, parents=True)

    output_model_path = str(path_to_save / config.output_model_fname)
    serialize_model(model, transformer, output_model_path)

    dev_features = make_features(transformer, dev_df)
    dev_target = extract_target(dev_df, config.feature_params)

    predicts = predict_model(model, dev_features, return_proba=True)

    metrics = evaluate_model(predicts, dev_target)
    with open(path_to_save / config.metric_fname, "w") as fout:
        json.dump(metrics, fout)

    return str(path_to_save / config.output_model_fname), metrics


@hydra.main(config_path="..")
def main(cfg):
    setup_logging(to_absolute_path(cfg.logging_path))
    train(cfg)


if __name__ == "__main__":
    main()
