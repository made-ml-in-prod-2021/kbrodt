import logging
import logging.config
from pathlib import Path

import hydra
import pandas as pd
import yaml
from hydra.utils import to_absolute_path

from ml_project.apis import load_model, predict_model
from ml_project.config import build_config
from ml_project.data import read_data
from ml_project.features import make_features


def setup_logging(logging_config_filepath):
    with open(logging_config_filepath) as config_in:
        config = yaml.safe_load(config_in)
        logging.config.dictConfig(config)


def predict(cfg):
    config = build_config(cfg)

    df = read_data(to_absolute_path(config.test_data_path))

    path_to_load = Path(to_absolute_path(config.experiment_path))
    output_model_path = str(path_to_load / config.output_model_fname)
    model, transformer = load_model(output_model_path)
    features = make_features(transformer, df)

    predicts = predict_model(model, features, return_proba=True)
    predicts = predicts[:, 1]

    predicts = pd.DataFrame(predicts, columns=["target"])
    predicts.to_csv(to_absolute_path(config.predict_path), index=False)

    return config.predict_path


@hydra.main()
def main(cfg):
    setup_logging(to_absolute_path(cfg.logging_path))
    predict(cfg)


if __name__ == "__main__":
    main()
