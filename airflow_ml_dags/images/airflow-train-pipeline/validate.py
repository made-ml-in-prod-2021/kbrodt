import json
from pathlib import Path

from clfit.apis import load_model, predict_model
from clfit.config import build_config
from clfit.data import read_data
from clfit.features import extract_target
from sklearn.metrics import accuracy_score

from constants import DEV_DATA_FNAME


def validate(config_path, input_data_dir, input_model_dir):
    config = build_config(config_path)
    input_model_path = Path(input_model_dir)
    model = load_model(str(input_model_path / config.output_model_fname))

    input_data_path = Path(input_data_dir)
    dev_df = read_data(str(input_data_path / DEV_DATA_FNAME))

    dev_target = extract_target(dev_df, config.feature_params)
    dev_df.drop(config.feature_params.target_col, axis=1, inplace=True)

    predicts = predict_model(model, dev_df, return_proba=False)

    metrics = {
        "accuracy_score": accuracy_score(dev_target.values, predicts),
    }
    with open(input_model_path / config.metric_fname, "w") as fout:
        json.dump(metrics, fout)
