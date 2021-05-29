import argparse
import os
from pathlib import Path

import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from clfit.apis import load_model, predict_model
from clfit.data import read_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input data",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model-path",
        type=str,
        help="Path to model",
    )
    group.add_argument(
        "--model-name",
        type=str,
        help="Model name",
    )
    parser.add_argument(
        "--model-stage",
        type=str,
        required=False,
        default="Production",
        help="Stage",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        required=True,
        help="Path to output predictions",
    )

    arguments = parser.parse_args()

    return arguments


def save_predictions(predicts, predictions_dir):
    predicts = pd.DataFrame(predicts, columns=["predictions"])
    predictions_path = Path(predictions_dir)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predicts.to_csv(predictions_path, index=False)


def main():
    arguments = parse_args()

    if arguments.model_path is not None:
        model = load_model(arguments.model_path)
    else:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{arguments.model_name}/{arguments.model_stage}"
        )

    data = read_data(arguments.input_data)

    predicts = predict_model(model, data, return_proba=False)
    save_predictions(predicts, arguments.output_data)


if __name__ == "__main__":
    main()
