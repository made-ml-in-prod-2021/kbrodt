import argparse
from pathlib import Path

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
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model",
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

    model = load_model(arguments.model_path)

    data = read_data(arguments.input_data)

    predicts = predict_model(model, data, return_proba=False)
    save_predictions(predicts, arguments.output_data)


if __name__ == "__main__":
    main()
