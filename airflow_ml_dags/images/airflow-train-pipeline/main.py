import argparse

from preprocess import data_preprocess
from split import train_test_split
from train import train
from validate import validate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool to preprocess, split, train and validate ml model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(help="choose step")
    setup_preprocess_parser(subparsers)
    setup_split_parser(subparsers)
    setup_train_parser(subparsers)
    setup_validate_parser(subparsers)

    arguments = parser.parse_args()

    return arguments


def callback_data_preprocess(arguments):
    data_preprocess(arguments.input_data, arguments.output_data)


def setup_preprocess_parser(subparsers):
    parser = subparsers.add_parser(
        "process",
        help="Preprocess data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to folder with raw data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        required=True,
        help="Path to folder to save processed data",
    )

    parser.set_defaults(callback=callback_data_preprocess)


def callback_train_test_split(arguments):
    train_test_split(arguments.input_data, arguments.test_size, arguments.seed)


def setup_split_parser(subparsers):
    parser = subparsers.add_parser(
        "split",
        help="Split data into train and test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to folder with processed data",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        required=False,
        default=0.1,
        help="Test size for splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random state",
    )

    parser.set_defaults(callback=callback_train_test_split)


def callback_train(arguments):
    train(arguments.config, arguments.input_data, arguments.output_model)


def setup_train_parser(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to folder with processed data",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Path to folder where model will be saved",
    )

    parser.set_defaults(callback=callback_train)


def callback_validate(arguments):
    validate(arguments.config, arguments.input_data, arguments.input_model)


def setup_validate_parser(subparsers):
    parser = subparsers.add_parser(
        "validate",
        help="Validate trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to folder with processed data",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to folder with model",
    )

    parser.set_defaults(callback=callback_validate)


def main():
    arguments = parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
