import argparse
import logging

import numpy as np
import requests
from clfit.data import read_data

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_DATA_PATH = "heart.csv"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-path",
        type=str,
        required=False,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--host",
        type=str,
        required=False,
        default=DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=DEFAULT_PORT,
    )

    arguments = parser.parse_args()

    return arguments


def main():
    arguments = parse_args()

    df = read_data(arguments.data_path)
    df.drop("target", inplace=True, axis=1)

    features = ["Id"]
    features.extend(df.columns)

    logger.info("request features %s", features)
    for row in df.itertuples():
        data = [x.item() if isinstance(x, np.generic) else x for x in row]
        logger.info("requests %s", data)

        request = {"data": [data], "features": features}
        response = requests.get(
            f"http://{arguments.host}:{arguments.port}/predict/",
            json=request,
        )

        logger.info("resonse status code %s", response.status_code)
        logger.info("resonse %s", response.json())


if __name__ == "__main__":
    main()
