from pathlib import Path

from clfit.config import build_config
from clfit.data import read_data, split_data
from constants import DEV_DATA_FNAME, TRAIN_DATA_FNAME


def train_test_split(config_path, input_data_dir):
    config = build_config(config_path)
    input_data_path = Path(input_data_dir)
    df = read_data(str(input_data_path / TRAIN_DATA_FNAME))

    train_df, dev_df = split_data(df, config.splitting_params)

    train_df.to_csv(input_data_path / TRAIN_DATA_FNAME, index=False)
    dev_df.to_csv(input_data_path / DEV_DATA_FNAME, index=False)
