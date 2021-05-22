from pathlib import Path

from clfit.config.split_params import SplittingParams
from clfit.data import read_data, split_data

from constants import DEV_DATA_FNAME, TRAIN_DATA_FNAME


def train_test_split(input_data_dir, test_size, seed):
    input_data_path = Path(input_data_dir)
    df = read_data(str(input_data_path / TRAIN_DATA_FNAME))

    splitting_params = SplittingParams(
        test_size=test_size,
        seed=seed,
    )
    train_df, dev_df = split_data(df, splitting_params)

    train_df.to_csv(input_data_path / TRAIN_DATA_FNAME, index=False)
    dev_df.to_csv(input_data_path / DEV_DATA_FNAME, index=False)
