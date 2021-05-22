from pathlib import Path

import pandas as pd
from clfit.data import read_data

from constants import DATA_FNAME, TARGET_FNAME, TRAIN_DATA_FNAME


def data_preprocess(input_data_dir, output_data_dir):
    input_data_path = Path(input_data_dir)

    df = read_data(str(input_data_path / DATA_FNAME))
    target = read_data(str(input_data_path / TARGET_FNAME))

    output_data_path = Path(output_data_dir)
    output_data_path.mkdir(parents=True, exist_ok=True)

    df_processed = pd.concat([df, target], axis=1)
    df_processed.to_csv(output_data_path / TRAIN_DATA_FNAME, index=False)
