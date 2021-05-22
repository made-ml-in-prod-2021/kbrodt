from pathlib import Path

from clfit.apis import serialize_model, train_model
from clfit.config import build_config
from clfit.data import read_data
from clfit.features import build_transformer, extract_target, make_features
from clfit.models import get_model
from constants import TRAIN_DATA_FNAME


def train(config_path, input_data_dir, output_model_dir):
    config = build_config(config_path)
    input_data_path = Path(input_data_dir)
    train_df = read_data(str(input_data_path / TRAIN_DATA_FNAME))

    train_target = extract_target(train_df, config.feature_params)
    train_df.drop(config.feature_params.target_col, axis=1, inplace=True)

    transformer = build_transformer(config.feature_params.feature_pipelines)
    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)

    model = get_model(config.train_params)
    train_model(model, train_features, train_target)

    output_model_path = Path(output_model_dir)
    output_model_path.mkdir(exist_ok=True, parents=True)

    output_model_path = str(output_model_path / config.output_model_fname)
    serialize_model(model, transformer, output_model_path)
