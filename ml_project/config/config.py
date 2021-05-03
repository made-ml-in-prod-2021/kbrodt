import logging
from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplittingParams
from .train_params import TrainingParams

logger = logging.getLogger(__name__)


@dataclass()
class Config:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


ConfigSchema = class_schema(Config)


def build_config(path: str) -> Config:
    logger.info("building config from %s", path)

    with open(path) as fin:
        schema = ConfigSchema()
        config_dict = yaml.safe_load(fin)
        config: Config = schema.load(config_dict)

        return config
