import logging
from dataclasses import dataclass
from typing import Union

import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplittingParams
from .train_params import TrainingParams

logger = logging.getLogger(__name__)


@dataclass()
class Config:
    input_data_path: str
    experiment_path: str
    output_model_fname: str
    metric_fname: str
    logging_path: str
    test_data_path: str
    predict_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


ConfigSchema = class_schema(Config)


def build_config(path_or_cfg: Union[str, dict]) -> Config:
    schema = ConfigSchema()

    if isinstance(path_or_cfg, str):
        logger.info("building config from %s", path_or_cfg)

        with open(path_or_cfg) as fin:
            config_dict = yaml.safe_load(fin)
    else:
        logger.info("building config from dict")
        config_dict = path_or_cfg

    config: Config = schema.load(config_dict)

    return config
