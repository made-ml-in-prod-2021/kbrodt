import tempfile
from textwrap import dedent

import yaml

from ml_project.config import build_config
from ml_project.config.config import ConfigSchema


def test_build_config():
    config_str = dedent(
        """
        input_data_path: "data/raw/heart.csv"
        logging_path: "configs/logging.conf.yml"

        experiment_path: "models/logreg"
        output_model_fname: "model.pkl"
        metric_fname: "metrics.json"
        test_data_path: "data/raw/heart.csv"
        predict_path: "models/logreg/predicts.csv"

        splitting_params:
          test_size: 0.1
          seed: 314159

        train_params:
          model_type: "LogisticRegression"

        feature_params:
          feature_pipelines:
            - name: "numeric"
              pipelines:
                - name: "SimpleImputer"
                  params:
                    strategy: "mean"
              columns:
                - "age"

            - name: "log1p"
              pipelines:
                - name: "Log1p"
              columns:
                - "trestbps"

            - name: "categorical"
              pipelines:
                - name: "OneHotEncoder"
              columns:
                - "cp"

            - name: "binary"
              pipelines: "passthrough"
              columns:
                - "sex"
          target_col: "target"
    """
    )

    config_dict = yaml.safe_load(config_str)
    fd, path = tempfile.mkstemp()
    with open(fd, "w") as fout:
        yaml.dump(config_dict, fout)

    config = build_config(path)
    config = ConfigSchema().dump(config)

    def is_subset(subset, superset):
        if isinstance(subset, dict):
            return all(
                key in superset and is_subset(val, superset[key])
                for key, val in subset.items()
            )

        if isinstance(subset, list) or isinstance(subset, set):
            return all(
                any(is_subset(subitem, superitem) for superitem in superset)
                for subitem in subset
            )

        # assume that subset is a plain value if none of the above match
        return subset == superset

    assert is_subset(config_dict, config)
