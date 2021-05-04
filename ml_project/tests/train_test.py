import tempfile
from pathlib import Path
from textwrap import dedent

import yaml

from tools.train import train


def test_train(faker):
    faker.set_arguments("num", {"min_value": 1, "max_value": 100})
    faker.set_arguments("cat", {"min_value": 0, "max_value": 4})
    faker.set_arguments("bin", {"min_value": 0, "max_value": 1})
    faker.set_arguments("target", {"min_value": 0, "max_value": 1})
    data = faker.csv(
        header=("num", "cat", "bin", "target"),
        data_columns=(
            "{{pyfloat:num}}",
            "{{pyint:cat}}",
            "{{pyint:bin}}",
            "{{pyint:target}}",
        ),
        num_rows=300,
        include_row_ids=False,
    ).replace("\r", "")

    fd, input_data_path = tempfile.mkstemp()
    with open(fd, "w") as f:
        f.write(data)

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
                - "num"

            - name: "log1p"
              pipelines:
                - name: "Log1p"
              columns:
                - "num"

            - name: "categorical"
              pipelines:
                - name: "OneHotEncoder"
              columns:
                - "cat"

            - name: "binary"
              pipelines: "passthrough"
              columns:
                - "bin"
          target_col: "target"
    """
    )

    experiment_path = tempfile.mkdtemp()

    config_dict = yaml.safe_load(config_str)
    config_dict["input_data_path"] = input_data_path
    config_dict["experiment_path"] = experiment_path
    fd, config_path = tempfile.mkstemp()
    with open(fd, "w") as fout:
        yaml.dump(config_dict, fout)

    path_to_model, metrics = train(config_path)

    assert Path(path_to_model).exists()

    for value in metrics.values():
        assert value >= 0
