from textwrap import dedent

import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def features():
    f = pd.DataFrame(
        [
            [0, 1, 0],
            [1, 2, 1],
        ]
    )

    return f


@pytest.fixture
def target():
    return pd.Series([0, 1])


@pytest.fixture
def model():
    model = LogisticRegression()

    return model


@pytest.fixture
def trained_model(model, features, target):
    model.fit(features, target)

    return model


@pytest.fixture
def transformer():
    tr = ColumnTransformer(["all", "passthrough", [0, 1, 2]])

    return tr


@pytest.fixture
def fake_dataset(faker):
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

    return data


@pytest.fixture
def config_dict():
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

    config_dict = yaml.safe_load(config_str)

    return config_dict


@pytest.fixture
def config_path(tmp_path, fake_dataset, config_dict):
    input_data_path = str(tmp_path / "dataset.csv")
    with open(input_data_path, "w") as f:
        f.write(fake_dataset)

    experiment_path = str(tmp_path / "experiment_path")
    predict_path = str(tmp_path / "predict_path")

    config_dict["input_data_path"] = input_data_path
    config_dict["test_data_path"] = input_data_path
    config_dict["experiment_path"] = experiment_path
    config_dict["predict_path"] = predict_path
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as fout:
        yaml.dump(config_dict, fout)

    return config_path
