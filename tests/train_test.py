import json
import tempfile
from pathlib import Path
from textwrap import dedent

import yaml

from ml_project.apis import evaluate_model, predict_model, serialize_model, train_model
from ml_project.config import build_config
from ml_project.data import read_data, split_data
from ml_project.features import build_transformer, extract_target, make_features
from ml_project.models import get_model


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

    config = build_config(config_path)

    df = read_data(config.input_data_path)
    train_df, dev_df = split_data(df, config.splitting_params)

    transformer = build_transformer(config.feature_params.feature_pipelines)
    transformer.fit(train_df)

    train_features = make_features(transformer, train_df)
    train_target = extract_target(train_df, config.feature_params)

    model = get_model(config.train_params)
    train_model(model, train_features, train_target)

    path_to_save = Path(config.experiment_path)
    path_to_save.mkdir(exist_ok=True, parents=True)

    serialize_model((model, transformer), path_to_save / config.output_model_fname)
    assert (path_to_save / config.output_model_fname).exists()

    dev_features = make_features(transformer, dev_df)
    dev_target = extract_target(dev_df, config.feature_params)

    predicts = predict_model(model, dev_features, return_proba=True)
    predicts = predicts[:, 1]

    metrics = evaluate_model(predicts, dev_target)
    with open(path_to_save / config.metric_fname, "w") as fout:
        json.dump(metrics, fout)

    del metrics

    assert (path_to_save / config.metric_fname).exists()
    with open(path_to_save / config.metric_fname) as fin:
        metrics = json.load(fin)

    for value in metrics.values():
        assert value >= 0
