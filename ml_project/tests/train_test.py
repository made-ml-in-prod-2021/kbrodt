from pathlib import Path

from tools.train import train


def test_train(config_path):
    path_to_model, metrics = train(config_path)

    assert Path(path_to_model).exists()

    for value in metrics.values():
        assert value >= 0
