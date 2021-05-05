from pathlib import Path

from tools.predict import predict
from tools.train import train


def test_predict(config_path):
    train(config_path)

    predict_path = predict(config_path)
    assert Path(predict_path).exists()
