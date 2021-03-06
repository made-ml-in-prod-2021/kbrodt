import datetime

from airflow.models import Variable
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
START_DATE = days_ago(8)
DATA_VOLUME_DIR = Variable.get("data_path")
MODEL_PATH = Variable.get("model_path")
CONFIGS_VOLUME_DIR = Variable.get("configs_path")
MLFLOW_TRACKING_URI = Variable.get("mlflow_tracking_uri")
NETWORK = Variable.get("network")
ARTIFACT_VOLUME = Variable.get("artifact_volume")
MODEL_NAME = Variable.get("model_name")
MODEL_STAGE = Variable.get("model_stage")

RAW_DATA_DIR = "/data/raw/{{ ds }}"
PROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
CONFIG_PATH = "/configs/config.yml"
MODEL_DIR = "/data/models/{{ ds }}"
PREDICTIONS_DIR = "/data/predictions/{{ ds }}"
