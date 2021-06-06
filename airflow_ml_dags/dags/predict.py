from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from constants import (
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MODEL_PATH,
    PREDICTIONS_DIR,
    RAW_DATA_DIR,
    START_DATE,
)

with DAG(
    "predict",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=START_DATE,
) as dag:
    wait_data = FileSensor(
        task_id="wait-for-data",
        filepath=str(Path(RAW_DATA_DIR) / "data.csv"),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    wait_model = FileSensor(
        task_id="wait-for-model",
        filepath=MODEL_PATH,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    cmd = (
        f"--input-data {RAW_DATA_DIR}/data.csv"
        f" --model-path {MODEL_PATH}"
        f" --output-data {PREDICTIONS_DIR}/predictions.csv"
    )
    predict = DockerOperator(
        image="airflow-ml-pipeline",
        command=cmd,
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
        entrypoint="python predict.py",
    )

    [wait_data, wait_model] >> predict
