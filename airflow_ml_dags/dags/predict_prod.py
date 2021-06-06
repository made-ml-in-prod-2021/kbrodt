from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from constants import (
    ARTIFACT_VOLUME,
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_STAGE,
    NETWORK,
    PREDICTIONS_DIR,
    RAW_DATA_DIR,
    START_DATE,
)

with DAG(
    "predict_prod",
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

    cmd = (
        f"--input-data {RAW_DATA_DIR}/data.csv"
        f" --model-name {MODEL_NAME}"
        f" --model-stage {MODEL_STAGE}"
        f" --output-data {PREDICTIONS_DIR}/predictions.csv"
    )
    predict = DockerOperator(
        image="airflow-ml-pipeline",
        command=cmd,
        network_mode=NETWORK,
        task_id="docker-airflow-predict-prod",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data", ARTIFACT_VOLUME],
        entrypoint="python predict.py",
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        },
    )

    wait_data >> predict
