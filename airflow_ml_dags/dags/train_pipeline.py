from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from constants import (
    ARTIFACT_VOLUME,
    CONFIG_PATH,
    CONFIGS_VOLUME_DIR,
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    NETWORK,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
)


def generate_operator(task_id, command):
    operator = DockerOperator(
        image="airflow-ml-pipeline",
        command=f"{task_id} {command}",
        network_mode=NETWORK,
        task_id=f"docker-airflow-train-pipeline-{task_id}",
        do_xcom_push=False,
        volumes=[
            f"{DATA_VOLUME_DIR}:/data",
            ARTIFACT_VOLUME,
            f"{CONFIGS_VOLUME_DIR}:/configs:ro",
        ],
        entrypoint="python main.py",
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        },
    )

    return operator


with DAG(
    "train_pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
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

    wait_target = FileSensor(
        task_id="wait-for-target",
        filepath=str(Path(RAW_DATA_DIR) / "target.csv"),
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocessing = generate_operator(
        "process",
        f"--input-data {RAW_DATA_DIR}" f" --output-data {PROCESSED_DATA_DIR}",
    )

    split = generate_operator(
        "split",
        f"--config {CONFIG_PATH}" f" --input-data {PROCESSED_DATA_DIR}",
    )

    train = generate_operator(
        "train",
        f"--config {CONFIG_PATH}"
        f" --input-data {PROCESSED_DATA_DIR}"
        f" --output-model {MODEL_DIR}",
    )

    validate = generate_operator(
        "validate",
        f"--config {CONFIG_PATH}"
        f" --input-data {PROCESSED_DATA_DIR}"
        f" --input-model {MODEL_DIR}",
    )

    [wait_data, wait_target] >> preprocessing >> split >> train >> validate
