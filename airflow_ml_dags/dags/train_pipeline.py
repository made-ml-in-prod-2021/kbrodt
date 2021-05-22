from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from constants import (
    CONFIG_PATH,
    CONFIGS_VOLUME_DIR,
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
)


def wait_fot_data_and_target(input_data_dir):
    input_data_path = Path(input_data_dir)

    return (input_data_path / "data.csv").exists() and (
        input_data_path / "target.csv"
    ).exists()


def generate_operator(task_id, command):
    operator = DockerOperator(
        image="airflow-ml-pipeline",
        command=command,
        network_mode="bridge",
        task_id=f"docker-airflow-train-pipeline-{task_id}",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data", f"{CONFIGS_VOLUME_DIR}:/configs:ro"],
        entrypoint="python main.py",
    )

    return operator


with DAG(
    "train_pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="@weekly",
    start_date=START_DATE,
) as dag:
    wait_data_and_target = PythonSensor(
        task_id="wait-for-data-and-target",
        python_callable=wait_fot_data_and_target,
        op_kwargs={
            "input_data_dir": RAW_DATA_DIR,
        },
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    cmd = f"process --input-data {RAW_DATA_DIR} --output-data {PROCESSED_DATA_DIR}"
    preprocessing = generate_operator("process", cmd)

    cmd = f"split --input-data {PROCESSED_DATA_DIR} --test-size 0.1 --seed 42"
    split = generate_operator("split", cmd)

    cmd = f"train --config {CONFIG_PATH} --input-data {PROCESSED_DATA_DIR} --output-model {MODEL_DIR}"
    train = generate_operator("train", cmd)

    cmd = f"validate --config {CONFIG_PATH} --input-data {PROCESSED_DATA_DIR} --input-model {MODEL_DIR}"
    validate = generate_operator("validate", cmd)

    wait_data_and_target >> preprocessing >> split >> train >> validate
