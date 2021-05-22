from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from constants import (
    DATA_VOLUME_DIR,
    DEFAULT_ARGS,
    MODEL_PATH,
    PREDICTIONS_DIR,
    RAW_DATA_DIR,
    START_DATE,
)


def wait_fot_data_and_model(input_data_dir):
    input_data_path = Path(input_data_dir)
    model_path = Path(MODEL_PATH)

    return (input_data_path / "data.csv").exists() and model_path.exists()


with DAG(
    "predict",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=START_DATE,
) as dag:
    wait_data_and_model = PythonSensor(
        task_id="wait-for-data-and-model",
        python_callable=wait_fot_data_and_model,
        op_kwargs={
            "input_data_dir": RAW_DATA_DIR,
        },
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

    wait_data_and_model >> predict
