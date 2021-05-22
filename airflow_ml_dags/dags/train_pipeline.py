import datetime
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
DATA_PATH = Variable.get("data_path")


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
        volumes=[f"{DATA_PATH}:/data"],
        entrypoint="python main.py",
    )

    return operator


with DAG(
    "train_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(8),
) as dag:
    wait_data_and_target = PythonSensor(
        task_id="wait-for-data-and-target",
        python_callable=wait_fot_data_and_target,
        op_kwargs={
            "input_data_dir": "/data/raw/{{ ds }}",
        },
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    cmd = (
        "process --input-data /data/raw/{{ ds }} --output-data /data/processed/{{ ds }}"
    )
    preprocessing = generate_operator("process", cmd)

    cmd = "split --input-data /data/processed/{{ ds }} --test-size 0.1 --seed 42"
    split = generate_operator("split", cmd)

    cmd = "train --config /config.yml --input-data /data/processed/{{ ds }} --output-model /data/models/{{ ds }}"
    train = generate_operator("train", cmd)

    cmd = "validate --config /config.yml --input-data /data/processed/{{ ds }} --input-model /data/models/{{ ds }}"
    validate = generate_operator("validate", cmd)

    wait_data_and_target >> preprocessing >> split >> train >> validate
