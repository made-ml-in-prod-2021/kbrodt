import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
DATA_PATH = Variable.get("data_path")


def generate_operator(task_id, command):
    operator =  DockerOperator(
        image="airflow-train-pipeline",
        command=command,
        network_mode="bridge",
        task_id=f"docker-airflow-train-pipeline-{task_id}",
        do_xcom_push=False,
        volumes=[f"{DATA_PATH}:/data"],
    )

    return operator


with DAG(
    "train_pipeine",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(8),
) as dag:
    cmd = "process --input-data /data/raw/{{ ds }} --output-data /data/processed/{{ ds }}"
    preprocessing = generate_operator("process", cmd)

    cmd = "split --input-data /data/processed/{{ ds }} --test-size 0.1 --seed 42"
    split = generate_operator("split", cmd)

    cmd = "train --config /config.yml --input-data /data/processed/{{ ds }} --output-model /data/models/{{ ds }}"
    train = generate_operator("train", cmd)

    cmd = "validate --config /config.yml --input-data /data/processed/{{ ds }} --input-model /data/models/{{ ds }}"
    validate = generate_operator("validate", cmd)

    preprocessing >> split >> train >> validate
