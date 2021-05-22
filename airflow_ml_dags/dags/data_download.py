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

with DAG(
    "data_download",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(8),
) as dag:
    data_download = DockerOperator(
        image="airflow-data-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-data-download",
        do_xcom_push=False,
        volumes=[f"{DATA_PATH}:/data"],
    )
