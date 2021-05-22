from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from constants import DATA_VOLUME_DIR, DEFAULT_ARGS, RAW_DATA_DIR, START_DATE

with DAG(
    "data_download",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=START_DATE,
) as dag:
    cmd = RAW_DATA_DIR
    data_download = DockerOperator(
        image="airflow-data-download",
        command=cmd,
        network_mode="bridge",
        task_id="docker-airflow-data-download",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUME_DIR}:/data"],
    )
