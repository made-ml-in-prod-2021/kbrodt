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
MODEL_PATH = Variable.get("model_path")


with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(8),
) as dag:
    cmd = (
        "--input-data /data/raw/{{ ds }}/data.csv"
        f" --model-path {MODEL_PATH}"
        " --output-data /data/predictions/{{ ds }}/predictions.csv"
    )
    predict = DockerOperator(
        image="airflow-predict",
        command=cmd,
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_PATH}:/data"],
    )
