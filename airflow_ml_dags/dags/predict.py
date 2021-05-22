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
MODEL_PATH = Variable.get("model_path")


def wait_fot_data_and_model(input_data_dir):
    input_data_path = Path(input_data_dir)
    model_path = Path(MODEL_PATH)

    return (input_data_path / "data.csv").exists() and model_path.exists()


with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(8),
) as dag:
    wait_data_and_model = PythonSensor(
        task_id="wait-for-data-and-model",
        python_callable=wait_fot_data_and_model,
        op_kwargs={
            "input_data_dir": "/data/raw/{{ ds }}",
        },
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

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

    wait_data_and_model >> predict
