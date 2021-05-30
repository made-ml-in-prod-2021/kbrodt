import pytest
from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    dag = DagBag(dag_folder="dags/", include_examples=False)

    return dag


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() in source.items()

    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


@pytest.mark.parametrize(
    "dag_id",
    [
        pytest.param("data_download"),
        pytest.param("train_pipeline"),
        pytest.param("predict"),
        pytest.param("predict_prod"),
    ],
)
def test_dag_loaded(dag_bag, dag_id):
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


@pytest.mark.parametrize(
    "source, dag_id",
    [
        pytest.param(
            {
                "docker-airflow-data-download": [],
            },
            "data_download",
        ),
        pytest.param(
            {
                "wait-for-data": ["docker-airflow-train-pipeline-process"],
                "wait-for-target": ["docker-airflow-train-pipeline-process"],
                "docker-airflow-train-pipeline-process": [
                    "docker-airflow-train-pipeline-split"
                ],
                "docker-airflow-train-pipeline-split": [
                    "docker-airflow-train-pipeline-train"
                ],
                "docker-airflow-train-pipeline-train": [
                    "docker-airflow-train-pipeline-validate"
                ],
                "docker-airflow-train-pipeline-validate": [],
            },
            "train_pipeline",
        ),
        pytest.param(
            {
                "wait-for-data": ["docker-airflow-predict"],
                "wait-for-model": ["docker-airflow-predict"],
                "docker-airflow-predict": [],
            },
            "predict",
        ),
        pytest.param(
            {
                "wait-for-data": ["docker-airflow-predict-prod"],
                "docker-airflow-predict-prod": [],
            },
            "predict_prod",
        ),
    ],
)
def test_dag_structure(dag_bag, source, dag_id):
    dag = dag_bag.get_dag(dag_id=dag_id)
    assert_dag_dict_equal(source, dag)
