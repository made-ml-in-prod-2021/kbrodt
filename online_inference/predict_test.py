import random

import pytest
from fastapi.testclient import TestClient

from app import app
from config import (
    CAT_OUT_OF_RANGE_ERR,
    FEATURES,
    FEATURES_MINMAX_VALS,
    INCORRECT_FEATURES_ERR,
    N_FEATURES,
    STATUS_CODE,
)

N_SAMPLES = 10


def make_request(data, features):
    return {
        "data": data,
        "features": features,
    }


@pytest.fixture
def no_data():
    samples = []

    return make_request(samples, FEATURES)


@pytest.fixture
def data():
    samples = [
        [
            random.randint(*min_max) if min_max is not None else random.random()
            for _, min_max in FEATURES_MINMAX_VALS
        ]
        for _ in range(N_SAMPLES)
    ]

    return make_request(samples, FEATURES)


@pytest.fixture
def data_bad_types():
    samples = [["kek"] * N_FEATURES] * N_SAMPLES

    return make_request(samples, FEATURES)


@pytest.fixture
def data_with_cat_vals_oor():
    samples = [
        [
            random.randint(*min_max) + max(min_max) + 1
            if min_max is not None
            else random.random()
            for _, min_max in FEATURES_MINMAX_VALS
        ]
        for _ in range(N_SAMPLES)
    ]

    return make_request(samples, FEATURES)


@pytest.fixture
def data_bad_order(data):
    samples = data["data"]
    samples = [sample for sample in samples]

    return make_request(samples, FEATURES[::-1])


@pytest.fixture
def data_without_one_column(data):
    samples = data["data"]
    samples = [sample[1:] for sample in samples]

    return make_request(samples, FEATURES)


@pytest.fixture
def data_with_extra_columns(data):
    samples = data["data"]
    samples = [sample + sample[1:] for sample in samples]

    return make_request(samples, FEATURES)


@pytest.fixture
def features_without_one_column(data):
    samples = data["data"]

    return make_request(samples, FEATURES[1:])


@pytest.fixture
def features_with_extra_columns(data):
    samples = data["data"]

    return make_request(samples, FEATURES + FEATURES[1:])


@pytest.fixture
def data_features_without_one_column(data):
    samples = data["data"]
    samples = [sample[1:] for sample in samples]

    return make_request(samples, FEATURES[1:])


@pytest.fixture
def data_features_with_extra_columns(data):
    samples = data["data"]
    samples = [sample + sample[1:] for sample in samples]

    return make_request(samples, FEATURES + FEATURES[1:])


@pytest.fixture
def data_without_one_column_features_with_extra_one(data):
    samples = data["data"]
    samples = [sample[1:] for sample in samples]

    return make_request(samples, FEATURES + FEATURES[1:])


@pytest.fixture
def data_with_extra_columns_features_without(data):
    samples = data["data"]
    samples = [sample + sample[1:] for sample in samples]

    return make_request(samples, FEATURES[1:])


def test_prediction_without_data(no_data):
    with TestClient(app) as client:
        response = client.get("/predict/", json=no_data)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert (
            response_json["detail"][0]["msg"]
            == "ensure this value has at least 1 items"
        )


def test_prediction_bad_types(data_bad_types):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_bad_types)
        response_json = response.json()

        assert len(response_json["detail"]) == 2 * N_SAMPLES * N_FEATURES
        assert response.status_code == STATUS_CODE
        assert all(
            item["msg"] == "value is not a valid float"
            for item in response_json["detail"][::2]
        )
        assert all(
            item["msg"] == "value is not a valid integer"
            for item in response_json["detail"][1::2]
        )


def test_prediction(data):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data)
        response_json = response.json()

        assert response.status_code == 200
        assert len(response_json) == N_SAMPLES


def test_prediction_with_categorical_values_out_of_range(data_with_cat_vals_oor):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_with_cat_vals_oor)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == 1
        assert all(
            item["msg"] == CAT_OUT_OF_RANGE_ERR for item in response_json["detail"]
        )


def test_prediction_incorrect_order(data_bad_order):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_bad_order)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == 1
        assert all(
            item["msg"] == INCORRECT_FEATURES_ERR for item in response_json["detail"]
        )


def test_prediction_incorrect_data_without_one_column(data_without_one_column):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_without_one_column)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES
        assert all(
            item["msg"] == f"ensure this value has at least {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_data_with_extra_columns(data_with_extra_columns):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_with_extra_columns)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES
        assert all(
            item["msg"] == f"ensure this value has at most {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_features_without_one_columns(features_without_one_column):
    with TestClient(app) as client:
        response = client.get("/predict/", json=features_without_one_column)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == 1
        assert all(
            item["msg"] == f"ensure this value has at least {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_features_with_extra_columns(features_with_extra_columns):
    with TestClient(app) as client:
        response = client.get("/predict/", json=features_with_extra_columns)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == 1
        assert all(
            item["msg"] == f"ensure this value has at most {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_data_features_without_one_columns(
    data_features_without_one_column,
):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_features_without_one_column)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES + 1
        assert all(
            item["msg"] == f"ensure this value has at least {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_data_features_with_extra_columns(
    data_features_with_extra_columns,
):
    with TestClient(app) as client:
        response = client.get("/predict/", json=data_features_with_extra_columns)
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES + 1
        assert all(
            item["msg"] == f"ensure this value has at most {N_FEATURES} items"
            for item in response_json["detail"]
        )


def test_prediction_incorrect_data_without_one_columns_features_with(
    data_without_one_column_features_with_extra_one,
):
    with TestClient(app) as client:
        response = client.get(
            "/predict/", json=data_without_one_column_features_with_extra_one
        )
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES + 1
        assert all(
            item["msg"] == f"ensure this value has at least {N_FEATURES} items"
            for item in response_json["detail"][:-1]
        )
        assert (
            response_json["detail"][-1]["msg"]
            == f"ensure this value has at most {N_FEATURES} items"
        )


def test_prediction_incorrect_data_with_extra_columns_features_without(
    data_with_extra_columns_features_without,
):
    with TestClient(app) as client:
        response = client.get(
            "/predict/", json=data_with_extra_columns_features_without
        )
        response_json = response.json()

        assert response.status_code == STATUS_CODE
        assert len(response_json["detail"]) == N_SAMPLES + 1
        assert all(
            item["msg"] == f"ensure this value has at most {N_FEATURES} items"
            for item in response_json["detail"][:-1]
        )
        assert (
            response_json["detail"][-1]["msg"]
            == f"ensure this value has at least {N_FEATURES} items"
        )
