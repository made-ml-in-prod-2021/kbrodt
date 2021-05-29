FROM python:3.9-slim-buster

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update \
    && apt-get install -y git gcc libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir "git+https://github.com/made-ml-in-prod-2021/kbrodt@homework1#egg=clfit&subdirectory=ml_project" \
        psycopg2 mlflow

COPY main.py \
    constants.py \
    preprocess.py \
    split.py \
    train.py \
    validate.py \
    predict.py \
    /
