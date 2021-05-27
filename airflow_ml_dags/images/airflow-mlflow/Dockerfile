FROM python:3.9-slim-buster

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir mlflow

EXPOSE 5000
