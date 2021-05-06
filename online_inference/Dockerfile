FROM python:3.9-slim-buster

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir "git+https://github.com/made-ml-in-prod-2021/kbrodt@homework1#egg=clfit&subdirectory=ml_project" \
    && pip install --no-cache-dir fastapi uvicorn \
    && mkdir /app

COPY app.py \
    config.py \
    model.pkl \
    /app/

WORKDIR /app

ENV PATH_TO_MODEL="model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
