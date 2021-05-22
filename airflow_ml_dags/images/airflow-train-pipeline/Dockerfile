FROM airflow-ml-base

COPY main.py \
    constants.py \
    preprocess.py \
    split.py \
    train.py \
    validate.py \
    config.yml \
    /

ENTRYPOINT ["python", "main.py"]
