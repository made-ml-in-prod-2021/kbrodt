# ML in prod: Homework 2

## Preresquistes

First follow `ml_project` to download data and train models. Then put them into this folder.
Or you can download the model via

```bash
wget https://github.com/made-ml-in-prod-2021/kbrodt/releases/download/v0.1/model.pkl
```

Build docker

```bash
docker build -t kbrodt/online_inference:v1 .
# docker push kbrodt/online_inference:v1
```

or pull

```bash
docker pull kbrodt/online_inference:v1
```

To make requests install dependencies

```bash
pip install -r requirements.txt
```

## Testing

```bash
PATH_TO_MODEL=model.pkl pytest -v
```

## Usage

Run docker

```bash
docker run --rm -p 8000:8000 kbrodt/online_inference:v1

# or without docker
PATH_TO_MODEL=model.pkl uvicorn app:app --host 0.0.0.0 --port 8000
```

and make requests

```bash
python make_request.py --host "localhost" --port "8000" --data-path "../ml_project/data/raw/heart.csv"
```

## Optimization

First I tried lightweight `Alpine` but python packages installation is [very slow](https://pythonspeed.com/articles/alpine-docker-python/).
So I moved to `python:3.9` (`886MB`), but quickly realised that `slim` version is better.

1. `python:3.9-slim-buster` (`115MB`)
2. Compose all commands into one in `RUN` and `COPY` layers
3. Install only necessary python packages

Final disk usage is `559MB`.

Of course we can tweak images by removing all unused libraries and packages.

## Usefull links

* [Docker best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [best Docker base image for Python](https://pythonspeed.com/articles/base-image-python-docker-images/)
* [pip install from git repo branch](https://stackoverflow.com/questions/20101834/pip-install-from-git-repo-branch)
* [pip install frin git subdirectory](https://stackoverflow.com/questions/13566200/how-can-i-install-from-a-git-subdirectory-with-pip)

## Roadmap

- [X] ветку назовите homework2, положите код в папку online_inference
- [X] Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI,
  так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих),
  должен быть endpoint /predict (3 балла)
- [X] Напишите тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)
- [X] Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла
- [X] Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном,
  типы не те и пр, в рамках вашей фантазии)
  (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 3 доп балла
  https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена
- [X] Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run),
  внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)
- [X] Оптимизируйте размер docker image (3 доп балла)
  (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)  -- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
- [X] опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)
- [X] напишите в readme корректные команды docker pull/run, которые должны привести к тому,
  что локально поднимется на inference ваша модель (1 балл)
  - Убедитесь, что вы можете протыкать его скриптом из пункта 3
- [X] проведите самооценку -- 1 доп балл
