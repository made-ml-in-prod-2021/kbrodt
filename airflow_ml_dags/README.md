# Homework 3: airflow

## Preresquistes

Configure `airflow` parameters in [`dags/constants.py`](dags/constants.py) and
training parameters in [`configs/config.yml`](configs/config.yml). Set paths to
data folders in envirenment variables in [`docker-compose.yml`](./docker-compose.yml).

## Run airflow

```bash
G_USER=USER@gmail.com G_PW=PASSWORD docker-compose up -d --build
docker-compose log -f
```

To stop

```bash
docker-compose down
docker system prune
docker volume prune
docker network prune
```

## Usefull links

* [setup airflow send email](https://stackoverflow.com/questions/51829200/how-to-set-up-airflow-send-email)
* [force a task to fail](https://stackoverflow.com/questions/43111506/how-do-i-force-a-task-on-airflow-to-fail)
* [docker network](https://docs.docker.com/network/)
* [link multiple docker containers via network](https://tjtelan.com/blog/how-to-link-multiple-docker-compose-via-network/)
* [set network in airflow docker operator](https://github.com/apache/airflow/issues/8418)
* [multiple databases with postgress docker](https://github.com/mrts/docker-postgresql-multiple-databases)
* [mlflow: no module `psycopg2`](https://github.com/3loc/charts/issues/2)
* [mlflow docker-compose example](https://github.com/aganse/docker_mlflow_db/blob/master/docker-compose.yaml)

## Roadmap

- [X] Поднимите airflow локально, используя docker compose (можно использовать
      из примера https://github.com/made-ml-in-prod-2021/airflow-examples/)
- [X] (5 баллов) Реализуйте dag, который генерирует данные для обучения модели
      (генерируйте данные, можете использовать как генератор синтетики из
      первой дз, так и что-то из датасетов sklearn), вам важно проэмулировать
      ситуации постоянно поступающих данных
  - записывайте данные в /data/raw/{{ ds }}/data.csv, /data/raw/{{ ds }}/target.csv
- [X] (10 баллов) Реализуйте dag, который обучает модель еженедельно, используя
      данные за текущий день. В вашем пайплайне должно быть как минимум 4
      стадии, но дайте волю своей фантазии=)
  - подготовить данные для обучения(например, считать из /data/raw/{{ ds }} и
    положить /data/processed/{{ ds }}/train_data.csv)
  - расплитить их на train/val
  - обучить модель на train (сохранить в /data/models/{{ ds }}
  - провалидировать модель на val (сохранить метрики к модельке)
- [X] Реализуйте dag, который использует модель ежедневно (5 баллов)
  - принимает на вход данные из пункта 1 (data.csv)
  - считывает путь до модельки из airflow variables(идея в том, что когда нам
    нравится другая модель и мы хотим ее на прод
  - делает предсказание и записывает их в /data/predictions/{{ds }}/predictions.csv
- [X] Реализуйте сенсоры на то, что данные готовы для дагов тренировки и
      обучения (3 доп балла)
- [X] вы можете выбрать 2 пути для выполнения ДЗ.
  - поставить все необходимые пакеты в образ с airflow и использовать bash
    operator, python operator (0 баллов)
  - использовать DockerOperator, тогда выполнение каждой из тасок должно
    запускаться в собственном контейнере
    - 1 из дагов реализован с помощью DockerOperator (5 баллов)
    - все даги реализованы только с помощью DockerOperator (10 баллов) (пример
      https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py).

По технике, вы можете использовать такую же структуру как в примере, пакую в
разные докеры скрипты, можете использовать общий докер с вашим пакетом, но с
разными точками входа для разных тасок.

Прикольно, если вы покажете, что для разных тасок можно использовать разный
набор зависимостей.

https://github.com/made-ml-in-prod-2021/airflow-examples/blob/main/dags/11_docker.py#L27
в этом месте пробрасывается путь с хостовой машины, используйте здесь путь типа
/tmp или считывайте из переменных окружения.

- [ ] Протестируйте ваши даги (5 баллов)
      https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- [X] В docker compose так же настройте поднятие mlflow и запишите туда
      параметры обучения, метрики и артефакт(модель) (5 доп баллов)
- [ ] вместо пути в airflow variables  используйте апи Mlflow Model Registry (5
      доп баллов)
  Даг для инференса подхватывает последнюю продакшен модель.
- [X] Настройте alert в случае падения дага (3 доп. балла)
  https://www.astronomer.io/guides/error-notifications-in-airflow
- [ ] традиционно, самооценка (1 балл)
