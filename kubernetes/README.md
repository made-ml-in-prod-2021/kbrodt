# Homework 4: kubernetes

## Prerequisites

Install [kubectl](https://kubernetes.io/docs/tasks/tools/)

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install kubectl ~/.local/bin/kubectl
```

and [minikube](https://minikube.sigs.k8s.io/docs/start/)

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
install minikube-linux-amd64 ~/.local/bin/minikube
```

Run cluster via `minikube start` and check `kubectl cluster-info`.

To delete all of the minikube clusters:

```bash
minikube delete --all
```

## Commands

### Pod: online-inference

Run following commands to create `online-inference` pod

```bash
kubectl apply -f ./online-inference-pod.yaml
kubectl get pods
```

To connect

```bash
kubectl port-forward pods/online-inference 8000:8000
```

and `online-inference` will be avalable at [localhost](http://localhost:8000).

To delete specific pod run

```bash
kubectl delete pods/online-inference
```

### Monitoring

```bash
kubectl get -w pods
kubectl get -w deployments.apps
kubectl get -w replicasets.apps
```

### Get a shell

```bash
kubectl exec --stdin --tty online-inference -- /bin/bash
```

## Helm

```bash
helm upgrade \
    --install online-inference-service \
    ./charts \
    --set replicas=3 \
    --set image.tag=v1

helm history online-inference-service
helm rollback online-inference-service 1
helm uninstall online-inference-service
```

## Roadmap

- [X] Установите kubectl
- [X] Разверните kubernetes
  
  Вы можете развернуть его в облаке:
  - https://cloud.google.com/kubernetes-engine
  - https://mcs.mail.ru/containers/
  - https://cloud.yandex.ru/services/managed-kubernetes
  
  Либо воспользоваться локальной инсталляцией
  - https://kind.sigs.k8s.io/docs/user/quick-start/
  - https://minikube.sigs.k8s.io/docs/start/

  Напишите, какой способ вы избрали. Убедитесь, с кластер поднялся (kubectl
  cluster-info) (5 баллов)

- [X] Напишите простой pod manifests для вашего приложения, назовите его
      online-inference-pod.yaml
      (https://kubernetes.io/docs/concepts/workloads/pods/)
      
  Задеплойте приложение в кластер (kubectl apply -f online-inference-pod.yaml),
  убедитесь, что все поднялось (kubectl get pods) Приложите скриншот, где
  видно, что все поднялось (4 балла)

- [X] Пропишите requests/limits и напишите зачем это нужно в описание PR
      закоммитьте файл online-inference-pod-resources.yaml (2 балл)
      **минимальные (должны быть физически) и максимальные (необязательно)
      требования по вычислительным ресурсам**

- [X] Модифицируйте свое приложение так, чтобы оно стартовало не сразу(с
      задержкой секунд 20-30) и падало спустя минуты работы.  Добавьте liveness
      и readiness пробы , посмотрите что будет происходить.  Напишите в
      описании -- чего вы этим добились. **под не поднимется, пока не будет
      готов сервис, а в случае его падения под перезапустится**

  Закоммититьте отдельный манифест online-inference-pod-probes.yaml (и
  изменение кода приложения) (3 балла)
  Опубликуйте ваше приложение(из ДЗ 2) с тэгом v2

- [X] Создайте replicaset, сделайте 3 реплики вашего приложения.
      (https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/)

  Ответьте на вопрос, что будет, если сменить докер образа в манифесте и
  одновременно с этим
  - а) уменьшить число реплик **убъются лишние реплики**
  - б) увеличить число реплик **досоздастся новые реплики**
  - Поды с какими версиями образа будут внутри будут в кластере? **новые будут
    создаваться с новыми версиями, а старые остануться прежними** (3 балла)
  Закоммитьте online-inference-replicaset.yaml

- [X] Опишите деплоймент для вашего приложения.
      (https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
  
  Играя с параметрами деплоя(maxSurge, maxUnavaliable), добейтесь ситуации,
  когда при деплое новой версии 
  - Есть момент времени, когда на кластере есть как все старые поды, так и все
    новые (опишите эту ситуацию) (закоммититьте файл
    online-inference-deployment-blue-green.yaml)
    **запрещаем удалять старые `maxUnavaliable=0%`, пока не создастся все
    новые `maxSurge=100%`**
  - одновременно с поднятием новых версии, гасятся старые (закоммитите файл
    online-inference-deployment-rolling-update.yaml) (3 балла)
    **создаём один новый, потом убиваем один старый: `maxUnavaliable=50%` и
    `maxSurge=50%`**

Бонусные активности:
- [X] Установить helm и оформить helm chart, включить в состав чарта ConfigMap
      и Service. -- 5 баллов
