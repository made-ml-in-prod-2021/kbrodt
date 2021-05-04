# ML in prod: Homework 1

## Data

Download [data](https://www.kaggle.com/ronitf/heart-disease-uci) and extract into folder `data/raw`

```bash
mkdir -p data/raw && unzip archive.zip -d data/raw
```

## Preresquistes

* [`Python 3`](https://www.python.org/)
* `virtualenv` (`pip install virtualenv`)

Create virtual envirenment

```bash
virtualenv venv
```

and activate it

```bash
. venv/bin/activate
```

Install all necessary packages

```bash
pip install -r requirements.txt
```

## EDA

Run

```bash
mkdir -p report && python tools/eda.py > report/README.md
```

to get some eda. See in [`report`](./report).

## Tests

```bash
pip install -r requirements_dev.txt
python -m pytest . -v --cov
```

## Usage

### Installation

```bash
pip install .
```

### Training

```bash
python train.py --config-name ./configs/config_lr.yml
python train.py --config-name ./configs/config_rf.yml
```

### Testing

```bash
python train.py --config-name ./configs/config_lr.yml
python train.py --config-name ./configs/config_rf.yml
```
