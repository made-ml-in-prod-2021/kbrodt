import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
import uvicorn
from clfit.apis import load_model, predict_model
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sklearn.pipeline import Pipeline

from config import ENTRY_POINT_MSG, ID_COL, STATUS_CODE, HeartDisease, Patient

TIME_TO_SLEEP = 20
TIME_TO_FAIL = 90

logger = logging.getLogger(__name__)
model: Optional[Pipeline] = None
app = FastAPI()
start_time = datetime.now()


def make_predict(
    data: List[List[Union[float, int]]], features: List[str], model: Pipeline
) -> List[HeartDisease]:
    df = pd.DataFrame(data, columns=features)
    predicts = predict_model(model, df.drop(ID_COL, axis=1), return_proba=True)

    response = [
        HeartDisease(id=index, probability=proba)
        for index, proba in zip(df[ID_COL].values, predicts)
    ]

    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=STATUS_CODE,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/")
def main():
    return ENTRY_POINT_MSG


@app.on_event("startup")
def prepare_model():
    global model

    time.sleep(TIME_TO_SLEEP)

    model_path = os.getenv("PATH_TO_MODEL")
    logger.info("load model from %s", model_path)

    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)

        raise RuntimeError(err)

    model = load_model(model_path)


@app.get("/health")
def health() -> bool:
    global model
    global start_time

    now = datetime.now()
    elapsed_time = now - start_time
    if elapsed_time.seconds > TIME_TO_FAIL:
        raise Exception('app is dead')

    return not (model is None)


@app.get("/predict", response_model=List[HeartDisease])
def predict(request: Patient):
    assert model is not None

    response = make_predict(request.data, request.features, model)

    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
