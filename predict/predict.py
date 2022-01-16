import sys
import logging
from typing import List, Optional, Any

import joblib
from fastapi import FastAPI
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


import uvicorn

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

v1_path = "v1"

logger.info("Loading API")
app = FastAPI(
    title="Street Group Bedroom Prediction",
    description="This API exposes predictions of number of bedrooms",
    version="1.0.0",
    openapi_url=f"/{v1_path}/openapi.json",
    docs_url=f"/{v1_path}/docs",
    redoc_url=f"/{v1_path}/redoc",
)

preprocessor_path = 'data/preprocessor.joblib'
logger.info(f"Loading preprocessor from {preprocessor_path}")
preprocessor = joblib.load(preprocessor_path)

model_path = 'data/optimal_model.joblib'
logger.info(f"Loading predictions from {model_path}")
predictions = joblib.load(model_path)


class Prediction(BaseModel):
    property_type: str
    total_floor_area: float
    number_habitable_rooms: int
    number_heated_rooms: int
    estimated_min_price: int
    estimated_max_price: int
    latitude: float
    longitude: float
    number_bedrooms: Optional[int]

    def __init__(self, **data) -> None:
        super().__init__(**data)


class PredictionOut(BaseModel):
    number_bedrooms: int
    confidence: List[float]
    result: Prediction

class MultiplePredictionsOut(BaseModel):
    result: List[Prediction]

class PerformanceReport(BaseModel):
    accuracy_score: Any
    confusion_matrix: Any
    report: Any


@app.post(f"/{v1_path}/house/predict-bedrooms")
def read_root(
    house: Prediction
) -> PredictionOut:

    X_predict_df = pd.DataFrame(house.__dict__, index=[0])
    X_transformed_df = preprocessor.transform(X_predict_df)
    y_prediction = predictions.predict(X_transformed_df)
    y_prediction_prob = predictions.predict_proba(X_transformed_df)

    house.number_bedrooms = int(y_prediction[0])

    return PredictionOut(
        number_bedrooms=int(y_prediction[0]),
        confidence=[float(v) for v in y_prediction_prob[0]],
        result=house,
    )

@app.post(f"/{v1_path}/houses/predict-bedrooms")
def read_root(
    requests: List[Prediction]
) -> MultiplePredictionsOut:

    X_predict_df = pd.DataFrame([s.__dict__ for s in requests])
    X_transformed_df = preprocessor.transform(X_predict_df)
    y_prediction = predictions.predict(X_transformed_df)

    # add prediction column to dataframe so we can unravel
    X_predict_df['number_bedrooms'] = y_prediction

    return MultiplePredictionsOut(
        result=list(map(lambda x: Prediction(**x), X_predict_df.to_dict(orient="records"))),
    )

@app.post(f"/{v1_path}/performance-report")
def read_root(
    requests: List[Prediction]
) -> PerformanceReport:

    Xy_test_df = pd.DataFrame([s.__dict__ for s in requests])

    columns = len(Xy_test_df.columns)
    attributes = columns - 1

    X_test_df = Xy_test_df.iloc[:, 0:attributes]
    y_test_df = Xy_test_df.iloc[:, attributes]

    X_transformed_df = preprocessor.transform(X_test_df)
    y_truth = y_test_df.values.ravel()
    y_prediction = predictions.predict(X_transformed_df)

    acc_score = accuracy_score(y_truth, y_prediction)
    confusion_mat = confusion_matrix(y_truth, y_prediction)
    report = classification_report(y_truth, y_prediction, output_dict=True)

    print(confusion_mat)

    logger.info(acc_score)
    logger.info(confusion_mat)
    logger.info(report)

    return PerformanceReport(
        accuracy_score=acc_score,
        confusion_matrix=[float(v) for v in confusion_mat[0]],
        report=report,
    )


@app.get(f"/{v1_path}/health")
def redirect():
    return {"detail": "healthy boi"}


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8080, reload=True)

