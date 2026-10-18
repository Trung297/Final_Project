import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils.inference import ChurnPredictor
from sqlalchemy.exc import SQLAlchemyError
from utils.db import make_engine
from utils.config import DB_CONFIG
from backend.repository import ChurnRepository
from pathlib import Path
from backend.batch_prediction import calculate_model_version
from backend.predictions_api import create_predictions_router
from backend.shap_api import create_shap_router

app = FastAPI(
    title="KKBOX Churn Prediction API",
    version="1.0.0"
)
predictor = ChurnPredictor()

engine = make_engine(DB_CONFIG)

repository = ChurnRepository(
    engine=engine,
    transaction_mode="standard"
)

model_version = calculate_model_version(
    Path(__file__).resolve().parents[1] / "artifacts"
)

app.include_router(
    create_predictions_router(
        engine=engine,
        active_model_version=model_version
    )
)

app.include_router(
    create_shap_router(
        repository=repository,
        predictor=predictor
    )
)

class PredictionRequest(BaseModel):
    features: dict[str, float | int | None]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df = pd.DataFrame([request.features])
        result = predictor.predict(df).iloc[0]

        return {
            "churn_score": float(result["churn_score"]),
            "prediction": int(result["prediction"]),
            "threshold": float(result["threshold"])
        }

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

class UserPredictionRequest(BaseModel):
    msno: str = Field(
        ...,
        min_length=1,
        description="Mã người dùng KKBOX"
    )

@app.post("/predict/user")
def predict_user(request: UserPredictionRequest):
    msno = request.msno.strip()
    if not msno:
        raise HTTPException(
            status_code=422,
            detail="msno không được để trống"
        )
    try:
        user_data = repository.get_user_features(
            msno=msno,
            observation_start="2017-02-01",
            observation_end="2017-02-28"
        )
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Không thể truy vấn MySQL"
        )
    if user_data.empty:
        raise HTTPException(
            status_code=404,
            detail="Người dùng không có hoạt động trong tháng 2/2017"
        )

    result = predictor.predict(user_data).iloc[0]

    churn_score = float(result["churn_score"])
    prediction = int(result["prediction"])
    threshold = float(result["threshold"])

    return {
        "msno": msno,
        "observation_window": {
            "start": "2017-02-01",
            "end": "2017-02-28"
        },
        "prediction_window": {
            "start": "2017-03-01",
            "end": "2017-03-31"
        },
        "churn_score": churn_score,
        "prediction": prediction,
        "threshold": threshold,
        "status": (
            "Churn" if prediction == 1
            else "Non-Churn"
        )
    }
