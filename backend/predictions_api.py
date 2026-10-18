"""Read-only FastAPI endpoints for results saved by backend.batch_prediction.

Integration in backend/main.py, after app and engine have been created:

    from pathlib import Path
    from backend.batch_prediction import calculate_model_version
    from backend.predictions_api import create_predictions_router

    version = calculate_model_version(Path(__file__).resolve().parents[1] / "artifacts")
    app.include_router(create_predictions_router(engine, active_model_version=version))

Endpoints:
    GET /predictions           - paginated, sortable, filterable results
    GET /predictions/summary   - aggregate KPIs for the same observation window
    GET /predictions/user      - one saved prediction, using msno query param

All SQL queries are SELECT-only, with an explicit observation window and model
version, so distinct batch runs/models are never mixed unintentionally.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Literal
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import (
    Engine, MetaData, Table, asc, desc, func, select,
)
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class PredictionItem(BaseModel):
    msno: str
    observation_start: date
    observation_end: date
    model_version: str
    churn_score: float
    prediction: int
    threshold: float
    predicted_at: datetime
    updated_at: datetime


class PredictionsPage(BaseModel):
    observation_start: date
    observation_end: date
    prediction_start: date
    prediction_end: date
    model_version: str
    total: int
    limit: int
    offset: int
    items: list[PredictionItem]


class PredictionsSummary(BaseModel):
    observation_start: date
    observation_end: date
    prediction_start: date
    prediction_end: date
    model_version: str
    total_users: int
    predicted_churn: int
    predicted_non_churn: int
    churn_rate: float
    avg_churn_score: float | None


def _prediction_window(observation_end: date) -> tuple[date, date]:
    first = observation_end + timedelta(days=1)
    last = date(first.year, first.month, monthrange(first.year, first.month)[1])
    return first, last


def _check_window(start: date, end: date) -> None:
    if start > end:
        raise HTTPException(status_code=422, detail="observation_start phải trước hoặc bằng observation_end")


def create_predictions_router(engine: Engine, active_model_version: str) -> APIRouter:
    """Build a router with an injected database engine and serving model ID."""
    if not isinstance(active_model_version, str) or len(active_model_version) != 64:
        raise ValueError("active_model_version phải là SHA-256 có 64 ký tự")

    # Reflect the exact table that batch_prediction.py writes to, once.
    predictions = Table("churn_predictions", MetaData(), autoload_with=engine)
    router = APIRouter(prefix="/predictions", tags=["Batch Predictions"])

    def _version(requested: str | None) -> str:
        return requested if requested is not None else active_model_version

    def _where(start: date, end: date, version: str):
        return [
            predictions.c.observation_start == start,
            predictions.c.observation_end == end,
            predictions.c.model_version == version,
        ]

    def _get_rows(stmt):
        try:
            with engine.connect() as connection:
                return connection.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.exception("Không thể truy vấn churn_predictions")
            raise HTTPException(status_code=503, detail="Không thể truy vấn MySQL") from exc

    @router.get("", response_model=PredictionsPage, summary="Danh sách dự đoán churn")
    def list_predictions(
        observation_start: date = Query(default=date(2017, 2, 1)),
        observation_end: date = Query(default=date(2017, 2, 28)),
        model_version: str | None = Query(default=None, min_length=64, max_length=64),
        prediction: int | None = Query(default=None, ge=0, le=1, description="0: Non-Churn, 1: Churn"),
        min_score: float | None = Query(default=None, ge=0.0, le=1.0),
        max_score: float | None = Query(default=None, ge=0.0, le=1.0),
        sort_by: Literal["churn_score", "msno", "predicted_at"] = Query(default="churn_score"),
        order: Literal["asc", "desc"] = Query(default="desc"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        _check_window(observation_start, observation_end)
        if min_score is not None and max_score is not None and min_score > max_score:
            raise HTTPException(status_code=422, detail="min_score không được lớn hơn max_score")

        version = _version(model_version)
        criteria = _where(observation_start, observation_end, version)
        if prediction is not None:
            criteria.append(predictions.c.prediction == prediction)
        if min_score is not None:
            criteria.append(predictions.c.churn_score >= min_score)
        if max_score is not None:
            criteria.append(predictions.c.churn_score <= max_score)

        sort_columns = {
            "churn_score": predictions.c.churn_score,
            "msno": predictions.c.msno,
            "predicted_at": predictions.c.predicted_at,
        }
        sort_column = sort_columns[sort_by]
        order_expr = asc(sort_column) if order == "asc" else desc(sort_column)
        query = (
            select(predictions)
            .where(*criteria)
            .order_by(order_expr, asc(predictions.c.msno))
            .limit(limit)
            .offset(offset)
        )
        count_query = select(func.count()).select_from(predictions).where(*criteria)
        try:
            with engine.connect() as connection:
                total = int(connection.execute(count_query).scalar_one())
                items = [dict(row) for row in connection.execute(query).mappings()]
        except SQLAlchemyError as exc:
            logger.exception("Không thể lấy danh sách churn_predictions")
            raise HTTPException(status_code=503, detail="Không thể truy vấn MySQL") from exc

        pred_start, pred_end = _prediction_window(observation_end)
        return {
            "observation_start": observation_start,
            "observation_end": observation_end,
            "prediction_start": pred_start,
            "prediction_end": pred_end,
            "model_version": version,
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }

    @router.get("/summary", response_model=PredictionsSummary, summary="KPI dự đoán churn")
    def get_predictions_summary(
        observation_start: date = Query(default=date(2017, 2, 1)),
        observation_end: date = Query(default=date(2017, 2, 28)),
        model_version: str | None = Query(default=None, min_length=64, max_length=64),
    ):
        _check_window(observation_start, observation_end)
        version = _version(model_version)
        statement = select(
            func.count().label("total_users"),
            func.coalesce(func.sum(predictions.c.prediction), 0).label("predicted_churn"),
            func.avg(predictions.c.churn_score).label("avg_churn_score"),
        ).where(*_where(observation_start, observation_end, version))
        try:
            with engine.connect() as connection:
                result = connection.execute(statement).mappings().one()
        except SQLAlchemyError as exc:
            logger.exception("Không thể thống kê churn_predictions")
            raise HTTPException(status_code=503, detail="Không thể truy vấn MySQL") from exc

        total = int(result["total_users"])
        churn = int(result["predicted_churn"])
        pred_start, pred_end = _prediction_window(observation_end)
        return {
            "observation_start": observation_start,
            "observation_end": observation_end,
            "prediction_start": pred_start,
            "prediction_end": pred_end,
            "model_version": version,
            "total_users": total,
            "predicted_churn": churn,
            "predicted_non_churn": total - churn,
            "churn_rate": round(churn / total, 6) if total else 0.0,
            "avg_churn_score": float(result["avg_churn_score"]) if total else None,
        }

    @router.get("/user", response_model=PredictionItem, summary="Dự đoán đã lưu của một người dùng")
    def get_user_prediction(
        msno: str = Query(..., min_length=1, description="Mã người dùng (cho phép có dấu /)"),
        observation_start: date = Query(default=date(2017, 2, 1)),
        observation_end: date = Query(default=date(2017, 2, 28)),
        model_version: str | None = Query(default=None, min_length=64, max_length=64),
    ):
        _check_window(observation_start, observation_end)
        version = _version(model_version)
        statement = select(predictions).where(
            *_where(observation_start, observation_end, version),
            predictions.c.msno == msno,
        ).limit(1)
        rows = _get_rows(statement)
        if not rows:
            raise HTTPException(status_code=404, detail="Không tìm thấy dự đoán đã lưu cho người dùng này")
        return dict(rows[0])

    return router
