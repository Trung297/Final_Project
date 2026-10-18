from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, func
from sqlalchemy.dialects.mysql import insert as mysql_insert

logger = logging.getLogger(__name__)

ARTIFACT_NAMES = (
    "xgboost_model.pkl",
    "xgboost_info.pkl",
    "inference_preprocessor.pkl",
)


def calculate_model_version(artifacts_dir: str | Path) -> str:
    folder = Path(artifacts_dir)
    digest = hashlib.sha256()
    for filename in ARTIFACT_NAMES:
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu artifact: {path}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} phải là YYYY-MM-DD") from exc


def _valid_batch(raw: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if raw.empty:
        raise ValueError("Repository trả về batch rỗng")
    if "msno" not in raw.columns or raw["msno"].isna().any():
        raise ValueError("Batch thiếu msno")
    if raw["msno"].duplicated().any():
        raise ValueError("Có msno trùng trong cùng batch")
    if len(predictions) != len(raw):
        raise ValueError("Inference trả về số dòng không khớp")
    needed = {"churn_score", "prediction", "threshold"}
    if not needed.issubset(predictions.columns):
        raise ValueError(f"Inference thiếu cột: {needed.difference(predictions.columns)}")
    if not raw.index.equals(predictions.index):
        raise ValueError("Thứ tự/index hàng thay đổi sau inference")
    if "msno" in predictions.columns and not predictions["msno"].equals(raw["msno"]):
        raise ValueError("Thứ tự msno thay đổi sau inference")
    for field in ("churn_score", "threshold"):
        values = pd.to_numeric(predictions[field], errors="raise")
        if not (values.notna().all() and values.between(0, 1).all()):
            raise ValueError(f"{field} phải nằm trong [0,1]")
    if not predictions["prediction"].isin([0, 1]).all():
        raise ValueError("prediction chỉ được nhận 0 hoặc 1")
    expected_labels = (
        predictions["churn_score"] >= predictions["threshold"]
    ).astype(int)
    if not expected_labels.equals(predictions["prediction"].astype(int)):
        raise ValueError("prediction không nhất quán với threshold")


def _upsert(engine: Any, table: Table, rows: list[dict[str, Any]]) -> None:
    """Write a whole batch atomically. Re-runs update rows, not duplicate them."""
    if engine.dialect.name == "mysql":
        statement = mysql_insert(table).values(rows)
        statement = statement.on_duplicate_key_update(
            churn_score=statement.inserted.churn_score,
            prediction=statement.inserted.prediction,
            threshold=statement.inserted.threshold,
            updated_at=func.current_timestamp(),
        )
    elif engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        statement = sqlite_insert(table).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["observation_start", "observation_end", "model_version", "msno"],
            set_={
                "churn_score": statement.excluded.churn_score,
                "prediction": statement.excluded.prediction,
                "threshold": statement.excluded.threshold,
                "updated_at": func.current_timestamp(),
            },
        )
    else:
        raise RuntimeError("Batch prediction chỉ hỗ trợ MySQL (SQLite dùng cho tests)")

    with engine.begin() as connection:
        connection.execute(statement)


def run_batch_prediction(
    *,
    repository: Any,
    predictor: Any,
    engine: Any,
    observation_start: str = "2017-02-01",
    observation_end: str = "2017-02-28",
    model_version: str,
    batch_size: int = 500,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    start = _date(observation_start, "observation_start")
    end = _date(observation_end, "observation_end")
    if start >= end:
        raise ValueError("observation_start phải trước observation_end")
    if not isinstance(model_version, str) or len(model_version) != 64 or any(
        c not in "0123456789abcdef" for c in model_version
    ):
        raise ValueError("model_version phải là SHA-256 hex gồm 64 ký tự")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size phải là số nguyên dương")
    if limit_batches is not None and (type(limit_batches) is not int or limit_batches < 1):
        raise ValueError("limit_batches phải là số nguyên dương")

    table = Table("churn_predictions", MetaData(), autoload_with=engine)
    total_processed = 0
    total_positive = 0
    batch_count = 0

    chunks = repository.iter_active_users_features(
        observation_start=observation_start,
        observation_end=observation_end,
        batch_size=batch_size,
    )
    try:
        selected_chunks = islice(chunks, limit_batches) if limit_batches is not None else chunks
        for raw in selected_chunks:
            predictions = predictor.predict(raw)
            _valid_batch(raw, predictions)

            records = []
            for user_id, (_, pred) in zip(raw["msno"].tolist(), predictions.iterrows()):
                records.append({
                    "msno": str(user_id),
                    "observation_start": start,
                    "observation_end": end,
                    "model_version": model_version,
                    "churn_score": float(pred["churn_score"]),
                    "prediction": int(pred["prediction"]),
                    "threshold": float(pred["threshold"]),
                })

            _upsert(engine, table, records)
            batch_count += 1
            total_processed += len(records)
            total_positive += int(predictions["prediction"].sum())
            logger.info("Batch %s: lưu %s người dùng (tổng %s)", batch_count, len(records), total_processed)
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()

    return {
        "observation_start": observation_start,
        "observation_end": observation_end,
        "model_version": model_version,
        "batches": batch_count,
        "processed": total_processed,
        "predicted_churn": total_positive,
        "completed_all_batches": limit_batches is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KKBOX batch churn predictions")
    parser.add_argument("--start", default="2017-02-01", help="Observation start YYYY-MM-DD")
    parser.add_argument("--end", default="2017-02-28", help="Observation end YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Chạy thử N batch; bỏ tham số để chạy hết")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from utils.config import DB_CONFIG
    from utils.db import make_engine
    from utils.inference import ChurnPredictor
    from backend.repository import ChurnRepository

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    engine = make_engine(DB_CONFIG)
    try:
        repo = ChurnRepository(engine, transaction_mode="standard")
        predictor = ChurnPredictor()
        version = calculate_model_version(artifacts_dir)
        result = run_batch_prediction(
            repository=repo,
            predictor=predictor,
            engine=engine,
            observation_start=args.start,
            observation_end=args.end,
            model_version=version,
            batch_size=args.batch_size,
            limit_batches=args.limit_batches,
        )
        print("\nBATCH PREDICTION RESULT")
        for key, value in result.items():
            print(f"{key}: {value}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
