from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
import logging

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

FEATURE_LABELS_VI = {
    "active_days": "Số ngày hoạt động",
    "total_secs": "Tổng thời gian nghe nhạc",
    "num_100": "Số lượt nghe gần hết bài",
    "num_25": "Số lượt nghe dưới 25% bài hát",
    "num_unq": "Số bài hát khác nhau",
    "city": "Mã thành phố",
    "age": "Tuổi",
    "registered_via": "Kênh đăng ký",
    "tx_count": "Số giao dịch",
    "total_paid": "Tổng số tiền thanh toán",
    "auto_renew_ratio": "Tỷ lệ giao dịch tự động gia hạn",
    "cancel_count": "Số giao dịch hủy",
    "decay_ratio": "Mức thay đổi thời gian nghe giữa hai nửa tháng",
    "skip_ratio": "Tỷ lệ lượt nghe ngắn",
    "avg_secs_per_active_day": "Thời gian nghe trung bình mỗi ngày hoạt động",
    "is_age_missing": "Cờ thiếu tuổi hợp lệ",
}


class ExplanationUnavailable(RuntimeError):
    """Stored explainer is absent, incompatible, or does not match the model."""

class FeatureContribution(BaseModel):
    feature: str
    label_vi: str
    scaled_value: float
    shap_value: float
    direction: str


class UserExplanation(BaseModel):
    msno: str
    observation_start: date
    observation_end: date
    prediction_start: date
    prediction_end: date
    churn_score: float
    prediction: int
    threshold: float
    status: str
    shap_unit: str
    base_value: float
    output_value: float
    baseline_probability: float
    reconstructed_probability: float
    contributions: list[FeatureContribution]
    top_features: list[FeatureContribution]
    summary_vi: str
    explanation_vi: list[str]
    note: str


def _raw_number(row: Any, feature: str) -> float | None:
    value = row.get(feature)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _format_feature_value_vi(feature: str, row: Any) -> str | None:
    number = _raw_number(row, feature)

    if feature == "decay_ratio":
        first = _raw_number(row, "secs_first_half")
        second = _raw_number(row, "secs_second_half")
        if first is None or second is None:
            return None
        trend = "tăng" if second > first else "giảm" if second < first else "không đổi"
        return (
            f"thời gian nghe nửa đầu {first / 3600:.1f} giờ, "
            f"nửa sau {second / 3600:.1f} giờ (xu hướng {trend})"
        )

    if feature == "skip_ratio":
        short = _raw_number(row, "num_25")
        full = _raw_number(row, "num_100")
        if short is None or full is None:
            return None
        ratio = short / (full + short + 1.0)
        return f"tỷ lệ lượt nghe ngắn {ratio:.1%}"

    if feature == "avg_secs_per_active_day":
        secs = _raw_number(row, "total_secs")
        days = _raw_number(row, "active_days")
        if secs is None or days is None:
            return None
        return f"trung bình {secs / (days + 1e-5) / 60:.1f} phút/ngày hoạt động"

    if feature == "is_age_missing":
        age = _raw_number(row, "age")
        return (
            "không có tuổi hợp lệ" if age is None or age <= 0 or age > 100
            else "có tuổi hợp lệ"
        )

    if feature == "age":
        if number is None or not (0 < number <= 100):
            return "không có thông tin tuổi hợp lệ"
        return f"{number:.0f} tuổi"

    if number is None:
        return None

    if feature == "active_days":
        return f"{number:.0f} ngày hoạt động trong kỳ quan sát"
    if feature == "total_secs":
        return f"{number / 3600:.1f} giờ nghe nhạc trong kỳ quan sát"
    if feature == "num_100":
        return f"{number:,.0f} lượt nghe gần hết bài"
    if feature == "num_25":
        return f"{number:,.0f} lượt nghe dưới 25% bài hát"
    if feature == "num_unq":
        return f"{number:,.0f} lượt nghe bài hát khác nhau, cộng gộp theo ngày"
    if feature == "tx_count":
        return f"{number:,.0f} giao dịch tính đến cuối kỳ quan sát"
    if feature == "total_paid":
        return f"{number:,.0f} đơn vị thanh toán tính đến cuối kỳ quan sát"
    if feature == "auto_renew_ratio":
        return f"{number:.1%} giao dịch sử dụng tự động gia hạn"
    if feature == "cancel_count":
        return f"{number:,.0f} giao dịch hủy tính đến cuối kỳ quan sát"
    if feature == "city":
        return f"mã thành phố {number:.0f}"
    if feature == "registered_via":
        return f"mã kênh đăng ký {number:.0f}"
    return f"giá trị {number:g}"


FEATURE_LEVELS_VI = {
    "active_days": ("Hoạt động thường xuyên", "Hoạt động ít ngày", "Số ngày hoạt động ở mức trung bình"),
    "total_secs": ("Thời lượng nghe nhạc nhiều", "Thời lượng nghe nhạc ít", "Thời lượng nghe nhạc ở mức trung bình"),
    "num_100": ("Nhiều lượt nghe gần hết bài", "Ít lượt nghe gần hết bài", "Lượt nghe gần hết bài ở mức trung bình"),
    "num_25": ("Nhiều lượt nghe ngắn", "Ít lượt nghe ngắn", "Lượt nghe ngắn ở mức trung bình"),
    "num_unq": ("Nghe nhiều bài hát khác nhau", "Nghe ít bài hát khác nhau", "Số bài hát khác nhau ở mức trung bình"),
    "tx_count": ("Số giao dịch nhiều", "Số giao dịch ít", "Số giao dịch ở mức trung bình"),
    "total_paid": ("Tổng chi tiêu cao", "Tổng chi tiêu thấp", "Tổng chi tiêu ở mức trung bình"),
    "auto_renew_ratio": ("Tỷ lệ tự động gia hạn cao", "Tỷ lệ tự động gia hạn thấp", "Tỷ lệ tự động gia hạn ở mức trung bình"),
    "cancel_count": ("Nhiều lượt hủy giao dịch", "Ít lượt hủy giao dịch", "Số lượt hủy giao dịch ở mức trung bình"),
    "skip_ratio": ("Tỷ lệ nghe ngắn cao", "Tỷ lệ nghe ngắn thấp", "Tỷ lệ nghe ngắn ở mức trung bình"),
    "avg_secs_per_active_day": ("Nghe nhạc nhiều mỗi ngày hoạt động", "Nghe nhạc ít mỗi ngày hoạt động", "Thời gian nghe mỗi ngày ở mức trung bình"),
}


def _business_label_vi(item: dict, row: Any) -> str:
    feature = item["feature"]
    if feature == "decay_ratio":
        first = _raw_number(row, "secs_first_half")
        second = _raw_number(row, "secs_second_half")
        if first is not None and second is not None:
            if second < first:
                return "Thời lượng nghe nhạc giảm về cuối tháng"
            if second > first:
                return "Thời lượng nghe nhạc tăng về cuối tháng"
            return "Thời lượng nghe nhạc ổn định giữa hai nửa tháng"

    if feature in FEATURE_LEVELS_VI:
        high, low, mid = FEATURE_LEVELS_VI[feature]
        z = float(item["scaled_value"])
        if z >= 0.5:
            return high
        if z <= -0.5:
            return low
        return mid

    return item["label_vi"]


def _describe_top_features_vi(
    top_features: list[dict], unit: str, raw: Any
) -> list[str]:
    row = raw.iloc[0]
    lines = []
    for item in top_features:
        feature = item["feature"]
        observed = _format_feature_value_vi(feature, row)
        title = _business_label_vi(item, row)
        details = f" ({observed})" if observed else ""
        value = float(item["shap_value"])

        if abs(value) < 1e-9:
            outcome = "Đây không phải dấu hiệu nổi bật đối với nguy cơ ngừng hoạt động."
        elif value > 0:
            outcome = "Đây là dấu hiệu đi kèm nguy cơ ngừng hoạt động cao hơn."
        else:
            outcome = "Đây là dấu hiệu gắn với khả năng tiếp tục hoạt động cao hơn."

        lines.append(f"{title}{details}. {outcome}")
    return lines


def _sigmoid(value: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -value)))


def _select_binary_shap(result: Any, expected_features: int) -> tuple[np.ndarray, float]:
    values = np.asarray(result.values, dtype=float)
    bases = np.asarray(result.base_values, dtype=float)

    if values.shape == (1, expected_features):
        contributions = values[0]
        if bases.size != 1:
            raise ExplanationUnavailable("SHAP base_value không phải một giá trị")
        base = float(bases.reshape(-1)[0])
    elif values.shape == (1, expected_features, 2):
        contributions = values[0, :, 1]
        if bases.size != 2:
            raise ExplanationUnavailable("SHAP base_values không đủ 2 lớp")
        base = float(bases.reshape(-1)[1])
    else:
        raise ExplanationUnavailable(f"SHAP values có shape không phù hợp: {values.shape}")

    if not np.isfinite(contributions).all() or not np.isfinite(base):
        raise ExplanationUnavailable("Giá trị SHAP không hữu hạn")
    return contributions, base


class ChurnShapService:
    def __init__(self, predictor: Any, explainer_path: str | Path | None = None):
        self.predictor = predictor
        default_path = Path(__file__).resolve().parents[1] / "artifacts" / "best_shap_explainer.pkl"
        self.explainer_path = Path(explainer_path) if explainer_path is not None else default_path

    @lru_cache(maxsize=1)
    def _load_explainer(self):
        if not self.explainer_path.is_file():
            raise ExplanationUnavailable(
                "Thiếu artifacts/best_shap_explainer.pkl; cần xuất explainer từ Notebook/SHAP.ipynb"
            )
        try:
            return joblib.load(self.explainer_path)
        except (OSError, ImportError, AttributeError, ValueError) as exc:
            raise ExplanationUnavailable("Không thể nạp SHAP explainer đã lưu") from exc

    def explain(self, raw, top_k: int = 8) -> dict:
        if len(raw) != 1:
            raise ValueError("SHAP API chỉ giải thích một người dùng mỗi lần")

        scaled = self.predictor.transform(raw)
        prediction = self.predictor.predict(raw).iloc[0]
        score = float(prediction["churn_score"])
        explainer = self._load_explainer()
        unit = getattr(getattr(explainer, "model", None), "model_output", "raw")
        if unit not in ("raw", "probability"):
            raise ExplanationUnavailable(f"Không hỗ trợ SHAP model_output={unit!r}")
        try:
            explanation = explainer(scaled)
        except Exception as exc:
            raise ExplanationUnavailable("Không thể tính SHAP; kiểm tra phiên bản SHAP/model") from exc

        feature_names = list(self.predictor.feature_names)
        if list(scaled.columns) != feature_names:
            raise ExplanationUnavailable("Thứ tự đặc trưng SHAP và mô hình không trùng nhau")
        values, base = _select_binary_shap(explanation, len(feature_names))
        output = float(base + values.sum())

        if unit == "raw":
            reconstructed = _sigmoid(output)
            baseline = _sigmoid(base)
            unit_label = "log_odds"
        else:
            reconstructed = output
            baseline = base
            unit_label = "probability"

        if not np.isclose(reconstructed, score, rtol=1e-4, atol=1e-4):
            raise ExplanationUnavailable(
                "SHAP explainer không khớp với XGBoost hiện tại. "
                "Cần tạo lại best_shap_explainer.pkl từ model hiện tại, không cần train lại."
            )

        contributions = [
            {
                "feature": name,
                "label_vi": FEATURE_LABELS_VI.get(name, name),
                "scaled_value": float(scaled.iloc[0][name]),
                "shap_value": float(value),
                "direction": "increase_churn" if value > 0 else "decrease_churn" if value < 0 else "neutral",
            }
            for name, value in zip(feature_names, values)
        ]
        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        pred_label = int(prediction["prediction"])
        return {
            "churn_score": score,
            "prediction": pred_label,
            "threshold": float(prediction["threshold"]),
            "status": "Churn" if pred_label == 1 else "Non-Churn",
            "shap_unit": unit_label,
            "base_value": base,
            "output_value": output,
            "baseline_probability": baseline,
            "reconstructed_probability": reconstructed,
            "contributions": contributions,
            "top_features": contributions[:top_k],
            "explanation_vi": _describe_top_features_vi(contributions[:top_k], unit_label, raw),
            "note": (
                "Các dấu hiệu thể hiện mối liên hệ với nguy cơ ngừng hoạt động, "
                "không khẳng định nguyên nhân trực tiếp. "
                "Mức nhiều/ít, cao/thấp được đánh giá tương đối theo dữ liệu tham chiếu, "
                "không phải ngưỡng cố định cho mọi nhóm khách hàng."
            ),
        }


def create_shap_router(repository: Any, predictor: Any, explainer_path: str | Path | None = None) -> APIRouter:
    router = APIRouter(prefix="/explain", tags=["SHAP Explanation"])
    service = ChurnShapService(predictor, explainer_path=explainer_path)

    @router.get("/user", response_model=UserExplanation, summary="Giải thích SHAP cho một người dùng")
    def explain_user(
        msno: str = Query(..., min_length=1),
        observation_start: date = Query(default=date(2017, 2, 1)),
        observation_end: date = Query(default=date(2017, 2, 28)),
        top_k: int = Query(default=8, ge=1, le=16),
    ):
        clean_msno = msno.strip()
        if not clean_msno:
            raise HTTPException(status_code=422, detail="msno không được để trống")
        if observation_start >= observation_end:
            raise HTTPException(status_code=422, detail="Observation Window không hợp lệ")
        last_day = monthrange(observation_start.year, observation_start.month)[1]
        if (observation_start.day != 1 or observation_end !=
                date(observation_start.year, observation_start.month, last_day)):
            raise HTTPException(status_code=422, detail="Observation Window phải là một tháng đầy đủ")

        try:
            raw = repository.get_user_features(
                msno=clean_msno,
                observation_start=observation_start.isoformat(),
                observation_end=observation_end.isoformat(),
            )
        except SQLAlchemyError as exc:
            logger.exception("Không thể truy vấn MySQL khi giải thích SHAP")
            raise HTTPException(status_code=503, detail="Không thể truy vấn MySQL") from exc

        if raw.empty:
            raise HTTPException(status_code=404, detail="Không tìm thấy hoạt động của người dùng trong kỳ quan sát")

        try:
            output = service.explain(raw, top_k=top_k)
        except ExplanationUnavailable as exc:
            logger.exception("Không thể cung cấp SHAP Explanation")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            logger.exception("Dữ liệu đầu vào không tương thích với inference")
            raise HTTPException(status_code=503, detail="Dữ liệu đầu vào không tương thích với mô hình") from exc

        pred_start = observation_end + timedelta(days=1)
        pred_end = date(pred_start.year, pred_start.month,
                        monthrange(pred_start.year, pred_start.month)[1])
        assessment = (
            "có nguy cơ ngừng hoạt động" if output["prediction"] == 1
            else "được dự đoán tiếp tục hoạt động"
        )
        summary_vi = (
            f"Dựa trên hoạt động từ {observation_start:%d/%m/%Y} đến "
            f"{observation_end:%d/%m/%Y}, người dùng {assessment} "
            f"trong tháng {pred_start:%m/%Y}. "
            f"Điểm nguy cơ ngừng hoạt động: {output['churn_score']:.1%}. "
            "Những dấu hiệu đáng chú ý được trình bày bên dưới."
        )
        return {
            "msno": clean_msno,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "prediction_start": pred_start,
            "prediction_end": pred_end,
            "summary_vi": summary_vi,
            **output,
        }

    return router
