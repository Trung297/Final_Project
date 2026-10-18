from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "active_days", "total_secs", "num_100", "num_25", "num_unq",
    "secs_first_half", "secs_second_half", "city", "age",
    "registered_via", "tx_count", "total_paid", "auto_renew_ratio",
    "cancel_count",
]
LOG_COLUMNS = [
    "total_secs", "num_100", "num_25", "num_unq",
    "total_paid", "avg_secs_per_active_day",
]
ZERO_IF_MISSING = ["tx_count", "total_paid", "auto_renew_ratio", "cancel_count"]
UNKNOWN_IF_MISSING = ["city", "age", "registered_via"]


class ChurnPredictor:
    def __init__(self, artifacts_dir=None):
        base_dir = Path(__file__).resolve().parent.parent
        base = Path(artifacts_dir) if artifacts_dir is not None else base_dir / "artifacts"

        self.model = joblib.load(base / "xgboost_model.pkl")
        self.model_info = joblib.load(base / "xgboost_info.pkl")
        preprocess = joblib.load(base / "inference_preprocessor.pkl")

        self.scaler = preprocess["scaler"]
        self.age_median = float(preprocess["valid_age_median"])
        self.feature_names = list(preprocess["feature_names"])
        self.threshold = float(self.model_info["best_threshold"])

        if self.feature_names != list(self.model_info["features"]):
            raise ValueError("Feature names/order mismatch between model and preprocessing artifacts")
        if len(self.feature_names) != 16:
            raise ValueError(f"Expected 16 features, got {len(self.feature_names)}")
        if getattr(self.scaler, "n_features_in_", len(self.feature_names)) != len(self.feature_names):
            raise ValueError("Scaler has an incompatible number of features")

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Turn snapshot aggregates (BEFORE notebook feature engineering) into scaled model features."""
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
        missing = [col for col in RAW_COLUMNS if col not in raw.columns]
        if missing:
            raise ValueError(f"Missing raw input columns: {missing}")

        data = raw.loc[:, RAW_COLUMNS].copy()
        for col in RAW_COLUMNS:
            data[col] = pd.to_numeric(data[col], errors="raise")
        for col in ZERO_IF_MISSING:
            data[col] = data[col].fillna(0)
        for col in UNKNOWN_IF_MISSING:
            data[col] = data[col].fillna(-1)
        if data.isna().any().any():
            raise ValueError(f"Unresolved missing values: {data.columns[data.isna().any()].tolist()}")
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError("Infinite values are not supported")

        # Mirrors Preprocessing.ipynb, cells 4-5, 15 and 17.
        data["decay_ratio"] = (
            (data["secs_second_half"] - data["secs_first_half"])
            / (data["secs_first_half"] + 1.0)
        ).clip(-1.0, 10.0)
        data["skip_ratio"] = data["num_25"] / (
            data["num_100"] + data["num_25"] + 1.0
        )
        data["avg_secs_per_active_day"] = (
            data["total_secs"] / (data["active_days"] + 1e-5)
        )
        for col in LOG_COLUMNS:
            data[col] = np.log1p(np.maximum(0, data[col]))

        data["is_age_missing"] = (
            (data["age"] <= 0) | (data["age"] > 100)
        ).astype(int)
        data.loc[data["is_age_missing"] == 1, "age"] = self.age_median

        data = data.drop(columns=["secs_first_half", "secs_second_half"])
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing engineered features: {missing_features}")
        features = data.loc[:, self.feature_names]
        scaled = self.scaler.transform(features)
        return pd.DataFrame(scaled, columns=self.feature_names, index=raw.index)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        scaled = self.transform(raw)
        scores = self.model.predict_proba(scaled)[:, 1]
        result = pd.DataFrame({
            "churn_score": scores,
            "prediction": (scores >= self.threshold).astype(int),
            "threshold": self.threshold,
        }, index=raw.index)
        if "msno" in raw.columns:
            result.insert(0, "msno", raw["msno"])
        return result


if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample = pd.read_pickle(Path(__file__).resolve().parent.parent / "artifacts" / "master_eda.pkl").head(5)
    print(predictor.predict(sample).to_string(index=False))
