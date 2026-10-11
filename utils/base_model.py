import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

sns.set_theme(style="whitegrid")

def cal_scale_pos_weight(y_train):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)
    print(f"Negative (0): {negative_count:,} | Positive (1): {positive_count:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    return scale_pos_weight


def train_model(X_train, y_train, model_type="logistic", **kwargs):
    model_type = model_type.lower().strip()

    if model_type in ["logistic", "logistic_regression"]:
        params = {
            "C": 0.5,
            "max_iter": 600,
            "class_weight": "balanced",
            "solver": "saga",
            "tol": 1e-3,
            "n_jobs": -1,
            "random_state": 42
        }
        params.update(kwargs)
        model = LogisticRegression(**params)

    elif model_type in ["decision_tree", "dt"]:
        params = {
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "random_state": 42
        }
        params.update(kwargs)
        model = DecisionTreeClassifier(**params)

    elif model_type in ["random_forest", "rf"]:
        params = {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 10,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }
        params.update(kwargs)
        model = RandomForestClassifier(**params)

    elif model_type in ["xgboost", "xgb"]:
        scale_pos_weight = kwargs.pop("scale_pos_weight", None)
        if scale_pos_weight is None:
            scale_pos_weight = cal_scale_pos_weight(y_train)

        params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        }
        params.update(kwargs)
        model = XGBClassifier(**params)

    # elif model_type in ["adaboost", "ada"]:
    #     base_max_depth = kwargs.pop("base_max_depth", 2)
    #     base_est = DecisionTreeClassifier(
    #         max_depth=base_max_depth,
    #         class_weight="balanced",
    #         random_state=42
    #     )
    #     params = {
    #         "estimator": base_est,
    #         "n_estimators": 150,
    #         "learning_rate": 0.1,
    #         "random_state": 42
    #     }
    #     params.update(kwargs)
    #     model = AdaBoostClassifier(**params)

    else:
        raise ValueError(f"Không hỗ trợ model_type='{model_type}'. Hãy chọn: 'logistic', 'decision_tree', 'random_forest' hoặc 'xgboost'")

    print(f"\n>>> Đang train mô hình: {model_type.upper()}...")
    model.fit(X_train, y_train)
    print(f">>> Đã train xong: {model_type.upper()}!")
    return model


def predict_proba_pos(model, X):
    return model.predict_proba(X)[:, 1]

def find_best_threshold(y_true, y_proba, start=0.10, end=0.90, step=0.01, metric="f1"):
    thresholds = np.arange(start, end + step, step)
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        })

    threshold_df = pd.DataFrame(rows)
    target_col = "f1_score" if metric == "f1" else metric
    best_row = threshold_df.sort_values(target_col, ascending=False).iloc[0]
    best_threshold = float(best_row["threshold"])

    print(f"Best threshold theo {metric}: {best_threshold:.2f}")
    return best_threshold, threshold_df


def evaluate_model(y_true, y_proba, threshold=0.5, title="MODEL EVALUATION"):
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    print(f"===== {title} (THRESHOLD = {threshold:.2f}) =====")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("PR-AUC   :", round(pr_auc, 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "Best Threshold": round(float(threshold), 2),
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Accuracy": round(acc, 4),
        "Precision Class 1": round(prec, 4),
        "Recall Class 1": round(rec, 4),
        "F1-score Class 1": round(f1, 4)
    }

def plot_threshold_metrics(threshold_df, best_threshold, title="Threshold Tuning"):
    plt.figure(figsize=(8, 4.5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1_score"], label="F1-score")
    plt.axvline(best_threshold, linestyle="--", color="red", label=f"Best Threshold = {best_threshold:.2f}")
    plt.title(f"Precision, Recall, F1-score theo Threshold - {title}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_custom(y_true, y_proba, threshold=0.5, title="Confusion Matrix"):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.title(f"{title} (Threshold = {threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curve_custom(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curve_custom(y_true, y_proba, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, title="Feature Importance"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Top {top_n} Important Features - {title}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    return fi_df

def save_model(model, model_path, info_path, best_threshold, feature_names, model_type, scale_pos_weight=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    model_info = {
        "model_type": model_type,
        "best_threshold": float(best_threshold),
        "scale_pos_weight": float(scale_pos_weight) if scale_pos_weight is not None else None,
        "features": list(feature_names)
    }
    joblib.dump(model_info, info_path)
    print(f"Đã lưu model: {model_path}")
    print(f"Đã lưu model info: {info_path}")