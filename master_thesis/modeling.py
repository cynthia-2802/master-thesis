from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .config import BacktestFold, Config
from .features import FeaturePreprocessor, select_feature_sets


def slice_time_window(
    df: pd.DataFrame,
    start: str,
    end: str,
    local_tz: str,
) -> pd.DataFrame:
    """Slices a timezone-aware frame using an inclusive-exclusive window.

    Args:
        df: Time-indexed data frame.
        start: Inclusive window start.
        end: Exclusive window end.
        local_tz: Analysis time zone.

    Returns:
        pd.DataFrame: Sliced data frame.
    """
    if df.empty:
        return df.copy()
    return df.loc[
        pd.Timestamp(start, tz=local_tz) : pd.Timestamp(end, tz=local_tz) - pd.Timedelta(hours=1)
    ].copy()


def final_test_split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the final train and untouched test blocks.

    Args:
        df: Full feature table.
        cfg: Runtime configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Final train and test frames.
    """
    if df.empty:
        return df.copy(), df.copy()
    train = slice_time_window(df, cfg.rolling_folds[0].train_start, cfg.test_start, cfg.local_tz)
    test = slice_time_window(df, cfg.test_start, cfg.test_end, cfg.local_tz)
    return train, test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes standard regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def spike_threshold_from_train(train_target: pd.Series, quantile: float) -> float:
    """Computes the spike threshold from the training target."""
    if train_target.empty:
        raise ValueError("Train target is empty; cannot compute spike threshold.")
    return float(train_target.quantile(quantile))


def add_spike_labels(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adds spike labels using a fixed threshold."""
    out = df.copy()
    out["spike_threshold"] = threshold
    out["is_spike"] = (out["price_eur_mwh"] >= threshold).astype(int)
    return out


def train_mean_models(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: list[str],
    cfg: Config,
    target_column: str = "price_eur_mwh",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
    """Fits LightGBM and XGBoost mean models on one split.

    Args:
        train_df: Training data frame.
        eval_df: Evaluation data frame.
        feature_names: Selected predictor names.
        cfg: Runtime configuration.
        target_column: Target column name.

    Returns:
        tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, object]]:
            Predictions, metrics, and fitted model artifacts.
    """
    if train_df.empty or eval_df.empty:
        raise ValueError("Train/eval split produced empty dataframes.")
    if not feature_names:
        raise ValueError("No features selected.")

    X_train_raw = train_df[feature_names].copy()
    X_eval_raw = eval_df[feature_names].copy()
    y_train = train_df[target_column].copy()
    y_eval = eval_df[target_column].copy()

    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_eval = preprocessor.transform(X_eval_raw)

    lgbm = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.random_state,
    )
    lgbm.fit(X_train, y_train)

    xgb = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    predictions = pd.DataFrame(
        {"y_true": y_eval, "pred_lgbm": lgbm.predict(X_eval), "pred_xgb": xgb.predict(X_eval)},
        index=eval_df.index,
    )
    metrics = {
        "lgbm": regression_metrics(y_eval.to_numpy(), predictions["pred_lgbm"].to_numpy()),
        "xgb": regression_metrics(y_eval.to_numpy(), predictions["pred_xgb"].to_numpy()),
    }
    artifacts: dict[str, object] = {
        "feature_names": list(X_train.columns),
        "preprocessor": preprocessor,
        "lgbm": lgbm,
        "xgb": xgb,
    }
    return predictions, metrics, artifacts


def train_quantile_models(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: list[str],
    cfg: Config,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    target_column: str = "price_eur_mwh",
) -> pd.DataFrame:
    """Fits LightGBM quantile models for one evaluation split.

    Args:
        train_df: Training data frame.
        eval_df: Evaluation data frame.
        feature_names: Selected predictor names.
        cfg: Runtime configuration.
        quantiles: Quantiles to estimate.
        target_column: Target column name.

    Returns:
        pd.DataFrame: Quantile predictions and true values.
    """
    X_train_raw = train_df[feature_names].copy()
    X_eval_raw = eval_df[feature_names].copy()
    y_train = train_df[target_column].copy()
    y_eval = eval_df[target_column].copy()

    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_eval = preprocessor.transform(X_eval_raw)

    out = pd.DataFrame(index=eval_df.index)
    for quantile in quantiles:
        model = LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=2500,
            learning_rate=0.02,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_state,
        )
        model.fit(X_train, y_train)
        out[f"q{int(quantile * 100):02d}"] = np.asarray(model.predict(X_eval))
    out["y_true"] = y_eval.to_numpy()
    return out


def shap_global_importance(
    model: object,
    X: pd.DataFrame,
    output_path: Path,
    background_size: int,
    max_display: int = 25,
) -> pd.DataFrame:
    """Computes global SHAP importance and saves a bar plot.

    Args:
        model: Fitted tree model.
        X: Feature matrix used for SHAP.
        output_path: Plot output path.
        background_size: Maximum SHAP background sample size.
        max_display: Maximum number of displayed features.

    Returns:
        pd.DataFrame: Mean absolute SHAP importance per feature.
    """
    if X.empty or X.shape[1] == 0:
        raise ValueError("SHAP input matrix is empty.")

    background = X.sample(background_size, random_state=42) if len(X) > background_size else X
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    shap_values = explainer(X, check_additivity=False)

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, 0]

    importance = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return importance


def run_for_feature_set(
    zone: str,
    feature_df: pd.DataFrame,
    cfg: Config,
    feature_set: Literal["autoregressive", "exogenous", "full"],
) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    """Runs rolling validation and final test for one feature set.

    Args:
        zone: Target zone.
        feature_df: Full modeling frame for the zone.
        cfg: Runtime configuration.
        feature_set: Feature specification to run.

    Returns:
        tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
            Predictions, metric rows, SHAP importance, and run-status rows.
    """
    selected_features = select_feature_sets(feature_df, target_column="price_eur_mwh")[feature_set]
    if not selected_features:
        raise ValueError(f"Feature set '{feature_set}' is empty.")

    combined_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    final_artifacts: dict[str, object] | None = None

    for fold in cfg.rolling_folds:
        fold_train, fold_validation = rolling_fold_split(feature_df, fold, cfg.local_tz)
        if len(fold_train) < cfg.min_train_rows or len(fold_validation) < cfg.min_validation_rows:
            raise ValueError(
                f"Not enough data for {fold.name}. train={len(fold_train)}, validation={len(fold_validation)}"
            )

        threshold = spike_threshold_from_train(fold_train["price_eur_mwh"], cfg.spike_quantile)
        fold_train = add_spike_labels(fold_train, threshold)
        fold_validation = add_spike_labels(fold_validation, threshold)

        validation_predictions, validation_metrics, artifacts = train_mean_models(
            fold_train, fold_validation, selected_features, cfg
        )
        validation_quantiles = train_quantile_models(fold_train, fold_validation, selected_features, cfg)
        fold_combined = validation_predictions.join(validation_quantiles.drop(columns=["y_true"]), how="left")
        fold_combined["zone"] = zone
        fold_combined["feature_set"] = feature_set
        fold_combined["dataset_split"] = "validation"
        fold_combined["fold"] = fold.name
        fold_combined["spike_threshold"] = threshold
        fold_combined["is_spike"] = fold_validation["is_spike"].astype(int)
        combined_frames.append(fold_combined)
        metric_rows.extend(metric_rows_for_split(zone, feature_set, fold.name, "validation", fold_combined, validation_metrics))
        status_rows.append({"zone": zone, "feature_set": feature_set, "fold": fold.name, "dataset_split": "validation", "status": "ok", "error": ""})
        final_artifacts = artifacts

    test_train_df, test_df = final_test_split(feature_df, cfg)
    if len(test_train_df) < cfg.min_train_rows or len(test_df) < cfg.min_test_rows:
        raise ValueError(f"Not enough data for final test. train={len(test_train_df)}, test={len(test_df)}")

    threshold = spike_threshold_from_train(test_train_df["price_eur_mwh"], cfg.spike_quantile)
    test_train_df = add_spike_labels(test_train_df, threshold)
    test_df = add_spike_labels(test_df, threshold)

    test_predictions, test_metrics, final_artifacts = train_mean_models(test_train_df, test_df, selected_features, cfg)
    test_quantiles = train_quantile_models(test_train_df, test_df, selected_features, cfg)
    test_combined = test_predictions.join(test_quantiles.drop(columns=["y_true"]), how="left")
    test_combined["zone"] = zone
    test_combined["feature_set"] = feature_set
    test_combined["dataset_split"] = "test"
    test_combined["fold"] = "final_test"
    test_combined["spike_threshold"] = threshold
    test_combined["is_spike"] = test_df["is_spike"].astype(int)
    combined_frames.append(test_combined)
    metric_rows.extend(metric_rows_for_split(zone, feature_set, "final_test", "test", test_combined, test_metrics))
    status_rows.append({"zone": zone, "feature_set": feature_set, "fold": "final_test", "dataset_split": "test", "status": "ok", "error": ""})

    combined = pd.concat(combined_frames).sort_index()
    if final_artifacts is None:
        raise ValueError("Model artifacts were not created.")

    preprocessor: FeaturePreprocessor = final_artifacts["preprocessor"]  # type: ignore[assignment]
    X_train_for_shap = preprocessor.transform(test_train_df[final_artifacts["feature_names"]])  # type: ignore[index]
    shap_importance = shap_global_importance(
        final_artifacts["lgbm"],
        X_train_for_shap,
        output_path=cfg.output_dir / f"{zone}__{feature_set}__shap_bar.png",
        background_size=cfg.shap_background_size,
    )
    shap_importance["zone"] = zone
    shap_importance["feature_set"] = feature_set
    return combined, metric_rows, shap_importance, status_rows


def rolling_fold_split(df: pd.DataFrame, fold: BacktestFold, local_tz: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds train and validation frames for one rolling fold."""
    train = slice_time_window(df, fold.train_start, fold.train_end, local_tz)
    validation = slice_time_window(df, fold.validation_start, fold.validation_end, local_tz)
    return train, validation


def metric_rows_for_split(
    zone: str,
    feature_set: str,
    fold_name: str,
    dataset_split: str,
    combined: pd.DataFrame,
    metrics: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Formats metric rows for one evaluation split."""
    rows: list[dict[str, object]] = []
    for model_name, metric in metrics.items():
        rows.append(
            {
                "zone": zone,
                "feature_set": feature_set,
                "fold": fold_name,
                "dataset_split": dataset_split,
                "sample": "all",
                "model": model_name,
                **metric,
            }
        )
        spike_mask = combined["is_spike"] == 1
        if spike_mask.any():
            rows.append(
                {
                    "zone": zone,
                    "feature_set": feature_set,
                    "fold": fold_name,
                    "dataset_split": dataset_split,
                    "sample": "spike_hours",
                    "model": model_name,
                    **regression_metrics(
                        combined.loc[spike_mask, "y_true"].to_numpy(),
                        combined.loc[spike_mask, f"pred_{model_name}"].to_numpy(),
                    ),
                }
            )
    return rows


def summarize_metrics(metric_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds validation summaries and selected-model tables.

    Args:
        metric_frame: Raw fold-level metric rows.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Validation summary table and selected-model table.
    """
    validation_summary = (
        metric_frame[metric_frame["dataset_split"] == "validation"]
        .groupby(["zone", "feature_set", "model", "sample"], as_index=False)[["rmse", "mae", "r2"]]
        .mean()
        .rename(columns={"rmse": "mean_rmse", "mae": "mean_mae", "r2": "mean_r2"})
    )
    selected_models = (
        validation_summary[validation_summary["sample"] == "all"]
        .sort_values(["zone", "mean_rmse", "mean_mae"], ascending=[True, True, True])
        .groupby("zone", as_index=False)
        .first()
        .rename(columns={"feature_set": "selected_feature_set", "model": "selected_model"})
    )
    return validation_summary, selected_models
