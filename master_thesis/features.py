from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import NORWAY_ZONES, RENEWABLE_KEYWORDS, SEASONAL_FEATURES
from .utils import sanitize_feature_name


class FeatureBuilder:
    """Transform raw zone-level data into leakage-safe, ML-ready features."""

    def __init__(
        self,
        local_tz: str = "Europe/Oslo",
        frequency: str = "h",
        renewable_keywords: tuple[str, ...] = RENEWABLE_KEYWORDS,
        price_lags: tuple[int, ...] = (1, 2, 3, 24, 48, 168),
        price_roll_windows: tuple[int, ...] = (3, 6, 12, 24, 72, 168),
        load_lags: tuple[int, ...] = (1, 3, 24, 168),
        load_roll_windows: tuple[int, ...] = (3, 6, 12, 24, 168),
        flow_lags: tuple[int, ...] = (1, 24, 168),
        outage_lags: tuple[int, ...] = (1, 24, 168),
        outage_roll_windows: tuple[int, ...] = (24, 168),
        gas_lags: tuple[int, ...] = (1, 24, 168),
        gas_roll_windows: tuple[int, ...] = (24, 168),
    ) -> None:
        self.local_tz = local_tz
        self.frequency = frequency
        self.renewable_keywords = renewable_keywords
        self.price_lags = price_lags
        self.price_roll_windows = price_roll_windows
        self.load_lags = load_lags
        self.load_roll_windows = load_roll_windows
        self.flow_lags = flow_lags
        self.outage_lags = outage_lags
        self.outage_roll_windows = outage_roll_windows
        self.gas_lags = gas_lags
        self.gas_roll_windows = gas_roll_windows

    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df.columns = [sanitize_feature_name(column) for column in df.columns]
        df = self._ensure_hourly_index(df)
        if "price_eur_mwh" not in df.columns:
            return pd.DataFrame()

        self._add_calendar_features(df)
        self._add_price_features(df)
        self._add_load_features(df)
        self._add_generation_features(df)
        self._add_wind_solar_forecast_features(df)
        self._add_flow_features(df)
        self._add_outage_features(df)
        self._add_gas_features(df)
        return df.dropna(subset=["price_eur_mwh"]).copy()

    def _ensure_hourly_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            raise ValueError("Input index must be timezone-aware.")
        df.index = df.index.tz_convert(self.local_tz)
        full_index = pd.date_range(df.index.min(), df.index.max(), freq=self.frequency, tz=self.local_tz)
        return df.reindex(full_index)

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> None:
        idx = pd.DatetimeIndex(df.index)
        df["hour"] = idx.hour
        df["day_of_week"] = idx.dayofweek
        df["month"] = idx.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    def _add_price_features(self, df: pd.DataFrame) -> None:
        shifted = df["price_eur_mwh"].shift(1)
        for lag in self.price_lags:
            df[f"price_eur_mwh_lag_{lag}"] = df["price_eur_mwh"].shift(lag)
        for window in self.price_roll_windows:
            df[f"price_eur_mwh_roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
            df[f"price_eur_mwh_roll_std_{window}"] = shifted.rolling(window, min_periods=window).std()

    def _add_load_features(self, df: pd.DataFrame) -> None:
        if "load_mw" in df.columns:
            shifted = df["load_mw"].shift(1)
            for lag in self.load_lags:
                df[f"load_mw_lag_{lag}"] = df["load_mw"].shift(lag)
            for window in self.load_roll_windows:
                df[f"load_mw_roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
                df[f"load_mw_roll_std_{window}"] = shifted.rolling(window, min_periods=window).std()
        if "load_forecast_mw" in df.columns:
            df["load_forecast_mw_lag_1"] = df["load_forecast_mw"].shift(1)

    def _add_generation_features(self, df: pd.DataFrame) -> None:
        generation_columns = [column for column in df.columns if column.startswith("gen_") and column.endswith("_mw")]
        if not generation_columns:
            return

        df["gen_total_mw"] = df[generation_columns].sum(axis=1)
        renewable_columns = [
            column
            for column in generation_columns
            if any(keyword.lower() in column.lower() for keyword in self.renewable_keywords)
        ]
        if renewable_columns:
            df["gen_renewable_mw"] = df[renewable_columns].sum(axis=1)
            df["gen_renewable_share"] = df["gen_renewable_mw"] / df["gen_total_mw"].replace(0, np.nan)

        df["gen_total_mw_lag_1"] = df["gen_total_mw"].shift(1)
        for column in renewable_columns:
            share_name = column.replace("_mw", "_share")
            df[share_name] = df[column] / df["gen_total_mw"].replace(0, np.nan)
            df[f"{column}_lag_1"] = df[column].shift(1)
            df[f"{share_name}_lag_1"] = df[share_name].shift(1)
        for column in ("gen_renewable_mw", "gen_renewable_share"):
            if column in df.columns:
                df[f"{column}_lag_1"] = df[column].shift(1)

    @staticmethod
    def _add_wind_solar_forecast_features(df: pd.DataFrame) -> None:
        forecast_columns = [column for column in df.columns if column.startswith("ws_forecast_") and column.endswith("_mw")]
        if not forecast_columns:
            return
        df["ws_forecast_total_mw"] = df[forecast_columns].sum(axis=1)
        df["ws_forecast_total_mw_lag_1"] = df["ws_forecast_total_mw"].shift(1)

    def _add_flow_features(self, df: pd.DataFrame) -> None:
        flow_columns = [column for column in df.columns if column.startswith("flow_") and column.endswith("_mw")]
        if not flow_columns:
            return
        df["flow_total_mw"] = df[flow_columns].sum(axis=1)
        for lag in self.flow_lags:
            df[f"flow_total_mw_lag_{lag}"] = df["flow_total_mw"].shift(lag)
        for column in flow_columns:
            df[f"{column}_lag_1"] = df[column].shift(1)

    def _add_outage_features(self, df: pd.DataFrame) -> None:
        for column in ("outage_unavailable_mw", "outage_event_count"):
            if column not in df.columns:
                continue
            shifted = df[column].shift(1)
            for lag in self.outage_lags:
                df[f"{column}_lag_{lag}"] = df[column].shift(lag)
            for window in self.outage_roll_windows:
                df[f"{column}_roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()

    def _add_gas_features(self, df: pd.DataFrame) -> None:
        if "gas_price_eur_mwh" not in df.columns:
            return
        shifted = df["gas_price_eur_mwh"].shift(1)
        for lag in self.gas_lags:
            df[f"gas_price_eur_mwh_lag_{lag}"] = df["gas_price_eur_mwh"].shift(lag)
        for window in self.gas_roll_windows:
            df[f"gas_price_eur_mwh_roll_mean_{window}"] = shifted.rolling(window, min_periods=window).mean()
            df[f"gas_price_eur_mwh_roll_std_{window}"] = shifted.rolling(window, min_periods=window).std()


class FeaturePreprocessor:
    """Training-consistent preprocessing."""

    def __init__(self) -> None:
        self.numeric_feature_names: list[str] = []
        self.median_values: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> FeaturePreprocessor:
        X_clean = X.copy().ffill().replace([np.inf, -np.inf], np.nan)
        numeric_columns = [column for column in X_clean.columns if pd.api.types.is_numeric_dtype(X_clean[column])]
        X_numeric = X_clean[numeric_columns]
        self.numeric_feature_names = list(X_numeric.columns)
        self.median_values = X_numeric.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.median_values is None:
            raise ValueError("Preprocessor is not fitted.")
        X_clean = X.copy().ffill().replace([np.inf, -np.inf], np.nan)
        X_numeric = X_clean.reindex(columns=self.numeric_feature_names)
        return X_numeric.fillna(self.median_values)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def base_feature_name(feature_name: str) -> str:
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def is_local_feature(feature_name: str) -> bool:
    return "__" not in feature_name


def is_price_history_feature(feature_name: str) -> bool:
    base = base_feature_name(feature_name)
    return base.startswith("price_eur_mwh_") and ("lag_" in base or "roll_" in base)


def is_seasonal_feature(feature_name: str) -> bool:
    return base_feature_name(feature_name) in SEASONAL_FEATURES


def is_leakage_safe_exogenous_feature(feature_name: str) -> bool:
    base = base_feature_name(feature_name)
    if base in {"price_eur_mwh", "is_spike", "spike_threshold"}:
        return False
    if is_seasonal_feature(base) or is_price_history_feature(base):
        return False
    if base.startswith("load_forecast_mw") or base.startswith("ws_forecast_"):
        return True
    if base in {"outage_unavailable_mw", "outage_event_count"}:
        return True
    if base.startswith("outage_unavailable_mw_") or base.startswith("outage_event_count_"):
        return "_lag_" in base or "_roll_" in base
    if base.startswith("load_mw_") or base.startswith("gas_price_eur_mwh_"):
        return "_lag_" in base or "_roll_" in base
    if base.startswith("gen_") or base.startswith("flow_"):
        return "_lag_" in base
    return False


def select_feature_sets(feature_df: pd.DataFrame, target_column: str) -> dict[str, list[str]]:
    all_columns = [column for column in feature_df.columns if column != target_column]
    autoregressive = sorted(
        column
        for column in all_columns
        if is_local_feature(column) and (is_seasonal_feature(column) or is_price_history_feature(column))
    )
    exogenous = sorted(column for column in all_columns if is_leakage_safe_exogenous_feature(column))
    full = sorted(set(autoregressive + exogenous + [column for column in all_columns if is_price_history_feature(column)]))
    return {
        "autoregressive": autoregressive,
        "exogenous": exogenous,
        "full": full,
    }


def build_zone_feature_frames(
    raw_all: pd.DataFrame,
    feature_builder: FeatureBuilder,
    gas_frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    zone_feature_frames: dict[str, pd.DataFrame] = {}
    for zone in NORWAY_ZONES:
        zone_columns = [column for column in raw_all.columns if column.startswith(f"{zone}__")]
        if not zone_columns:
            continue
        zone_raw = raw_all[zone_columns].copy()
        zone_raw.columns = [column.split("__", 1)[1] for column in zone_raw.columns]
        if not gas_frame.empty:
            zone_raw = zone_raw.join(gas_frame, how="left")
        zone_feature_frames[zone] = feature_builder.build(zone_raw)
    return zone_feature_frames


def assemble_modeling_frame(zone: str, zone_feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = zone_feature_frames[zone].copy()
    if base.empty:
        return base

    other_frames: list[pd.DataFrame] = []
    for other_zone, other_frame in zone_feature_frames.items():
        if other_zone == zone or other_frame.empty:
            continue
        shareable = [
            column
            for column in other_frame.columns
            if is_leakage_safe_exogenous_feature(column) or is_price_history_feature(column)
        ]
        if not shareable:
            continue
        prefixed = other_frame[shareable].copy()
        prefixed.columns = [f"{other_zone}__{column}" for column in prefixed.columns]
        other_frames.append(prefixed)

    return pd.concat([base, *other_frames], axis=1).sort_index() if other_frames else base
