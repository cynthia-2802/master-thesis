from __future__ import annotations

import pandas as pd

from .config import Config
from .constants import CROSSBORDER_PAIRS, NORWAY_ZONES
from .data import EntsoeDataLoader, ExternalMarketDataLoader
from .features import FeatureBuilder, assemble_modeling_frame, build_zone_feature_frames
from .modeling import run_for_feature_set, summarize_metrics
from .utils import sanitize_feature_name


def filter_flow_pairs_for_zone(zone: str, all_pairs: tuple[tuple[str, str], ...]) -> list[tuple[str, str]]:
    """Filters interconnector pairs to those touching one zone."""
    return [(a, b) for (a, b) in all_pairs if a == zone or b == zone]


def load_or_download_raw_data(cfg: Config, data_loader: EntsoeDataLoader) -> pd.DataFrame:
    """Loads cached raw data or downloads it if needed.

    Args:
        cfg: Runtime configuration.
        data_loader: ENTSO-E data loader.

    Returns:
        pd.DataFrame: Combined raw dataset for all zones.
    """
    if cfg.raw_cache_path.exists():
        print(f"Raw data already exists at {cfg.raw_cache_path}, skipping data loading.")
        return pd.read_parquet(cfg.raw_cache_path)

    all_zone_raw_frames: list[pd.DataFrame] = []
    full_flows = data_loader.fetch_crossborder_flows(CROSSBORDER_PAIRS)

    for zone in NORWAY_ZONES:
        zone_raw = data_loader.fetch_zone_data(zone)
        zone_pairs = filter_flow_pairs_for_zone(zone, CROSSBORDER_PAIRS)
        zone_flow_columns = [sanitize_feature_name(f"flow_{a}_to_{b}_mw") for (a, b) in zone_pairs]
        zone_flows = full_flows.reindex(columns=[column for column in zone_flow_columns if column in full_flows.columns])
        zone_merged = zone_raw.join(zone_flows, how="left")
        zone_prefixed = zone_merged.copy()
        zone_prefixed.columns = [f"{zone}__{column}" for column in zone_prefixed.columns]
        all_zone_raw_frames.append(zone_prefixed)

    raw_all = pd.concat(all_zone_raw_frames, axis=1).sort_index()
    if not raw_all.empty and isinstance(raw_all.index, pd.DatetimeIndex):
        raw_all.index = raw_all.index.tz_convert(cfg.local_tz)
    raw_all.to_parquet(cfg.raw_cache_path)
    return raw_all


def run_pipeline(cfg: Config) -> None:
    """Runs the full thesis pipeline from raw data to saved outputs.

    Args:
        cfg: Runtime configuration.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    data_loader = EntsoeDataLoader(cfg)
    external_loader = ExternalMarketDataLoader(cfg)
    feature_builder = FeatureBuilder(local_tz=cfg.local_tz, frequency=cfg.frequency)

    raw_all = load_or_download_raw_data(cfg, data_loader)
    gas_frame = external_loader.load_gas_prices()
    zone_feature_frames = build_zone_feature_frames(raw_all, feature_builder, gas_frame)

    all_predictions: list[pd.DataFrame] = []
    all_metric_rows: list[dict[str, object]] = []
    all_shap_frames: list[pd.DataFrame] = []
    all_status_rows: list[dict[str, object]] = []

    for zone in NORWAY_ZONES:
        if zone not in zone_feature_frames or zone_feature_frames[zone].empty:
            print(f"[SKIP] {zone}: no usable feature table.")
            continue

        model_frame = assemble_modeling_frame(zone, zone_feature_frames)
        if model_frame.empty:
            print(f"[SKIP] {zone}: empty modeling frame.")
            continue

        model_frame.to_parquet(cfg.output_dir / f"{zone}__features.parquet")
        for feature_set in ("autoregressive", "exogenous", "full"):
            try:
                predictions, metric_rows, shap_frame, status_rows = run_for_feature_set(zone, model_frame, cfg, feature_set)
                all_predictions.append(predictions)
                all_metric_rows.extend(metric_rows)
                all_shap_frames.append(shap_frame)
                all_status_rows.extend(status_rows)
                predictions.to_parquet(cfg.output_dir / f"{zone}__{feature_set}__predictions.parquet")
                print(f"[OK] {zone}/{feature_set}: rows={len(predictions)} shap_features={len(shap_frame)}")
            except Exception as exc:
                for fold in [fold.name for fold in cfg.rolling_folds] + ["final_test"]:
                    dataset_split = "test" if fold == "final_test" else "validation"
                    all_status_rows.append(
                        {
                            "zone": zone,
                            "feature_set": feature_set,
                            "fold": fold,
                            "dataset_split": dataset_split,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                print(f"[SKIP] {zone}/{feature_set}: {type(exc).__name__}: {exc}")

    if all_predictions:
        pd.concat(all_predictions).sort_index().to_parquet(cfg.output_dir / "all_zones__predictions.parquet")
    if all_metric_rows:
        metric_frame = pd.DataFrame(all_metric_rows).sort_values(
            ["zone", "feature_set", "fold", "dataset_split", "sample", "model"]
        )
        metric_frame.to_csv(cfg.output_dir / "metrics.csv", index=False)
        validation_summary, selected_models = summarize_metrics(metric_frame)
        validation_summary.to_csv(cfg.output_dir / "metrics_validation_summary.csv", index=False)
        selected_models.to_csv(cfg.output_dir / "selected_models.csv", index=False)
    if all_status_rows:
        pd.DataFrame(all_status_rows).sort_values(["zone", "feature_set", "fold"]).to_csv(
            cfg.output_dir / "run_status.csv",
            index=False,
        )
    if all_shap_frames:
        (
            pd.concat(all_shap_frames)
            .sort_values(["zone", "feature_set", "mean_abs_shap"], ascending=[True, True, False])
            .to_csv(cfg.output_dir / "shap_importance.csv", index=False)
        )

    print(f"Saved raw: {cfg.raw_cache_path}")
    print(f"Gas control source: {cfg.gas_data_path or cfg.default_gas_path}")
    if all_metric_rows:
        print(f"Saved metrics: {cfg.output_dir / 'metrics.csv'}")
        print(f"Saved validation summary: {cfg.output_dir / 'metrics_validation_summary.csv'}")
        print(f"Saved selected models: {cfg.output_dir / 'selected_models.csv'}")
    if all_status_rows:
        print(f"Saved run status: {cfg.output_dir / 'run_status.csv'}")
    if all_predictions:
        print(f"Saved predictions: {cfg.output_dir / 'all_zones__predictions.parquet'}")
    if all_shap_frames:
        print("Saved per-zone SHAP bar plots (PNG) for each feature set.")
