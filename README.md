# Master Thesis Pipeline

Run the end-to-end forecasting pipeline with:

```bash
uv run thesis-pipeline
```

Required environment variables:

- `ENTSOE_API_KEY`: ENTSO-E API key.

- `GAS_PRICE_PATH`: Path to a CSV or Parquet file with columns `timestamp` and `gas_price_eur_mwh`.

The pipeline:

- pulls hourly ENTSO-E data for NO1-NO5 over the fixed 2023-2025 study window,
- chunks long API requests to support multi-year downloads,
- reuses cached raw ENTSO-E data when `entsoe_no_experiment/raw_2023_2025.parquet` already exists,
- adds outage-based features from ENTSO-E unavailability data,
- downloads TTF gas prices to `entsoe_no_experiment/ttf_gas_price.parquet` when no gas file is supplied,
- builds cross-zone predictors from the other Norwegian bidding zones, and
- exports per-zone features, predictions, metrics, and SHAP summaries to `entsoe_no_experiment/`.

Code layout:

- `master_thesis/config.py`: runtime configuration and cache paths.
- `master_thesis/constants.py`: zones and cross-border pair definitions.
- `master_thesis/data.py`: ENTSO-E loading, outage aggregation, and gas download/loading.
- `master_thesis/features.py`: feature engineering and feature-set selection.
- `master_thesis/modeling.py`: train/test split, model training, spike metrics, and SHAP.
- `master_thesis/orchestration.py`: raw-cache reuse, rolling-origin backtest execution, and result summaries.
- `master_thesis/pipeline.py`: thin CLI entrypoint.

Evaluation:

- Validation uses four rolling-origin folds from 2024-07-01 through 2025-06-30.
- Final test remains untouched at 2025-07-01 through 2025-12-31.
- Raw fold-level metrics are saved to `metrics.csv`.
- Averaged validation tables are saved to `metrics_validation_summary.csv`.
- Selected presentation models are saved to `selected_models.csv`.
- Per-run success and failure status for `autoregressive`, `exogenous`, and `full` is saved to `run_status.csv`.
