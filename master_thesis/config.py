from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestFold:
    """Defines one expanding-window validation fold.

    Attributes:
        name: Fold identifier used in saved outputs.
        train_start: Inclusive start of the training window.
        train_end: Exclusive end of the training window.
        validation_start: Inclusive start of the validation window.
        validation_end: Exclusive end of the validation window.
    """

    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class Config:
    """Stores runtime settings for the thesis pipeline.

    Attributes:
        api_key: ENTSO-E API key.
        output_dir: Directory for cached data and model outputs.
        entsoe_tz: Time zone used by ENTSO-E requests.
        local_tz: Local analysis time zone.
        study_start: Inclusive start of the raw data study window.
        study_end: Exclusive end of the raw data study window.
        test_start: Inclusive start of the final test block.
        test_end: Exclusive end of the final test block.
        frequency: Base data frequency.
        fetch_chunk_months: Chunk size for general ENTSO-E requests.
        unavailability_chunk_months: Chunk size for outage requests.
        spike_quantile: Train-sample quantile used to define spikes.
        random_state: Random seed for model training and sampling.
        min_train_rows: Minimum allowed train rows.
        min_validation_rows: Minimum allowed validation rows.
        min_test_rows: Minimum allowed test rows.
        shap_background_size: Maximum SHAP background sample size.
        gas_data_path: Optional explicit path to gas price data.
        gas_datetime_column: Timestamp column name in gas data.
        gas_price_column: Price column name in gas data.
    """

    api_key: str
    output_dir: Path = Path("./entsoe_no_experiment")
    entsoe_tz: str = "Europe/Brussels"
    local_tz: str = "Europe/Oslo"
    study_start: str = "2023-01-01 00:00"
    study_end: str = "2026-01-01 00:00"
    test_start: str = "2025-07-01 00:00"
    test_end: str = "2026-01-01 00:00"
    frequency: str = "h"
    fetch_chunk_months: int = 12
    unavailability_chunk_months: int = 6
    spike_quantile: float = 0.95
    random_state: int = 42
    min_train_rows: int = 24 * 60
    min_validation_rows: int = 24 * 7
    min_test_rows: int = 24 * 7
    shap_background_size: int = 2000
    gas_data_path: Path | None = None
    gas_datetime_column: str = "timestamp"
    gas_price_column: str = "gas_price_eur_mwh"

    @classmethod
    def from_env(cls) -> Config:
        """Builds a config instance from environment variables.

        Returns:
            Config: Parsed runtime configuration.

        Raises:
            ValueError: If the ENTSO-E API key is missing.
        """
        api_key = os.getenv("ENTSOE_API_KEY", "")
        if not api_key:
            raise ValueError("Missing ENTSOE_API_KEY in environment.")

        gas_path = os.getenv("GAS_PRICE_PATH")
        return cls(
            api_key=api_key,
            gas_data_path=Path(gas_path) if gas_path else None,
        )

    @property
    def raw_cache_path(self) -> Path:
        """Returns the parquet path used for cached raw data."""
        return self.output_dir / f"raw_{self.study_start[:4]}_{int(self.study_end[:4]) - 1}.parquet"

    @property
    def default_gas_path(self) -> Path:
        """Returns the default parquet path for gas price data."""
        return self.output_dir / "ttf_gas_price.parquet"

    @property
    def rolling_folds(self) -> tuple[BacktestFold, ...]:
        """Returns the configured rolling-origin validation folds."""
        return (
            BacktestFold(
                name="fold_1",
                train_start="2023-01-01 00:00",
                train_end="2024-07-01 00:00",
                validation_start="2024-07-01 00:00",
                validation_end="2024-10-01 00:00",
            ),
            BacktestFold(
                name="fold_2",
                train_start="2023-01-01 00:00",
                train_end="2024-10-01 00:00",
                validation_start="2024-10-01 00:00",
                validation_end="2025-01-01 00:00",
            ),
            BacktestFold(
                name="fold_3",
                train_start="2023-01-01 00:00",
                train_end="2025-01-01 00:00",
                validation_start="2025-01-01 00:00",
                validation_end="2025-04-01 00:00",
            ),
            BacktestFold(
                name="fold_4",
                train_start="2023-01-01 00:00",
                train_end="2025-04-01 00:00",
                validation_start="2025-04-01 00:00",
                validation_end="2025-07-01 00:00",
            ),
        )
