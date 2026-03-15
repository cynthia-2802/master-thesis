from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestFold:
    """One expanding-window validation fold."""

    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class Config:
    """Runtime configuration for data retrieval, feature engineering, and evaluation."""

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
        return self.output_dir / f"raw_{self.study_start[:4]}_{int(self.study_end[:4]) - 1}.parquet"

    @property
    def default_gas_path(self) -> Path:
        return self.output_dir / "ttf_gas_price.parquet"

    @property
    def rolling_folds(self) -> tuple[BacktestFold, ...]:
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
