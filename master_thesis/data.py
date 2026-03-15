from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import yfinance as yf
from entsoe.entsoe import EntsoePandasClient

from .config import Config
from .utils import sanitize_feature_name


class EntsoeDataLoader:
    """Loads ENTSO-E data used by the forecasting pipeline."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = EntsoePandasClient(api_key=cfg.api_key)
        self.start_utc, self.end_utc = self._time_window()

    def _time_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(self.cfg.study_start, tz=self.cfg.local_tz).tz_convert(self.cfg.entsoe_tz)
        end = pd.Timestamp(self.cfg.study_end, tz=self.cfg.local_tz).tz_convert(self.cfg.entsoe_tz)
        if end <= start:
            raise ValueError("study_end must be after study_start.")
        return start, end

    def _iter_chunks(self, months: int) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
        chunk_start = self.start_utc
        while chunk_start < self.end_utc:
            chunk_end = min(chunk_start + pd.DateOffset(months=months), self.end_utc)
            yield chunk_start, chunk_end
            chunk_start = chunk_end

    @staticmethod
    def _as_series(obj: object, series_name: str) -> pd.Series:
        if isinstance(obj, pd.DataFrame):
            if obj.shape[1] != 1:
                raise ValueError(f"Expected a single column for '{series_name}', got {list(obj.columns)}")
            series = obj.iloc[:, 0].copy()
            series.name = series_name
            return series
        if isinstance(obj, pd.Series):
            series = obj.copy()
            series.name = series_name
            return series
        raise TypeError(f"Expected pandas Series or 1-col DataFrame for '{series_name}', got {type(obj)}")

    @staticmethod
    def _as_frame(obj: object, prefix: str, suffix: str) -> pd.DataFrame:
        if isinstance(obj, pd.Series):
            frame = obj.to_frame()
        elif isinstance(obj, pd.DataFrame):
            frame = obj.copy()
        else:
            raise TypeError(f"Expected pandas Series or DataFrame for '{prefix}', got {type(obj)}")
        frame.columns = [sanitize_feature_name(f"{prefix}{column}{suffix}") for column in frame.columns]
        return frame

    @staticmethod
    def _concat_chunks(chunks: Iterable[pd.DataFrame | pd.Series]) -> pd.DataFrame | pd.Series:
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks).sort_index()
        return combined[~combined.index.duplicated(keep="last")]

    def _query_series_in_chunks(
        self,
        query_fn: Callable[[pd.Timestamp, pd.Timestamp], object],
        series_name: str,
        months: int | None = None,
    ) -> pd.Series:
        months = months or self.cfg.fetch_chunk_months
        chunks: list[pd.Series] = []
        for start, end in self._iter_chunks(months):
            result = query_fn(start, end)
            if result is not None:
                chunks.append(self._as_series(result, series_name))

        combined = self._concat_chunks(chunks)
        if isinstance(combined, pd.DataFrame):
            raise TypeError(f"Expected Series for '{series_name}', got DataFrame.")
        return combined

    def _query_frame_in_chunks(
        self,
        query_fn: Callable[[pd.Timestamp, pd.Timestamp], object],
        prefix: str,
        suffix: str,
        months: int | None = None,
    ) -> pd.DataFrame:
        months = months or self.cfg.fetch_chunk_months
        chunks: list[pd.DataFrame] = []
        for start, end in self._iter_chunks(months):
            result = query_fn(start, end)
            if result is not None:
                chunks.append(self._as_frame(result, prefix=prefix, suffix=suffix))

        combined = self._concat_chunks(chunks)
        if isinstance(combined, pd.Series):
            return combined.to_frame()
        return combined

    def fetch_zone_data(self, zone: str) -> pd.DataFrame:
        """Fetches all available raw inputs for one bidding zone.

        Args:
            zone: Norwegian bidding zone code.

        Returns:
            pd.DataFrame: Time-indexed raw data for the zone.
        """
        pieces: list[pd.DataFrame] = []
        pieces.append(
            self._query_series_in_chunks(
                lambda start, end: self.client.query_day_ahead_prices(zone, start=start, end=end),
                "price_eur_mwh",
            ).to_frame()
        )

        optional_series = (
            ("load_mw", self.client.query_load),
            ("load_forecast_mw", self.client.query_load_forecast),
        )
        for column_name, query_fn in optional_series:
            try:
                pieces.append(
                    self._query_series_in_chunks(
                        lambda start, end, fn=query_fn: fn(zone, start=start, end=end),
                        column_name,
                    ).to_frame()
                )
            except Exception:
                pass

        optional_frames = (
            ("gen_", "_mw", self.client.query_generation),
            ("ws_forecast_", "_mw", self.client.query_wind_and_solar_forecast),
        )
        for prefix, suffix, query_fn in optional_frames:
            try:
                frame = self._query_frame_in_chunks(
                    lambda start, end, fn=query_fn: fn(zone, start=start, end=end),
                    prefix=prefix,
                    suffix=suffix,
                )
                if not frame.empty:
                    pieces.append(frame)
            except Exception:
                pass

        try:
            outage_frame = self.fetch_outages(zone)
            if not outage_frame.empty:
                pieces.append(outage_frame)
        except Exception:
            pass

        merged = pd.concat(pieces, axis=1).sort_index()
        merged.columns = [sanitize_feature_name(column) for column in merged.columns]
        return merged

    def fetch_outages(self, zone: str) -> pd.DataFrame:
        """Fetches and aggregates outage data for one zone.

        Args:
            zone: Norwegian bidding zone code.

        Returns:
            pd.DataFrame: Hourly outage features.
        """
        outage_chunks: list[pd.DataFrame] = []
        for start, end in self._iter_chunks(self.cfg.unavailability_chunk_months):
            outages = self.client.query_unavailability_of_generation_units(zone, start=start, end=end)
            if outages is not None and not outages.empty:
                outage_chunks.append(outages)

        if not outage_chunks:
            return pd.DataFrame()

        outages = pd.concat(outage_chunks, axis=0, ignore_index=False)
        outages = outages.drop_duplicates(subset=["mrid", "revision", "start", "end"], keep="last")
        return self._aggregate_outages_to_hourly(outages)

    def _aggregate_outages_to_hourly(self, outages: pd.DataFrame) -> pd.DataFrame:
        """Aggregates unit-level outage records to hourly features.

        Args:
            outages: Raw outage records from ENTSO-E.

        Returns:
            pd.DataFrame: Hourly outage magnitude and event counts.
        """
        local_index = pd.date_range(
            self.start_utc.tz_convert(self.cfg.local_tz),
            self.end_utc.tz_convert(self.cfg.local_tz) - pd.Timedelta(hours=1),
            freq=self.cfg.frequency,
            tz=self.cfg.local_tz,
        )
        hourly = pd.DataFrame(
            {"outage_unavailable_mw": 0.0, "outage_event_count": 0.0},
            index=local_index,
        )

        outage_data = outages.copy()
        outage_data["start"] = pd.to_datetime(outage_data["start"], utc=True, errors="coerce").dt.tz_convert(
            self.cfg.local_tz
        )
        outage_data["end"] = pd.to_datetime(outage_data["end"], utc=True, errors="coerce").dt.tz_convert(
            self.cfg.local_tz
        )
        outage_data["nominal_power"] = pd.to_numeric(outage_data["nominal_power"], errors="coerce").fillna(0.0)
        outage_data["avail_qty"] = pd.to_numeric(outage_data["avail_qty"], errors="coerce").fillna(0.0)
        outage_data["unavailable_mw"] = (outage_data["nominal_power"] - outage_data["avail_qty"]).clip(lower=0.0)
        outage_data = outage_data.dropna(subset=["start", "end"])
        outage_data = outage_data[outage_data["end"] > outage_data["start"]]

        outage_records = outage_data[["start", "end", "unavailable_mw"]].to_dict("records")
        for outage in outage_records:
            start_ts = pd.Timestamp(outage["start"])
            end_ts = pd.Timestamp(outage["end"])
            hour_start = start_ts.floor(self.cfg.frequency)
            hour_end = (end_ts - pd.Timedelta(microseconds=1)).floor(self.cfg.frequency)
            active_hours = pd.date_range(hour_start, hour_end, freq=self.cfg.frequency, tz=self.cfg.local_tz)
            active_hours = active_hours.intersection(pd.DatetimeIndex(hourly.index))
            if active_hours.empty:
                continue
            hourly.loc[active_hours, "outage_unavailable_mw"] += float(outage["unavailable_mw"])
            hourly.loc[active_hours, "outage_event_count"] += 1.0

        return hourly

    def fetch_crossborder_flows(self, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
        """Fetches hourly cross-border flows for selected area pairs.

        Args:
            pairs: Iterable of `(from_area, to_area)` pairs.

        Returns:
            pd.DataFrame: Hourly flow series for available pairs.
        """
        flow_series: list[pd.Series] = []
        for from_area, to_area in pairs:
            column_name = sanitize_feature_name(f"flow_{from_area}_to_{to_area}_mw")
            try:
                series = self._query_series_in_chunks(
                    lambda start, end, a=from_area, b=to_area: self.client.query_crossborder_flows(
                        a, b, start=start, end=end
                    ),
                    column_name,
                )
                flow_series.append(series)
            except Exception:
                continue

        if not flow_series:
            return pd.DataFrame()
        return pd.concat(flow_series, axis=1).sort_index()


class ExternalMarketDataLoader:
    """Loads or downloads external market controls."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def load_gas_prices(self) -> pd.DataFrame:
        """Loads gas prices from parquet and aligns them to the study window.

        Returns:
            pd.DataFrame: Hourly gas price series.

        Raises:
            ValueError: If the gas file is invalid or missing required columns.
        """
        path = self.cfg.gas_data_path or self.cfg.default_gas_path
        if not path.exists():
            self.download_gas_prices(path)
        if path.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError(f"Gas price file must be a parquet file: {path}")

        frame = pd.read_parquet(path)
        required = {self.cfg.gas_datetime_column, self.cfg.gas_price_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing gas columns: {sorted(missing)}")
        gas = frame[[self.cfg.gas_datetime_column, self.cfg.gas_price_column]].copy()
        gas[self.cfg.gas_datetime_column] = pd.to_datetime(gas[self.cfg.gas_datetime_column], errors="coerce")
        gas[self.cfg.gas_price_column] = pd.to_numeric(gas[self.cfg.gas_price_column], errors="coerce")
        gas = gas.dropna(subset=[self.cfg.gas_datetime_column, self.cfg.gas_price_column])

        gas = gas.set_index(self.cfg.gas_datetime_column).sort_index()
        gas_index = pd.DatetimeIndex(gas.index)
        if gas_index.tz is None:
            gas.index = gas_index.tz_localize(self.cfg.local_tz)
        else:
            gas.index = gas_index.tz_convert(self.cfg.local_tz)

        gas = gas.resample(self.cfg.frequency).ffill()
        gas = gas.loc[
            pd.Timestamp(self.cfg.study_start, tz=self.cfg.local_tz) : pd.Timestamp(
                self.cfg.study_end, tz=self.cfg.local_tz
            )
            - pd.Timedelta(hours=1)
        ]
        gas.columns = [sanitize_feature_name(column) for column in gas.columns]
        return gas

    def download_gas_prices(self, output_path: Path) -> Path:
        """Downloads TTF gas prices and stores them as parquet.

        Args:
            output_path: Destination parquet path.

        Returns:
            Path: Saved parquet path.

        Raises:
            ValueError: If Yahoo Finance returns no gas data.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gas = yf.download("TTF=F", start=self.cfg.study_start[:10], end=self.cfg.study_end[:10], progress=False)
        if gas is None or gas.empty:
            raise ValueError("Failed to fetch gas price data from Yahoo Finance.")

        gas = gas[["Close"]].reset_index()
        gas.columns = [self.cfg.gas_datetime_column, self.cfg.gas_price_column]
        gas.to_parquet(output_path)
        return output_path
