"""Live-vs-backtest tracking for Dashboard V3.

Operators check this every EOD: did today's return land inside or outside
the band the backtest told us to expect? The check is intentionally crude
(±2σ around the historical daily mean), but in practice this is the single
strongest signal that a strategy has stopped behaving like its backtest —
worth a flag before a small drift becomes a real loss.

Configure expected stats once via ``alpha/live/expected_pnl.yaml``:

    pod_dv2_caspersky_live:
      daily_mean_return_float: 0.00012
      daily_volatility_float:  0.0118
      band_sigma_float:        2.0   # optional, default 2.0
      sample_count_int:        252   # optional, for the operator's reference

Missing pod / missing file → no comparison shown (graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXPECTED_PNL_PATH_STR = "alpha/live/expected_pnl.yaml"
DEFAULT_BAND_SIGMA_FLOAT = 2.0


@dataclass
class ExpectedPnlEntry:
    pod_id_str: str
    daily_mean_return_float: float
    daily_volatility_float: float
    band_sigma_float: float = DEFAULT_BAND_SIGMA_FLOAT
    sample_count_int: int | None = None

    @property
    def lower_band_float(self) -> float:
        return self.daily_mean_return_float - self.band_sigma_float * self.daily_volatility_float

    @property
    def upper_band_float(self) -> float:
        return self.daily_mean_return_float + self.band_sigma_float * self.daily_volatility_float


@dataclass
class TrackingComparison:
    has_data_bool: bool
    actual_return_float: float | None
    expected_mean_float: float | None
    expected_volatility_float: float | None
    band_sigma_float: float
    lower_band_float: float | None
    upper_band_float: float | None
    is_outside_band_bool: bool
    severity_str: str
    summary_str: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_data_bool": self.has_data_bool,
            "actual_return_float": self.actual_return_float,
            "expected_mean_float": self.expected_mean_float,
            "expected_volatility_float": self.expected_volatility_float,
            "band_sigma_float": self.band_sigma_float,
            "lower_band_float": self.lower_band_float,
            "upper_band_float": self.upper_band_float,
            "is_outside_band_bool": self.is_outside_band_bool,
            "severity_str": self.severity_str,
            "summary_str": self.summary_str,
        }


def load_expected_pnl_map(
    expected_pnl_path_str: str = DEFAULT_EXPECTED_PNL_PATH_STR,
) -> dict[str, ExpectedPnlEntry]:
    expected_pnl_path_obj = Path(expected_pnl_path_str)
    if not expected_pnl_path_obj.exists():
        return {}
    raw_obj = yaml.safe_load(expected_pnl_path_obj.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_obj, dict):
        raise ValueError(f"{expected_pnl_path_str}: expected a top-level mapping.")
    entry_map_dict: dict[str, ExpectedPnlEntry] = {}
    for pod_id_str, raw_entry_obj in raw_obj.items():
        if not isinstance(raw_entry_obj, dict):
            continue
        try:
            mean_float = float(raw_entry_obj["daily_mean_return_float"])
            vol_float = float(raw_entry_obj["daily_volatility_float"])
        except (KeyError, TypeError, ValueError):
            continue
        band_sigma_float = float(raw_entry_obj.get("band_sigma_float", DEFAULT_BAND_SIGMA_FLOAT))
        sample_count_int_obj = raw_entry_obj.get("sample_count_int")
        sample_count_int: int | None
        try:
            sample_count_int = int(sample_count_int_obj) if sample_count_int_obj is not None else None
        except (TypeError, ValueError):
            sample_count_int = None
        entry_map_dict[str(pod_id_str)] = ExpectedPnlEntry(
            pod_id_str=str(pod_id_str),
            daily_mean_return_float=mean_float,
            daily_volatility_float=vol_float,
            band_sigma_float=band_sigma_float,
            sample_count_int=sample_count_int,
        )
    return entry_map_dict


def build_tracking_comparison(
    pod_id_str: str,
    actual_daily_return_float: float | None,
    expected_pnl_map_dict: dict[str, ExpectedPnlEntry],
) -> TrackingComparison:
    entry_obj = expected_pnl_map_dict.get(pod_id_str)
    if entry_obj is None or actual_daily_return_float is None:
        return TrackingComparison(
            has_data_bool=False,
            actual_return_float=actual_daily_return_float,
            expected_mean_float=entry_obj.daily_mean_return_float if entry_obj else None,
            expected_volatility_float=entry_obj.daily_volatility_float if entry_obj else None,
            band_sigma_float=entry_obj.band_sigma_float if entry_obj else DEFAULT_BAND_SIGMA_FLOAT,
            lower_band_float=entry_obj.lower_band_float if entry_obj else None,
            upper_band_float=entry_obj.upper_band_float if entry_obj else None,
            is_outside_band_bool=False,
            severity_str="gray",
            summary_str="no expected-pnl baseline configured",
        )
    is_outside_band_bool = (
        actual_daily_return_float < entry_obj.lower_band_float
        or actual_daily_return_float > entry_obj.upper_band_float
    )
    severity_str = "yellow" if is_outside_band_bool else "green"
    summary_str = _format_summary_str(actual_daily_return_float, entry_obj, is_outside_band_bool)
    return TrackingComparison(
        has_data_bool=True,
        actual_return_float=actual_daily_return_float,
        expected_mean_float=entry_obj.daily_mean_return_float,
        expected_volatility_float=entry_obj.daily_volatility_float,
        band_sigma_float=entry_obj.band_sigma_float,
        lower_band_float=entry_obj.lower_band_float,
        upper_band_float=entry_obj.upper_band_float,
        is_outside_band_bool=is_outside_band_bool,
        severity_str=severity_str,
        summary_str=summary_str,
    )


def _format_summary_str(
    actual_float: float,
    entry_obj: ExpectedPnlEntry,
    is_outside_band_bool: bool,
) -> str:
    actual_str = f"{actual_float * 100:+.2f}%"
    mean_str = f"{entry_obj.daily_mean_return_float * 100:+.2f}%"
    band_pct_str = f"{entry_obj.band_sigma_float * entry_obj.daily_volatility_float * 100:.2f}%"
    band_warning_str = " — outside band" if is_outside_band_bool else " — within band"
    return f"Today {actual_str} · Expected {mean_str} ±{band_pct_str}{band_warning_str}"
