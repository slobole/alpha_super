"""ALFRED real-time vintage fetcher for DGS10, DGS3MO, DAAA, DBAA (cached; polite batching).

One request returns several vintages of the same series:
  https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=S,S,...&vintage_date=d1,d2,...
Column SERIES_YYYYMMDD == requested vintage date -> valid real-time vintage.  When the requested date precedes the
first ALFRED vintage, ALFRED silently returns another vintage (label differs) or an empty body -> marked invalid.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from def_common import CACHE

ALFRED_DIR = CACHE / "alfred"
ALFRED_DIR.mkdir(parents=True, exist_ok=True)
BASE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def _fetch(series: str, dates: list[pd.Timestamp]) -> pd.DataFrame | None:
    ids = ",".join([series] * len(dates))
    vds = ",".join(d.strftime("%Y-%m-%d") for d in dates)
    url = f"{BASE}?id={ids}&vintage_date={vds}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                body = resp.read().decode("utf-8")
            if not body.strip():
                return None
            return pd.read_csv(io.StringIO(body))
        except Exception as exc:  # network hiccup: back off
            print("retry", series, dates[0].date(), exc)
            time.sleep(5 * (attempt + 1))
    return None


def vintage_panel(series: str, vintage_dates, batch: int = 10) -> dict[pd.Timestamp, pd.Series]:
    """Return {vintage_date: observation series (float, NaNs dropped)} for valid vintages only."""
    out: dict[pd.Timestamp, pd.Series] = {}
    vintage_dates = sorted({pd.Timestamp(d).normalize() for d in vintage_dates})

    def handle(chunk):
        tag = f"{series}_{chunk[0]:%Y%m%d}_{chunk[-1]:%Y%m%d}_{len(chunk)}"
        path = ALFRED_DIR / f"{tag}.pkl"
        if path.exists():
            df = pd.read_pickle(path)
        else:
            df = _fetch(series, chunk)
            if df is None:
                df = pd.DataFrame()
            df.to_pickle(path)
            time.sleep(1.0)
        if df.empty:
            # a pre-first-vintage date can blank the whole multi-vintage response: split
            if len(chunk) > 1:
                mid = len(chunk) // 2
                handle(chunk[:mid])
                handle(chunk[mid:])
            return
        df = df.set_index("observation_date")
        df.index = pd.to_datetime(df.index)
        for d in chunk:
            col = f"{series}_{d:%Y%m%d}"
            if col in df.columns:
                ser = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
                ser.name = series
                out[d] = ser

    for i in range(0, len(vintage_dates), batch):
        handle(vintage_dates[i:i + batch])
    return out


def _fetch_windows(series: str, dates: list[pd.Timestamp], window_days: int) -> pd.DataFrame | None:
    ids = ",".join([series] * len(dates))
    vds = ",".join(d.strftime("%Y-%m-%d") for d in dates)
    cosd = ",".join((d - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d") for d in dates)
    coed = ",".join(d.strftime("%Y-%m-%d") for d in dates)
    url = f"{BASE}?id={ids}&vintage_date={vds}&cosd={cosd}&coed={coed}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                body = resp.read().decode("utf-8")
            if not body.strip():
                return None
            return pd.read_csv(io.StringIO(body))
        except Exception as exc:
            print("retry", series, dates[0].date(), exc, flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def window_panel(series: str, vintage_dates, window_days: int = 45, batch: int = 10) -> dict[pd.Timestamp, pd.Series]:
    """{vintage_date: observations in (vintage_date - window_days, vintage_date] as known on vintage_date}."""
    out: dict[pd.Timestamp, pd.Series] = {}
    vintage_dates = sorted({pd.Timestamp(d).normalize() for d in vintage_dates})
    for i in range(0, len(vintage_dates), batch):
        chunk = vintage_dates[i:i + batch]
        path = ALFRED_DIR / f"win{window_days}_{series}_{chunk[0]:%Y%m%d}_{chunk[-1]:%Y%m%d}_{len(chunk)}.pkl"
        if path.exists():
            df = pd.read_pickle(path)
        else:
            df = _fetch_windows(series, chunk, window_days)
            df = pd.DataFrame() if df is None else df
            df.to_pickle(path)
            time.sleep(0.5)
        if df.empty:
            continue
        df = df.set_index("observation_date")
        df.index = pd.to_datetime(df.index)
        for d in chunk:
            col = f"{series}_{d:%Y%m%d}"
            if col in df.columns:
                ser = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
                ser = ser[(ser.index > d - pd.Timedelta(days=window_days)) & (ser.index <= d)]
                ser.name = series
                out[d] = ser
    return out


def full_vintage(series: str, vintage_date) -> pd.Series | None:
    d = pd.Timestamp(vintage_date).normalize()
    path = ALFRED_DIR / f"full_{series}_{d:%Y%m%d}.pkl"
    if path.exists():
        df = pd.read_pickle(path)
    else:
        df = _fetch(series, [d])
        df = pd.DataFrame() if df is None else df
        df.to_pickle(path)
        time.sleep(0.5)
    col = f"{series}_{d:%Y%m%d}"
    if df.empty or col not in df.columns:
        return None
    ser = pd.to_numeric(df.set_index("observation_date")[col], errors="coerce").dropna().astype(float)
    ser.index = pd.to_datetime(ser.index)
    ser.name = series
    return ser
