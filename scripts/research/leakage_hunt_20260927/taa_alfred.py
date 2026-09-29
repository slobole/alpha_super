"""ALFRED real-time vintage fetcher (cached, polite) for the TAA macro-vintage study.

ALFRED graph CSV accepts several vintages in one request:
    https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=DTB3,DTB3&vintage_date=2020-03-31,2020-04-30
returning columns DTB3_20200331, DTB3_20200430 (the series as it was known on each vintage date).
Empirical check (2026-09-27): vintage 2020-03-31 ends at observation 2020-03-30, i.e. the H.15 value for date t
is first published on t+1 -> a month-end decision at Close_T cannot know DTB3_T.

Responses are cached as raw CSV in the scratchpad; batches of 12 vintages; 2 s pause between live requests.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

from taa_common import SCRATCH

ALFRED_DIR = SCRATCH / "alfred"
ALFRED_DIR.mkdir(parents=True, exist_ok=True)
BATCH = 12


class VintageNotAvailable(RuntimeError):
    """ALFRED returns 404 when any requested vintage_date predates the series' first archived vintage
    (T5YIE: first vintage between 2013-06-28 and 2014-01-31 in our probes)."""


def _fetch(series_id: str, vintage_list: list[pd.Timestamp]) -> pd.DataFrame:
    tag = f"{series_id}_{vintage_list[0]:%Y%m%d}_{vintage_list[-1]:%Y%m%d}_{len(vintage_list)}"
    path = ALFRED_DIR / f"{tag}.csv"
    if not path.exists():
        ids = ",".join([series_id] * len(vintage_list))
        vds = ",".join(f"{v:%Y-%m-%d}" for v in vintage_list)
        url = f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={ids}&vintage_date={vds}"
        req = urllib.request.Request(url, headers={"User-Agent": "alpha-super-leakage-audit/1.0 (research)"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    txt = r.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:  # 404 = no ALFRED vintage on/before a requested date
                if exc.code == 404:
                    raise VintageNotAvailable(url) from exc
                print("retry", attempt, exc, flush=True)
                time.sleep(10 * (attempt + 1))
            except Exception as exc:  # pragma: no cover - network
                print("retry", attempt, exc, flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"ALFRED fetch failed: {url}")
        path.write_text(txt, encoding="utf-8")
        time.sleep(2.0)
    df = pd.read_csv(path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    return df.set_index(df.columns[0])


# First archived ALFRED vintage found by probing (404 before it): T5YIE between 2014-01-24 and 2014-01-31.
FIRST_VINTAGE = {"T5YIE": pd.Timestamp("2014-01-31")}


def vintage_frame(series_id: str, vintage_dates) -> dict[pd.Timestamp, pd.Series]:
    """{vintage_date -> series as known on that date (numeric, NaN dropped)}; empty series if ALFRED has no
    vintage on/before that date."""
    vds = sorted(pd.Timestamp(v) for v in set(pd.DatetimeIndex(vintage_dates)))
    out: dict[pd.Timestamp, pd.Series] = {}
    first = FIRST_VINTAGE.get(series_id)
    if first is not None:
        for v in [v for v in vds if v < first]:
            out[v] = pd.Series(dtype=float, name=series_id)
        vds = [v for v in vds if v >= first]
    for i in range(0, len(vds), BATCH):
        chunk = vds[i:i + BATCH]
        try:
            frames = [(chunk, _fetch(series_id, chunk))]
        except VintageNotAvailable:
            frames = []
            for v in chunk:  # fall back to single-vintage requests; missing ones -> empty series
                try:
                    frames.append(([v], _fetch(series_id, [v])))
                except VintageNotAvailable:
                    frames.append(([v], pd.DataFrame()))
                    time.sleep(1.0)
        for sub, df in frames:
          for v in sub:
            col = f"{series_id}_{v:%Y%m%d}"
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
            else:
                s = pd.Series(dtype=float)
            s.index = pd.DatetimeIndex(s.index)
            s.name = series_id
            out[v] = s.sort_index()
    return out
