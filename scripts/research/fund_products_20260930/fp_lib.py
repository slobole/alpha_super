"""Fund products study (SPEC_FROZEN.md): families, family blocks, grid mixes, path metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GA_DIR = HERE.parent / "growth_aggressive_20260930"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

import ga_lib as ga  # noqa: E402
from ga_lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book, lib  # noqa: E402,F401

STUDY = ga.WT_REPO / "results" / "research" / "portfolio" / "fund_products_20260930"
SEED0 = 20260929
FAMILIES = {
    "TAA": ["taa3x", "taa3x_1n", "taa2x_1n", "taa_1n_qld", "taa_1n_sso"],
    "NDX": ["ndx_vxn", "ndx_atr", "ndx_natr20"],
    "SMR": ["dv2", "dv2_adv", "dv2_floor", "hpi_vote", "hpi_ibs_rsi"],
    "EMR": ["etf_dv2", "downshock", "disp"],
    "DEF": ["core5", "btal_qqq", "trinity"],
    "EOM": ["eom_flow"],
}
FAMILY_LABEL = {"TAA": "רוטציה טקטית (הגנה קודם)", "NDX": "מומנטום נאסד״ק 100", "SMR": "היפוך לממוצע במניות",
                "EMR": "היפוך לממוצע בקרנות סל", "DEF": "מאקרו הגנתי", "EOM": "זרימות סוף חודש", "CASH": "אג״ח קצר (BIL)"}
DAILY = {"dv2", "dv2_adv", "dv2_floor", "hpi_vote", "hpi_ibs_rsi", "etf_dv2", "downshock", "disp", "trinity"}
NOT_LIVE_TRADABLE = {"eom_flow"}
LINES = {"LOW-TOUCH": ["TAA", "NDX", "DEF", "CASH"],
         "MAIN": ["TAA", "NDX", "SMR", "EMR", "DEF", "CASH"],
         "TARGET": ["TAA", "NDX", "SMR", "EMR", "DEF", "EOM", "CASH"]}
GRID_STEP = {"LOW-TOUCH": 0.05, "MAIN": 0.10, "TARGET": 0.10}
TAA_CAP = 0.70


def ledger(event_str: str, **fields) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    rec = {"event_str": event_str, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
           "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (STUDY / "experiment_ledger.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec, default=str) + "\n")


def tier(alias: str, meta: dict) -> str:
    return "cash" if alias == TBILL else meta[alias]["tier_str"]


def grid(n_blocks: int, step: float) -> np.ndarray:
    """All weight vectors on the simplex with the given step (n_blocks columns)."""
    k = int(round(1 / step))
    out = []
    for combo in combinations_with_replacement(range(n_blocks), k):
        w = np.bincount(combo, minlength=n_blocks) / k
        out.append(w)
    return np.unique(np.array(out), axis=0)


def mix_metrics(R: np.ndarray, W: np.ndarray, rf: np.ndarray, chunk: int = 800) -> dict:
    """Constant-daily-weight mixes (N x B returns, M x B weights): CAGR (252/yr), excess Sharpe over rf (A1), max DD."""
    n = R.shape[0]
    cagr, sharpe, dd = np.empty(len(W)), np.empty(len(W)), np.empty(len(W))
    for a in range(0, len(W), chunk):
        w = W[a:a + chunk]
        r = R @ w.T
        nav = np.cumprod(1.0 + r, axis=0)
        cagr[a:a + chunk] = nav[-1] ** (252.0 / n) - 1.0
        x = r - rf[:, None]
        sd = x.std(axis=0, ddof=1)
        sharpe[a:a + chunk] = np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
        peak = np.maximum.accumulate(np.vstack([np.ones((1, r.shape[1])), nav]), axis=0)
        dd[a:a + chunk] = (np.vstack([np.ones((1, r.shape[1])), nav]) / peak - 1.0).min(axis=0)
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd}
