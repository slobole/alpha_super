"""Build the cached NDX data (trimmed + untrimmed universe) and verify both modules' loaders return the same data.

Usage: uv run python scripts/research/leakage_hunt_20260927/ndx_build_cache.py
"""

from __future__ import annotations

import json
import pickle

import pandas as pd

import ndx_common as nc


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1))
        return True
    except AssertionError as exc:
        print("DIFF:", str(exc)[:500])
        return False


def main() -> None:
    report = {}
    trimmed = nc.load_data("trimmed")  # natr_vxn loader
    atr_data = nc.build_data("trimmed", key="atr_vxn")
    report["loaders_identical"] = {
        name: frames_equal(trimmed[name], atr_data[name]) for name in ("pricing", "universe", "schedule", "vxn")
    }
    untrimmed = nc.load_data("untrimmed")
    pr = trimmed["pricing"]
    report["trimmed"] = {"pricing_shape": list(pr.shape), "universe_shape": list(trimmed["universe"].shape),
                         "first_date": str(pr.index[0].date()), "last_date": str(pr.index[-1].date()),
                         "n_symbols": int(len(trimmed["universe"].columns)),
                         "has_unadjusted_close_all": bool(all((s, "Unadjusted Close") in pr.columns
                                                              for s in trimmed["universe"].columns)),
                         "vxn_first": str(trimmed["vxn"].index[0].date()), "vxn_last": str(trimmed["vxn"].index[-1].date())}
    report["untrimmed"] = {"pricing_shape": list(untrimmed["pricing"].shape),
                           "universe_shape": list(untrimmed["universe"].shape),
                           "n_symbols": int(len(untrimmed["universe"].columns))}
    t_u, u_u = trimmed["universe"], untrimmed["universe"]
    common_cols = t_u.columns.intersection(u_u.columns)
    diff = (u_u[common_cols].reindex(t_u.index).fillna(0) - t_u[common_cols]).abs()
    report["membership_rows_restored"] = int(diff.to_numpy().sum())
    report["symbols_with_restored_rows"] = int((diff.sum() > 0).sum())
    report["symbols_only_untrimmed"] = sorted(set(u_u.columns) - set(t_u.columns))[:50]
    (nc.OUT / "cache_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
