"""Owner question after the momentum decision (2026-10-04): "so maybe it is better to simply buy QQQ?"

Descriptive numbers on series that already exist; no new candidate, nothing selected. The legs are those of the
registered decision pack (ndx_momentum_decision.py, registration ndx_momentum_decision_controls_20261004):
    L             the LIVE pod
    capsule       E2 + 40% sector cap
    QQQ gated     the registered control: QQQ held at L's own invested fraction of the day before, idle part at T-bills
    QQQ           buy and hold (total return)

1. The legs alone, full period and four sub-periods.
2. The live book (60% TAA 3x / 40% leg, monthly rebalance) with each leg; paired stationary-bootstrap
   P(book with L > book with QQQ gated); the book's 2022; daily correlation of each leg with TAA 3x.
3. Small-account friction: L and the capsule started on 2023-01-03 at USD 12K to 1M (whole shares, USD 1 minimum fee,
   the engine's cost model). The QQQ legs carry no fee here: one order a month, negligible at any size.

    uv run python scripts/research/scout_robustness_20261002/ndx_momentum_qqq_question.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import ndx_momentum_decision as decision
from alpha.scout.engines.weights import CostModel, simulate
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import performance_dict
from alpha.scout.stations.robustness import paired_sharpe_probability
from plans import PLAN_DICT
from run import PodRunner, cash_credited, market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_momentum_qqq_question.json"
RECENT_START_STR = "2023-01-01"
WINDOW_DICT = {"full 2000-09 to 2026-10": (decision.START_STR, None), "2000-09 to 2013-09": (decision.START_STR, "2013-09-30"),
               "2013-10 to 2021-09": ("2013-10-01", "2021-09-30"), "2021-10 on": ("2021-10-01", None), "2023-01 on": (RECENT_START_STR, None)}
POD_SIZE_TUPLE = (12_000, 25_000, 50_000, 100_000, 1_000_000)
METRIC_TUPLE = ("cagr_float", "volatility_float", "sharpe_float", "max_drawdown_float")


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    ndx_vxn.sector_group_dict([s for s in inputs.close_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR], 1)
    name_dict = {"L": decision.LIVE_STR, "capsule": decision.DESIGN_STR}
    weight_dict = {k: decision.averaged_weight_df(inputs, decision.FINALIST_DICT[n]) for k, n in name_dict.items()}
    run_dict = {k: decision.simulate_weight_df(inputs, tbill_ser, w) for k, w in weight_dict.items()}
    date_index = run_dict["L"]["daily"].index
    qqq_ser = decision.etf_return_df(inputs.close_df.index, tbill_ser)["QQQ"].reindex(date_index)
    tbill_day_ser = tbill_ser.reindex(date_index).fillna(0.0)
    invested_ser = run_dict["L"]["invested"].shift(1).reindex(date_index).fillna(0.0).clip(0.0, 1.0)
    leg_dict = {"L": run_dict["L"]["daily"], "capsule": run_dict["capsule"]["daily"],
                "QQQ gated": invested_ser * qqq_ser + (1.0 - invested_ser) * tbill_day_ser, "QQQ": qqq_ser}
    out = {"legs": {}, "book": {}, "friction": {}, "invested_mean": float(invested_ser.mean())}

    for leg_str, leg_ser in leg_dict.items():
        out["legs"][leg_str] = {}
        for window_str, (a, b) in WINDOW_DICT.items():
            perf = performance_dict(leg_ser.loc[a:b].dropna(), tbill_ser)
            out["legs"][leg_str][window_str] = {k: float(perf[k]) for k in METRIC_TUPLE}
        out["legs"][leg_str]["year_2022"] = float((1.0 + leg_ser.loc["2022"]).prod() - 1.0)

    taa_ser = PodRunner(PLAN_DICT["taa_3x"], tbill_ser=tbill_ser).daily({}).loc[PLAN_DICT["taa_3x"].eval_start_str:]
    proxy_ser = pd.read_csv(decision.PROXY_PATH, index_col=0, parse_dates=True)["taa_btal_tqqq"].dropna()
    long_taa_ser = pd.concat([proxy_ser.loc[proxy_ser.index < taa_ser.index[0]], taa_ser])
    for label_str, taa_leg_ser in (("2012-11 on (real TAA)", taa_ser), ("2008-03 on (TAA proxy before 2012-11)", long_taa_ser)):
        book_dict = {k: decision.book_ser(pd.DataFrame({"taa": taa_leg_ser, "ndx": s}), decision.BOOK_WEIGHT_DICT) for k, s in leg_dict.items()}
        row = {k: {**{m: float(performance_dict(b, tbill_ser)[m]) for m in METRIC_TUPLE}, "year_2022": float((1.0 + b.loc["2022"]).prod() - 1.0)}
               for k, b in book_dict.items()}
        row["p_l_book_gt_gated_qqq_book"] = paired_sharpe_probability(book_dict["L"], book_dict["QQQ gated"])
        row["p_capsule_book_gt_gated_qqq_book"] = paired_sharpe_probability(book_dict["capsule"], book_dict["QQQ gated"])
        row["p_l_book_gt_qqq_book"] = paired_sharpe_probability(book_dict["L"], book_dict["QQQ"])
        row["corr_with_taa_daily"] = {k: float(s.corr(taa_leg_ser)) for k, s in leg_dict.items()}
        out["book"][label_str] = row

    for key_str, weight_df in weight_dict.items():
        column_list = list(weight_df.columns[(weight_df != 0).any(axis=0)])
        recent_weight_df = weight_df.loc[RECENT_START_STR:, column_list]
        out["friction"][key_str] = {}
        for size_int in POD_SIZE_TUPLE:
            result = simulate(inputs.open_df[column_list], inputs.close_df[column_list], inputs.dividend_df[column_list], recent_weight_df,
                              start_date=RECENT_START_STR, capital_float=float(size_int), share_unit_mode_str="historical",
                              unadjusted_close_df=inputs.raw_close_df[column_list], cost_model=CostModel())
            perf = performance_dict(cash_credited(result, inputs.close_df, tbill_ser), tbill_ser)
            out["friction"][key_str][str(size_int)] = {"cagr_float": float(perf["cagr_float"]), "sharpe_float": float(perf["sharpe_float"]),
                                                       "order_count_int": int(len(result.trade_df)), "fee_float": float(result.trade_df["fee_float"].sum()),
                                                       "end_value_float": float(result.total_value_ser.iloc[-1])}
    out["friction"]["QQQ gated, 2023-01 on, cagr_float"] = out["legs"]["QQQ gated"]["2023-01 on"]["cagr_float"]
    OUTPUT_PATH.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{x:6.1%}"

    print("legs (CAGR / Vol / Sharpe / Max DD), then 2022")
    for leg_str, row in out["legs"].items():
        for window_str in WINDOW_DICT:
            v = row[window_str]
            print(f"  {leg_str:10s} {window_str:24s} {pct(v['cagr_float'])} {pct(v['volatility_float'])} {v['sharpe_float']:5.2f} {pct(v['max_drawdown_float'])}")
        print(f"  {leg_str:10s} 2022 {pct(row['year_2022'])}")
    for label_str, row in out["book"].items():
        print("book", label_str)
        for key_str in leg_dict:
            v = row[key_str]
            print(f"  leg = {key_str:10s} CAGR {pct(v['cagr_float'])} Vol {pct(v['volatility_float'])} Sharpe {v['sharpe_float']:.2f} Max DD {pct(v['max_drawdown_float'])} 2022 {pct(v['year_2022'])}")
        print(f"  P(L book > gated QQQ book) {row['p_l_book_gt_gated_qqq_book']:.2f} | P(capsule book > gated QQQ book) {row['p_capsule_book_gt_gated_qqq_book']:.2f} "
              f"| P(L book > QQQ book) {row['p_l_book_gt_qqq_book']:.2f} | corr with TAA {({k: round(v, 2) for k, v in row['corr_with_taa_daily'].items()})}")
    print("started 2023-01-03 at a small pod (CAGR / Sharpe / orders / fees)")
    for key_str in weight_dict:
        for size_str, v in out["friction"][key_str].items():
            print(f"  {key_str:8s} ${int(size_str):>9,} {pct(v['cagr_float'])} {v['sharpe_float']:.2f} {v['order_count_int']} ${v['fee_float']:,.0f}")
    print(f"  QQQ gated, 2023-01 on: {pct(out['friction']['QQQ gated, 2023-01 on, cagr_float'])}")


if __name__ == "__main__":
    main()
