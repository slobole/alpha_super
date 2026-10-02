"""A14 robustness diagnostics: contribution concentration, market-timing convexity, component ablation and the
random-parameter percentile (alpha/scout/stations/robustness.py), on the plans fixed in plans.py.

    uv run python scripts/research/scout_robustness_20261002/run.py            # the five planned pods, then the rest
    uv run python scripts/research/scout_robustness_20261002/run.py taa_3x     # one planned pod
    uv run python scripts/research/scout_robustness_20261002/run.py --others   # contribution + timing on the others

Writes results/scout/robustness/<pod>/robustness.pkl and summary.json (main checkout), adds the section to the pod's
re-audition bundle and re-renders its card (results/scout/cards/<pod>_reaudition*.html).
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from alpha.scout.card import render_card
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import sharpe_float, tbill_daily_ser
from alpha.scout.stations.robustness import (
    RandomParameterResult,
    ablation_dict,
    contribution_dict,
    random_parameter_summary,
    timing_dict,
)
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from plans import DRAW_COUNT_INT, PLAN_DICT, SEED_INT, RobustPlan

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness"
REAUDITION_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition"
CARD_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "cards"
WORKER_COUNT_INT = 8


def pod_dir_str(name_str: str) -> str:
    return name_str.replace(" ", "_").replace("/", "-")


# ---------------------------------------------------------------- building a pod
class PodRunner:
    """Runs the live rule of a planned pod with spec fields overridden. Inputs are loaded once per traded universe
    (a TAA ablation can change the defensive list or the fallback); `live_inputs` seeds the live universe's inputs
    (the random-draw workers receive them from the parent instead of reading Norgate eight times at once)."""

    def __init__(self, plan: RobustPlan, live_inputs=None, tbill_ser: pd.Series | None = None):
        self.plan, self.family_dict, self.inputs_dict, self.tbill_ser = plan, {}, {}, tbill_ser
        self.result_cache_dict, self.cache_bool = {}, live_inputs is None  # workers get live_inputs: no cache there
        if live_inputs is not None:
            self.inputs_dict[self._key({})] = live_inputs

    def _key(self, override_dict: dict) -> tuple:
        if self.plan.kind_str != "taa":
            return ("live",)
        from alpha.scout.specs import taa_3x

        config = dataclasses.replace(taa_3x.VARIANT_DICT[self.plan.variant_str].config, **override_dict)
        return (tuple(config.defensive_tuple), config.fallback_str)

    def inputs(self, override_dict: dict):
        from alpha.scout.specs import core5, ndx_vxn, taa_3x

        key_tuple = self._key(override_dict)
        if key_tuple not in self.inputs_dict:
            if self.plan.kind_str == "taa":
                config = dataclasses.replace(taa_3x.VARIANT_DICT[self.plan.variant_str].config, **override_dict)
                self.inputs_dict[key_tuple] = taa_3x.load_inputs(config=config)
            elif self.plan.kind_str == "ndx":
                self.inputs_dict[key_tuple] = ndx_vxn.load_inputs()
            else:
                self.inputs_dict[key_tuple] = core5.load_inputs()
        return self.inputs_dict[key_tuple]

    def _family(self, override_dict: dict):
        from alpha.scout import family as f

        key_tuple = self._key(override_dict)
        if key_tuple not in self.family_dict:
            inputs, plan = self.inputs(override_dict), self.plan
            if plan.kind_str == "taa":
                self.family_dict[key_tuple] = f.taa_variant_family(plan.variant_str, inputs)
            elif plan.kind_str == "ndx":
                self.family_dict[key_tuple] = {"ndx_vxn": f.ndx_vxn_family, "ndx_natr20_vxn": f.ndx_natr20_vxn_family}[plan.variant_str](inputs)
            else:
                self.family_dict[key_tuple] = f.core5_family(inputs)
        return self.family_dict[key_tuple]

    def result(self, override_dict: dict):
        """The engine run; cached in the parent (ablation reads each run twice: T-bill cash and 0% cash), never in a
        random-draw worker (each draw is new, and a stock pod's result is tens of MB)."""
        key_str = repr(sorted(override_dict.items()))
        if key_str in self.result_cache_dict:
            return self.result_cache_dict[key_str]
        family = self._family(override_dict)
        result = family.run_config({**family.live_config_dict, **override_dict})
        if self.cache_bool:
            self.result_cache_dict[key_str] = result
        return result

    def daily(self, override_dict: dict, result=None) -> pd.Series:
        """Net daily returns with idle cash credited at the T-bill rate (cash realism; the engine pays 0%):
            r'(t) = r(t) + idle(t-1) x tbill(t),  idle = (V - long market value) / V, clipped to [0, 1]
        Short proceeds are not idle (excluded); BIL held as an asset already earns its own yield. A return-level
        credit (the extra interest is not re-invested into later sizing): a small first-order approximation."""
        result = result if result is not None else self.result(override_dict)
        if self.tbill_ser is None:
            raise ValueError("The cash credit needs the T-bill series.")
        position_df = result.daily_position_df
        close_df = self.inputs(override_dict).close_df.reindex(index=position_df.index, columns=position_df.columns)
        long_value_ser = (position_df * close_df).clip(lower=0.0).sum(axis=1, min_count=0)
        idle_ser = (1.0 - long_value_ser / result.total_value_ser).clip(0.0, 1.0)
        tbill_ser = self.tbill_ser.reindex(result.daily_return_ser.index).fillna(0.0)
        return result.daily_return_ser + idle_ser.shift(1).fillna(0.0) * tbill_ser

    def daily_zero_cash(self, override_dict: dict) -> pd.Series:
        return self.result(override_dict).daily_return_ser


_WORKER: dict = {}


def _worker_init(plan_key_str: str, live_inputs, tbill_ser: pd.Series) -> None:
    _WORKER["runner"] = PodRunner(PLAN_DICT[plan_key_str], live_inputs, tbill_ser)
    _WORKER["plan"] = PLAN_DICT[plan_key_str]


def _worker_draw(config_dict: dict) -> dict:
    plan = _WORKER["plan"]
    try:
        runner = _WORKER["runner"]
        result = runner.result(config_dict)
        # The first rebalance decision (an all-cash target counts: a regime-off start is a decision, not a warm-up).
        # A draw whose warm-up runs past the window start would be judged on fewer days: listed as failed instead.
        first_ts = result.position_after_rebalance_df.index.min()
        if pd.isna(first_ts) or first_ts > pd.Timestamp(plan.eval_start_str):
            return {"config": config_dict, "sharpe_float": float("nan"), "error_str": f"first decision {first_ts}, after the window start"}
        return {"config": config_dict, "sharpe_float": sharpe_float(runner.daily(config_dict, result).loc[plan.eval_start_str:SEAL_END_STR])}
    except Exception as error_obj:  # noqa: BLE001 - a draw the spec refuses counts as failed, and is listed
        return {"config": config_dict, "sharpe_float": float("nan"), "error_str": repr(error_obj)[:200]}


# ---------------------------------------------------------------- shared inputs
def market_inputs():
    from alpha.scout.reaudit import factor_daily_df
    from alpha.scout.specs import taa_3x

    taa_inputs = taa_3x.load_inputs()
    date_index = pd.bdate_range("1998-01-01", pd.Timestamp.today().normalize())
    tbill_ser = tbill_daily_ser(taa_inputs.dtb3_ser, date_index)
    factor_df = factor_daily_df(date_index, tbill_ser)
    # No zero-fill before a fund existed (QQQ lists 1999-03): _monthly turns those months into NaN, and they drop out.
    return {"SPY": factor_df["SPY"], "QQQ": factor_df["QQQ"]}, tbill_ser


def save(name_str: str, robustness_dict: dict) -> None:
    pod_path = OUTPUT_DIR_PATH / pod_dir_str(name_str)
    pod_path.mkdir(parents=True, exist_ok=True)
    with (pod_path / "robustness.pkl").open("wb") as file_obj:
        pickle.dump(robustness_dict, file_obj)

    def plain(value_obj):
        if isinstance(value_obj, pd.Series):
            return {str(k): float(v) for k, v in value_obj.items()}
        if isinstance(value_obj, dict):
            return {str(k): plain(v) for k, v in value_obj.items()}
        if isinstance(value_obj, (list, tuple)):
            return [plain(v) for v in value_obj]
        if isinstance(value_obj, (np.floating, np.integer)):
            return value_obj.item()
        return value_obj

    (pod_path / "summary.json").write_text(json.dumps(plain(robustness_dict), indent=2, default=str), encoding="utf-8")
    bundle_path = REAUDITION_DIR_PATH / pod_dir_str(name_str) / "bundle.pkl"
    if bundle_path.exists():
        with bundle_path.open("rb") as file_obj:
            bundle = pickle.load(file_obj)
        bundle["robustness"] = robustness_dict
        with bundle_path.open("wb") as file_obj:
            pickle.dump(bundle, file_obj)
        card_path_list = sorted(CARD_DIR_PATH.glob(f"{pod_dir_str(name_str)}_reaudition*.html"))
        for card_path in card_path_list or [CARD_DIR_PATH / f"{pod_dir_str(name_str)}_reaudition.html"]:
            card_path.write_text(render_card(bundle), encoding="utf-8")


# ---------------------------------------------------------------- one planned pod
def run_plan(plan_key_str: str, market_dict: dict, tbill_ser: pd.Series) -> dict:
    plan = PLAN_DICT[plan_key_str]
    started_float = time.time()
    runner = PodRunner(plan, tbill_ser=tbill_ser)
    live_result = runner.result({})
    robustness_dict = {"plan_key_str": plan_key_str, "eval_start_str": plan.eval_start_str}
    robustness_dict["contribution"] = contribution_dict(live_result, plan.eval_start_str, SEAL_END_STR)
    robustness_dict["timing"] = timing_dict(runner.daily({}), market_dict, tbill_ser, plan.eval_start_str, SEAL_END_STR)
    print(f"  [{plan.name_str}] contribution + timing ({time.time() - started_float:.0f}s)", flush=True)
    robustness_dict["ablation"] = ablation_dict(runner.daily, list(plan.ablation_list), plan.eval_start_str, SEAL_END_STR,
                                                alt_run_fn=runner.daily_zero_cash)
    print(f"  [{plan.name_str}] ablation ({time.time() - started_float:.0f}s)", flush=True)

    rng_obj = np.random.default_rng(SEED_INT)
    config_list = [plan.sample_fn(rng_obj) for _ in range(DRAW_COUNT_INT)]
    with Pool(WORKER_COUNT_INT, initializer=_worker_init, initargs=(plan_key_str, runner.inputs({}), tbill_ser)) as pool_obj:
        draw_list = pool_obj.map(_worker_draw, config_list, chunksize=max(1, DRAW_COUNT_INT // (WORKER_COUNT_INT * 4)))
    random_result = RandomParameterResult(robustness_dict["ablation"]["live_sharpe_float"], draw_list)
    robustness_dict["random"] = {
        "summary": random_parameter_summary(random_result), "sharpe_list": [d["sharpe_float"] for d in draw_list],
        "draw_list": draw_list, "box_str": " ".join((plan.sample_fn.__doc__ or plan.sample_fn.__name__).split()),
        "error_list": [d for d in draw_list if "error_str" in d],
    }
    print(f"  [{plan.name_str}] random parameters ({time.time() - started_float:.0f}s)", flush=True)
    save(plan.name_str, robustness_dict)
    return robustness_dict


# ---------------------------------------------------------------- the other re-audited families
def other_family_dict() -> dict:
    from alpha.scout import family as f

    out_dict = {name_str: (lambda v=v: f.taa_variant_family(v)) for v, name_str in f.TAA_FAMILY_NAME_DICT.items() if v not in ("taa_3x", "taa_lin_1n_qqq")}
    out_dict.update({"NDX ATR": f.ndx_atr_family, "NDX NATR20": f.ndx_natr20_family,
                     "Compass": lambda: f.compass_family("compass"), "Compass QQQ": lambda: f.compass_family("compass_qqq"),
                     "TFI": f.tfi_family, "Trinity": f.trinity_family, "EOM": f.eom_family, "Sector IBS VOX IYR": f.sector_ibs_family,
                     "DV2": f.dv2_family, "DV2 Nasdaq 100": f.dv2_ndx_family})
    out_dict.update({name_str: (lambda v=v: f.dispersion_ibs_family(v)) for v, name_str in f.DISPERSION_IBS_NAME_DICT.items()})
    out_dict.update({name_str: (lambda v=v: f.hpi_family(v)) for v, name_str in f.HPI_FAMILY_NAME_DICT.items()})
    return out_dict


def run_other(name_str: str, family_fn, market_dict: dict, tbill_ser: pd.Series) -> dict:
    family = family_fn()
    result = family.run_config(family.live_config_dict)
    daily_ser = result.daily_return_ser
    start_str = str(daily_ser.loc[daily_ser.ne(0)].index.min().date())
    robustness_dict = {"plan_key_str": None, "eval_start_str": start_str,
                       "contribution": contribution_dict(result, start_str, SEAL_END_STR),
                       "timing": timing_dict(daily_ser, market_dict, tbill_ser, start_str, SEAL_END_STR)}
    save(name_str, robustness_dict)
    return robustness_dict


def main() -> None:
    argument_list = sys.argv[1:]
    market_dict, tbill_ser = market_inputs()
    started_float = time.time()
    if argument_list in ([], ["--all"]) or any(a in PLAN_DICT for a in argument_list):
        for plan_key_str in [a for a in argument_list if a in PLAN_DICT] or list(PLAN_DICT):
            print(f"== {PLAN_DICT[plan_key_str].name_str}", flush=True)
            rob = run_plan(plan_key_str, market_dict, tbill_ser)
            c, r = rob["contribution"], rob["random"]["summary"]
            print(f"   contribution {c['verdict_str']} (top-1 share {c['top_share_dict'].get(1, float('nan')):.0%}); "
                  f"timing {[t['verdict_str'] for t in rob['timing']]}; min spec: {rob['ablation']['min_spec_str']}; "
                  f"random {r['verdict_str']} (median {r['median_float']:.2f} vs live {r['live_sharpe_float']:.2f}) ({time.time() - started_float:.0f}s)", flush=True)
    if argument_list in ([], ["--all"], ["--others"]):
        for name_str, family_fn in other_family_dict().items():
            try:
                rob = run_other(name_str, family_fn, market_dict, tbill_ser)
            except Exception as error_obj:  # noqa: BLE001 - one failing family must not stop the others; it is printed
                print(f"== {name_str}: FAILED {error_obj!r}"[:300], flush=True)
                continue
            c = rob["contribution"]
            print(f"== {name_str}: contribution {c['verdict_str']} (top-1 share {c['top_share_dict'].get(1, float('nan')):.0%}, "
                  f"{c['traded_count_int']} assets); timing {[t['verdict_str'] for t in rob['timing']]} ({time.time() - started_float:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
