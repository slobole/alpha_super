"""Validate the sector ETF IBS MCPT replicas against the engine family on every grid configuration (in sample).

For each pod: the replica (`fast_daily_list` on the unshuffled `mcpt_matrix`, gross) against the parity-engine family
(`FamilyRunner.run_config`, net of the given costs), on the engine's dates up to 2022-12-30. Prints the daily return
correlation per configuration (min / median), and the Spearman correlation of the configurations' Sharpe ratios.

    uv run python scripts/research/scout_p6c_sector_ibs_20261002/validate_replicas.py [pod ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy import stats

from alpha.scout.engines.weights import CostModel
from alpha.scout.metrics import sharpe_float

SEAL_END_STR = "2022-12-30"
GROSS_COST = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0)


def validate(name_str: str, family, module, inputs, fast_kwarg_dict: dict) -> dict:
    date_index, matrix = module.mcpt_matrix(inputs)
    config_list = family.config_list()
    started_float = time.time()
    fast_list = module.fast_daily_list(matrix, date_index, config_list, **fast_kwarg_dict)
    fast_seconds_float = time.time() - started_float
    row_list = []
    for config_dict, fast_vec in zip(config_list, fast_list):
        fast_ser = pd.Series(fast_vec, index=date_index)
        row_dict = {"label": family.label_str(config_dict)}
        for cost_str, cost_model in (("net", CostModel()), ("gross", GROSS_COST)):
            engine_ser = family.run_config(config_dict, cost_model).daily_return_ser.loc[:SEAL_END_STR]
            common_index = engine_ser.index.intersection(date_index)
            row_dict[f"corr_{cost_str}"] = float(np.corrcoef(engine_ser.loc[common_index], fast_ser.loc[common_index])[0, 1])
            row_dict[f"sharpe_engine_{cost_str}"] = sharpe_float(engine_ser.loc[common_index])
        row_dict["sharpe_fast"] = sharpe_float(fast_ser.loc[common_index])
        row_list.append(row_dict)
    frame = pd.DataFrame(row_list)
    summary_dict = {
        "pod": name_str, "configs": len(frame), "fast_seconds": round(fast_seconds_float, 2),
        "min_corr_net": frame["corr_net"].min(), "median_corr_net": frame["corr_net"].median(),
        "min_corr_gross": frame["corr_gross"].min(),
        "spearman_sharpe_net": stats.spearmanr(frame["sharpe_engine_net"], frame["sharpe_fast"]).statistic,
        "spearman_sharpe_gross": stats.spearmanr(frame["sharpe_engine_gross"], frame["sharpe_fast"]).statistic,
    }
    print(frame.round(3).to_string(), flush=True)
    print(summary_dict, flush=True)
    return summary_dict


def main() -> None:
    from alpha.scout.family import dispersion_ibs_family, sector_ibs_family
    from alpha.scout.specs import sector_dispersion_ibs, sector_ibs

    name_list = sys.argv[1:] or ["sector_ibs_vox_iyr", *sector_dispersion_ibs.VARIANT_DICT]
    summary_list = []
    for name_str in name_list:
        if name_str == "sector_ibs_vox_iyr":
            inputs = sector_ibs.load_inputs()
            summary_list.append(validate(name_str, sector_ibs_family(inputs), sector_ibs, inputs, {}))
        else:
            inputs = sector_dispersion_ibs.load_inputs(name_str)
            config = sector_dispersion_ibs.VARIANT_DICT[name_str].config
            summary_list.append(validate(name_str, dispersion_ibs_family(name_str, inputs), sector_dispersion_ibs, inputs,
                                         {"base_config": config}))
    print(pd.DataFrame(summary_list).round(4).to_string())


if __name__ == "__main__":
    main()
