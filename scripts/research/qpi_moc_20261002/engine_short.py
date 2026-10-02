"""Current engine run of strategy_mr_qpi_ibs_rsi_exit 2004-2007 for replica parity (research-only)."""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import run_variant
s = run_variant(show_display_bool=False, save_results_bool=False, end_date_str="2007-12-31")
out = Path(__file__).resolve().parents[3] / "results/research/qpi_moc_20261002/engine_2004_2007_transactions.csv"
tx = s.get_transactions()
pd.DataFrame(tx).to_csv(out)
print("ok", out)
