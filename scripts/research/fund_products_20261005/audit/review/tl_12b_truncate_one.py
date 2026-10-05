"""Timing lens 12b: the ndx_atr_cap truncation run alone (the parallel attempt hit a Norgate watchlist read error)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tl_12_truncate as t
if __name__ == "__main__":
    print(t.run("ndx_atr_cap"), flush=True)
