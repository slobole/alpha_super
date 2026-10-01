"""Attach the S3 results (run_s3.py) to the P5 re-audition bundles of the two LIVE pods and re-render their cards.

    uv run python scripts/research/scout_p6_reaudition_20261002/attach_s3.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.card import grade_str, render_card
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout"
NOTE_DICT = {
    "TAA_3x": ("W", (
        "Over the ETFs' full histories (to 2022), the momentum ranking of the five defensive ETFs does not predict their next-month "
        "returns: neither across assets nor asset by asset. The VIX gate does predict risk: next-month QQQ volatility is 1.75 times "
        "higher when the gate is off. The pod's edge comes from holding the leveraged Nasdaq fallback while volatility is calm, not "
        "from the defensive rotation. For class W these are diagnostics; the burden is on S5 and S6."
    )),
    "NDX_VXN": ("X", (
        "The score ranks stocks: its monthly rank correlation with next-month returns among eligible members is positive (t 2.55), "
        "and the top 10 beat the eligible average. The S5 MCPT still fails, because its per-asset null keeps each stock's long-run "
        "mean return: a momentum rule that mostly finds stocks that are strong over their whole life earns as much on shuffled "
        "histories. What S3 sees is real in sample; what the gate does not see is timing information beyond each stock's own drift."
    )),
}


def main() -> None:
    for pod_dir_str, (class_str, note_str) in NOTE_DICT.items():
        pod_dir_path = ROOT_PATH / "reaudition" / pod_dir_str
        with (pod_dir_path / "bundle.pkl").open("rb") as file_obj:
            bundle = pickle.load(file_obj)
        bundle["s3"] = {"class_str": class_str, "note_str": note_str, "result": json.loads((pod_dir_path / "s3.json").read_text(encoding="utf-8"))}
        with (pod_dir_path / "bundle.pkl").open("wb") as file_obj:
            pickle.dump(bundle, file_obj)
        (ROOT_PATH / "cards" / f"{pod_dir_str}_reaudition_20261002.html").write_text(render_card(bundle), encoding="utf-8")
        print(pod_dir_str, grade_str(bundle))


if __name__ == "__main__":
    main()
