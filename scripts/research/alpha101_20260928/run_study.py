"""Runner for the Alpha101 study (research only). Frozen plan: docs/research/ALPHA101_PREREG_20260928.md.

    python scripts/research/alpha101_20260928/run_study.py --alphas U1 | U2       evaluate the 100 alphas -> z store
    python ... --controls U1 | U2                                                REV1 / REV5 signal store
    python ... --ic U1 | U2                                                      daily IC of the alphas (-> WF weights)
    python ... --composites U1 | U2                                              C_EQ, C_WF, C_EQ_noInd panels + weights
    python ... --stage-a U1 | U2                                                 Stage A tables
    python ... --v1 | --v2 | --v3 | --parity | --cbil                            checks (section 9)
    python ... --grid U1 --costs engine,stress --workers 6                       24 long-only cells
    python ... --extras U1                                                       HEDGED, REV pods, IndNeutralize-free pod
    python ... --small-accounts | --u2-candidates                                after the candidates are known
    python ... --analyze                                                         books, rule, confidence, charts, summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpha101_20260928 import common  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", choices=("U1", "U2"))
    parser.add_argument("--controls", choices=("U1", "U2"))
    parser.add_argument("--ic", choices=("U1", "U2"))
    parser.add_argument("--composites", choices=("U1", "U2"))
    parser.add_argument("--stage-a", choices=("U1", "U2"))
    parser.add_argument("--v1", action="store_true")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--v3", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--cbil", action="store_true")
    parser.add_argument("--grid", choices=("U1", "U2"))
    parser.add_argument("--extras", choices=("U1", "U2"))
    parser.add_argument("--small-accounts", action="store_true")
    parser.add_argument("--u2-candidates", action="store_true")
    parser.add_argument("--costs", default="engine,stress")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    common.RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    cost_list = [c for c in args.costs.split(",") if c]
    if args.alphas:
        from alpha101_20260928 import alphas

        alphas.compute_all(args.alphas)
    if args.controls:
        from alpha101_20260928 import composites

        composites.build_controls(args.controls)
    if args.ic:
        from alpha101_20260928 import composites

        composites.build_alpha_ic(args.ic)
    if args.composites:
        from alpha101_20260928 import composites

        composites.build_composites(args.composites)
    if args.stage_a:
        from alpha101_20260928 import stage_a

        stage_a.run(args.stage_a)
    if args.cbil:
        from alpha101_20260928 import checks

        checks.check_cbil()
    if args.v1 or args.v2 or args.v3:
        from alpha101_20260928 import invariance

        if args.v1:
            invariance.run_v1()
        if args.v2:
            invariance.run_v2()
        if args.v3:
            invariance.run_v3()
    if args.parity:
        from alpha101_20260928 import engine_parity

        engine_parity.run_parity()
    if args.grid:
        from alpha101_20260928 import run_pods

        run_pods.run_grid(args.grid, cost_list, args.workers)
    if args.extras:
        from alpha101_20260928 import run_pods

        run_pods.run_extras(args.extras, cost_list, args.workers)
    if args.small_accounts:
        from alpha101_20260928 import run_pods

        run_pods.run_small_accounts()
    if args.u2_candidates:
        from alpha101_20260928 import run_pods

        run_pods.run_u2_candidates(cost_list, args.workers)
    if args.analyze:
        from alpha101_20260928 import analyze

        analyze.main()


if __name__ == "__main__":
    main()
