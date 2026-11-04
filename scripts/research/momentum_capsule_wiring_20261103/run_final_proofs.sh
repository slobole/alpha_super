#!/usr/bin/env bash
# Every proof of the wiring on the final code, one after the other: parallel Norgate reads are avoided on purpose.
# Usage: run_final_proofs.sh <python> <temporary snapshot root> <gate base json>. No real client root, release file, broker or task is touched.
set -euo pipefail
PY="$1"; ROOT="$2"; BASE_JSON="$3"
HERE="scripts/research/momentum_capsule_wiring_20261103"
OUT="results/research/momentum_capsule_wiring_20261103"
RESEARCH_ROWS="../alpha_super/results/scout/residual_momentum_20261031/report_wiring/closed_design_rows.parquet"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
stamp() { echo "== $1 $(date +%H:%M:%S)"; }

for LEG in 1 2 3; do
  stamp "parity $LEG"
  "$PY" "$HERE/parity_rows.py" "$LEG" "$RESEARCH_ROWS"
done

rm -f "$OUT"/snapshot_decision_*.json
for T in 2026-10-30 2026-10-16; do
  for PROFILE in norgate_eod_ndx_momentum_legs_pit norgate_eod_r1000_momentum_legs_pit; do
    stamp "export $PROFILE $T"
    S=$(date +%s)
    "$PY" scripts/export_norgate_snapshot.py --snapshot-root "$ROOT/$T" --profile "$PROFILE" --snapshot-date "$T" --overwrite
    echo "export seconds $(( $(date +%s) - S ))"
  done
  stamp "decisions $T"
  "$PY" "$HERE/snapshot_decision.py" "$ROOT/$T" "$T" 1 2 3
done

for LEG in 1 2 3; do
  stamp "replay cut $LEG"
  "$PY" "$HERE/replay_cut.py" "$LEG" 40
done

for MODULE in strategy_mo_capsule_leg1_natr_ndx strategy_mo_capsule_leg2_residual_ndx; do
  stamp "vanilla $MODULE"
  "$PY" scripts/research/run_strategy_analysis.py "strategies.momentum.$MODULE" --analysis vanilla
done

stamp "gate"
"$PY" scripts/research/mr_capsule_liq_build_20261101/gate_ndx_taa.py "$(pwd)" "$OUT/gate_new.json" 2>&1 | tee "$OUT/gate_new.log"
"$PY" scripts/research/mr_capsule_liq_build_20261101/gate_ndx_taa.py compare "$BASE_JSON" "$OUT/gate_new.json"
stamp "PROOFS DONE"
