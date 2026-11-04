#!/usr/bin/env bash
# Every proof of the wiring on the final code, one after the other: parallel Norgate reads are avoided on purpose.
# Usage: run_final_proofs.sh <python> <temporary snapshot root> <gate base json>. No real client root, release file, broker or task is touched.
set -u
PY="$1"; ROOT="$2"; BASE_JSON="$3"
HERE="scripts/research/momentum_capsule_wiring_20261103"
OUT="results/research/momentum_capsule_wiring_20261103"
RESEARCH_ROWS="../alpha_super/results/scout/residual_momentum_20261031/report_wiring/closed_design_rows.parquet"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
stamp() { echo "== $1 $(date +%H:%M:%S)"; }

for LEG in 1 2 3; do
  stamp "parity $LEG"
  "$PY" "$HERE/parity_rows.py" "$LEG" "$RESEARCH_ROWS" 2>/dev/null | grep -E '"(decisions|same|names_differ|missing|largest_weight_gap_when_same)"|PARITY DONE' | tr -d '\n'; echo
done

rm -f "$OUT"/snapshot_decision_*.json
for T in 2026-10-30 2026-10-16; do
  for PROFILE in norgate_eod_ndx_momentum_legs_pit norgate_eod_r1000_momentum_legs_pit; do
    stamp "export $PROFILE $T"
    S=$(date +%s)
    "$PY" scripts/export_norgate_snapshot.py --snapshot-root "$ROOT/$T" --profile "$PROFILE" --snapshot-date "$T" --overwrite 2>&1 | grep -v "it/s\|s/it\|Norgate Data\|^$" | tail -1 | cut -c1-200
    echo "export seconds $(( $(date +%s) - S ))"
  done
  stamp "decisions $T"
  "$PY" "$HERE/snapshot_decision.py" "$ROOT/$T" "$T" 1 2 3 2>&1 | tail -1 | cut -c1-100
done

for LEG in 1 2 3; do
  stamp "replay cut $LEG"
  "$PY" "$HERE/replay_cut.py" "$LEG" 40 2>/dev/null | tail -1 | cut -c1-200
done

for MODULE in strategy_mo_capsule_leg1_natr_ndx strategy_mo_capsule_leg2_residual_ndx; do
  stamp "vanilla $MODULE"
  "$PY" scripts/research/run_strategy_analysis.py "strategies.momentum.$MODULE" --analysis vanilla 2>&1 | grep -E "^vanilla|^FAIL" | cut -c1-120
done

stamp "gate"
"$PY" scripts/research/mr_capsule_liq_build_20261101/gate_ndx_taa.py "$(pwd)" "$OUT/gate_new.json" > "$OUT/gate_new.log" 2>&1
"$PY" scripts/research/mr_capsule_liq_build_20261101/gate_ndx_taa.py compare "$BASE_JSON" "$OUT/gate_new.json" | cut -c1-200
stamp "PROOFS DONE"
