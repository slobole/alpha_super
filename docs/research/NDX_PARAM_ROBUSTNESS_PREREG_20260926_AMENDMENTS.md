# Amendments to NDX_PARAM_ROBUSTNESS_PREREG_20260926.md

The frozen PREREG file is left byte-identical (SHA-256 `0cc15db565a4ba6cdcce21dfa43a5655558b7249987e0374d9484b4a51d35476`,
recorded in `results/research/ndx_param_robustness_20260926/prereg_freeze.json` at 2026-09-26 21:00:11 +03:00, before
the grids ran at about 21:07). Everything below was written on 2026-09-26 AFTER the results, mostly in response to the
quant-pitfalls review. No amendment changes the decision ("keep L") or the shadow line.

## A1 - Inverse-vol budget when fewer than N names qualify (disclosure of an implementation reading)

PREREG section 4 says the IV weights sum "to the VXN-scaled exposure". The code gives IV the same budget equal weight
would invest: (count / N) x the scaled exposure. The two readings differ only when fewer than N names are eligible:
4 of 233 invested decisions at N = 30, none at N <= 20. The stage-2 plateau centre is A0 (N 10, equal weight), so no
verdict depends on it.

## A2 - Freeze evidence

PREREG section 0 says the freeze is the git commit that adds the file. No commit was made (the owner had not asked for
one). The freeze evidence is the SHA-256 plus timestamp file written before any grid cell was computed. Post-freeze code
edits, all disclosed:
- core replica, 21:06, after the parity gate and before the grids: suppressed divide-by-zero warnings (no numeric change);
- analysis script, after the grids: R4 made to follow PREREG section 8 when the plateau centre is A0; post-hoc section
  (labelled); chart tick fix; A3-A5 below.

## A3 - Rebalance offsets start on different dates

Schedules with k > 0 first trade in mid-February 2000, k <= 0 in January, because a December-1999 anchor would need a
December-1998 close that the data (from 1999-01-04) does not have. The PREREG standalone window (from 2000-01-04)
therefore compares different start dates across offsets. Added sensitivity: standalone timing-luck statistics and the
standalone Reality Check re-measured from the common start 2000-04-01. The stage-4 candidate (median over offsets per
buffer) and all G3 numbers are unaffected.

## A4 - ADV window

The capacity proxy used a 20-session median with at least 15 valid sessions; PREREG says 20 sessions. Changed to 20
valid sessions and all grids re-run; no capacity figure changed at the reported precision.

## A5 - A0-centred stages in the decision

PREREG section 8 read literally: a stage whose plateau centre is A0 can also "pass" (which would mean switching from L
to A0), and the shadow line is chosen over all stage candidates, A0-centred ones included. The first analysis excluded
A0-centred stages from both; the code now follows the literal reading. Nothing passes R1, and the shadow line is
unchanged (stage 3, worst-block margin -0.19, against -0.26 and -0.23 for the A0-centred stages).

## A6 - Invariance check V2 extended

V2 was specified for the stage-1 lists. It now also rebuilds every k = 0 cell of every stage (73 configurations,
including IV weights, SMA50/200, the QQQ gate, VXN settings and buffers) from the decision-day view; all are identical.
The NaN mismatches reported for sigma63 (24 name-dates) and ROC12-1 (151) involve zero names that were PIT members with a
close on the decision day: they are names without a price that day, where R(t) is undefined. They cannot be selected.
