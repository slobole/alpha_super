# Research ledger

`scout_ledger.jsonl` is Scout's append-only, hash-chained record of every
registration, trial, test result, vault opening and gate result
(`alpha/scout/ledger.py`, design in `docs/plans/SCOUT_DESIGN.md` section 8).

Rules:

- Never edit, delete or reorder a line. Corrections are new rows.
- Write it only from the main checkout (or Pakal through the editable install of
  that checkout). Rows written in a feature worktree would diverge from main.
- Check it with `uv run python -m alpha.scout verify`.
