# Other accounts (accounts without a strategy)

One rule, nothing to configure:

- **The LIVE release YAMLs decide what a strategy is.** Their accounts get Pods,
  Pod returns and trading, exactly as before.
- **Every other account in the Flex is one of the client's other accounts**
  (for example a vault, or a sub-account that is not used). It never fails the
  sync, never gets a Pod, a scheduler, orders or a Pod row.

Keep every account in the Flex query. Nothing is added to the dashboard config,
the releases or the database by hand.

## How an other account is counted

```text
[Release YAMLs] -----------> [Pods, Pod returns, trading]      (unchanged)
[Flex: every account] -----> [Client NAV / P&L / TWR totals]
                                 + "Other accounts (not traded)" rows
```

- **Found from the saved Flex.** `alpha/live/reporting_accounts.py` reads the
  raw Flex XML the sync already keeps (`flex_import`, read-only). Any account
  that is not a strategy account (saved Pod bindings or current LIVE releases)
  is an other account.
- **Starts on its first active day:** the first saved day with any nonzero
  money field (NAV, deposits, fees, interest...), so money that came in and
  went out the same day still counts. An account that never had any activity
  is left out and never shown, in any currency.
- **Membership is never taken back.** It is read from every saved import,
  including ones a later Flex correction replaced. If a correction drops the
  account, its days become missing (totals withheld), not zero.
- **Becomes a Pod on the Pod's first day.** If an account gets a LIVE YAML
  later, it stays an other account up to the day before the Pod's first
  return day and is the Pod from then on. Client history does not change.
  Each new Flex import automatically saves which accounts were strategies at
  import time. This internal evidence survives range corrections and duplicate
  imports; it needs no configuration. An ordinary Pod's first EOD remains only
  its baseline, while an account already observed as Other keeps that day.
  Existing stores are upgraded automatically during normal sync, using their
  saved bindings for legacy imports. Role changes completed before that
  evidence was recorded cannot be reconstructed from the old XML alone.
  Reads remain read-only, including against stores not yet upgraded.
- **Continuous ownership across role changes:** an Other-to-Pod handover does
  not create a weekend or holiday NAV requirement. Every exchange session,
  observed non-session movement and real ownership entry/exit is still checked.
  Missing, empty or non-finite required NAV/TWR never proves an unused account.
- **Name:** the IBKR alias (Account Information `acctAlias`) when the query has
  it, else `Other account …` plus the last four characters. Full account IDs
  never reach the screens or the client report.
- **In the totals:** client NAV, P&L and TWR include it. A transfer between it
  and a strategy account on the same day nets to zero.
- **Shown apart:** a grey "not traded" row on the Overview allocation, the
  Performance page and in contribution, an "Other accounts" table in the client
  PDF, and its own rows in the Pods CSV.
- **Missing is never zero.** Once counted, a day without its NAV withholds the
  client totals for that day. Pod numbers are not affected.
- **USD only**, like the strategy accounts. An active non-USD other account is
  named in the financial error instead of being dropped from the totals.

## Trade-off

An account in the Flex without a YAML used to fail the sync. Now it is counted
as an other account and shown as a grey row, so a forgotten strategy YAML shows
up on the screen instead of as a sync error. Add a new Pod's account to the
Flex after its LIVE YAML is in the releases folder (see the runbook); days
before the Pod's first return day count as an other account.
