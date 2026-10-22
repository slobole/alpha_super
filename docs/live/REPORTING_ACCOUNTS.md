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
- **Starts on its first money day:** the first saved day with a nonzero
  starting or ending NAV. Before that it held nothing. An account that never
  held money is left out and never shown.
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
- **USD only**, like the strategy accounts. A non-USD other account is named in
  the financial error instead of being dropped from the totals.

## Trade-off

An account in the Flex without a YAML used to fail the sync. Now it is counted
as an other account and shown as a grey row, so a forgotten strategy YAML shows
up on the screen instead of as a sync error. Add a new Pod's account to the
Flex after its LIVE YAML is in the releases folder (see the runbook), or its
first days count as an other account and not as Pod days.
