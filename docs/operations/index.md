---
title: Operations
description: Entry point for reviewed operating procedures.
document_type: how-to
authority: guide
risk_scope: live
---

# Operations

!!! danger "Use reviewed procedures only"
    An operating page is published here only after its commands, expected results, stop conditions, recovery boundaries, and affected LIVE surfaces have been checked.

<div class="grid cards" markdown>

-   :material-account-cog-outline: **New Client Onboarding**

    Follow the short setup path, then run one final Norgate and per-Pod check.

    **Status:** Master checklist published; detailed procedures are being audited one at a time.

    [:octicons-arrow-right-24: Open the onboarding map](client-onboarding.md)

-   :material-bank-transfer: **Set Up IBKR Flex Performance**

    Create the client Query, initialize the read-only Performance Shadow, verify account mapping, and schedule the daily refresh.

    **Status:** Reviewed against the current CLI, task scripts, and IBKR documentation.

    [:octicons-arrow-right-24: Open the procedure](ibkr-flex-performance-setup.md)

-   :material-server-network: **VPS Runtime Checklist**

    Start the Norgate and client-VPS runtimes, then prove the Pods, monitoring, and Flex data are healthy.

    **Status:** Command review complete; production VPS proof still required.

    [:octicons-arrow-right-24: Open the checklist](vps-runtime-checklist.md)

</div>

## Source material awaiting publication

These files remain available in the repository but are not yet presented as audited portal instructions:

| Need | Existing source |
|---|---|
| LIVE entry and advanced operation | `LIVE_START_HERE.md`, `docs/live/LIVE_RUNBOOK.md` |
| Initial user setup | `docs/live/LIVE_USER_SETUP_QUICK.md` |
| Debugging | `docs/live/DEBUGGING_RUNBOOK.md` |
| Norgate snapshot and client VPS | `docs/live/NORGATE_SNAPSHOT_V1.md` |
| Dashboard operation | `docs/live/DASHBOARD_V3_RUNBOOK.md` |

!!! warning "Meaning of this list"
    A path proves that source material exists. It does not prove that every command is current. Promote one procedure at a time after code and runtime verification.
