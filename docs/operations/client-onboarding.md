---
title: New Client Onboarding
description: Short checklist for preparing a Windows VPS and proving it is ready for a new LIVE client.
document_type: how-to
authority: guide
risk_scope: live
status: draft
source_paths:
  - docs/live/LIVE_USER_SETUP_QUICK.md
  - docs/live/LIVE_RUNBOOK.md
  - docs/live/NORGATE_SNAPSHOT_V1.md
  - docs/live/DASHBOARD_V3_RUNBOOK.md
---

# New Client Onboarding

!!! abstract "Goal"
    Prepare one Windows VPS for one client, verify every Pod, and only then approve LIVE activation.

!!! warning "Draft"
    This checklist is still being completed. The final checks do not submit orders. LIVE activation is blocked until its separate procedure is reviewed.

## Flow

~~~mermaid
flowchart LR
    A["VPS"] --> B["Secure access"]
    B --> C["Code"]
    C --> D["Norgate"]
~~~

~~~mermaid
flowchart LR
    A["IBKR / TWS"] --> B["Pods"]
    B --> C["Doctors"]
    C --> D{"Final approval"}
    D -->|"Approved"| E["LIVE"]
    D -->|"Not approved"| X["Stop"]
~~~

## Setup

### 1. Open the VPS

- **Open a Windows server in Hyonix, location Ashburn.**
- Budget: about **$24/month**. Do not use the $12 plan until it is tested and approved.
- Set Windows power mode to **High performance**.
- Disable PowerShell **QuickEdit**.
- Install Windows updates and restart.

### 2. Secure the VPS

- Install Tailscale and connect it to the approved operations network.
- Disable key expiry only if the access policy allows it.
- Do not expose trading services through public ports.
- Record who owns the Windows administrator and Tailscale access.

Check:

~~~powershell
tailscale status
~~~

### 3. Install the project

- Install VS Code, Git, and **uv**.
- Sign in to GitHub with the approved operations user.
- Clone **alpha_super** into **Documents\workspace**.

~~~powershell
cd C:\Users\Administrator\Documents\workspace\alpha_super
uv sync
~~~

Transfer **config.env** privately. Never paste it into chat or commit it to Git.

### 4. One client only

- Store this client's manifests under **alpha/live/releases/[client_id]/**.
- Every LIVE manifest in this checkout must have the same **user_id**.
- One Pod must map to one strategy, one IBKR account route, and one state database.
- Use a separate production checkout for another client.

### 5. Connect Norgate

Add the real values to **config.env**:

~~~text
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true
NORGATE_CLIENT_ID=<client_id>
NORGATE_RELEASES_ROOT=alpha/live/releases/<client_id>
NORGATE_SNAPSHOT_ROOT=C:\alpha\norgate_snapshots
NORGATE_API_URL=http://<norgate_tailscale_ip>:8787
NORGATE_API_TOKEN=<secret>
~~~

The Norgate Doctor runs in Step 9, after the Pod manifests exist.

### 6. IBKR and TWS

- Finish the client accounts, POA or linked-account access, trading permissions, and market data in IBKR.
- Install the approved standalone/offline TWS version.
- Set the TWS time zone to **America/New_York**.
- Confirm TWS can see every account route used by the client's Pods.

Stop if an account is missing or the logged-in user has the wrong permissions.

### 7. Create the Pods

Copy the required templates from **docs/live/release_templates/** into the client release folder.

For the readiness check:

~~~yaml
deployment:
  enabled_bool: true

execution:
  auto_submit_enabled_bool: false
~~~

Do not start a LIVE scheduler yet.

### 8. Start monitoring

Start the local dashboard:

~~~powershell
uv run python -m alpha.live.dashboard_v3 --host 127.0.0.1 --port 8080
~~~

Open:

~~~text
http://127.0.0.1:8080
~~~

Then configure Healthchecks.io and the LIVE watchdog. Monitoring must work before LIVE starts.

!!! warning "Watchdog task"
    The scheduled-task setup is not yet reviewed. Do not register or replace the production task from this page.

### 9. Final pre-LIVE check

Run these from the repository root.

Norgate Doctor:

~~~powershell
uv run python scripts\doctor_norgate_client.py --client-id YOUR_CLIENT_ID --releases-root alpha/live/releases/YOUR_CLIENT_ID
~~~

One strategy:

~~~powershell
uv run python -m alpha.live.runner doctor --mode live --releases-root alpha/live/releases/YOUR_CLIENT_ID --pod-id YOUR_POD_ID
~~~

Use the automatic check below for final acceptance. It finds every LIVE Pod, so none can be skipped.

<details markdown="1">
<summary><strong>Copy/paste automatic check for every Pod</strong></summary>

~~~powershell
$ErrorActionPreference = 'Stop'

$repoRoot = 'C:\Users\Administrator\Documents\workspace\alpha_super'
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath .\pyproject.toml)) {
    throw 'pyproject.toml is missing. Wrong checkout.'
}
if (-not (Test-Path -LiteralPath .\.git)) {
    throw '.git is missing. Wrong checkout.'
}

$clientId = 'YOUR_CLIENT_ID'
$allReleasesRoot = (Resolve-Path -LiteralPath .\alpha\live\releases).Path
$releaseRoot = Join-Path $allReleasesRoot $clientId

if (-not (Test-Path -LiteralPath $releaseRoot)) {
    throw "Client release folder is missing: $releaseRoot"
}

$manifestJson = @'
import json
import sys
from pathlib import Path
from alpha.live.release_manifest import load_release_list

all_releases_root_str = sys.argv[1]
client_id_str = sys.argv[2]
release_root_path_obj = Path(sys.argv[3]).resolve()
release_list = [
    release_obj
    for release_obj in load_release_list(all_releases_root_str)
    if release_obj.mode_str == "live"
]

if not release_list:
    raise SystemExit("No LIVE manifests found.")
if {release_obj.user_id_str for release_obj in release_list} != {client_id_str}:
    raise SystemExit("LIVE manifests do not belong to one client.")
if any(
    not Path(release_obj.source_path_str).resolve().is_relative_to(release_root_path_obj)
    for release_obj in release_list
):
    raise SystemExit("A LIVE manifest exists outside the selected client folder.")

account_route_list = [release_obj.account_route_str for release_obj in release_list]
if len(account_route_list) != len(set(account_route_list)):
    raise SystemExit("Duplicate IBKR account route.")
if any(not release_obj.enabled_bool for release_obj in release_list):
    raise SystemExit("A LIVE manifest is disabled and cannot be checked.")
if any(release_obj.auto_submit_enabled_bool for release_obj in release_list):
    raise SystemExit("Auto-submit must remain false during readiness.")

print(json.dumps([
    {
        "pod_id_str": release_obj.pod_id_str,
        "strategy_import_str": release_obj.strategy_import_str,
        "account_route_str": release_obj.account_route_str,
        "db_path_str": f"alpha/live/state/live/{release_obj.pod_id_str}.sqlite3",
    }
    for release_obj in release_list
]))
'@ | uv run python - $allReleasesRoot $clientId $releaseRoot

if ($LASTEXITCODE -ne 0) {
    throw 'Manifest inventory failed.'
}

$manifestList = @($manifestJson | ConvertFrom-Json)
$manifestList | Format-Table pod_id_str, strategy_import_str, account_route_str, db_path_str

uv run python scripts\doctor_norgate_client.py --client-id $clientId --releases-root $releaseRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Norgate Doctor failed.'
}

foreach ($manifest in $manifestList) {
    $podId = $manifest.pod_id_str
    Write-Host "=== $podId ==="

    $doctorJson = uv run python -m alpha.live.runner doctor --mode live --releases-root $releaseRoot --pod-id $podId --json
    if ($LASTEXITCODE -ne 0) {
        throw "Doctor failed for $podId."
    }

    $doctor = $doctorJson | ConvertFrom-Json
    if ($doctor.overall_verdict_str -ne 'PASS') {
        throw "Doctor is $($doctor.overall_verdict_str) for $podId."
    }
    if ($doctor.release_dict.strategy_import_str -ne $manifest.strategy_import_str) {
        throw "Strategy mismatch for $podId."
    }
    if ($doctor.release_dict.account_route_str -ne $manifest.account_route_str) {
        throw "Account mismatch for $podId."
    }

    uv run python -m alpha.live.runner status --mode live --releases-root $releaseRoot --pod-id $podId
    if ($LASTEXITCODE -ne 0) {
        throw "Status failed for $podId."
    }

    uv run python -m alpha.live.scheduler_service next_due --mode live --releases-root $releaseRoot --pod-id $podId
    if ($LASTEXITCODE -ne 0) {
        throw "Next Due failed for $podId."
    }
}

Write-Host "PRE-LIVE CHECK: PASS ($($manifestList.Count) Pods)"
~~~

</details>

Required:

- Norgate ends with **RESULT: PASS**.
- Each Pod Doctor shows the correct strategy and account.
- Every discovered Pod must reach **VERDICT: PASS** at its real due time.
- Any **WAIT**, **BLOCK**, command failure, or mismatch means the check is incomplete.
- Success ends with **PRE-LIVE CHECK: PASS (N Pods)**.

!!! danger "No trading commands"
    These diagnostics may update snapshots, logs, or health artifacts. They do not submit orders. Do not add **tick**, **serve**, **submit_vplan**, or another trading command.

### 10. Activate LIVE

!!! danger "LIVE activation is not available yet"
    Do not start LIVE from this page. Use the dedicated activation procedure only after it is tested on the VPS and marked **Reviewed**.

After approved activation:

1. Require fresh green monitoring evidence for every Pod.
2. Capture the first trusted broker EOD snapshot.
3. Complete [IBKR Flex Performance setup](ibkr-flex-performance-setup.md).

## Pre-LIVE readiness

Pre-LIVE readiness is complete only when:

- [ ] The final command block prints **PRE-LIVE CHECK: PASS**.
- [ ] The discovered Pod count and account routes match the approved client list.
- [ ] The production watchdog task is installed and proven through its reviewed procedure.
- [ ] Final LIVE approval is recorded.

Full onboarding is complete only after:

- [ ] LIVE activation runs through a **Reviewed** procedure.
- [ ] Every scheduler has the expected Pod, account, process owner, and database.
- [ ] Every Pod produces fresh green monitoring evidence after activation.
- [ ] Every Pod has a trusted broker EOD snapshot.
- [ ] IBKR Flex performance is verified.

The original Word notes remain source material. Personal emails, account IDs, credentials, and private chat links are intentionally excluded.
