# Dashboard V3 — one-page deploy runbook

Operator console for the multi-pod live book. Flask + Jinja + HTMX, no Node,
no build step. Lives on the trading VPS, reached over Tailscale only.

## Prereqs (VPS)

- Python 3.12 with `uv` installed.
- The repo already cloned to `/srv/alpha` (or wherever the live engine runs).
- Tailscale installed and logged in (`tailscale status` shows the VPS in your tailnet).

## Bring it up

No application username/password is required. The former
`ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR` setting is ignored and may be removed.
Keep the listener on loopback. For another machine, use Tailscale Serve HTTPS
with operator-only tailnet access. Do not publish the service publicly or use
Tailscale Funnel. Anyone allowed to reach it can view its financial data;
with `--enable-actions`, they can also operate its advanced controls.

```bash
cd /srv/alpha
uv sync --locked
uv run python -m alpha.live.dashboard_v3 --host 127.0.0.1 --port 8080 --read-only
```

On the VPS itself, open `http://127.0.0.1:8080/`. For a laptop,
first configure the HTTPS proxy below; a loopback listener is not directly
reachable at `http://<vps-hostname>:8080/`.

`--read-only` blocks operational POSTs and action previews server-side, prevents
notification sends/state writes and disables file-generating trade-sheet GETs.
It does not merely hide buttons. Enabling actions requires replacing it with
`--enable-actions` under a separately approved operational rollout; removing
`--read-only` alone does not enable actions. Client pages never submit trading commands.

For client ownership, financial reporting and the synthetic preview, read
[CLIENT_OPERATOR_WORKSPACE.md](CLIENT_OPERATOR_WORKSPACE.md). When a registry is
configured, `/` opens the client directory; `/vps` retains the explicitly global
advanced workspace. No deployment is performed by this document.

### config.env is loaded automatically

On startup the dashboard reads `config.env` from the repo root and exports
every `KEY=value` line into the process environment — same loader the live
runner uses. This is critical on a VPS that does **not** have the local Norgate
Data Updater installed: set

```ini
ALPHA_USE_NORGATE_SNAPSHOT_BOOL=true
```

in `config.env` and the dashboard's data builders will use local snapshots
instead of retrying NDU ten times per refresh.

Pass `--skip-env-file` only when the host already exports the required
environment variables (e.g. when the systemd unit sets them inline).

## Make it permanent (systemd)

Copy `docs/live/dashboard_v3.service` to `/etc/systemd/system/dashboard_v3.service`,
adjust `User=`, `WorkingDirectory=`, and `Environment=` to match your install,
then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dashboard_v3
sudo systemctl status dashboard_v3
```

Logs go to journald:

```bash
journalctl -u dashboard_v3 -f
```

## Expose via Tailscale (recommended)

The service binds to `127.0.0.1:8080`. Tailscale-serve makes that visible
across the tailnet with free HTTPS:

```bash
tailscale serve --bg --https 443 127.0.0.1:8080
tailscale serve status
```

Open `https://<vps-hostname>.<tailnet>.ts.net/` from any device on the tailnet.

## Discord red-alert notifications (optional)

Create a Discord webhook in your private server, then set the env var before
`systemctl start`:

```ini
Environment=ALPHA_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

The default **read-only dashboard sends no notifications and writes no alert
state**, even with a webhook configured. Use the independent scheduled watchdog
below for unattended alerts; do not enable trading actions just to get alerts.

In an explicitly actions-enabled dashboard, top-bar polling can send one alert
per Pod/Inspector transition to red. It is browser-driven, not a background
monitor. Its state is `alpha/live/logs/notification_state.json`. Missing webhook
means silent. Recovery permits a later red transition to alert again.

## Live OPS Inspector

Dashboard V3 surfaces the Inspector verdict near the top of each mode page. The
CLI view is:

```powershell
uv run python -m alpha.live.runner ops_report --mode live --json
```

The Inspector report is read-only. It reads the existing dashboard summary and
POD evidence, then applies the contract from
`docs/live/INSPECTOR_CONTRACT.md`: unknown is not green, stale is not green, and
silence is caught by a separate heartbeat.

The scheduled watchdog (`scripts/live_ops_watchdog.py`, see
`docs/live/LIVE_RUNBOOK.md`) is the supported way to run the Inspector and the
heartbeat on a timer. It keeps its own notification state file
(`alpha/live/logs/watchdog_notification_state.json`), so when both the dashboard
in actions-enabled mode and the watchdog have `ALPHA_DISCORD_WEBHOOK_URL` set, the same red transition
can alert twice — harmless, accepted.

## Live vs Backtest comparison

The pod detail page includes a compact Live vs Backtest card when a comparison
artifact exists. Viewing a saved artifact is read-only. It shows observed live
fills/state against a reference; inspect dates, capital, release and accounting
basis before treating it as a same-condition comparison. Legacy artifacts are
not automatically certified decision-time replays.

Generating a new comparison from Operator Tools (`Live vs Backtest`) does not
send orders, but it writes artifacts and may populate data caches. This action
is disabled under `--read-only`; it is not a passive status check. When explicitly
authorized in a write-enabled console, its artifacts are written under:

```text
results/live_reference_compare/<mode>/<pod_id>/<timestamp>/
```

The dashboard card links to `index.html` and `trade_fill_diff.csv`. The CSV is
the first table to inspect when asking how live differed from the backtest: live
shares, backtest shares, share diff, live average fill, backtest fill, price
diff bps, notional diff, and a plain note such as `matched` or `backtest trade
without matching live fill`.

## Manual Broker Ticket

The pod detail page includes `Operator Tools -> Manual Ticket` for paper/live
pods. This is a break-glass IBKR order ticket, not a strategy-state workflow.

Supported v1 fields:

- asset symbol;
- side: `BUY` or `SELL`;
- order type: `MKT` or `LMT`;
- integer share quantity;
- limit price for `LMT`;
- operator name and reason.

The ticket submits exactly the entered order as a `DAY` order through the
configured pod broker route and writes `manual_order_submit_requested` plus
`manual_order_submit_completed` or `manual_order_submit_failed` to the live
event JSONL log. It does not read live quotes, read broker positions, rebuild a
VPlan, reconcile fills, or update strategy state. Check IBKR first, then use
the ticket as a logged alternative to typing the same order directly in IBKR.

## Verify Tailscale-only exposure

From the VPS:

```bash
ss -tlnp | grep 8080      # should show 127.0.0.1:8080 only
curl http://127.0.0.1:8080/healthz
```

From a non-tailnet machine on the VPS's public IP:

```bash
curl -m 3 http://<vps-public-ip>:8080/  # should hang / connection refused
```

If that returns HTML, fix the bind before logging off.

## Routes cheatsheet

| Path | Purpose |
|---|---|
| `/clients` | Operator-only client selection; never an investor portal |
| `/clients/<client>/<view>` | Overview, performance, strategies, exposure, activity, diagnostics, report |
| `/vps` | Advanced all-VPS scope; client selection does not apply here |
| `/live`, `/paper`, `/incubation` | Mode pages |
| `/journal` | Operator intervention log |
| `/healthz` | Authenticated application check; not proof of broker/process health |
| `/fragments/top-bar` | Polled (5s) — also runs notification check |
| `/fragments/health-strip` | Polled (15s) |
| `/fragments/schedule-strip` | Polled (30s) |
| `/fragments/pod-detail/<id>` | Expanded detail (polled 5s while open) |
| `/fragments/events-tail/<id>` | Live event log (polled 5s while open) |
| `/fragments/equity-chart/<id>?window=30d\|90d\|all` | SVG curve |
| `/api/action-token` | Same-origin action token; not sufficient without a one-use preview approval |
| `/fragments/command-catalog/<id>` | Fixed PowerShell Copy controls, separate from action previews; disabled read-only |
| `POST /api/pods/<id>/manual-order-preview` | Validate/freeze a manual ticket; no broker submission |
| `POST /api/pods/<id>/manual-order` | Consume frozen ticket approval once and dispatch |
| `POST /api/pods/<id>/diff/run` | Live vs Backtest |
| `POST /api/pods/<id>/actions/<name>` | tick / submit_vplan / reconcile / eod_snapshot |
| `GET /api/jobs/<id>` | Job status (HTML for HTMX, JSON otherwise) |

## Where things live

Supported startup defaults to **read-only**. Enabling advanced operations requires
the explicit `--enable-actions` flag and trusted operator network access. Keep that flag
out of the read-only service template and demo. Preview approval lasts 120 seconds,
is bound to the exact target/release/saved state and cannot be reused after dispatch.
Tick can create new plans; direct submit/reconcile are bound to captured plan intent.
After an uncertain result inspect broker/job evidence before creating a new approval.
Single dashboard process only; CLI/scheduler execution does not inherit UI approvals.

- Code: `alpha/live/dashboard_v3/`
- Templates: `alpha/live/dashboard_v3/templates/`
- Static: `alpha/live/dashboard_v3/static/` (vendored HTMX + tiny JS for new-event badge)
- Tests: `tests/test_dashboard_v3_*.py`
- Operator journal: `alpha/live/logs/operator_journal.jsonl`
- Notification state: `alpha/live/logs/notification_state.json`
