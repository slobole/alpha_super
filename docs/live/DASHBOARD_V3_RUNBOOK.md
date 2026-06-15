# Dashboard V3 — one-page deploy runbook

Operator console for the multi-pod live book. Flask + Jinja + HTMX, no Node,
no build step. Lives on the trading VPS, reached over Tailscale only.

## Prereqs (VPS)

- Python 3.12 with `uv` installed.
- The repo already cloned to `/srv/alpha` (or wherever the live engine runs).
- Tailscale installed and logged in (`tailscale status` shows the VPS in your tailnet).

## Bring it up

```bash
cd /srv/alpha
uv sync
uv run python -m alpha.live.dashboard_v3 --host 127.0.0.1 --port 8080
```

That's it for a manual smoke test. From your laptop (also on the tailnet):

```
http://<vps-hostname>:8080/
```

If Tailscale's MagicDNS is on you can use the hostname directly; otherwise use
the Tailscale IP from `tailscale ip -4` on the VPS.

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

On every pod's green/yellow/gray → red transition, the dashboard fires a single
message. State is persisted in `alpha/live/logs/notification_state.json`, so
recovering and re-failing fires a fresh alert. Missing env var = silent.

The same notification path also watches the Live OPS Inspector rollup. If the
Inspector itself turns red because the report is stale or a required proof is
missing, it sends one red-transition message and then waits for recovery before
firing again.

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
and the watchdog have `ALPHA_DISCORD_WEBHOOK_URL` set, the same red transition
can alert twice — harmless, accepted.

## Live vs Backtest comparison

The pod detail page includes a compact Live vs Backtest card when a comparison
artifact exists. It is read-only: it compares observed live fills and state to a
same-condition reference backtest, and it does not send orders.

Run it from Operator Tools with `Live vs Backtest`. Artifacts are written under:

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
curl -s http://127.0.0.1:8080/healthz   # ok
```

From a non-tailnet machine on the VPS's public IP:

```bash
curl -m 3 http://<vps-public-ip>:8080/  # should hang / connection refused
```

If that returns HTML, fix the bind before logging off.

## Routes cheatsheet

| Path | Purpose |
|---|---|
| `/live`, `/paper`, `/incubation` | Mode pages |
| `/journal` | Operator intervention log |
| `/healthz` | Plain-text health check |
| `/fragments/top-bar` | Polled (5s) — also runs notification check |
| `/fragments/health-strip` | Polled (15s) |
| `/fragments/schedule-strip` | Polled (30s) |
| `/fragments/pod-detail/<id>` | Expanded detail (polled 5s while open) |
| `/fragments/events-tail/<id>` | Live event log (polled 5s while open) |
| `/fragments/equity-chart/<id>?window=30d\|90d\|all` | SVG curve |
| `/api/action-token` | Token for action POSTs |
| `POST /api/pods/<id>/diff/run` | Live vs Backtest |
| `POST /api/pods/<id>/actions/<name>` | tick / submit_vplan / reconcile / eod_snapshot |
| `GET /api/jobs/<id>` | Job status (HTML for HTMX, JSON otherwise) |

## Where things live

- Code: `alpha/live/dashboard_v3/`
- Templates: `alpha/live/dashboard_v3/templates/`
- Static: `alpha/live/dashboard_v3/static/` (vendored HTMX + tiny JS for new-event badge)
- Tests: `tests/test_dashboard_v3_*.py`
- Operator journal: `alpha/live/logs/operator_journal.jsonl`
- Notification state: `alpha/live/logs/notification_state.json`
