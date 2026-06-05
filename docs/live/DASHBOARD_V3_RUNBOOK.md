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
