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

## Live-vs-backtest band (optional)

Drop expected-PnL stats into `alpha/live/expected_pnl.yaml`:

```yaml
pod_dv2_caspersky_live:
  daily_mean_return_float: 0.00012
  daily_volatility_float:  0.0118
  band_sigma_float: 2.0
  sample_count_int: 252
```

Each pod's EOD card then shows "Today +X% · Expected ±Y%" with a yellow ⚠ when
the day lands outside the band. No file or no entry = the line is just hidden.

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
| `POST /api/pods/<id>/diff/run` | Run DIFF |
| `POST /api/pods/<id>/actions/<name>` | tick / submit_vplan / reconcile / eod_snapshot |
| `GET /api/jobs/<id>` | Job status (HTML for HTMX, JSON otherwise) |

## Where things live

- Code: `alpha/live/dashboard_v3/`
- Templates: `alpha/live/dashboard_v3/templates/`
- Static: `alpha/live/dashboard_v3/static/` (vendored HTMX + tiny JS for new-event badge)
- Tests: `tests/test_dashboard_v3_*.py`
- Operator journal: `alpha/live/logs/operator_journal.jsonl`
- Notification state: `alpha/live/logs/notification_state.json`
