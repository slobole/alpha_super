# Live Release Templates

TL;DR: these files are safe examples only. Real client release YAMLs live under
`alpha/live/releases/<client_id>/` on each VPS and are ignored by Git.

## Workflow

1. Pick the template for the wired strategy.
2. Copy it to a local client folder:

```powershell
New-Item -ItemType Directory -Force alpha\live\releases\<client_id>
Copy-Item docs\live\release_templates\pod_qpi_daily_moo.yaml.example `
  alpha\live\releases\<client_id>\pod_qpi_01.yaml
```

3. Edit the local copy:

- `identity.user_id`: the client folder/id.
- `identity.release_id`: unique deployment version.
- `identity.pod_id`: stable POD/sleeve id.
- `broker.account_route`: real IBKR paper/live account route.
- `deployment.mode`: `paper` or `live`.
- `deployment.enabled_bool`: keep `false` until the VPS doctor and broker checks pass.
- `execution.pod_budget_fraction_float`: capital slice for this POD.

4. Validate before running:

```powershell
uv run python -m alpha.live.runner status `
  --mode paper `
  --releases-root alpha/live/releases/<client_id> `
  --json
```

For a snapshot-mode client VPS, also run:

```powershell
uv run python scripts\doctor_norgate_client.py
```

## Repo Rule

Do not commit real release YAMLs. They are client-specific state and can block
`git pull` on another VPS. The tracked contract is:

```text
docs/live/release_templates/*.yaml.example
alpha/live/releases/your_user/README.md
```

Actual runtime files are local:

```text
alpha/live/releases/<client_id>/*.yaml
```

## Wired Templates

| Template | Wired strategy | Data profile |
|---|---|---|
| `pod_dv2_daily_moo.yaml.example` | `strategies.dv2.strategy_mr_dv2:DVO2Strategy` | `norgate_eod_sp500_pit` |
| `pod_qpi_daily_moo.yaml.example` | `strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy` | `norgate_eod_sp500_pit` |
| `pod_taa_btal_fallback_tqqq_vix_cash_monthly_open.yaml.example` | `strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash` | `norgate_eod_etf_plus_vix_helper` |
| `pod_taa_btal_1n_fallback_tqqq_vix_cash_monthly_open.yaml.example` | `strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash` | `norgate_eod_etf_plus_vix_helper` |
| `pod_taa_btal_linearity_1n_fallback_qqq_vix_cash_monthly_open.yaml.example` | `strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash` | `norgate_eod_etf_plus_vix_helper` |
| `pod_ndx_atr_normalized_monthly_open.yaml.example` | `strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy` | `norgate_eod_ndx_pit` |
| `pod_ndx_atr_normalized_vxn_scaled_monthly_open.yaml.example` | `strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy` | `norgate_eod_ndx_pit_plus_vxn_helper` |
