# Local Release Folder

TL;DR: this directory is a placeholder only. Real release YAMLs are local per
VPS/client and are ignored by Git.

Use tracked templates from:

```text
docs/live/release_templates/
```

Copy a template into a client folder:

```powershell
New-Item -ItemType Directory -Force alpha\live\releases\<client_id>
Copy-Item docs\live\release_templates\pod_qpi_daily_moo.yaml.example `
  alpha\live\releases\<client_id>\pod_qpi_01.yaml
```

Then edit the local copy:

- `identity.user_id`: client id / release folder name.
- `identity.release_id`: unique deployment version.
- `identity.pod_id`: stable POD state id.
- `broker.account_route`: real IBKR paper/live account route.
- `deployment.mode`: `paper` or `live`.
- `deployment.enabled_bool`: keep `false` until the VPS is verified.
- `execution.pod_budget_fraction_float`: capital slice for this POD.

Do not commit local release YAMLs. They are different for each client and can
block `git pull` on another VPS.
