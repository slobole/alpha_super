TL;DR: it looks like “many parameters”, but in practice only `params:` are the **strategy parameters**. Everything else is **live deployment metadata**: who runs it, on which account, when it runs, how it executes, and whether it is enabled.

The clean way to think about it is:

\[
\text{Live Release} = \text{Identity} + \text{Routing} + \text{Timing} + \text{Strategy Ref} + \text{Strategy Params}
\]

So this file is not “the strategy config only”.  
It is “one approved live deployment of a strategy”.

Your example:

```yaml
release_id: user_001.pod_dv2.daily_moo.v1
user_id: user_001
pod_id: pod_dv2_01
account_route: U1234567
strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy
mode: paper
signal_clock_str: eod_close_plus_10m
execution_policy_str: next_open_moo
data_profile_str: norgate_eod_sp500_pit
params:
  max_positions_int: 10
  capital_base_float: 100000.0
risk_profile_str: standard_equity_mr
enabled_bool: true
```

## 1. Identity fields

### `release_id`
Example:
```yaml
release_id: user_001.pod_dv2.daily_moo.v1
```

What it does:
- unique id for this exact live release

Why it exists:
- you may run the same strategy in different forms later:
  - paper vs live
  - different accounts
  - different timing
  - different parameter sets

So this is the exact deployment version, not just “DV2”.

Used by:
- manifest validation and storage in [release_manifest.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/release_manifest.py)
- plan storage in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)

---

### `user_id`
Example:
```yaml
user_id: user_001
```

What it does:
- says who owns this deployment logically

Why it exists:
- later you want multiple users / managed accounts / multiple clients
- keeps releases grouped under one owner

Used by:
- release model and state persistence in [models.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/models.py)
- order plans and pod state in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)

---

### `pod_id`
Example:
```yaml
pod_id: pod_dv2_01
```

What it does:
- identifies one live pod

Why it exists:
- in v1, the pod is the main stateful live unit
- pod state includes:
  - current holdings
  - cash
  - strategy state
  - reconciliation history

So:

\[
\text{pod} = \text{one strategy deployment unit}
\]

In v1 effectively:
- `1 pod = 1 strategy + 1 account + 1 release`

Used by:
- pod state lookup in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)
- uniqueness rule in [release_manifest.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/release_manifest.py)

## 2. Routing field

### `account_route`
Example:
```yaml
account_route: U1234567
```

What it does:
- tells the order clerk which broker account should receive the orders

Why it exists:
- the strategy should not know broker account ids
- routing belongs to live infra, not research logic

Used by:
- broker session checks and snapshots in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)

## 3. Strategy reference

### `strategy_import_str`
Example:
```yaml
strategy_import_str: strategies.dv2.strategy_mr_dv2:DVO2Strategy
```

What it does:
- points to the research strategy that should be hosted live

Why it exists:
- you do **not** want copied `live_strategy.py` files
- the live layer imports the original research strategy

So the live target is:

\[
\text{target position} = f(\text{research strategy code}, \text{params}, \text{approved snapshot})
\]

Used by:
- strategy selection in [strategy_host.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/strategy_host.py)

## 4. Environment field

### `mode`
Example:
```yaml
mode: paper
```

What it does:
- declares whether this release is approved for `paper` or `live`

Why it exists:
- same strategy can exist in paper first, then live later
- runner should block execution if environment does not match

Used by:
- mode check in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)

## 5. Timing fields

### `signal_clock_str`
Example:
```yaml
signal_clock_str: eod_close_plus_10m
```

What it does:
- says **when the strategy is allowed to make its decision**

Why it exists:
- quant correctness
- timing must be explicit

For this release it means:
- build the decision snapshot after the daily close, with a small buffer

Used by:
- scheduling logic in [scheduler_utils.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/scheduler_utils.py)

---

### `execution_policy_str`
Example:
```yaml
execution_policy_str: next_open_moo
```

What it does:
- says **how and when orders should be sent**

Why it exists:
- decision time and execution time are not the same thing

For this release:
- decision from yesterday’s EOD data
- submit for next open
- broker order type becomes `MOO`

Used by:
- scheduling in [scheduler_utils.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/scheduler_utils.py)
- broker order type mapping in [strategy_host.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/strategy_host.py)

## 6. Data contract field

### `data_profile_str`
Example:
```yaml
data_profile_str: norgate_eod_sp500_pit
```

What it does:
- says what data contract this release expects

Why it exists:
- the same strategy logic can require different data stacks
- here it documents:
  - Norgate
  - EOD
  - point-in-time S&P 500 universe

Right now in v1:
- this is partly metadata
- and partly a control hook
- for example, `same_day_moc + intraday` is explicitly blocked until real intraday adapters exist

Used by:
- validation and future data routing
- intraday guard in [strategy_host.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/strategy_host.py)

## 7. Actual strategy parameters

### `params`
Example:
```yaml
params:
  max_positions_int: 10
  capital_base_float: 100000.0
```

This is the only place that contains the actual strategy-level knobs.

### `max_positions_int`
What it does:
- sets the maximum number of simultaneous DV2 positions

Why it exists:
- it directly affects DV2 sizing and slot logic

Used by:
- DV2 host in [strategy_host.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/strategy_host.py)

---

### `capital_base_float`
What it does:
- initial capital for a brand-new pod if no existing state is present

Why it exists:
- on first run, the pod needs starting cash
- later, real broker state / pod state takes over

Used by:
- default pod initialization in [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)
- strategy instantiation in [strategy_host.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/strategy_host.py)

So for your DV2 manifest, the true strategy params are only:

\[
\theta = \{ \text{max\_positions\_int},\ \text{capital\_base\_float} \}
\]

Everything else is deployment metadata.

## 8. Risk label

### `risk_profile_str`
Example:
```yaml
risk_profile_str: standard_equity_mr
```

What it does:
- labels which risk policy should apply

Why it exists:
- you want a generic live system where risk checks are not hard-coded into each strategy

Important:
- in the current v1 implementation, this is mostly a **reserved hook / metadata**
- it is stored and carried through, but not yet deeply enforced by a separate risk engine

So it is there because the architecture needs it, even if v1 still uses a minimal risk gate.

## 9. Activation flag

### `enabled_bool`
Example:
```yaml
enabled_bool: true
```

What it does:
- turns the release on or off

Why it exists:
- easiest operational kill switch
- lets you keep the manifest without deleting it

Used by:
- due-release selection and execution filtering in [release_manifest.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/release_manifest.py) and [runner.py](C:/Users/User/Documents/workspace/alpha_super/alpha/live/runner.py)

## Why there are “so many” fields

Because this file is doing two jobs at once:

1. **strategy hosting**
2. **live deployment control**

If it only held strategy parameters, it would be very short.  
But then the runner would not know:

- which account to trade
- whether it is paper or live
- when to run
- how to execute
- whether it is enabled
- which pod state to load

So the manifest needs both layers.

The simplest mental split is:

- **Research parameters**
  - `params`
- **Live deployment metadata**
  - everything else

## The shortest practical interpretation of your file

This manifest says:

- run the research strategy `DVO2Strategy`
- for user `user_001`
- inside pod `pod_dv2_01`
- routed to account `U1234567`
- in `paper` mode
- build signals after daily close
- send orders for next open as `MOO`
- use Norgate EOD point-in-time S&P 500 data
- use `max_positions = 10`
- start with `100,000` if the pod has no prior state
- tag it with risk profile `standard_equity_mr`
- and keep it enabled

If you want, next I can give you a **leaner manifest version** and tell you:
- which fields are truly mandatory
- which fields could later get sane defaults
- and which ones are just future-proofing.