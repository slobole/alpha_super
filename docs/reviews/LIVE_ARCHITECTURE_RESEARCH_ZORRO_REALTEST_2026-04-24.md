# Live Architecture Research: RealTest, Zorro, And Our Pod Model

TL;DR: I think your core idea is right. A pod is an independent strategy
sleeve. It should live in its own real IBKR subaccount/account route, with its
own cash, positions, and broker truth. In that model, the architecture is not
broken. I would not rewrite everything today. I would make the system easier to
operate, add a proper incubation/forward-testing layer, and keep learning from
RealTest and Zorro.

## The Human Version

When you say:

```text
POD = one independent strategy sleeve
```

that is the right mental model.

Example:

```text
Client Edy has $1,000,000

Sleeve 1: DV2 strategy  -> IBKR subaccount A
Sleeve 2: QPI strategy  -> IBKR subaccount B
Sleeve 3: TAA strategy  -> IBKR subaccount C
```

Those sleeves should not know about each other. They should not share positions.
They should not need to coordinate. Each sleeve looks at its own account,
decides what it wants, trades, and reconciles itself.

The simple math is:

```text
total_client_equity = sleeve_1_equity + sleeve_2_equity + sleeve_3_equity
```

and inside each sleeve:

```text
orders_today = strategy_decision + current_broker_account_state
```

That is enough. No need to overcomplicate the idea.

My earlier warning was only about a different setup:

```text
one IBKR account
many internal "pod" labels inside that same account
```

That is where things get messy, because the broker only reports one combined
position list. If every pod is a real subaccount/account route, the problem
mostly disappears.

## Current Verdict

I would keep the current direction:

```text
strategy decides -> execution prepares orders -> broker sends/fills -> system reconciles
```

That is good architecture.

I would improve the packaging around it:

- make the config easier to read;
- make the operator workflow simpler;
- make `serve --user edy` the normal operating mode;
- add a dashboard/status view;
- add a clean incubation system for strategies before they go live;
- make MOO/MOC paper testing expectations explicit.

I would not copy Zorro completely. I would not copy RealTest completely. I
would steal the parts that fit us.

## What RealTest Teaches Us

RealTest is very close to your world because it is built around daily systems,
multi-strategy portfolios, and tomorrow's orders.

The important RealTest idea is:

```text
research system -> generate tomorrow's orders -> send through OrderClerk -> read actual fills -> use those fills tomorrow
```

RealTest's own docs say Tomorrow's Orders are for daily systems where the next
orders can be known while the market is closed. It explicitly says it is not a
live intraday engine. That is fine, because most of your current strategies are
daily or monthly.

OrderClerk is the most relevant part. It sends order lists to IBKR, receives
execution reports, and keeps a live trade list. The key detail: each order keeps
the strategy name, so tomorrow's orders can be based on the actual live
positions for that strategy.

What we should copy:

- a simple "orders for next session" view;
- a clear review screen before submission;
- actual fills become tomorrow's starting point;
- every live order knows which sleeve/strategy it came from;
- multi-strategy reporting is first-class.

What we should not copy blindly:

- RealTest is not designed as a full live service;
- it is not a general Python research/live stack;
- same-day close and intraday workflows are awkward unless you add live data.

Best lesson from RealTest:

```text
for daily strategies, a clean order-generation and reconciliation workflow beats a complicated live engine
```

Sources: [RealTest home](https://mhptrading.com/), [Tomorrow's Orders](https://mhptrading.com/docs/topics/idh-topic1145.htm), [OrderClerk](https://mhptrading.com/docs/topics/idh-topic1143.htm), [OrdersMode](https://mhptrading.com/docs/topics/idh-topic11164.htm).

## What Zorro Teaches Us

Zorro is more of a full trading platform. One script can research, backtest,
optimize, connect to brokers, trade live, resume state, and show live status.

That is powerful. It is also more dangerous if we copy it too literally.

What Zorro does well:

- one clear place to run the system;
- live status pages;
- broker/data plugins;
- account and asset lists;
- resume after restart;
- multiple live instances/accounts in the paid version;
- strong integration between research and live.

What I would not copy:

- too much logic living inside one script;
- automatic retraining while live as a default habit;
- optimizer-first thinking;
- mixing strategy logic and broker mechanics too tightly.

The best Zorro lesson for us is operational:

```text
the live system should feel like one product, not a bag of scripts
```

Right now our engine idea is okay, but the operator experience can be smoother.
Zorro is good at that.

Sources: [Zorro features](https://zorro-project.com/features.php), [Zorro trade mode](https://zorro-project.com/manual/en/trading.htm), [Zorro broker plugin API](https://zorro-project.com/manual/en/brokerplugin.htm).

## What I Would Change

### 1. Make Pods Human And Concrete

A pod should mean:

```text
one strategy sleeve, one broker subaccount, one independent state
```

That should be written everywhere in the live docs.

Do not describe a pod mainly as a config object. Describe it like this:

```text
This sleeve runs DV2 for Edy.
It trades account A.
It has its own cash and positions.
It does not know about the other sleeves.
```

### 2. Make The Config Read Like A Release Card

The config should answer simple questions:

- Who owns this sleeve?
- Which IBKR account does it trade?
- Which strategy runs here?
- When does it decide?
- When does it send orders?
- Is it manual or automatic?
- Is it enabled?

That is it. The file should feel like an operating card, not a database row.

### 3. Add A Better Status Screen

Before changing deep architecture, I would improve the live status output.

I want to see:

```text
Edy
  DV2 sleeve: waiting for next open / no issues
  QPI sleeve: order plan ready / manual review needed
  TAA sleeve: next decision at month end
```

This matters more than another abstraction.

### 4. Separate Paper Testing From Real Execution Testing

Carlos's point is important. IBKR paper can test the logic and the flow, but
paper fills are not proof that MOO/MOC execution behaves like live.

The practical ladder should be:

```text
backtest -> paper flow test -> tiny live test -> scale slowly
```

For MOC strategies, the live signal must be computed before the auction cutoff.
Do not backtest with the final close and pretend that exact final-close signal
was available before the close.

The clean MOC idea is:

```text
signal from pre-close live snapshot -> submit MOC -> fill at official close
```

Not:

```text
signal from official close -> submit MOC
```

*** CRITICAL*** A same-day close strategy must not use information from the
final close if the order had to be submitted before the final close was known.

### 5. Build An Incubation Layer

This is the biggest idea from Carlos's Clearinghouse/Ledger setup.

You want a place where many strategies can run before they are promoted to real
money. Not a notebook. Not a random paper account. A real incubation system.

Plain English:

```text
Strategies place paper orders.
The system pretends to fill MOO/MOC at official open/close.
It records cash, positions, trades, and stats.
After enough time, a strategy can be promoted to a real pod.
```

That would be very valuable.

This is separate from live IBKR execution. It is a forward-testing laboratory.

### 6. Keep The Current Core, But Make It Cleaner

I would keep this:

```text
decision -> order plan -> submit -> reconcile
```

I would clean this:

- fewer confusing names;
- less noisy persistence;
- better logs;
- simpler operator commands;
- clearer separation between research, incubation, and live.

## Should We Start A New Project?

Maybe, but not as a replacement yet.

I would do it only as a side-by-side prototype:

```text
alpha_live_next
```

The goal would be to learn and simplify, not to immediately replace production.

The test is simple:

```text
same data + same sleeve state -> same decisions and same orders
```

If the new version cannot match the current one, it is not ready.

## My Recommendation

Do not rewrite everything now.

Do this instead:

0. Make every future change simple enough that a human can read it without the agent who wrote it.
1. Keep the pod model: one independent strategy sleeve per real IBKR subaccount.
2. Rewrite the live docs and config language so a human can operate it.
3. Improve `serve --user edy` and the status view.
4. Treat paper as a system-flow test, not an execution-quality test.
5. Add a small-live validation stage before scaling.
6. Build an incubation layer inspired by RealTest OrderClerk and Carlos's Ledger.
7. Later, if the cleaner prototype proves itself, migrate slowly.

My bottom line:

```text
The architecture is not fundamentally wrong.
The pod idea is right.
The next improvement is operational clarity and incubation, not a giant rewrite.
```
