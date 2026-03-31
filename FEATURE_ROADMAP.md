# Feature Roadmap

This file is for future capabilities only. It is not the place for current assumptions, current gaps, or house philosophy.

Use this file to track features that should be added to move the research stack toward a stronger institutional platform.

## Feature Template

Each feature should be described with:

- Feature
- Why it matters
- Design intent
- Dependencies
- Acceptance criteria
- Status

## Planned Features

### Walk-Forward Backtesting

**Why it matters**

Walk-forward testing helps separate research specification from repeated in-sample tuning and provides a more realistic estimate of how model updates would behave through time.

\[
\text{Research window} \rightarrow \text{test window} \rightarrow \text{roll forward} \rightarrow \text{repeat}
\]

**Design intent**

Add a framework for repeated train-validate-test style evaluation over sequential time windows without changing the core causal execution rules of the engine.

**Dependencies**

- clear window specification API
- repeatable strategy parameter interface
- reporting layer that can aggregate per-window results
- explicit rules for what is allowed to change between windows

**Acceptance criteria**

- can define multiple sequential windows
- each test window uses only information available before that window
- output includes per-window metrics and aggregate metrics
- the workflow is hard to misuse in a way that creates leakage

**Status**

Planned

### Cash Reserve Buffer

**Why it matters**

A reserve cash buffer can reduce margin pressure and execution friction when overnight target sizing drifts because the open differs from the prior close.

\[
V^{investable}_t = (1-b)\,V^{close}_{t-1}
\]

with an initial candidate of:

\[
b = 0.02
\]

**Design intent**

Add an optional reserve cash setting for overnight target-allocation strategies so the strategy intentionally leaves some capital unallocated before market-on-open style fills.

**Dependencies**

- strategy-level configuration for reserve cash
- clear reporting of investable capital versus total equity
- tests around residual cash behavior after rebalances

**Acceptance criteria**

- reserve cash can be enabled or disabled explicitly
- target sizing uses investable capital rather than full prior-close value when enabled
- reporting makes residual cash visible after rebalances
- the feature does not change strategies that do not opt in

**Status**

Planned

## Future Candidates

No additional feature candidates are committed yet. Add new items only when the design intent is concrete enough to guide implementation.
