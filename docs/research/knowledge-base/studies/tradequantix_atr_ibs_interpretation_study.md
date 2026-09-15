---
title: "ATR-band ETF dip buying with staged entries"
description: "Declared four-fund staged ETF interpretation passes the frozen economic gate but remains diagnostic because auction/source/short-later evidence is incomplete."
document_type: research
authority: guide
risk_scope: research
source_paths:
  - "pakal-research/reports/tradequantix_atr_ibs_interpretation_study/knowledge_record.json"
  - "C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/REPORT.md"
  - "C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/REPORT_FULL.md"
  - "C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/research_spec_frozen.json"
  - "C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/research.ipynb"
  - "C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/run_manifest.json"
---

<!-- GENERATED FILE. EDIT THE CANONICAL KNOWLEDGE RECORD. -->

# ATR-band ETF dip buying with staged entries

!!! warning "Research-only"
    This page summarizes saved research evidence. It does not authorize LIVE, allocation, or deployment.

## TL;DR

> **Verdict:** Declared four-fund staged ETF interpretation passes the frozen economic gate but remains diagnostic because auction/source/short-later evidence is incomplete.

> **Status:** `diagnostic`

> **Disposition:** `promising_component`

> **Replication:** `not_reproducible`

## Research question

Does a declared four-fund staged ATR/IBS dip strategy retain net value and tolerable drawdown, and do child entries add value beyond the first tranche?

## Exact setup

| Field | Value |
| --- | --- |
| Signal family | staged_etf_mean_reversion |
| Universe | ["Four source-selected US-listed funds SPXL,TMF,TQQQ,UGL; GBTC omitted, three unredistributed slots"] |
| Decision | Completed CloseT; opening-known split/sell-liability safety only reduces prior instructions |
| Fill | NextClose DAY buy limit, C_D<=limit fixed atT |
| Primary cost layer | central_research |
| Last reviewed | 2026-09-14T17:21:17.956729+00:00 |

## Timing and overnight attribution

```text
information available: Completed CloseT; opening-known split/sell-liability safety only reduces prior instructions
primary executable fill: NextClose DAY buy limit, C_D<=limit fixed atT
```

If final `Close_T` data formed the signal, a hypothetical `Close_T` entry is diagnostic only unless a separate pre-close protocol was modeled. Comparing that diagnostic with `Open_(T+1)` and applying the compounded return identity shows whether the apparent edge occurred in the overnight gap. Missing attribution numbers mean **not tested**, not zero.

| Attribution field | Value |
| --- | --- |
| Status | not_tested |
| Diagnostic Path | Selection-conditioned pre-fill CloseT to CloseD identity only |
| Executable Path | No alternative nextOpen portfolio evaluated; source buys atNextClose |
| Method | Saved filled signed-share dollar decomposition, not causal open counterfactual |
| Headline Result | The day-before-fill movement is not earned strategy return |
| Metrics | {"after_open_dollars": -304748.66384595016, "decision_to_close_dollars": -541996.391782463, "fills": 1627, "identity_error_max": 0.0, "interpretation": "Pre-fill selected movement only, not earned return or open strategy", "opening_gap_dollars": -237247.7279365128} |
| Artifact | C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/tables/same_quantity_timing.csv |

## Primary metrics

| Metric | Value |
| --- | ---: |
| Period | 2019-01-02:2026-04-02 |
| Universe | Four source-selected US-listed funds SPXL,TMF,TQQQ,UGL; GBTC omitted, three unredistributed slots |
| Cost Layer | central_research |
| Cagr | 4.63% |
| Annualized Volatility | 6.34% |
| Sharpe | 0.745 |
| Maximum Drawdown | -7.04% |
| Turnover | 617.93% |

## Four separate verdicts

| Question | Conclusion |
| --- | --- |
| Source Replication | Unavailable exact source, explicit interpretation |
| Predictive Value | No pristine independent prediction claim; fixed survivor basket and short later period |
| Economic Value | Declared four-fund staged ETF interpretation passes the frozen economic gate but remains diagnostic because auction/source/short-later evidence is incomplete. |
| Promotion | Research-only diagnostic; no trading approval |

## Key findings

| Feature | Role | Direction | Status | Effect | Action |
| --- | --- | --- | --- | --- | --- |
| ATR/IBS three-stage prior-high recovery | entry | Long dip buying | diagnostic | Validation CAGR0.046262, MDD-0.070446 | No trading implementation; preserve result and avoid same-history tuning |

## Visual evidence

![01-equity_drawdown.png](../assets/tradequantix_atr_ibs_interpretation_study/01-equity_drawdown.png)


## Limitations

- Not exact author replication; clipped child LLV/filter tail and final UGL body are interpreted
- GBTC excluded for unresolved economics; no renormalization
- Current-vintage fixed survivor basket and author-selected parameters; no true PITselection claim
- Original cache omitted currency/security_name and stable before-after vintage; retained prior QA, no invented metadata
- No padding; ex-date entitlement cash is not pay-date liquidity proof
- Daily closes do not prove closing-auction access, partial fills or quantity capacity
- Fractional post-split claims held, no cash-in-lieu model
- Short post-publication window and related seen history
- No calibrated impact, cash yield or independent execution proof
- Same-quantity pre-fill timing identity is selection-conditioned and not trading return
- Frozen gate failures: 

## Next gates

- New independently frozen evidence only; do not retune seen history

## Sources

- `{"content_id": "sha256:ff7aa17a0d5dc873837309212086161dd5b82b01d8a8e6637903fdb2d6ea44cc", "location": "C:/Users/User/Documents/workspace/0_papers/TradeQuantiX_PDFs/TradeQuantiX_PDFs_WHITE/etf-mean-reversion-methods-four-systems.pdf", "read_complete": true, "role": "Source mechanism, incomplete author code; full read receipt in prior F011 intake", "source_id": "14"}`

## Canonical artifacts

| Artifact | Pakal path |
| --- | --- |
| Source Rule Map | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/SOURCE_RULE_MAP.md` |
| Hypothesis Registry | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/hypothesis_registry.json` |
| Experiment Ledger | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/experiment_ledger.jsonl` |
| Decision Log | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/decision_log.jsonl` |
| Frozen Specification | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/research_spec_frozen.json` |
| Concise Report | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/REPORT.md` |
| Full Report | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/REPORT_FULL.md` |
| Knowledge Record | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/knowledge_record.json` |
| Manifest | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/run_manifest.json` |
| Research State | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/research_state.json` |
| Notebook | `C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/research.ipynb` |
| Primary Source Code | `["C:/Users/User/Documents/workspace/pakal/pakal-research/tradequantix_atr_ibs_interpretation_engine.py"]` |
| Primary Tables | `["C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/tables/period_metrics.csv"]` |
| Primary Charts | `["C:/Users/User/Documents/workspace/pakal/pakal-research/reports/tradequantix_atr_ibs_interpretation_study/charts/equity_drawdown.png"]` |
