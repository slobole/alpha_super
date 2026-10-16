"""Assemble the isolated TAA/NDX family review from source and saved evidence."""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as xml_module
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
from scripts.research.wired_signoff_20261015.taa_ndx_native_replay import VARIANT_DICT

OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results/review/wired_signoff_20261015"
TAA_VARIANT_LIST = [variant_str for variant_str, (family_str, _, _) in VARIANT_DICT.items() if family_str == "taa"]
NDX_VARIANT_LIST = [variant_str for variant_str, (family_str, _, _) in VARIANT_DICT.items() if family_str == "ndx"]


def finding_dict(identifier_str, priority_str, confidence_str, affected_list, issue_str, direction_str, impact_str,
        mitigation_str, source_list, evidence_list=None):
    return {"id_str": identifier_str, "priority_str": priority_str, "confidence_str": confidence_str,
        "affected_variant_list": affected_list, "issue_description_str": issue_str,
        "expected_bias_direction_str": direction_str, "impact_level_str": impact_str,
        "mitigation_str": mitigation_str, "source_list": source_list, "evidence_list": evidence_list or []}


def main() -> None:
    lag_path = OUTPUT_ROOT_PATH / "taa_ndx_dtb3_lag.json"
    lag_dict = json.loads(lag_path.read_text(encoding="utf-8"))
    capacity_path = OUTPUT_ROOT_PATH / "taa_ndx_btal_capacity.json"
    capacity_dict = json.loads(capacity_path.read_text(encoding="utf-8"))
    for study_dict in (lag_dict, capacity_dict):
        for row_dict in [*study_dict["input_manifest_list"], *study_dict["source_manifest_list"]]:
            if hashlib.sha256((REPO_ROOT_PATH / row_dict["path"]).read_bytes()).hexdigest() != row_dict["sha256"]:
                raise ValueError("Bounded study source or saved input changed; rerun the dependent saved-data diagnostic.")
    findings_list = [
        finding_dict("TAA-HELPER-01", "P1_before_unconditional_signal_signoff", "high_reproduced_actual_host",
            TAA_VARIANT_LIST,
            "A current execution-price endpoint does not prove VIX/SPY helper endpoints are current. Removing only the final VIX row can silently select the prior day's gate under the current month-end label. All three actual hosts accepted a synthetic 100% cash-to-fallback flip without a VIX source-date field.",
            "Either direction; silent different trade intent, not systematic alpha", "High when the missing helper changes the monthly fallback gate",
            "Before unconditional signal acceptance, validate/attest per-symbol observed month-end helper endpoints or explicitly approve a bounded prior-asof policy with visible source dates and warnings. A current index heartbeat alone is insufficient.",
            ["strategies/taa_df/strategy_taa_df_fallback_vix_cash_variant_utils.py:130-179", "alpha/live/strategy_host.py:1011-1020"],
            ["taa_ndx_adversarial_final.xml::test_actual_taa_host_can_flip_full_fallback_on_one_missing_vix_close (3 cases)"]),
        finding_dict("NDX-VXN-02", "P2_operator_visibility", "high_reproduced_actual_host",
            ["ndx_atr_vxn", "ndx_e2_atr", "ndx_e2_natr20"],
            "VXN is deliberately latest-prior-asof with no maximum age. Legacy host accepts a month-old VXN without an age warning. E2 also explicitly preserves prior-asof VXN; this is approved behavior, not proof of an invalid signal.",
            "Either direction; stale VXN changes exposure", "Medium; an old finite VXN can produce materially different NAV exposure",
            "Expose exact VXN observation date, trading-session age and the approved freshness policy in Doctor/dashboard and signoff. Keep a change to strict exact-date VXN as a separate owner decision.",
            ["strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py:175-197", "alpha/live/strategy_host.py:1208-1230", "docs/live/NDX_E2_WIRING_20261011.md:74"],
            ["taa_ndx_adversarial_final.xml::test_legacy_ndx_actual_host_accepts_month_old_vxn_without_warning", "tests/test_live_ndx_e2_freshness.py::test_fresh_endpoint_checks_preserve_padded_history_and_prior_asof_vxn"]),
        finding_dict("MONTHLY-SIZING-03", "P2_accept_explicit_execution_bridge", "high_source_and_tests",
            list(VARIANT_DICT),
            "Research fixes share quantities from Close_T NAV/price; monthly LIVE fixes weights at T but floors weight*actual broker NAV*budget/current pre-submit reference quote. Equal weight maps do not certify equal shares or P&L. The reference is not the future opening auction print.",
            "Either direction; rebalance turnover and overnight exposure differ", "Medium; exact share/P&L parity claims would be false",
            "Sign the weight/signal contract and the pre-open sizing bridge separately. Preserve stored Close_T targets, broker budget/reference quotes and resulting shares. Historical Sep28 TAA delta estimates are legacy evidence, not fresh $1M estimates.",
            ["strategies/taa_df/strategy_taa_df.py:493-506", "strategies/momentum/strategy_mo_atr_normalized_ndx.py:903-923", "alpha/live/execution_engine.py:286-319", "ASSUMPTIONS_AND_GAPS.md:263"]),
        finding_dict("MONTHLY-ACCOUNTING-04", "P2_comparison_interpretation", "high_source_verified",
            list(VARIANT_DICT),
            "Canonical vanilla runs with Dividend use the net dividend ledger (25% withholding, ex-date economic accrual). LIVE reference explicitly disables that ledger and remains price_return_ledger_v1. Broker settlements, interest and dividend timing therefore differ from both research and stored reference.",
            "Price-only reference generally understates a long portfolio versus net-dividend research; broker interest/timing create further drift",
            "Medium; false parity alarms or false confidence if accounting bases are mixed",
            "Show accounting contract version beside every chart. Separate weight/decision parity, price-only reference tracking, dividends, interest, fees and unexplained drift. Do not promote a P&L match to execution proof.",
            ["alpha/live/reference_compare.py:31,221-226", "alpha/engine/strategy.py:57,336-337", "ASSUMPTIONS_AND_GAPS.md:249-250"]),
        finding_dict("TAA-UNITS-05", "P2_research_measurement_limit", "high_source_plus_legacy_measurement",
            TAA_VARIANT_LIST,
            "TAA still floors adjusted-share counts and charges commissions in those ledger units; it does not opt into the historical-equivalent whole-share path now enabled for NDX. The September audit measured conservative TQQQ commission distortion of roughly +0.32/+0.38 percentage points/year after correction for rank/equal variants; that numerical estimate was not rerun here.",
            "Mostly conservative fees; rounding changes are not guaranteed one-directional",
            "Medium for historical TQQQ metrics; small for large present-day accounts does not remove historical unit mismatch",
            "Keep the current canonical backtest labeled with its actual share-unit policy. Treat a future TAA unit conversion as a separately scoped quantitative change with before/after results and dividend/corporate-action tests.",
            ["strategies/taa_df/strategy_taa_df.py:473,497", "alpha/engine/strategy.py:94-95", "strategies/momentum/strategy_mo_atr_normalized_ndx.py:720,912-919", "docs/research/STRATEGY_READINESS_AUDIT_20260928.md:98-104"]),
        finding_dict("E2-SECTORS-06", "P2_historical_claim_limit", "high_declared_contract",
            ["ndx_e2_atr", "ndx_e2_natr20"],
            "Sector caps use current-at-export GICS labels historically. Frozen map hashes make forward decisions auditable but cannot recreate historical sector knowledge, especially reclassifications. Four names per sector is at most40% intended NAV before gaps, not an exact live exposure cap.",
            "Possible anticipatory historical bias; magnitude not established by mechanics parity",
            "Medium for historical inference; high if historical PIT sector parity is claimed",
            "Retain the explicit current-label approximation and current-map reconstruction label. Sign forward plumbing separately from historical sector evidence. Use archived sector labels for a stronger historical claim.",
            ["ASSUMPTIONS_AND_GAPS.md:251", "docs/live/NDX_E2_WIRING_20261011.md:48,76-87", "strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap.py:91-143"]),
        finding_dict("MONTHLY-EXECUTION-07", "P2_forward_acceptance_and_capacity", "high_source_primary_provider_definition",
            list(VARIANT_DICT),
            "Norgate US Open is the first eligible regular consolidated-tape trade from any venue, not necessarily the primary exchange opening-auction price received by MOO. Canonical2.5bps/side is a cost assumption. No offline test establishes actual auction fills, partials or capacity. A compounded account is much larger than a constant$1M sleeve. TAA's measured historical and recent BTAL limitations are separately classified in TAA-BTAL-HISTORY-11 and TAA-BTAL-CURRENT-12.",
            "Unknown fill difference; omitted impact/partials can be optimistic",
            "High if backtest fills are represented as verified executable auction prices",
            "Use fresh native-Turnover capacity plus extra-cost diagnostics, then post-deployment read-only broker evidence for fill price, official auction reference where available, rejects/partials and net costs. Treat historical/current ADV separately and do not reuse the old1.8M cap as a current guarantee.",
            ["strategies/momentum/strategy_mo_atr_normalized_ndx.py:119-121", "strategies/taa_df/strategy_taa_df_fallback_variant_utils.py:99-102", "docs/research/STRATEGY_READINESS_AUDIT_20260928.md:100-103,454-458"],
            ["https://norgatedata.com/data-package-faq.php", "results/review/wired_signoff_20261015/audit_statistics/*/statistics.json"]),
        finding_dict("MONTHLY-SELECTION-08", "P2_investment_claim_limit", "high_research_records_not_new_inference",
            list(VARIANT_DICT),
            "TAA rank/TQQQ was selected from48 sibling variants after seeing the history. NDX has151 parameter configurations plus later ranking/filter/sector work (counts overlap; full unique family count is not reconstructible from these citations alone). Old NDX Reality Check p0.61 does not establish an incremental winner. Scout E2-cap selection verdict versus matched-exposure QQQ is UNCLEAR, and historical data through2026 are already exposed.",
            "Optimistic selected-winner inference; no untouched full-history holdout",
            "High for claims of proven edge, not a mechanics defect",
            "Disclose search exposure, keep all prior studies labeled by engine/cash/cost basis and avoid adding their trial counts blindly. This audit freezes7 existing identities and performs no optimization. Separate reproducibility signoff from expected-return/allocation judgment.",
            ["docs/research/STRATEGY_READINESS_AUDIT_20260928.md:104-105", "docs/research/NDX_PARAM_ROBUSTNESS_PREREG_20260926.md:105-106", "docs/research/NDX_PARAM_ROBUSTNESS_REPORT_20260926.md:83", "docs/research/MOMENTUM_DECISION_20261004.md:66-68,118-123"]),
        finding_dict("NDX-HOST-09", "P3_defense_in_depth", "high_reproduced_and_layered_guards_verified",
            ["ndx_atr", "ndx_atr_vxn"],
            "Legacy NDX raw host can return a past target execution date when invoked midmonth. This is NOT a demonstrated stale-order vulnerability: scheduler due/freshness, runner build/submit expiry and Doctor expiry guards block the ordinary path.",
            "None established through the guarded production runner", "Low after surrounding defenses; useful adapter contract hardening",
            "Optionally align raw legacy NDX host rejection with TAA/E2 and retain regression coverage. Do not label production execution unsafe solely from the raw-host probe.",
            ["alpha/live/strategy_host.py:1139-1240", "alpha/live/scheduler_utils.py:516-581", "alpha/live/runner.py:2888-2905,3348-3376", "alpha/live/doctor.py:1061-1077"],
            ["taa_ndx_adversarial_final.xml::test_legacy_ndx_actual_host_returns_past_execution_without_stale_guard (2 cases)"]),
        finding_dict("TAA-DTB3-10", "P2_accept_declared_publication_approximation", "high_fresh_frozen_input_sensitivity",
            ["taa_rank_tqqq", "taa_equal_tqqq"],
            "The cash hurdle uses current-vintage DTB3 observation dates without an explicit publication delay. G-015 accepts this approximation; a source comment describes latest-published availability more strongly than the data establish. A fresh one-XNYS-session delay changes0 of169 completed monthly target tables for EACH momentum variant. Closest margin is UUP September2025,0.2547basis points of monthly score. Linearity does not useDTB3.",
            "Potential either-direction threshold selection; no target changes under this bounded lag test",
            "Medium model-availability limitation; not the reproduced missing-VIX-row failure",
            "Retain G-015 as explicit owner-accepted model policy. Show current-vintage/assumed-publication status. The zero-flip result bounds only one exchange-session delay on these frozen bytes; it does not prove historical release timestamps or rule out revisions or longer delays.",
            ["strategies/taa_df/strategy_taa_df.py:183-190,294-307,315-325", "ASSUMPTIONS_AND_GAPS.md:G-015"],
            ["taa_ndx_dtb3_lag.json:169months per variant,0flips; weekend/MemorialDay/GoodFriday calendar controls; planted availability-crossing detected for both variants"]),
        finding_dict("TAA-BTAL-HISTORY-11", "P1_before_historical_1m_execution_cost_signoff", "high_reproduced_saved_order_evidence",
            TAA_VARIANT_LIST,
            "BTAL historical fills at$1M are not qualified by the2.5bps baseline. Fixed$1M executed-order maxima are16.934845x dailyADV for rank and10.283948x for equal on2017-07-03, and9.744140x for linearity on2017-04-03. The first pair uses only$19,471.50 preceding20session median turnover, against$329,746.84/$200,243.89 orders; linearity uses$20,765.50 against$202,341.94. Full BTAL order counts98/98/101 include24/24/29 missingADV orders, respectively. Known orders above100%ADV number5/6/3. Reproducible ledgers do not establish executable historical returns at this size.",
            "Historical net performance can be optimistic when complete low-cost fills are assumed at multiples of aggregate daily trading; exact drag is unmeasured",
            "High for signing historical net results as executable at$1M; independent of the verified monthly target mechanics",
            "Do not sign historical$1M execution costs or represented investable returns as qualified. Preserve reproducibility evidence with this blocker, qualify historical liquidity/auction-impact assumptions using a separately scoped capacity model or clearly restrict the performance claim. ETF secondary-marketADV is a warning diagnostic, not a hard impossibility bound; creation/redemption and underlying liquidity require separate evidence. Better recent liquidity does not repair old fills. Do not silently remove early years, optimize signals or change capital to make this pass.",
            ["ASSUMPTIONS_AND_GAPS.md:238", "scripts/research/wired_signoff_20261015/audit_statistics.py:214-249",
                "scripts/research/wired_signoff_20261015/taa_ndx_btal_capacity.py"],
            ["taa_ndx_btal_capacity.json:variant_result_dict.*.period_dict.full", "audit_statistics/strategy_taa_df_btal*/order_diagnostics.csv.gz"]),
        finding_dict("TAA-BTAL-CURRENT-12", "P2_before_current_1m_auction_capacity_acceptance", "high_reproduced_saved_order_evidence",
            TAA_VARIANT_LIST,
            "Recent liquidity is substantially better but is not$1M MOO cost qualification. Over trailing3years, fixed$1M BTAL order maxima are6.0573%/6.0574%/6.1792% of preceding20session median dailyADV (rank/equal/linearity);12/18,9/18 and9/15 observed orders exceed G-012's1% extrapolation threshold, with0missingADV. All3 have0BTAL executed orders in trailing1year. At2026-10-14's$6,013,174.75 median daily turnover, retrospective scaling of the3year order sizes gives maxima4.4313%/3.3412%/3.4035%; these are not current intended orders or opening-auction volume.",
            "Unknown auction impact/partial-fill bias; lower recent participation does not prove baseline2.5bps fills",
            "Medium before accepting a current$1M auction budget; high if described as already certified capacity",
            "Before the NEXT actual BTAL order, measure current intended notional and native-turnover date/coverage; flag every MOO order>1%dailyADV as extrapolation per G-012. Qualify account-specific opening fills, partials and TCA before scale. Keep fixed$1M, original compounded account, historical per-orderADV and endpoint scaling separate. Zero last-year trades provide no recent fill evidence.",
            ["ASSUMPTIONS_AND_GAPS.md:238", "scripts/research/wired_signoff_20261015/audit_statistics.py:288-327",
                "scripts/research/wired_signoff_20261015/taa_ndx_btal_capacity.py"],
            ["taa_ndx_btal_capacity.json:variant_result_dict.*.period_dict.last3y,last1y,current_endpoint_diagnostic_dict"]),
    ]
    test_path_list = [OUTPUT_ROOT_PATH / filename_str for filename_str in
        ("taa_ndx_focused_tests_verified.xml", "taa_ndx_quant_integrity_tests.xml", "taa_ndx_adversarial_final.xml")]
    test_node_set = set()
    for test_path in test_path_list:
        for case_obj in xml_module.parse(test_path).iter("testcase"):
            if all(case_obj.find(tag_str) is None for tag_str in ("failure", "error", "skipped")):
                test_node_set.add(f"{case_obj.get('classname')}::{case_obj.get('name')}")
    variant_review_list = []
    for variant_str, (family_str, import_str, _) in VARIANT_DICT.items():
        module_obj = import_module(import_str.split(":")[0])
        key_str = import_str.split(":")[0].rsplit(".", 1)[-1]
        baseline_path = OUTPUT_ROOT_PATH / "baselines" / key_str / "summary.json"
        baseline_dict = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
        replay_path = OUTPUT_ROOT_PATH / "taa_ndx_native" / f"{family_str}_actual_host_replay.json"
        replay_result_dict = json.loads(replay_path.read_text(encoding="utf-8"))["variant_result_dict"].get(variant_str) if replay_path.exists() else None
        variant_review_list.append({"variant_str": variant_str, "strategy_import_str": import_str,
            "audit_model_budget_float": 1_000_000, "config_dict": asdict(module_obj.DEFAULT_CONFIG),
            "baseline_result_dict": baseline_dict, "fresh_actual_host_replay_dict": replay_result_dict,
            "dtb3_publication_lag_sensitivity_dict": lag_dict["variant_result_dict"].get(variant_str),
            "btal_capacity_review_dict": capacity_dict["variant_result_dict"].get(variant_str),
            "finding_id_list": [row_dict["id_str"] for row_dict in findings_list if variant_str in row_dict["affected_variant_list"]],
            "research_verdict_str": ("historical_execution_cost_not_qualified_at_1m" if family_str == "taa" else "conditional_model_evidence_not_proof_of_edge"),
            "execution_verdict_str": ("conditional_weight_mechanics_verified" if replay_result_dict and replay_result_dict["mechanics_pass_bool"] else "pending_fresh_24_informative_month_host_replay"),
            "forward_broker_acceptance_str": "not_tested_no_broker_authority"})
    source_path_list = ["alpha/live/strategy_host.py", "alpha/live/ndx_e2_adapter.py", "alpha/live/execution_engine.py",
        "alpha/live/reference_compare.py", "strategies/taa_df/strategy_taa_df.py",
        "strategies/taa_df/strategy_taa_df_fallback_vix_cash_variant_utils.py",
        "strategies/taa_df/strategy_taa_df_btal_linearity.py", "strategies/taa_df/strategy_taa_df_btal_linearity_1n.py",
        "strategies/momentum/strategy_mo_atr_normalized_ndx.py", "strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py",
        "strategies/momentum/strategy_mo_natr20_ndx_vxn_scaled.py", "strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap.py",
        "strategies/momentum/strategy_mo_natr20_ndx_vxn_scaled_sector_cap.py"]
    artifact_path_list = [*test_path_list, *(REPO_ROOT_PATH / path_str for path_str in source_path_list)]
    artifact_path_list.extend(REPO_ROOT_PATH / (import_str.split(":")[0].replace(".", "/") + ".py")
        for _, import_str, _ in VARIANT_DICT.values())
    artifact_path_list.extend((REPO_ROOT_PATH / "results/review/e2_memory_20261014/refresh_snapshot_root/norgate_eod_ndx_e2_pit_plus_vxn_helper/2026-09-30") / name_str
        for name_str in ("manifest.json", "prices.parquet", "universe.parquet"))
    artifact_path_list.extend([OUTPUT_ROOT_PATH / "frozen_manifest.json", Path(__file__),
        Path(__file__).with_name("taa_ndx_native_replay.py"), Path(__file__).with_name("taa_ndx_test_audit.py"),
        Path(__file__).with_name("taa_ndx_taa_frozen_capture.py"), Path(__file__).with_name("taa_ndx_dtb3_lag.py"), lag_path,
        Path(__file__).with_name("taa_ndx_btal_capacity.py"), capacity_path])
    artifact_path_list.extend(REPO_ROOT_PATH / row_dict["path"] for row_dict in capacity_dict["input_manifest_list"])
    artifact_path_list.extend((OUTPUT_ROOT_PATH / "taa_ndx_native").glob("*.json"))
    result_dict = {"schema_version_str": "wired_signoff.review.v1", "generated_at_utc_str": datetime.now(UTC).isoformat(),
        "production_modified_bool": False, "broker_verified_bool": False, "audit_model_budget_float": 1_000_000,
        "findings_list": findings_list, "variant_review_list": variant_review_list,
        "mechanics_dict": {
            "timing_str": "T=actual final XNYS session of month; known Close_T inputs freeze intended weights; next month's first regular open fills in research; MOO lifecycle uses pre-submit broker quote/NAV to calculate live whole shares.",
            "taa_basket_list": ["GLD", "UUP", "TLT", "DBC", "BTAL"],
            "taa_momentum_formula_str": "m_i,T=(C_TR_i,T/C_TR_i,T-1month-1 + C_TR_i,T/C_TR_i,T-3months-1 + C_TR_i,T/C_TR_i,T-6months-1 + C_TR_i,T/C_TR_i,T-12months-1)/4; h_T=(1+DTB3_T/100)^(1/12)-1; keep slot iff m>h.",
            "taa_dtb3_policy_str": "G-015: current-vintage datedDTB3, no explicit publicationlag; not archived release evidence. Fresh frozen one-XNYS-session availability delay produces0/169 monthly target flips in BOTH rank/equal variants, with holiday controls and planted positive controls. Closest monthly score-hurdle margin0.2547bps atUUP September2025. Linearity has noDTB3 input.",
            "taa_btal_capacity_str": "Historical$1M cost signoff BLOCKED: fixed1M BTAL order maxima16.9348x/10.2839x/9.7441x dailyADV; fullhistory missingADV24/24/29orders. Recent3y fixed1M maxima6.0573%/6.0574%/6.1792%, stillG012>1%extrapolation. Zero last1y BTALtrades. EndpointADV6.013M is separate retrospective scaling context, not an intended auctionorder or historicalcost repair.",
            "taa_rank_weights_str": "Descending momentum slots5/15,4/15,3/15,2/15,1/15; failed slots sum intoTQQQ fallback. Equal version uses1/5 each.",
            "taa_linearity_formula_str": "For L in21,63,126,252: OLS logprice against0..L-1 gives slope beta and R2; R2adj=1-(1-R2)*(L-1)/(L-2); composite=mean(beta*R2adj). Keep each1/5 defensive slot iff composite>0, failed slots intoQQQ. No DTB3 threshold for this variant; negative adjustedR2 is retained by literal formula.",
            "taa_fallback_gate_str": "r_SPY,d=C_d/C_d-1-1; RV20=std_ddof0(last20 daily returns)*sqrt252*100; allow fallback iff RV20<VIX_asofT. Otherwise fallback weight becomes literal0%return cash. Defensive weights remain.",
            "ndx_universe_str": "Norgate historical Nasdaq100 PIT membership with exact tails; defaults history1999, backtest2000; SPY regime requires C_T>SMA200_T; stock requires C_T>SMA100_T; noROC>0 rule.",
            "ndx_atr_formula_str": "ROC12=C_monthend,T/C_monthend,T-12-1; TR_d=max(H-L,abs(H-C_prev),abs(L-C_prev)); ATR_nominal20=mean(TR_adjusted,last20)*UnadjustedClose_T/AdjustedClose_T; dollarATRscore=ROC12/ATR_nominal20; NATRscore=ROC12/(ATR_nominal20/UnadjustedClose_T). RawClose anchor rebases the entire adjusted ATR window, avoiding artificial rawsplit jumps.",
            "ndx_selection_str": "Rank eligible finite scores descending, symbol ascending for ties; take at most10; each10% before VXN. VXN scale=clip22/VXN_asofT,.25,1.0. BaseATR has noVXNscale. Missing slots/capped-out names are not renormalized and remaincash.",
            "ndx_e2_str": "Same parentrules with atmost4 names per currentGICSlevel1sector; UNKNOWN is one cappedbucket. Each component is an independent dedicatedaccount; the research50/50E2combination is not a third live strategy or automatic accounttransfer.",
            "prices_and_cash_str": "TAA return signals useTOTALRETURN; NDX price-level features/fills/marks useCAPITALSPECIAL with decision-timeUnadjustedClose anchors; benchmark$SPX label reads$SPXTR. Positivecash0%, unfinancednegativecash diagnostic; vanilla ordinarycashdividends net25%withholding, economicentitlementaccrual before nextopen.",
            "cost_formula_str": "Modeled fill per side is nextOpen*(1+sign(delta)*0.00025); commission=max1USD,0.005USD*abs(shares), subject to the actual adjusted/historical-unit policy. No partialfill or verifiedauctionimpact assumption.",
            "backtest_sizing_formula_str": "ResearchCloseTshares=trunc(NAV_CloseT*weight/CloseT) in the strategy's declared shareunits; fill atOpenTplus1. LIVEshares=floor(actualBrokerNAV_submit*budgetFraction*weight/referenceQuote_submit); orderdelta=target-actualBrokerHolding.",
        },
        "fresh_test_unique_passed_int": len(test_node_set), "fresh_test_nodeid_list": sorted(test_node_set),
        "test_xml_path_list": [str(path_obj.relative_to(REPO_ROOT_PATH)).replace("\\", "/") for path_obj in test_path_list],
        "quant_integrity_str": "Current-source offline tests: TAA monthly/VRP prefix and independent-symbol scale invariance; default21/63/126/252 linearity windows and plantedT+1 control12/12; dollarATR andNATR20 exact-row causal prefix and plantedT+1 controls12/12 each; E2 future rows, window equivalence, stale snapshot/map/endpoint failure tests.",
        "study_search_str": "No new strategy search;7 frozen existing identities. Historical families have at least48 TAA variants and151 NDX configurations plus overlapping later work. No fresh multiplicity-adjusted alpha claim.",
        "dtb3_lag_study_dict": lag_dict,
        "btal_capacity_study_dict": capacity_dict,
        "legacy_evidence_list": ["Sep28 TAA167/168 successful exact dates pervariant,1 error each; these are old artifacts.",
            "Sep28 legacyNDX: VXN51/54,baseATR23/26 successes; oldbase informative coverage was only12.",
            "Oct12 E2:30/30 decisions across15dates total2components; fewer than24dates percomponent.",
            "Scout/October research tables often credit idlecash atT-bills, unlike canonical0%cash; do not copy those metrics onto vanilla cards."],
        "not_proven_list": ["Real broker fills/funding/partials/capacity", "Current deployed revision, account settings or enabledrelease", "Historical PIT GICS labels", "Archived FRED publication vintages", "Exact quantity/P&L parity across Close_T research and pre-open broker sizing", "Untouched holdout alpha or a complete historical unique trial count"],
        "artifact_manifest_list": [{"path": str(path_obj.relative_to(REPO_ROOT_PATH)).replace("\\", "/"), "bytes": path_obj.stat().st_size,
            "sha256": hashlib.sha256(path_obj.read_bytes()).hexdigest()} for path_obj in sorted(set(artifact_path_list)) if path_obj.exists()]}
    output_path = OUTPUT_ROOT_PATH / "taa_ndx_review.json"
    output_path.write_text(json.dumps(result_dict, indent=2, default=str, allow_nan=False), encoding="utf-8")
    evidence_line_list = ["TAA / NDX current-source offline review", "",
        "Budget: USD1,000,000 initial model capital per strategy; no allocation/deployment/broker operation.",
        f"Fresh offline test union: {len(test_node_set)} unique passing cases, plus34 subtests in focused suite.",
        "Actual-host replay status (weight maps and exact decision/execution dates):"]
    for variant_dict in variant_review_list:
        replay_dict = variant_dict["fresh_actual_host_replay_dict"]
        evidence_line_list.append(f"- {variant_dict['variant_str']}: " + (
            f"{replay_dict['exact_int']}/{replay_dict['attempted_int']} exact; {replay_dict['informative_exact_int']} nonzero informative dates; USD1M modelbudget."
            if replay_dict else "pending fresh historical host replay; old evidence is not a substitute."))
    evidence_line_list.extend(["", "Mechanics:",
        "- Close_T inputs freeze intended weights; research fixes quantities at Close_T; live monthly execution floors actual brokerNAV*weight*budget/pre-submit quote before next month's first open.",
        "- TAA: five defensive ETFs, 1/3/6/12month TR momentum versus monthly DTB3 hurdle; rank or equal slots. Linearity uses mean adjustedR2*slope on logprices over21/63/126/252days versuszero. Failed slots go TQQQ/QQQ only when SPYRV20<VIX; otherwise0%cash.",
        "- NDX: PIT Nasdaq100, SPY>SMA200, stock>SMA100, top10 ROC12/nominalATR20 or ROC12/(ATR20/rawClose); VXNclip22/VXN,.25,1 scales10%slots. E2 permits atmost4names/currentGICSsector; missing slots remaincash.",
        "- Vanilla fills/marks CAPITALSPECIAL; benchmark$SPXTR; cost2.5bps/side+$0.005/share,min$1; ordinarydividends net25%; positivecash0%, negativecashfinancing unmodeled.",
        "", "Material conclusions:"])
    evidence_line_list.extend(f"- {finding_obj['id_str']} ({finding_obj['priority_str']}): {finding_obj['issue_description_str']}" for finding_obj in findings_list)
    evidence_line_list.extend(["", "Limits: E2 historical replay uses one frozen current-sector map and reconstructed historical metadata; it proves conditional mechanics, not historical PIT labels or native transport freshness. No actual broker execution, current VPS/account state, untouched holdout or exact research/live P&L equivalence is certified.",
        "Detailed formulas, parameters, sources, hashes, nodeids and per-variant conditions: taa_ndx_review.json."])
    (OUTPUT_ROOT_PATH / "taa_ndx_evidence.txt").write_text("\n".join(evidence_line_list) + "\n", encoding="utf-8")
    print(json.dumps({"output_path_str": str(output_path), "unique_tests_int": len(test_node_set),
        "variant_status_dict": {row_dict["variant_str"]: row_dict["execution_verdict_str"] for row_dict in variant_review_list}}))


if __name__ == "__main__":
    main()
