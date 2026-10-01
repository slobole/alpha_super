"""The research card (design section 11): one self-contained HTML page per family and run.

Order follows QUANT_PHILOSOPHY.md: conclusion first (grade and verdict strip), then the intuition (what each station
found, in a sentence), then the detail (tables and charts). Charts are inline PNG (matplotlib, Agg).

Grades (S7.6, D22): REJECTED only on a hard fail (evidence against the idea: here, a live configuration that loses
money at twice the costs); WATCHLIST on any soft fail or a failed gate; CANDIDATE when S4-S6 pass. A re-audition
card never demotes a pod by itself (section 10): the decision stays the owner's.
"""

from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alpha.scout.gate.deviations import ENGINE_DEVIATION_TUPLE

STYLE_STR = """
:root { --ink:#1d2433; --muted:#5b6475; --line:#d9dee7; --bg:#ffffff; --pass:#1e7b4f; --warn:#a86a00; --fail:#b3261e; --info:#3a5a8c; }
body { font-family: Segoe UI, Arial, sans-serif; color: var(--ink); background: var(--bg); max-width: 1100px; margin: 24px auto; padding: 0 16px; line-height: 1.45; }
h1 { font-size: 24px; margin-bottom: 4px; } h2 { font-size: 18px; margin-top: 32px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.sub { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px; } th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: right; }
th { background: #f4f6fa; } td.l, th.l { text-align: left; }
.grade { display: inline-block; padding: 6px 14px; border-radius: 6px; color: #fff; font-weight: 600; font-size: 18px; }
.PASS { color: var(--pass); font-weight: 600; } .WARN { color: var(--warn); font-weight: 600; } .FAIL { color: var(--fail); font-weight: 600; } .INFO { color: var(--info); }
.box { border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; background: #fafbfd; }
img { max-width: 100%; }
td.chosen { outline: 3px solid #1d2433; } td.live { font-weight: 700; text-decoration: underline; }
"""


def _png(fig) -> str:
    buffer_obj = io.BytesIO()
    fig.savefig(buffer_obj, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer_obj.getvalue()).decode()}">'


def _verdict_class(verdict_str: str) -> str:
    for key_str in ("PASS", "WARN", "FAIL", "INFO"):
        if verdict_str.startswith(key_str):
            return key_str
    return "INFO"


def _pct(value_float: float, digits_int: int = 1) -> str:
    return f"{value_float:+.{digits_int}%}" if np.isfinite(value_float) else "n/a"


def s3_check_list(bundle: dict) -> list:
    """S3 rows for the verdict strip. Class W: diagnostics (design S3: "the burden moves to S5 and S6"); class X:
    soft checks."""
    s3_dict = bundle.get("s3")
    if not s3_dict:
        return []
    row_list = list(s3_dict["result"]["check_list"]) + ([s3_dict["result"]["vix_gate"]["check"]] if "vix_gate" in s3_dict["result"] else [])
    if s3_dict["class_str"] == "W":
        return [(name_str, "INFO" if verdict_str != "PASS" else "PASS", detail_str + (" (diagnostic for class W)" if verdict_str != "PASS" else ""))
                for name_str, verdict_str, detail_str in row_list]
    return row_list


def grade_str(bundle: dict) -> str:
    s4, s5, s6 = bundle["s4"], bundle["s5"], bundle["s6"]
    # D22: a hard fail is evidence against the idea: a live configuration that loses money at twice the costs, or an
    # S3 hard criterion (class E / X) that fails (wrong sign, or significantly against).
    if s4.cost_dict["live"]["fail_bool"] or any(v == "FAIL" for _, v, _ in s3_check_list(bundle)):
        return "REJECTED"
    verdict_list = [v for _, v, _ in s3_check_list(bundle) + s4.check_list + s5.check_list + s6.check_list if v != "INFO"]
    if any(v.startswith(("FAIL", "WARN")) for v in verdict_list):
        return "WATCHLIST"
    # CANDIDATE means S0-S6 passed.
    return "CANDIDATE" if bundle.get("s3") else "CANDIDATE (S3 pending)"


def _surface_html(s4) -> str:
    """Grid Sharpe tables; for a 3-axis grid, one table per value of the first axis."""
    name_list, shape_tuple = s4.plateau_dict["name_list"], s4.plateau_dict["grid_shape_tuple"]
    sharpe_mat = s4.sharpe_ser.to_numpy().reshape(shape_tuple)
    label_mat = np.array(list(s4.sharpe_ser.index), dtype=object).reshape(shape_tuple)
    low_float, high_float = np.nanmin(sharpe_mat), np.nanmax(sharpe_mat)

    def cell(label_str, value_float) -> str:
        share_float = 0.0 if high_float == low_float else (value_float - low_float) / (high_float - low_float)
        colour_str = f"rgba(30,123,79,{0.08 + 0.55 * share_float:.2f})"
        class_list = (["chosen"] if label_str == s4.chosen_label_str else []) + (["live"] if label_str == s4.live_label_str else [])
        return f'<td class="{" ".join(class_list)}" style="background:{colour_str}">{value_float:.2f}</td>'

    def table(mat, labels, row_name, col_name, row_values, col_values, title_str="") -> str:
        head_str = "".join(f"<th>{html.escape(str(v))}</th>" for v in col_values)
        body_str = "".join(
            f"<tr><th class='l'>{html.escape(str(row_values[i]))}</th>" + "".join(cell(labels[i, j], mat[i, j]) for j in range(mat.shape[1])) + "</tr>"
            for i in range(mat.shape[0])
        )
        return f"<p class='sub'>{title_str}rows: {row_name}, columns: {col_name}</p><table><tr><th class='l'></th>{head_str}</tr>{body_str}</table>"

    grid_dict = s4.grid_param_dict
    if len(shape_tuple) == 2:
        return table(sharpe_mat, label_mat, name_list[0], name_list[1], grid_dict[name_list[0]], grid_dict[name_list[1]])
    return "".join(
        table(sharpe_mat[k], label_mat[k], name_list[1], name_list[2], grid_dict[name_list[1]], grid_dict[name_list[2]], f"{name_list[0]} = {grid_dict[name_list[0]][k]}; ")
        for k in range(shape_tuple[0])
    )


def render_card(bundle: dict) -> str:
    s4, s5, s6, post_dict = bundle["s4"], bundle["s5"], bundle["s6"], bundle["post_adoption"]
    s4.grid_param_dict = {k: tuple("-".join(map(str, v)) if isinstance(v, tuple) else v for v in values) for k, values in bundle["family"]["grid"].items()}
    pod_str, grade = bundle["pod_str"], grade_str(bundle)
    colour_str = {"CANDIDATE": "#1e7b4f", "WATCHLIST": "#a86a00", "REJECTED": "#b3261e"}[grade.split(" ")[0]]
    live_metric, post_metric = s4.metric_dict["live"]["in_sample"], s4.metric_dict["live"]["post_seal"]
    part_list = [f"<h1>{html.escape(pod_str)}: Scout re-audition card</h1>",
                 (f"<p class='sub'>Family {bundle['family']['family_id_str']} · retro registration · prior trials {bundle['prior_trial_count_int']} · "
                  f"in sample {live_metric['start_str']} to {live_metric['end_str']} (vault sealed at 2023-01-01) · research only, no live change</p>"),
                 f"<p><span class='grade' style='background:{colour_str}'>{grade}</span></p>"]

    strip_str = "".join(
        f"<tr><td class='l'>{station_str}</td><td class='l'>{html.escape(name_str)}</td><td class='l {_verdict_class(v)}'>{html.escape(v)}</td><td class='l'>{html.escape(d)}</td></tr>"
        for station_str, check_list in (("S3", s3_check_list(bundle)), ("S4", s4.check_list), ("S5", s5.check_list), ("S6", s6.check_list))
        for name_str, v, d in check_list
    )
    part_list.append(f"<h2>Verdict strip</h2><table><tr><th class='l'>Station</th><th class='l'>Check</th><th class='l'>Verdict</th><th class='l'>Key number</th></tr>{strip_str}</table>")
    part_list.append(
        "<div class='box'><b>In one paragraph.</b> "
        f"In sample the live configuration earned a Sharpe of {live_metric['sharpe_float']:.2f} ({_pct(live_metric['cagr_float'])} a year, "
        f"max drawdown {live_metric['max_drawdown_float']:.0%}) on its live month-end decision day; over the {len(s4.luck_dict['live']['sharpe_by_offset'])} decision days "
        f"of the luck band the median is <b>{s4.luck_dict['live']['median_float']:.2f}</b>, the number to plan with. It ranks {s4.live_dict['rank_int']} of {s4.live_dict['grid_size_int']} in its grid, "
        f"and the plateau rule {'chooses it' if s4.live_dict['is_chosen_bool'] else 'chooses ' + html.escape(s4.chosen_label_str)}. "
        + " ".join(f"MCPT {html.escape(c.name_str)}: p {c.p_value_float:.3f} ({c.verdict_str})." for c in s5.mcpt_list)
        + f" Net alpha after the fullest factor set: {_pct(s6.spanning_list[-1]['net_alpha_annual_float'])} a year (t {s6.spanning_list[-1]['net_alpha_t_float']:.2f}). "
        + (f"Since 2023 (seen, not a clean test): Sharpe {post_metric['sharpe_float']:.2f}. " if post_metric else "")
        + f"Since the current rule was adopted ({post_dict['adoption_date_str']}): {post_dict['sessions_int']} sessions; a live record would need about "
        f"{post_dict['min_track_record_months_float']:.0f} months to show, at 95% confidence, that the in-sample Sharpe is above zero.</div>"
    )

    # ---- S3
    if bundle.get("s3"):
        s3_result = bundle["s3"]["result"]
        part_list.append("<h2>S3: does the signal carry information?</h2>")
        part_list.append(f"<p>{html.escape(bundle['s3']['note_str'])}</p>")
        if "per_asset_list" in s3_result:
            asset_rows_str = "".join(f"<tr><td class='l'>{r['asset_str']}</td><td>{r['slope_float'] * 100:+.2f}%</td><td>{r['t_float']:.2f}</td><td>{r['months_int']}</td></tr>"
                                     for r in s3_result["per_asset_list"])
            part_list.append("<p>Per asset: next-month excess return per standard deviation of the score.</p>"
                             f"<table><tr><th class='l'>Asset</th><th>Slope / month</th><th>t</th><th>Months</th></tr>{asset_rows_str}</table>")
        if "vix_gate" in s3_result:
            gate = s3_result["vix_gate"]
            part_list.append(f"<p>Gate: next-month volatility {gate['vol_on_float']:.0%} when on ({gate['months_on_int']} months) vs {gate['vol_off_float']:.0%} when off "
                             f"({gate['months_off_int']} months); mean return {gate['mean_on_float']:+.2%} vs {gate['mean_off_float']:+.2%} a month.</p>")

    # ---- S4
    part_list.append("<h2>S4: the strategy in full reality</h2>")
    part_list.append(f"<p>Net Sharpe per configuration, in sample. Outlined: the plateau choice. <u>Underlined</u>: the live configuration. "
                     f"Plateau ratio {s4.plateau_dict['plateau_ratio_float']:.2f} ({s4.plateau_dict['verdict_str']}); peak {html.escape(s4.plateau_dict['peak_label_str'])} "
                     f"at {s4.plateau_dict['peak_sharpe_float']:.2f}.</p>")
    part_list.append(_surface_html(s4))
    luck = s4.luck_dict["live"]
    fig, axis = plt.subplots(figsize=(8, 2.6))
    axis.bar(list(luck["sharpe_by_offset"]), list(luck["sharpe_by_offset"].values()), color="#3a5a8c")
    axis.set_xlabel("decision sessions before month end"); axis.set_ylabel("Sharpe"); axis.set_title("Luck band (live configuration)")
    part_list.append(f"<p>Luck band: min {luck['min_float']:.2f}, median {luck['median_float']:.2f}, max {luck['max_float']:.2f}; the worst offset "
                     f"({luck['worst_offset_int']}) is the planning case for drawdowns and the S8 monitor.</p>" + _png(fig))
    cost_rows_str = "".join(
        f"<tr><td class='l'>{role}</td><td>{c['gross_sharpe_float']:.2f}</td><td>{c['net_sharpe_float']:.2f}</td><td>{c['stressed_sharpe_float']:.2f}</td>"
        f"<td>{'> 200' if not np.isfinite(c['breakeven_slippage_per_side_float']) else f"{c['breakeven_slippage_per_side_float'] * 1e4:.0f}"} bp</td><td>{c['annual_turnover_float']:.1f}x</td></tr>"
        for role, c in s4.cost_dict.items()
    )
    part_list.append("<table><tr><th class='l'>Config</th><th>Gross Sharpe</th><th>Net</th><th>2x costs + 10 bp</th><th>Breakeven slippage / side</th><th>Turnover / yr</th></tr>"
                     f"{cost_rows_str}</table>")
    metric_key_list = [("cagr_float", "CAGR", _pct), ("volatility_float", "Volatility", lambda v: f"{v:.1%}"), ("sharpe_float", "Sharpe", lambda v: f"{v:.2f}"),
                       ("excess_sharpe_float", "Excess Sharpe (T-bills)", lambda v: f"{v:.2f}"), ("sortino_float", "Sortino", lambda v: f"{v:.2f}"),
                       ("max_drawdown_float", "Max drawdown", lambda v: f"{v:.1%}"), ("calmar_float", "Calmar", lambda v: f"{v:.2f}"),
                       ("longest_underwater_days_int", "Longest underwater (sessions)", str), ("skew_float", "Skew", lambda v: f"{v:.2f}"),
                       ("expected_shortfall_95_float", "95% ES (daily)", lambda v: f"{v:.2%}"), ("positive_month_share_float", "Positive months", lambda v: f"{v:.0%}")]
    small, large = s4.account_dict["small"], s4.account_dict["institutional"]
    metric_rows_str = "".join(
        f"<tr><td class='l'>{label}</td><td>{fmt(live_metric.get(key, float('nan')))}</td><td>{fmt(post_metric[key]) if key in post_metric else ''}</td>"
        f"<td>{fmt(small.get(key, float('nan'))) if key in small else ''}</td><td>{fmt(large.get(key, float('nan'))) if key in large else ''}</td></tr>"
        for key, label, fmt in metric_key_list
    )
    part_list.append("<table><tr><th class='l'>Live configuration</th><th>In sample ($100K)</th><th>2023 on (seen)</th><th>$30K account</th><th>$10M account</th></tr>"
                     f"{metric_rows_str}</table>")
    year_str = "".join(f"<td>{_pct(v, 0)}</td>" for v in live_metric["year_return_dict"].values())
    part_list.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in live_metric["year_return_dict"]) + f"</tr><tr>{year_str}</tr></table>")
    streak = s4.streak_dict["live"]
    part_list.append(f"<p>Longest losing streak: {streak['longest_losing_months_int']} months (independent reorderings: median {streak['independent_median_int']}; p {streak['p_float']:.2f}).</p>")

    # ---- S5
    part_list.append("<h2>S5: is the search manufacturing winners?</h2>")
    for component in s5.mcpt_list:
        fig, axis = plt.subplots(figsize=(7, 2.4))
        axis.hist(component.null_score_vec, bins=40, color="#9aa7bd")
        axis.axvline(component.observed_float, color="#b3261e", linewidth=2)
        axis.set_title(f"MCPT, {component.name_str}: real {component.observed_float:.2f} (red) vs {len(component.null_score_vec)} permuted histories")
        part_list.append(f"<p><b>{html.escape(component.name_str)}</b>: <span class='{_verdict_class(component.verdict_str)}'>{component.verdict_str}</span>, "
                         f"p {component.p_value_float:.3f}. Null: {html.escape(component.null_str)}; score: {html.escape(component.score_str)}. {html.escape(component.note_str)}</p>" + _png(fig))
    dsr = s5.dsr_dict
    part_list.append(f"<p>Correlation-aware DSR (warn): plateau choice p {dsr['chosen']['p_value_float']:.3f}, live p {dsr['live']['p_value_float']:.3f}; "
                     f"{dsr['prior_trial_count_int']} prior trials counted as independent draws.</p>")
    wf = s5.walk_forward_dict
    oos_value_ser = (1 + wf["oos_return_ser"]).cumprod()
    fig, axis = plt.subplots(figsize=(8, 2.6))
    axis.plot(oos_value_ser.index, oos_value_ser.to_numpy(), color="#1d2433"); axis.set_title("Walk-forward, registered design: stitched out-of-sample value")
    design_rows_str = "".join(f"<tr><td class='l'>{r.design_str}</td><td>{r.oos_sharpe_float:.2f}</td><td>{r.efficiency_float:.2f}</td></tr>"
                              for r in wf["design_df"].itertuples() if np.isfinite(r.oos_sharpe_float))
    part_list.append(f"<p>Walk-forward (diagnostic): OOS Sharpe {wf['oos_sharpe_float']:.2f}, efficiency {wf['efficiency_float']:.2f}; "
                     f"{wf['positive_design_share_float']:.0%} of designs positive. PBO (diagnostic) {s5.pbo_dict['pbo_float']:.2f}.</p>" + _png(fig)
                     + f"<table><tr><th class='l'>Design</th><th>OOS Sharpe</th><th>Efficiency</th></tr>{design_rows_str}</table>")

    # ---- S6
    part_list.append("<h2>S6: value to the book</h2>")
    span_rows_str = "".join(
        f"<tr><td class='l'>{html.escape(r['model_str'])}</td><td>{_pct(r['gross_alpha_annual_float'])}</td><td>{r['gross_alpha_t_float']:.2f}</td>"
        f"<td>{_pct(r['net_alpha_annual_float'])}</td><td>{r['net_alpha_t_float']:.2f}</td>"
        f"<td class='l'>{', '.join(f'{k} {v:.2f}' for k, v in r['beta_dict'].items())}</td><td>{r['months_int']}</td></tr>"
        for r in s6.spanning_list
    )
    part_list.append("<p>Monthly excess returns on factor sets, Newey-West t (lag 3), over the months every factor exists (see the Months column: a "
                     "model with the other pod starts when both pods trade). Fama-French factors are not stored offline; ETF factors stand in. "
                     "TREND is 12-1 time-series momentum on SPY, EFA, EEM, IEF, TLT, GLD, DBC and UUP.</p>"
                     "<table><tr><th class='l'>Factors</th><th>Gross alpha / yr</th><th>t</th><th>Net alpha / yr</th><th>t</th><th class='l'>Betas</th><th>Months</th></tr>"
                     f"{span_rows_str}</table>")
    slot = s6.slot_dict
    part_list.append(f"<p>T-bill slot ({html.escape(slot['book_str'])}, from {slot['start_str']}): book Sharpe {slot['book_sharpe_with_float']:.2f} with the pod vs "
                     f"{slot['book_sharpe_tbills_float']:.2f} with T-bills in its slot; P(pod better) {slot['probability_better_float']:.2f} "
                     f"(<span class='{slot['verdict_str']}'>{slot['verdict_str']}</span>, bar 0.80). The engine's idle cash earns 0%; crediting T-bills "
                     "to it is a small correction (review: NDX P 0.47 to 0.49).</p>")
    div = s6.diversification_dict
    crisis_head_list = [k for k in div["crisis_list"][0] if k != "window_str"] if div["crisis_list"] else []
    crisis_rows_str = "".join("<tr><td class='l'>" + html.escape(r["window_str"]) + "</td>" + "".join(f"<td>{_pct(r[k], 0)}</td>" for k in crisis_head_list) + "</tr>" for r in div["crisis_list"])
    part_list.append("<p>Correlations (each on its own overlap): " + ", ".join(f"{k} {v:.2f} (from {div['correlation_start_dict'][k]})" for k, v in div["correlation_dict"].items())
                     + f". On the book's worst 5% of days the pod averaged {div['mean_on_book_worst_5pct_float']:.2%} (book {div['book_mean_on_worst_5pct_float']:.2%}).</p>")
    if crisis_rows_str:
        part_list.append("<table><tr><th class='l'>Crisis window</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in crisis_head_list) + f"</tr>{crisis_rows_str}</table>")
    cap = s6.capacity_dict
    part_list.append(f"<p>Capacity at 1% of 63-session ADV for the 95th-percentile order: ${cap['recent_3y_aum_float'] / 1e6:,.1f}M on the last three years' volumes "
                     f"(${cap['full_history_aum_float'] / 1e6:,.1f}M over the full history); binding asset {html.escape(cap['binding_asset_str'])}. "
                     "A simplified estimate, not the capacity v2 study.</p>")

    # ---- deviations, untested, post-adoption
    deviation_rows_str = "".join(f"<tr><td class='l'>{d.deviation_id_str}</td><td class='l'>{html.escape(d.expected_bias_direction_str)}</td><td class='l'>{html.escape(d.impact_level_str)}</td></tr>"
                                 for d in ENGINE_DEVIATION_TUPLE
                                 if not bundle.get("strategy_module_str") or bundle["strategy_module_str"].rsplit(".", 1)[-1] in d.affected_strategy_list
                                 or bundle["strategy_module_str"] in d.affected_strategy_list)
    part_list.append(f"<h2>Known engine deviations</h2><table><tr><th class='l'>Deviation</th><th class='l'>Bias</th><th class='l'>Impact</th></tr>{deviation_rows_str}</table>")
    part_list.append("<h2>What was not tested, and why</h2><ul>"
                     + ("" if bundle.get("s3") else "<li>S3 edge study: scheduled with P6.</li>")
                     + "<li>S7 vault: contaminated (2023 on was seen); replaced by the post-adoption period, too short to judge.</li>"
                     "<li>Fama-French factors (not stored offline); capacity v2 study (the estimate here is simplified).</li>"
                     "<li>The MCPT search is a fast replica of the engine (daily correlation 0.98-0.995, gross, constant weights within the month). Its plateau "
                     "pick can differ from the engine's (NDX: roc 15 vs roc 6); the verdicts were checked against every configuration's replica score.</li>"
                     "<li>Luck band: decision offsets 0-15 sessions before month end (offsets near the month length would skip months).</li></ul>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(pod_str)} Scout card</title><style>{STYLE_STR}</style></head><body>{''.join(part_list)}</body></html>"
