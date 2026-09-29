"""Assemble the fund product menu report (single self-contained HTML page).

Inputs: the frozen spec, books/ and inventory/ outputs, sleeve metadata, and the
narrative in report_text.yaml (written after the results, and clearly marked as
interpretation). Output: results/.../fund_product_menu_20260923/report/fund_product_menu.html
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import report_lib as rl  # noqa: E402

HERE_PATH = Path(__file__).resolve().parent
BOOK_DIR_PATH = common.STUDY_DIR_PATH / "books"
INVENTORY_DIR_PATH = common.STUDY_DIR_PATH / "inventory"
REPORT_DIR_PATH = common.STUDY_DIR_PATH / "report"
# The defensive chapter's numbers come from the follow-up study (scripts/research/simple_defensive_20260924/).
DEFENSIVE_SECTION_PATH = common.REPO_ROOT_PATH / "results" / "research" / "portfolio" / "simple_defensive_20260924" / "report_section.json"
# The capacity chapter's numbers come from scripts/research/defensive_capacity_20260924/.
CAPACITY_DIR_PATH = common.REPO_ROOT_PATH / "results" / "research" / "portfolio" / "defensive_capacity_20260924"
AMENDMENT_PATH = HERE_PATH / "amendments.yaml"

ENGINE_COLOR_VAR_DICT = {
    "MR_S": "--e-1", "MR_X": "--e-2", "MOM": "--e-3", "TAA": "--e-4",
    "MACRO": "--e-5", "BOND": "--e-6", "FLOW": "--e-7", "HEDGE": "--e-8",
}
PRODUCT_COLOR_VAR_LIST = ["--p-def", "--p-bal", "--p-gro", "--p-agg"]


def load_all() -> dict:
    spec_dict = yaml.safe_load((HERE_PATH / "frozen_spec.yaml").read_text(encoding="utf-8"))
    text_dict = yaml.safe_load((HERE_PATH / "report_text_static.yaml").read_text(encoding="utf-8"))
    text_dict.update(yaml.safe_load((HERE_PATH / "report_text_method.yaml").read_text(encoding="utf-8")))
    text_dict.update(yaml.safe_load((HERE_PATH / "report_text.yaml").read_text(encoding="utf-8")))
    data_dict = {
        "spec": spec_dict,
        "text": text_dict,
        "run": json.loads((BOOK_DIR_PATH / "run_summary.json").read_text(encoding="utf-8")),
        "metrics": pd.read_csv(BOOK_DIR_PATH / "headline_metrics.csv"),
        "weights": pd.read_csv(BOOK_DIR_PATH / "product_weights.csv"),
        "risk": pd.read_csv(BOOK_DIR_PATH / "product_risk_shares.csv"),
        "corr_products": pd.read_csv(BOOK_DIR_PATH / "product_correlations_daily.csv", index_col=0),
        "crisis_long": pd.read_csv(BOOK_DIR_PATH / "crisis_long_window.csv"),
        "crisis_exact": pd.read_csv(BOOK_DIR_PATH / "crisis_exact_window.csv"),
        "sub": pd.read_csv(BOOK_DIR_PATH / "subperiods_exact.csv"),
        "years_long": pd.read_csv(BOOK_DIR_PATH / "calendar_years_long.csv", index_col=0),
        "years_exact": pd.read_csv(BOOK_DIR_PATH / "calendar_years_exact.csv", index_col=0),
        "boot": pd.read_csv(BOOK_DIR_PATH / "bootstrap_exact.csv", index_col=0),
        "perturb": pd.read_csv(BOOK_DIR_PATH / "perturbation_exact.csv"),
        "split": pd.read_csv(BOOK_DIR_PATH / "estimation_split.csv"),
        "alt": pd.read_csv(BOOK_DIR_PATH / "alternatives_exact.csv"),
        "overlay": pd.read_csv(BOOK_DIR_PATH / "hedge_overlays_exact.csv"),
        "stress": pd.read_csv(BOOK_DIR_PATH / "stress_cases_exact.csv"),
        "drags": pd.read_csv(BOOK_DIR_PATH / "sleeve_stress_drags_exact.csv"),
        "gates": pd.read_csv(BOOK_DIR_PATH / "gates_product.csv", index_col=0),
        "gates_ladder": pd.read_csv(BOOK_DIR_PATH / "gates_ladder.csv"),
        "nav_exact": pd.read_csv(BOOK_DIR_PATH / "nav_exact.csv.gz", index_col=0, parse_dates=True),
        "nav_long": pd.read_csv(BOOK_DIR_PATH / "nav_long.csv.gz", index_col=0, parse_dates=True),
        "sleeve_full": pd.read_csv(INVENTORY_DIR_PATH / "sleeve_stats_full.csv", index_col=0),
        "sleeve_common": pd.read_csv(INVENTORY_DIR_PATH / "sleeve_stats_common.csv", index_col=0),
        "activity": pd.read_csv(INVENTORY_DIR_PATH / "sleeve_activity.csv", index_col=0),
        "corr_sleeves": pd.read_csv(INVENTORY_DIR_PATH / "corr_monthly_common.csv", index_col=0),
        "corr_stress": pd.read_csv(INVENTORY_DIR_PATH / "corr_stress_common.csv", index_col=0),
        "metadata": common.load_sleeve_metadata_dict(),
    }
    parity_path = common.STUDY_DIR_PATH / "parity" / "pm_parity.csv"
    data_dict["parity"] = pd.read_csv(parity_path) if parity_path.exists() else None
    data_dict["defensive"] = json.loads(DEFENSIVE_SECTION_PATH.read_text(encoding="utf-8")) if DEFENSIVE_SECTION_PATH.exists() else None
    if (CAPACITY_DIR_PATH / "auction_capacity_summary.csv").exists():
        data_dict["capacity"] = {
            "moo_orders": pd.read_csv(CAPACITY_DIR_PATH / "moo_order_sizes.csv"),
            "worked": pd.read_csv(CAPACITY_DIR_PATH / "capacity_summary.csv").query("window == 'recent_3y'").set_index("product"),
            "curve": pd.read_csv(CAPACITY_DIR_PATH / "capacity_curve.csv"),
            "wall": pd.read_csv(CAPACITY_DIR_PATH / "etf_concentration.csv").set_index("product"),
            "volume": pd.read_csv(CAPACITY_DIR_PATH / "etf_median_dollar_volume_by_year_musd.csv", index_col=0),
            "assets": json.loads((CAPACITY_DIR_PATH / "etf_assets_sources.json").read_text(encoding="utf-8")),
        }
    else:
        data_dict["capacity"] = None
    amendment_list = yaml.safe_load(AMENDMENT_PATH.read_text(encoding="utf-8"))["amendments"] if AMENDMENT_PATH.exists() else []
    data_dict["amendments"] = {a["product_id_str"]: a for a in amendment_list}
    return data_dict


def metric_row(data_dict: dict, window_str: str, series_str: str) -> pd.Series:
    frame_df = data_dict["metrics"]
    match_df = frame_df[(frame_df["window_str"] == window_str) & (frame_df["series_str"] == series_str)]
    if match_df.empty:
        raise KeyError(f"No metrics for {window_str}/{series_str}")
    return match_df.iloc[0]


def product_list(data_dict: dict) -> list[dict]:
    return [p for p in data_dict["spec"]["products"] if p.get("listed_bool", True)]


def engine_of_alias_dict(spec_dict: dict) -> dict[str, str]:
    mapping_dict = {}
    for engine_str, engine_dict in spec_dict["engines"].items():
        for key_str in ("primary_str", "secondary_str"):
            if engine_dict.get(key_str):
                mapping_dict.setdefault(engine_dict[key_str], engine_str)
    for alias_str, proxy_str in spec_dict.get("long_window_alias_swap", {}).items():
        mapping_dict.setdefault(proxy_str, mapping_dict.get(alias_str, "TAA"))
    mapping_dict.setdefault("trinity", "MACRO")
    return mapping_dict


def sleeve_name(data_dict: dict, alias_str: str) -> str:
    return data_dict["text"]["sleeve_names"].get(alias_str, alias_str)


# ─── sections ───────────────────────────────────────────────────────────────


def section_header(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    run_dict = data_dict["run"]
    return f"""
<header>
  <div class="eyebrow">{rl.esc(text_dict['eyebrow'])}</div>
  <h1>{rl.esc(text_dict['title'])}</h1>
  <p class="lede">{text_dict['lede']}</p>
  <div class="meta-row">
    <span><strong>Exact window</strong> {rl.esc(run_dict['exact_window'][0])} → {rl.esc(run_dict['exact_window'][1])}</span>
    <span><strong>Long window (proxies)</strong> {rl.esc(run_dict['long_window'][0])} → {rl.esc(run_dict['long_window'][1])}</span>
    <span><strong>Sleeves reviewed</strong> 25 (9 WIRED, 16 PM_READY)</span>
    <span><strong>Spec frozen</strong> <span class="mono">{rl.esc(run_dict['spec_sha256_str'][:12])}</span></span>
    <span><strong>Built</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}</span>
  </div>
  <nav class="toc" aria-label="Sections">{text_dict['toc_html']}</nav>
</header>"""


def pods_label_html(data_dict: dict, product_id_str: str) -> str:
    weight_df = data_dict["weights"]
    rows_df = weight_df[(weight_df["product_id_str"] == product_id_str) & weight_df["weight_float"].notna()]
    rows_df = rows_df.sort_values("weight_float", ascending=False)
    engine_map_dict = engine_of_alias_dict(data_dict["spec"])
    part_list = []
    for row in rows_df.itertuples():
        color_var_str = ENGINE_COLOR_VAR_DICT[engine_map_dict[row.alias_str]]
        part_list.append(
            f'<li><span class="swatch" style="background:var({color_var_str})"></span>'
            f'<span>{rl.esc(sleeve_name(data_dict, row.alias_str))}</span><strong>{rl.pct(row.weight_float, 0)}</strong></li>'
        )
    return f'<ul class="pod-list">{"".join(part_list)}</ul>'



def min_account_float(data_dict: dict, product_id_str: str) -> float:
    """Smallest book at which every pod clears its own funding floor (DV2/HPI: $25K)."""
    floor_dict = data_dict["text"]["pod_floor_usd"]
    weight_df = data_dict["weights"]
    rows_df = weight_df[(weight_df["product_id_str"] == product_id_str) & weight_df["weight_float"].notna()]
    need_list = [floor_dict.get(r.alias_str, floor_dict["default"]) / r.weight_float for r in rows_df.itertuples()]
    return float(max(need_list)) if need_list else float("nan")


def amendment_badge_html(data_dict: dict, product_id_str: str) -> str:
    """A post-freeze owner amendment is labelled on the product card itself."""
    note_str = data_dict["text"].get("amendment_card_notes", {}).get(product_id_str)
    if product_id_str not in data_dict["amendments"] or not note_str:
        return ""
    return f'<p class="amend"><span class="pill amended">Amended</span> {note_str}</p>'


def section_menu(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    card_by_line_dict = {"main": [], "low_touch": []}
    for position_int, product_dict in enumerate(product_list(data_dict)):
        product_id_str = product_dict["product_id_str"]
        exact_row = metric_row(data_dict, "exact_annual", product_id_str)
        long_row = metric_row(data_dict, "long_annual", product_id_str)
        color_var_str = product_dict.get("color_var_str", PRODUCT_COLOR_VAR_LIST[min(position_int, 3)])
        card_by_line_dict[product_dict["line_str"]].append(f"""
<article class="menu-card" aria-labelledby="card-{rl.esc(product_id_str)}">
  <div class="rung"><span class="bar" style="background:var({color_var_str})"></span>{rl.esc(product_dict['rung_label_str'])}</div>
  <h3 id="card-{rl.esc(product_id_str)}">{rl.esc(product_dict['display_name_str'])}</h3>
  <p class="job">{rl.esc(text_dict['product_jobs'][product_id_str])}</p>
  {amendment_badge_html(data_dict, product_id_str)}
  <dl>
    <dt>CAGR, {rl.esc(data_dict['run']['exact_window'][0][:4])}–26</dt><dd>{rl.pct(exact_row['cagr_float'])}</dd>
    <dt>Volatility</dt><dd>{rl.pct(exact_row['volatility_float'])}</dd>
    <dt>Max drawdown</dt><dd>{rl.pct(exact_row['max_drawdown_float'])}</dd>
    <dt>Max drawdown incl. 2008 (proxy)</dt><dd>{rl.pct(long_row['max_drawdown_float'])}</dd>
    <dt>Worst 12 months</dt><dd>{rl.pct(exact_row['worst_12m_float'])}</dd>
    <dt>Sharpe (rf 0 / over T-bills)</dt><dd>{rl.num(exact_row['sharpe_rf0_float'])} / {rl.num(exact_row['sharpe_excess_float'])}</dd>
    <dt>Beta to S&amp;P 500</dt><dd>{rl.num(exact_row['beta_spx_float'])}</dd>
    <dt>Minimum account (est.)</dt><dd>{rl.money(min_account_float(data_dict, product_id_str))}</dd>
  </dl>
  <div class="pods">{pods_label_html(data_dict, product_id_str)}</div>
</article>""")
    return f"""
<section id="menu">
  <h2>The menu</h2>
  <div class="prose">{text_dict['menu_intro_html']}</div>
  <h3 class="line-head">Main line <span class="note">· all seven engines</span></h3>
  <div class="menu-cards cols-3">{''.join(card_by_line_dict['main'])}</div>
  <h3 class="line-head">Low-touch line <span class="note">· monthly-style trading, no closing-auction orders</span></h3>
  <div class="menu-cards cols-4">{''.join(card_by_line_dict['low_touch'])}</div>
  <p class="note">{text_dict['pod_floor_note_html']}</p>
  {text_dict['recommendation_html']}
</section>"""


def headline_table_html(data_dict: dict, window_str: str, include_legacy_bool: bool) -> str:
    header_list = ["Book", "CAGR", "over T-bills", "Volatility", "Sharpe", "Sharpe over T-bills", "Max DD", "Longest under water",
                   "Worst year", "Worst 12m", "Beta", "Down capture"]
    row_list, class_list = [], []

    def add(series_str: str, label_str: str, window_key_str: str, class_str: str = "") -> None:
        row = metric_row(data_dict, window_key_str, series_str)
        row_list.append([
            rl.esc(label_str), rl.pct(row["cagr_float"]), rl.pct(row["cagr_float"] - row["tbill_cagr_float"], signed_bool=True),
            rl.pct(row["volatility_float"]), rl.num(row["sharpe_rf0_float"]), rl.num(row["sharpe_excess_float"]),
            rl.pct(row["max_drawdown_float"]), f"{row['longest_underwater_days_int'] / 30.44:.0f} mo",
            rl.pct(row["worst_year_float"]), rl.pct(row["worst_12m_float"]), rl.num(row["beta_spx_float"]),
            rl.pct(row["down_capture_float"], 0),
        ])
        class_list.append(class_str)

    product_window_str = "exact_annual" if window_str == "exact" else "long_annual"
    bench_window_str = "exact_benchmark" if window_str == "exact" else "long_benchmark"
    for product_dict in product_list(data_dict):
        add(product_dict["product_id_str"], product_dict["display_name_str"], product_window_str)
    first_bench_bool = True
    for bench_str in ("S&P 500 TR", "60/40", "T-bills"):
        add(bench_str, bench_str, bench_window_str, "bench sep" if first_bench_bool else "bench")
        first_bench_bool = False
    if include_legacy_bool:
        first_legacy_bool = True
        for legacy_id_str, label_str in data_dict["text"]["legacy_labels"].items():
            add(legacy_id_str, label_str, "exact_legacy", "legacy sep" if first_legacy_bool else "legacy")
            first_legacy_bool = False
    return rl.table(header_list, row_list, class_list)


def composition_payload(data_dict: dict) -> dict:
    risk_df = data_dict["risk"]
    engine_map_dict = engine_of_alias_dict(data_dict["spec"])
    row_list = []
    for product_dict in product_list(data_dict):
        product_id_str = product_dict["product_id_str"]
        share_df = risk_df[risk_df["product_id_str"] == product_id_str].copy()
        share_df["engine_str"] = share_df["alias_str"].map(engine_map_dict)
        capital_ser = share_df.groupby("engine_str")["start_weight_float"].sum()
        risk_ser = share_df.groupby("engine_str")["variance_share_float"].sum().clip(lower=0.0)
        for label_str, value_ser in ((f"{product_dict['display_name_str']} · capital", capital_ser), (f"{product_dict['display_name_str']} · risk", risk_ser)):
            row_list.append({
                "label": label_str,
                "segments": [{"key": e, "value": float(value_ser.get(e, 0.0))} for e in ENGINE_COLOR_VAR_DICT if value_ser.get(e, 0.0) > 0],
            })
    key_list = [
        {"key": e, "label": data_dict["spec"]["engines"][e]["label_str"], "colorVar": v}
        for e, v in ENGINE_COLOR_VAR_DICT.items() if e in data_dict["spec"]["engines"] and e != "HEDGE"
    ]
    return {"rows": row_list, "keys": key_list}


def line_legend_html(products: list[dict]) -> str:
    legend_list = []
    for product_dict in products:
        legend_list.append(f'<span class="k"><span class="line" style="border-color:var({product_dict["color_var_str"]})"></span>{rl.esc(product_dict["display_name_str"])}</span>')
    legend_list.append('<span class="k"><span class="line" style="border-color:var(--b-spx)"></span>S&amp;P 500 TR</span>')
    legend_list.append('<span class="k"><span class="line dashed" style="border-color:var(--b-6040)"></span>60/40</span>')
    return f'<div class="legend">{"".join(legend_list)}</div>'


def section_performance(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    products = product_list(data_dict)
    run_dict = data_dict["run"]
    figure_list = []
    for line_str, line_label_str in (("main", "Main line"), ("low_touch", "Low-touch line")):
        line_products = [p for p in products if p["line_str"] == line_str]
        legend_html = line_legend_html(line_products)
        figure_list.append(f"""
  <h3>{rl.esc(line_label_str)}</h3>
  <figure>
    <figcaption><strong>Growth of $1, {rl.esc(run_dict['exact_window'][0])} → {rl.esc(run_dict['exact_window'][1])}</strong> · log scale · annual reset to target weights · net of the engine's modeled costs</figcaption>
    {legend_html}
    <div id="chart-nav-exact-{line_str}" class="viz"></div>
  </figure>
  <figure>
    <figcaption><strong>Drawdown from the running peak, same window</strong></figcaption>
    {legend_html}
    <div id="chart-dd-exact-{line_str}" class="viz"></div>
  </figure>
  <figure>
    <figcaption><strong>Growth of $1 including 2008, {rl.esc(run_dict['long_window'][0])} → {rl.esc(run_dict['long_window'][1])}</strong> · the 3x BTAL TAA is replaced for the whole period by its 2x no-BTAL sibling at the same weight, the only version with 2008 history · log scale</figcaption>
    {legend_html}
    <div id="chart-nav-long-{line_str}" class="viz"></div>
  </figure>""")
    return f"""
<section id="performance">
  <h2>How the products behaved</h2>
  <div class="prose">{text_dict['performance_intro_html']}</div>
  {''.join(figure_list)}
  <h3>Numbers on the exact window</h3>
  {headline_table_html(data_dict, 'exact', include_legacy_bool=True)}
  <p class="note">{text_dict['headline_table_note_html']}</p>
  <h3>Numbers including 2008 (proxy window)</h3>
  {headline_table_html(data_dict, 'long', include_legacy_bool=False)}
  <figure>
    <figcaption><strong>Risk and return, exact window</strong> · filled dots are the new products, rings the current ladder books, squares the benchmarks</figcaption>
    <div class="legend"><span class="k"><span class="dot" style="background:var(--p-bal)"></span>new product</span><span class="k"><span class="ring" style="border-color:var(--legacy)"></span>current ladder book</span><span class="k"><span class="sq" style="background:var(--b-spx)"></span>benchmark</span></div>
    <div id="chart-scatter" class="viz"></div>
  </figure>
</section>"""


def section_crises(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    products = product_list(data_dict)
    crisis_df = data_dict["crisis_long"]
    header_list = ["Episode", "Dates"] + [p["display_name_str"] for p in products] + ["S&P 500 TR", "60/40"]
    row_list = []
    for _, row in crisis_df.iterrows():
        cell_list = [f'<td class="l episode">{rl.esc(row["episode_str"])}</td>', f'<td class="l nowrap small">{rl.esc(row["start_str"])} → {rl.esc(row["end_str"])}</td>']
        for product_dict in products:
            cell_list.append(rl.heat_cell(row.get(product_dict["product_id_str"], np.nan)))
        cell_list.append(rl.heat_cell(row.get("S&P 500 TR", np.nan)))
        cell_list.append(rl.heat_cell(row.get("60/40", np.nan)))
        row_list.append(cell_list)
    return f"""
<section id="crises">
  <h2>Stress episodes</h2>
  <div class="prose">{text_dict['crises_intro_html']}</div>
  {rl.table(header_list, row_list, table_class_str='heat', left_column_count_int=2)}
  <p class="note">{text_dict['crises_note_html']}</p>
</section>"""


DEFENSIVE_LABEL_DICT = {"DISP": "Dispersion", "FI": "Tactical FI", "DOWNSHOCK": "Sector downshock"}


def month_label(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%b %Y")


def corr_cell_html(value_float: float) -> str:
    css_str = "h-p3" if value_float >= 0.7 else "h-p2" if value_float >= 0.5 else "h-p1" if value_float >= 0.3 else "h-0" if value_float > -0.1 else "h-n1"
    return f'<td class="h {css_str}">{rl.num(value_float, 2)}</td>'


def section_defensive(data_dict: dict) -> str:
    """The defensive shelf: few-pod alternatives to the menu's defensive rung, their crises, tails and correlations,
    and the owner's post-freeze amendment to Low-touch Defensive. Numbers: simple_defensive_20260924/report_section.json."""
    payload_dict = data_dict["defensive"]
    if payload_dict is None:
        return ""
    text_dict = data_dict["text"]
    book_by_id_dict = {b["id"]: b for b in payload_dict["books"]}

    main_id_list = ["core5", "core5_btal", "core5_btal_fi", "core5_btal_disp", "core5_btal_down", "core5_btalspy", "lt_def", "def", "ladder_1"]
    main_header_list = ["Book", "Pods", "Min. account", "Trade days / yr", "CAGR", "Sharpe", "Sharpe over T-bills", "Max DD",
                        "Max DD incl. 2008", "CVaR 5% (21 days)", "Worst crisis", "Crisis corr. to S&amp;P"]
    main_row_list, main_class_list = [], []
    for book_id_str in main_id_list:
        book = book_by_id_dict[book_id_str]
        highlight_bool = book_id_str in ("core5_btal", "lt_def")
        main_row_list.append([
            f'<td class="l nowrap">{"<strong>" if highlight_bool else ""}{rl.esc(book["name"])}{"</strong>" if highlight_bool else ""}</td>',
            f"{book['pods']}", rl.money(book["min_account"]), f"{book['trade_days']:.0f}",
            rl.pct(book["cagr"]), rl.num(book["sharpe"]), rl.num(book["sharpe_excess"]), rl.pct(book["maxdd"]), rl.pct(book["maxdd_long"]),
            rl.pct(book["cvar5_21d"]), rl.pct(book["worst_crisis"]), rl.num(book["corr_spx_crisis"]),
        ])
        main_class_list.append("sep" if book_id_str == "lt_def" else ("legacy" if book["group"] == "reference" else ""))

    crisis_id_list = ["core5", "core5_btal", "core5_btal_fi", "core5_btal_disp", "lt_def", "def"]
    crisis_header_list = ["Episode", "Dates"] + [rl.esc(book_by_id_dict[i]["name"]) for i in crisis_id_list] + ["S&amp;P 500 TR"]
    crisis_row_list = []
    for episode_dict in payload_dict["episodes"]:
        label_str = episode_dict["label"]
        cell_list = [f'<td class="l episode">{rl.esc(label_str)}</td>', f'<td class="l nowrap small">{rl.esc(episode_dict["start"])} → {rl.esc(episode_dict["end"])}</td>']
        cell_list += [rl.heat_cell(book_by_id_dict[i]["crises"][label_str]) for i in crisis_id_list]
        cell_list.append(rl.heat_cell(payload_dict["spx_crises"][label_str]))
        crisis_row_list.append(cell_list)

    corr_dict = payload_dict["pod_corr"]
    label_list = [DEFENSIVE_LABEL_DICT.get(label_str, label_str) for label_str in corr_dict["labels"]]
    corr_row_list = []
    for i, label_a_str in enumerate(label_list):
        cell_list = [rl.esc(label_a_str)]
        for j in range(len(label_list)):
            if i == j:
                cell_list.append('<td class="h h-na">·</td>')
            elif j > i:
                cell_list.append(corr_cell_html(corr_dict["all"][i][j]))
            else:
                cell_list.append(corr_cell_html(corr_dict["stress"][i][j]))
        corr_row_list.append(cell_list)

    sleeve_row_list = []
    proxy_note_html = ' <span class="note">(proxy before Oct 2012)</span>'
    for sleeve_dict in payload_dict["sleeves_2008"]:
        sleeve_row_list.append([
            f'<td class="l nowrap">{rl.esc(DEFENSIVE_LABEL_DICT.get(sleeve_dict["name"], sleeve_dict["name"]))}{proxy_note_html if sleeve_dict["proxy_before_2012"] else ""}</td>',
            f'<span class="pill {"wired" if sleeve_dict["tier"] == "wired" else ""}">{rl.esc(sleeve_dict["tier"].upper())}</span>',
            rl.heat_cell(sleeve_dict["y2008"]), rl.heat_cell(sleeve_dict["gfc"]), rl.pct(sleeve_dict["maxdd"]),
            f'<td class="nowrap small">{rl.esc(month_label(sleeve_dict["maxdd_peak"]))} → {rl.esc(month_label(sleeve_dict["maxdd_trough"]))}</td>',
        ])

    variant_header_list = ["Low-touch Defensive", "Pods", "CAGR", "Sharpe", "Sharpe incl. 2008", "Max DD", "Max DD incl. 2008",
                           "CVaR 5% (21 days)", "Worst crisis", "Crisis corr. to S&amp;P"]
    variant_row_list, variant_class_list = [], []
    for variant_dict in payload_dict["lt_def_variants"]:
        chosen_bool = variant_dict["id"] == "amended"
        variant_row_list.append([
            f'<td class="l">{"<strong>" if chosen_bool else ""}{rl.esc(variant_dict["name"])}{"</strong>" if chosen_bool else ""}</td>',
            f"{variant_dict['pods']}", rl.pct(variant_dict["cagr"]), rl.num(variant_dict["sharpe"]), rl.num(variant_dict["sharpe_long"]),
            rl.pct(variant_dict["maxdd"]), rl.pct(variant_dict["maxdd_long"]), rl.pct(variant_dict["cvar5_21d"]), rl.pct(variant_dict["worst_crisis"]),
            rl.num(variant_dict["corr_spx_crisis"]),
        ])
        variant_class_list.append("")
    lt_weight_df = data_dict["weights"]
    lt_weight_df = lt_weight_df[(lt_weight_df["product_id_str"] == "LT_DEF") & lt_weight_df["weight_float"].notna()].sort_values("weight_float", ascending=False)
    weights_html = " · ".join(f"{rl.esc(sleeve_name(data_dict, row.alias_str))} {rl.pct(row.weight_float, 0)}" for row in lt_weight_df.itertuples())

    search_row_list = []
    for row_dict in sorted(payload_dict["search"], key=lambda r: (r["pods"], -r["sharpe_excess"])):
        search_row_list.append([
            f'<td class="l">{rl.esc(row_dict["book"].replace("DOWNSHOCK", "Sector downshock").replace("DISP", "Dispersion").replace("FI", "Tactical FI"))}</td>',
            f"{row_dict['pods']}", rl.pct(row_dict["cagr"]), rl.num(row_dict["sharpe"]), rl.num(row_dict["sharpe_excess"]), rl.pct(row_dict["maxdd_long"]),
            f'<span class="pill {"pass" if row_dict["qualifies"] else "fail"}">{"yes" if row_dict["qualifies"] else "no"}</span>',
        ])
    return f"""
<section id="defensive">
  <h2>The defensive shelf</h2>
  <div class="prose">{text_dict['defensive_intro_html']}</div>
  {text_dict['defensive_verdict_html']}
  <h3>Pods against protection</h3>
  {rl.table(main_header_list, main_row_list, main_class_list, table_class_str='wrap-head', left_column_count_int=1)}
  <p class="note">{text_dict['defensive_table_note_html']}</p>
  <h3>Crisis by crisis</h3>
  <div class="prose">{text_dict['defensive_crises_intro_html']}</div>
  {rl.table(crisis_header_list, crisis_row_list, table_class_str='heat wrap-head', left_column_count_int=2)}
  <p class="note">{text_dict['defensive_crises_note_html']}</p>
  <h3>How the pods move together</h3>
  <p class="prose">{text_dict['defensive_corr_intro_html']}</p>
  {rl.table([""] + [rl.esc(label_str) for label_str in label_list], corr_row_list, table_class_str='heat')}
  <p class="note">{text_dict['defensive_corr_note_html']}</p>
  <h3>The 2008 test</h3>
  <div class="prose">{text_dict['defensive_2008_html']}</div>
  {rl.table(["Sleeve", "Tier", "2008 calendar year", "S&amp;P 500 peak to trough, May 2008 to Mar 2009", "Worst drawdown since Mar 2008", "When"], sleeve_row_list, table_class_str='wrap-head', left_column_count_int=1)}
  <h3 id="amendment-a1">Amendment A1: Inflation Compass leaves Low-touch Defensive</h3>
  <div class="prose">{text_dict['defensive_amendment_html']}</div>
  {rl.table(variant_header_list, variant_row_list, variant_class_list, table_class_str='wrap-head', left_column_count_int=1)}
  <p class="note">Amended weights: {weights_html}. {text_dict['defensive_amendment_note_html']}</p>
  <details>
    <summary>All 23 simple books from the search, and which passed the pre-declared rules</summary>
    {rl.table(["Book (equal capital, annual reset)", "Pods", "CAGR", "Sharpe", "Sharpe over T-bills", "Max DD incl. 2008", "Passed"], search_row_list, left_column_count_int=1)}
    <p class="note">{text_dict['defensive_search_note_html']}</p>
  </details>
</section>"""


def section_capacity(data_dict: dict) -> str:
    """How much money the defensive products can take, by execution method, on today's liquidity."""
    capacity_dict = data_dict["capacity"]
    if capacity_dict is None:
        return ""
    text_dict = data_dict["text"]
    moo_df, worked_df, wall_df = capacity_dict["moo_orders"], capacity_dict["worked"], capacity_dict["wall"]

    def usd(value_float: float) -> str:
        if value_float >= 1e9:
            return f"${value_float / 1e9:.1f}B"
        if value_float >= 1e6:
            return f"${value_float / 1e6:,.0f}M" if value_float >= 1e7 or float(value_float / 1e6).is_integer() else f"${value_float / 1e6:.1f}M"
        return f"${value_float / 1e3:,.0f}K"

    def cell(recommended_obj, outer_obj=None) -> str:
        recommended_str = usd(recommended_obj) if not rl.is_missing(recommended_obj) else "under $25K"
        if outer_obj is None:
            return recommended_str
        outer_str = usd(outer_obj) if not rl.is_missing(outer_obj) else "–"
        return f'{recommended_str} <span class="note">({outer_str})</span>'

    product_list_str = ["CORE5 alone", "CORE5 + BTAL_QQQ", "Low-touch Defensive (A1)", "Defensive (main line)"]
    moo_row_list = []
    for product_str in product_list_str:
        other_row = moo_df[(moo_df["product"] == product_str) & (moo_df["group"] == "everything else")].iloc[0]
        thin_row = moo_df[(moo_df["product"] == product_str) & (moo_df["group"].str.startswith("thin"))].iloc[0]
        moo_row_list.append([
            f'<td class="l nowrap">{rl.esc(product_str.replace(" (A1)", ""))}</td>',
            usd(other_row["aum_max_hits_1%"]), usd(thin_row["aum_median_hits_1%"]), usd(thin_row["aum_p95_hits_5%"]),
            f'{usd(thin_row["aum_max_hits_5%"])} <span class="note">({rl.esc(thin_row["largest_order_ticker"])})</span>',
            usd(thin_row["aum_max_hits_10%"]),
        ])
    moo_header_list = ["Product", "Other ETFs and stocks: largest order at 1%", "Thin three: typical order at 1%", "Thin three: 95th-percentile order at 5%",
                       "Thin three: largest order at 5%", "…at 10%"]

    wall_column_dict = {"BTAL": "aum_at_10pct_of_btal_musd", "UUP": "aum_at_10pct_of_uup_musd", "DBC": "aum_at_10pct_of_dbc_musd"}
    row_list = []
    for product_str in product_list_str:
        worked_row = worked_df.loc[product_str]
        if product_str in wall_df.index:
            wall_ser = pd.Series({k: wall_df.at[product_str, c] for k, c in wall_column_dict.items()}).dropna()
            wall_str = f"{usd(wall_ser.min() * 1e6)} <span class=\"note\">({wall_ser.idxmin()})</span>"
        else:
            wall_str = "–"
        row_list.append([
            f'<td class="l nowrap">{rl.esc(product_str.replace(" (A1)", ""))}</td>',
            cell(worked_row["recommended__one_day"], worked_row["outer__one_day"]),
            cell(worked_row["recommended__up_to_5_days"], worked_row["outer__up_to_5_days"]),
            cell(worked_row["recommended__5_days_wrappers_as_blocks"]),
            wall_str,
        ])
    header_list = ["Product", "Worked over the day", "Worked over up to 5 days", "…and thin three as blocks", "Wall: 10% of an ETF"]

    volume_df = capacity_dict["volume"]
    assets_dict = capacity_dict["assets"]
    holds_dict = {"BTAL": "long/short basket of US stocks", "UUP": "US dollar index futures", "DBC": "commodity futures"}
    last_year_dict = {t: float(volume_df.loc[volume_df.index.max(), t]) for t in holds_dict}
    def volume_usd(value_musd_float: float) -> str:
        return f"${value_musd_float:.2f}M" if value_musd_float < 1.0 else f"${value_musd_float:.0f}M"

    etf_row_list = []
    for ticker_str, holds_str in holds_dict.items():
        etf_row_list.append([
            f'<td class="l nowrap"><strong>{ticker_str}</strong></td>', f'<td class="l">{rl.esc(holds_str)}</td>',
            volume_usd(volume_df.at[2013, ticker_str]), volume_usd(volume_df.at[2019, ticker_str]), volume_usd(volume_df.at[2025, ticker_str]),
            volume_usd(last_year_dict[ticker_str]), usd(assets_dict[f"{ticker_str}_net_assets_usd"]),
        ])
    return f"""
<section id="capacity">
  <h2>Capacity of the defensive products</h2>
  <div class="prose">{text_dict['capacity_intro_html']}</div>
  {text_dict['capacity_verdict_html']}
  <h3>Market-on-open: product size at which orders reach a share of daily volume</h3>
  {rl.table(moo_header_list, moo_row_list, table_class_str='wrap-head', left_column_count_int=1)}
  <p class="note">{text_dict['capacity_moo_note_html']}</p>
  <h3>Largest product size with worked orders</h3>
  {rl.table(header_list, row_list, table_class_str='wrap-head', left_column_count_int=1)}
  <p class="note">{text_dict['capacity_table_note_html']}</p>
  <h3>The three thin ETFs</h3>
  {rl.table(["ETF", "Holds", "Daily volume 2013", "2019", "2025", f"{int(volume_df.index.max())} so far", "Fund assets"], etf_row_list, table_class_str='wrap-head', left_column_count_int=2)}
  <p class="note">{text_dict['capacity_etf_note_html']}</p>
</section>"""


def section_years(data_dict: dict) -> str:
    products = product_list(data_dict)
    years_df = data_dict["years_long"]
    header_list = ["Year"] + [p["display_name_str"] for p in products] + ["S&P 500 TR", "60/40", "T-bills"]
    row_list = []
    exact_start_year_int = int(data_dict["run"]["exact_window"][0][:4])
    for year_int, row in years_df.iterrows():
        label_str = f"{year_int}" + ("*" if year_int < exact_start_year_int or year_int == exact_start_year_int else "")
        cell_list = [label_str]
        for product_dict in products:
            cell_list.append(rl.heat_cell(row.get(product_dict["product_id_str"], np.nan)))
        for bench_str in ("S&P 500 TR", "60/40", "T-bills"):
            cell_list.append(rl.heat_cell(row.get(bench_str, np.nan)))
        row_list.append(cell_list)
    return f"""
<details>
  <summary>Calendar-year returns (long window)</summary>
  {rl.table(header_list, row_list, table_class_str='heat')}
  <p class="note">{data_dict['text']['years_note_html']}</p>
</details>"""


def section_construction(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    spec_dict = data_dict["spec"]
    products = product_list(data_dict)
    weight_df = data_dict["weights"]
    risk_df = data_dict["risk"]
    engine_map_dict = engine_of_alias_dict(spec_dict)
    alias_order_list = []
    for engine_str in ENGINE_COLOR_VAR_DICT:
        for alias_str, mapped_str in engine_map_dict.items():
            if mapped_str == engine_str and alias_str in set(weight_df.loc[weight_df["weight_float"].notna(), "alias_str"]) and alias_str not in alias_order_list:
                alias_order_list.append(alias_str)
    header_list = ["Sleeve"] + [p["display_name_str"] for p in products]
    row_list = []
    for alias_str in alias_order_list:
        engine_str = engine_map_dict[alias_str]
        cell_list = [
            f'<td class="l nowrap"><span class="swatch" style="background:var({ENGINE_COLOR_VAR_DICT[engine_str]})"></span>{rl.esc(sleeve_name(data_dict, alias_str))}</td>',
        ]
        for product_dict in products:
            match_df = weight_df[(weight_df["product_id_str"] == product_dict["product_id_str"]) & (weight_df["alias_str"] == alias_str)]
            risk_match_df = risk_df[(risk_df["product_id_str"] == product_dict["product_id_str"]) & (risk_df["alias_str"] == alias_str)]
            if match_df.empty or pd.isna(match_df.iloc[0]["weight_float"]):
                cell_list.append('<span class="note">–</span>')
            else:
                risk_float = risk_match_df.iloc[0]["variance_share_float"] if not risk_match_df.empty else np.nan
                cell_list.append(f"<strong>{rl.pct(match_df.iloc[0]['weight_float'], 0)}</strong> <span class=\"note\">({rl.pct(max(risk_float, 0.0), 0)} of risk)</span>")
        row_list.append(cell_list)
    dial_row_list = []
    for product_dict in products:
        result_dict = data_dict["run"]["products"][product_dict["product_id_str"]]
        risk_df = data_dict["risk"]
        own_df = risk_df[risk_df["product_id_str"] == product_dict["product_id_str"]]
        engine_map_dict = engine_of_alias_dict(spec_dict)
        stabilizer_risk_float = own_df[own_df["alias_str"].map(lambda a: spec_dict["engines"][engine_map_dict[a]]["bucket_str"] == "stabilizer")]["variance_share_float"].sum()
        dial_row_list.append([
            f'<td class="l nowrap">{rl.esc(product_dict["display_name_str"])}</td>',
            rl.esc("Main" if product_dict["line_str"] == "main" else "Low-touch"),
            rl.pct(product_dict.get("target_volatility_float"), 1) if product_dict.get("target_volatility_float") else "none: return engines only",
            rl.pct(1.0 - sum(w for a, w in result_dict["weight_dict"].items() if spec_dict["engines"][engine_map_dict[a]]["bucket_str"] == "stabilizer"), 0),
            rl.pct(sum(w for a, w in result_dict["weight_dict"].items() if spec_dict["engines"][engine_map_dict[a]]["bucket_str"] == "stabilizer"), 0),
            rl.pct(max(stabilizer_risk_float, 0.0), 0),
            rl.pct(result_dict["ex_ante_volatility_float"], 1),
            rl.pct(product_dict["max_drawdown_budget_float"], 0),
            f"{len(result_dict['weight_dict'])}",
        ])
    return f"""
<section id="construction">
  <h2>What is inside each product</h2>
  <div class="prose">{text_dict['construction_intro_html']}</div>
  <figure>
    <figcaption><strong>Capital versus risk, by engine</strong> · capital = starting weights; risk = each engine's share of the book's realised variance on the exact window (annual reset)</figcaption>
    <div class="legend">{''.join(f'<span class="k"><span class="sw" style="background:var({v})"></span>{rl.esc(spec_dict["engines"][e]["label_str"])}</span>' for e, v in ENGINE_COLOR_VAR_DICT.items() if e in spec_dict["engines"] and e != "HEDGE")}</div>
    <div id="chart-composition" class="viz"></div>
  </figure>
  {rl.table(header_list, row_list, left_column_count_int=1)}
  <p class="note">Bold = starting capital weight; in brackets = that sleeve's share of the product's realised risk on the exact window.</p>
  <h3>How each product's weights were set</h3>
  {rl.table(["Product", "Line", "Volatility target", "Return engines (capital)", "Stabilizers (capital)", "Stabilizers (share of risk)", "Ex-ante volatility", "Drawdown budget", "Pods"], dial_row_list, left_column_count_int=2)}
  <div class="prose">{text_dict['construction_after_html']}</div>
</section>"""


def section_method(data_dict: dict) -> str:
    return f"""
<section id="method">
  <h2>How the menu was built</h2>
  {data_dict['text']['method_html']}
</section>"""


def section_building_blocks(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    spec_dict = data_dict["spec"]
    full_df = data_dict["sleeve_full"]
    common_df = data_dict["sleeve_common"]
    activity_df = data_dict["activity"]
    engine_map_dict = engine_of_alias_dict(spec_dict)
    role_dict = text_dict["sleeve_roles"]
    header_list = ["Sleeve", "Tier", "Engine / role in menu", "History from", "CAGR", "Vol", "Sharpe", "Max DD", "Beta",
                   "CAGR 2012+", "Max DD 2012+", "Trade days / yr", "Idle cash", "Evidence note"]
    row_list = []
    order_list = text_dict["sleeve_order"]
    for alias_str in order_list:
        full_row = full_df.loc[alias_str]
        common_row = common_df.loc[alias_str] if alias_str in common_df.index else None
        activity_row = activity_df.loc[alias_str]
        tier_str = data_dict["metadata"][alias_str]["tier_str"]
        engine_str = engine_map_dict.get(alias_str)
        swatch_html = f'<span class="swatch" style="background:var({ENGINE_COLOR_VAR_DICT[engine_str]})"></span>' if engine_str else '<span class="swatch" style="background:var(--base)"></span>'
        row_list.append([
            f'<td class="l nowrap">{swatch_html}{rl.esc(sleeve_name(data_dict, alias_str))}</td>',
            f'<span class="pill {"wired" if tier_str == "wired" else ""}">{rl.esc(tier_str.upper())}</span>',
            f'<td class="l wrap-cell">{rl.esc(role_dict.get(alias_str, ""))}</td>',
            rl.esc(full_row["first_invested_date_str"]),
            rl.pct(full_row["cagr_float"]), rl.pct(full_row["volatility_float"]), rl.num(full_row["sharpe_rf0_float"]),
            rl.pct(full_row["max_drawdown_float"]), rl.num(full_row["beta_spx_float"]),
            rl.pct(common_row["cagr_float"]) if common_row is not None else "–",
            rl.pct(common_row["max_drawdown_float"]) if common_row is not None else "–",
            f"{activity_row['trade_days_per_year_float']:.0f}",
            rl.pct(activity_row["mean_positive_cash_weight_float"], 0),
            f'<td class="l wrap-cell small">{rl.esc(text_dict["evidence_notes"].get(alias_str, ""))}</td>',
        ])
    corr_df = data_dict["corr_sleeves"]
    rep_list = [a for a in text_dict["correlation_sleeves"] if a in corr_df.index]
    corr_header_list = [""] + [rl.esc(text_dict["short_names"].get(a, a)) for a in rep_list]
    corr_row_list = []
    for alias_a in rep_list:
        cell_list = [rl.esc(sleeve_name(data_dict, alias_a))]
        for alias_b in rep_list:
            value_float = corr_df.loc[alias_a, alias_b]
            if alias_a == alias_b:
                cell_list.append('<td class="h h-na">·</td>')
            else:
                css_str = "h-p3" if value_float >= 0.7 else "h-p2" if value_float >= 0.5 else "h-p1" if value_float >= 0.3 else "h-0" if value_float > -0.1 else "h-n1"
                cell_list.append(f'<td class="h {css_str}">{rl.num(value_float, 2)}</td>')
        corr_row_list.append(cell_list)
    return f"""
<section id="building-blocks">
  <h2>The building blocks</h2>
  <div class="prose">{text_dict['blocks_intro_html']}</div>
  {text_dict['engine_table_html']}
  <details open>
    <summary>All 25 sleeves: standalone record, cadence and evidence</summary>
    {rl.table(header_list, row_list, left_column_count_int=1)}
    <p class="note">{text_dict['sleeve_table_note_html']}</p>
  </details>
  <h3>How the chosen sleeves move together</h3>
  <p class="prose">{text_dict['correlation_intro_html']}</p>
  {rl.table(corr_header_list, corr_row_list, table_class_str='heat')}
  <p class="note">{text_dict['correlation_note_html']}</p>
</section>"""


def section_trust(data_dict: dict) -> str:
    text_dict = data_dict["text"]
    products = product_list(data_dict)
    boot_df = data_dict["boot"]
    gates_df = data_dict["gates"]
    perturb_df = data_dict["perturb"]
    sub_df = data_dict["sub"]
    alt_df = data_dict["alt"]
    stress_df = data_dict["stress"]
    overlay_df = data_dict["overlay"]
    split_df = data_dict["split"]
    ladder_df = data_dict["gates_ladder"]

    boot_header_list = ["Book", "CAGR, 90% range", "Sharpe, 90% range", "Max drawdown, 90% range", "P(beats T-bills)", "P(Sharpe > 60/40)"]
    boot_row_list = []
    prob_col_str = [c for c in boot_df.columns if c.startswith("prob_sharpe_above_")][0]
    for series_str, label_str in [(p["product_id_str"], p["display_name_str"]) for p in products] + [("S&P 500 TR", "S&P 500 TR"), ("60/40", "60/40")]:
        row = boot_df.loc[series_str]
        boot_row_list.append([
            rl.esc(label_str),
            f"{rl.pct(row['cagr_p05_float'])} to {rl.pct(row['cagr_p95_float'])}",
            f"{rl.num(row['sharpe_p05_float'])} to {rl.num(row['sharpe_p95_float'])}",
            f"{rl.pct(row['maxdd_p05_float'])} to {rl.pct(row['maxdd_p95_float'])}",
            rl.pct(row["prob_excess_cagr_positive_float"], 0),
            rl.pct(row[prob_col_str], 0) if series_str != "60/40" else "–",
        ])
    perturb_header_list = ["Product", "Draws", "Sharpe, 90% range", "CAGR, 90% range", "Max drawdown, 90% range", "Within drawdown budget", "Chosen book's Sharpe percentile"]
    perturb_row_list = []
    for product_dict in products:
        sub_perturb_df = perturb_df[perturb_df["product_id_str"] == product_dict["product_id_str"]]
        gate_row = gates_df.loc[product_dict["product_id_str"]]
        perturb_row_list.append([
            rl.esc(product_dict["display_name_str"]), f"{len(sub_perturb_df)}",
            f"{rl.num(sub_perturb_df['sharpe_rf0_float'].quantile(0.05))} to {rl.num(sub_perturb_df['sharpe_rf0_float'].quantile(0.95))}",
            f"{rl.pct(sub_perturb_df['cagr_float'].quantile(0.05))} to {rl.pct(sub_perturb_df['cagr_float'].quantile(0.95))}",
            f"{rl.pct(sub_perturb_df['max_drawdown_float'].quantile(0.05))} to {rl.pct(sub_perturb_df['max_drawdown_float'].quantile(0.95))}",
            rl.pct(gate_row["perturbation_share_within_dd_budget_float"], 0),
            rl.pct(gate_row["base_sharpe_percentile_in_perturbation_float"], 0),
        ])
    sub_header_list = ["Book"] + [f"Part {i}: {sub_df[sub_df['part_int'] == i]['start_str'].iloc[0][:7]} → {sub_df[sub_df['part_int'] == i]['end_str'].iloc[0][:7]}" for i in (1, 2, 3)]
    sub_row_list = []
    for series_str, label_str in [(p["product_id_str"], p["display_name_str"]) for p in products] + [("60/40", "60/40"), ("S&P 500 TR", "S&P 500 TR")]:
        cell_list = [rl.esc(label_str)]
        for part_int in (1, 2, 3):
            match_df = sub_df[(sub_df["series_str"] == series_str) & (sub_df["part_int"] == part_int)]
            row = match_df.iloc[0]
            cell_list.append(f"{rl.pct(row['cagr_float'])} <span class=\"note\">Sharpe {rl.num(row['sharpe_rf0_float'])} · DD {rl.pct(row['max_drawdown_float'])}</span>")
        sub_row_list.append(cell_list)
    alt_header_list = ["Product", "Construction", "CAGR", "Vol", "Sharpe", "Max DD", "ES 95% 21d"]
    alt_row_list = []
    for product_dict in products:
        product_id_str = product_dict["product_id_str"]
        base_row = metric_row(data_dict, "exact_annual", product_id_str)
        alt_row_list.append([rl.esc(product_dict["display_name_str"]), '<td class="l"><strong>chosen template</strong></td>', rl.pct(base_row["cagr_float"]), rl.pct(base_row["volatility_float"]),
                             rl.num(base_row["sharpe_rf0_float"]), rl.pct(base_row["max_drawdown_float"]), rl.pct(base_row["es95_21d_float"])])
        for alt_row in alt_df[alt_df["product_id_str"] == product_id_str].itertuples():
            alt_row_list.append(["", f'<td class="l">{rl.esc(text_dict["alternative_labels"].get(alt_row.alternative_str, alt_row.alternative_str))}</td>',
                                 rl.pct(alt_row.cagr_float), rl.pct(alt_row.volatility_float), rl.num(alt_row.sharpe_rf0_float),
                                 rl.pct(alt_row.max_drawdown_float), rl.pct(alt_row.es95_21d_float)])
    stress_header_list = ["Product", "Modeled", "Costs + financing stress", "Cash earns T-bill − 0.5%"]
    stress_row_list = []
    for product_dict in products:
        product_id_str = product_dict["product_id_str"]
        base_row = metric_row(data_dict, "exact_annual", product_id_str)
        stress_row = stress_df[(stress_df["product_id_str"] == product_id_str) & (stress_df["case_str"] == "stress")].iloc[0]
        upside_row = stress_df[(stress_df["product_id_str"] == product_id_str) & (stress_df["case_str"] == "cash_upside")].iloc[0]
        stress_row_list.append([
            rl.esc(product_dict["display_name_str"]),
            f"{rl.pct(base_row['cagr_float'])} <span class=\"note\">Sharpe {rl.num(base_row['sharpe_rf0_float'])}</span>",
            f"{rl.pct(stress_row['cagr_float'])} <span class=\"note\">Sharpe {rl.num(stress_row['sharpe_rf0_float'])}</span>",
            f"{rl.pct(upside_row['cagr_float'])} <span class=\"note\">Sharpe {rl.num(upside_row['sharpe_rf0_float'])}</span>",
        ])
    overlay_header_list = ["Product", "Hedge", "Variant", "CAGR", "Vol", "Sharpe", "Max DD", "ES 95% 21d"]
    overlay_row_list = []
    for row in overlay_df.itertuples():
        overlay_row_list.append([
            rl.esc(next(p["display_name_str"] for p in products if p["product_id_str"] == row.product_id_str) if any(p["product_id_str"] == row.product_id_str for p in products) else row.product_id_str),
            rl.esc(sleeve_name(data_dict, row.overlay_alias_str)), rl.esc("with 10% hedge" if row.variant_str == "with" else "without (same window)"),
            rl.pct(row.cagr_float), rl.pct(row.volatility_float), rl.num(row.sharpe_rf0_float), rl.pct(row.max_drawdown_float), rl.pct(row.es95_21d_float),
        ])
    split_header_list = ["Product", "Weights estimated on", "CAGR", "Vol", "Sharpe", "Max DD", "Largest weight change vs chosen"]
    split_row_list = []
    weight_df = data_dict["weights"]
    for product_dict in products:
        product_id_str = product_dict["product_id_str"]
        chosen_ser = weight_df[(weight_df["product_id_str"] == product_id_str) & weight_df["weight_float"].notna()].set_index("alias_str")["weight_float"]
        for half_str, label_str in (("first_half", "first half only"), ("second_half", "second half only")):
            half_df = split_df[(split_df["product_id_str"] == product_id_str) & (split_df["estimation_str"] == half_str)]
            if half_df.empty:
                continue
            half_ser = half_df.set_index("alias_str")["weight_float"]
            max_change_float = (half_ser.reindex(chosen_ser.index.union(half_ser.index)).fillna(0) - chosen_ser.reindex(chosen_ser.index.union(half_ser.index)).fillna(0)).abs().max()
            first_row = half_df.iloc[0]
            split_row_list.append([rl.esc(product_dict["display_name_str"]), label_str, rl.pct(first_row["book_cagr_float"]), rl.pct(first_row["book_volatility_float"]),
                                   rl.num(first_row["book_sharpe_float"]), rl.pct(first_row["book_max_drawdown_float"]), f"{max_change_float * 100:.0f} pts"])
    gate_header_list = ["Product", "G1 drawdown budget (exact / with 2008)", "G2 beats T-bills", "G5 weight plateau", "G6 positive in every third"]
    gate_row_list = []

    def pill(flag_obj) -> str:
        if flag_obj is None or (isinstance(flag_obj, float) and np.isnan(flag_obj)):
            return '<span class="pill">n/a</span>'
        flag_bool = str(flag_obj).lower() in ("true", "1")
        return f'<span class="pill {"pass" if flag_bool else "fail"}">{"pass" if flag_bool else "fail"}</span>'

    for product_dict in products:
        gate_row = gates_df.loc[product_dict["product_id_str"]]
        gate_row_list.append([
            rl.esc(product_dict["display_name_str"]),
            f"{pill(gate_row['G1_drawdown_within_budget_exact_bool'])} / {pill(gate_row['G1_drawdown_within_budget_long_bool'])}",
            pill(gate_row["G2_beats_tbills_bool"]), pill(gate_row["G5_plateau_bool"]), pill(gate_row["G6_positive_excess_every_third_bool"]),
        ])
    ladder_row_list = []
    for row in ladder_df.itertuples():
        lower_label_str = next((p["display_name_str"] for p in data_dict["spec"]["products"] if p["product_id_str"] == row.lower_str), row.lower_str)
        upper_label_str = next((p["display_name_str"] for p in data_dict["spec"]["products"] if p["product_id_str"] == row.upper_str), row.upper_str)
        ladder_row_list.append([
            f"{rl.esc(lower_label_str)} → {rl.esc(upper_label_str)}",
            pill(row.G3_monotone_bool) if isinstance(row.G3_monotone_bool, (bool, np.bool_)) or str(row.G3_monotone_bool) in ("True", "False") else '<span class="pill">cross-line</span>',
            rl.num(row.daily_corr_float), pill(row.G4_distinct_bool),
        ])
    return f"""
<section id="trust">
  <h2>How much to trust these numbers</h2>
  <div class="prose">{text_dict['trust_intro_html']}</div>
  <h3>Pre-declared gates</h3>
  {rl.table(gate_header_list, gate_row_list)}
  {rl.table(["Adjacent pair", "G3 ladder order", "Daily correlation", "G4 distinct (< 0.95)"], ladder_row_list)}
  <p class="note">{text_dict['gates_note_html']}</p>
  <h3>Luck: resampled history</h3>
  <div class="prose">{text_dict['bootstrap_intro_html']}</div>
  <figure>
    <figcaption><strong>Sharpe ratio, 90% bootstrap range</strong> · dot = measured on the actual history · dashed line = 60/40 measured</figcaption>
    <div id="chart-boot" class="viz"></div>
  </figure>
  {rl.table(boot_header_list, boot_row_list)}
  <h3>Judgment: what if the splits were different?</h3>
  <div class="prose">{text_dict['perturbation_intro_html']}</div>
  {rl.table(perturb_header_list, perturb_row_list)}
  <h3>Time: three equal sub-periods</h3>
  {rl.table(sub_header_list, sub_row_list)}
  <h3>Estimation: weights from half the data</h3>
  {rl.table(split_header_list, split_row_list)}
  <h3>Simpler constructions on the same sleeves</h3>
  <div class="prose">{text_dict['alternatives_intro_html']}</div>
  {rl.table(alt_header_list, alt_row_list)}
  <h3>Frictions the engine does not charge, and cash it does not credit</h3>
  <div class="prose">{text_dict['stress_intro_html']}</div>
  {rl.table(stress_header_list, stress_row_list)}
  <h3>Adding a crisis hedge</h3>
  <div class="prose">{text_dict['overlay_intro_html']}</div>
  {rl.table(overlay_header_list, overlay_row_list, left_column_count_int=3)}
  <h3>Parity with the real PortfolioManager</h3>
  <p class="prose">{text_dict['parity_intro_html']}</p>
  {parity_table_html(data_dict)}
</section>"""


def parity_table_html(data_dict: dict) -> str:
    parity_df = data_dict["parity"]
    if parity_df is None:
        return ""
    row_list = []
    for row in parity_df.itertuples():
        row_list.append([
            rl.esc(data_dict["text"]["legacy_labels"].get(row.book_str, row.book_str)), rl.esc(row.window_str),
            rl.pct(row.pm_cagr, 3), rl.pct(row.matrix_cagr, 3), rl.pct(row.pm_vol, 2), rl.pct(row.matrix_vol, 2),
            f"{row.daily_corr:.6f}", rl.pct(row.tracking_error, 3),
        ])
    return rl.table(["Book", "Window", "PortfolioManager CAGR", "Study CAGR", "PM volatility", "Study volatility", "Daily correlation", "Tracking error / yr"], row_list)


def section_text_block(data_dict: dict, key_str: str, section_id_str: str, title_str: str) -> str:
    return f"""
<section id="{section_id_str}">
  <h2>{rl.esc(title_str)}</h2>
  {data_dict['text'][key_str]}
</section>"""


def chart_payload(data_dict: dict) -> dict:
    products = product_list(data_dict)
    nav_exact_df = data_dict["nav_exact"]
    nav_long_df = data_dict["nav_long"]
    bench_list = ["S&P 500 TR", "60/40"]
    payload_dict = {"lines": {}}
    for line_str in ("main", "low_touch"):
        line_products = [p for p in products if p["line_str"] == line_str]
        id_list = [p["product_id_str"] for p in line_products]
        meta_list = ([{"id": p["product_id_str"], "name": p["display_name_str"], "colorVar": p["color_var_str"]} for p in line_products]
                     + [{"id": "S&P 500 TR", "name": "S&P 500 TR", "colorVar": "--b-spx"}, {"id": "60/40", "name": "60/40", "colorVar": "--b-6040", "dashed": True}])
        payload_dict["lines"][line_str] = {
            "meta": meta_list,
            "navExact": rl.weekly_nav_payload(nav_exact_df, id_list + bench_list),
            "ddExact": rl.weekly_drawdown_payload(nav_exact_df, id_list + bench_list),
            "navLong": rl.weekly_nav_payload(nav_long_df, id_list + bench_list),
        }
    scatter_list = []
    for p in products:
        row = metric_row(data_dict, "exact_annual", p["product_id_str"])
        scatter_list.append({"x": row["volatility_float"], "y": row["cagr_float"], "label": p["display_name_str"], "colorVar": p["color_var_str"], "kind": "product",
                             "extra": {"value": rl.pct(row["max_drawdown_float"]), "label": "max drawdown"}})
    for legacy_id_str, label_str in data_dict["text"]["legacy_labels"].items():
        row = metric_row(data_dict, "exact_legacy", legacy_id_str)
        scatter_list.append({"x": row["volatility_float"], "y": row["cagr_float"], "label": label_str, "colorVar": "--legacy", "kind": "legacy",
                             "extra": {"value": rl.pct(row["max_drawdown_float"]), "label": "max drawdown"}})
    for bench_str, color_var_str in (("S&P 500 TR", "--b-spx"), ("60/40", "--b-6040"), ("T-bills", "--b-tbill")):
        row = metric_row(data_dict, "exact_benchmark", bench_str)
        scatter_list.append({"x": row["volatility_float"], "y": row["cagr_float"], "label": bench_str, "colorVar": color_var_str, "kind": "bench",
                             "extra": {"value": rl.pct(row["max_drawdown_float"]), "label": "max drawdown"}})
    boot_df = data_dict["boot"]
    boot_row_list = []
    for p in products:
        row = boot_df.loc[p["product_id_str"]]
        boot_row_list.append({"label": p["display_name_str"], "lo": row["sharpe_p05_float"], "hi": row["sharpe_p95_float"], "mid": row["sharpe_p50_float"],
                              "point": metric_row(data_dict, "exact_annual", p["product_id_str"])["sharpe_rf0_float"], "pointLabel": "measured Sharpe", "colorVar": p["color_var_str"]})
    for legacy_id_str, label_str in data_dict["text"]["legacy_labels"].items():
        row = boot_df.loc[legacy_id_str]
        boot_row_list.append({"label": label_str, "lo": row["sharpe_p05_float"], "hi": row["sharpe_p95_float"], "mid": row["sharpe_p50_float"],
                              "point": metric_row(data_dict, "exact_legacy", legacy_id_str)["sharpe_rf0_float"], "pointLabel": "measured Sharpe", "colorVar": "--legacy"})
    for bench_str, color_var_str in (("S&P 500 TR", "--b-spx"), ("60/40", "--b-6040")):
        row = boot_df.loc[bench_str]
        boot_row_list.append({"label": bench_str, "lo": row["sharpe_p05_float"], "hi": row["sharpe_p95_float"], "mid": row["sharpe_p50_float"],
                              "point": metric_row(data_dict, "exact_benchmark", bench_str)["sharpe_rf0_float"], "pointLabel": "measured Sharpe", "colorVar": color_var_str})
    payload_dict.update({
        "scatter": scatter_list,
        "composition": composition_payload(data_dict),
        "boot": {"rows": boot_row_list, "ref": float(metric_row(data_dict, "exact_benchmark", "60/40")["sharpe_rf0_float"])},
    })
    return payload_dict


BOOT_JS = """
(function () {
  const data = JSON.parse(document.getElementById('chart-data').textContent);
  function lines(meta, payload) {
    const rows = meta.map((m) => ({ name: m.name, colorVar: m.colorVar, dashed: !!m.dashed, bench: m.colorVar.startsWith('--b-'), values: payload.series[m.id] }));
    return rows.filter((r) => r.bench).concat(rows.filter((r) => !r.bench));
  }
  function draw() {
    for (const lineKey of Object.keys(data.lines)) {
      const L = data.lines[lineKey];
      FundCharts.lineChart(document.getElementById('chart-nav-exact-' + lineKey), { dates: L.navExact.dates, series: lines(L.meta, L.navExact), yLog: true, endLabels: true, height: 330, ariaLabel: 'Growth of one dollar, exact window, ' + lineKey });
      FundCharts.lineChart(document.getElementById('chart-dd-exact-' + lineKey), { dates: L.ddExact.dates, series: lines(L.meta, L.ddExact), yFormat: 'pct', yMax: 0, height: 250, ariaLabel: 'Drawdowns, exact window, ' + lineKey });
      FundCharts.lineChart(document.getElementById('chart-nav-long-' + lineKey), { dates: L.navLong.dates, series: lines(L.meta, L.navLong), yLog: true, endLabels: true, height: 330, ariaLabel: 'Growth of one dollar including 2008, ' + lineKey });
    }
    FundCharts.scatterChart(document.getElementById('chart-scatter'), { points: data.scatter, xLabel: 'Volatility (annualised)', yLabel: 'CAGR', height: 400, ariaLabel: 'Risk and return' });
    FundCharts.stackedBars(document.getElementById('chart-composition'), { rows: data.composition.rows, keys: data.composition.keys, labelWidth: 210, ariaLabel: 'Capital and risk by engine' });
    FundCharts.dotWhisker(document.getElementById('chart-boot'), { rows: data.boot.rows, refValue: data.boot.ref, format: 'num', labelWidth: 210, ariaLabel: 'Bootstrap Sharpe ranges' });
  }
  draw();
  let timer = null;
  window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(draw, 150); });
  const observer = new MutationObserver(draw);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
"""


def main() -> int:
    data_dict = load_all()
    REPORT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    css_str = (HERE_PATH / "report_assets" / "report.css").read_text(encoding="utf-8")
    charts_js_str = (HERE_PATH / "report_assets" / "charts.js").read_text(encoding="utf-8")
    body_html = "".join([
        section_header(data_dict),
        section_menu(data_dict),
        section_performance(data_dict),
        section_crises(data_dict),
        section_years(data_dict),
        section_defensive(data_dict),
        section_capacity(data_dict),
        section_construction(data_dict),
        section_method(data_dict),
        section_building_blocks(data_dict),
        section_trust(data_dict),
        section_text_block(data_dict, "legacy_html", "vs-ladder", "What changes versus the current ladder"),
        section_text_block(data_dict, "limits_html", "limits", "Limitations and open decisions"),
        section_text_block(data_dict, "appendix_html", "appendix", "Appendix: provenance and definitions"),
    ])
    page_html = f"""<title>{rl.esc(data_dict['text']['page_title'])}</title>
<meta name="description" content="{rl.esc(data_dict['text']['page_description'])}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>{css_str}</style>
<div class="wrap">
{body_html}
<footer class="note" style="margin-top:48px;border-top:1px solid var(--rule);padding-top:12px">{data_dict['text']['footer_html']}</footer>
</div>
<script id="chart-data" type="application/json">{rl.json_payload(chart_payload(data_dict))}</script>
<script>{charts_js_str}</script>
<script>{BOOT_JS}</script>
"""
    output_path = REPORT_DIR_PATH / "fund_product_menu.html"
    output_path.write_text(page_html, encoding="utf-8")
    print(f"wrote {output_path} ({len(page_html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
