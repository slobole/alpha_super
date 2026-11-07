"""Read-only Scout pages in Bench: /scout, /scout/shadows, /scout/ledger, /scout/families.

Everything here reads files that the Scout jobs already wrote (the daily status, shadow
reports, pod health, FDR report, the ledger, research cards). Nothing is written, and a
missing file renders as "no data yet", never as an error. The data roots are module-level
constants so tests can point them at a temporary folder.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

from flask import Blueprint, render_template, url_for
from markupsafe import Markup

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
SCOUT_ROOT_PATH = REPO_ROOT_PATH / "results" / "scout"
LEDGER_PATH = REPO_ROOT_PATH / "research_ledger" / "scout_ledger.jsonl"
CARDS_ROOT_PATH = SCOUT_ROOT_PATH / "cards"
CARDS_ARTIFACT_PREFIX_STR = "scout/cards/"  # the cards folder's path under results/, for /artifact/

BAND_HORIZON_INT = 252
STALE_DAYS_INT = 5
GRADE_ORDER_LIST = ["CANDIDATE", "WATCHLIST", "DIAGNOSTIC", "REJECTED", "NOT_TESTED"]

blueprint_obj = Blueprint("scout", __name__)


# ── file readers ─────────────────────────────────────────────────────────
def _latest_dated_file_path(folder_path: Path, name_str: str) -> Path | None:
    try:
        folder_list = sorted(p for p in folder_path.iterdir() if p.is_dir() and (p / name_str).is_file())
    except OSError:
        return None
    return folder_list[-1] / name_str if folder_list else None


def _read_json(path_obj: Path | None):
    if path_obj is None:
        return None
    try:
        return json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _latest_json(subfolder_str: str, name_str: str):
    return _read_json(_latest_dated_file_path(SCOUT_ROOT_PATH / subfolder_str, name_str))


def _ledger_rows(row_type_str: str) -> list[dict]:
    """Rows of one type; unreadable lines are skipped so a partial last line cannot break a page."""
    row_list: list[dict] = []
    try:
        with LEDGER_PATH.open("r", encoding="utf-8") as ledger_file:
            for line_str in ledger_file:
                try:
                    row_dict = json.loads(line_str)
                except ValueError:
                    continue
                if isinstance(row_dict, dict) and row_dict.get("row_type_str") == row_type_str:
                    row_list.append(row_dict)
    except OSError:
        pass
    return row_list


def _family_description_dict() -> dict:
    try:
        from alpha.scout.families import FAMILY_DESCRIPTION_DICT

        return dict(FAMILY_DESCRIPTION_DICT)
    except Exception:
        return {}


def _grade_word_str(verdict_str: str | None) -> str:
    word_str = re.split(r"[\s(]", (verdict_str or "").strip(), maxsplit=1)[0]
    return word_str or "UNGRADED"


def _pct_str(value_obj, digits_int: int = 1) -> str:
    return f"{value_obj * 100:+.{digits_int}f}%" if isinstance(value_obj, (int, float)) else "–"


def _num_str(value_obj, digits_int: int = 2) -> str:
    return f"{value_obj:.{digits_int}f}" if isinstance(value_obj, (int, float)) else "–"


def _parse_date(text_obj) -> date | None:
    try:
        return date.fromisoformat(str(text_obj)[:10])
    except ValueError:
        return None


# ── overview ─────────────────────────────────────────────────────────────
def _status_row_list(daily_dict: dict) -> list[dict]:
    part_dict = daily_dict.get("parts") or {}
    row_list: list[dict] = []

    def add(name_str, state_str, word_str, line_str):
        row_list.append({"name_str": name_str, "state_str": state_str, "word_str": word_str, "line_str": line_str})

    def failed(name_str, label_str) -> bool:
        part_obj = part_dict.get(name_str)
        if part_obj is None:
            add(label_str, "grey", "No data", "This part is not in the daily status.")
            return True
        if not part_obj.get("ok_bool", True):
            add(label_str, "red", "Failed", str(part_obj.get("error_str") or part_obj.get("summary_str") or "The part failed."))
            return True
        return False

    if not failed("ledger", "Ledger"):
        add("Ledger", "green", "Intact", f"{part_dict['ledger'].get('rows_int', '–')} rows, chain intact")
    if not failed("shadow", "Shadows"):
        status_dict = part_dict["shadow"].get("status_by_registration_dict") or {}
        count_dict: dict[str, int] = {}
        for status_str in status_dict.values():
            count_dict[status_str] = count_dict.get(status_str, 0) + 1
        words_str = ", ".join(f"{n} {s.lower().replace('_', ' ')}" for s, n in sorted(count_dict.items()))
        state_str = "red" if any(s in ("ERROR", "RED") for s in count_dict) else (
            "amber" if any(s in ("AMBER", "NO_DATA") for s in count_dict) else "green")
        add("Shadows", state_str, "Watching", f"{len(status_dict)} shadows: {words_str}" if status_dict else "No shadows yet")
    if not failed("health", "Live pod health"):
        health_obj = part_dict["health"]
        status_dict = health_obj.get("status_by_pod_dict") or {}
        if not status_dict:
            add("Live pod health", "grey", str(health_obj.get("status_str") or "No data").title().replace("_", " "),
                "No live pod data was available.")
        else:
            worst_str = "RED" if "RED" in status_dict.values() else ("AMBER" if "AMBER" in status_dict.values() else "GREEN")
            add("Live pod health", {"RED": "red", "AMBER": "amber"}.get(worst_str, "green"), worst_str.title(),
                ", ".join(f"{k} {str(v).lower()}" for k, v in status_dict.items()))
    if not failed("fdr", "Discovery test"):
        fdr_obj = part_dict["fdr"]
        add("Discovery test", "green", "Done",
            f"{len(fdr_obj.get('discoveries') or [])} discoveries from {fdr_obj.get('primary_tests_int', '–')} tests "
            f"across {fdr_obj.get('registrations_int', '–')} registrations")
    if not failed("kb", "Knowledge base"):
        kb_obj = part_dict["kb"]
        overdue_int = len(kb_obj.get("overdue_id_list") or [])
        gap_int = kb_obj.get("gap_count_int") or 0
        state_str = "red" if overdue_int else ("amber" if gap_int else "green")
        add("Knowledge base", state_str, {"red": "Overdue", "amber": "Pending", "green": "Up to date"}[state_str],
            f"{overdue_int} overdue, {gap_int} pending")
    return row_list


def _pod_row_list() -> list[dict]:
    report_list = _latest_json("pod_health", "pod_health.json")
    if not isinstance(report_list, list):
        return []
    row_list = []
    for report_dict in report_list:
        live_dict = report_dict.get("live") or {}
        as_of_date = _parse_date(report_dict.get("as_of_str"))
        end_date = _parse_date(live_dict.get("end_str"))
        behind_int = live_dict.get("sessions_behind_int") or 0
        stale_bool = bool(live_dict.get("stale_bool")) or behind_int > 0 or bool(
            as_of_date and end_date and (as_of_date - end_date).days > STALE_DAYS_INT)
        status_str = str(report_dict.get("status_str") or "NO_DATA")
        drawdown_obj = live_dict.get("current_drawdown_float")
        row_list.append({
            "label_str": report_dict.get("label_str") or report_dict.get("pod_id_str") or "pod",
            "status_str": status_str.title().replace("_", " "),
            "state_str": {"RED": "red", "AMBER": "amber", "GREEN": "green"}.get(status_str, "grey"),
            "through_str": str(live_dict.get("end_str") or "no data"),
            "stale_bool": stale_bool,
            "behind_int": behind_int,
            "return_str": _pct_str(live_dict.get("cumulative_return_float")),
            "drawdown_str": _pct_str(-abs(drawdown_obj)) if isinstance(drawdown_obj, (int, float)) else "–",
            "sessions_int": live_dict.get("session_count_int"),
        })
    return row_list


@blueprint_obj.route("/scout")
def scout_overview_fn() -> str:
    daily_dict = _latest_json("daily", "daily_status.json")
    daily_dict = daily_dict if isinstance(daily_dict, dict) else {}
    return render_template(
        "scout.html", active_nav="scout", scout_tab_str="overview", has_data_bool=bool(daily_dict),
        as_of_str=daily_dict.get("as_of_str"), written_str=str(daily_dict.get("written_utc_str") or "")[:16].replace("T", " "),
        overall_ok_bool=daily_dict.get("ok_bool"), status_row_list=_status_row_list(daily_dict),
        pod_row_list=_pod_row_list(), shadow_row_list=_shadow_row_list(with_chart_bool=False),
    )


# ── shadows ──────────────────────────────────────────────────────────────
def _read_csv_rows(path_obj: Path) -> list[dict]:
    try:
        with path_obj.open("r", encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except (OSError, ValueError, csv.Error):  # unreadable or not UTF-8: the page shows no sessions, never a 500
        return []


def _expected_return_list(registration_id_str: str, start_row: dict | None) -> list[float]:
    path_str = (start_row or {}).get("expected_path_str")
    candidate_list = []
    if path_str:
        candidate_list.append(REPO_ROOT_PATH / str(path_str).replace("\\", "/"))
    candidate_list.append(SCOUT_ROOT_PATH / "shadow" / registration_id_str / "expected.parquet")
    for path_obj in candidate_list:
        if not path_obj.is_file():
            continue
        try:
            import pandas as pd

            frame_obj = pd.read_parquet(path_obj)
            return [float(v) for v in frame_obj["expected_return_float"].dropna().tolist()]
        except Exception:
            continue
    return []


def _band_dict(return_list: list[float], horizon_int: int) -> dict | None:
    """Mean path and 5th to 95th percentile of cumulative return over windows of horizon_int sessions.

    Simple method, no resampling: every window of the expected daily returns of that length
    (thinned to about 600 windows) is compounded, then each step takes the mean and the
    percentiles across windows. Windows overlap, so the band is a guide, not a test.
    """
    if len(return_list) < horizon_int + 20:
        return None
    import numpy as np

    series_arr = np.asarray(return_list, dtype=float)
    last_start_int = len(series_arr) - horizon_int
    stride_int = max(1, last_start_int // 600)
    path_arr = np.vstack([np.cumprod(1 + series_arr[s:s + horizon_int]) - 1 for s in range(0, last_start_int + 1, stride_int)])
    return {
        "mean_list": path_arr.mean(axis=0).tolist(),
        "low_list": np.percentile(path_arr, 5, axis=0).tolist(),
        "high_list": np.percentile(path_arr, 95, axis=0).tolist(),
    }


def _forward_path_list(registration_id_str: str) -> list[tuple[str, float]]:
    point_list = []
    for row_dict in _read_csv_rows(SCOUT_ROOT_PATH / "shadow" / registration_id_str / "forward_log.csv"):
        try:
            point_list.append((row_dict["date_str"], float(row_dict["return_float"])))
        except (KeyError, TypeError, ValueError):
            continue
    point_list.sort()
    return point_list


def _chart_svg(forward_list: list[tuple[str, float]], band_obj: dict | None) -> Markup:
    """Inline SVG: range band, expected mean line, forward path. x is sessions since the shadow start."""
    width_int, height_int, left_int, right_int, top_int, bottom_int = 640, 220, 46, 12, 10, 24
    cumulative_list, level_float = [], 1.0
    for _, return_float in forward_list:
        level_float *= 1 + return_float
        cumulative_list.append(level_float - 1)
    horizon_int = max(BAND_HORIZON_INT, len(cumulative_list))
    value_list = cumulative_list + (band_obj["low_list"] + band_obj["high_list"] if band_obj else [])
    if not value_list:
        return Markup("")
    low_float, high_float = min(value_list + [0.0]), max(value_list + [0.0])
    pad_float = (high_float - low_float) * 0.06 or 0.01
    low_float, high_float = low_float - pad_float, high_float + pad_float

    def sx(index_float: float) -> float:
        return left_int + (width_int - left_int - right_int) * index_float / max(horizon_int, 1)

    def sy(value_float: float) -> float:
        return top_int + (height_int - top_int - bottom_int) * (high_float - value_float) / (high_float - low_float)

    def points(value_seq) -> list[str]:
        return [f"{sx(i + 1):.1f},{sy(v):.1f}" for i, v in enumerate(value_seq)]

    part_list = [f'<svg class="sc-chart" viewBox="0 0 {width_int} {height_int}" role="img" '
                 f'aria-label="Forward path against the expected range">']
    for tick_int in range(5):
        value_float = low_float + (high_float - low_float) / 4 * tick_int
        part_list.append(
            f'<line class="sc-grid" x1="{left_int}" x2="{width_int - right_int}" y1="{sy(value_float):.1f}" y2="{sy(value_float):.1f}"/>'
            f'<text class="sc-tick" x="{left_int - 6}" y="{sy(value_float) + 3:.1f}" text-anchor="end">{value_float * 100:.0f}%</text>')
    part_list.append(f'<line class="sc-zero" x1="{left_int}" x2="{width_int - right_int}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    for index_int in range(0, horizon_int + 1, 63):
        part_list.append(f'<text class="sc-tick" x="{sx(index_int):.1f}" y="{height_int - 8}" text-anchor="middle">{index_int}</text>')
    if band_obj:
        polygon_list = points(band_obj["high_list"]) + list(reversed(points(band_obj["low_list"])))
        part_list.append(f'<polygon class="sc-band" points="{" ".join(polygon_list)}"/>')
        part_list.append(f'<polyline class="sc-mean" points="{" ".join(points(band_obj["mean_list"]))}"/>')
    if cumulative_list:
        part_list.append(f'<polyline class="sc-fwd" points="{" ".join(points(cumulative_list))}"/>')
        part_list.append(f'<circle class="sc-dot" cx="{sx(len(cumulative_list)):.1f}" cy="{sy(cumulative_list[-1]):.1f}" r="3"/>')
    part_list.append("</svg>")
    return Markup("".join(part_list))


def _shadow_row_list(*, with_chart_bool: bool) -> list[dict]:
    report_list = _latest_json("shadow/reports", "shadow.json")
    if not isinstance(report_list, list):
        return []
    start_by_id_dict = {row.get("registration_id_str"): row for row in _ledger_rows("shadow_start")}
    row_list = []
    for report_dict in report_list:
        id_str = str(report_dict.get("pod_id_str") or "")
        shadow_dict = report_dict.get("shadow") or {}
        live_dict = report_dict.get("live") or {}
        revision_int = (report_dict.get("log") or {}).get("revisions_total_int")
        if revision_int is None:
            revision_int = len(_read_csv_rows(SCOUT_ROOT_PATH / "shadow" / id_str / "forward_revisions.csv"))
        status_str = str(report_dict.get("status_str") or "NO_DATA")
        progress_float = shadow_dict.get("progress_float")
        row_dict = {
            "id_str": id_str,
            "status_str": status_str.title().replace("_", " "),
            "state_str": {"RED": "red", "ERROR": "red", "AMBER": "amber", "GREEN": "green"}.get(status_str, "grey"),
            "start_str": shadow_dict.get("start_date_str") or report_dict.get("monitoring_start_str") or "–",
            "months_str": _num_str(shadow_dict.get("forward_months_float"), 1),
            "trl_str": _num_str(shadow_dict.get("min_trl_months_float"), 1),
            "progress_pct_int": int(round(min(max(progress_float or 0.0, 0.0), 1.0) * 100)),
            "sharpe_str": _num_str(shadow_dict.get("expected_annual_sharpe_float")),
            "return_str": _pct_str(live_dict.get("cumulative_return_float")),
            "sessions_int": shadow_dict.get("forward_sessions_int") or 0,
            "revision_int": revision_int or 0,
            "chart": None, "chart_note_str": "",
        }
        if with_chart_bool:
            forward_list = _forward_path_list(id_str)
            band_obj = _band_dict(_expected_return_list(id_str, start_by_id_dict.get(id_str)), BAND_HORIZON_INT)
            row_dict["chart"] = _chart_svg(forward_list, band_obj) if (forward_list or band_obj) else None
            note_list = []
            if not forward_list:
                note_list.append(f"The forward log starts {row_dict['start_str']}; no sessions are recorded yet.")
            if not band_obj:
                note_list.append("The expected backtest series is not available, so there is no range.")
            row_dict["chart_note_str"] = " ".join(note_list)
        row_list.append(row_dict)
    return row_list


@blueprint_obj.route("/scout/shadows")
def scout_shadows_fn() -> str:
    return render_template("scout_shadows.html", active_nav="scout", scout_tab_str="shadows",
                           shadow_row_list=_shadow_row_list(with_chart_bool=True), horizon_int=BAND_HORIZON_INT)


# ── ledger and families ──────────────────────────────────────────────────
def _normal_key_str(text_str: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text_str.lower()).strip("_")


def _card_link_dict() -> dict:
    """Normalised card file stem -> file name."""
    try:
        name_list = sorted(p.name for p in CARDS_ROOT_PATH.glob("*.html"))
    except OSError:
        return {}
    return {_normal_key_str(Path(n).stem): n for n in name_list}


def _card_name_for(registration_id_str: str, card_dict: dict) -> str | None:
    """Exact match on the registration id, or the id equals the card name plus a date suffix. Nothing looser."""
    key_str = _normal_key_str(registration_id_str)
    if key_str in card_dict:
        return card_dict[key_str]
    for stem_str, name_str in card_dict.items():
        if re.fullmatch(re.escape(stem_str) + r"_\d{8}", key_str):
            return name_str
    return None


def _grade_by_id_dict() -> dict:
    grade_dict: dict[str, str] = {}
    for row_dict in _ledger_rows("station_verdict"):  # ledger order, so the last grade row wins
        if row_dict.get("station_str") == "grade":
            grade_dict[row_dict.get("registration_id_str")] = row_dict.get("verdict_str") or ""
    return grade_dict


@blueprint_obj.route("/scout/ledger")
def scout_ledger_fn() -> str:
    grade_dict = _grade_by_id_dict()
    card_dict = _card_link_dict()
    row_list = []
    for row_dict in _ledger_rows("registration"):
        id_str = row_dict.get("registration_id_str") or ""
        card_name_str = _card_name_for(id_str, card_dict)
        row_list.append({
            "id_str": id_str, "family_str": row_dict.get("family_id_str") or "–",
            "grade_str": _grade_word_str(grade_dict.get(id_str)) if id_str in grade_dict else "UNGRADED",
            "grade_detail_str": grade_dict.get(id_str, ""), "trials_int": row_dict.get("prior_trials_int") or 0,
            "retro_bool": bool(row_dict.get("retro_bool")), "date_str": str(row_dict.get("utc_ts_str") or "")[:10],
            "card_url_str": url_for("artifact_fn", rel_path_str=CARDS_ARTIFACT_PREFIX_STR + card_name_str) if card_name_str else None,
        })
    row_list.reverse()  # newest first
    return render_template(
        "scout_ledger.html", active_nav="scout", scout_tab_str="ledger", row_list=row_list,
        family_list=sorted({r["family_str"] for r in row_list}), grade_list=sorted({r["grade_str"] for r in row_list}),
    )


@blueprint_obj.route("/scout/families")
def scout_families_fn() -> str:
    description_dict = _family_description_dict()
    grade_dict = _grade_by_id_dict()
    fdr_dict = _latest_json("fdr", "fdr_report.json")
    fdr_dict = fdr_dict if isinstance(fdr_dict, dict) else {}
    discovery_count_dict: dict[str, int] = {}
    for fdr_row in fdr_dict.get("rows") or []:
        if fdr_row.get("discovery_bool"):
            key_str = fdr_row.get("family_id_str")
            discovery_count_dict[key_str] = discovery_count_dict.get(key_str, 0) + 1
    family_dict: dict[str, dict] = {}

    def family_obj(id_str: str) -> dict:
        return family_dict.setdefault(id_str, {"id_str": id_str, "description_str": description_dict.get(id_str, ""),
                                               "registrations_int": 0, "trials_int": 0, "grade_dict": {}})

    for id_str in description_dict:
        family_obj(id_str)
    for row_dict in _ledger_rows("registration"):
        entry_dict = family_obj(row_dict.get("family_id_str") or "unknown")
        entry_dict["registrations_int"] += 1
        entry_dict["trials_int"] += row_dict.get("prior_trials_int") or 0
        registration_id_str = row_dict.get("registration_id_str")
        grade_str = _grade_word_str(grade_dict[registration_id_str]) if registration_id_str in grade_dict else "UNGRADED"
        entry_dict["grade_dict"][grade_str] = entry_dict["grade_dict"].get(grade_str, 0) + 1
    row_list = list(family_dict.values())
    for entry_dict in row_list:
        entry_dict["discoveries_int"] = discovery_count_dict.get(entry_dict["id_str"], 0)
    row_list.sort(key=lambda e: (-e["registrations_int"], e["id_str"]))
    grade_column_list = [g for g in GRADE_ORDER_LIST + ["UNGRADED"] if any(g in e["grade_dict"] for e in row_list)]
    return render_template("scout_families.html", active_nav="scout", scout_tab_str="families", row_list=row_list,
                           grade_column_list=grade_column_list, has_fdr_bool=bool(fdr_dict))


def register_routes(app_obj) -> None:
    app_obj.register_blueprint(blueprint_obj)
