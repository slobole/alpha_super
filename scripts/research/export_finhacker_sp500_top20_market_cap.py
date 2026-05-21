"""
Export FinHacker S&P 500 top-20 market-cap data for SP10 research.

This is a research-data ingestion helper, not an official point-in-time data
feed. The output shape is intentionally compatible with
strategy_mo_sp10_market_cap_rotation.py:

    date,symbol,market_cap

Data lineage:
- The top-20 page identifies the companies that appear in annual top-20 lists.
- Each linked company page contains month-end market-cap history.
- Norgate remains responsible for PIT S&P 500 membership and historical prices.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd


SOURCE_URL_STR = "https://www.finhacker.cz/en/top-20-sp-500-companies-by-market-cap/"
DEFAULT_CACHE_DIR = Path(".cache") / "finhacker_sp500_top20_market_cap"
DEFAULT_MONTHLY_OUTPUT_CSV_PATH = Path("data") / "external" / "finhacker_sp500_top20_market_cap_monthly.csv"
DEFAULT_ANNUAL_OUTPUT_CSV_PATH = Path("data") / "external" / "finhacker_sp500_top20_market_cap_annual_top20.csv"
DEFAULT_METADATA_JSON_PATH = Path("data") / "external" / "finhacker_sp500_top20_market_cap_metadata.json"
USER_AGENT_STR = "Mozilla/5.0 (compatible; alpha_super research data audit)"


# The top-20 page exposes company names and stock-page links, not machine-ready
# Norgate symbols. Keep the mapping explicit so a symbol/name ambiguity does not
# silently enter a backtest.
SLUG_TO_NORGATE_SYMBOL_DICT: dict[str, str] = {
    "3m": "MMM",
    "abbvie": "ABBV",
    "advanced-micro-devices": "AMD",
    "alphabet": "GOOGL",
    "altria-group": "MO",
    "amazon": "AMZN",
    "american-international-group": "AIG",
    "amgen": "AMGN",
    "apple": "AAPL",
    "att": "T",
    "bank-of-america": "BAC",
    "berkshire-hathaway": "BRK.B",
    "boeing": "BA",
    "bristol-myers-squibb": "BMY",
    "broadcom": "AVGO",
    "chevron": "CVX",
    "cisco-systems": "CSCO",
    "citigroup": "C",
    "coca-cola": "KO",
    "costco-wholesale": "COST",
    "eli-lilly": "LLY",
    "exxon-mobil": "XOM",
    "fannie-mae": "FNMA",
    "ge-aerospace": "GE",
    "home-depot": "HD",
    "intel": "INTC",
    "international-business-machines": "IBM",
    "johnson-johnson": "JNJ",
    "jpmorgan-chase": "JPM",
    "kellanova": "K-202512",
    "mastercard": "MA",
    "mcdonalds": "MCD",
    "merck": "MRK",
    "meta-platforms": "META",
    "microsoft": "MSFT",
    "micron-technology": "MU",
    "nvidia": "NVDA",
    "oracle": "ORCL",
    "palantir-technologies": "PLTR",
    "paypal-holdings": "PYPL",
    "pepsico": "PEP",
    "pfizer": "PFE",
    "philip-morris-international": "PM",
    "procter-gamble": "PG",
    "qualcomm": "QCOM",
    "schlumberger": "SLB",
    "tesla": "TSLA",
    "united-parcel-service": "UPS",
    "unitedhealth-group": "UNH",
    "verizon-communications": "VZ",
    "visa": "V",
    "walmart": "WMT",
    "walt-disney": "DIS",
    "wells-fargo": "WFC",
}


@dataclass(frozen=True)
class StockPageRef:
    company_name_str: str
    slug_str: str
    symbol_str: str
    source_url_str: str


def _strip_html_tags(text_str: str) -> str:
    return html.unescape(re.sub(r"<.*?>", "", text_str, flags=re.DOTALL)).strip()


def _slug_from_stock_url(source_url_str: str) -> str:
    path_part_str = urlparse(source_url_str).path.rstrip("/").split("/")[-1]
    return path_part_str.removesuffix("-market-cap")


def parse_market_cap_float(market_cap_text_str: str) -> float:
    normalized_text_str = html.unescape(market_cap_text_str).replace("$", "").replace(",", "").strip()
    match_obj = re.fullmatch(r"(?P<value_float>\d+(?:\.\d+)?)(?P<unit_str>[TBM])?", normalized_text_str)
    if match_obj is None:
        raise ValueError(f"Unsupported market-cap text: {market_cap_text_str!r}")

    unit_str = match_obj.group("unit_str") or ""
    scale_float = {"": 1.0, "M": 1_000_000.0, "B": 1_000_000_000.0, "T": 1_000_000_000_000.0}[unit_str]
    return float(match_obj.group("value_float")) * scale_float


def parse_source_effective_date(top_page_html_str: str) -> pd.Timestamp | None:
    match_obj = re.search(
        r"Data as of\s+(?P<date_html_str>.*?(?:</strong>|PM ET|AM ET))",
        top_page_html_str,
        flags=re.DOTALL,
    )
    if match_obj is None:
        return None
    date_text_str = _strip_html_tags(match_obj.group("date_html_str"))
    date_match_obj = re.search(r"(?P<date_str>[A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text_str)
    if date_match_obj is None:
        return None
    return pd.Timestamp(date_match_obj.group("date_str")).normalize()


def parse_stock_page_refs(top_page_html_str: str, base_url_str: str = SOURCE_URL_STR) -> list[StockPageRef]:
    row_pattern_obj = re.compile(
        r'<a class="name-link" href="(?P<href_str>[^"]+)">.*?<strong>(?P<company_html_str>.*?)</strong>',
        flags=re.DOTALL,
    )
    stock_ref_by_url_dict: dict[str, StockPageRef] = {}
    for match_obj in row_pattern_obj.finditer(top_page_html_str):
        source_url_str = urljoin(base_url_str, match_obj.group("href_str"))
        slug_str = _slug_from_stock_url(source_url_str)
        if slug_str not in SLUG_TO_NORGATE_SYMBOL_DICT:
            raise KeyError(f"No Norgate symbol mapping for FinHacker slug: {slug_str}")

        if source_url_str in stock_ref_by_url_dict:
            continue

        stock_ref_by_url_dict[source_url_str] = StockPageRef(
            company_name_str=_strip_html_tags(match_obj.group("company_html_str")),
            slug_str=slug_str,
            symbol_str=SLUG_TO_NORGATE_SYMBOL_DICT[slug_str],
            source_url_str=source_url_str,
        )

    return sorted(stock_ref_by_url_dict.values(), key=lambda stock_ref: stock_ref.slug_str)


def parse_annual_top20_rows(top_page_html_str: str, base_url_str: str = SOURCE_URL_STR) -> pd.DataFrame:
    source_effective_date_ts = parse_source_effective_date(top_page_html_str)
    source_effective_year_int = None if source_effective_date_ts is None else int(source_effective_date_ts.year)

    section_pattern_obj = re.compile(
        r"Largest 20 S&amp;P 500 Companies by Market Cap in (?P<year_int>\d{4})</summary>"
        r"\s*(?P<table_html_str><table.*?</table>)",
        flags=re.DOTALL,
    )
    row_pattern_obj = re.compile(
        r'<tr><td class="idx">(?P<rank_int>\d+)</td>'
        r'.*?<a class="name-link" href="(?P<href_str>[^"]+)">'
        r'.*?<strong>(?P<company_html_str>.*?)</strong>'
        r".*?</a></td><td class=\"cap\">(?P<market_cap_text_str>.*?)</td>",
        flags=re.DOTALL,
    )

    row_dict_list: list[dict[str, object]] = []
    for section_match_obj in section_pattern_obj.finditer(top_page_html_str):
        snapshot_year_int = int(section_match_obj.group("year_int"))
        if source_effective_year_int == snapshot_year_int and source_effective_date_ts is not None:
            snapshot_date_ts = source_effective_date_ts
        else:
            snapshot_date_ts = pd.Timestamp(year=snapshot_year_int, month=12, day=31)

        for row_match_obj in row_pattern_obj.finditer(section_match_obj.group("table_html_str")):
            source_url_str = urljoin(base_url_str, row_match_obj.group("href_str"))
            slug_str = _slug_from_stock_url(source_url_str)
            symbol_str = SLUG_TO_NORGATE_SYMBOL_DICT[slug_str]
            market_cap_text_str = _strip_html_tags(row_match_obj.group("market_cap_text_str"))
            row_dict_list.append(
                {
                    "date": snapshot_date_ts.strftime("%Y-%m-%d"),
                    "snapshot_year_int": snapshot_year_int,
                    "rank_int": int(row_match_obj.group("rank_int")),
                    "symbol": symbol_str,
                    "company_name": _strip_html_tags(row_match_obj.group("company_html_str")),
                    "market_cap": parse_market_cap_float(market_cap_text_str),
                    "market_cap_text": market_cap_text_str,
                    "source_url": source_url_str,
                }
            )

    annual_top20_df = pd.DataFrame(row_dict_list)
    if annual_top20_df.empty:
        raise RuntimeError("No annual top-20 rows were parsed from the FinHacker page.")
    return annual_top20_df.sort_values(["snapshot_year_int", "rank_int"]).reset_index(drop=True)


def parse_monthly_market_cap_rows(stock_page_html_str: str, stock_ref: StockPageRef) -> pd.DataFrame:
    row_pattern_obj = re.compile(
        r"<time datetime='(?P<date_str>\d{4}-\d{2}-\d{2})'>.*?</time>"
        r"</td><td>(?P<market_cap_text_str>.*?)</td>",
        flags=re.DOTALL,
    )
    row_dict_list: list[dict[str, object]] = []
    for match_obj in row_pattern_obj.finditer(stock_page_html_str):
        market_cap_text_str = _strip_html_tags(match_obj.group("market_cap_text_str"))
        row_dict_list.append(
            {
                "date": match_obj.group("date_str"),
                "symbol": stock_ref.symbol_str,
                "company_name": stock_ref.company_name_str,
                "market_cap": parse_market_cap_float(market_cap_text_str),
                "market_cap_text": market_cap_text_str,
                "source_url": stock_ref.source_url_str,
            }
        )

    monthly_market_cap_df = pd.DataFrame(row_dict_list)
    if monthly_market_cap_df.empty:
        raise RuntimeError(f"No monthly market-cap rows parsed from {stock_ref.source_url_str}")

    monthly_market_cap_df["date"] = pd.to_datetime(monthly_market_cap_df["date"]).dt.strftime("%Y-%m-%d")
    return (
        monthly_market_cap_df.drop_duplicates(["date", "symbol"])
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def fetch_html_text(
    source_url_str: str,
    cache_path: Path,
    refresh_bool: bool,
    max_retries_int: int,
    retry_delay_sec_float: float,
) -> str:
    if cache_path.exists() and not refresh_bool:
        return cache_path.read_text(encoding="utf-8")

    last_error: Exception | None = None
    for attempt_int in range(max_retries_int + 1):
        try:
            request_obj = Request(source_url_str, headers={"User-Agent": USER_AGENT_STR})
            with urlopen(request_obj, timeout=30) as response_obj:
                html_text_str = response_obj.read().decode("utf-8", errors="replace")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html_text_str, encoding="utf-8")
            return html_text_str
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt_int >= max_retries_int:
                break
            time.sleep(retry_delay_sec_float * float(attempt_int + 1))

    raise RuntimeError(f"Failed to fetch {source_url_str}: {last_error}") from last_error


def build_monthly_market_cap_df(
    stock_ref_list: Iterable[StockPageRef],
    cache_dir_path: Path,
    refresh_bool: bool,
    delay_sec_float: float,
    max_retries_int: int,
    retry_delay_sec_float: float,
) -> pd.DataFrame:
    monthly_market_cap_df_list: list[pd.DataFrame] = []
    for stock_ref in stock_ref_list:
        cache_path = cache_dir_path / f"{stock_ref.slug_str}.html"
        stock_page_html_str = fetch_html_text(
            source_url_str=stock_ref.source_url_str,
            cache_path=cache_path,
            refresh_bool=refresh_bool,
            max_retries_int=max_retries_int,
            retry_delay_sec_float=retry_delay_sec_float,
        )
        monthly_market_cap_df_list.append(
            parse_monthly_market_cap_rows(
                stock_page_html_str=stock_page_html_str,
                stock_ref=stock_ref,
            )
        )
        if delay_sec_float > 0.0:
            time.sleep(delay_sec_float)

    monthly_market_cap_df = pd.concat(monthly_market_cap_df_list, ignore_index=True)
    return monthly_market_cap_df.sort_values(["date", "symbol"]).reset_index(drop=True)


def write_outputs(
    monthly_market_cap_df: pd.DataFrame | None,
    annual_top20_df: pd.DataFrame,
    metadata_dict: dict[str, object],
    monthly_output_csv_path: Path,
    annual_output_csv_path: Path,
    metadata_json_path: Path,
) -> None:
    annual_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    annual_top20_df.to_csv(annual_output_csv_path, index=False)

    if monthly_market_cap_df is not None:
        monthly_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        monthly_market_cap_df.to_csv(monthly_output_csv_path, index=False)

    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json_path.write_text(json.dumps(metadata_dict, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--source-url", default=SOURCE_URL_STR)
    parser_obj.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser_obj.add_argument("--monthly-output-csv", type=Path, default=DEFAULT_MONTHLY_OUTPUT_CSV_PATH)
    parser_obj.add_argument("--annual-output-csv", type=Path, default=DEFAULT_ANNUAL_OUTPUT_CSV_PATH)
    parser_obj.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON_PATH)
    parser_obj.add_argument("--annual-only", action="store_true")
    parser_obj.add_argument("--refresh", action="store_true")
    parser_obj.add_argument("--delay-sec", type=float, default=1.5)
    parser_obj.add_argument("--max-retries", type=int, default=4)
    parser_obj.add_argument("--retry-delay-sec", type=float, default=30.0)
    return parser_obj.parse_args()


def main() -> None:
    args = parse_args()
    top_page_cache_path = args.cache_dir / "top20.html"
    top_page_html_str = fetch_html_text(
        source_url_str=args.source_url,
        cache_path=top_page_cache_path,
        refresh_bool=args.refresh,
        max_retries_int=args.max_retries,
        retry_delay_sec_float=args.retry_delay_sec,
    )

    stock_ref_list = parse_stock_page_refs(top_page_html_str=top_page_html_str, base_url_str=args.source_url)
    annual_top20_df = parse_annual_top20_rows(top_page_html_str=top_page_html_str, base_url_str=args.source_url)
    monthly_market_cap_df = None
    if not args.annual_only:
        monthly_market_cap_df = build_monthly_market_cap_df(
            stock_ref_list=stock_ref_list,
            cache_dir_path=args.cache_dir,
            refresh_bool=args.refresh,
            delay_sec_float=args.delay_sec,
            max_retries_int=args.max_retries,
            retry_delay_sec_float=args.retry_delay_sec,
        )

    metadata_dict = {
        "source_url": args.source_url,
        "source_effective_date": (
            None
            if parse_source_effective_date(top_page_html_str) is None
            else parse_source_effective_date(top_page_html_str).strftime("%Y-%m-%d")
        ),
        "stock_page_count": len(stock_ref_list),
        "annual_row_count": int(len(annual_top20_df)),
        "monthly_row_count": None if monthly_market_cap_df is None else int(len(monthly_market_cap_df)),
        "symbol_list": [stock_ref.symbol_str for stock_ref in stock_ref_list],
        "caveat": (
            "Third-party compiled/backfilled market-cap data. Use for research only; "
            "Norgate still supplies PIT S&P 500 membership and prices."
        ),
    }
    write_outputs(
        monthly_market_cap_df=monthly_market_cap_df,
        annual_top20_df=annual_top20_df,
        metadata_dict=metadata_dict,
        monthly_output_csv_path=args.monthly_output_csv,
        annual_output_csv_path=args.annual_output_csv,
        metadata_json_path=args.metadata_json,
    )

    if monthly_market_cap_df is None:
        print(f"Wrote annual top-20 CSV: {args.annual_output_csv}")
    else:
        print(f"Wrote monthly market-cap CSV: {args.monthly_output_csv}")
    print(f"Wrote annual audit CSV: {args.annual_output_csv}")
    print(f"Wrote metadata JSON: {args.metadata_json}")


if __name__ == "__main__":
    main()
