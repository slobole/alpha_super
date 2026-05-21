import unittest

from scripts.research.export_finhacker_sp500_top20_market_cap import (
    StockPageRef,
    parse_annual_top20_rows,
    parse_market_cap_float,
    parse_monthly_market_cap_rows,
    parse_source_effective_date,
    parse_stock_page_refs,
)


class FinHackerMarketCapExportTests(unittest.TestCase):
    def test_parse_market_cap_float_scales_units(self):
        self.assertEqual(parse_market_cap_float("$1.25T"), 1_250_000_000_000.0)
        self.assertEqual(parse_market_cap_float("999.5B"), 999_500_000_000.0)
        self.assertEqual(parse_market_cap_float("12M"), 12_000_000.0)

    def test_parse_stock_page_refs_maps_slugs_to_norgate_symbols(self):
        top_page_html_str = """
        <a class="name-link" href="/en/stocks/apple-market-cap/"><strong>Apple</strong></a>
        <a class="name-link" href="/en/stocks/berkshire-hathaway-market-cap/"><strong>Berkshire Hathaway</strong></a>
        <a class="name-link" href="/en/stocks/apple-market-cap/"><strong>Apple</strong></a>
        """

        stock_ref_list = parse_stock_page_refs(
            top_page_html_str=top_page_html_str,
            base_url_str="https://www.finhacker.cz/en/top-20-sp-500-companies-by-market-cap/",
        )

        symbol_list = [stock_ref.symbol_str for stock_ref in stock_ref_list]
        self.assertEqual(symbol_list, ["AAPL", "BRK.B"])

    def test_parse_annual_top20_rows_extracts_ranked_market_caps(self):
        top_page_html_str = """
        <p>Data as of May 20, 2026, 4:00 PM ET</p>
        Largest 20 S&amp;P 500 Companies by Market Cap in 2026</summary>
        <table><tbody>
        <tr><td class="idx">1</td><td><a class="name-link" href="/en/stocks/nvidia-market-cap/"><strong>NVIDIA</strong></a></td><td class="cap">5.433T</td><td></td></tr>
        </tbody></table>
        Largest 20 S&amp;P 500 Companies by Market Cap in 2025</summary>
        <table><tbody>
        <tr><td class="idx">2</td><td><a class="name-link" href="/en/stocks/apple-market-cap/"><strong>Apple</strong></a></td><td class="cap">4.017T</td><td></td></tr>
        </tbody></table>
        """

        annual_top20_df = parse_annual_top20_rows(top_page_html_str)

        self.assertEqual(len(annual_top20_df), 2)
        self.assertEqual(annual_top20_df.loc[0, "date"], "2025-12-31")
        self.assertEqual(annual_top20_df.loc[1, "date"], "2026-05-20")
        self.assertEqual(annual_top20_df.loc[1, "symbol"], "NVDA")

    def test_parse_monthly_market_cap_rows_extracts_hidden_rows(self):
        stock_page_html_str = """
        <tr class='month-row year-2025' style='display:none;'><td>
        <time datetime='2025-12-31'>December 31, 2025</time></td><td>$4.533T</td><td>x</td></tr>
        <tr class='month-row year-2025' style='display:none;'><td>
        <time datetime='2025-11-28'>November 28, 2025</time></td><td>$4.302T</td><td>x</td></tr>
        """
        stock_ref = StockPageRef(
            company_name_str="NVIDIA",
            slug_str="nvidia",
            symbol_str="NVDA",
            source_url_str="https://www.finhacker.cz/en/stocks/nvidia-market-cap/",
        )

        monthly_market_cap_df = parse_monthly_market_cap_rows(
            stock_page_html_str=stock_page_html_str,
            stock_ref=stock_ref,
        )

        self.assertEqual(len(monthly_market_cap_df), 2)
        self.assertEqual(monthly_market_cap_df.loc[0, "date"], "2025-11-28")
        self.assertEqual(monthly_market_cap_df.loc[1, "market_cap"], 4_533_000_000_000.0)

    def test_parse_source_effective_date(self):
        source_effective_date_ts = parse_source_effective_date(
            'Data as of <strong><span class="asof-date">May 20, 2026</span>'
            '<span class="asof-time">, 4:00 PM ET</span></strong>'
        )

        self.assertEqual(str(source_effective_date_ts.date()), "2026-05-20")


if __name__ == "__main__":
    unittest.main()
