import unittest

import pandas as pd

from scripts.research.run_dividend_total_return_parity_study import (
    build_dividend_event_df,
    build_dividend_parity_frame_df,
    compute_symbol_summary_dict,
)


class DividendTotalReturnParityStudyTests(unittest.TestCase):
    @staticmethod
    def make_price_frame_tuple() -> tuple[pd.DataFrame, pd.DataFrame]:
        date_idx = pd.bdate_range("2024-01-02", periods=4)
        capital_price_df = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 99.0, 101.0],
                "Dividend": [0.0, 1.0, 0.0, 0.0],
            },
            index=date_idx,
        )
        total_return_price_df = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0, 102.02020202020202],
            },
            index=date_idx,
        )
        return capital_price_df, total_return_price_df

    def test_next_session_dividend_reproduces_total_return(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()

        parity_frame_df = build_dividend_parity_frame_df(
            capital_price_df=capital_price_df,
            total_return_price_df=total_return_price_df,
        )

        ex_date_ts = parity_frame_df.index[2]
        self.assertAlmostEqual(
            float(parity_frame_df.loc[ex_date_ts, "modeled_total_return_float"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(parity_frame_df.loc[ex_date_ts, "norgate_total_return_float"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                parity_frame_df.loc[
                    ex_date_ts,
                    "same_session_placebo_error_bps_float",
                ]
            ),
            -100.0,
        )

    def test_summary_passes_exact_parity_and_retains_placebo_failure(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()
        parity_frame_df = build_dividend_parity_frame_df(
            capital_price_df=capital_price_df,
            total_return_price_df=total_return_price_df,
        )

        summary_dict = compute_symbol_summary_dict(
            symbol_str="TEST",
            parity_frame_df=parity_frame_df,
        )

        self.assertTrue(summary_dict["pass_bool"])
        self.assertEqual(summary_dict["dividend_event_count_int"], 1)
        self.assertAlmostEqual(
            float(summary_dict["event_mean_absolute_error_bps_float"]),
            0.0,
        )
        self.assertAlmostEqual(
            float(summary_dict["terminal_wealth_error_bps_float"]),
            0.0,
        )
        self.assertLess(
            float(
                summary_dict[
                    "fixed_share_cash_vs_total_return_error_bps_float"
                ]
            ),
            0.0,
        )
        self.assertGreater(
            float(
                summary_dict[
                    "same_session_placebo_max_absolute_error_bps_float"
                ]
            ),
            0.0,
        )

    def test_event_audit_maps_entitlement_to_next_session(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()
        parity_frame_df = build_dividend_parity_frame_df(
            capital_price_df=capital_price_df,
            total_return_price_df=total_return_price_df,
        )

        event_df = build_dividend_event_df(
            symbol_str="TEST",
            parity_frame_df=parity_frame_df,
        )

        self.assertEqual(len(event_df), 1)
        self.assertEqual(
            event_df.iloc[0]["entitlement_date_str"],
            parity_frame_df.index[1].date().isoformat(),
        )
        self.assertEqual(
            event_df.iloc[0]["ex_date_str"],
            parity_frame_df.index[2].date().isoformat(),
        )
        self.assertAlmostEqual(
            float(event_df.iloc[0]["dividend_per_share_float"]),
            1.0,
        )

    def test_missing_dividend_field_fails_loud(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()

        with self.assertRaisesRegex(ValueError, "Dividend"):
            build_dividend_parity_frame_df(
                capital_price_df=capital_price_df.drop(columns=["Dividend"]),
                total_return_price_df=total_return_price_df,
            )

    def test_null_dividend_value_fails_loud(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()
        capital_price_df.loc[capital_price_df.index[1], "Dividend"] = None

        with self.assertRaisesRegex(ValueError, "Dividend"):
            build_dividend_parity_frame_df(
                capital_price_df=capital_price_df,
                total_return_price_df=total_return_price_df,
            )

    def test_calendar_mismatch_fails_loud(self):
        capital_price_df, total_return_price_df = self.make_price_frame_tuple()

        with self.assertRaisesRegex(ValueError, "calendars differ"):
            build_dividend_parity_frame_df(
                capital_price_df=capital_price_df,
                total_return_price_df=total_return_price_df.iloc[:-1],
            )


if __name__ == "__main__":
    unittest.main()
