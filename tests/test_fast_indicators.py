import unittest

import numpy as np
import pandas as pd

from alpha.indicators import dv2_indicator as public_dv2_indicator
from alpha.indicators import dv2_indicator_reference as public_dv2_indicator_reference
from alpha.indicators import qp_indicator as public_qp_indicator
from alpha.indicators import qp_indicator_reference as public_qp_indicator_reference
from alpha.engine.dv2_indicator_fast import dv2_indicator_fast
from alpha.engine.indicators import dv2_indicator, qp_indicator
from alpha.engine.qp_indicator_fast import qp_indicator_fast


class FastIndicatorTests(unittest.TestCase):
    def test_dv2_indicator_fast_matches_reference_and_preserves_warmup(self):
        date_index = pd.date_range("2020-01-01", periods=40, freq="D")
        base_price_arr = np.linspace(100.0, 120.0, len(date_index))
        close_price_ser = pd.Series(base_price_arr, index=date_index, dtype=float)
        high_price_ser = close_price_ser + 1.5
        low_price_ser = close_price_ser - 1.5
        close_price_ser.iloc[7] = np.nan
        high_price_ser.iloc[7] = np.nan
        low_price_ser.iloc[7] = np.nan
        close_price_ser.iloc[20:24] = 111.0
        high_price_ser.iloc[20:24] = 112.5
        low_price_ser.iloc[20:24] = 109.5

        reference_value_ser = dv2_indicator(
            close_price_ser,
            high_price_ser,
            low_price_ser,
            length=8,
        )
        fast_value_ser = dv2_indicator_fast(
            close_price_ser,
            high_price_ser,
            low_price_ser,
            length_int=8,
        )

        pd.testing.assert_index_equal(fast_value_ser.index, close_price_ser.index)
        self.assertTrue(fast_value_ser.iloc[:8].isna().all())
        pd.testing.assert_series_equal(
            fast_value_ser,
            reference_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

        public_fast_value_ser = public_dv2_indicator(
            close_price_ser,
            high_price_ser,
            low_price_ser,
            length_int=8,
        )
        public_reference_value_ser = public_dv2_indicator_reference(
            close_price_ser,
            high_price_ser,
            low_price_ser,
            length_int=8,
        )

        pd.testing.assert_series_equal(
            public_fast_value_ser,
            fast_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )
        pd.testing.assert_series_equal(
            public_reference_value_ser,
            reference_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

    def test_qp_indicator_fast_matches_reference_and_preserves_nan_semantics(self):
        date_index = pd.date_range("2015-01-01", periods=420, freq="D")
        close_price_arr = 100.0 + np.sin(np.arange(len(date_index)) / 9.0) * 3.0
        close_price_ser = pd.Series(close_price_arr, index=date_index, dtype=float)
        close_price_ser.iloc[13] = np.nan
        close_price_ser.iloc[150:155] = 101.0

        reference_value_ser = qp_indicator(
            close_price_ser,
            window=3,
            lookback_years=1,
        )
        fast_value_ser = qp_indicator_fast(
            close_price_ser,
            window_int=3,
            lookback_years_int=1,
        )

        pd.testing.assert_index_equal(fast_value_ser.index, close_price_ser.index)
        self.assertTrue(fast_value_ser.iloc[:252].isna().all())
        pd.testing.assert_series_equal(
            fast_value_ser,
            reference_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )

        public_fast_value_ser = public_qp_indicator(
            close_price_ser,
            window_int=3,
            lookback_years_int=1,
        )
        public_reference_value_ser = public_qp_indicator_reference(
            close_price_ser,
            window_int=3,
            lookback_years_int=1,
        )

        pd.testing.assert_series_equal(
            public_fast_value_ser,
            fast_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )
        pd.testing.assert_series_equal(
            public_reference_value_ser,
            reference_value_ser,
            check_names=False,
            check_exact=False,
            atol=1e-12,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
