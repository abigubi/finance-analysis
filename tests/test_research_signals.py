"""Run: python -m unittest discover -s tests -p test_research_signals.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Quant"))
from research_signals import backtest, breadth, compute_features, normalize_prices, prior_z


def sample_prices(n=1000):
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, 0.017, n)
    close = 100 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Close": close, "Open": close * 1.0002,
        "High": close * 1.01, "Low": close * 0.99,
        "Volume": rng.integers(1_000_000, 9_000_000, n),
    }, index=idx)


class ResearchSignalTests(unittest.TestCase):
    def test_no_future_observations_affect_past_features(self):
        a = sample_prices()
        b = a.copy()
        # Rewrite future history after the 750th row. Nothing before then changes.
        b.loc[b.index[750]:, "Close"] *= 1.9
        for key in ("Open", "High", "Low"):
            b.loc[b.index[750]:, key] *= 1.9
        one = compute_features(a)
        two = compute_features(b)
        for col in ("Sharpe", "SharpeZ", "ReturnZ", "Return20Z",
                    "VolumeZ", "PanicDays6", "Curvature",
                    "EntryCross", "ExitCross", "Capitulation"):
            pd.testing.assert_series_equal(
                one[col].iloc[:750], two[col].iloc[:750], check_names=False,
            )

    def test_sharpe_baseline_uses_prior_observations(self):
        result = compute_features(sample_prices())
        sharpe = result["Sharpe"]
        mean = result["SharpeMean"]
        i = mean.first_valid_index()
        self.assertIsNotNone(i)
        pos = result.index.get_loc(i)
        expected = sharpe.iloc[pos-252:pos].mean()
        self.assertAlmostEqual(mean.iloc[pos], expected, places=10)

    def test_normalization_excludes_current_observation(self):
        s = pd.Series([1, 1, 1, 1, 2, 3, 4, 5, 100.])
        z = prior_z(s, 4)
        baseline = s.iloc[4:8]
        self.assertAlmostEqual(z.iloc[8],
                               (100 - baseline.mean()) / baseline.std(), places=10)

    def test_next_session_open_execution(self):
        idx = pd.bdate_range("2026-01-01", periods=6)
        frame = pd.DataFrame({
            "SharpeZ": [0, -2.5, -2.3, -1.0, 2.5, 2.0],
            "Open": [100, 101, 102, 103, 104, 105],
        }, index=idx)
        trades = backtest(frame, entry_z=-2, exit_z=2,
                          max_hold=30, cost_bps_per_side=0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Entry price"], 102)
        self.assertEqual(trades.iloc[0]["Exit price"], 105)
        self.assertEqual(trades.iloc[0]["Holding days"], 3)
        self.assertAlmostEqual(trades.iloc[0]["Return %"], (105/102-1)*100)

    def test_adjusted_ohlcv_multiindex(self):
        data = sample_prices(10)
        multi = data.copy()
        multi.columns = pd.MultiIndex.from_product([data.columns, ["MSFT"]])
        result = normalize_prices(multi, "MSFT")
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result["Close"].iloc[-1], data["Close"].iloc[-1])

    def test_breadth_uses_observed_not_future_prices(self):
        idx = pd.bdate_range("2026-01-01", periods=4)
        close = pd.DataFrame({"AAA": [10, 11, 12, 11],
                              "BBB": [10, 9, 10, 11]}, index=idx)
        b = breadth(close)
        self.assertAlmostEqual(b["AD Ratio"].iloc[1], 0)
        self.assertAlmostEqual(b["AD Ratio"].iloc[2], 1)
        self.assertAlmostEqual(b["AD Ratio"].iloc[3], 0)


if __name__ == "__main__":
    unittest.main()
