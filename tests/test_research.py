import unittest

import numpy as np
import polars as pl
import torch

from src.data import FOLDS, HORIZON, features, market, samples, training_mask
from src.models import SequenceModel
from src.optimize import empirical_cvar, hold, optimize, performance, scenarios


class ResearchTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        a, x = market()
        b, y = market()
        np.testing.assert_array_equal(x, y)
        self.assertTrue(a.equals(b))

    def test_future_changes_cannot_change_past_features(self):
        frame, _ = market()
        changed = frame.with_columns(pl.when(pl.col("day")>200).then(pl.lit(.99)).otherwise(pl.col("return")).alias("return"))
        a, b = features(frame), features(changed)
        self.assertTrue(a.filter(pl.col("day")<=200).equals(b.filter(pl.col("day")<=200)))

    def test_purge_removes_overlapping_labels(self):
        f, r = market()
        _, _, days, _ = samples(f, r)
        for start, _ in FOLDS:
            train = training_mask(days, start)
            self.assertLess((days[train]+HORIZON).max(), start)
            self.assertFalse(train[days==start-HORIZON].any())

    def test_target_uses_next_five_days(self):
        f, r = market()
        x, y, days, assets = samples(f, r)
        d, a = days[0], assets[0]
        self.assertAlmostEqual(y[0, 0]/100, np.prod(1+r[d+1:d+6, a])-1, places=7)
        self.assertTrue(np.isfinite(x).all())

    def test_sequence_models_have_two_output_heads(self):
        for kind in ["lstm", "transformer"]:
            out = SequenceModel(kind)(torch.zeros(4, 12, 9))
            self.assertEqual(tuple(out.shape), (4, 2))
            self.assertTrue(torch.isfinite(out).all())

    def test_cvar_discrete_tail(self):
        self.assertAlmostEqual(empirical_cvar([1, 2, 3, 4], .5), 3.5)
        self.assertAlmostEqual(empirical_cvar([1, 2, 3, 4], .95), 4.)

    def test_scenario_means_and_volatility_match_forecasts(self):
        history = np.random.default_rng(2).normal(size=(40, 6))
        mu, vol = np.arange(6)*.001, np.linspace(.01, .03, 6)
        result = scenarios(history, mu, vol)
        np.testing.assert_allclose(result.mean(axis=0), mu, atol=1e-12)
        np.testing.assert_allclose(result.std(axis=0, ddof=1), vol)

    def test_optimizer_respects_exposures_and_turnover(self):
        previous = np.ones(6)/6
        s = np.random.default_rng(3).normal(0, .01, size=(40, 6))
        w, info = optimize(np.arange(6)*.001, s, previous)
        self.assertFalse(info["fallback"])
        self.assertAlmostEqual(w.sum(), 1, places=7)
        self.assertGreaterEqual(w.min(), -1e-7)
        self.assertLessEqual(w.max(), .350001)
        self.assertLessEqual(w[:3].sum(), .650001)
        self.assertLessEqual(w[3:5].sum(), .650001)
        self.assertLessEqual(np.abs(w-previous).sum(), .300001)
        self.assertLessEqual(empirical_cvar(-s @ w), .060001)

    def test_infeasible_cvar_retains_holdings_and_reports_failure(self):
        previous = np.ones(6)/6
        w, info = optimize(np.zeros(6), np.full((20, 6), -.5), previous)
        self.assertTrue(info["fallback"])
        np.testing.assert_array_equal(w, previous)

    def test_cost_reduces_wealth_and_weights_drift(self):
        weights = np.ones(6)/6
        daily = np.zeros((5, 6)); daily[:, 0] = .01
        gross, drift = hold(weights, daily)
        net, _ = hold(weights, daily, fee=.001)
        self.assertLess(np.prod(1+net), np.prod(1+gross))
        self.assertGreater(drift[0], weights[0])
        self.assertAlmostEqual(drift.sum(), 1.)

    def test_drawdown_includes_initial_wealth(self):
        m = performance([-.10, .01, .02])
        self.assertAlmostEqual(m["max_drawdown"], -.10)


if __name__ == "__main__":
    unittest.main()
