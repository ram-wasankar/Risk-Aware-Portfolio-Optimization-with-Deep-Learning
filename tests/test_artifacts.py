import json
from pathlib import Path
import unittest

import numpy as np
import polars as pl

from src.optimize import performance


@unittest.skipUnless(Path("results/metrics.json").exists(), "Run python -m src.run to generate study artifacts")
class ArtifactTests(unittest.TestCase):
    def test_saved_returns_reconcile_to_metrics(self):
        m = json.loads(Path("results/metrics.json").read_text())
        daily = pl.read_csv("results/daily_returns.csv")
        for name, expected in m["strategies"].items():
            data = daily.filter(pl.col("strategy")==name)["net_return"].to_numpy()
            self.assertEqual(len(data), 240)
            actual = performance(data)
            for key in actual:
                self.assertAlmostEqual(actual[key], expected[key], places=10)

    def test_base_strategy_trace_constraints(self):
        traces = json.loads(Path("results/weights.json").read_text())
        for row in traces:
            if row["strategy"] not in ["xgboost", "lstm", "transformer"]:
                continue
            self.assertFalse(row["fallback"])
            w = np.array(row["weights"])
            self.assertAlmostEqual(w.sum(), 1, places=7)
            self.assertGreaterEqual(w.min(), -1e-7)
            self.assertLessEqual(w.max(), .350001)
            self.assertLessEqual(w[:3].sum(), .650001)
            self.assertLessEqual(w[3:5].sum(), .650001)
            self.assertLessEqual(row["total_traded"], .300001)
            self.assertLessEqual(row["scenario_cvar"], .060001)

    def test_tracking_export_reconciles(self):
        m = json.loads(Path("results/metrics.json").read_text())
        tracking = json.loads(Path("results/tracking_export.json").read_text())
        self.assertEqual(len(tracking), 8)
        for run in tracking:
            self.assertEqual(run["status"], "FINISHED")
            self.assertEqual(run["run_id"], m["run_ids"][run["name"]])
            if run["name"] in m["strategies"]:
                for key, value in m["strategies"][run["name"]].items():
                    self.assertAlmostEqual(run["metrics"][key], value)
