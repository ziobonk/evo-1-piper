"""Validate standalone JSON latency summaries."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inference_latency import LatencyStats


class LatencyStatsTests(unittest.TestCase):
    def test_metrics_have_independent_counts_and_ignore_invalid_measurements(self):
        stats = LatencyStats(("model_ms", "processing_ms", "missing_ms"))
        for record in ({"model_ms": 10, "processing_ms": 5}, {"model_ms": 30},
                       {"model_ms": float("nan"), "processing_ms": -1}):
            stats.observe(record)
        result = stats.summary()
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["metrics"]["model_ms"]["mean_ms"], 20)
        self.assertEqual(result["metrics"]["model_ms"]["min_ms"], 10)
        self.assertEqual(result["metrics"]["model_ms"]["max_ms"], 30)
        self.assertEqual(result["metrics"]["processing_ms"]["count"], 1)
        self.assertIsNone(result["metrics"]["missing_ms"]["mean_ms"])

    def test_atomic_report_is_one_json_document_and_can_be_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "latency.json"
            stats = LatencyStats(("model_ms",))
            stats.save(path, role="server", run_id="test")
            self.assertEqual(json.loads(path.read_text())["sample_count"], 0)
            stats.observe({"model_ms": 5})
            stats.save(path, role="server", run_id="test")
            report = json.loads(path.read_text())
            self.assertEqual(report["metrics"]["model_ms"]["mean_ms"], 5)
            self.assertEqual(report["run_id"], "test")
            self.assertEqual(len(list(path.parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
