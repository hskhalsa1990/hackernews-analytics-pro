import tempfile
import unittest
from pathlib import Path
import pandas as pd
from hn_predictive import train_virality_model, detect_engagement_anomalies, export_predictive_research

class PredictiveResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_json(Path(__file__).parent / "demo_stories.jsonl", lines=True)

    def test_classifier_metrics_and_holdout(self):
        _, metrics, predictions = train_virality_model(self.data)
        self.assertEqual(metrics["split"], "chronological 80/20")
        self.assertGreaterEqual(metrics["roc_auc"], 0)
        self.assertEqual(len(predictions), metrics["test_rows"])
        self.assertNotIn("score", metrics["predictors"])
        self.assertIn("high_score_probability", predictions)

    def test_anomaly_detector_returns_flags(self):
        result = detect_engagement_anomalies(self.data)
        self.assertIn("is_anomaly", result)
        self.assertTrue(result.is_anomaly.any())

    def test_export_writes_three_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_predictive_research(self.data, tmp)
            for key in ("metrics_file", "predictions_file", "anomalies_file"):
                self.assertTrue(Path(result[key]).exists())

    def test_small_dataset_rejected_for_training(self):
        with self.assertRaises(ValueError):
            train_virality_model(self.data.head(10))

if __name__ == "__main__":
    unittest.main()
