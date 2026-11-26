import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from hn_research_v8 import build_topic_network, explain_classifier, export_v8_research

class V8ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_json(Path(__file__).parent / "demo_stories.jsonl", lines=True)

    def test_topic_network_schema_and_edges(self):
        sample = pd.DataFrame({"title": ["Python data science tools", "Python data science research", "Rust systems programming"]})
        graph = build_topic_network(sample, min_cooccurrence=2)
        self.assertEqual(graph["documents"], 3)
        self.assertTrue(any(e["source"] == "data" and e["target"] == "python" for e in graph["edges"]))

    def test_explainability_has_raw_feature_names(self):
        result = explain_classifier(self.data, repeats=2)
        self.assertIn("feature", result)
        self.assertIn("importance_mean", result)
        self.assertNotIn("score", result.feature.tolist())

    def test_export_writes_all_v8_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_v8_research(self.data, tmp, repeats=2)
            for name in summary["artifacts"].values():
                self.assertTrue(Path(name).exists())
            payload = json.loads((Path(tmp) / "v8_topic_network.json").read_text())
            self.assertIn("nodes", payload); self.assertIn("edges", payload)

if __name__ == "__main__":
    unittest.main()
