import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hn_data_loader import load_stories
from hn_cleaning import clean_stories
from hn_analytics_core import build_kpis, analyze_stories
from hn_advanced import engagement_analysis, trend_analysis, compare_segments, statistical_summary

class PipelineTests(unittest.TestCase):
    def setUp(self):
        raw = load_stories(ROOT / "demo_stories.jsonl")
        self.df = clean_stories(raw)

    def test_demo_dataset_loads(self):
        self.assertGreater(len(self.df), 0)

    def test_title_words_are_extracted(self):
        self.assertGreater(int(self.df["title_word_count"].max()), 0)

    def test_kpis_have_expected_fields(self):
        kpis = build_kpis(self.df)
        self.assertIn("total_stories", kpis)
        self.assertEqual(kpis["total_stories"], len(self.df))

    def test_advanced_analysis_outputs(self):
        self.assertIn("authors", engagement_analysis(self.df))
        self.assertGreaterEqual(len(trend_analysis(self.df)), 1)
        self.assertIn("mean", compare_segments(self.df).columns)
        self.assertIn("score_iqr", statistical_summary(self.df))

    def test_analysis_tables_exist(self):
        result = analyze_stories(self.df)
        self.assertEqual(set(result), {"hourly", "domains", "title_lengths", "daily"})

if __name__ == "__main__":
    unittest.main()
