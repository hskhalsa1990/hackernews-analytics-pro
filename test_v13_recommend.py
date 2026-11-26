import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_recommend_v13 import (
    recommend_stories, find_near_duplicates, export_v13_recommendations,
)


class V13RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "title": [
                "Python machine learning tools for data science",
                "Machine learning tools in Python for data science",
                "A guide to sourdough bread baking at home",
                "New database engine improves query performance",
            ],
            "by": ["a", "b", "c", "d"],
            "score": [100, 80, 50, 70],
        })

    def test_recommendations_rank_similar_story_first_and_exclude_seed(self):
        result = recommend_stories(self.frame, seed_index=0, top_k=3, min_similarity=0)
        self.assertEqual(int(result.iloc[0].source_row), 1)
        self.assertNotIn(0, result.source_row.tolist())
        self.assertTrue(result.similarity.between(0, 1).all())

    def test_duplicate_detection_finds_paraphrased_lexical_match(self):
        result = find_near_duplicates(self.frame, threshold=0.45)
        self.assertTrue(((result.left_row == 0) & (result.right_row == 1)).any())

    def test_export_outputs_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v13_recommendations(self.frame, tmp, seed_index=0)
            for key in ("recommendations_file", "duplicates_file", "summary_file"):
                self.assertTrue(Path(result[key]).exists())
            summary = json.loads(Path(result["summary_file"]).read_text())
            self.assertEqual(summary["version"], "13.0.0")

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError): recommend_stories(self.frame, seed_index=99)
        with self.assertRaises(ValueError): recommend_stories(self.frame, top_k=0)
        with self.assertRaises(ValueError): find_near_duplicates(self.frame, threshold=0)
        with self.assertRaises(ValueError): recommend_stories(pd.DataFrame({"title": [""]}))


if __name__ == "__main__":
    unittest.main()
