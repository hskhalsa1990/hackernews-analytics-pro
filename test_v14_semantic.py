import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_semantic_v14 import semantic_recommendations, discover_topics, export_v14_semantic


class SemanticV14Tests(unittest.TestCase):
    def setUp(self):
        self.stories = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "title": [
                "Python machine learning libraries improve model training",
                "New Python tools for machine learning model development",
                "Rust compiler performance and memory safety improvements",
                "How databases optimize SQL query execution",
                "Neural networks and deep learning model training",
                "Space telescope discovers a distant exoplanet",
            ],
            "by": ["a", "b", "c", "d", "e", "f"],
            "score": [10, 20, 15, 12, 30, 8],
        })

    def test_tfidf_semantic_recommendations(self):
        recs, meta = semantic_recommendations(self.stories, 0, top_k=3, min_similarity=0.0, backend="tfidf")
        self.assertEqual(len(recs), 3)
        self.assertNotIn(0, recs.source_row.tolist())
        self.assertEqual(meta["backend"], "tfidf-fallback")
        self.assertTrue(recs.semantic_similarity.between(-1, 1).all())

    def test_mmr_diversity_parameter_validated(self):
        with self.assertRaises(ValueError):
            semantic_recommendations(self.stories, diversity=1.5, backend="tfidf")

    def test_topic_clustering(self):
        topics, assignments, meta = discover_topics(self.stories, n_clusters=3, backend="tfidf")
        self.assertEqual(len(topics), 3)
        self.assertEqual(len(assignments), len(self.stories))
        self.assertEqual(assignments.topic_cluster.nunique(), 3)
        self.assertEqual(meta["backend"], "tfidf-fallback")

    def test_export_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v14_semantic(self.stories, tmp, n_clusters=3, backend="tfidf")
            for key in ("recommendations_file", "topics_file", "assignments_file", "summary_file"):
                self.assertTrue(Path(result[key]).exists())

    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            semantic_recommendations(self.stories, seed_index=99, backend="tfidf")


if __name__ == "__main__":
    unittest.main()
