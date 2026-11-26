import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hn_graph_v15 import build_author_topic_graph, export_v15_graph


class GraphV15Tests(unittest.TestCase):
    def setUp(self):
        self.stories = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "title": [
                "Python machine learning tools and libraries",
                "Python machine learning model training",
                "Rust compiler memory safety and speed",
                "Rust systems programming compiler tools",
                "SQL database query optimization techniques",
                "Database indexing and SQL performance",
                "Neural networks deep learning research",
                "Neural network model training research",
            ],
            "by": ["alice", "alice", "bob", "bob", "alice", "cara", "cara", "dave"],
            "score": [10, 20, 15, 18, 12, 9, 30, 25],
        })
        self.assignments = pd.DataFrame({"topic_cluster": [0, 0, 1, 1, 2, 2, 3, 3]})

    def test_bipartite_graph_and_author_metrics(self):
        authors, topics, edges, overlaps, graph, meta = build_author_topic_graph(
            self.stories, self.assignments)
        self.assertEqual(len(authors), 4)
        self.assertEqual(len(topics), 4)
        self.assertEqual(len(edges), 6)
        self.assertTrue((authors.story_count >= 1).all())
        self.assertTrue(authors.normalized_topic_entropy.between(0, 1).all())
        self.assertEqual(meta["author_nodes"], 4)
        self.assertEqual(len(graph["nodes"]), 8)
        self.assertIn("not collaboration", graph["semantics"])

    def test_author_overlap_uses_weighted_jaccard(self):
        _, _, _, overlaps, _, _ = build_author_topic_graph(self.stories, self.assignments)
        pair = overlaps[(overlaps.author_a == "alice") & (overlaps.author_b == "bob")]
        self.assertTrue(pair.empty)  # Their assigned topic sets do not overlap.
        pair2 = overlaps[(overlaps.author_a == "alice") & (overlaps.author_b == "cara")]
        self.assertEqual(len(pair2), 1)
        self.assertGreater(pair2.iloc[0].weighted_jaccard, 0)

    def test_missing_author_rows_are_excluded(self):
        data = self.stories.copy()
        data.loc[0, "by"] = None
        authors, topics, edges, overlaps, graph, meta = build_author_topic_graph(data, self.assignments)
        self.assertEqual(meta["stories_with_author_and_topic"], 7)
        self.assertNotIn("author:None", [n["id"] for n in graph["nodes"]])

    def test_assignment_length_validation(self):
        with self.assertRaises(ValueError):
            build_author_topic_graph(self.stories, self.assignments.iloc[:-1])

    def test_export_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_v15_graph(self.stories, tmp, n_clusters=3, backend="tfidf")
            for key in ("authors_file", "topics_file", "edges_file", "overlaps_file", "graph_file", "summary_file"):
                self.assertTrue(Path(result[key]).exists())
            graph = json.loads(Path(result["graph_file"]).read_text())
            self.assertIn("nodes", graph)
            self.assertIn("links", graph)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            build_author_topic_graph(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
