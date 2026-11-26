import unittest
import pandas as pd
from hn_robustness_v17 import _betweenness, build_v17_robustness, export_v17_robustness


def _stories():
    rows = [('alice', 'Python data science tools', 10), ('alice', 'Python machine learning models', 20),
            ('bob', 'Python data engineering platform', 8), ('carol', 'Space telescope astronomy mission', 15),
            ('carol', 'Astronomy space science discovery', 12), ('dave', 'Space mission telescope science', 9),
            ('erin', 'Python space data science', 11), ('erin', 'Telescope astronomy python tools', 7)]
    return pd.DataFrame([{'by': a, 'title': t, 'score': s} for a, t, s in rows])


class V17RobustnessTests(unittest.TestCase):
    def test_betweenness_path_center(self):
        nodes = [{'id': x} for x in 'abc']
        links = [{'source': 'a', 'target': 'b'}, {'source': 'b', 'target': 'c'}]
        bt = _betweenness(nodes, links)
        self.assertAlmostEqual(bt['b'], 1.0); self.assertEqual(bt['a'], 0.0)

    def test_build_is_deterministic_and_bounded(self):
        a1, s1, t1, m1 = build_v17_robustness(_stories(), n_clusters=2, repeats=20, top_k=2, seed=7)
        a2, *_ = build_v17_robustness(_stories(), n_clusters=2, repeats=20, top_k=2, seed=7)
        pd.testing.assert_frame_equal(a1, a2)
        self.assertEqual(len(a1), 5)
        self.assertTrue(((a1.top_k_frequency >= 0) & (a1.top_k_frequency <= 1)).all())
        self.assertTrue((a1.ci_low <= a1.ci_high).all())
        self.assertEqual(len(s1), 4)
        self.assertIn('not influence', m1['interpretation'])

    def test_invalid_args(self):
        with self.assertRaises(ValueError): build_v17_robustness(_stories(), repeats=0)
        with self.assertRaises(ValueError): build_v17_robustness(_stories(), damping=1.5)

    def test_export(self, ):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            r = export_v17_robustness(_stories(), d, n_clusters=2, repeats=10, top_k=2)
            self.assertTrue(os.path.exists(r['summary_file']))
if __name__ == '__main__':
    unittest.main()
