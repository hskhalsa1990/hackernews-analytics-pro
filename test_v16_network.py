import unittest
import pandas as pd
from hn_network_v16 import _weighted_pagerank, _components, build_v16_network

class V16NetworkTests(unittest.TestCase):
    def test_pagerank_is_normalized(self):
        nodes=[{'id':'a'},{'id':'b'},{'id':'c'}]
        links=[{'source':'a','target':'b','weight':2},{'source':'b','target':'c','weight':1}]
        scores=_weighted_pagerank(nodes,links)
        self.assertAlmostEqual(sum(scores.values()),1.0,places=9)
        self.assertTrue(all(v>=0 for v in scores.values()))

    def test_components_threshold(self):
        authors=['a','b','c']
        overlaps=pd.DataFrame([{'author_a':'a','author_b':'b','weighted_jaccard':0.8},
                               {'author_a':'b','author_b':'c','weighted_jaccard':0.1}])
        ids, groups=_components(authors,overlaps,0.5)
        self.assertEqual(ids['a'],ids['b'])
        self.assertNotEqual(ids['a'],ids['c'])
        self.assertEqual(len(groups),2)

    def test_build_v16_network(self):
        stories=pd.DataFrame([
            {'by':'alice','title':'Python data science tools','score':10},
            {'by':'alice','title':'Python machine learning models','score':20},
            {'by':'bob','title':'Python data engineering platform','score':8},
            {'by':'carol','title':'Space telescope astronomy mission','score':15},
            {'by':'carol','title':'Astronomy space science discovery','score':12},
            {'by':'dave','title':'Space mission telescope science','score':9},
        ])
        authors,topics,communities,graph,meta=build_v16_network(stories,n_clusters=2,backend='tfidf',similarity_threshold=0.0)
        self.assertEqual(len(authors),4)
        self.assertGreaterEqual(len(topics),1)
        self.assertAlmostEqual(authors.pagerank.sum()+topics.pagerank.sum(),1.0,places=7)
        self.assertEqual(len(graph['nodes']),len(authors)+len(topics))
        self.assertIn('not influence',graph['network_analysis']['interpretation'])

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            build_v16_network(pd.DataFrame([{'by':'a','title':'hello'}]),similarity_threshold=2)

if __name__=='__main__':
    unittest.main()
