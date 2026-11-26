import unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from hn_data_loader import load_stories
from hn_cleaning import clean_stories
from hn_nlp import sentiment_analysis
from hn_reporting import export_report
class V4Tests(unittest.TestCase):
 def setUp(self): self.df=clean_stories(load_stories(ROOT/"demo_stories.jsonl"))
 def test_sentiment_columns(self):
  out=sentiment_analysis(self.df); self.assertIn("sentiment_label",out); self.assertTrue(set(out.sentiment_label)<= {"positive","negative","neutral"})
 def test_export_report(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   result=export_report(self.df,d); self.assertIn("kpis",result); self.assertTrue((Path(d)/"executive_summary.json").exists())
if __name__=="__main__": unittest.main()
