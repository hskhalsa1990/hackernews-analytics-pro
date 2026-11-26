import unittest
from pathlib import Path
try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Streamlit is an optional dependency
    AppTest = None

DASH = Path(__file__).resolve().parent / "hn_dashboard.py"


@unittest.skipIf(AppTest is None, "streamlit not installed")
class DashboardSmokeTests(unittest.TestCase):
    def test_app_loads_without_exception(self):
        at = AppTest.from_file(str(DASH), default_timeout=60).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.tabs), 15)
