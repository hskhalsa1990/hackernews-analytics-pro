import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Imported lazily or only by optional entry points; not required at import time.
OPTIONAL_MODULES = {"hn_dashboard.py", "hn_visualization.py"}
IMPORT_NAME = {"sklearn": "scikit-learn"}


class DeclaredDependencyTests(unittest.TestCase):
    def test_top_level_third_party_imports_are_declared(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M).group(1).lower()
        declared = set(re.findall(r'"([a-z0-9_.-]+)', block))
        missing = []
        for path in (ROOT / "hn_analytics").glob("*.py"):
            if path.name in OPTIONAL_MODULES:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                m = re.match(r"(?:import|from)\s+([A-Za-z_][\w]*)", line)  # column 0 only
                if not m or m.group(1) in sys.stdlib_module_names or m.group(1).startswith("hn_") or m.group(1) == "__future__":
                    continue
                name = IMPORT_NAME.get(m.group(1), m.group(1)).lower()
                if name not in declared:
                    missing.append((path.name, m.group(1)))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
