"""v18.4 doctor tests."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hn_doctor as doctor


class DoctorTests(unittest.TestCase):
    def test_python_version_gate(self):
        self.assertEqual(doctor.check_python((3, 12, 0))[1], "ok")
        self.assertEqual(doctor.check_python((3, 9, 18))[1], "fail")

    def test_missing_required_package_is_failure(self):
        with mock.patch.object(doctor, "_has", lambda m: m != "requests"):
            results = {n: s for n, s, _ in doctor.check_packages()}
        self.assertEqual(results["required:requests"], "fail")
        self.assertEqual(results["required:pandas"], "ok")

    def test_missing_optional_package_is_warning_only(self):
        with mock.patch.object(doctor, "_has", lambda m: m not in doctor.OPTIONAL):
            _, failed = doctor.run_checks("/nonexistent.sqlite", "/nonexistent.jsonl")
        self.assertFalse(failed)

    def test_database_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "h.sqlite"
            self.assertEqual(doctor.check_database(db)[1], "warn")  # absent
            with sqlite3.connect(db) as c:
                c.execute("CREATE TABLE story_snapshots (id INTEGER, observed_at TEXT, rank INTEGER)")
            self.assertEqual(doctor.check_database(db)[1], "ok")
            with sqlite3.connect(db) as c:
                c.execute("DROP TABLE story_snapshots")
                c.execute("CREATE TABLE story_snapshots (id INTEGER, observed_at TEXT)")
            self.assertEqual(doctor.check_database(db)[1], "warn")  # legacy schema
            db.write_text("not a sqlite file at all")
            self.assertEqual(doctor.check_database(db)[1], "fail")


if __name__ == "__main__":
    unittest.main()
