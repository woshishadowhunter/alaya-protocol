import gc
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from alaya.cli import main


class CliTests(unittest.TestCase):
    def test_plant_reinforce_activate_flow(self):
        tmp = tempfile.mkdtemp()
        try:
            db = str(Path(tmp) / "seeds.db")
            out = StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--db", db, "plant", "--lesson", "Align stakeholders first", "--guidance", "Map interests", "--tags", "project,stakeholder", "--applies", "multi-party project", "--source", "case-1", "--evidence", "Draft failed"]), 0)
            seed_id = json.loads(out.getvalue())["id"]
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["--db", db, "reinforce", seed_id, "--polarity", "support", "--source", "case-2", "--evidence", "Alignment worked"]), 0)
            activated = StringIO()
            with redirect_stdout(activated):
                self.assertEqual(main(["--db", db, "activate", "stakeholder project"]), 0)
            result = json.loads(activated.getvalue())
            self.assertEqual(result[0]["seed"]["status"], "active")
            self.assertIn("explanation", result[0])
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
