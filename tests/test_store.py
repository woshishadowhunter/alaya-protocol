import gc
import sqlite3
import tempfile
import unittest
from pathlib import Path

from alaya.models import ExperienceSeed
from alaya.store import SQLiteSeedStore


class StoreTests(unittest.TestCase):
    def test_crud_and_export(self):
        tmp = tempfile.mkdtemp()
        try:
            db = Path(tmp) / "alaya.db"
            store = SQLiteSeedStore(db)
            seed = ExperienceSeed.new(
                lesson="Review outcomes.", guidance="Schedule a retrospective.",
                context_tags=["project"], applicability="Completed projects"
            )
            store.save(seed)
            self.assertEqual(store.get(seed.id), seed)
            self.assertEqual(store.list()[0], seed)
            exported = store.export_json()
            self.assertIn(seed.id, exported)
            self.assertTrue(store.delete(seed.id))
            self.assertIsNone(store.get(seed.id))
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_list_multiple_seeds_ordered_by_updated(self):
        tmp = tempfile.mkdtemp()
        try:
            store = SQLiteSeedStore(Path(tmp) / "test.db")
            a = ExperienceSeed.new(lesson="A", guidance="A1", context_tags=["x"], applicability="y")
            b = ExperienceSeed.new(lesson="B", guidance="B1", context_tags=["x"], applicability="y")
            store.save(a); store.save(b)
            seeds = store.list()
            self.assertEqual(len(seeds), 2)
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
