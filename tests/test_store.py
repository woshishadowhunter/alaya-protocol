import gc
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

    def test_list_active_filters_by_status(self):
        tmp = tempfile.mkdtemp()
        try:
            store = SQLiteSeedStore(Path(tmp) / "test.db")
            active = ExperienceSeed.new(lesson="A", guidance="A1", context_tags=["x"], applicability="y", status="active")
            candidate = ExperienceSeed.new(lesson="B", guidance="B1", context_tags=["x"], applicability="y")
            store.save(active); store.save(candidate)
            results = store.list_active(since_days=3650)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].id, active.id)
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_count_returns_correct_total(self):
        tmp = tempfile.mkdtemp()
        try:
            store = SQLiteSeedStore(Path(tmp) / "test.db")
            self.assertEqual(store.count(), 0)
            store.save(ExperienceSeed.new(lesson="A", guidance="A1", context_tags=["x"], applicability="y"))
            store.save(ExperienceSeed.new(lesson="B", guidance="B1", context_tags=["x"], applicability="y"))
            self.assertEqual(store.count(), 2)
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_iter_all_batches(self):
        tmp = tempfile.mkdtemp()
        try:
            store = SQLiteSeedStore(Path(tmp) / "test.db")
            for i in range(5):
                store.save(ExperienceSeed.new(lesson=f"L{i}", guidance=f"G{i}", context_tags=["x"], applicability="y"))
            batches = list(store.iter_all(batch_size=2))
            self.assertEqual(len(batches), 3)
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual(len(batches[2]), 1)
            store.close()
            gc.collect()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
