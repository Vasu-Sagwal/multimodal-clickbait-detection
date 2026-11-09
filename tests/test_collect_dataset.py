import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import collect_dataset as collector


class InputTests(unittest.TestCase):
    def test_reads_current_csv_headers_and_deduplicates_video_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submissions.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["url", "category", "label"])
                w.writeheader()
                w.writerow({"url":"https://youtu.be/abcdefghijk", "category":"Gaming", "label":"clickbait"})
                w.writerow({"url":"https://youtube.com/watch?v=abcdefghijk", "category":"Gaming", "label":"1"})
            rows, report = collector.load_csv(path, ["Gaming"])
        self.assertEqual([r["video_id"] for r in rows], ["abcdefghijk"])
        self.assertEqual(rows[0]["label"], 1)
        self.assertIn("duplicate", report["issues"][0]["issue"])

    def test_rejects_invalid_video_url_without_dropping_other_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submissions.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["video_url", "category", "label"])
                w.writeheader()
                w.writerow({"video_url":"bad", "category":"Gaming", "label":"0"})
                w.writerow({"video_url":"https://youtu.be/abcdefghijk", "category":"Gaming", "label":"0"})
            rows, report = collector.load_csv(path, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(report["issues"]), 1)

    def test_preserves_categories_missing_from_old_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submissions.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["video_url", "category", "label"])
                w.writeheader()
                w.writerow({"video_url":"https://youtu.be/abcdefghijk", "category":"Other / Uncategorized", "label":"0"})
            rows, report = collector.load_csv(path, ["Gaming"])
        self.assertEqual(rows[0]["category"], "Other / Uncategorized")
        self.assertEqual(len(report["issues"]), 0)
        self.assertEqual(len(report["warnings"]), 1)


class ApiTests(unittest.TestCase):
    def test_metadata_uses_batches_of_fifty_and_maps_by_id(self):
        calls = []
        def get(endpoint, params, cost=1):
            ids = params["id"].split(",")
            calls.append(ids)
            return {"items":[{"id":i,"snippet":{"title":i,"thumbnails":{}}} for i in ids]}
        ids = [f"id{i}" for i in range(101)]
        class Client:
            def get(self, *args, **kwargs): return get(*args, **kwargs)
        result, paused = collector.metadata_batches(Client(), ids)
        self.assertEqual([len(x) for x in calls], [50,50,1])
        self.assertEqual(result["id100"]["title"], "id100")
        self.assertIsNone(paused)

    def test_quota_ledger_stops_before_safety_buffer(self):
        state = {"quota":{"default":{"date":collector.quota_day(), "used":8}}}
        client = collector.YouTubeClient(["secret"], state, daily_limit=10, buffer=2)
        with self.assertRaises(collector.QuotaPaused):
            client.get("videos", {"id":"abcdefghijk"})


class CommentTests(unittest.TestCase):
    def test_comment_requests_are_separate_and_max_25(self):
        calls=[]
        def get(endpoint, params, cost=1):
            calls.append(params.copy())
            return {"items":[{"snippet":{"topLevelComment":{"snippet":{"textDisplay":"x"}}}}]}
        class Client:
            def get(self, *args, **kwargs): return get(*args, **kwargs)
        client=Client()
        top=collector.comments(client,"abcdefghijk","relevance",25)
        newest=collector.comments(client,"abcdefghijk","time",25)
        self.assertEqual([x["order"] for x in calls],["relevance","time"])
        self.assertTrue(all(x["maxResults"]=="25" for x in calls))
        self.assertEqual(top["count"],1)
        self.assertEqual(newest["count"],1)


if __name__ == "__main__":
    unittest.main()
