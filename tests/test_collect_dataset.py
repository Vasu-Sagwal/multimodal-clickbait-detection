import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import collect_dataset as collector


class NormalizationTests(unittest.TestCase):
    def test_labels(self):
        for value, expected in (("  NORMAL ", 0), ("ClickBait", 1), (" 0 ", 0), ("1", 1)):
            self.assertEqual(collector.normalize_label(value), expected)
        with self.assertRaises(collector.CollectionError):
            collector.normalize_label("other")

    def test_categories_and_video_url_formats(self):
        categories = ["Technology"]
        self.assertEqual(collector.validate_row({"video_url":"https://youtu.be/abcdefghijk", "category":" technology ", "label":"0", "reviewer":"R"}, 2, categories), ("abcdefghijk", "Technology", 0))
        expected = "abcdefghijk"
        for url in ("https://youtube.com/watch?v=abcdefghijk", "https://www.youtube.com/shorts/abcdefghijk", "https://youtube.com/embed/abcdefghijk", "https://youtube.com/live/abcdefghijk"):
            self.assertEqual(collector.youtube_video_id(url), expected)


class CommentTests(unittest.TestCase):
    def test_separate_orders_and_limits(self):
        calls = []

        def api(endpoint, params):
            calls.append(params.copy())
            order = params["order"]
            offset = 0 if "pageToken" not in params else 2
            return {"items": [{"snippet":{"topLevelComment":{"snippet":{"textDisplay":f"{order}-{offset+i}", "publishedAt":"2020-01-01T00:00:00Z", "likeCount":i}}}} for i in range(min(2, 3-offset))], **({"nextPageToken":"next"} if offset == 0 else {})}

        with patch.object(collector, "api_get", side_effect=api):
            top = collector.collect_comments("abcdefghijk", "secret", 3, "relevance")
            newest = collector.collect_comments("abcdefghijk", "secret", 3, "time")
        self.assertEqual(top["count"], 3)
        self.assertEqual(newest["count"], 3)
        self.assertEqual({call["order"] for call in calls}, {"relevance", "time"})
        self.assertTrue(all(int(call["maxResults"]) <= 20 for call in calls))

    def test_each_comment_collection_is_capped_at_twenty(self):
        calls = []

        def api(endpoint, params):
            calls.append(params)
            return {"items": [{"snippet":{"topLevelComment":{"snippet":{"textDisplay":str(i)}}}} for i in range(20)]}

        with patch.object(collector, "api_get", side_effect=api):
            comments = collector.collect_comments("abcdefghijk", "dummy", 100, "relevance")
        self.assertEqual(comments["count"], 20)
        self.assertEqual(int(calls[0]["maxResults"]), 20)


class CollectionPreflightTests(unittest.TestCase):
    def test_csv_duplicates_skipped_and_existing_skipped_by_default(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            manifest = root / "videos.json"
            config = root / "dataset_config.json"
            config.write_text(json.dumps({"expected_categories":["Technology"]}), encoding="utf-8")
            manifest.write_text(json.dumps([{"video_id":"lmnopqrstuv", "label":0, "label_reviewer":"old", "label_notes":"keep"}]), encoding="utf-8")
            source = root / "rows.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_url","category","label","reviewer","notes"])
                writer.writeheader()
                writer.writerow({"video_url":"https://youtu.be/abcdefghijk", "category":"Technology", "label":"normal", "reviewer":"A"})
                writer.writerow({"video_url":"https://youtube.com/watch?v=abcdefghijk", "category":"Technology", "label":"1", "reviewer":"B"})
                writer.writerow({"video_url":"https://youtu.be/lmnopqrstuv", "category":" technology ", "label":"0", "reviewer":"C"})
            with patch.object(collector, "ROOT", root), patch.object(collector, "MANIFEST", manifest), patch.object(collector, "CONFIG", config), patch.dict("os.environ", {"YOUTUBE_API_KEY":"dummy"}), patch.object(collector, "fetch_video") as fetch:
                code = collector.collect(source, 20, False)
            self.assertEqual(code, 0)
            fetch.assert_not_called()


class AuditTests(unittest.TestCase):
    def test_audit_integer_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            categories = [f"Category{i}" for i in range(10)]
            (root / "dataset_config.json").write_text(json.dumps({"expected_categories":categories, "videos_per_category":2, "labels_per_category":{"0":1,"1":1}}), encoding="utf-8")
            (root / "videos.json").write_text(json.dumps([{"category":cat,"label":label} for cat in categories for label in (0,1)]), encoding="utf-8")
            with patch.object(collector, "ROOT", root), patch.object(collector, "CONFIG", root / "dataset_config.json"), patch.object(collector, "MANIFEST", root / "videos.json"), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(collector.run_audit(), 0)
            self.assertIn("normal=0: 1, clickbait=1: 1", output.getvalue())


if __name__ == "__main__":
    unittest.main()

