import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import collect_dataset as collector


class InputTests(unittest.TestCase):
    def test_reads_current_csv_headers_and_deduplicates_video_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
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
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
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
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
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
    def test_api_keys_map_to_projects_in_matching_order(self):
        with patch.dict("os.environ", {"YOUTUBE_API_KEYS":"dummy-a,dummy-b", "YOUTUBE_API_KEY_PROJECTS":"project-a,project-b"}, clear=False):
            creds = collector.api_credentials()
        self.assertEqual([project for _, project in creds], ["project-a", "project-b"])

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
        client = collector.YouTubeClient([("dummy-key", "default")], state, daily_limit=10, buffer=2)
        with self.assertRaises(collector.QuotaPaused):
            client.get("videos", {"id":"abcdefghijk"})

    def test_exhausted_project_is_skipped_and_other_project_is_used(self):
        state = {"quota":{"project-a":{"date":collector.quota_day(), "used":8, "total_used":8}}}
        client = collector.YouTubeClient([("dummy-a1","project-a"), ("dummy-a2","project-a"), ("dummy-b","project-b")], state, 10, 2)
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"items":[]}'
        with patch.object(collector, "urlopen", return_value=Response()) as request:
            self.assertEqual(client.get("videos", {"id":"abcdefghijk"}), {"items":[]})
        self.assertEqual(request.call_count, 1)
        self.assertEqual(state["quota"]["project-a"]["used"], 8)
        self.assertEqual(state["quota"]["project-b"]["used"], 1)

    def test_api_quota_error_fails_over_only_to_a_different_project(self):
        state={}
        client=collector.YouTubeClient([("dummy-a1","project-a"),("dummy-a2","project-a"),("dummy-b","project-b")],state,100,10)
        quota_error=collector.HTTPError("https://example.invalid",403,"Forbidden",None,io.BytesIO(b'{"error":{"errors":[{"reason":"quotaExceeded"}]}}'))
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return b'{"items":[]}'
        with patch.object(collector,"urlopen",side_effect=[quota_error,Response()]) as request:
            self.assertEqual(client.get("videos",{"id":"abcdefghijk"}),{"items":[]})
        self.assertEqual(request.call_count,2)
        self.assertEqual(state["quota"]["project-a"]["used"],1)
        self.assertEqual(state["quota"]["project-b"]["used"],1)

    def test_dataset_profile_reports_count_mismatches(self):
        rows = [{"category":"Gaming", "label":0}, {"category":"Gaming", "label":1}]
        report = {"warnings":[]}
        collector.validate_dataset_config(rows, report, {"target_total_videos":3, "category_counts":{"Gaming":1}, "label_counts":{"0":2,"1":1}})
        self.assertEqual(len(report["warnings"]), 3)


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


class PipelineIntegrationTests(unittest.TestCase):
    def test_small_csv_runs_and_resume_skips_completed_modalities(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root=Path(tmp)
            (root/"dataset_config.json").write_text(json.dumps({"expected_categories":["Gaming"],"target_total_videos":1,"category_counts":{"Gaming":1},"label_counts":{"0":1,"1":0}}),encoding="utf-8")
            with (root/"submissions.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=["video_url","category","label"]); w.writeheader()
                w.writerow({"video_url":"https://youtu.be/abcdefghijk","category":"Gaming","label":"0"})
            api_calls=[]
            def fake_get(_self, endpoint, params, cost=1):
                api_calls.append(endpoint)
                if endpoint=="videos":
                    return {"items":[{"id":"abcdefghijk","snippet":{"title":"Sample","thumbnails":{"high":{"url":"https://example.invalid/thumb.jpg"}}}}]}
                return {"items":[{"snippet":{"topLevelComment":{"snippet":{"textDisplay":"sample"}}}}]}
            with patch.object(collector,"ROOT",root), patch.object(collector,"MANIFEST",root/"videos.json"), patch.object(collector,"CONFIG",root/"dataset_config.json"), patch.object(collector,"STATE",root/"collection_state.json"), patch.object(collector.YouTubeClient,"get",fake_get), patch.object(collector,"thumbnail",return_value=("Thumbnail/abcdefghijk.jpg",None)), patch.object(collector,"collect_transcript",return_value={"status":"available","language":"en","segments":[],"error":None}), patch.dict("os.environ",{"YOUTUBE_API_KEYS":"dummy-key","YOUTUBE_API_KEY_PROJECTS":"project-a"},clear=False):
                self.assertEqual(collector.collect(Path("submissions.csv"),25,10000,100),0)
                record=json.loads((root/"videos.json").read_text(encoding="utf-8"))[0]
                self.assertEqual(record["category"],"Gaming")
                self.assertEqual(record["label"],0)
                self.assertEqual(record["comments_collected"],{"top":1,"newest":1})
                self.assertEqual(collector.collect(Path("submissions.csv"),25,10000,100),0)
            self.assertEqual(api_calls,["videos","commentThreads","commentThreads"])


if __name__ == "__main__":
    unittest.main()
