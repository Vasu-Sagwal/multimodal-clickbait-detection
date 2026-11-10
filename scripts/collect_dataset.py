#!/usr/bin/env python3
"""Resumable extraction of YouTube modalities for the current submissions CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta, time as datetime_time
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "videos.json"
CONFIG = ROOT / "dataset_config.json"
STATE = ROOT / "collection_state.json"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
LABELS = {"normal": 0, "clickbait": 1, "0": 0, "1": 1}
MODALITIES = ("metadata", "thumbnail", "transcript", "comments_top")
interrupt_requested = False


class CollectionError(RuntimeError):
    pass


class QuotaPaused(CollectionError):
    pass


def load_env_file() -> None:
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_keys() -> list[str]:
    raw = os.getenv("YOUTUBE_API_KEYS", "") or os.getenv("YOUTUBE_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip() and k.strip() != "replace_me"]


def api_credentials() -> list[tuple[str, str]]:
    keys = api_keys()
    if not keys:
        return []
    raw_projects = os.getenv("YOUTUBE_API_KEY_PROJECTS", "").strip()
    if raw_projects:
        projects = [value.strip() for value in raw_projects.split(",")]
        if len(projects) != len(keys) or any(not value for value in projects):
            raise CollectionError("YOUTUBE_API_KEY_PROJECTS must list one non-empty project ID per key, in matching order.")
    else:
        default_project = os.getenv("YOUTUBE_QUOTA_PROJECT", "default").strip() or "default"
        projects = [default_project] * len(keys)
    credentials, assigned = [], {}
    for key, project in zip(keys, projects):
        if key in assigned and assigned[key] != project:
            raise CollectionError("The same API key cannot be assigned to multiple project IDs.")
        assigned[key] = project
        if (key, project) not in credentials:
            credentials.append((key, project))
    return credentials


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"Cannot read {path.name}: {exc}") from exc


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def video_id_from(value: str) -> str:
    parsed = urlparse((value or "").strip())
    host = (parsed.hostname or "").lower()
    vid = ""
    if host == "youtu.be":
        vid = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            bits = [p for p in parsed.path.split("/") if p]
            if len(bits) >= 2 and bits[0] in {"embed", "shorts", "live"}:
                vid = bits[1]
    if not VIDEO_ID_RE.fullmatch(vid):
        raise CollectionError("not a valid YouTube video URL")
    return vid


def load_csv(path: Path, categories: list[str]) -> tuple[list[dict], dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers or not any(h in headers for h in ("video_url", "url", "video_id")):
            raise CollectionError("CSV needs a video_url, url, or video_id column.")
        entries, issues, warnings, seen = [], [], [], set()
        submitted_rows = 0
        canonical = {str(c).strip().casefold(): c for c in categories}
        for line, row in enumerate(reader, 2):
            submitted_rows += 1
            try:
                raw_id = (row.get("video_id") or "").strip()
                vid = raw_id if VIDEO_ID_RE.fullmatch(raw_id) else video_id_from(row.get("video_url") or row.get("url") or "")
                if vid in seen:
                    issues.append({"row": line, "video_id": vid, "issue": "duplicate video id; only first occurrence used"})
                    continue
                seen.add(vid)
                cat = (row.get("category") or "").strip()
                label = (row.get("label") or "").strip()
                if not cat or not label:
                    raise CollectionError("category and label columns are required")
                if categories and cat.casefold() not in canonical:
                    warnings.append({"row": line, "video_id": vid, "issue": f"category {cat!r} is not listed in dataset_config.json; preserving CSV value"})
                if label.casefold() not in LABELS:
                    raise CollectionError("label must be 0/normal or 1/clickbait")
                entries.append({"video_id": vid, "video_url": f"https://www.youtube.com/watch?v={vid}",
                                "category": canonical.get(cat.casefold(), cat), "label": LABELS[label.casefold()],
                                "label_reviewer": (row.get("reviewer") or row.get("label_reviewer") or "").strip(),
                                "label_notes": (row.get("notes") or row.get("label_notes") or "").strip()})
            except CollectionError as exc:
                issues.append({"row": line, "issue": str(exc)})
    return entries, {"headers": headers, "submitted_rows": submitted_rows, "issues": issues, "warnings": warnings}


def validate_dataset_config(entries: list[dict], report: dict, config: dict) -> None:
    """Compare the immutable CSV with its checked-in dataset summary."""
    warnings = report.setdefault("warnings", [])
    target = config.get("target_total_videos")
    if target is not None and len(entries) != int(target):
        warnings.append({"issue":f"dataset has {len(entries)} unique valid videos; config expects {target}"})
    expected_categories = config.get("category_counts", {})
    if expected_categories:
        actual = {}
        for entry in entries:
            actual[entry["category"]] = actual.get(entry["category"], 0) + 1
        for category in sorted(set(expected_categories) | set(actual)):
            expected, observed = int(expected_categories.get(category, 0)), actual.get(category, 0)
            if expected != observed:
                warnings.append({"issue":f"category count for {category!r} is {observed}; config expects {expected}"})
    expected_labels = config.get("label_counts", {})
    for label in ("0", "1"):
        if label in expected_labels:
            observed = sum(entry["label"] == int(label) for entry in entries)
            if observed != int(expected_labels[label]):
                warnings.append({"issue":f"label {label} count is {observed}; config expects {expected_labels[label]}"})


class YouTubeClient:
    # videos.list costs 1 unit; commentThreads.list costs 1 unit.
    def __init__(self, keys: list[str] | list[tuple[str, str]], state: dict, daily_limit: int, buffer: int, checkpoint=None):
        default_project = os.getenv("YOUTUBE_QUOTA_PROJECT", "default")
        self.credentials = [(item, default_project) if isinstance(item, str) else item for item in keys]
        self.state = state
        self.daily_limit, self.buffer = daily_limit, buffer
        self.key_index = 0
        self.bad_keys: set[int] = set()
        self.exhausted_projects: set[str] = set()
        self.checkpoint = checkpoint

    def quota_for(self, project: str) -> dict:
        quota = self.state.setdefault("quota", {}).setdefault(project, {"date":quota_day(), "used":0, "total_used":0})
        quota.setdefault("total_used", quota.get("used", 0))
        if quota.get("date") != quota_day():
            quota.update(date=quota_day(), used=0)
        return quota

    def available_calls(self) -> int:
        projects = {project for _, project in self.credentials}
        return sum(max(0, self.daily_limit-self.buffer-self.quota_for(project).get("used", 0)) for project in projects)

    def get(self, endpoint: str, params: dict, cost: int = 1):
        errors = []
        saw_quota_exhaustion = False
        for _ in range(len(self.credentials)):
            idx = self.key_index % len(self.credentials)
            if idx in self.bad_keys:
                self.key_index = (idx + 1) % len(self.credentials)
                continue
            key, project = self.credentials[idx]
            if project in self.exhausted_projects:
                self.key_index = (idx + 1) % len(self.credentials)
                saw_quota_exhaustion = True
                continue
            quota = self.quota_for(project)
            if quota.get("used", 0) + cost > self.daily_limit-self.buffer:
                self.exhausted_projects.add(project)
                self.key_index = (idx + 1) % len(self.credentials)
                saw_quota_exhaustion = True
                continue
            # Keep using this key for successive requests. Advance only when
            # this key is invalid or its project's quota has run out.
            quota["used"] = quota.get("used", 0) + cost
            quota["total_used"] = quota.get("total_used", 0) + cost
            if self.checkpoint:
                self.checkpoint()
            query = dict(params, key=key)
            request = Request(f"https://www.googleapis.com/youtube/v3/{endpoint}?{urlencode(query)}", headers={"Accept":"application/json"})
            try:
                # Count every attempted request conservatively, including API errors.
                with urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data
            except HTTPError as exc:
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    reason = next((e.get("reason", "") for e in body.get("error", {}).get("errors", [])), "")
                except Exception:
                    reason = ""
                if reason in {"quotaExceeded", "dailyLimitExceeded", "dailyLimitExceededUnreg"}:
                    self.exhausted_projects.add(project)
                    self.key_index = (idx + 1) % len(self.credentials)
                    saw_quota_exhaustion = True
                    if self.checkpoint:
                        self.checkpoint()
                    continue
                if reason in {"keyInvalid", "authError", "ipRefererBlocked"} or exc.code == 401:
                    self.bad_keys.add(idx)
                    self.key_index = (idx + 1) % len(self.credentials)
                    errors.append(reason or f"HTTP {exc.code}")
                    if self.checkpoint:
                        self.checkpoint()
                    continue
                raise CollectionError(f"YouTube API {endpoint} request failed ({reason or f'HTTP {exc.code}'}).") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise CollectionError(f"YouTube API {endpoint} temporary request failure: {type(exc).__name__}.") from exc
        if saw_quota_exhaustion:
            raise QuotaPaused("All configured YouTube API projects reached their daily safety limit; progress saved for reset.")
        raise CollectionError("No usable YouTube API credentials remain: " + ", ".join(errors))


def metadata_batches(client: YouTubeClient, ids: list[str], on_batch=None) -> tuple[dict[str, dict], str | None]:
    output = {}
    error_seen = False
    for index, vid in enumerate(ids, start=1):
        try:
            data = client.get("videos", {"part":"snippet", "id":vid})
        except QuotaPaused:
            return output, "quota"
        except CollectionError as exc:
            print(f"Metadata request failed for video {vid}: {exc}", file=sys.stderr)
            error_seen = True
            continue
        item_output = {}
        for item in data.get("items", []):
            sn = item.get("snippet", {})
            item_output[item["id"]] = {"title": sn.get("title"), "description": sn.get("description"), "tags": sn.get("tags", []),
                "channel_title": sn.get("channelTitle"), "published_at": sn.get("publishedAt"), "category_id": sn.get("categoryId"),
                "thumbnails": sn.get("thumbnails", {})}
        output.update(item_output)
        if on_batch:
            on_batch([vid], item_output, index)
    return output, "error" if error_seen else None


def comments(client: YouTubeClient, vid: str, order: str, limit: int) -> dict:
    items, token, retries = [], None, 0
    while len(items) < limit:
        params = {"part":"snippet", "videoId":vid, "maxResults":str(min(25, limit-len(items))), "textFormat":"plainText", "order":order}
        if token: params["pageToken"] = token
        try:
            page = client.get("commentThreads", params)
        except QuotaPaused: raise
        except CollectionError as exc:
            msg = str(exc).lower()
            if retries < 2 and ("temporary" in msg or "request failed" in msg):
                time.sleep(2 ** retries)
                retries += 1
                continue
            status = "disabled" if "disabled" in msg else "error" if "temporary" in msg or "request failed" in msg else "unavailable"
            return {"status":status, "count":len(items), "items":items, "error":str(exc)}
        retries = 0
        for it in page.get("items", []):
            sn = it.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            items.append({"text":sn.get("textDisplay", ""), "published_at":sn.get("publishedAt"), "like_count":sn.get("likeCount", 0)})
            if len(items) >= limit: break
        token = page.get("nextPageToken")
        if not token or not page.get("items"): break
    return {"status":"available" if items else "empty", "count":len(items), "items":items, "error":None}


def thumbnail(vid: str, variants: dict) -> tuple[str | None, str | None]:
    order = ("maxres", "standard", "high", "medium", "default")
    root = ROOT / "Thumbnail"
    root.mkdir(parents=True, exist_ok=True)
    # Keep established .jpg naming and verify it is nonempty before reuse.
    dest = root / f"{vid}.jpg"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.relative_to(ROOT).as_posix(), None
    errors = []
    for quality in order:
        info = variants.get(quality, {})
        url = info.get("url")
        if not url: continue
        try:
            request = Request(url, headers={"User-Agent":"dataset-collector/1.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read()
            if not data: raise OSError("empty image response")
            tmp = dest.with_suffix(".jpg.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return dest.relative_to(ROOT).as_posix(), None
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            errors.append(type(exc).__name__)
    return None, "; ".join(errors) if errors else "No thumbnail URL available."


def collect_transcript(vid: str) -> dict:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return {"status":"error", "language":None, "segments":[], "error":"youtube-transcript-api is not installed"}
    for attempt in range(3):
        try:
            api = YouTubeTranscriptApi()
            try:
                tr = api.fetch(vid, languages=["en"])
            except Exception:
                tr = api.fetch(vid)
            segs = tr.to_raw_data() if hasattr(tr, "to_raw_data") else list(tr)
            return {"status":"available", "language":getattr(tr, "language_code", None), "segments":segs, "error":None}
        except Exception as exc:
            if attempt < 2 and any(x in type(exc).__name__.lower() for x in ("network", "request", "timeout", "connection")):
                time.sleep(2 ** attempt)
                continue
            name = type(exc).__name__.lower()
            status = "disabled" if "disabled" in name else "unavailable" if any(x in name for x in ("notranscript", "videounavailable")) else "error"
            return {"status":status, "language":None, "segments":[], "error":type(exc).__name__}


def status_row(state: dict, vid: str) -> dict:
    return state.setdefault("videos", {}).setdefault(vid, {k:{"status":"pending"} for k in MODALITIES})


def quota_day() -> str:
    """YouTube Data API quota resets at midnight Pacific Time."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


def seconds_until_quota_reset() -> int:
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    next_midnight = datetime.combine(now.date() + timedelta(days=1), datetime_time.min, tzinfo=now.tzinfo)
    return max(1, int((next_midnight - now).total_seconds()) + 60)


def active_records(records: dict, entries: list[dict]) -> list[dict]:
    """The manifest follows current CSV membership; old modality files stay on disk."""
    ids = {entry["video_id"] for entry in entries}
    return sorted((record for vid, record in records.items() if vid in ids), key=lambda r:r["video_id"])


def hydrate_existing_state(state: dict, records: dict, entries: list[dict]) -> None:
    """Import already collected work from the legacy manifest and files once."""
    for entry in entries:
        vid = entry["video_id"]
        record = records.get(vid, {})
        statuses = status_row(state, vid)
        if statuses["metadata"].get("status") == "pending" and record.get("metadata", {}).get("title"):
            statuses["metadata"] = {"status":"available", "source":"existing manifest"}
        thumb = ROOT / str(record.get("thumbnail_path") or "")
        if statuses["thumbnail"].get("status") == "pending" and thumb.is_file() and thumb.stat().st_size > 0:
            statuses["thumbnail"] = {"status":"available", "source":"existing file"}
        transcript_path = ROOT / str(record.get("transcript_path") or "")
        if statuses["transcript"].get("status") == "pending" and transcript_path.is_file():
            value = read_json(transcript_path, {})
            status = value.get("status", record.get("transcript_status"))
            if status in {"available", "disabled", "unavailable"}:
                statuses["transcript"] = {"status":status, "source":"existing file"}
        for modality, key in (("comments_top", "top"),):
            if statuses[modality].get("status") != "pending":
                continue
            path_value = record.get("comments_paths", {}).get(key)
            if not path_value:
                continue
            comment_path = ROOT / path_value
            value = read_json(comment_path, {})
            if comment_path.is_file() and value.get("status") in {"available", "empty", "disabled", "unavailable"}:
                statuses[modality] = {"status":value["status"], "count":value.get("count",0), "source":"existing file"}


def done(state: dict, vid: str, modality: str) -> bool:
    return status_row(state, vid).get(modality, {}).get("status") in {"available", "empty", "disabled", "unavailable", "missing"}


def collect(input_path: Path, limit: int, daily_limit: int, buffer: int) -> int:
    load_env_file()
    try:
        credentials = api_credentials()
    except CollectionError as exc:
        print(f"Credential configuration error: {exc}", file=sys.stderr)
        return 2
    if not credentials:
        print("Set YOUTUBE_API_KEYS (comma separated) in .env.", file=sys.stderr)
        return 2
    config = read_json(CONFIG, {})
    if not input_path.is_absolute(): input_path = ROOT / input_path
    try:
        entries, validation = load_csv(input_path, config.get("expected_categories", []))
        validate_dataset_config(entries, validation, config)
    except (OSError, CollectionError, csv.Error) as exc:
        print(f"Input error: {exc}", file=sys.stderr); return 2
    state = read_json(STATE, {"videos":{}, "quota":{}})
    client = YouTubeClient(credentials, state, daily_limit, buffer, checkpoint=lambda: atomic_json(STATE,state))
    records = {x.get("video_id"):x for x in read_json(MANIFEST, []) if x.get("video_id")}
    # Submission rows update only their human annotations; modality data is retained.
    for entry in entries:
        old = records.get(entry["video_id"], {})
        records[entry["video_id"]] = {**old, **entry}
    hydrate_existing_state(state, records, entries)
    missing_metadata = [e["video_id"] for e in entries if not done(state,e["video_id"],"metadata")]
    projected_calls = len(missing_metadata)
    metadata_paused = client.available_calls() < projected_calls
    atomic_json(MANIFEST, active_records(records, entries))
    atomic_json(STATE,state)
    def checkpoint_metadata(batch, result, batch_index):
        for vid in batch:
            if vid in result:
                records[vid]["metadata"] = result[vid]
                status_row(state,vid)["metadata"] = {"status":"available", "updated_at":datetime.now(timezone.utc).isoformat()}
            else:
                status_row(state,vid)["metadata"] = {"status":"unavailable", "error":"Video not returned by videos.list"}
    metadata, metadata_stop = metadata_batches(client, missing_metadata, checkpoint_metadata)
    metadata_paused = metadata_paused or metadata_stop == "quota"
    metadata_error = metadata_stop == "error"
    for vid in missing_metadata:
        if vid in metadata:
            records[vid]["metadata"] = metadata[vid]
            status_row(state,vid)["metadata"] = {"status":"available", "updated_at":datetime.now(timezone.utc).isoformat()}
        elif not metadata_paused:
            status_row(state,vid)["metadata"] = {"status":"error", "error":"Metadata request failed; retry on next run"} if metadata_error else {"status":"unavailable", "error":"Video not returned by videos.list"}
    atomic_json(MANIFEST, active_records(records, entries))
    atomic_json(STATE,state)
    paused = metadata_paused
    try:
        for entry in entries:
            vid = entry["video_id"]
            if interrupt_requested: break
            rec = records[vid]
            st = status_row(state,vid)
            if st.get("metadata",{}).get("status") != "available": continue
            # Thumbnail URL selection is retained in metadata for future verification/redownload.
            if not done(state,vid,"thumbnail"):
                path, err = thumbnail(vid, rec.get("metadata",{}).get("thumbnails",{}))
                rec["thumbnail_path"] = path
                thumbnail_status = "available" if path else ("missing" if err == "No thumbnail URL available." else "error")
                st["thumbnail"] = {"status":thumbnail_status, "error":err}
                if "thumbnails" in rec.get("metadata",{}): rec["metadata"].pop("thumbnails",None)
                atomic_json(MANIFEST, active_records(records, entries))
                atomic_json(STATE,state)
            if not done(state,vid,"transcript"):
                value = collect_transcript(vid)
                path = ROOT / "transcript" / f"{vid}.json"
                atomic_json(path,value)
                rec.update(transcript_path=path.relative_to(ROOT).as_posix(), transcript_status=value["status"])
                st["transcript"] = {"status":value["status"], "error":value.get("error")}
                atomic_json(MANIFEST, active_records(records, entries))
                atomic_json(STATE,state)
            label, order = "comments_top", "relevance"
            if not done(state,vid,label):
                value = comments(client,vid,order,limit)
                path = ROOT / "comments" / f"{vid}_top.json"
                atomic_json(path,value)
                rec.setdefault("comments_paths",{})["top"] = path.relative_to(ROOT).as_posix()
                rec.setdefault("comments_status",{})["top"] = value["status"]
                rec.setdefault("comments_collected",{})["top"] = value["count"]
                st[label] = {"status":value["status"], "count":value["count"], "error":value.get("error")}
                atomic_json(MANIFEST, active_records(records, entries))
                atomic_json(STATE,state)
            rec["collection_timestamp"] = datetime.now(timezone.utc).isoformat()
            atomic_json(MANIFEST, active_records(records, entries))
            atomic_json(STATE,state)
    except QuotaPaused as exc:
        paused = True
        print(str(exc))
    finally:
        atomic_json(MANIFEST, active_records(records, entries))
        atomic_json(STATE,state)
    write_report(entries, validation, state)
    print(f"Processed {len(entries)} unique valid submissions; {len(validation['issues'])} malformed/duplicate rows. {'Extraction paused; progress saved. Rerun to resume.' if paused else ''}")
    if paused:
        return 75
    return 0 if not validation["issues"] and not metadata_error else 1


def write_report(entries: list[dict], validation: dict, state: dict) -> None:
    videos = state.get("videos", {})
    counts = {m:sum(1 for e in entries if done(state,e["video_id"],m) and videos[e["video_id"]][m]["status"] == "available") for m in MODALITIES}
    incomplete = [{"video_id":e["video_id"], "modalities":[m for m in MODALITIES if not done(state,e["video_id"],m)]} for e in entries if any(not done(state,e["video_id"],m) for m in MODALITIES)]
    quota_projects = state.get("quota",{})
    today = quota_day()
    used_today = sum(value.get("used",0) for value in quota_projects.values() if value.get("date") == today)
    used_total = sum(value.get("total_used",value.get("used",0)) for value in quota_projects.values())
    report = {"generated_at":datetime.now(timezone.utc).isoformat(), "total_submitted_rows":validation["submitted_rows"],
        "total_unique_valid_videos":len(entries), "validation_issues":validation["issues"], "validation_warnings":validation.get("warnings",[]), "completed":counts,
        "failed_or_unavailable":{m:sum(1 for e in entries if videos.get(e["video_id"],{}).get(m,{}).get("status") in {"error","unavailable","disabled","missing"}) for m in MODALITIES},
        "estimated_api_quota_units_used_total":used_total, "estimated_api_quota_units_used_today":used_today,
        "quota_by_project":{name:{"date":value.get("date"),"used_today":value.get("used",0),"used_total":value.get("total_used",value.get("used",0))} for name,value in quota_projects.items()},
        "remaining_incomplete_tasks":incomplete}
    atomic_json(ROOT/"collection_report.json",report)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="data/submissions.csv")
    parser.add_argument("--max-comments",type=int,default=None)
    parser.add_argument("--daily-quota",type=int,default=None)
    parser.add_argument("--quota-buffer",type=int,default=None)
    parser.add_argument("--no-wait-for-quota",action="store_true",help="Save progress and exit instead of waiting for the next daily quota reset.")
    args=parser.parse_args()
    config=read_json(CONFIG,{})
    max_comments=args.max_comments if args.max_comments is not None else int(config.get("default_comment_limit",25))
    daily_quota=args.daily_quota if args.daily_quota is not None else int(config.get("daily_quota_limit_per_project",10000))
    quota_buffer=args.quota_buffer if args.quota_buffer is not None else int(config.get("quota_safety_buffer_per_project",100))
    if not 0 <= max_comments <= 25: parser.error("--max-comments must be 0..25")
    if daily_quota <= quota_buffer: parser.error("daily quota must exceed quota buffer")
    def stop(_sig,_frame):
        global interrupt_requested
        interrupt_requested=True
    signal.signal(signal.SIGINT,stop)
    while True:
        result = collect(Path(args.input),max_comments,daily_quota,quota_buffer)
        if result != 75 or args.no_wait_for_quota:
            return 1 if result == 75 else result
        delay = seconds_until_quota_reset()
        print(f"Quota safety stop reached. Progress is saved; resuming after the Pacific Time daily reset in about {delay // 3600} hours.")
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            print("Stopped while waiting. Run the collector again to resume.")
            return 130


if __name__ == "__main__":
    raise SystemExit(main())

