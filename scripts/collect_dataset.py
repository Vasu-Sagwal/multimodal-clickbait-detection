#!/usr/bin/env python3
"""Collect metadata and public modalities for manually labeled YouTube videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "videos.json"
CONFIG = ROOT / "dataset_config.json"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
VALID_LABELS = {"normal", "clickbait"}


class CollectionError(RuntimeError):
    """A recoverable error for one submitted video."""


def load_env_file() -> None:
    """Read simple KEY=VALUE pairs without overriding environment variables."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def youtube_video_id(value: str) -> str:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
                video_id = parts[1]
    if not VIDEO_ID_RE.fullmatch(video_id):
        raise CollectionError(f"Not a recognized YouTube video URL: {value}")
    return video_id


def api_get(endpoint: str, params: dict[str, str]) -> dict:
    from urllib.parse import urlencode

    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "clickbait-dataset-collector/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            error = detail.get("error", {})
            reasons = [item.get("reason") for item in error.get("errors", []) if item.get("reason")]
            message = error.get("message", str(exc))
            if reasons:
                message += f" ({', '.join(reasons)})"
        except Exception:
            message = str(exc)
        raise CollectionError(f"YouTube API {endpoint} failed: {message}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CollectionError(f"YouTube API {endpoint} request failed: {exc}") from exc


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_manifest() -> list[dict]:
    if not MANIFEST.exists() or not MANIFEST.read_text(encoding="utf-8").strip():
        return []
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise CollectionError("videos.json must contain a JSON array.")
    return data


def fetch_video(video_id: str, api_key: str) -> dict:
    payload = api_get("videos", {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": api_key,
    })
    items = payload.get("items", [])
    if not items:
        raise CollectionError("Video was not returned by the YouTube Data API (it may be private, removed, or unavailable).")
    item = items[0]
    snippet = item.get("snippet", {})
    details = item.get("contentDetails", {})
    stats = item.get("statistics", {})
    return {
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "channel_title": snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "category_id": snippet.get("categoryId"),
        "tags": snippet.get("tags", []),
        "duration": details.get("duration"),
        "view_count": int(stats["viewCount"]) if stats.get("viewCount", "").isdigit() else None,
        "like_count": int(stats["likeCount"]) if stats.get("likeCount", "").isdigit() else None,
        "comment_count": int(stats["commentCount"]) if stats.get("commentCount", "").isdigit() else None,
        "thumbnails": snippet.get("thumbnails", {}),
    }


def download_thumbnail(video_id: str, thumbnails: dict) -> tuple[str | None, str | None]:
    candidates = sorted(
        thumbnails.items(),
        key=lambda pair: {"maxres": 0, "standard": 1, "high": 2, "medium": 3, "default": 4}.get(pair[0], 5),
    )
    last_error = None
    for quality, info in candidates:
        url = info.get("url")
        if not url:
            continue
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        rel_path = Path("Thumbnail") / f"{video_id}{suffix}"
        dest = ROOT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            request = Request(url, headers={"User-Agent": "clickbait-dataset-collector/1.0"})
            with urlopen(request, timeout=30) as response, dest.open("wb") as out:
                out.write(response.read())
            return rel_path.as_posix(), None
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            last_error = str(exc)
    return None, last_error or "No thumbnail URL was supplied by the API."


def collect_transcript(video_id: str) -> dict:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return {"status": "unavailable", "language": None, "segments": [], "error": "Install requirements.txt to enable transcript collection."}
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        segments = transcript.to_raw_data() if hasattr(transcript, "to_raw_data") else list(transcript)
        language = getattr(transcript, "language_code", None)
        normalized = [
            {"start": segment.get("start"), "duration": segment.get("duration"), "text": segment.get("text", "")}
            for segment in segments
        ]
        return {"status": "available", "language": language or "en", "segments": normalized, "error": None}
    except Exception as first_error:
        # Compatibility with youtube-transcript-api releases before the instance API.
        try:
            old = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return {"status": "available", "language": "en", "segments": old, "error": None}
        except Exception as exc:
            return {"status": "unavailable", "language": None, "segments": [], "error": str(exc or first_error)}


def collect_comments(video_id: str, api_key: str, limit: int) -> dict:
    comments = []
    page_token = None
    try:
        while len(comments) < limit:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": str(min(100, limit - len(comments))),
                "textFormat": "plainText",
                "order": "relevance",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = api_get("commentThreads", params)
            for item in payload.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                comments.append({
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "like_count": snippet.get("likeCount", 0),
                })
                if len(comments) >= limit:
                    break
            page_token = payload.get("nextPageToken")
            if not page_token or not payload.get("items"):
                break
        return {"status": "available" if comments else "empty", "count": len(comments), "items": comments, "error": None}
    except CollectionError as exc:
        status = "disabled_or_unavailable" if "commentsDisabled" in str(exc) or "disabled" in str(exc).lower() else "unavailable"
        return {"status": status, "count": len(comments), "items": comments, "error": str(exc)}


def validate_row(row: dict, row_number: int, categories: set[str]) -> tuple[str, str, str]:
    url = (row.get("video_url") or "").strip()
    category = (row.get("category") or "").strip()
    label = (row.get("label") or "").strip().lower()
    reviewer = (row.get("reviewer") or "").strip()
    if not url or not category or not label or not reviewer:
        raise CollectionError(f"Row {row_number}: video_url, category, label, and reviewer are required.")
    if label not in VALID_LABELS:
        raise CollectionError(f"Row {row_number}: label must be 'normal' or 'clickbait'.")
    if categories and category not in categories:
        raise CollectionError(f"Row {row_number}: category {category!r} is not listed in dataset_config.json.")
    return youtube_video_id(url), category, label


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}


def run_audit() -> int:
    config = load_config()
    categories = config.get("expected_categories", [])
    if not categories or any(str(item).startswith("REPLACE_WITH_") for item in categories):
        print("Set the ten real category names in dataset_config.json before running the audit.")
        return 2
    if len(categories) != 10 or len(set(categories)) != 10:
        print("dataset_config.json must list exactly 10 unique categories.")
        return 2
    records = read_manifest()
    failed = False
    target = int(config.get("videos_per_category", 20))
    expected = config.get("labels_per_category", {"normal": target // 2, "clickbait": target // 2})
    for category in categories:
        rows = [record for record in records if record.get("category") == category]
        counts = {label: sum(record.get("label") == label for record in rows) for label in VALID_LABELS}
        ok = len(rows) == target and all(counts.get(label, 0) == int(expected.get(label, 0)) for label in VALID_LABELS)
        print(f"{category}: {len(rows)}/{target}; normal={counts['normal']}, clickbait={counts['clickbait']} {'OK' if ok else 'INCOMPLETE'}")
        failed |= not ok
    extra = sorted({record.get("category") for record in records} - set(categories))
    if extra:
        print("Unconfigured categories in videos.json: " + ", ".join(map(str, extra)))
        failed = True
    print(f"Total videos: {len(records)}; target: {len(categories) * target}")
    return 1 if failed else 0


def collect(input_path: Path, max_comments: int, refresh: bool) -> int:
    load_env_file()
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        print("Set YOUTUBE_API_KEY in your environment or .env file.", file=sys.stderr)
        return 2
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    config = load_config()
    categories = set(config.get("expected_categories", []))
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        print(f"Cannot read input CSV {input_path}: {exc}", file=sys.stderr)
        return 2
    records = read_manifest()
    by_id = {record.get("video_id"): record for record in records}
    errors = 0
    seen_in_file = set()
    for row_number, row in enumerate(rows, start=2):
        try:
            video_id, category, label = validate_row(row, row_number, categories)
            if video_id in seen_in_file:
                raise CollectionError(f"Row {row_number}: duplicate video URL in this CSV.")
            seen_in_file.add(video_id)
            if video_id in by_id and not refresh:
                print(f"SKIP {video_id}: already present (use --refresh to recollect).")
                continue

            metadata = fetch_video(video_id, api_key)
            thumbnail_path, thumbnail_error = download_thumbnail(video_id, metadata.pop("thumbnails"))
            transcript = collect_transcript(video_id)
            transcript_path = Path("transcript") / f"{video_id}.json"
            atomic_json(ROOT / transcript_path, transcript)
            comments = collect_comments(video_id, api_key, max_comments)
            comments_path = Path("comments") / f"{video_id}.json"
            atomic_json(ROOT / comments_path, comments)

            record = {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "category": category,
                "label": label,
                "label_reviewer": (row.get("reviewer") or "").strip(),
                "label_notes": (row.get("notes") or "").strip(),
                "metadata": metadata,
                "thumbnail_path": thumbnail_path,
                "transcript_path": transcript_path.as_posix(),
                "transcript_status": transcript["status"],
                "comments_path": comments_path.as_posix(),
                "comments_status": comments["status"],
                "comments_collected": comments["count"],
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_warnings": [message for message in (
                    f"Thumbnail unavailable: {thumbnail_error}" if thumbnail_error else None,
                    f"Transcript unavailable: {transcript['error']}" if transcript.get("error") else None,
                    f"Comments: {comments['error']}" if comments.get("error") else None,
                ) if message],
            }
            by_id[video_id] = record
            records = sorted(by_id.values(), key=lambda item: item.get("video_id", ""))
            atomic_json(MANIFEST, records)
            print(f"OK {video_id}: {category} / {label}")
            time.sleep(0.1)
        except (CollectionError, OSError, csv.Error, json.JSONDecodeError) as exc:
            errors += 1
            print(f"ERROR: {exc}", file=sys.stderr)
    print(f"Finished: {len(records)} records in videos.json; {errors} row(s) failed.")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/submissions.csv", help="CSV containing manually labeled video submissions.")
    parser.add_argument("--max-comments", type=int, default=None, help="Maximum top-level comments per video (default from dataset_config.json).")
    parser.add_argument("--refresh", action="store_true", help="Recollect videos already present in videos.json.")
    parser.add_argument("--audit", action="store_true", help="Check the 10 × 20 balanced collection target.")
    args = parser.parse_args()
    if args.audit:
        return run_audit()
    config = load_config()
    max_comments = args.max_comments if args.max_comments is not None else int(config.get("default_comment_limit", 100))
    if max_comments < 0:
        parser.error("--max-comments cannot be negative")
    return collect(Path(args.input), max_comments, args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
