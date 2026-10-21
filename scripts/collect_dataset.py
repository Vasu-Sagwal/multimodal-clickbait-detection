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
LABELS = {"normal": 0, "clickbait": 1, "0": 0, "1": 1}


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
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com"}:
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
    secret = params.get("key")

    def redact(message: str) -> str:
        return message.replace(secret, "[REDACTED]") if secret else message

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
        raise CollectionError(f"YouTube API {endpoint} failed: {redact(message)}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CollectionError(f"YouTube API {endpoint} request failed: {redact(str(exc))}") from exc


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


def collect_comments(video_id: str, api_key: str, limit: int, order: str) -> dict:
    limit = min(max(0, limit), 20)
    comments = []
    page_token = None
    try:
        while len(comments) < limit:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": str(min(100, limit - len(comments))),
                "textFormat": "plainText",
                "order": order,
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


def normalize_label(value: object) -> int:
    label = "" if value is None else str(value).strip().lower()
    if label not in LABELS:
        raise CollectionError("label must be 0 (normal) or 1 (clickbait).")
    return LABELS[label]


def normalize_manifest_label(value: object) -> int:
    """Read current integer labels and legacy text labels consistently for audits/resume."""
    if isinstance(value, bool):
        raise CollectionError(f"Invalid label in videos.json: {value!r}")
    try:
        return normalize_label(value)
    except CollectionError as exc:
        raise CollectionError(f"Invalid label in videos.json: {value!r}") from exc


def validate_row(row: dict, row_number: int, categories: list[str]) -> tuple[str, str, int]:
    url = (row.get("video_url") or "").strip()
    category = (row.get("category") or "").strip()
    label_value = row.get("label")
    reviewer = (row.get("reviewer") or "").strip()
    if not url or not category or label_value is None or not str(label_value).strip() or not reviewer:
        raise CollectionError(f"Row {row_number}: video_url, category, label, and reviewer are required.")
    try:
        label = normalize_label(label_value)
    except CollectionError as exc:
        raise CollectionError(f"Row {row_number}: {exc}") from exc
    canonical = {item.strip().casefold(): item for item in categories}
    if categories and category.casefold() not in canonical:
        raise CollectionError(f"Row {row_number}: category {category!r} is not listed in dataset_config.json.")
    return youtube_video_id(url), canonical.get(category.casefold(), category), label


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}


def run_audit() -> int:
    config = load_config()
    categories = config.get("expected_categories", [])
    if not categories or any(str(item).startswith("REPLACE_WITH_") for item in categories):
        print("Set the ten real category names in dataset_config.json before running the audit.")
        return 2
    if len(categories) != 10 or len({str(item).strip().casefold() for item in categories}) != 10:
        print("dataset_config.json must list exactly 10 unique categories.")
        return 2
    records = read_manifest()
    failed = False
    target = int(config.get("videos_per_category", 100))
    expected = config.get("labels_per_category", {"0": target // 2, "1": target // 2})
    for category in categories:
        rows = [record for record in records if str(record.get("category", "")).strip().casefold() == category.strip().casefold()]
        counts = {0: sum(normalize_manifest_label(record.get("label")) == 0 for record in rows), 1: sum(normalize_manifest_label(record.get("label")) == 1 for record in rows)}
        normal_expected = int(expected.get("0", expected.get("normal", target // 2)))
        clickbait_expected = int(expected.get("1", expected.get("clickbait", target // 2)))
        ok = len(rows) == target and counts[0] == normal_expected and counts[1] == clickbait_expected
        print(f"{category}: {len(rows)}/{target}; normal=0: {counts[0]}, clickbait=1: {counts[1]} {'OK' if ok else 'INCOMPLETE'}")
        failed |= not ok
    extra = sorted({record.get("category") for record in records if str(record.get("category", "")).strip().casefold() not in {c.strip().casefold() for c in categories}})
    if extra:
        print("Unconfigured categories in videos.json: " + ", ".join(map(str, extra)))
        failed = True
    print(f"Total videos: {len(records)}; target: {len(categories) * target}")
    return 1 if failed else 0


def collect(input_path: Path, max_comments: int, refresh: bool) -> int:
    load_env_file()
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    config = load_config()
    categories = config.get("expected_categories", [])
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                rows.append((row, reader.line_num))
    except OSError as exc:
        print(f"Cannot read input CSV {input_path}: {exc}", file=sys.stderr)
        return 2
    records = read_manifest()
    by_id = {record.get("video_id"): record for record in records}
    errors = 0
    validated = []
    by_csv_id: dict[str, list[dict]] = {}
    # Full pre-validation pass: no video, thumbnail, transcript, or comment API work occurs here.
    for serial, (row, physical) in enumerate(rows, start=1):
        url = (row.get("video_url") or "").strip()
        if url:
            try:
                candidate_id = youtube_video_id(url)
                by_csv_id.setdefault(candidate_id, []).append({"video_id": candidate_id, "row": row, "serial": serial, "physical": physical})
            except CollectionError:
                pass
        try:
            video_id, category, label = validate_row(row, physical, categories)
            entry = {"video_id": video_id, "category": category, "label": label, "row": row, "serial": serial, "physical": physical}
            validated.append(entry)
        except (CollectionError, OSError, csv.Error, json.JSONDecodeError) as exc:
            errors += 1
            print(f"ERROR: {exc}", file=sys.stderr)

    duplicate_ids = {video_id for video_id, entries in by_csv_id.items() if len(entries) > 1}
    for video_id, entries in by_csv_id.items():
        exists = video_id in by_id
        if video_id in duplicate_ids or exists:
            serials = ", ".join(str(e["serial"]) for e in entries)
            lines = ", ".join(str(e["physical"]) for e in entries)
            kind = "duplicate within CSV" if video_id in duplicate_ids else "already exists in videos.json"
            if video_id in duplicate_ids and exists:
                kind += " and already exists in videos.json"
            print(f"DUPLICATE: Video ID {video_id} - CSV serials {serials} (physical lines {lines}); {kind}. Skipping all occurrences.")
            for entry in entries:
                row = entry["row"]
                print(f"  serial {entry['serial']} (line {entry['physical']}): reviewer={row.get('reviewer','')!r}, label={row.get('label','')!r}")

    if not api_key or api_key == "replace_me":
        print("Set YOUTUBE_API_KEY in your environment or .env file.", file=sys.stderr)
        return 2

    for entry in validated:
        video_id, category, label, row = entry["video_id"], entry["category"], entry["label"], entry["row"]
        if video_id in duplicate_ids:
            continue
        existing = by_id.get(video_id)
        if existing:
            try:
                existing_label = normalize_manifest_label(existing.get("label"))
            except CollectionError as exc:
                errors += 1
                print(f"ERROR: Video ID {video_id} has an invalid existing label: {exc}", file=sys.stderr)
                continue
            if existing_label != label:
                errors += 1
                print(f"ERROR: Video ID {video_id} has conflicting existing label; refusing to overwrite it.", file=sys.stderr)
                continue
            if not refresh:
                print(f"SKIP {video_id}: already present (use --refresh to recollect).")
                continue

        try:
            metadata = fetch_video(video_id, api_key)
            thumbnail_path, thumbnail_error = download_thumbnail(video_id, metadata.pop("thumbnails"))
            transcript = collect_transcript(video_id)
            transcript_path = Path("transcript") / f"{video_id}.json"
            atomic_json(ROOT / transcript_path, transcript)
            comments = {order: collect_comments(video_id, api_key, max_comments, order) for order in ("relevance", "time")}
            comments_paths = {"relevance": Path("comments") / f"{video_id}_top.json", "time": Path("comments") / f"{video_id}_newest.json"}
            for order, path in comments_paths.items():
                atomic_json(ROOT / path, comments[order])

            record = {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "category": category,
                "label": label,
                "label_reviewer": (row.get("reviewer") or existing.get("label_reviewer", "") if existing else (row.get("reviewer") or "")).strip(),
                "label_notes": (row.get("notes") or existing.get("label_notes", "") if existing else (row.get("notes") or "")).strip(),
                "metadata": metadata,
                "thumbnail_path": thumbnail_path,
                "transcript_path": transcript_path.as_posix(),
                "transcript_status": transcript["status"],
                "comments_paths": {"top": comments_paths["relevance"].as_posix(), "newest": comments_paths["time"].as_posix()},
                "comments_status": {"top": comments["relevance"]["status"], "newest": comments["time"]["status"]},
                "comments_collected": {"top": comments["relevance"]["count"], "newest": comments["time"]["count"]},
                "collection_timestamp": datetime.now(timezone.utc).isoformat(),
                "collection_warnings": [message for message in (
                    f"Thumbnail unavailable: {thumbnail_error}" if thumbnail_error else None,
                    f"Transcript unavailable: {transcript['error']}" if transcript.get("error") else None,
                    *(f"Comments ({name}): {item['error']}" for name, item in (("top", comments["relevance"]), ("newest", comments["time"])) if item.get("error")),
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
    parser.add_argument("--audit", action="store_true", help="Check the 10 × 100 balanced collection target.")
    args = parser.parse_args()
    if args.audit:
        return run_audit()
    config = load_config()
    max_comments = args.max_comments if args.max_comments is not None else int(config.get("default_comment_limit", 20))
    if not 0 <= max_comments <= 20:
        parser.error("--max-comments must be between 0 and 20 per collection")
    return collect(Path(args.input), max_comments, args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())

