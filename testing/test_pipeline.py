#!/usr/bin/env python3
"""Run all saved clickbait models for one YouTube video URL.

The only YouTube Data API request is videos.list. The thumbnail is then
downloaded from the URL returned by that response; it does not use API quota.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TEXT_DIR = MODELS / "text_classifier"
FUSION_WEIGHTS = MODELS / "fusion_weights.joblib"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipelineError(RuntimeError):
    """A clear, user-facing pipeline error."""


def load_env_file() -> None:
    """Load local .env values without replacing environment variables."""
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def video_id_from_url(value: str) -> str:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").lower()
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {
        "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
        "youtube-nocookie.com", "www.youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            segments = [part for part in parsed.path.split("/") if part]
            if len(segments) >= 2 and segments[0] in {"embed", "shorts", "live"}:
                video_id = segments[1]
    if not VIDEO_ID_RE.fullmatch(video_id):
        raise PipelineError("Provide a valid YouTube watch, youtu.be, shorts, embed, or live URL.")
    return video_id


def fetch_video(video_id: str, api_key: str) -> dict:
    """Make exactly one YouTube Data API request for all model inputs."""
    query = urlencode({
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": api_key,
    })
    request = Request(
        f"https://www.googleapis.com/youtube/v3/videos?{query}",
        headers={"Accept": "application/json", "User-Agent": "clickbait-pipeline/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise PipelineError(f"YouTube API request failed with HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PipelineError(f"YouTube API request failed: {type(exc).__name__}.") from exc

    items = payload.get("items", [])
    if not items:
        raise PipelineError("The video was not returned; it may be private, removed, or unavailable.")
    item = items[0]
    snippet = item.get("snippet", {})
    statistics = item.get("statistics", {})
    return {
        "title": snippet.get("title") or "",
        "description": snippet.get("description") or "",
        "duration": item.get("contentDetails", {}).get("duration") or "",
        "views": statistics.get("viewCount"),
        "likes": statistics.get("likeCount"),
        "thumbnails": snippet.get("thumbnails", {}),
    }


def download_thumbnail(thumbnails: dict) -> Path:
    for quality in ("maxres", "standard", "high", "medium", "default"):
        url = thumbnails.get(quality, {}).get("url")
        if not url:
            continue
        try:
            with urlopen(Request(url, headers={"User-Agent": "clickbait-pipeline/1.0"}), timeout=30) as response:
                image_bytes = response.read()
            if image_bytes:
                suffix = Path(urlparse(url).path).suffix or ".jpg"
                temp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                temp.write(image_bytes)
                temp.close()
                return Path(temp.name)
        except (HTTPError, URLError, OSError, TimeoutError):
            continue
    raise PipelineError("Could not download a thumbnail returned by YouTube.")


def duration_to_seconds(value: object) -> float:
    match = ISO_DURATION_RE.fullmatch(str(value).strip().upper())
    if match:
        factors = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1}
        return sum(float(match.group(name) or 0) * factor for name, factor in factors.items())
    simple = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s?", str(value).strip().lower())
    if simple:
        return float(simple.group(1))
    raise PipelineError(f"Cannot parse the video's duration: {value!r}")


def is_lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:120].startswith(b"version https://git-lfs.github.com/spec/v1")


def require_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise PipelineError(f"Missing saved model: {path.relative_to(ROOT)}")
    if is_lfs_pointer(path):
        raise PipelineError(
            f"{path.relative_to(ROOT)} is a Git LFS pointer. Run 'git lfs pull' in the repository first."
        )


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # The fine-tuned checkpoint contains the encoder weights, so construct
        # the matching DistilBERT architecture without downloading base weights.
        self.encoder = DistilBertModel(DistilBertConfig(vocab_size=vocab_size))
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.encoder.config.dim, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden[:, 0]))


class CNNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(512, 1))

    def forward(self, images):
        return self.backbone(images).squeeze(1)


def load_models():
    text_checkpoint = TEXT_DIR / "best_model.pt"
    cnn_checkpoint = MODELS / "resnet18_best.pt"
    ann_checkpoint = MODELS / "ann_model.joblib"
    for checkpoint in (text_checkpoint, cnn_checkpoint, ann_checkpoint, FUSION_WEIGHTS):
        require_checkpoint(checkpoint)

    tokenizer_dir = TEXT_DIR / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise PipelineError("Missing saved tokenizer at models/text_classifier/tokenizer.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    text_model = TextClassifier(vocab_size=tokenizer.vocab_size).to(DEVICE)
    text_model.load_state_dict(torch.load(text_checkpoint, map_location=DEVICE, weights_only=True)["model_state_dict"])
    text_model.eval()

    cnn_model = CNNClassifier().to(DEVICE)
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=DEVICE, weights_only=True)["model_state_dict"])
    cnn_model.eval()

    ann = joblib.load(ann_checkpoint)
    expected_features = ["views", "likes", "duration_seconds", "like_ratio"]
    if ann.get("features") != expected_features:
        raise PipelineError("The ANN feature schema is not the expected trained schema.")

    fusion = joblib.load(FUSION_WEIGHTS)
    if fusion.get("model_columns") != ["text_probability", "ann_probability", "cnn_probability"]:
        raise PipelineError("The saved fusion weights do not match the three model scores.")
    weights = np.asarray(fusion.get("weights"), dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise PipelineError("The saved fusion weights are invalid.")
    return tokenizer, text_model, cnn_model, ann, fusion


def predict(video: dict, thumbnail_path: Path, models) -> dict:
    tokenizer, text_model, cnn_model, ann, fusion = models
    cleaned_description = URL_RE.sub("[URL]", video["description"])
    encoded = tokenizer(
        text=video["title"],
        text_pair=cleaned_description,
        max_length=256,
        truncation="only_second",
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    with Image.open(thumbnail_path) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        text_logits = text_model(
            input_ids=encoded["input_ids"].to(DEVICE),
            attention_mask=encoded["attention_mask"].to(DEVICE),
        ).float()
        text_probability = float(text_logits.softmax(1)[0, 1].cpu())
        cnn_probability = float(cnn_model(image_tensor).float().sigmoid()[0].cpu())

    views = float(video["views"]) if str(video["views"] or "").isdigit() else 0.0
    likes = float(video["likes"]) if str(video["likes"] or "").isdigit() else 0.0
    seconds = duration_to_seconds(video["duration"])
    ann_features = np.array([[
        np.log1p(views),
        np.log1p(likes),
        np.log1p(seconds),
        likes / views if views else 0.0,
    ]])
    ann_probability = float(ann["model"].predict_proba(ann["scaler"].transform(ann_features))[:, 1][0])

    ordered_scores = np.array([text_probability, ann_probability, cnn_probability])
    weights = np.asarray(fusion["weights"], dtype=float)
    final_probability = float(np.average(ordered_scores, weights=weights))
    threshold = float(fusion.get("threshold", 0.5))
    return {
        "thumbnail_score": cnn_probability,
        "text_score": text_probability,
        "metadata_score": ann_probability,
        "fusion_score": final_probability,
        "final_prediction": int(final_probability >= threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_url", help="YouTube video URL to classify.")
    args = parser.parse_args()

    try:
        load_env_file()
        api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
        if not api_key or api_key == "replace_me":
            raise PipelineError("Set YOUTUBE_API_KEY in .env before running this script.")
        video = fetch_video(video_id_from_url(args.video_url), api_key)
        thumbnail = download_thumbnail(video.pop("thumbnails"))
        try:
            scores = predict(video, thumbnail, load_models())
        finally:
            thumbnail.unlink(missing_ok=True)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Thumbnail score: {scores['thumbnail_score']:.4f}")
    print(f"Text score (title + description): {scores['text_score']:.4f}")
    print(f"Metadata score: {scores['metadata_score']:.4f}")
    print(f"Fusion score: {scores['fusion_score']:.4f}")
    print(f"Final prediction: {'clickbait (1)' if scores['final_prediction'] else 'not clickbait (0)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
