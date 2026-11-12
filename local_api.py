"""Run the clickbait inference pipeline on this computer for the hosted UI."""
from __future__ import annotations

import hmac
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from testing.test_pipeline import (
    PipelineError,
    download_thumbnail,
    fetch_video,
    load_env_file,
    load_models,
    predict,
    video_id_from_url,
)

ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT / ".local-api-token"
load_env_file()
CUSTOM_WEB_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("WEB_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


def _load_or_create_token() -> str:
    if TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    else:
        token = secrets.token_urlsafe(32)
        try:
            with TOKEN_FILE.open("x", encoding="utf-8") as token_file:
                token_file.write(token)
        except FileExistsError:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()

    print("Local API token (paste into the website; kept only in this file and console):", file=sys.stderr)
    print(token, file=sys.stderr)
    return token


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env_file()
    app.state.api_token = _load_or_create_token()
    app.state.models = None
    app.state.model_error = None
    try:
        app.state.models = await run_in_threadpool(load_models)
    except Exception as exc:
        app.state.model_error = str(exc)
        print(f"Model loading failed: {exc}", file=sys.stderr)
    yield


app = FastAPI(title="Local Clickbait Inference", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *CUSTOM_WEB_ORIGINS],
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_private_network=True,
)


@app.middleware("http")
async def allow_local_network_preflight(request, call_next):
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network", "").lower() == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class AnalyzeRequest(BaseModel):
    video_url: str = Field(min_length=8, max_length=2048)


def _require_token(supplied: str | None, expected: str) -> None:
    if not supplied or not supplied.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Enter the local API token shown in the FastAPI console.")
    if not hmac.compare_digest(supplied[7:].strip(), expected):
        raise HTTPException(status_code=401, detail="The local API token is incorrect.")


@app.get("/health")
async def health():
    return {
        "service": "local-clickbait-inference",
        "models_ready": app.state.models is not None,
    }


@app.post("/analyze")
async def analyze(payload: AnalyzeRequest, authorization: str | None = Header(default=None)):
    _require_token(authorization, app.state.api_token)
    if app.state.models is None:
        raise HTTPException(status_code=503, detail="Local models are not ready. Check the FastAPI console.")

    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        raise HTTPException(status_code=503, detail="Set YOUTUBE_API_KEY in the project's local .env file.")

    def run_inference():
        thumbnail_path: Path | None = None
        try:
            video_id = video_id_from_url(payload.video_url)
            video = fetch_video(video_id, api_key)
            thumbnail_path = download_thumbnail(video.pop("thumbnails"))
            scores = predict(video, thumbnail_path, app.state.models)
            return {
                "video_id": video_id,
                "title": video["title"],
                "duration": video["duration"],
                "views": video["views"],
                "likes": video["likes"],
                **scores,
            }
        except PipelineError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if thumbnail_path is not None:
                thumbnail_path.unlink(missing_ok=True)

    return await run_in_threadpool(run_inference)
