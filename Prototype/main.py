"""Authenticated API for the hosted video knowledge search product."""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from media_tools import probe_video
from product_service import ProductService
from telemetry import HTTP_REQUESTS, HTTP_SECONDS, SEARCH_RESULTS, start_metrics

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(HERE / "runtime"))).resolve()
PUBLIC_DEPLOYMENT = os.getenv("PUBLIC_DEPLOYMENT") == "1"
service = ProductService(DATA_DIR)


@asynccontextmanager
async def lifespan(_app):
    start_metrics()
    yield


app = FastAPI(title="MomentFlow", docs_url=None if PUBLIC_DEPLOYMENT else "/docs", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.middleware("http")
async def record_request(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        status = 500
        raise
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(path, request.method, str(status)).inc()
        HTTP_SECONDS.labels(path, request.method).observe(time.perf_counter() - started)


class Feedback(BaseModel):
    video_id: str
    query: str
    timestamp_sec: float
    value: str


def check_origin(request: Request):
    origin = request.headers.get("origin")
    expected = os.getenv("APP_ORIGIN") or os.getenv("RENDER_EXTERNAL_URL")
    if origin and expected and origin.rstrip("/") != expected.rstrip("/"):
        raise HTTPException(403, "Unexpected origin")


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": "multimodal-v1", "indexed_frames": service.frame_count()}


@app.get("/api/videos")
def videos():
    return {"videos": service.list_videos()}


@app.post("/api/upload", status_code=202)
async def upload(request: Request, file: UploadFile = File(...)):
    check_origin(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(400, "Upload an MP4, MOV, AVI, or MKV")
    upload_dir = DATA_DIR / "incoming"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / (uuid.uuid4().hex + ".part")
    max_bytes = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
    size = 0
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, "Video exceeds upload limit")
                digest.update(chunk)
                target.write(chunk)
        if not size:
            raise HTTPException(400, "Empty video")
        try:
            metadata = probe_video(temporary, max_duration=float(os.getenv("MAX_DURATION_SEC", "1800")))
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        existing = service.find_sha(digest.hexdigest())
        if existing:
            return {"video_id": existing, "status": "already_uploaded"}
        video_id = uuid.uuid4().hex
        video_dir = DATA_DIR / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        final_path = video_dir / (video_id + suffix)
        temporary.replace(final_path)
        try:
            service.add_video(video_id, file.filename or "video", final_path, metadata, digest.hexdigest())
        except Exception:
            final_path.unlink(missing_ok=True)
            raise
        return {"video_id": video_id, "status": "queued"}
    finally:
        temporary.unlink(missing_ok=True)


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str, request: Request):
    check_origin(request)
    try:
        deleted = service.delete_video(video_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, "Video not found")
    return {"deleted": True}


@app.get("/api/search")
def search(q: str, limit: int = 8):
    if not 1 <= len(q.strip()) <= 300 or not 1 <= limit <= 20:
        raise HTTPException(400, "Invalid query or limit")
    try:
        results = service.search(q, limit)
        SEARCH_RESULTS.observe(len(results))
        return {"query": q, "results": results}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/feedback")
def feedback(payload: Feedback, request: Request):
    check_origin(request)
    if service.video_path(payload.video_id) is None:
        raise HTTPException(404, "Video not found")
    try:
        service.add_feedback(payload.video_id, payload.query, payload.timestamp_sec, payload.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved": True}


@app.get("/api/media/{video_id}")
def media(video_id: str):
    path = service.video_path(video_id)
    if path is None or not path.is_file():
        raise HTTPException(404, "Video not found")
    return FileResponse(path)


@app.get("/api/thumbnail/{video_id}")
def thumbnail(video_id: str):
    if service.video_path(video_id) is None:
        raise HTTPException(404, "Video not found")
    path = service.thumbnail_path(video_id)
    if path is None:
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")
