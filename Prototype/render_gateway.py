"""
MomentFlow - Render Web Gateway (Lightweight Reverse Proxy)
===========================================================
Target Environment: Render Free Web Service (512 MB RAM, 0.1 CPU, No GPU)
Memory Footprint: ~65 MB RAM (Zero PyTorch / CUDA dependencies)
Role: Serves static Web UI, handles user sessions, and proxies video inference 
      to the active Kaggle GPU Cloudflare Tunnel.
"""

import os
import time
import httpx
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# Dynamic or environment-configured Kaggle backend URL
KAGGLE_BACKEND_URL = os.getenv("KAGGLE_BACKEND_URL", "").rstrip("/")

app = FastAPI(
    title="MomentFlow - Render Gateway",
    description="Lightweight Web Gateway proxying video grounding inference to Kaggle GPU backend.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SetBackendPayload(BaseModel):
    backend_url: str


@app.get("/")
def home():
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return HTMLResponse("<h2>MomentFlow Gateway Live</h2>")


@app.get("/api/health")
async def health():
    backend_status = "unconfigured"
    backend_vram = None
    if KAGGLE_BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{KAGGLE_BACKEND_URL}/health")
                if res.status_code == 200:
                    backend_status = "connected"
                    backend_vram = res.json().get("vram_allocated_gb")
                else:
                    backend_status = f"unhealthy (status {res.status_code})"
        except Exception as e:
            backend_status = f"disconnected ({str(e)})"

    return {
        "status": "healthy",
        "service": "MomentFlow-Render-Gateway",
        "memory_friendly": True,
        "kaggle_backend_url": KAGGLE_BACKEND_URL or "NOT_CONFIGURED",
        "kaggle_gpu_status": backend_status,
        "kaggle_vram_gb": backend_vram
    }


@app.post("/api/set_backend")
def set_backend(payload: SetBackendPayload):
    """Allows updating the Kaggle Cloudflare Tunnel URL without redeploying Render."""
    global KAGGLE_BACKEND_URL
    url = payload.backend_url.strip().rstrip("/")
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(400, "URL must start with http:// or https://")
    KAGGLE_BACKEND_URL = url
    return {"status": "success", "kaggle_backend_url": KAGGLE_BACKEND_URL}


@app.post("/extract_moments")
@app.post("/api/extract_moments")
async def extract_moments(
    file: UploadFile = File(...),
    query: str = Form(...),
    sample_fps: float = Form(1.0),
    top_k: int = Form(3)
):
    """Proxies the video file and prompt directly to the Kaggle GPU backend."""
    global KAGGLE_BACKEND_URL
    if not KAGGLE_BACKEND_URL:
        raise HTTPException(
            status_code=503,
            detail="Kaggle GPU Backend URL is not set. Set KAGGLE_BACKEND_URL in Render environment or via /api/set_backend."
        )

    t0 = time.perf_counter()
    content = await file.read()
    files = {"file": (file.filename, content, file.content_type or "video/mp4")}
    data = {
        "query": query,
        "sample_fps": str(sample_fps),
        "top_k": str(top_k)
    }

    try:
        # Stream request to Kaggle with a 5-minute timeout for large videos
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{KAGGLE_BACKEND_URL}/extract_moments",
                files=files,
                data=data
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Kaggle GPU Error: {resp.text}"
                )
            result = resp.json()
            result["gateway_roundtrip_latency_sec"] = round(time.perf_counter() - t0, 3)
            return result
    except httpx.ConnectError:
        raise HTTPException(502, f"Failed to connect to Kaggle backend at {KAGGLE_BACKEND_URL}. Check if Kaggle tunnel is running.")
    except httpx.TimeoutException:
        raise HTTPException(504, "Kaggle GPU backend timed out processing the video.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

