"""
MomentFlow - Apple 2-Model Architecture Kaggle Production Deployment
=====================================================================
Target Environment: Kaggle Notebook with Nvidia Tesla T4 or P100 (16 GB VRAM)
Runtime: PyTorch 2.x + CUDA 12.x + FP16 Precision
Networking: FastAPI + Uvicorn + Cloudflare Tunnel (pycloudflared)

Models Served:
  1. apple/MobileCLIP-B (OpenCLIP Torch FP16 on CUDA:0) - 0.8 GB VRAM
  2. apple/FastVLM-0.5B (AutoModelForCausalLM FP16 on CUDA:0) - 1.8 GB VRAM
Total VRAM Footprint: ~2.6 GB (Leaving >13.4 GB VRAM for high-concurrency batching)
"""

import os
import sys
import time
import math
import tempfile
import threading
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pycloudflared import try_cloudflare

# ---------------------------------------------------------------------------
# Hardware & CUDA Check
# ---------------------------------------------------------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[MomentFlow Kaggle] Target Device: {DEVICE}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"[MomentFlow Kaggle] GPU Name: {props.name}")
    print(f"[MomentFlow Kaggle] Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    print(f"[MomentFlow Kaggle] CUDA Capability: {props.major}.{props.minor}")

# ---------------------------------------------------------------------------
# Apple 2-Model Stack Loaders
# ---------------------------------------------------------------------------
from apple_encoder import AppleMobileClipEncoder
from apple_grounder import AppleFastVLMVerifier

# ---------------------------------------------------------------------------
# Global Server App & State
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MomentFlow - Apple 2-Model Video Grounding API",
    description="Scalable Video Temporal Action Grounding powered by Apple MobileCLIP-B and Apple FastVLM-0.5B on Nvidia GPU.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mobile_clip: Optional[AppleMobileClipEncoder] = None
fast_vlm: Optional[AppleFastVLMVerifier] = None

@app.on_event("startup")
def startup_event():
    global mobile_clip, fast_vlm
    print(f"[MomentFlow] Loading Model 1 (MobileCLIP-B) & Model 2 (FastVLM-0.5B) on {DEVICE}...")
    mobile_clip = AppleMobileClipEncoder(device=DEVICE)
    fast_vlm = AppleFastVLMVerifier(device=DEVICE)
    print(f"[MomentFlow] Both Apple models successfully loaded on {DEVICE} and ready for inference!")

from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Mount static UI if available
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(STATIC_DIR):
    # Check parent directory
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "Prototype", "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def home():
    index_path = os.path.join(STATIC_DIR, "index.html") if os.path.isdir(STATIC_DIR) else ""
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MomentFlow - Apple 2-Model Serving</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #0b0f19; color: #f3f4f6; }
            .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }
            h1 { color: #60a5fa; margin-top: 0; }
            a { color: #38bdf8; text-decoration: none; font-weight: 500; }
            a:hover { text-decoration: underline; }
            .tag { display: inline-block; background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; margin-right: 6px; }
            pre { background: #030712; padding: 12px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚀 MomentFlow Apple 2-Model Serving API</h1>
            <p>Video Temporal Action Grounding powered by <strong>Apple MobileCLIP-B</strong> and <strong>Apple FastVLM-0.5B</strong> on Nvidia GPU.</p>
            <div style="margin: 15px 0;">
                <span class="tag">CUDA FP16</span>
                <span class="tag">MobileCLIP-B (147M)</span>
                <span class="tag">FastVLM-0.5B</span>
                <span class="tag">2.6 GB VRAM</span>
            </div>
            <ul>
                <li><a href="/docs" target="_blank">📖 Interactive OpenAPI / Swagger Documentation</a></li>
                <li><a href="/health" target="_blank">🩺 Health & GPU Memory Status (/health)</a></li>
            </ul>
        </div>
        <div class="card">
            <h3>Quick cURL Verification:</h3>
            <pre>curl -X POST "http://localhost:8000/extract_moments" \\
  -F "file=@sample_video.mp4" \\
  -F "query=cutting tomatoes" \\
  -F "sample_fps=1.0" \\
  -F "top_k=3"</pre>
        </div>
    </body>
    </html>
    """)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 3),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        }
    return {
        "status": "healthy",
        "service": "MomentFlow-Apple-Kaggle",
        "device": DEVICE,
        "gpu": gpu_info,
        "models": {
            "model_1": "apple/MobileCLIP-B (147M)",
            "model_2": "apple/FastVLM-0.5B (0.5B)"
        }
    }

class TextVerifyRequest(BaseModel):
    query: str
    base64_image: Optional[str] = None

@app.post("/extract_moments")
async def extract_moments(
    file: UploadFile = File(...),
    query: str = Form(...),
    sample_fps: float = Form(1.0),
    top_k: int = Form(3),
    window_padding: float = Form(4.0)
):
    """
    Two-Stage Temporal Action Grounding Pipeline:
      Stage 1: Sample video frames at `sample_fps` -> Extract vectors with MobileCLIP-B -> Cosine similarity scan.
      Stage 2: Top-K peak moment extraction -> FastVLM-0.5B verification of peak keyframes.
    """
    t_start = time.perf_counter()
    if mobile_clip is None or fast_vlm is None:
        raise HTTPException(status_code=503, detail="Models are initializing...")

    # Save uploaded video to temp file
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: Decode & sample frames
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        step_frames = max(1, int(round(fps / max(0.1, sample_fps))))
        timestamps = []
        pil_frames = []

        curr_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame % step_frames == 0:
                t_sec = curr_frame / fps
                timestamps.append(t_sec)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
            curr_frame += 1
        cap.release()

        if not pil_frames:
            raise HTTPException(status_code=400, detail="Could not decode video frames.")

        # Step 2: Model 1 Vector Search (MobileCLIP-B)
        t_stage1 = time.perf_counter()
        q_vec = mobile_clip.text(query)
        frame_vecs = mobile_clip.images(pil_frames)
        similarities = frame_vecs @ q_vec
        s1_latency = time.perf_counter() - t_stage1

        # Step 3: Peak selection (Suppression within window)
        sorted_indices = np.argsort(-similarities)
        chosen_indices = []
        min_distance_sec = window_padding * 1.5

        for idx in sorted_indices:
            cand_time = timestamps[idx]
            if all(abs(cand_time - timestamps[c]) >= min_distance_sec for c in chosen_indices):
                chosen_indices.append(idx)
            if len(chosen_indices) >= top_k:
                break

        # Step 4: Model 2 Keyframe Verification (FastVLM-0.5B)
        moments = []
        for rank, idx in enumerate(chosen_indices, start=1):
            peak_time = timestamps[idx]
            sim_score = float(similarities[idx])
            start_sec = max(0.0, peak_time - window_padding)
            end_sec = min(duration, peak_time + window_padding)

            keyframe = pil_frames[idx]
            t_v = time.perf_counter()
            verified, explanation = fast_vlm.verify_action(keyframe, query)
            v_lat = time.perf_counter() - t_v

            moments.append({
                "rank": rank,
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "peak": round(peak_time, 2),
                "clip_score": round(sim_score, 4),
                "verified": verified,
                "explanation": explanation,
                "verification_latency": round(v_lat, 4)
            })

        total_latency = time.perf_counter() - t_start
        return {
            "status": "success",
            "query": query,
            "video_duration_seconds": round(duration, 2),
            "sampled_frames_count": len(pil_frames),
            "stage1_scan_latency_seconds": round(s1_latency, 4),
            "total_latency_seconds": round(total_latency, 4),
            "moments": moments
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------------------------
# Runner with Cloudflare Public Tunnel
# ---------------------------------------------------------------------------
def run_server(port: int = 8000):
    print(f"\n[MomentFlow Kaggle] Starting FastAPI backend on 0.0.0.0:{port}...")
    # Launch Cloudflare tunnel in separate thread
    def tunnel_thread():
        time.sleep(2)
        try:
            print("[Cloudflare] Requesting public HTTPS tunnel...")
            tunnel = try_cloudflare(port=port)
            print("\n" + "="*70)
            print("🚀 MOMENTFLOW PUBLIC CLOUDFLARE TUNNEL URL:")
            print(f"👉 {tunnel.tunnel}")
            print(f"👉 Interactive Swagger UI: {tunnel.tunnel}/docs")
            print("="*70 + "\n")
        except Exception as e:
            print(f"[Cloudflare Tunnel Warning] {e}")

    t = threading.Thread(target=tunnel_thread, daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run_server()

