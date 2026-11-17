# MomentFlow (Apple 2-Model Architecture)

MomentFlow is an on-device video intelligence and temporal moment search engine powered by a **clean 2-model Apple vision stack**:

1. **Model 1 (Timeline Filter & Embedder): [apple_encoder.py](file:///Users/abhinav/Desktop/jobs/projects/MomentFlow/Prototype/apple_encoder.py)** — **Apple MobileCLIP-B** (CVPR 2024). Scans raw video frames at **114+ fps** on Apple Silicon MPS into 512-dimensional vectors. Query matching runs in **$< 10\text{ ms}$** via matrix dot-product ($\mathbf{S} = \mathbf{V} \cdot \mathbf{q}$).
2. **Model 2 (Deep Action Grounder & Verifier): [apple_grounder.py](file:///Users/abhinav/Desktop/jobs/projects/MomentFlow/Prototype/apple_grounder.py)** — **Apple FastVLM-0.5B** (CVPR 2025). Efficient vision-language model pairing an Apple MobileCLIP vision tower with a compact Qwen2 causal language backbone. Inspects candidate keyframes to verify action presence and reject out-of-domain / absent queries.

---

## Architecture & Data Flow

```
[Uploaded Video]
       │
       ▼ (1 fps sampling & scene cuts)
[Model 1: Apple MobileCLIP-B] ──► Encodes 114+ fps into SQLite (512-d unit vectors)
       │
[Natural Language Query]
       │
       ▼ (Text Embedding: q)
[Timeline Dot-Product Scan: S = V · q] ──► < 10 ms fast timeline filtering
       │
       ▼ (Top 2-3 Candidate Peaks)
[Model 2: Apple FastVLM-0.5B] ──► Temporal Verification & Explanation (~0.7s)
       │
       ▼
[Playable Action Interval [t_start, t_end] + Playback Stream]
```

---

## File Structure

```
Prototype/
├── apple_encoder.py        # Model 1: Apple MobileCLIP-B text & image encoder
├── apple_grounder.py       # Model 2: Apple FastVLM-0.5B action verifier
├── search_core.py          # SQLite index storage and vector operations
├── product_service.py      # Unified 2-stage coordinator (MobileCLIP + FastVLM)
├── media_tools.py          # FFmpeg video sampling and web-stream transcoding
├── main.py                 # FastAPI server (API endpoints & web UI)
├── worker.py               # Background durable indexing worker
├── render_gateway.py       # Lightweight Render proxy gateway (<65 MB RAM)
├── requirements-render.txt # Strict minimal dependencies for Render
├── kaggle_deploy.py        # Standalone Kaggle GPU production server
├── kaggle_deployment.ipynb # Turnkey Kaggle GPU Notebook (Tesla T4 FP16)
├── static/                 # Clean web search UI
├── requirements.txt        # Full local & GPU dependencies
└── compose.yaml            # Docker Compose configuration
```

---

## Quickstart (Apple Silicon & Kaggle)

### 1. Install Dependencies
```bash
pip install -r Prototype/requirements.txt
```

### 2. Start the Indexing Worker
```bash
PYTHONPATH=Prototype python3 Prototype/worker.py
```

### 3. Start the Web Application
```bash
PYTHONPATH=Prototype python3 -m uvicorn main:app --app-dir Prototype --host 127.0.0.1 --port 7860
```
Open `http://127.0.0.1:7860/` in your browser.

---

## Kaggle Deployment
To deploy on Kaggle Notebooks (free Nvidia Tesla T4 / P100 16GB GPU):
* The code runs identically on CUDA.
* Total VRAM footprint is **$< 4\text{ GB}$**, fitting into Kaggle's free GPU tier.
