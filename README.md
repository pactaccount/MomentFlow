# MomentFlow: Apple 2-Model Video Intelligence & Moment Grounding

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Models](https://img.shields.io/badge/Models-Apple_MobileCLIP_|_FastVLM-black.svg)](https://github.com/apple/ml-mobileclip)

**MomentFlow** is an ultra-fast, zero-cost video temporal action grounding and moment retrieval engine powered by an enterprise **Apple 2-Model vision-language architecture**:

1. **Model 1 (Temporal Candidate Retriever)**: [`apple/MobileCLIP-B`](https://huggingface.co/apple/MobileCLIP-B) (147M parameters) — FastViT hybrid vision encoder trained via Multi-Modal Reinforced Training (MMRT), scanning video trajectories at **78+ fps**.
2. **Model 2 (Keyframe Spatial-Semantic Verifier)**: [`apple/FastVLM-0.5B`](https://huggingface.co/apple/FastVLM-0.5B) (0.5B parameters) — High-resolution vision-language model equipped with FastViTHD token compression, achieving an **85× reduction in Time-To-First-Token (TTFT)**.

Given any natural language action query (e.g., *"sprinkling shredded cheese on pizza"* or *"cutting hot dogs into thirds"*), MomentFlow identifies the exact start and end timestamps ($[t_{\text{start}}, t_{\text{end}}]$) and confirms action presence with natural language rationales in **$< 0.65\text{s}$ total latency**.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Ingestion ["1. Video Ingestion (1.0 fps Uniform Frame Sampling)"]
        V["Uploaded Video"] --> Probe["Probe Video & Transcode Web H.264 Playback"]
        V --> Frames["Sample Frames at 1.0 fps"]
        Frames --> M1["Model 1: Apple MobileCLIP-B<br/>FastViT Hybrid Vision Encoder"]
        M1 --> SQLite[("SQLite: Normalized 512-d Frame Vectors")]
    end

    subgraph Grounding ["2. Temporal Scan & FastVLM Verification (< 0.65s Latency)"]
        Q["User Query"] --> M1_Txt["Model 1: Apple MobileCLIP-B Text Encoder"]
        M1_Txt --> DotProd["Cosine Similarity Dot-Product S = V · q<br/>(< 5 ms full timeline scan)"]
        SQLite --> DotProd
        DotProd --> PeakPick["Temporal Candidate Peak Selector"]
        PeakPick --> M2["Model 2: Apple FastVLM-0.5B<br/>CVPR 2025 FastViTHD 144 Tokens"]
        M2 --> Decision{"Action Verified?"}
        Decision -- "YES" --> UI["Stream Video at Action Interval (t_start, t_end) - HTTP 206"]
        Decision -- "NO" --> Reject["Calibrated Absent Query Rejection"]
    end
```

---

## Empirical Benchmark Performance

All metrics measured on real `.mp4` video files using Apple Silicon MPS / Nvidia CUDA FP16:

| Benchmark | Dataset Split | Sample Count | Primary Metric | Primary Score | Latency |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **MSR-VTT** | Official 1k Test Split | 100 Clips | **Recall@1 / MedR** | **61.0% / 1.0** | 78.1 fps / 116.5 qps |
| **Charades-STA** | Multi-Action Benchmark | 15 Videos / 45 Queries | **R@1 ($t\text{IoU} \ge 0.3$)** | **40.0%** | 0.65s / query |
| **YouCook2** | Official Validation Split | 12 Videos / 48 Steps | **R@1 ($t\text{IoU} \ge 0.3$)** | **35.4%** | 0.63s / query |
| **Real-World Uncut** | 5.5 min Continuous Tutorial | 1 Video / 18 Queries | **Absent Rejection** | **75.0%** | 0.61s / query |

---

## Local Development Quickstart

### 1. Install Dependencies
```bash
pip install -r Prototype/requirements.txt
```

### 2. Launch Background Worker & Web Server
```bash
# Terminal 1: Background Video Indexing Worker
PYTHONPATH=Prototype python3 Prototype/worker.py

# Terminal 2: FastAPI Web Server
PYTHONPATH=Prototype python3 -m uvicorn main:app --app-dir Prototype --host 127.0.0.1 --port 7860
```
Open `http://127.0.0.1:7860/` in your browser.

---

## License & Attribution

- **Code License**: This project is open-sourced under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
- **Model Attribution**:
  - `apple/MobileCLIP-B`: Released by Apple Machine Learning Research under the [Apple Sample Code License](https://github.com/apple/ml-mobileclip).
  - `apple/FastVLM-0.5B`: Released by Apple Machine Learning Research under the [Apple Research License](https://huggingface.co/apple/FastVLM-0.5B).
