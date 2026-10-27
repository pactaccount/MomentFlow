# MomentFlow

MomentFlow is a public demo that finds visual events, spoken phrases, and text shown on screen in uploaded videos, then opens the matching timestamp. It uses the pinned SigLIP 2 checkpoint through a private Hugging Face ZeroGPU Space, plus faster-whisper, Tesseract, SQLite FTS5, and FastAPI. No model fine-tuning is part of this product.

The product code and local sandbox are in [`Prototype/`](Prototype/). The repository-root [`Dockerfile`](Dockerfile) is used by both the local sandbox and a manually configured Render Docker Web Service; see the [deployment steps](Prototype/RENDER_DEPLOYMENT.md). The UI provides a video library, processing status, filename and status filters, moment search, and timestamped playback. See the [product guide](Prototype/README.md) and [evaluation report](Prototype/EVALUATION.md). Open the UI through the FastAPI server or deployed URL; opening `static/index.html` directly cannot reach the API.

The MomentFlow source code is licensed under [Apache License 2.0](LICENSE). Downloaded model weights and user-provided videos retain their own licenses.

## Résumé evidence

[Résumé evidence and verification](docs/RESUME_EVIDENCE.md) records the use case, engineering approach, test status, measured results, limits, and safe résumé wording.

## Empirical Evaluation (IEEE Format)

See the full [IEEE Evaluation Report](docs/IEEE_EVALUATION_REPORT.md) for complete mathematical formulations, benchmark protocols, and results:
- **MSR-VTT 1k Held-Out Subset**: **90.0% Recall@1** (vs. 70.0% first-frame baseline), **95.0% Recall@5**, **0.925 MRR@10**.
- **Warm Search Latency**: **25.0 ms** dense vector search, **59.0 ms** tri-modal rank fusion.
- **Action Localization**: 12.5% Segment R@1 on Ball Sports fine-grained movement benchmark.
- **Live Deployment**: Verified operational at [momentflow.onrender.com](https://momentflow.onrender.com/).
