# MomentFlow

MomentFlow is a public demo that finds visual events, spoken phrases, and text shown on screen in uploaded videos, then opens the matching timestamp. It uses the pinned SigLIP 2 checkpoint through a private Hugging Face ZeroGPU Space, plus faster-whisper, Tesseract, SQLite FTS5, and FastAPI. No model fine-tuning is part of this product.

The product code and local sandbox are in [`Prototype/`](Prototype/); [`render.yaml`](render.yaml) defines its Render Free deployment. The UI provides a video library, processing status, filename and status filters, moment search, and timestamped playback. See the [product guide](Prototype/README.md), [deployment steps](Prototype/RENDER_DEPLOYMENT.md), and [evaluation report](Prototype/EVALUATION.md). Open the UI through the FastAPI server or deployed URL; opening `static/index.html` directly cannot reach the API.

The MomentFlow source code is licensed under [Apache License 2.0](LICENSE). Downloaded model weights and user-provided videos retain their own licenses.
