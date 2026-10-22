# MomentFlow product

MomentFlow accepts a video, indexes visual frames, spoken words, and on-screen text, then returns searchable timestamps and playback. It is a public demo with no sign-in. The web app and indexing worker share one Python process on Render Free; the same container can run in the local sandbox. SigLIP 2 runs in the private Hugging Face Space `pactaccount/momentflow-siglip2`. There is no fine-tuning.

## Pipeline

1. FastAPI validates and stores an uploaded video, then queues an indexing job in SQLite.
2. The worker uses OpenCV and PySceneDetect to sample frames and scene changes. Tesseract extracts visible text. Faster-whisper `base.en` transcribes speech when audio is present.
3. The worker sends frame batches to the pinned SigLIP 2 model in the private Hugging Face Space and commits visual vectors, transcript segments, and OCR text to SQLite/FTS5 together.
4. Search embeds the text query through the same Space, fuses visual, speech, and screen-text evidence, and returns ranked video moments.

The pinned visual checkpoint is `google/siglip2-base-patch16-224` revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The Space app is in [`hf_zerogpu_space/`](hf_zerogpu_space/). The Docker build is defined by the repository-root [`Dockerfile`](../Dockerfile); Render Web Service settings are documented in [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).

## Keys and settings

For **local sandbox testing**, edit the already-created, Git-ignored [`Prototype/.env`](.env). The safe template is [`.env.example`](.env.example). See the [complete setup guide](RENDER_DEPLOYMENT.md) before deploying.

| Name | What to enter | Where it is used |
| --- | --- | --- |
| `HF_SPACE_ID` | Already set to `pactaccount/momentflow-siglip2` | Identifies the private model Space. Not secret. |
| `HF_TOKEN` | Your fine-grained Hugging Face token with read access to that Space | Server-to-Space calls. Secret. |
| `ASR_MODEL` | Leave `base.en` | Same speech model in sandbox and Render. Not secret. |

**`HF_TOKEN` is the only secret.** It is already present locally and must never be placed in chat or Git. On **Render**, enter it in the Web Service's **Environment** page; Render does not read your local `.env` file. No Oracle, Modal, Roboflow, Liquid AI, Render API, or GitHub API key is required for the selected dashboard deployment. Render must have access to this GitHub repository through your connected Git provider account.

## Sandbox first

1. The Hugging Face Space is **Running** with the three files from `hf_zerogpu_space/` at its root and **ZeroGPU** selected in its Hardware settings. The corrected `app.py` has been uploaded and verified.
2. Keep `HF_TOKEN` in `Prototype/.env`. Keep the file private.
3. Install and start Docker Desktop if you want the container sandbox. From `Prototype/`, run `docker compose up --build -d`, then open `http://127.0.0.1:7860`. There is no sign-in.
4. Upload one short video you can demo. Wait for **Ready**, then test a visible event, a spoken phrase, and text shown on screen. Check `docker compose logs --tail=100 product` if indexing fails. The sandbox uses the same Dockerfile, model Space, worker, ranking code, and UI as Render. Its bind-mounted `runtime/` lasts across local restarts; Render Free's `/data` does not.
5. Run `python3 -m unittest discover -s . -p 'test_*.py' -q` and `node --check static/app.js`. Use [`evaluate.py`](evaluate.py) with a labeled timestamp manifest for Recall@K metrics; see [evaluation](EVALUATION.md).

Live Space verification passed for the pinned model with normalized 768-dimensional text and image embeddings. Render Free memory testing remains pending. Its 512 MB RAM and 0.1 CPU may be insufficient for full video indexing with `base.en`; see [deployment steps](RENDER_DEPLOYMENT.md).
