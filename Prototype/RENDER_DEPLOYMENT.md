# MomentFlow demo: exact setup and deployment guide

## Which service does what?

| Service | Runs | Where its settings go |
| --- | --- | --- |
| Hugging Face Space `pactaccount/momentflow-siglip2` | Pinned SigLIP 2 image and text embeddings on ZeroGPU | Three application files in the **root of that Space repository**; ZeroGPU in its Settings |
| Render Web Service `momentflow-demo` | Public website, video upload, indexing worker, speech transcription, OCR, SQLite/FTS search | MomentFlow GitHub repository and Render Environment variables |
| Your Mac | Optional sandbox using the same Dockerfile as Render | `Prototype/.env` for Docker Compose; this file is not uploaded |

The Space is **private**. That hides it from unauthenticated visitors, who can see a 404. The Render website can still call it using `HF_TOKEN`. The website itself has no sign-in. Anyone with the Render URL can upload, search, and delete demo videos; use only videos you are comfortable exposing in a public demo.

## Current Space status and API fix

The Space contains the three application files and is **Running**. The first live verification found that `/health` returned an empty tuple because the deployed Gradio functions had no declared return types. The corrected `app.py` now declares `-> str` for all three endpoints and matches the file deployed to the Space. A follow-up verification passed: the pinned model returned normalized text and image embeddings of 768 dimensions. Pushing MomentFlow to GitHub does not update the separate Space repository.

A **404 in an ordinary browser** is expected for this private Space unless you are signed in to the owner account. Render will call it with `HF_TOKEN`.

The local files to review are:

```text
MomentFlow/Prototype/hf_zerogpu_space/README.md
MomentFlow/Prototype/hf_zerogpu_space/app.py
MomentFlow/Prototype/hf_zerogpu_space/requirements.txt
```

The Space root now contains `README.md`, `app.py`, and `requirements.txt`, with the corrected `app.py` deployed. **Settings → Hardware** should remain **ZeroGPU**. No API key is needed inside the Space: it downloads a public, pinned SigLIP 2 model. The `HF_TOKEN` stays only in the product backend.

The product repository is at `https://github.com/pactaccount/MomentFlow` on `main`. The remaining deployment action is configuring and testing the Render Web Service.

## Exact environment values

The only secret is `HF_TOKEN`. `APP_TOKEN` was removed with the demo sign-in flow. Do not add a Hugging Face username/password, Render API key, GitHub API key, Oracle key, model checkpoint path, or an `.env` file to Git.

### On your Mac: `MomentFlow/Prototype/.env`

Open the existing hidden `.env` file in a text editor. It is already Git-ignored. It should contain these three settings for the current plan (with your real token in place of the placeholder):

```dotenv
HF_SPACE_ID=pactaccount/momentflow-siglip2
HF_TOKEN=hf_your_actual_token_here
ASR_MODEL=base.en
```

| Setting | How to obtain or choose it | Purpose |
| --- | --- | --- |
| `HF_SPACE_ID` | Keep the exact owner/name shown above | Selects the private model Space; not secret |
| `HF_TOKEN` | Hugging Face **Settings → Access Tokens**; use a token that can read this private Space | Allows the Render/backend process to call the Space; secret |
| `ASR_MODEL` | Leave `base.en` | English faster-whisper checkpoint used only when a video has audio; not a token |

The existing local `HF_TOKEN` was checked without printing it and successfully read the Space metadata. There is no need to paste it into chat. If you replace it, use a read token or a fine-grained token with read access to this Space. Keep the token on one line with no surrounding quotes or spaces. Do not commit `.env`.

### On Hugging Face

There is **no project `.env` file** to add to the Space and **no token to paste into its Variables and Secrets** for this design. Its configuration is the YAML header of its root `README.md`, the application code, its `requirements.txt`, and the selected ZeroGPU hardware.

### On Render

Render does **not** read your Mac's `.env`. For a manually created **Docker Web Service**, add these values in the service's **Environment** page:

```text
HF_SPACE_ID=pactaccount/momentflow-siglip2
PUBLIC_DEPLOYMENT=1
DATA_DIR=/data
```

Render supplies `PORT` and its service URL automatically. Its service URL is used for the same-origin check on requests that change data. Add **`HF_TOKEN`** as a secret Environment variable in the same page. Do not put the token's value in `render.yaml` or Git. `render.yaml` is optional and does not configure a Web Service created manually.

`ASR_MODEL` is optional on Render because the backend defaults to `base.en`. Set `ASR_MODEL=base.en` in Render Environment only if you want it visible there; do not choose another checkpoint for a comparison until the baseline has passed.

## Manual Render Web Service setup (no Blueprint)

The reported build failure, `failed to read dockerfile: open Dockerfile: no such file or directory`, occurs because a manual Render service defaults to the repository root while this Dockerfile is in `Prototype/`. Fix the existing service rather than creating a Blueprint:

1. Open **Render Dashboard → your Web Service → Settings → Build & Deploy**. Set **Root Directory** to `Prototype` (no leading slash), **Dockerfile Path** to `./Dockerfile`, and **Docker Build Context Directory** to `.`. Render resolves the Dockerfile and build context relative to the Root Directory. Select **Docker** as the runtime. The Dockerfile's `CMD` starts the app; no custom Docker command is needed.
2. Set the service plan to **Free** and **Health Check Path** to `/api/health`. Use GitHub repo `pactaccount/MomentFlow`, branch `main`. Render changes to root directory can prompt you to update relative build fields; confirm the values above.
3. Under **Environment**, add `HF_SPACE_ID=pactaccount/momentflow-siglip2`, `PUBLIC_DEPLOYMENT=1`, `DATA_DIR=/data`, and secret `HF_TOKEN=<your token>`. `ASR_MODEL=base.en` is optional because it is the default. Do not set `PORT`; Render provides it.
4. The Hugging Face Space is **Running** and `python3 verify_hf_space.py` has passed. No further Space command is needed unless its app changes.
5. In Render, choose **Manual Deploy → Deploy latest commit**. If the build fails again, read the first error after `load build definition`; if startup fails, inspect runtime logs. Open the Render URL, upload a short public-demo video, wait for **Ready**, and search for a visible event.

A Render **Static Site** can serve only the browser assets. This product needs its Python API, upload processing, indexing worker, SQLite database, OCR, ffmpeg, and private Hugging Face token. Use a Web Service for the complete product.

## What is verified and what is still uncertain?

Local unit tests, JavaScript syntax, Compose configuration, and the web UI served through FastAPI have passed. The responsive library, filter, search layout, upload dialog, and browser-compatible playback were checked with temporary test data. Live Space verification has passed with normalized 768-dimensional text and image embeddings from the pinned model. Container and Render runtime tests are pending. Render Free's 512 MB RAM and 0.1 CPU may not be enough for video decoding, OCR, and `base.en` transcription together. If the first real deployment exhausts memory, keep the features intact and move the worker to a larger instance or a tested external speech service; do not claim that the free deployment is production-ready before measuring it.

Render Free storage is ephemeral: videos and SQLite rows can disappear on restart or redeploy, which the user accepted for this demo. ZeroGPU has shared queues and usage quotas, so it is not an always-ready dedicated inference server.

Official references: [Hugging Face Space visibility](https://huggingface.co/docs/hub/spaces-overview), [upload files in the Hub UI](https://huggingface.co/docs/hub/repositories-getting-started), [Space README configuration](https://huggingface.co/docs/hub/spaces-config-reference), [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [Hugging Face tokens](https://huggingface.co/docs/hub/security-tokens), [Gradio API endpoints](https://www.gradio.app/docs/gradio/api), [Render monorepo settings](https://render.com/docs/monorepo-support), [Render environment variables](https://render.com/docs/configure-environment-variables), [Render Free limits](https://render.com/docs/free).
