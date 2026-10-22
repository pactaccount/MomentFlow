# MomentFlow demo: exact setup and deployment guide

## Which service does what?

| Service | Runs | Where its settings go |
| --- | --- | --- |
| Hugging Face Space `pactaccount/momentflow-siglip2` | Pinned SigLIP 2 image and text embeddings on ZeroGPU | Three application files in the **root of that Space repository**; ZeroGPU in its Settings |
| Render Web Service `momentflow-demo` | Public website, video upload, indexing worker, speech transcription, OCR, SQLite/FTS search | MomentFlow GitHub repository and Render Environment variables |
| Your Mac | Optional sandbox using the same Dockerfile as Render | `Prototype/.env` for Docker Compose; this file is not uploaded |

The Space is **private**. That hides it from unauthenticated visitors, who can see a 404. The Render website can still call it using `HF_TOKEN`. The website itself has no sign-in. Anyone with the Render URL can upload, search, and delete demo videos; use only videos you are comfortable exposing in a public demo.

## Why the Space currently fails

The Space exists and the token can read it, but its repository contains only `.gitattributes` and a starter `README.md`. Hugging Face reports `NO_APP_FILE`: there is no root-level `app.py` to run. Creating a Space or adding a token does not upload application code. These are separate problems:

1. **404 in an ordinary browser:** expected for this private Space unless you are signed in to the owner account. If you are signed in and still see a 404, verify you are using the `pactaccount` account and the exact URL `https://huggingface.co/spaces/pactaccount/momentflow-siglip2`.
2. **`NO_APP_FILE` when using the valid token:** the Space needs the three prepared files below. It cannot perform inference until its build finishes and its status says **Running**.

The local files to review are:

```text
MomentFlow/Prototype/hf_zerogpu_space/README.md
MomentFlow/Prototype/hf_zerogpu_space/app.py
MomentFlow/Prototype/hf_zerogpu_space/requirements.txt
```

After review, sign in to the `pactaccount` Hugging Face account and open the Space's **Files** tab. At the repository root, use **Add file → Upload files** for the prepared `app.py` and `requirements.txt`; replace the starter `README.md` with the prepared `README.md` using **Edit** or **Add file → Upload files**. Commit the changes in the Hugging Face UI. The root listing must show `README.md`, `app.py`, and `requirements.txt` directly; do not upload a folder named `hf_zerogpu_space`. The README YAML sets `sdk: gradio` and `app_file: app.py`. Confirm **Settings → Hardware → ZeroGPU**. Wait for **Running**, then check **Build** and **Runtime** logs if it does not start. No API key needs to be placed in the Space: it downloads a public, pinned SigLIP 2 model. The `HF_TOKEN` stays only in the product backend.

Uploading these files changes the remote Space repository; wait for the user's review before doing that or pushing the product repository.

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

Render does **not** read your Mac's `.env`. When creating the Blueprint from the reviewed MomentFlow GitHub repository, `render.yaml` supplies these public configuration values automatically:

```text
HF_SPACE_ID=pactaccount/momentflow-siglip2
PUBLIC_DEPLOYMENT=1
DATA_DIR=/data
```

Render supplies `PORT` and its service URL automatically. Its service URL is used for the same-origin check on requests that change data. The only value you enter yourself in Render is **`HF_TOKEN`**, as a secret Environment variable. In the initial Blueprint creation flow, Render prompts for it because `render.yaml` uses `sync: false`. For an existing service, open **Render Dashboard → momentflow-demo → Environment → Add Environment Variable**, set **Key** to `HF_TOKEN`, paste the token into **Value**, save, and redeploy. Do not put the token's value in `render.yaml` or Git.

`ASR_MODEL` is optional on Render because the backend defaults to `base.en`. Set `ASR_MODEL=base.en` in Render Environment only if you want it visible there; do not choose another checkpoint for a comparison until the baseline has passed.

## Exact order to bring the demo online

1. **Review the three Space files** listed above. After approval, place them at the Space repository root and wait for the Space to show **Running**. A starter page or `NO_APP_FILE` means the model API is still unavailable.
2. **Verify the model API once** from `Prototype/` with `python3 verify_hf_space.py`. The script reads the local `.env` without displaying the token. It calls one text embedding and one image embedding and checks that both are finite, normalized vectors of the same dimension. Run this only after the Space shows **Running**.
3. **Review the product repository**, then create the GitHub repository and authorize its first push. The repository root must contain `render.yaml`, with `Prototype/Dockerfile` beneath it.
4. **Create a Render Blueprint** from that GitHub repository. Confirm it creates one **Free Docker Web Service** named `momentflow-demo` with health path `/api/health`. When prompted, enter `HF_TOKEN`. Do not enter `APP_TOKEN`.
5. **Open the Render URL directly**; there is no login. Upload a short video you may publicly demo. Wait until the library row says **Ready**, then search for a visible event. If the clip has audio or readable screen text, also test a spoken phrase and visible text. Check Render logs if the job fails.
6. **Run the same short-video test in the sandbox** with Docker Desktop installed and running: from `Prototype/`, run `docker compose up --build -d`, open `http://127.0.0.1:7860`, and inspect `docker compose logs --tail=100 product` if needed. This uses the same Dockerfile, worker, model Space, and UI as Render. Docker Desktop is not currently installed on the Mac, so container verification remains pending.

## What is verified and what is still uncertain?

Local unit tests, JavaScript syntax, and Compose configuration have passed. Live SigLIP inference has **not** passed because the Space has no app file. Container and Render tests are pending. Render Free's limited memory/CPU may not be enough for video decoding, OCR, and `base.en` transcription together. If the first real deployment exhausts memory, keep the features intact and move the worker to a larger instance or a tested external speech service; do not claim that the free deployment is production-ready before measuring it.

Render Free storage is ephemeral: videos and SQLite rows can disappear on restart or redeploy, which the user accepted for this demo. ZeroGPU has shared queues and usage quotas, so it is not an always-ready dedicated inference server.

Official references: [Hugging Face Space visibility](https://huggingface.co/docs/hub/spaces-overview), [upload files in the Hub UI](https://huggingface.co/docs/hub/repositories-getting-started), [Space README configuration](https://huggingface.co/docs/hub/spaces-config-reference), [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu), [Hugging Face tokens](https://huggingface.co/docs/hub/security-tokens), [Render environment variables](https://render.com/docs/configure-environment-variables), [Render Blueprint secrets](https://render.com/docs/blueprint-spec), [Render Free limits](https://render.com/docs/free).
