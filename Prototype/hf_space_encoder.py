"""MomentFlow visual encoder backed by a private Hugging Face ZeroGPU Space."""
from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

from search_core import MODEL_ID, MODEL_REVISION, normalize


class HuggingFaceSpaceEncoder:
    def __init__(self, space_id=None, token=None, client=None):
        self.space_id = space_id or os.environ["HF_SPACE_ID"]
        self.token = token or os.environ["HF_TOKEN"]
        if not self.token:
            raise ValueError("HF_TOKEN is required for the private inference Space")
        if client is None:
            from gradio_client import Client

            client = Client(self.space_id, token=self.token, verbose=False, httpx_kwargs={"timeout": 180})
        self.client = client
        self.validated = False

    def _validate(self):
        if self.validated:
            return
        info = json.loads(self.client.predict(api_name="/health"))
        if info.get("model") != MODEL_ID or info.get("revision") != MODEL_REVISION:
            raise RuntimeError("Space model revision differs from the indexed SigLIP 2 revision")
        self.validated = True

    def text(self, query):
        self._validate()
        result = json.loads(self.client.predict(query, api_name="/embed_text"))
        vectors = np.asarray(result["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != 1:
            raise RuntimeError("Space returned an invalid text vector")
        return normalize(vectors)[0]

    def images(self, images):
        if not 1 <= len(images) <= 8:
            raise ValueError("Send 1–8 images per inference batch")
        self._validate()
        encoded = []
        for image in images:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        result = json.loads(self.client.predict(json.dumps(encoded), api_name="/embed_images"))
        vectors = np.asarray(result["vectors"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(images):
            raise RuntimeError("Space returned the wrong image count")
        return normalize(vectors)
