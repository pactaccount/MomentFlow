"""Apple MobileCLIP-B visual and text encoder for fast video timeline retrieval."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image

logger = logging.getLogger("apple_encoder")

MODEL_NAME = "MobileCLIP-B"
PRETRAINED_TAG = "datacompdr"


def normalize(vectors: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.cpu().float().numpy()
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class AppleMobileClipEncoder:
    """Fast on-device vision-language encoder backed by Apple MobileCLIP-B."""

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.info("Initializing Apple MobileCLIP-B on %s...", self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_TAG
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.embedding_dim = 512

    def text(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
        return normalize(features)[0]

    def images(self, images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensors)
        return normalize(features)

