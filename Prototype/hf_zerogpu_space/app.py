"""Gradio API for the same pinned SigLIP 2 image/text embeddings as MomentFlow."""
from __future__ import annotations

import base64
import io
import json

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor


MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

# ZeroGPU requires model placement on cuda at module load time.
model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def normalize(vector):
    array = np.asarray(vector, dtype=np.float32)
    return (array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)).tolist()


def health():
    return json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION})


@spaces.GPU(duration=60)
def embed_text(text: str):
    if not 1 <= len(text) <= 300:
        raise ValueError("Text must contain 1–300 characters")
    inputs = processor(text=[text], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        result = model.get_text_features(**inputs)
        features = getattr(result, "pooler_output", result)
    return json.dumps({"vectors": [normalize(features.cpu().float().numpy())[0]]})


@spaces.GPU(duration=120)
def embed_images(images_json: str):
    items = json.loads(images_json)
    if not isinstance(items, list) or not 1 <= len(items) <= 8:
        raise ValueError("Send 1–8 base64 PNG images")
    images = []
    for item in items:
        if not isinstance(item, str) or len(item) > 12_000_000:
            raise ValueError("Encoded image is too large")
        try:
            raw = base64.b64decode(item, validate=True)
            if len(raw) > 8_000_000:
                raise ValueError("Image is too large")
            image = Image.open(io.BytesIO(raw))
            if image.width * image.height > 4_000_000:
                raise ValueError("Image resolution is too large")
            images.append(image.convert("RGB"))
        except (ValueError, UnidentifiedImageError) as exc:
            raise ValueError("Invalid image") from exc
    inputs = processor(images=images, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        result = model.get_image_features(**inputs)
        features = getattr(result, "pooler_output", result)
    return json.dumps({"vectors": normalize(features.cpu().float().numpy())})


with gr.Blocks(title="MomentFlow SigLIP 2 API") as demo:
    gr.Markdown("Private SigLIP 2 embedding API for MomentFlow. Use the API endpoints from your server.")
    gr.api(health, api_name="health")
    gr.api(embed_text, api_name="embed_text")
    gr.api(embed_images, api_name="embed_images")


if __name__ == "__main__":
    demo.launch()
