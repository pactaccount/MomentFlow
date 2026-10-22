"""One text and one image call to verify a private ZeroGPU embedding Space."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from hf_space_encoder import HuggingFaceSpaceEncoder


def main():
    local_env = Path(__file__).with_name(".env")
    if local_env.is_file():
        for line in local_env.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator and key in {"HF_SPACE_ID", "HF_TOKEN"}:
                os.environ.setdefault(key, value.strip())
    encoder = HuggingFaceSpaceEncoder(os.environ["HF_SPACE_ID"], os.environ["HF_TOKEN"])
    text = encoder.text("a red ball")
    image = encoder.images([Image.new("RGB", (224, 224), "red")])[0]
    assert text.shape == image.shape and text.size > 0
    assert np.isfinite(text).all() and np.isfinite(image).all()
    assert np.isclose(np.linalg.norm(text), 1, atol=1e-4)
    assert np.isclose(np.linalg.norm(image), 1, atol=1e-4)
    print(f"OK: pinned model, text/image embeddings ({text.size} dimensions)")


if __name__ == "__main__":
    main()
