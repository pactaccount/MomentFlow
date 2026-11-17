"""Apple FastVLM verifier and action grounder for second-stage candidate verification."""
from __future__ import annotations

import logging
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("apple_grounder")

MODEL_ID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200


def expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    """Pad image to square matching FastVLM aspect ratio specification."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def tokenizer_image_token(prompt: str, tokenizer: AutoTokenizer, image_token_index: int = IMAGE_TOKEN_INDEX) -> torch.Tensor:
    """Tokenize prompt containing <image> token into token IDs with IMAGE_TOKEN_INDEX placeholder."""
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]
    token_ids = []
    offset = 1 if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id else 0
    if offset == 1:
        token_ids.append(chunks[0][0])
    for i, chunk in enumerate(chunks):
        if i > 0:
            token_ids.append(image_token_index)
        token_ids.extend(chunk[offset:])
    return torch.tensor(token_ids, dtype=torch.long)


class AppleFastVLMVerifier:
    """Apple FastVLM-0.5B Vision-Language verifier for second-stage grounding."""

    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logger.info("Loading Apple FastVLM (%s) on %s...", model_id, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device in ("mps", "cuda") else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        self.image_processor = vision_tower.image_processor
        logger.info("Apple FastVLM successfully initialized.")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        rgb_image = image.convert("RGB")
        mean_bg = tuple(int(x * 255) for x in self.image_processor.image_mean)
        square_image = expand2square(rgb_image, mean_bg)
        tensor = self.image_processor.preprocess(square_image, return_tensors="pt")["pixel_values"][0]
        tensor = tensor.unsqueeze(0).to(self.device)
        if self.device in ("mps", "cuda"):
            tensor = tensor.half()
        return tensor

    def verify_action(self, image: Image.Image, query: str, max_new_tokens: int = 50) -> Tuple[bool, str]:
        """Verify whether the candidate frame matches the query action."""
        image_tensor = self.preprocess_image(image)
        prompt = (
            f"<|im_start|>system\nYou are a concise video analysis assistant.<|im_end|>\n"
            f"<|im_start|>user\n<image>\n"
            f"Question: Does this scene clearly show '{query}'? Describe what is occurring in one sentence.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer_image_token(prompt, self.tokenizer).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        explanation = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # Check semantic consistency or negation
        lower_exp = explanation.lower()
        query_words = set(query.lower().split())
        matched_words = [w for w in query_words if len(w) > 3 and w in lower_exp]
        verified = len(matched_words) >= max(1, len(query_words) // 2)

        return verified, explanation

