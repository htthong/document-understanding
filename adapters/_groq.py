import base64
import io
import json
import os
import re
import time
from typing import List, Dict, Any

from PIL import Image

from adapters.base import BaseAdapter


DEFAULT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_BASE_URL = "https://api.groq.com"
DEFAULT_MAX_COMPLETION_TOKENS = 4096


class GroqAdapter(BaseAdapter):
    """Adapter for Groq-hosted models via the OpenAI-compatible API.

    Groq currently serves two vision-capable models for document images:
      - qwen/qwen3.6-27b  (up to 5 images per request)
      - qwen/qwen3.8-27b  (up to 3 images per request)

    ``model_path`` is interpreted as the Groq model ID (or leave empty to use
    ``GROQ_MODEL``, defaulting to qwen3.6-27b). The API key is read from the
    ``GROQ_API_KEY`` environment variable, or passed via ``load_model(api_key=...)``.
    """

    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self.model_name = model_path or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.api_key = kwargs.get("api_key") or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set the environment variable "
                "GROQ_API_KEY or pass api_key=... to load_model()."
            )

        try:
            from groq import (
                Groq,
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
                APIStatusError,
            )
        except ImportError as e:
            raise ImportError(
                "The 'groq' SDK is not installed. Install it with: pip install groq"
            ) from e

        self._APIStatusError = APIStatusError
        self._APIConnectionError = APIConnectionError
        self.max_completion_tokens = int(
            kwargs.get("max_completion_tokens")
            or os.environ.get("GROQ_MAX_COMPLETION_TOKENS", DEFAULT_MAX_COMPLETION_TOKENS)
        )
        self.client = Groq(api_key=self.api_key, base_url=kwargs.get("base_url", DEFAULT_BASE_URL))
        print(f"[{self.__class__.__name__}] Ready (model={self.model_name}, provider=groq, "
              f"max_completion_tokens={self.max_completion_tokens})")

    # ── helpers ──────────────────────────────────────────────────────────

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_completion_tokens: int = None,
        max_retries: int = 3,
    ) -> str:
        budget = max_completion_tokens or self.max_completion_tokens
        content = []
        for img in images:
            b64 = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        for attempt in range(max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_completion_tokens=budget,
                )
                return completion.choices[0].message.content or ""
            except self._APIStatusError as e:
                if getattr(e, "status_code", None) == 413:
                    if budget <= 256:
                        raise
                    budget = max(256, int(budget * 0.5))
                    print(
                        f"[{self.__class__.__name__}] 413 token limit — "
                        f"reducing max_completion_tokens to {budget} and retrying"
                    )
                    continue
                if getattr(e, "status_code", None) in (429, 500, 502, 503, 504):
                    if attempt == max_retries:
                        raise
                    wait = 2 ** (attempt + 1)
                    print(
                        f"[{self.__class__.__name__}] RETRY {attempt + 1}/{max_retries}: "
                        f"HTTP {e.status_code} — waiting {wait}s"
                    )
                    time.sleep(wait)
                    continue
                raise
            except self._APIConnectionError as e:
                if attempt == max_retries:
                    raise
                wait = 2 ** (attempt + 1)
                print(
                    f"[{self.__class__.__name__}] RETRY {attempt + 1}/{max_retries}: "
                    f"{type(e).__name__} — waiting {wait}s"
                )
                time.sleep(wait)

    # ── Group A: Markdown ────────────────────────────────────────────────

    def predict_md(self, image_paths: List[str], prompt: str) -> str:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self._generate(images, prompt)

    # ── Group B: Detection ───────────────────────────────────────────────

    def predict_detection(self, image_path: str) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        prompt = (
            "Analyze this document image. Detect all layout regions and return "
            "a JSON array where each object has: "
            "\"category\" (title|text|figure|figure_caption|table|table_caption|"
            "table_footnote|isolate_formula|formula_caption|abandon), "
            "\"bbox\" as [x, y, width, height] in pixel coords, "
            "\"score\" as confidence 0-1. Output ONLY the JSON array."
        )

        raw = self._generate([image], prompt)
        return self._parse_detection(raw, width, height)

    def _parse_detection(self, raw: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        CAT_MAP = {
            "title": 0, "text": 1, "abandon": 2, "figure": 3,
            "figure_caption": 4, "table": 5, "table_caption": 6,
            "table_footnote": 7, "isolate_formula": 8, "formula_caption": 9,
        }

        results = []
        for item in items:
            cat_name = item.get("category", "text")
            category_id = CAT_MAP.get(cat_name, 1)
            bbox = item.get("bbox", [0, 0, 0, 0])
            x, y, w, h = [float(v) for v in bbox]
            x = max(0.0, min(x, img_w))
            y = max(0.0, min(y, img_h))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            results.append({
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "score": float(item.get("score", 0.9)),
                "text": item.get("text", ""),
            })
        return results

    # ── Group C: Recognition ─────────────────────────────────────────────

    def predict_recognition(self, image_path: str, layout_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        results = []
        for det in layout_dets:
            cat = det.get("category_type", "")
            poly = det.get("poly", [])
            if len(poly) < 8:
                results.append(det)
                continue

            xmin, ymin = poly[0], poly[1]
            xmax = max(poly[0], poly[2], poly[4], poly[6])
            ymax = max(poly[1], poly[3], poly[5], poly[7])
            xmin = max(0, min(int(xmin), width))
            ymin = max(0, min(int(ymin), height))
            xmax = min(int(xmax), width)
            ymax = min(int(ymax), height)
            if xmax <= xmin or ymax <= ymin:
                results.append(det)
                continue

            crop = image.crop((xmin, ymin, xmax, ymax))

            if cat == "table":
                prompt = (
                    "Extract the table from this image. Return the table in HTML format "
                    "wrapped in <table> tags. Output ONLY the HTML."
                )
            elif cat in ("equation_isolated", "equation_inline"):
                prompt = (
                    "Extract the mathematical formula from this image. "
                    "Return it in LaTeX format. Output ONLY the LaTeX."
                )
            else:
                prompt = (
                    "Extract all text from this image. "
                    "Return the text exactly as it appears. Output ONLY the text."
                )

            pred = self._generate([crop], prompt, max_completion_tokens=2048)
            new_det = dict(det)
            new_det["pred"] = pred.strip()
            results.append(new_det)

        return results