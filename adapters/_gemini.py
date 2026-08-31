import json
import os
import re
import time
from typing import List, Dict, Any

from PIL import Image

from adapters.base import BaseAdapter


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_MAX_RETRIES = 3


class GeminiAdapter(BaseAdapter):
    """Adapter for Gemini models via the google-genai SDK.

    ``model_path`` is interpreted as the Gemini model ID (or leave empty to use
    ``GEMINI_MODEL``, defaulting to gemini-2.5-flash). The API key is read from the
    ``GEMINI_API_KEY`` environment variable, or passed via ``load_model(api_key=...)``.
    """

    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        self.model_name = model_path or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Set the environment variable "
                "GEMINI_API_KEY or pass api_key=... to load_model()."
            )

        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'google-genai' SDK is not installed. Install it with: pip install google-genai"
            ) from e

        self.max_output_tokens = int(
            kwargs.get("max_output_tokens")
            or os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
        )
        self.max_retries = int(
            kwargs.get("max_retries")
            or os.environ.get("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        self.client = genai.Client(api_key=self.api_key)
        print(f"[{self.__class__.__name__}] Ready (model={self.model_name}, provider=gemini, "
              f"max_output_tokens={self.max_output_tokens}, max_retries={self.max_retries})")

    # ── helpers ──────────────────────────────────────────────────────────

    def _generate(
        self,
        images: List[Image.Image],
        prompt: str,
        max_output_tokens: int = None,
        max_retries: int = None,
    ) -> str:
        from google import genai
        from google.genai import types

        budget = max_output_tokens or self.max_output_tokens
        retries = max_retries if max_retries is not None else self.max_retries

        contents = [prompt] + images

        for attempt in range(retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(max_output_tokens=budget),
                )
                return response.text or ""
            except genai.errors.APIError as e:
                if attempt == retries:
                    raise
                wait = 2 ** (attempt + 1)
                print(
                    f"[{self.__class__.__name__}] RETRY {attempt + 1}/{retries}: "
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

            pred = self._generate([crop], prompt, max_output_tokens=2048)
            new_det = dict(det)
            new_det["pred"] = pred.strip()
            results.append(new_det)

        return results