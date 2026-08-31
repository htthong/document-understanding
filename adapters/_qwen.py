import json
import re
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from adapters.base import BaseAdapter


class QwenAdapter(BaseAdapter):
    """Adapter for local Qwen2-VL / Qwen2.5-VL models."""

    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        print(f"[{self.__class__.__name__}] Loading model from: {model_path}")
        self.device = device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_path)

    # ── helpers ──────────────────────────────────────────────────────────

    def _generate(self, images: List[Image.Image], prompt: str, max_new_tokens: int = 8192) -> str:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)

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

            pred = self._generate([crop], prompt, max_new_tokens=2048)
            new_det = dict(det)
            new_det["pred"] = pred.strip()
            results.append(new_det)

        return results
