from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseAdapter(ABC):
    """Abstract base class for all model adapters.

    Every adapter must implement load_model(). Then implement whichever
    predict methods correspond to the evaluation modes you need:

    - predict_md       → Group A: Markdown eval (end2end, md2md, multipage_*)
    - predict_detection → Group B: Detection eval (layout_detection, formula_detection)
    - predict_recognition → Group C: Recognition eval (ocr, formula/table_recognition)

    Raise NotImplementedError from any method you do not support.
    """

    @abstractmethod
    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        """Load model weights and processor/tokenizer into memory."""
        ...

    def predict_md(self, image_paths: List[str], prompt: str) -> str:
        """Group A: Convert page images to raw Markdown text."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_md()")

    def predict_detection(self, image_path: str) -> List[Dict[str, Any]]:
        """Group B: Detect layout regions on a single page image.

        Returns a list of dicts, each with:
            category_id: int   — matches categories dict (e.g. 0=title, 1=text, ...)
            bbox: [x, y, w, h] — COCO-style axis-aligned bounding box in pixels
            score: float       — confidence 0-1
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_detection()")

    def predict_recognition(self, image_path: str, layout_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group C: Recognize content for each layout region from GT annotations.

        Takes GT layout_dets (with poly, category_type, etc.) and returns
        the same list with a 'pred' field added to each entry containing
        the recognized text/latex/html.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_recognition()")
