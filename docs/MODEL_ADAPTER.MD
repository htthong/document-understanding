# Hướng Dẫn Mẫu: Thêm Model Adapter Mới

Tài liệu này quy định cấu trúc mẫu và quy trình chuẩn để tích hợp bất kỳ mô hình local mới nào vào hệ thống.

---

## 1. Quy Trình 3 Bước Thêm Mô Hình Mới

1. **Tạo file adapter mới** trong thư mục `adapters/` theo **Template Mẫu** bên dưới.
2. **Đăng ký tên mô hình** vào `ADAPTER_REGISTRY` trong `adapters/__init__.py`.
3. **Cài đặt thư viện bổ sung** (nếu mô hình mới yêu cầu thư viện đặc thù).

---

## 2. Template Mẫu Cho Adapter Mới (`adapters/<model_name>.py`)

Mọi adapter mới bắt buộc kế thừa `BaseAdapter` và triển khai `load_model` cùng các phương thức predict phù hợp với evaluation mode cần hỗ trợ.

```python
# adapters/your_model.py
from typing import List, Dict, Any
from adapters.base import BaseAdapter


class YourModelAdapter(BaseAdapter):
    """Adapter mẫu cho mô hình mới."""

    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        """Nạp trọng số và processor/tokenizer của mô hình vào bộ nhớ."""
        print(f"[{self.__class__.__name__}] Loading model from: {model_path}")
        self.device = device

        # TODO: Khởi tạo model & processor tại đây
        # self.processor = AutoProcessor.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path).to(device)
        pass

    def predict_md(self, image_paths: List[str], prompt: str) -> str:
        """Group A: Chuyển ảnh trang thành Markdown thô."""
        # TODO: Chạy suy luận, trả về chuỗi Markdown
        return "raw markdown string"

    def predict_detection(self, image_path: str) -> List[Dict[str, Any]]:
        """Group B: Phát hiện các vùng layout trên một trang ảnh.

        Trả về danh sách dict, mỗi dict chứa:
            category_id: int   — ID danh mục (xem bảng bên dưới)
            bbox: [x, y, w, h] — COCO-style bounding box (pixel)
            score: float       — độ tin cậy 0-1
        """
        return [
            {"category_id": 1, "bbox": [10, 20, 300, 150], "score": 0.95},
        ]

    def predict_recognition(self, image_path: str, layout_dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group C: Nhận diện nội dung cho mỗi vùng GT.

        Nhận vào GT layout_dets (có poly, category_type, ...).
        Trả về danh sách tương tự với trường 'pred' được thêm vào.
        """
        results = []
        for det in layout_dets:
            new_det = dict(det)
            new_det["pred"] = "recognized content"  # text, LaTeX, hoặc HTML
            results.append(new_det)
        return results
```

---

## 3. Ba Phương Thức Predict

### `predict_md(image_paths, prompt) -> str` — Group A

| Param | Kiểu | Mô tả |
|---|---|---|
| `image_paths` | `List[str]` | Danh sách đường dẫn ảnh các trang trong sliding window |
| `prompt` | `str` | Prompt đã thay `{num_images}` |

Trả về: **chuỗi Markdown thô** — nội dung document được chuyển đổi từ ảnh.

Lưu ý: File `.md` sẽ được lưu trực tiếp, không parse thành JSON.

### `predict_detection(image_path) -> List[Dict]` — Group B

| Param | Kiểu | Mô tả |
|---|---|---|
| `image_path` | `str` | Đường dẫn một ảnh trang |

Trả về: **danh sách dict**, mỗi dict:

```python
{
    "category_id": 0,          # int — ID danh mục
    "bbox": [x, y, w, h],     # COCO-style axis-aligned bounding box
    "score": 0.95              # float — confidence
}
```

Bảng `category_id`:

| ID | Category | ID | Category |
|----|----------|-----|----------|
| 0 | title | 5 | table |
| 1 | text | 6 | table_caption |
| 2 | abandon | 7 | table_footnote |
| 3 | figure | 8 | isolate_formula |
| 4 | figure_caption | 9 | formula_caption |

### `predict_recognition(image_path, layout_dets) -> List[Dict]` — Group C

| Param | Kiểu | Mô tả |
|---|---|---|
| `image_path` | `str` | Đường dẫn một ảnh trang |
| `layout_dets` | `List[Dict]` | GT annotations từ OmniDocBench JSON |

Trả về: **danh sách dict giống layout_dets** với trường `pred` thêm vào:

```python
{
    "category_type": "text_block",
    "poly": [x1, y1, ..., x4, y4],
    "text": "GT text content",      # trường gốc từ GT
    "pred": "recognized text"       # trường mới từ adapter
}
```

Nội dung `pred` tùy theo `category_type`:
- `text_block`, `title` → văn bản thuần túy
- `equation_isolated`, `equation_inline` → LaTeX
- `table` → HTML `<table>...</table>`

---

## 4. Đăng Ký Mô Hình Vào Registry (`adapters/__init__.py`)

```python
# adapters/__init__.py
from typing import Dict, Type
from adapters.base import BaseAdapter

# 1. Import class adapter mới
from adapters.your_model import YourModelAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    # 2. Đăng ký key-value (key dùng làm tham số --model_type khi chạy CLI)
    "your_model_key": YourModelAdapter,
}

def get_adapter(model_type: str) -> BaseAdapter:
    if model_type not in ADAPTER_REGISTRY:
        raise ValueError(f"Mô hình '{model_type}' chưa được hỗ trợ. Danh sách: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[model_type]()
```

---

## 5. Lệnh Chạy Thực Thi

### Inference — Tạo file dự đoán

```bash
# Group A: Markdown
poetry run python adapters/run_inference.py \
    --model_type your_model_key \
    --model_path /path/to/weights \
    --image_root /path/to/benchmark \
    --save_root /path/to/output \
    --prompt DrDocBench/tools/model_infer/prompt_v3.txt \
    --eval_mode md \
    --num_pages 5

# Group B: Detection
poetry run python adapters/run_inference.py \
    --model_type your_model_key \
    --model_path /path/to/weights \
    --image_root /path/to/benchmark \
    --save_root /path/to/output \
    --eval_mode detection

# Group C: Recognition
poetry run python adapters/run_inference.py \
    --model_type your_model_key \
    --model_path /path/to/weights \
    --image_root /path/to/benchmark \
    --save_root /path/to/output \
    --gt_root /path/to/gt_root \
    --eval_mode recognition
```

### Các tham số CLI

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `--model_type` | Yes | — | Key trong `ADAPTER_REGISTRY` |
| `--model_path` | Yes | — | Đường dẫn thư mục chứa trọng số mô hình |
| `--image_root` | Yes | — | Thư mục gốc chứa `subject/doc_id/images/` |
| `--save_root` | Yes | — | Thư mục lưu kết quả |
| `--eval_mode` | No | `md` | `md`, `detection`, hoặc `recognition` |
| `--prompt` | md mode | — | Đường dẫn file prompt `.txt` |
| `--num_pages` | No | `5` | Kích thước sliding window |
| `--gt_root` | recognition | — | Đường dẫn thư mục GT |
| `--device` | No | `cuda` | Thiết bị chạy mô hình |
| `--overwrite` | No | `False` | Ghi đè kết quả đã tồn tại |

### Evaluation

```bash
cd DrDocBench

# Group A: Markdown eval
python tools/multipage_pdf_validation.py \
    --config /path/to/output/md/md2md_eval.yaml \
    --save_name model_name_eval

# Group B: Detection eval
python tools/multipage_pdf_validation.py \
    --config configs/layout_detection.yaml \
    --pred_path /path/to/output/detection/layout_detection.json

# Group C: Recognition eval
python tools/multipage_pdf_validation.py \
    --config configs/ocr.yaml \
    --gt_root /path/to/output/recognition/ocr.json
```

### Chuẩn bị GT làm prediction (sanity check)

```bash
poetry run python adapters/prepare_gt_pred.py \
    --gt_root /home/htthong/lab/Datasets/dev \
    --output_dir /tmp/gt_pred
```

Tạo ra 3 thư mục con:
- `md/` — flat `.md` files + `page_info.json` + `md2md_eval.yaml`
- `detection/` — `layout_detection.json` + `formula_detection.json`
- `recognition/` — `ocr.json` + `formula_recognition.json` + `table_recognition.json`

---

## 6. Ví Dụ: Qwen Adapter

```python
# adapters/qwen.py
from typing import List, Dict, Any
import json, re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from adapters.base import BaseAdapter


class QwenAdapter(BaseAdapter):
    def load_model(self, model_path, device="cuda", **kwargs):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, **kwargs
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _generate(self, images, prompt, max_new_tokens=8192):
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def predict_md(self, image_paths, prompt):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self._generate(images, prompt)

    def predict_detection(self, image_path):
        image = Image.open(image_path).convert("RGB")
        prompt = ("Detect layout regions. Return JSON array with: "
                  "\"category\" (title|text|figure|table|...), "
                  "\"bbox\" [x,y,w,h], \"score\" 0-1.")
        raw = self._generate([image], prompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        items = json.loads(match.group())
        CAT = {"title":0,"text":1,"abandon":2,"figure":3,"figure_caption":4,
               "table":5,"table_caption":6,"table_footnote":7,"isolate_formula":8,"formula_caption":9}
        return [{"category_id": CAT.get(i.get("category","text"),1),
                 "bbox": i.get("bbox",[0,0,0,0]), "score": float(i.get("score",0.9))} for i in items]

    def predict_recognition(self, image_path, layout_dets):
        image = Image.open(image_path).convert("RGB")
        results = []
        for det in layout_dets:
            cat = det.get("category_type","")
            poly = det.get("poly",[])
            if len(poly) < 8:
                results.append(det); continue
            crop = image.crop((poly[0], poly[1], max(poly[0],poly[2],poly[4],poly[6]),
                               max(poly[1],poly[3],poly[5],poly[7])))
            if cat == "table":
                prompt = "Extract table as HTML. Output ONLY <table>...</table>."
            elif "equation" in cat:
                prompt = "Extract formula as LaTeX. Output ONLY the LaTeX."
            else:
                prompt = "Extract text. Output ONLY the text."
            new_det = dict(det)
            new_det["pred"] = self._generate([crop], prompt, max_new_tokens=2048).strip()
            results.append(new_det)
        return results
```
