# Hướng Dẫn Mẫu: Thêm Model Adapter Mới

Tài liệu này quy định cấu trúc mẫu và quy trình chuẩn để tích hợp bất kỳ mô hình local mới nào vào hệ thống.

---

## 1. Quy Trình 3 Bước Thêm Mô Hình Mới

1. **Tạo file adapter mới** trong thư mục `adapters/` theo **Template Mẫu** bên dưới.
2. **Đăng ký tên mô hình** vào `ADAPTER_REGISTRY` trong `adapters/__init__.py`.
3. **Cài đặt thư viện bổ sung** (nếu mô hình mới yêu cầu thư viện đặc thù).

---

## 2. Template Mẫu Cho Adapter Mới (`adapters/<model_name>.py`)

Mọi adapter mới bắt buộc kế thừa `BaseAdapter` và triển khai 2 hàm chính: `load_model` và `predict`.

```python
# adapters/your_model.py
from typing import List, Dict, Any
from PIL import Image
from adapters.base import BaseAdapter


class YourModelAdapter(BaseAdapter):
    """Adapter mẫu cho mô hình mới."""

    def load_model(self, model_path: str, device: str = "cuda", **kwargs) -> None:
        """1. Nạp trọng số và processor/tokenizer của mô hình vào bộ nhớ."""
        print(f"[{self.__class__.__name__}] Loading model from: {model_path}")
        self.device = device
        
        # TODO: Khởi tạo model & processor tại đây
        # self.processor = AutoProcessor.from_pretrained(model_path)
        # self.model = AutoModel.from_pretrained(model_path).to(device)
        pass

    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """2. Nhận đường dẫn ảnh, thực thi suy luận và trả về kết quả chuẩn OmniDocBench."""
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # TODO: Chạy suy luận với mô hình
        # raw_output = self.model.generate(...)

        # TODO: Parse kết quả raw_output thành danh sách layout_dets
        layout_dets = self._parse_output(raw_output, width, height)
        return layout_dets

    def _parse_output(self, raw_output: Any, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Hàm phụ trợ: Ép kiểu dữ liệu ra về đúng định dạng OmniDocBench."""
        layout_dets = []

        # Ví dụ bóc tách dữ liệu từ output của mô hình:
        # for item in raw_output:
        #     xmin, ymin, xmax, ymax = item["bbox"]  # Tọa độ pixel hoặc normalized
        #     
        #     layout_dets.append({
        #         "category_id": item.get("label", "text"),  # text, title, table, formula, figure
        #         "poly": self.bbox_to_poly([xmin, ymin, xmax, ymax]), # Chuyển thành 8 điểm
        #         "text": item.get("text", "")
        #     })

        return layout_dets

```

---

## 3. Quy Chuẩn Đầu Ra (OmniDocBench Format)

Hàm `predict()` bắt buộc trả về một `list` các `dict` có cấu trúc chính xác như sau:

```python
[
    {
        "category_id": "text",  # Tùy chọn: "text", "title", "table", "formula", "figure"
        "poly": [x1, y1, x2, y2, x3, y3, x4, y4],  # Tọa độ pixel 8 điểm (4 góc)
        "text": "Nội dung nhận diện..."  # Chuỗi văn bản, HTML (cho table), hoặc LaTeX (cho formula)
    }
]

```

> **Lưu ý về Tọa độ (`poly`):** Sử dụng hàm tiện ích `self.bbox_to_poly([xmin, ymin, xmax, ymax])` có sẵn từ `BaseAdapter` để tự động chuyển Bounding Box 4 điểm thành Polygon 8 điểm chuẩn pixel.

---

## 4. Đăng Ký Mô Hình Vào Registry (`adapters/__init__.py`)

Sau khi viết xong file adapter, mở `adapters/__init__.py` và thêm tên định danh của mô hình vào `ADAPTER_REGISTRY`:

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

Sau khi hoàn tất đăng ký, mô hình mới có thể chạy ngay lập tức qua lệnh:

```bash
poetry run python adapters/run_inference.py --model_type your_model_key --model_path /path/to/weights

```