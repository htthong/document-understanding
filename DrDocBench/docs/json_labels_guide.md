# DrDocBench - JSON Labels Guide

## Tổng quan

DrDocBench sử dụng hai loại JSON chính:
1. **GT JSON** - Ground truth annotations (đ标注)
2. **Result JSON** - Kết quả evaluation

---

## 1. JSON trong Ground Truth (GT)

**Vị trí:** `{gt_root}/{subject}/{doc_id}/json/{doc_id}_page_{N}.json`

Cấu trúc tổng:
```json
[{
  "page_info": {...},
  "layout_dets": [...],
  "extra": {...}
}]
```

### 1.1 `page_info` - Thông tin trang

| Field | Kiểu | Ý nghĩa | Ví dụ |
|-------|------|---------|-------|
| `page_name` | string | UUID định danh tài liệu | `"adf0ea9c-f744-4f64-9f74-66a9d276f371"` |
| `page_no` | int | Số trang (1-based) | `32` |
| `height` | int | Chiều cao ảnh (px) | `1014` |
| `width` | int | Chiều rộng ảnh (px) | `671` |
| `image_path` | string | Đường dẫn相对 ảnh | `"../images/.../page_32.jpg"` |
| `page_attribute` | object | Thuộc tính trang (xem 1.2) | |

### 1.2 `page_attribute` - Thuộc tính trang

| Field | Kiểu | Các giá trị | Ý nghĩa |
|-------|------|-------------|---------|
| `data_source` | string | `"book"`, `"academic_literature"` | Nguồn tài liệu |
| `subject` | string | `"ARCHITECTURE"`, `"MEDICAL"`,... | Chủ đề tài liệu |
| `challenge_type` | string | `"perception"`, `"structural_reconstruction"` | Loại thách thức |
| `language` | string | `"english"` | Ngôn ngữ |
| `layout` | string | `"single_column"`, `"double_column"`, `"other_layout"` | Bố cục trang |
| `special_issue` | array | `[]` (thường rỗng) | Tags đặc biệt |

### 1.3 `layout_dets` - Các phần tử layout trên trang

Mỗi object trong array đại diện cho 1 phần tử trên trang:

| Field | Kiểu | Ý nghĩa |
|-------|------|---------|
| `category_type` | string | **Loại phần tử** (xem 1.4) |
| `poly` | array[8 float] | Bounding box 4 góc: `[x1,y1, x2,y2, x3,y3, x4,y4]` |
| `order` | int | **Thứ tự đọc** (1-based, từ trên xuống, trái sang phải) |
| `anno_id` | int | ID duy nhất trong trang |
| `text` | string | Nội dung text (chỉ có cho text elements) |
| `ignore` | bool | Có bỏ qua khi evaluate không |
| `attribute` | object | Thuộc tính element (xem 1.5) |
| `merge_list` | array | Các block con đã merge (khi nhiều block nhỏ gộp lại) |

### 1.4 `category_type` - Loại phần tử

| Giá trị | Ý nghĩa | Có `text`? |
|---------|---------|-----------|
| `text_block` | Đoạn văn bản | ✅ |
| `title` | Tiêu đề section/chương | ✅ |
| `header` | Header trên cùng trang | ✅ |
| `page_number` | Số trang | ✅ |
| `figure` | Ảnh minh họa | ❌ |
| `figure_caption` | Chú thích ảnh | ✅ |
| `table` | Bảng | ❌ |
| `table_caption` | Chú thích bảng | ✅ |
| `table_footnote` | Chú thích cuối bảng | ✅ |
| `equation_isolated` | Công thức display (LaTeX) | ✅ |
| `equation_caption` | Chú thích công thức | ✅ |
| `figure_footnote` | Chú thích cuối ảnh | ✅ |
| `footer` | Footer dưới cùng trang | ✅ |

### 1.5 `attribute` - Thuộc tính element

| Field | Kiểu | Các giá trị |
|-------|------|-------------|
| `text_language` | string | `"text_english"`, `"text_chinese"` |
| `text_background` | string | `"white"`, `"single_colored"` |
| `text_rotate` | string | `"normal"`, `"rotated_90"`, `"rotated_180"` |

### 1.6 `extra` - Quan hệ giữa các phần tử

```json
{
  "relation": [
    {
      "source_anno_id": 3,      // anno_id của phần tử nguồn
      "target_anno_id": 4,      // anno_id của phần tử đích (-1 nếu tự tham chiếu)
      "relation_type": "parent_son"  // Loại quan hệ
    }
  ]
}
```

| `relation_type` | Ý nghĩa |
|-----------------|---------|
| `parent_son` | Quan hệ cha-con (ví dụ: figure ↔ figure_caption) |
| `truncated` | Text bị cắt ở trang này, tiếp tục ở trang sau (`target_anno_id = -1`) |

---

## 2. JSON trong Result (Kết quả evaluation)

**Vị trí:** `./result/{save_name}_*.json`

### 2.1 `{save_name}_metric_result.json` - Tổng hợp metrics

Cấu trúc:
```json
{
  "text_block": { "all": {...}, "group": {...}, "page": {...} },
  "display_formula": { "all": {...}, "group": {...}, "page": {...} },
  "table": { "all": {...}, "group": {...}, "page": {...} },
  "reading_order": { "all": {...}, "group": {...}, "page": {...} }
}
```

#### Level `all` - Điểm tổng hợp

| Sub-key | Ý nghĩa |
|---------|---------|
| `ALL_page_avg` | Trung bình edit distance theo trang |
| `edit_whole` | Edit distance trên toàn bộ text (concatenate) |
| `edit_sample_avg` | Trung bình edit distance từng sample |

Ví dụ:
```json
{
  "Edit_dist": {
    "ALL_page_avg": 0.0087,
    "edit_whole": 0.0099,
    "edit_sample_avg": 0.0101
  }
}
```

#### Level `group` - Phân tích theo thuộc tính element

```json
{
  "Edit_dist": {
    "text_background: white": 0.0088,
    "text_background: single_colored": 0.0133,
    "text_language: text_english": 0.0101
  },
  "sample_count": {
    "text_background: white": 35,
    "text_background: single_colored": 14,
    "text_language: text_english": 49
  }
}
```

#### Level `page` - Phân tích theo thuộc tính trang

```json
{
  "Edit_dist": {
    "ALL": 0.0087,
    "challenge_type: perception": 0.0069,
    "challenge_type: structural_reconstruction": 0.0105,
    "data_source: academic_literature": 0.0081,
    "data_source: book": 0.0092,
    "layout: single_column": 0.0102,
    "subject: ARCHITECTURE": 0.0087
  }
}
```

### 2.2 `{save_name}_text_block_result.json` - Chi tiết text block

Mỗi object trong array:
```json
{
  "gt_idx": [4],
  "gt": "but - an inventory of architectural 'blanks'...",
  "pred_idx": [2],
  "pred": "but - an inventory of architectural 'blanks'...",
  "edit": 0.0,
  "gt_category_type": "text_block",
  "pred_category_type": "text_block",
  "gt_position": [5],
  "pred_position": 358,
  "gt_attribute": [{
    "text_language": "text_english",
    "text_background": "single_colored",
    "text_rotate": "normal"
  }],
  "img_id": "adf0ea9c-..._page_32-32.jpg",
  "norm_gt": "butaninventoryofarchitecturalblanks",
  "norm_pred": "butaninventoryofarchitecturalblanks",
  "upper_len": 130,
  "metric": { "Edit_dist": 0.0 },
  "Edit_num": 0
}
```

| Field | Ý nghĩa |
|-------|---------|
| `gt` | Text ground truth |
| `pred` | Text prediction |
| `edit` | Edit distance (0.0 = hoàn hảo) |
| `gt_idx` / `pred_idx` | Index của element được match |
| `gt_category_type` | Loại element GT |
| `pred_category_type` | Loại element prediction |
| `gt_attribute` | Thuộc tính element GT |
| `norm_gt` / `norm_pred` | Text đã normalize (bỏ space/punctuation) |
| `upper_len` | Độ dài normalized text (số ký tự) |
| `metric` | `{ "Edit_dist": 0.0 }` |
| `Edit_num` | Số phép edit cần thiết |

### 2.3 `{save_name}_reading_order_result.json` - Chi tiết reading order

```json
{
  "gt": [5, 6, 7, 8],
  "pred": [5, 6, 7, 8],
  "img_id": "adf0ea9c-..._page_32-32.jpg",
  "edit": 0.0,
  "upper_len": 4,
  "metric": { "Edit_dist": 0.0 },
  "Edit_num": 0
}
```

| Field | Ý nghĩa |
|-------|---------|
| `gt` | Thứ tự đọc GT (danh sách order values) |
| `pred` | Thứ tự đọc prediction |
| `edit` | Edit distance (0.0 = hoàn hảo) |
| `Edit_num` | Số phép edit cần thiết |

### 2.4 `{save_name}_per_page_edit.json` - Điểm theo từng trang

```json
{
  "adf0ea9c-..._page_32-32.jpg": 0.0,
  "adf0ea9c-..._page_39-39.jpg": 0.0738,
  "adf0ea9c-..._page_45-45.jpg": 0.0186
}
```

| Key | Value | Ý nghĩa |
|-----|-------|---------|
| Tên ảnh | float | Edit distance trên trang đó (0.0 = hoàn hảo) |

---

## 3. So sánh GT JSON vs Pred Markdown

| Thuộc tính | GT JSON | Pred Markdown |
|------------|---------|---------------|
| **Định dạng** | Structured JSON | Plain markdown |
| **Thứ tự đọc** | Có `order` field | Theo thứ tự trong file |
| **Loại element** | `category_type` rõ ràng | Không có metadata |
| **Bounding box** | Có `poly` | Không có |
| **Truncated text** | Được merge qua relation | Được giữ nguyên từng block |
| **Formulas** | `equation_isolated` với LaTeX | `$...$` hoặc `$$...$$` |
| **Tables** | `table` element | `<table>` hoặc `\begin{tabular}` |

**Lưu ý:** GT JSON có cấu trúc phong phú hơn markdown, dẫn đến mismatch khi evaluate (đây là design pattern, không phải bug).

---

## 4. Ví dụ thực tế

### 4.1 Text block bị truncated

**JSON annotations:**
```json
{
  "anno_id": 5, "order": 3,
  "text": "The image here seems transparent...It becomes",
  "truncated": [12]
}
{
  "anno_id": 12, "order": 10,
  "text": "*A View of the World...strange to me."
}
```

**Khi load GT:** `get_page_elements()` merge → 1 block (480 chars)

**Markdown file:**
```markdown
The image here seems transparent...It becomes

*A View of the World...strange to me.
```

**Khi parse pred:** `md_tex_filter()` split theo `\n\n` → 2 blocks riêng biệt

**Kết quả:** Edit distance > 0 (mismatch do cách xử lý truncated)

---

## 5. Các loại metric

| Metric | Áp dụng cho | Ý nghĩa |
|--------|-------------|---------|
| `Edit_dist` | text_block, display_formula, table, reading_order | Levenshtein distance (normalized) |
| `TEDS` | table | Tree Edit Distance-based Similarity |
| `TEDS_structure_only` | table | TEDS chỉ so sánh cấu trúc |
| `CDM` | display_formula | Content Difference Metric (visual) |
| `BLEU` | text_block | BLEU score |
| `METEOR` | text_block | METEOR score |
