# Qwen Local Model Adapter + Evaluation Pipeline Report

## 1. What Was Built

### `adapters/` folder structure

```
adapters/
├── __init__.py              # ADAPTER_REGISTRY + get_adapter()
├── base.py                  # BaseAdapter ABC
├── qwen.py                  # QwenAdapter for local Qwen2-VL
├── run_inference.py         # CLI: sliding window inference → .md output
└── prepare_gt_pred.py       # CLI: copy GT markdowns to flat pred dir
```

### Files and their roles

#### `adapters/base.py`
Abstract base class. Two methods:
- `load_model(model_path, device)` — load weights into memory
- `predict(image_paths: List[str], prompt: str) -> str` — returns raw Markdown text

#### `adapters/qwen.py`
`QwenAdapter` implementation:
- Uses `Qwen2VLForConditionalGeneration` + `AutoProcessor` from `transformers`
- `predict()` loads all images in the sliding window, builds a multimodal chat message with the prompt, runs generation, returns raw Markdown string
- Requires: `transformers`, `torch`, `Pillow`

#### `adapters/run_inference.py`
CLI entry point for local model inference:
- Reads `--prompt` file, substitutes `{num_images}` with window size
- Builds sliding windows over page images
- Calls `adapter.predict(window_paths, prompt)` per window
- Saves output as `{doc_id}_page_{start}-{end}.md` — directly compatible with `MultipageEnd2EndDataset`

```bash
poetry run python adapters/run_inference.py \
    --model_type qwen \
    --model_path /path/to/weights \
    --image_root /path/to/benchmark \
    --save_root /path/to/output \
    --prompt DrDocBench/tools/model_infer/prompt_v3.txt \
    --num_pages 5
```

#### `adapters/prepare_gt_pred.py`
Copies GT per-page markdowns from dev dataset to a flat directory:

```bash
poetry run python adapters/prepare_gt_pred.py \
    --gt_root /home/htthong/lab/Datasets/dev \
    --output_dir /tmp/gt_pred
```

Currently just copies `.md` files. **Does not yet build `page_info.json` or generate eval config.**

---

## 2. Dev Dataset Structure

Path: `/home/htthong/lab/Datasets/dev/`

```
dev/
  {subject}/                          # 33 BISAC subjects
    {uuid}/
      {uuid}.md                       # whole-document markdown (57 docs)
      images/
        page_{N}.jpg                  # page images
      json/
        {uuid}_page_{N}.json          # per-page GT annotation (836 files)
      mds/
        {uuid}_{N}.md                 # per-page markdown (836 files)
        imgs/                         # embedded figure images
```

**836 pages, 66 documents, 33 subjects.**

Each per-page JSON has:
```json
[{
  "page_info": {
    "image_path": "page_1.jpg",
    "page_no": 1,
    "width": 1200,
    "height": 1600,
    "page_attribute": {
      "data_source": "book",
      "subject": "POETRY",
      "language": "english",
      "layout": "single_column",
      "challenge_type": "perception"
    }
  },
  "layout_dets": [...],
  "extra": {"relation": [...]}
}]
```

---

## 3. DrDocBench Evaluation Pipeline

### Entry point

```bash
cd DrDocBench
python tools/multipage_pdf_validation.py \
    --config configs/<config>.yaml \
    --gt_root <path>       # override ground_truth.data_path
    --pred_path <path>     # override prediction.data_path
    --save_name <name>     # result file prefix
```

### Config structure (YAML)

```yaml
end2end_eval:                    # must match EVAL_TASK_REGISTRY key
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM_plain]
    table:
      metric: [TEDS, Edit_dist]
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: md2md_dataset  # must match DATASET_REGISTRY key
    ground_truth:
      data_path: /path/to/gt/mds
      page_info: /path/to/page_info.json   # OPTIONAL — for per-page breakdown
    prediction:
      data_path: /path/to/predictions
    match_method: quick_match
```

### How `page_info_path` works (critical)

In `multipage_pdf_validation.py` lines 103-108:
```python
if cfg[task_name]['dataset']['ground_truth'].get('page_info'):
    val_task(val_dataset, metrics_list,
             cfg[task_name]['dataset']['ground_truth']['page_info'], save_name)
else:
    val_task(val_dataset, metrics_list,
             cfg[task_name]['dataset']['ground_truth']['data_path'], save_name)
```

In `end2end_run_eval.py` lines 14-24:
```python
if os.path.isdir(page_info_path):
    md_flag = True        # skip file loading, page_info = {}
else:
    md_flag = False
    with open(page_info_path, 'r') as f:   # ← FAILS if file missing
        pages = json.load(f)
```

**Rule:**
- If `page_info` is set in config → must be a valid JSON file path
- If `page_info` is absent → `data_path` (a directory) is passed → `md_flag = True` → skips page_info loading → no per-page attribute breakdown

### `Md2MdDataset` matching flow

```
1. Iterate gt_folder/*.md
2. For each file, look for same-named .md in pred_folder
3. Parse both with md_tex_filter() → {text_all, equation_isolated, latex_table, html_table}
4. Match GT vs pred via match_gt2pred_quick (Hungarian + fuzzy)
5. Return matched samples for: text_block, display_formula, table, reading_order
```

### Output files (in `./result/`)

```
{save_name}_metric_result.json              # aggregated scores
{save_name}_{element}_result.json           # per-sample matches
{save_name}_{element}_per_page_edit.json    # per-page edit distances
{save_name}_{element}_per_table_TEDS.json   # per-table TEDS (tables)
{save_name}_{element}_per_sample_CDM.json   # per-sample CDM (formulas)
```

---

## 4. The Problem

`md2md.yaml` has:
```yaml
ground_truth:
  data_path: ./demo_data/omnidocbench_demo/mds
  page_info: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json  # ← DOES NOT EXIST
```

This causes `FileNotFoundError` when running eval.

---

## 5. The Fix

**Remove or replace the `page_info` line.** Two options:

### Option A: Skip page_info (simpler)

Remove `page_info` from config. Evaluation works, but no per-page attribute breakdown (no results split by subject, layout, data_source, etc.).

### Option B: Build page_info.json from per-page JSONs (recommended)

Combine all `{uuid}_page_{N}.json` files into one `page_info.json`:
```json
[
  {"page_info": {"image_path": "page_1.jpg", "page_attribute": {...}}},
  {"page_info": {"image_path": "page_2.jpg", "page_attribute": {...}}},
  ...
]
```

This gives full per-page breakdown like the existing `gt_sanity_dev_2p_metric_result.json` shows.

---

## 6. Proposed Complete Script

A single script (`adapters/prepare_gt_pred.py`) that does everything:

1. Copy per-page `.md` files to flat directory
2. Build `page_info.json` from per-page JSONs
3. Generate eval config with correct paths

```bash
poetry run python adapters/prepare_gt_pred.py \
    --gt_root /home/htthong/lab/Datasets/dev \
    --output_dir /tmp/gt_pred
```

Output:
```
/tmp/gt_pred/
  {uuid}_{page_no}.md       # 836 flat markdown files
  page_info.json            # combined page info
  md2md_eval.yaml           # ready-to-use config
```

Then run:
```bash
cd DrDocBench
python tools/multipage_pdf_validation.py \
    --config /tmp/gt_pred/md2md_eval.yaml \
    --save_name gt_sanity
```

---

## 7. Registered Names

| Registry | Key | Class |
|---|---|---|
| ADAPTER_REGISTRY | `qwen` | `QwenAdapter` |
| DATASET_REGISTRY | `md2md_dataset` | `Md2MdDataset` |
| EVAL_TASK_REGISTRY | `end2end_eval` | `End2EndEval` |

---

## 8. Existing GT Sanity Results

Already computed at `/home/htthong/lab/results/result/gt_sanity_dev_2p_*`:
- text_block Edit_dist: 0.058 (page avg)
- display_formula Edit_dist: 0.002
- table TEDS: 0.974
- reading_order Edit_dist: 0.015

These were computed with a 2-page sliding window, presumably with a properly built `page_info.json`.
