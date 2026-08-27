import argparse
import json
import os
import re
import shutil


def poly_to_bbox(poly):
    """Convert 8-point polygon [x1,y1,x2,y2,x3,y3,x4,y4] to [L,U,R,D] (xyxy).
    
    Uses the same logic as DrDocBench's poly2bbox: L=poly[0], U=poly[1], R=poly[2], D=poly[5].
    """
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    return [L, U, R, D]


_FORMULA_PATTERN = re.compile(
    r'(\$\$.*?\$\$|\\\[.*?\\\]|\$[^$]+\$|\\\(.*?\\\))',
    re.DOTALL,
)
_DOLLAR_PATTERN = re.compile(
    r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)',
    re.DOTALL,
)


def _normalize_formula_in_md(match):
    """Pre-normalize a formula found in MD text so that DrDocBench's 
    normalized_formula() applied to it is idempotent (returns same string)."""
    from utils.data_preprocess import normalized_formula
    raw = match.group(0)
    return normalized_formula(raw)


def copy_md_files(gt_root, output_dir):
    count = 0
    for root, dirs, files in os.walk(gt_root):
        if os.path.basename(root) != "mds":
            continue
        for f in files:
            if f.endswith(".md"):
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, f)
                shutil.copy2(src, dst)
                count += 1
    return count


def normalize_formulas_in_md_files(output_dir):
    """Pre-normalize formula content in all MD files so that DrDocBench's
    GT normalization (raw content) matches pred normalization (normalized_formula(content)).
    
    DrDocBench bug: match.py line 91 uses raw content for GT formulas,
    but line 79 applies normalized_formula() for pred formulas.
    Since normalized_formula() is idempotent, pre-normalizing makes both paths produce the same output.
    
    We normalize the INNER content only, preserving delimiters ($$..$$, \[..\], $..$, \(..\))
    so md_tex_filter can still extract them.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DrDocBench'))
    from utils.data_preprocess import normalized_formula
    
    count = 0
    for fname in os.listdir(output_dir):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(output_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = _normalize_md_formulas(content, normalized_formula)
        
        if new_content != content:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            count += 1
    return count


def _normalize_md_formulas(content, normalized_formula_fn):
    """Replace inner formula content with normalized version, preserving delimiters."""
    result = []
    pos = 0
    while pos < len(content):
        # Try to match a formula delimiter at current position
        matched = False
        for opener, closer, inner_group in [
            ('$$', '$$', 2),   # $$...$$ → opener=$$, inner starts at pos+2
            ('\\[', '\\]', 2), # \[...\]
            ('\\(', '\\)', 2), # \(...\)
            ('$', '$', 1),     # $...$ → need to check not $$
        ]:
            olen = len(opener)
            if content[pos:pos + olen] == opener:
                # For $, check it's not $$
                if opener == '$' and pos + 1 < len(content) and content[pos + 1] == '$':
                    continue
                # Find closer
                closer_pos = content.find(closer, pos + olen)
                if closer_pos == -1:
                    continue
                inner = content[pos + olen:closer_pos]
                normalized_inner = normalized_formula_fn(inner)
                result.append(content[pos:pos + olen])  # opener
                result.append(normalized_inner)
                result.append(content[closer_pos:closer_pos + len(closer)])  # closer
                pos = closer_pos + len(closer)
                matched = True
                break
        if not matched:
            result.append(content[pos])
            pos += 1
    return ''.join(result)


def build_page_info(gt_root, output_dir):
    page_info = []
    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            json_dir = os.path.join(subject_path, doc_id, "json")
            if not os.path.isdir(json_dir):
                continue
            for jfile in sorted(os.listdir(json_dir)):
                if not jfile.endswith(".json"):
                    continue
                with open(os.path.join(json_dir, jfile), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and "page_info" in data[0]:
                    page_info.append(data[0])

    out_path = os.path.join(output_dir, "page_info.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(page_info, f, ensure_ascii=False, indent=2)
    return len(page_info)


def build_detection_json(gt_root, output_dir):
    CATEGORIES = {
        "title": 0, "text_block": 1, "abandon": 2, "figure": 3,
        "figure_caption": 4, "table": 5, "table_caption": 6,
        "table_footnote": 7, "equation_isolated": 8, "formula_caption": 9,
    }
    CAT_MAP = {
        "title": 0, "text_block": 1, "abandon": 2, "figure": 3,
        "figure_caption": 4, "table": 5, "table_caption": 6,
        "table_footnote": 7, "equation_isolated": 8, "formula_caption": 9,
        "equation_inline": 8, "reference": 1, "code_algorithm": 1,
        "code_algorithm_caption": 1, "header": 2, "footer": 2,
        "page_number": 2, "page_footnote": 2,
    }

    all_results = []
    pages_processed = 0

    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            json_dir = os.path.join(subject_path, doc_id, "json")
            if not os.path.isdir(json_dir):
                continue
            for jfile in sorted(os.listdir(json_dir)):
                if not jfile.endswith(".json"):
                    continue
                with open(os.path.join(json_dir, jfile), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:
                    continue

                image_name = os.path.splitext(data[0]["page_info"]["image_path"])[0]
                for det in data[0].get("layout_dets", []):
                    cat = det.get("category_type", "text_block")
                    category_id = CAT_MAP.get(cat, 1)
                    poly = det.get("poly", [0, 0, 0, 0, 0, 0, 0, 0])
                    bbox = poly_to_bbox(poly)
                    all_results.append({
                        "image_name": image_name,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": 1.0,
                    })
                pages_processed += 1

    output = {"categories": {str(v): k for k, v in CATEGORIES.items()}, "results": all_results}

    layout_path = os.path.join(output_dir, "layout_detection.json")
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    formula_results = [r for r in all_results if r["category_id"] == 8]
    formula_output = {
        "categories": {"0": "isolate_formula"},
        "results": formula_results,
    }
    formula_path = os.path.join(output_dir, "formula_detection.json")
    with open(formula_path, "w", encoding="utf-8") as f:
        json.dump(formula_output, f, ensure_ascii=False, indent=2)

    return pages_processed, len(all_results)


def build_detection_gt(gt_root, output_dir):
    all_pages = []
    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            json_dir = os.path.join(subject_path, doc_id, "json")
            if not os.path.isdir(json_dir):
                continue
            for jfile in sorted(os.listdir(json_dir)):
                if not jfile.endswith(".json"):
                    continue
                with open(os.path.join(json_dir, jfile), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:
                    continue
                all_pages.append(data[0])

    gt_path = os.path.join(output_dir, "det_gt.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    return len(all_pages), gt_path


def build_detection_config(gt_file, pred_file, output_dir):
    config = {
        "detection_eval": {
            "metrics": ["COCODet"],
            "dataset": {
                "dataset_name": "detection_dataset_simple_format",
                "ground_truth": {"data_path": gt_file},
                "prediction": {"data_path": pred_file},
            },
            "categories": {
                "eval_cat": {
                    "block_level": [
                        "title", "text", "abandon", "figure", "figure_caption",
                        "table", "table_caption", "table_footnote",
                        "isolate_formula", "formula_caption",
                    ]
                },
                "gt_cat_mapping": {
                    "equation_isolated": "isolate_formula",
                    "equation_inline": "isolate_formula",
                    "table": "table",
                    "figure": "figure",
                    "title": "title",
                    "text_block": "text",
                    "figure_caption": "figure_caption",
                    "table_caption": "table_caption",
                    "table_footnote": "table_footnote",
                    "formula_caption": "formula_caption",
                    "header": "abandon",
                    "footer": "abandon",
                    "page_number": "abandon",
                    "page_footnote": "abandon",
                    "reference": "text",
                    "code_algorithm": "text",
                    "code_algorithm_caption": "text",
                    "figure_footnote": "text",
                },
                "pred_cat_mapping": {
                    "title": "title",
                    "text_block": "text",
                    "abandon": "abandon",
                    "figure": "figure",
                    "figure_caption": "figure_caption",
                    "table": "table",
                    "table_caption": "table_caption",
                    "table_footnote": "table_footnote",
                    "equation_isolated": "isolate_formula",
                    "formula_caption": "formula_caption",
                },
            },
        }
    }
    import yaml
    out_path = os.path.join(output_dir, "detection_eval.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return out_path


def build_recognition_json(gt_root, output_dir):
    all_pages = []
    all_flat = []
    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            json_dir = os.path.join(subject_path, doc_id, "json")
            if not os.path.isdir(json_dir):
                continue
            for jfile in sorted(os.listdir(json_dir)):
                if not jfile.endswith(".json"):
                    continue
                with open(os.path.join(json_dir, jfile), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not data:
                    continue

                page_data = data[0]
                new_dets = []
                for det in page_data.get("layout_dets", []):
                    cat = det.get("category_type", "")
                    new_det = dict(det)
                    if cat in ("equation_isolated", "equation_inline"):
                        new_det["pred"] = det.get("latex", "")
                    elif cat == "table":
                        html_content = det.get("html", "")
                        if not html_content.strip():
                            continue
                        new_det["pred"] = html_content
                    elif cat in ("text_block", "title", "reference"):
                        new_det["pred"] = det.get("text", "")
                    else:
                        new_det["pred"] = det.get("text", "")
                    new_dets.append(new_det)
                    all_flat.append(new_det)

                page_entry = {
                    "page_info": page_data["page_info"],
                    "layout_dets": new_dets,
                }
                all_pages.append(page_entry)

    ocr_path = os.path.join(output_dir, "ocr.json")
    with open(ocr_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    formula_pages = [
        {
            "page_info": p["page_info"],
            "layout_dets": [d for d in p["layout_dets"] if d.get("category_type") in ("equation_isolated", "equation_inline")]
        }
        for p in all_pages
        if any(d.get("category_type") in ("equation_isolated", "equation_inline") for d in p["layout_dets"])
    ]
    formula_path = os.path.join(output_dir, "formula_recognition.json")
    with open(formula_path, "w", encoding="utf-8") as f:
        json.dump(formula_pages, f, ensure_ascii=False, indent=2)

    table_pages = [
        {
            "page_info": p["page_info"],
            "layout_dets": [d for d in p["layout_dets"] if d.get("category_type") == "table"]
        }
        for p in all_pages
        if any(d.get("category_type") == "table" for d in p["layout_dets"])
    ]
    table_path = os.path.join(output_dir, "table_recognition.json")
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(table_pages, f, ensure_ascii=False, indent=2)

    return len(formula_pages), len(table_pages), len(all_pages)


def build_md_config(output_dir):
    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "display_formula": {"metric": ["Edit_dist", "CDM_plain"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "md2md_dataset",
                "ground_truth": {
                    "data_path": output_dir,
                },
                "prediction": {"data_path": output_dir},
                "match_method": "quick_match",
            },
        }
    }
    out_path = os.path.join(output_dir, "md2md_eval.yaml")
    import yaml
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build all prediction formats from GT dataset for evaluation"
    )
    parser.add_argument("--gt_root", type=str, required=True, help="Path to Datasets/dev")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    md_dir = os.path.join(args.output_dir, "md")
    det_dir = os.path.join(args.output_dir, "detection")
    rec_dir = os.path.join(args.output_dir, "recognition")
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)

    print("=== Group A: Markdown ===")
    md_count = copy_md_files(args.gt_root, md_dir)
    norm_count = normalize_formulas_in_md_files(md_dir)
    page_count = build_page_info(args.gt_root, md_dir)
    config_path = build_md_config(md_dir)
    print(f"  Copied {md_count} .md files")
    print(f"  Pre-normalized formulas in {norm_count} files")
    print(f"  Built page_info.json ({page_count} pages)")
    print(f"  Generated {config_path}")

    print("\n=== Group B: Detection ===")
    pages, dets = build_detection_json(args.gt_root, det_dir)
    gt_count, gt_path = build_detection_gt(args.gt_root, det_dir)
    pred_path = os.path.join(det_dir, "layout_detection.json")
    det_config = build_detection_config(gt_path, pred_path, det_dir)
    print(f"  {pages} pages -> {dets} detections")
    print(f"  -> {pred_path}")
    print(f"  -> {os.path.join(det_dir, 'formula_detection.json')}")
    print(f"  -> {gt_path} ({gt_count} pages)")
    print(f"  -> {det_config}")

    print("\n=== Group C: Recognition ===")
    formulas, tables, total = build_recognition_json(args.gt_root, rec_dir)
    print(f"  {total} annotations ({formulas} formulas, {tables} tables)")
    print(f"  -> {os.path.join(rec_dir, 'ocr.json')}")
    print(f"  -> {os.path.join(rec_dir, 'formula_recognition.json')}")
    print(f"  -> {os.path.join(rec_dir, 'table_recognition.json')}")


if __name__ == "__main__":
    main()
