import argparse
import json
import os
import re

from adapters import get_adapter


def get_page_number(filename: str) -> int:
    match = re.search(r"page_(\d+)", filename)
    return int(match.group(1)) if match else -1


def collect_page_files(doc_dir: str) -> list[str]:
    files = [
        os.path.join(doc_dir, f)
        for f in os.listdir(doc_dir)
        if f.endswith((".jpg", ".png"))
    ]
    return sorted(files, key=lambda p: get_page_number(os.path.basename(p)))


def build_windows(page_files: list[str], num_pages: int) -> list[list[str]]:
    n = len(page_files)
    if n <= num_pages:
        return [page_files]
    return [page_files[i : i + num_pages] for i in range(n - num_pages + 1)]


def load_prompt(prompt_path: str, num_images: int = 1) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("{num_images}", str(num_images))


def run_md(adapter, args):
    save_root = args.save_root.rstrip("/")
    if args.model_type not in save_root:
        save_root = f"{save_root}_{args.model_type}"
    save_root = f"{save_root}_{args.num_pages}p"
    os.makedirs(save_root, exist_ok=True)

    total, ok, skip = 0, 0, 0
    for subject in sorted(os.listdir(args.image_root)):
        subject_path = os.path.join(args.image_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            doc_path = os.path.join(subject_path, doc_id)
            if not os.path.isdir(doc_path):
                continue
            images_path = os.path.join(doc_path, "images")
            if not os.path.isdir(images_path):
                continue
            page_files = collect_page_files(images_path)
            if not page_files:
                continue

            for window in build_windows(page_files, args.num_pages):
                total += 1
                page_nums = [get_page_number(os.path.basename(p)) for p in window]
                start_page, end_page = page_nums[0], page_nums[-1]

                output_filename = f"{doc_id}_page_{start_page}-{end_page}.md"
                subject_dir = os.path.join(save_root, subject)
                output_path = os.path.join(subject_dir, output_filename)

                if not args.overwrite and os.path.exists(output_path):
                    skip += 1
                    continue

                os.makedirs(subject_dir, exist_ok=True)
                prompt = load_prompt(args.prompt, num_images=len(window))
                response = adapter.predict_md(window, prompt)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response)
                ok += 1
                print(f"[{args.model_type}] {subject}/{doc_id} page_{start_page}-{end_page} -> {len(response)} chars")

    print(f"\nDone: {ok} generated, {skip} skipped, {total} total windows")


def run_detection(adapter, args):
    save_dir = os.path.join(args.save_root, "detection")
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    categories = {
        "0": "title", "1": "text", "2": "abandon", "3": "figure",
        "4": "figure_caption", "5": "table", "6": "table_caption",
        "7": "table_footnote", "8": "isolate_formula", "9": "formula_caption",
    }

    total = 0
    for subject in sorted(os.listdir(args.image_root)):
        subject_path = os.path.join(args.image_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            doc_path = os.path.join(subject_path, doc_id)
            if not os.path.isdir(doc_path):
                continue
            images_path = os.path.join(doc_path, "images")
            if not os.path.isdir(images_path):
                continue

            for page_file in sorted(os.listdir(images_path)):
                if not page_file.endswith((".jpg", ".png")):
                    continue
                image_path = os.path.join(images_path, page_file)
                image_name = os.path.splitext(page_file)[0]

                dets = adapter.predict_detection(image_path)
                for det in dets:
                    det["image_name"] = image_name
                all_results.extend(dets)
                total += 1
                print(f"[{args.model_type}] {subject}/{doc_id}/{page_file} -> {len(dets)} regions")

    output = {"categories": categories, "results": all_results}
    out_path = os.path.join(save_dir, "layout_detection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {total} pages, {len(all_results)} detections -> {out_path}")


def run_recognition(adapter, args):
    save_dir = os.path.join(args.save_root, "recognition")
    os.makedirs(save_dir, exist_ok=True)

    gt_root = args.gt_root
    total = 0
    for subject in sorted(os.listdir(gt_root)):
        subject_path = os.path.join(gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for doc_id in sorted(os.listdir(subject_path)):
            doc_path = os.path.join(subject_path, doc_id)
            if not os.path.isdir(doc_path):
                continue

            images_path = os.path.join(doc_path, "images")
            json_dir = os.path.join(doc_path, "json")
            if not os.path.isdir(images_path) or not os.path.isdir(json_dir):
                continue

            for jfile in sorted(os.listdir(json_dir)):
                if not jfile.endswith(".json"):
                    continue
                gt_path = os.path.join(json_dir, jfile)
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_data = json.load(f)
                if not gt_data:
                    continue

                page_info = gt_data[0].get("page_info", {})
                image_path = os.path.join(images_path, page_info.get("image_path", ""))
                if not os.path.exists(image_path):
                    continue

                layout_dets = gt_data[0].get("layout_dets", [])
                pred_dets = adapter.predict_recognition(image_path, layout_dets)

                gt_data[0]["layout_dets"] = pred_dets
                out_path = os.path.join(save_dir, jfile)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(gt_data, f, ensure_ascii=False, indent=2)
                total += 1
                print(f"[{args.model_type}] {subject}/{doc_id}/{jfile} -> {len(pred_dets)} annotations")

    print(f"\nDone: {total} pages -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run local model inference (md / detection / recognition)"
    )
    parser.add_argument("--model_type", type=str, required=True, help="Adapter key (e.g. qwen)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root folder containing subject/doc_id/images/ subfolders")
    parser.add_argument("--save_root", type=str, required=True, help="Folder to save results")
    parser.add_argument("--prompt", type=str, default=None, help="Path to prompt .txt file (required for md mode)")
    parser.add_argument("--num_pages", type=int, default=5, help="Sliding window size for md mode")
    parser.add_argument("--eval_mode", type=str, default="md", choices=["md", "detection", "recognition"],
                        help="Evaluation mode: md (markdown), detection (layout bbox), recognition (ocr/formula/table)")
    parser.add_argument("--gt_root", type=str, default=None,
                        help="GT root path (required for recognition mode, e.g. Datasets/dev)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    adapter = get_adapter(args.model_type)
    adapter.load_model(args.model_path, device=args.device)

    if args.eval_mode == "md":
        if not args.prompt:
            parser.error("--prompt is required for md mode")
        run_md(adapter, args)
    elif args.eval_mode == "detection":
        run_detection(adapter, args)
    elif args.eval_mode == "recognition":
        if not args.gt_root:
            parser.error("--gt_root is required for recognition mode")
        run_recognition(adapter, args)


if __name__ == "__main__":
    main()
