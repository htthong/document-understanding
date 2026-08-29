import os
import re
import sys
import argparse

MODEL_INFER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_infer")
sys.path.insert(0, MODEL_INFER_DIR)
from infer_utils import build_windows


def parse_md_page(filename, doc_id):
    m = re.match(re.escape(doc_id) + r"_(\d+)\.md$", filename)
    return int(m.group(1)) if m else None


def collect_md_pages(mds_dir, doc_id):
    pages = []
    try:
        entries = os.listdir(mds_dir)
    except OSError:
        return pages
    for f in entries:
        page_no = parse_md_page(f, doc_id)
        if page_no is not None:
            pages.append((page_no, os.path.join(mds_dir, f)))
    pages.sort(key=lambda x: x[0])
    return pages


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def convert_subject(gt_root, save_root, subject, num_pages):
    subject_dir = os.path.join(gt_root, subject)
    for doc_id in sorted(os.listdir(subject_dir)):
        doc_dir = os.path.join(subject_dir, doc_id)
        if not os.path.isdir(doc_dir):
            continue
        mds_dir = os.path.join(doc_dir, "mds")
        if not os.path.isdir(mds_dir):
            continue

        pages = collect_md_pages(mds_dir, doc_id)
        if not pages:
            continue

        path_by_page = dict(pages)
        page_nums = [p for p, _ in pages]

        for window in build_windows(page_nums, num_pages):
            start, end = window[0], window[-1]
            parts = []
            for page_no in window:
                content = read_text(path_by_page[page_no])
                if content:
                    parts.append(content)
            if not parts:
                continue

            out_dir = os.path.join(save_root, subject)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{doc_id}_page_{start}-{end}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description="Build sliding-window pseudo-predictions from per-page GT markdown.")
    parser.add_argument("--gt_root", type=str,
                        default="../Datasets/DrDocBench/dev")
    parser.add_argument("--save_root", type=str,
                        default="../pred_gt")
    parser.add_argument("--num_pages", type=int, default=5)
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subjects to process (omit for all)")
    args = parser.parse_args()

    subjects_filter = set()
    if args.subjects:
        subjects_filter = {s.strip() for s in args.subjects.split(",") if s.strip()}

    for subject in sorted(os.listdir(args.gt_root)):
        if subjects_filter and subject not in subjects_filter:
            continue
        subject_path = os.path.join(args.gt_root, subject)
        if not os.path.isdir(subject_path):
            continue
        convert_subject(args.gt_root, args.save_root, subject, args.num_pages)


if __name__ == "__main__":
    main()