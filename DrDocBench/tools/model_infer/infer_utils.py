"""
Provider-agnostic utilities shared across inference scripts.

Usage
-----
    from infer_utils import encode_image, get_page_number, build_windows, collect_page_files, load_prompt, add_subjects_arg
"""

import base64
import os
import re


def get_page_number(filename: str) -> int:
    """Extract the integer page number from a filename like 'page_42.jpg'."""
    match = re.search(r'page_(\d+)', filename)
    return int(match.group(1)) if match else -1


def encode_image(image_path: str) -> str:
    """Return the base64-encoded contents of an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_windows(page_files: list[str], num_pages: int) -> list[list[str]]:
    """
    Produce sliding windows of size `num_pages` over a sorted list of page paths.

    - If the document has fewer pages than `num_pages`, one window with all
      pages is returned (handles single-page documents naturally).
    - Otherwise windows are [0..k-1], [1..k], [2..k+1], …
    """
    n = len(page_files)
    if n <= num_pages:
        return [page_files]
    return [page_files[i:i + num_pages] for i in range(n - num_pages + 1)]


def collect_page_files(doc_dir: str) -> list[str]:
    """
    Return page image paths inside `doc_dir`, sorted by page number.
    Accepts .jpg and .png files whose names contain 'page_N'.
    """
    files = [
        os.path.join(doc_dir, f)
        for f in os.listdir(doc_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    return sorted(files, key=lambda p: get_page_number(os.path.basename(p)))


def add_subjects_arg(parser) -> None:
    """Add --subjects to an argparse parser (shared across all inference scripts)."""
    parser.add_argument(
        "--subjects", type=str, default=None,
        help="Comma-separated list of subjects to process (e.g. MUSIC). "
             "Omit to process all subjects.")


def parse_subjects(args) -> set:
    """Return a set of subject names from args.subjects, or empty set (= all)."""
    if not args.subjects:
        return set()
    return {s.strip() for s in args.subjects.split(',') if s.strip()}


def load_prompt(prompt_path: str, num_images: int = 1) -> str:
    """
    Load a prompt from a .txt file and substitute `{num_images}` with the
    actual number of images in the current window.

    Example prompt.txt:
        You are given {num_images} consecutive pages of a document. Convert them...
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("{num_images}", str(num_images))
