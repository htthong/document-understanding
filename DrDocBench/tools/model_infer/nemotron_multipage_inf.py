"""
Multi-page sliding-window Nemotron img2md inference.

Output files: {save_root}/{subject}/{doc_id}_page_{start}-{end}.md
"""
import collections
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
import argparse
import os
from tqdm import tqdm

from infer_utils import encode_image, get_page_number, build_windows, collect_page_files, load_prompt, add_subjects_arg, parse_subjects

random.seed(42)

API_KEY = open("api_key.txt", "r").read().strip()
BASE_URL = open("base_url.txt", "r").read().strip()

FAILED_JOBS_FILE = "failed_jobs.jsonl"

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

_rate_limiter = None


class _TokenBucket:
    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval


def get_response(image_paths, prompt, model, max_retries=3):
    content = []
    for image_path in image_paths:
        img_str = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
        })
    content.append({"type": "text", "text": prompt})

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for attempt in range(max_retries + 1):
        if _rate_limiter is not None:
            _rate_limiter.acquire()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            return completion.choices[0].message.content
        except _RETRYABLE as e:
            if attempt == max_retries:
                raise
            cap = min(2 ** (attempt + 1), 60)
            wait = random.uniform(0, cap)
            print(f"[RETRY {attempt + 1}/{max_retries}] {type(e).__name__}: {e} — retrying in {wait:.1f}s")
            time.sleep(wait)


def process_window(subject, doc_id, window_paths, save_root, model, prompt_path,
                   overwrite=False, failed_jobs_lock=None, max_retries=3):
    page_nums = [get_page_number(os.path.basename(p)) for p in window_paths]
    start_page, end_page = page_nums[0], page_nums[-1]

    output_filename = f"{doc_id}_page_{start_page}-{end_page}.md"
    subject_save_root = os.path.join(save_root, subject)
    output_path = os.path.join(subject_save_root, output_filename)

    if not overwrite and os.path.exists(output_path):
        return f"[SKIP] Already exists: {output_filename}"

    os.makedirs(subject_save_root, exist_ok=True)

    missing = [p for p in window_paths if not os.path.exists(p)]
    if missing:
        error_msg = f"Missing page files: {missing}"
        _record_failure(subject, doc_id, window_paths, error_msg, "MISSING_PAGE",
                        save_root, failed_jobs_lock)
        return f"[FAIL:MISSING_PAGE] {output_filename}: {error_msg}"

    try:
        prompt = load_prompt(prompt_path, num_images=len(window_paths))
        response = get_response(window_paths, prompt, model, max_retries=max_retries)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        return f"[OK] {output_filename}"
    except Exception as e:
        error_msg = str(e)
        _record_failure(subject, doc_id, window_paths, error_msg, "API_ERROR",
                        save_root, failed_jobs_lock)
        return f"[FAIL:API_ERROR] {output_filename}: {error_msg}"


def _record_failure(subject, doc_id, window_paths, error_msg, error_type, save_root, lock):
    entry = json.dumps({
        "subject": subject,
        "doc_id": doc_id,
        "window_paths": window_paths,
        "error": error_msg,
        "error_type": error_type,
    })
    failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
    with lock:
        with open(failed_jobs_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def load_failed_jobs(save_root):
    path = os.path.join(save_root, FAILED_JOBS_FILE)
    if not os.path.exists(path):
        print(f"[INFO] No failed jobs file found at {path}")
        return [], [], None

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    retryable = [e for e in entries if e["error_type"] != "MISSING_PAGE"]
    permanent = [e for e in entries if e["error_type"] == "MISSING_PAGE"]

    if permanent:
        print(f"[WARN] {len(permanent)} MISSING_PAGE failure(s) skipped (fix source data):")
        for e in permanent:
            print(f"       {e['doc_id']} — {e['error']}")

    if retryable:
        error_counts = collections.Counter(e["error"] for e in retryable)
        print(f"[INFO] {len(retryable)} API_ERROR job(s) queued for retry (from {path}). Error breakdown:")
        for msg, n in error_counts.most_common():
            print(f"       {n:3}x  {msg}")
    else:
        print(f"[INFO] 0 API_ERROR job(s) queued for retry (from {path})")

    processing_path = path + ".processing"
    os.rename(path, processing_path)

    return [(e["subject"], e["doc_id"], e["window_paths"]) for e in retryable], permanent, processing_path


def main():
    parser = argparse.ArgumentParser(description="Multi-page sliding-window Nemotron inference for document images")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="nemotron-nano-12b-v2-vl")
    parser.add_argument("--threads", type=int, default=50)
    parser.add_argument("--num_pages", type=int, default=2)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--rps", type=float, default=10.0,
                        help="Max requests per second (token-bucket rate limiter)")
    parser.add_argument("--debug", action="store_true",
                        help="Process only the first 10 documents, overwrite existing results")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only the jobs recorded in failed_jobs.jsonl from a previous run.")
    add_subjects_arg(parser)
    args = parser.parse_args()

    save_root = args.save_root

    if args.model not in save_root:
        print(f"Model name '{args.model}' not found in save_root path '{save_root}'. "
              "Please ensure the model name is included in the save_root path for clarity.")

    save_root = save_root.rstrip("/") + f"_{args.num_pages}p"

    if "close" in args.image_root:
        save_root = os.path.join(save_root, "close_source")
    elif "open" in args.image_root:
        save_root = os.path.join(save_root, "open_source")
    elif "sell" in args.image_root:
        save_root = os.path.join(save_root, "sell")

    os.makedirs(save_root, exist_ok=True)

    processing_path = None
    permanent_entries = []
    if args.retry_failed:
        work_items, permanent_entries, processing_path = load_failed_jobs(save_root)
        if not work_items:
            print("Nothing to retry.")
            if permanent_entries and processing_path and os.path.exists(processing_path):
                failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
                with open(failed_jobs_path, "a", encoding="utf-8") as f:
                    for e in permanent_entries:
                        f.write(json.dumps(e) + "\n")
                os.remove(processing_path)
            return
        print(f"Retrying {len(work_items)} previously failed jobs.")
    else:
        subjects_filter = parse_subjects(args)
        doc_entries = []
        for subject in sorted(os.listdir(args.image_root)):
            subject_path = os.path.join(args.image_root, subject)
            if not os.path.isdir(subject_path):
                continue
            if subjects_filter and subject not in subjects_filter:
                continue
            for doc_id in sorted(os.listdir(subject_path)):
                doc_path = os.path.join(subject_path, doc_id)
                if not os.path.isdir(doc_path):
                    continue
                images_path = os.path.join(doc_path, "images")
                if os.path.isdir(images_path):
                    doc_entries.append((subject, doc_id, images_path))

        if args.debug:
            doc_entries = doc_entries[:10]

        work_items = []
        for subject, doc_id, images_path in doc_entries:
            page_files = collect_page_files(images_path)
            if not page_files:
                continue
            for window in build_windows(page_files, args.num_pages):
                work_items.append((subject, doc_id, window))

        print(f"Total inference calls: {len(work_items)} "
              f"(across {len(doc_entries)} documents, window size={args.num_pages})")

    global _rate_limiter
    _rate_limiter = _TokenBucket(args.rps)
    print(f"Rate limiter: {args.rps} req/s")

    failed_jobs_lock = threading.Lock()

    def worker(item):
        subject, doc_id, window_paths = item
        return process_window(
            subject, doc_id, window_paths, save_root, args.model, args.prompt,
            overwrite=args.debug,
            failed_jobs_lock=failed_jobs_lock,
            max_retries=args.max_retries,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(tqdm(executor.map(worker, work_items),
                                total=len(work_items), desc="Processing"))
    except KeyboardInterrupt:
        if processing_path and os.path.exists(processing_path):
            os.rename(processing_path, os.path.join(save_root, FAILED_JOBS_FILE))
            print("\n[INFO] Run interrupted — failed_jobs.jsonl restored. Re-run with --retry-failed to resume.")
        raise

    if processing_path and os.path.exists(processing_path):
        if permanent_entries:
            failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
            with open(failed_jobs_path, "a", encoding="utf-8") as f:
                for e in permanent_entries:
                    f.write(json.dumps(e) + "\n")
        os.remove(processing_path)

    ok           = sum(1 for r in results if r.startswith("[OK]"))
    skip         = sum(1 for r in results if r.startswith("[SKIP]"))
    fail_missing = sum(1 for r in results if r.startswith("[FAIL:MISSING_PAGE]"))
    fail_api     = sum(1 for r in results if r.startswith("[FAIL:API_ERROR]"))

    print(f"\nDone: {ok} succeeded, {skip} skipped, "
          f"{fail_missing} failed (missing page), {fail_api} failed (API error)")

    total_fail = fail_missing + fail_api
    if total_fail:
        failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
        print(f"{total_fail} failed jobs saved to: {failed_jobs_path}")
        print("Re-run with --retry-failed to retry API errors.")


if __name__ == "__main__":
    main()
