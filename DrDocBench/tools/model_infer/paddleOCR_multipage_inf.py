"""
Multi-page sliding-window PaddleOCR API (Baidu VL Parser) img2md inference.

Bundles each sliding window of pages into a single in-memory PDF and submits
it to the Baidu PaddleVL Parser API as one task (max 500 pages). The API
returns a per-page `pages` array which is concatenated into the window output.

Output files: {save_root}/{subject}/{doc_id}_page_{start}-{end}.md
"""
import base64
import collections
import io
import json
import os
import threading
import time
import argparse
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from infer_utils import get_page_number, build_windows, collect_page_files, add_subjects_arg, parse_subjects

FAILED_JOBS_FILE = "failed_jobs.jsonl"

SUBMIT_URL = "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task"
QUERY_URL  = "https://aip.baidubce.com/rest/2.0/brain/online/v2/paddle-vl-parser/task/query"

POLL_INTERVAL_S = 3
POLL_MAX_TRIES  = 120   # ~6 minutes before giving up

# Global rate limiter for submit calls (shared across all threads)
_submit_lock = threading.Lock()
_last_submit_time = 0.0
_submit_gap_s = 0.5


_QPS_RETRY_DELAYS = [5, 10, 20, 40, 60]  # seconds to wait after each error-18 hit


def _images_to_pdf_bytes(image_paths: list[str]) -> bytes:
    """Bundle page images into a single in-memory PDF."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    buf = io.BytesIO()
    if len(images) == 1:
        images[0].save(buf, format="PDF")
    else:
        images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def _submit_task(access_token: str, file_bytes: bytes, file_name: str) -> str:
    """Submit a file to the API and return the task_id."""
    global _last_submit_time

    url = f"{SUBMIT_URL}?access_token={access_token}"
    file_data = base64.b64encode(file_bytes).decode("utf-8")
    data = {
        "file_data": file_data,
        "file_url":  "",
        "file_name": file_name,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for attempt, retry_delay in enumerate([0] + _QPS_RETRY_DELAYS):
        if retry_delay:
            time.sleep(retry_delay)

        with _submit_lock:
            now = time.time()
            wait = _submit_gap_s - (now - _last_submit_time)
            if wait > 0:
                time.sleep(wait)
            _last_submit_time = time.time()

        resp = requests.post(url, headers=headers, data=data, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        error_code = body.get("error_code", 0)
        if error_code == 0:
            return body["result"]["task_id"]
        if error_code == 18:
            next_delay = _QPS_RETRY_DELAYS[attempt] if attempt < len(_QPS_RETRY_DELAYS) else None
            if next_delay is None:
                break
            print(f"[WARN] QPS limit hit for {file_name}, "
                  f"retry {attempt + 1}/{len(_QPS_RETRY_DELAYS)} in {next_delay}s")
            continue
        raise RuntimeError(f"Submit error {error_code}: {body.get('error_msg')}")

    raise RuntimeError(f"Submit error 18: QPS limit still hit after {len(_QPS_RETRY_DELAYS)} retries")


def _poll_task(access_token: str, task_id: str) -> dict:
    """Poll until the task completes and return the result dict."""
    url = f"{QUERY_URL}?access_token={access_token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    for _ in range(POLL_MAX_TRIES):
        resp = requests.post(url, headers=headers, data={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if body.get("error_code", 0) != 0:
            raise RuntimeError(f"Query error {body['error_code']}: {body.get('error_msg')}")
        result = body["result"]
        status = result.get("status")
        if status == "success":
            return result
        if status == "failed":
            raise RuntimeError(f"Task failed: {result.get('task_error')}")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Task {task_id} did not finish after {POLL_MAX_TRIES} polls")


def run_api_on_window(window_paths: list[str], access_token: str) -> tuple:
    """Bundle window pages into a PDF, submit as one task. Returns (markdown_text, parse_data_dict)."""
    page_nums = [get_page_number(os.path.basename(p)) for p in window_paths]
    pdf_name = f"window_{page_nums[0]}-{page_nums[-1]}.pdf"
    pdf_bytes = _images_to_pdf_bytes(window_paths)

    task_id = _submit_task(access_token, pdf_bytes, pdf_name)
    result = _poll_task(access_token, task_id)

    parse_result_url = result.get("parse_result_url")
    if not parse_result_url:
        raise RuntimeError(f"No parse_result_url in response for task {result.get('task_id')}")
    parse_resp = requests.get(parse_result_url, timeout=60)
    parse_resp.raise_for_status()
    parse_data = parse_resp.json()
    pages = parse_data.get("pages", [])
    markdown = "\n\n".join(page.get("text", "") for page in pages)
    return markdown, parse_data


def process_window(subject, doc_id, window_paths, save_root, access_token,
                   overwrite=False, failed_jobs_lock=None):
    if failed_jobs_lock is None:
        failed_jobs_lock = threading.Lock()

    page_nums  = [get_page_number(os.path.basename(p)) for p in window_paths]
    start_page = page_nums[0]
    end_page   = page_nums[-1]

    base_name         = f"{doc_id}_page_{start_page}-{end_page}"
    subject_save_root = os.path.join(save_root, subject)
    output_path       = os.path.join(subject_save_root, base_name + ".md")
    json_path         = os.path.join(subject_save_root, base_name + ".json")

    if not overwrite and os.path.exists(output_path):
        return f"[SKIP] Already exists: {base_name}.md"

    missing = [p for p in window_paths if not os.path.exists(p)]
    if missing:
        error_msg = f"Missing page files: {missing}"
        _record_failure(subject, doc_id, window_paths, error_msg, "MISSING_PAGE",
                        save_root, failed_jobs_lock)
        return f"[FAIL:MISSING_PAGE] {base_name}.md: {error_msg}"

    os.makedirs(subject_save_root, exist_ok=True)

    try:
        markdown, parse_data = run_api_on_window(window_paths, access_token)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parse_data, f, ensure_ascii=False, indent=2)
        return f"[OK] {base_name}.md"
    except Exception as e:
        error_msg = str(e)
        _record_failure(subject, doc_id, window_paths, error_msg, "PROC_ERROR",
                        save_root, failed_jobs_lock)
        return f"[FAIL:PROC_ERROR] {base_name}.md: {error_msg}"


def _record_failure(subject, doc_id, window_paths, error_msg, error_type, save_root, lock):
    entry = json.dumps({
        "subject":      subject,
        "doc_id":       doc_id,
        "window_paths": window_paths,
        "error":        error_msg,
        "error_type":   error_type,
    })
    failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
    with lock:
        with open(failed_jobs_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def load_failed_jobs(save_root):
    """Load retryable (PROC_ERROR) jobs from a previous failed_jobs.jsonl.

    MISSING_PAGE entries are skipped — fix source data and re-run normally.
    The file is renamed to .processing before starting; restored on interrupt.
    """
    path = os.path.join(save_root, FAILED_JOBS_FILE)
    if not os.path.exists(path):
        print(f"[INFO] No failed jobs file found at {path}")
        return [], None

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    retryable = [e for e in entries if e["error_type"] == "PROC_ERROR"]
    permanent = [e for e in entries if e["error_type"] == "MISSING_PAGE"]

    if permanent:
        print(f"[WARN] {len(permanent)} MISSING_PAGE failure(s) skipped (fix source data):")
        for e in permanent:
            print(f"       {e['doc_id']} — {e['error']}")

    if retryable:
        error_counts = collections.Counter(e["error"] for e in retryable)
        print(f"[INFO] {len(retryable)} PROC_ERROR job(s) queued for retry. Error breakdown:")
        for msg, n in error_counts.most_common():
            print(f"       {n:3}x  {msg}")
    else:
        print("[INFO] 0 PROC_ERROR job(s) queued for retry")

    processing_path = path + ".processing"
    os.rename(path, processing_path)

    return [(e["subject"], e["doc_id"], e["window_paths"]) for e in retryable], processing_path


def main():
    parser = argparse.ArgumentParser(
        description="Multi-page sliding-window PaddleOCR API img2md inference")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root folder containing subject/doc_id/images/ subfolders")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Folder to save markdown results")
    parser.add_argument("--access_token", type=str,
                        default="baidu_access_token.txt",
                        help="Baidu API access token (or set BAIDU_ACCESS_TOKEN env var)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of parallel worker threads")
    parser.add_argument("--num_pages", type=int, default=2,
                        help="Sliding window size: consecutive pages per output file")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only the first 10 documents, "
                             "overwrite existing results")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only the jobs recorded in failed_jobs.jsonl from a "
                             "previous run.")
    add_subjects_arg(parser)
    parser.add_argument("--submit-delay", type=float, default=0.5,
                        help="Minimum seconds between submit API calls across all threads "
                             "(default: 0.5). Increase if hitting QPS limits.")
    args = parser.parse_args()

    global _submit_gap_s
    _submit_gap_s = args.submit_delay

    if not args.access_token:
        parser.error("--access_token is required (or set BAIDU_ACCESS_TOKEN env var)")
    else:
        access_token = open(args.access_token).read().strip() if os.path.isfile(args.access_token) else args.access_token

    save_root = args.save_root.rstrip("/") + f"_{args.num_pages}p"

    if "close" in args.image_root:
        save_root = os.path.join(save_root, "close_source")
    elif "open" in args.image_root:
        save_root = os.path.join(save_root, "open_source")
    elif "sell" in args.image_root:
        save_root = os.path.join(save_root, "sell")

    os.makedirs(save_root, exist_ok=True)

    processing_path = None
    if args.retry_failed:
        work_items, processing_path = load_failed_jobs(save_root)
        if not work_items:
            print("Nothing to retry.")
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

        print(f"Total windows: {len(work_items)} "
              f"(across {len(doc_entries)} documents, window size={args.num_pages})")

    failed_jobs_lock = threading.Lock()

    def worker(item):
        subject, doc_id, window_paths = item
        return process_window(
            subject, doc_id, window_paths, save_root, access_token,
            overwrite=args.debug,
            failed_jobs_lock=failed_jobs_lock,
        )

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(tqdm(executor.map(worker, work_items),
                                total=len(work_items), desc="Processing"))
    except KeyboardInterrupt:
        if processing_path and os.path.exists(processing_path):
            os.rename(processing_path, os.path.join(save_root, FAILED_JOBS_FILE))
            print("\n[INFO] Run interrupted — failed_jobs.jsonl restored. "
                  "Re-run with --retry-failed to resume.")
        raise

    if processing_path and os.path.exists(processing_path):
        os.remove(processing_path)

    ok           = sum(1 for r in results if r.startswith("[OK]"))
    skip         = sum(1 for r in results if r.startswith("[SKIP]"))
    fail_missing = sum(1 for r in results if r.startswith("[FAIL:MISSING_PAGE]"))
    fail_proc    = sum(1 for r in results if r.startswith("[FAIL:PROC_ERROR]"))

    print(f"\nDone: {ok} succeeded, {skip} skipped, "
          f"{fail_missing} failed (missing page), {fail_proc} failed (processing error)")

    total_fail = fail_missing + fail_proc
    if total_fail:
        failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
        print(f"{total_fail} failed jobs saved to: {failed_jobs_path}")
        print("Re-run with --retry-failed to retry processing errors.")


if __name__ == "__main__":
    main()
