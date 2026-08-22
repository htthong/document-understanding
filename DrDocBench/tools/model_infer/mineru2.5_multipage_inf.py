"""
Multi-page sliding-window MinerU 2.5 API img2md inference.

For each window: page images are bundled into a single in-memory PDF and
submitted to MinerU as one document so MinerU processes all pages together
with full cross-page context. The result zip is parsed via content_list.json
(items are in document reading order, so no page splitting is needed).

Output files: {save_root}/{subject}/{doc_id}_page_{start}-{end}.md
"""
import collections
import io
import json
import os
import random
import threading
import time
import argparse
import zipfile
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from infer_utils import get_page_number, build_windows, collect_page_files, add_subjects_arg, parse_subjects

FAILED_JOBS_FILE = "failed_jobs.jsonl"

BATCH_SUBMIT_URL = "https://mineru.net/api/v4/file-urls/batch"
BATCH_POLL_URL_TMPL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"

POLL_INTERVAL_S = 10
POLL_MAX_TRIES = 120  # 20 minutes max
SUBMIT_MAX_RETRIES = 5
SUBMIT_RETRY_BASE_S = 30  # exponential backoff base for 429

_TERMINAL_STATES = {"done", "failed"}


def _headers(token: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _images_to_pdf_bytes(image_paths: list[str]) -> bytes:
    """Bundle page images into a single in-memory PDF."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    buf = io.BytesIO()
    if len(images) == 1:
        images[0].save(buf, format="PDF")
    else:
        images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    return buf.getvalue()


def _submit_single_file(token: str, filename: str) -> tuple[str, str]:
    """
    Register one file in a new batch, retrying on 429 with exponential backoff.
    Returns (batch_id, upload_url).
    """
    payload = {"files": [{"name": filename, "data_id": "0"}], "model_version": "vlm"}
    for attempt in range(SUBMIT_MAX_RETRIES):
        resp = requests.post(BATCH_SUBMIT_URL, headers=_headers(token), json=payload, timeout=60)
        if resp.status_code == 429:
            delay = SUBMIT_RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 10)
            time.sleep(delay)
            continue
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Batch submit failed: {body.get('msg')}")
        return body["data"]["batch_id"], body["data"]["file_urls"][0]
    raise RuntimeError(f"Batch submit failed after {SUBMIT_MAX_RETRIES} retries: 429 Too Many Requests")


def _poll_batch_all(token: str, batch_id: str, num_files: int) -> list[dict]:
    """
    Poll until ALL files in the batch reach a terminal state (done/failed).
    Returns the full extract_result list.
    """
    url = BATCH_POLL_URL_TMPL.format(batch_id=batch_id)
    for _ in range(POLL_MAX_TRIES):
        resp = requests.get(url, headers=_headers(token), timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Batch poll error: {body.get('msg')}")
        results = body["data"].get("extract_result", [])
        if len(results) == num_files and all(
            r.get("state") in _TERMINAL_STATES for r in results
        ):
            failed = [r for r in results if r.get("state") == "failed"]
            if failed:
                msgs = "; ".join(
                    f"{r['file_name']}: {r.get('err_msg', '')}" for r in failed
                )
                raise RuntimeError(f"Some pages failed extraction: {msgs}")
            return results
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(
        f"Batch {batch_id} did not finish after {POLL_MAX_TRIES * POLL_INTERVAL_S}s"
    )


def _zip_to_md(zip_url: str) -> str:
    """
    Download a MinerU result zip and return its markdown string.
    Prefers content_list.json; falls back to the .md file.
    """
    resp = requests.get(zip_url, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        cl_name = next((n for n in names if n.endswith("content_list.json")), None)
        md_name = next((n for n in names if n.endswith(".md")), None)
        if cl_name:
            with zf.open(cl_name) as f:
                return _content_list_to_md(json.load(f))
        if md_name:
            with zf.open(md_name) as f:
                return f.read().decode("utf-8")
    raise RuntimeError(f"Result zip has no content_list.json or .md: {zip_url}")


def _coerce_str(val) -> str:
    """MinerU fields like table_caption / img_caption can be a list or a string."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val if v)
    return val or ""


def _content_list_to_md(content_list: list) -> str:
    """Convert a content_list to a markdown string. Items are iterated in document order."""
    parts = []
    for item in content_list:
        item_type = item.get("type", "")
        if item_type in ("text", "equation"):
            text = item.get("text", "").strip()
        elif item_type == "table":
            caption = _coerce_str(item.get("table_caption", ""))
            body = _coerce_str(item.get("table_body", ""))
            text = "\n".join(filter(None, [caption, body])).strip()
        elif item_type == "image":
            text = _coerce_str(item.get("img_caption", "")).strip()
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _process_window(token: str, window: list[str]) -> str:
    """
    Bundle the window's pages into one PDF, submit to MinerU as a single document
    so all pages are processed together, then return the extracted markdown.
    """
    page_nums = [get_page_number(os.path.basename(p)) for p in window]
    pdf_name = f"window_{page_nums[0]}-{page_nums[-1]}.pdf"

    pdf_bytes = _images_to_pdf_bytes(window)
    batch_id, upload_url = _submit_single_file(token, pdf_name)

    up = requests.put(upload_url, data=pdf_bytes, timeout=300)
    if up.status_code != 200:
        raise RuntimeError(f"PDF upload failed: HTTP {up.status_code}")

    entries = _poll_batch_all(token, batch_id, 1)
    return _zip_to_md(entries[0]["full_zip_url"])


def process_document(subject, doc_id, page_files, save_root, token,
                     num_pages, overwrite=False, failed_jobs_lock=None):
    if failed_jobs_lock is None:
        failed_jobs_lock = threading.Lock()

    subject_save_root = os.path.join(save_root, subject)
    windows = build_windows(page_files, num_pages)

    # Skip early if all windows for this document already exist
    if not overwrite:
        if all(
            os.path.exists(os.path.join(
                subject_save_root,
                f"{doc_id}_page_{get_page_number(os.path.basename(w[0]))}"
                f"-{get_page_number(os.path.basename(w[-1]))}.md",
            ))
            for w in windows
        ):
            return [f"[SKIP] All windows exist: {doc_id}"]

    missing = [p for p in page_files if not os.path.exists(p)]
    if missing:
        error_msg = f"Missing page files: {missing}"
        _record_failure(subject, doc_id, page_files, error_msg, "MISSING_PAGE",
                        save_root, failed_jobs_lock)
        return [f"[FAIL:MISSING_PAGE] {doc_id}: {error_msg}"]

    os.makedirs(subject_save_root, exist_ok=True)

    results = []
    recorded_failure = False
    for window in windows:
        page_nums = [get_page_number(os.path.basename(p)) for p in window]
        start_page, end_page = page_nums[0], page_nums[-1]
        fname = f"{doc_id}_page_{start_page}-{end_page}.md"
        output_path = os.path.join(subject_save_root, fname)

        if not overwrite and os.path.exists(output_path):
            results.append(f"[SKIP] Already exists: {fname}")
            continue

        try:
            md = _process_window(token, window)
        except Exception as e:
            error_msg = str(e)
            if not recorded_failure:
                _record_failure(subject, doc_id, page_files, error_msg, "PROC_ERROR",
                                save_root, failed_jobs_lock)
                recorded_failure = True
            results.append(f"[FAIL:PROC_ERROR] {doc_id}: {error_msg}")
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        results.append(f"[OK] {fname}")

    return results


def _record_failure(subject, doc_id, page_files, error_msg, error_type, save_root, lock):
    entry = json.dumps({
        "subject": subject,
        "doc_id": doc_id,
        "page_files": page_files,
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

    # Write permanent failures back immediately so they survive regardless of
    # how this retry run exits (early return, KeyboardInterrupt, or success).
    if permanent:
        with open(path, "w", encoding="utf-8") as f:
            for e in permanent:
                f.write(json.dumps(e) + "\n")

    return [(e["subject"], e["doc_id"], e["page_files"]) for e in retryable], processing_path


def main():
    parser = argparse.ArgumentParser(
        description="Multi-page sliding-window MinerU 2.5 API img2md inference")
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--api_key", type=str,
                        default="mineru_api_key.txt",
                        help="Path to a file containing the MinerU API token, or the raw token")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num_pages", type=int, default=2)
    parser.add_argument("--debug", action="store_true",
                        help="Process only the first 10 documents, overwrite existing results")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only jobs recorded in failed_jobs.jsonl from a previous run")
    add_subjects_arg(parser)
    args = parser.parse_args()

    token = (
        open(args.api_key).read().strip()
        if os.path.isfile(args.api_key)
        else args.api_key
    )

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
            if processing_path and os.path.exists(processing_path):
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
            if page_files:
                work_items.append((subject, doc_id, page_files))

        print(f"Total documents: {len(work_items)} (window size={args.num_pages})")

    failed_jobs_lock = threading.Lock()

    def worker(item):
        subject, doc_id, page_files = item
        return process_document(
            subject, doc_id, page_files, save_root, token,
            args.num_pages,
            overwrite=args.debug,
            failed_jobs_lock=failed_jobs_lock,
        )

    all_results = []
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for doc_results in tqdm(executor.map(worker, work_items),
                                    total=len(work_items), desc="Processing"):
                all_results.extend(doc_results)
    except KeyboardInterrupt:
        if processing_path and os.path.exists(processing_path):
            os.rename(processing_path, os.path.join(save_root, FAILED_JOBS_FILE))
            print("\n[INFO] Interrupted — failed_jobs.jsonl restored. Re-run with --retry-failed to resume.")
        raise

    if processing_path and os.path.exists(processing_path):
        os.remove(processing_path)

    ok           = sum(1 for r in all_results if r.startswith("[OK]"))
    skip         = sum(1 for r in all_results if r.startswith("[SKIP]"))
    fail_missing = sum(1 for r in all_results if r.startswith("[FAIL:MISSING_PAGE]"))
    fail_proc    = sum(1 for r in all_results if r.startswith("[FAIL:PROC_ERROR]"))

    print(f"\nDone: {ok} succeeded, {skip} skipped, "
          f"{fail_missing} failed (missing page), {fail_proc} failed (processing error)")
    total_fail = fail_missing + fail_proc
    if total_fail:
        print(f"{total_fail} failed jobs saved to: {os.path.join(save_root, FAILED_JOBS_FILE)}")
        print("Re-run with --retry-failed to retry processing errors.")


if __name__ == "__main__":
    main()
