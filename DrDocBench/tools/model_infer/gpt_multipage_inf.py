import collections
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError
import argparse
import os
from tqdm import tqdm

from infer_utils import encode_image, get_page_number, build_windows, collect_page_files, load_prompt, add_subjects_arg, parse_subjects

random.seed(42)

API_KEY = open("api_key.txt", "r").read().strip()
BASE_URL = open("base_url.txt", "r").read().strip()

FAILED_JOBS_FILE = "failed_jobs.jsonl"

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_CONTENT_FILTER_PHRASES = (
    "Your input image may contain content that is not allowed by our content safety system.",
    "content_policy_violation",
)


class ContentFilterError(RuntimeError):
    pass


def get_gpt_response(image_paths, prompt, model="gpt-4o", max_retries=3):
    """Send one or more page images plus a prompt to the model.

    Retries transient errors (429, 5xx, timeout, connection reset) up to
    `max_retries` times with exponential backoff (2, 4, 8 … seconds).
    Permanent errors (e.g. 400 context-length) are raised immediately.
    """
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
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
            )
            return completion.choices[0].message.content
        except BadRequestError as e:
            if any(phrase in str(e) for phrase in _CONTENT_FILTER_PHRASES):
                raise ContentFilterError(str(e)) from e
            raise
        except _RETRYABLE as e:
            if attempt == max_retries:
                raise
            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s, …
            print(f"[RETRY {attempt + 1}/{max_retries}] {type(e).__name__}: {e} — retrying in {wait}s")
            time.sleep(wait)


def process_window(subject, doc_id, window_paths, save_root, model, prompt_path,
                   overwrite=False, failed_jobs_lock=None, max_retries=3):
    """Process one sliding window (list of page image paths) and save the result.

    Failures are classified as MISSING_PAGE (permanent) or API_ERROR (transient).
    Failed jobs are appended to failed_jobs.jsonl in save_root for later retry.
    """
    page_nums = [get_page_number(os.path.basename(p)) for p in window_paths]
    start_page = page_nums[0]
    end_page = page_nums[-1]

    output_filename = f"{doc_id}_page_{start_page}-{end_page}.md"
    subject_save_root = os.path.join(save_root, subject)
    output_path = os.path.join(subject_save_root, output_filename)

    if not overwrite and os.path.exists(output_path):
        return f"[SKIP] Already exists: {output_filename}"

    os.makedirs(subject_save_root, exist_ok=True)

    # Pre-flight: verify all page images exist before making any API call.
    missing = [p for p in window_paths if not os.path.exists(p)]
    if missing:
        error_msg = f"Missing page files: {missing}"
        _record_failure(subject, doc_id, window_paths, error_msg, "MISSING_PAGE",
                        save_root, failed_jobs_lock)
        return f"[FAIL:MISSING_PAGE] {output_filename}: {error_msg}"

    try:
        prompt = load_prompt(prompt_path, num_images=len(window_paths))
        response = get_gpt_response(window_paths, prompt, model, max_retries=max_retries)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)
        return f"[OK] {output_filename}"
    except ContentFilterError as e:
        error_msg = str(e)
        _record_failure(subject, doc_id, window_paths, error_msg, "CONTENT_FILTER",
                        save_root, failed_jobs_lock)
        return f"[FAIL:CONTENT_FILTER] {output_filename}: {error_msg}"
    except Exception as e:
        error_msg = str(e)
        _record_failure(subject, doc_id, window_paths, error_msg, "API_ERROR",
                        save_root, failed_jobs_lock)
        return f"[FAIL:API_ERROR] {output_filename}: {error_msg}"


def _record_failure(subject, doc_id, window_paths, error_msg, error_type, save_root, lock):
    """Append a failed job entry to failed_jobs.jsonl in a thread-safe way."""
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
    """Load retryable work items from a previous failed_jobs.jsonl.

    Only API_ERROR entries are returned for retry — MISSING_PAGE failures
    are permanent and require fixing the data, not re-running the job.
    The file is cleared after loading so this run starts fresh.
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

    _non_retryable = {"MISSING_PAGE", "CONTENT_FILTER"}
    retryable = [e for e in entries if e["error_type"] not in _non_retryable]
    permanent = [e for e in entries if e["error_type"] in _non_retryable]

    if permanent:
        missing = [e for e in permanent if e["error_type"] == "MISSING_PAGE"]
        filtered = [e for e in permanent if e["error_type"] == "CONTENT_FILTER"]
        if missing:
            print(f"[WARN] {len(missing)} MISSING_PAGE failure(s) skipped (fix source data):")
            for e in missing:
                print(f"       {e['doc_id']} — {e['error']}")
        if filtered:
            print(f"[INFO] {len(filtered)} CONTENT_FILTER failure(s) skipped (permanent, left in failed_jobs):")

    if retryable:
        error_counts = collections.Counter(e["error"] for e in retryable)
        print(f"[INFO] {len(retryable)} API_ERROR job(s) queued for retry (from {path}). Error breakdown:")
        for msg, n in error_counts.most_common():
            print(f"       {n:3}x  {msg}")
    else:
        print(f"[INFO] 0 API_ERROR job(s) queued for retry (from {path})")

    # Rename to .processing so the original is preserved if the run is
    # interrupted (Ctrl+C, OOM, etc.). It is deleted only after the run
    # completes — see the finally block in main().
    processing_path = path + ".processing"
    os.rename(path, processing_path)

    return [(e["subject"], e["doc_id"], e["window_paths"]) for e in retryable], permanent, processing_path


def main():
    parser = argparse.ArgumentParser(description="Multi-page sliding-window GPT inference for document images")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root folder (e.g. /root/close_source_benchmark or /root/open_source) containing subject/doc_id/images/ subfolders")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Folder to save markdown results")
    parser.add_argument("--model", type=str, default="gpt-5.5", help="Model name")
    parser.add_argument("--threads", type=int, default=100, help="Number of parallel threads")
    parser.add_argument("--num_pages", type=int, default=5,
                        help="Sliding window size: number of consecutive pages sent per inference call")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Path to a .txt file containing the prompt. "
                             "Use {num_images} anywhere in the file to insert the window size.")
    parser.add_argument("--max-retries", type=int, default=20,
                        help="Max in-request retries for transient API errors (429, 5xx, timeout) "
                             "with exponential backoff. Permanent errors (e.g. 400) are never retried.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only the first 10 documents, overwrite existing results")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only the jobs recorded in failed_jobs.jsonl from a previous run. "
                             "Clears the file before starting so new failures are tracked fresh.")
    add_subjects_arg(parser)
    args = parser.parse_args()

    image_root = args.image_root
    save_root = args.save_root

    if args.model not in save_root:
        print(f"Model name '{args.model}' not found in save_root path '{save_root}'. "
              "Please ensure the model name is included in the save_root path for clarity.")
        if "gpt-4o" in save_root:
            save_root = save_root.replace("gpt-4o", args.model)
            print(f"Updated save_root to: {save_root}")
        else:
            print("Unable to automatically update save_root. "
                  "Please manually ensure the model name is included in the save_root path.")
            return

    save_root = save_root.rstrip("/") + f"_{args.num_pages}p"

    if "close" in image_root:
        save_root = os.path.join(save_root, "close_source")
    elif "open" in image_root:
        save_root = os.path.join(save_root, "open_source")
    elif "sell" in image_root:
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
        doc_entries = []  # (subject, doc_id, images_path)
        for subject in sorted(os.listdir(image_root)):
            subject_path = os.path.join(image_root, subject)
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
        # Restore the .processing file so the job list isn't lost.
        if processing_path and os.path.exists(processing_path):
            os.rename(processing_path, os.path.join(save_root, FAILED_JOBS_FILE))
            print(f"\n[INFO] Run interrupted — original failed_jobs.jsonl restored. "
                  f"Re-run with --retry-failed to resume.")
        raise

    if processing_path and os.path.exists(processing_path):
        if permanent_entries:
            failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
            with open(failed_jobs_path, "a", encoding="utf-8") as f:
                for e in permanent_entries:
                    f.write(json.dumps(e) + "\n")
        os.remove(processing_path)

    ok            = sum(1 for r in results if r.startswith("[OK]"))
    skip          = sum(1 for r in results if r.startswith("[SKIP]"))
    fail_missing  = sum(1 for r in results if r.startswith("[FAIL:MISSING_PAGE]"))
    fail_api      = sum(1 for r in results if r.startswith("[FAIL:API_ERROR]"))
    fail_filtered = sum(1 for r in results if r.startswith("[FAIL:CONTENT_FILTER]"))

    print(f"\nDone: {ok} succeeded, {skip} skipped, "
          f"{fail_missing} failed (missing page), {fail_api} failed (API error), "
          f"{fail_filtered} failed (content filter)")

    total_fail = fail_missing + fail_api + fail_filtered
    if total_fail:
        failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
        print(f"{total_fail} failed jobs saved to: {failed_jobs_path}")
        print("Re-run with --retry-failed to retry API errors.")


if __name__ == "__main__":
    main()
