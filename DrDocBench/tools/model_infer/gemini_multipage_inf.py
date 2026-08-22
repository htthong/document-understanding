import asyncio
import aiohttp
import collections
import json
import os
import random
import re
import argparse
from tqdm import tqdm

from infer_utils import encode_image, get_page_number, build_windows, collect_page_files, load_prompt, add_subjects_arg, parse_subjects

API_KEY = open("api_key.txt", "r").read().strip()
BASE_URL = open("base_url.txt", "r").read().strip()

FAILED_JOBS_FILE = "failed_jobs.jsonl"

# HTTP status codes that indicate a permanent failure — no point retrying.
# 400: bad request (e.g. context-length exceeded, malformed payload).
_PERMANENT_STATUS = {400}

# Codes we treat as transient and will retry with exponential backoff.
# 429: rate-limited. 5xx: server-side errors.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _jitter(attempt, base=2.0, cap=60.0):
    """Full-jitter exponential backoff: random in [0, min(cap, base^(attempt+1))].
    Spreads concurrent retries so they don't all hit the server at the same instant.
    """
    return random.uniform(0, min(cap, base ** (attempt + 1)))


def clean_markdown(markdown_text):
    if markdown_text.strip().startswith("```markdown"):
        markdown_text = markdown_text.strip()[len("```markdown"):].strip()
    if markdown_text.strip().endswith("```"):
        markdown_text = markdown_text.strip()[:-len("```")].strip()
    return markdown_text


async def get_gemini_response(session, image_paths, prompt, model, max_retries=10):
    """Send one or more page images plus a prompt to Gemini in a single request.

    Retries transient HTTP errors (429, 5xx) and connection errors with
    exponential backoff (2, 4, 8 … seconds). Permanent errors (e.g. 400)
    are raised immediately. Empty responses are also retried.
    """
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(p)}"}}
        for p in image_paths
    ]
    content.append({"type": "text", "text": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": content},
        ],
    }

    for attempt in range(max_retries + 1):
        try:
            async with session.post(
                url=f"{BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status in _PERMANENT_STATUS:
                    body = await response.text()
                    raise RuntimeError(f"Permanent HTTP {response.status}: {body}")

                if response.status not in (200, *_RETRYABLE_STATUS):
                    body = await response.text()
                    raise RuntimeError(f"Unexpected HTTP {response.status}: {body}")

                if response.status in _RETRYABLE_STATUS:
                    body = await response.text()
                    if attempt == max_retries:
                        raise RuntimeError(f"HTTP {response.status} after {max_retries} retries: {body}")
                    wait = _jitter(attempt)
                    print(f"[RETRY {attempt + 1}/{max_retries}] HTTP {response.status} — retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                result = await response.json()
                content = result["choices"][0]["message"]["content"]

                if not content or content.strip() == "":
                    if attempt == max_retries:
                        raise RuntimeError("Empty response after max retries")
                    wait = _jitter(attempt)
                    print(f"[RETRY {attempt + 1}/{max_retries}] Empty response — retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue

                return clean_markdown(content)

        except RuntimeError:
            raise  # permanent errors — propagate immediately
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries:
                raise RuntimeError(f"Connection error after {max_retries} retries: {e}") from e
            wait = _jitter(attempt)
            print(f"[RETRY {attempt + 1}/{max_retries}] {type(e).__name__}: {e} — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)


async def process_window(session, semaphore, subject, doc_id, window_paths, save_root, model,
                         prompt_path, overwrite, failed_jobs_lock, max_retries):
    page_nums = [get_page_number(os.path.basename(p)) for p in window_paths]
    start_page, end_page = page_nums[0], page_nums[-1]

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
        await _record_failure(subject, doc_id, window_paths, error_msg, "MISSING_PAGE", save_root, failed_jobs_lock)
        return f"[FAIL:MISSING_PAGE] {output_filename}: {error_msg}"

    async with semaphore:
        try:
            prompt = load_prompt(prompt_path, num_images=len(window_paths))
            response = await get_gemini_response(session, window_paths, prompt, model, max_retries=max_retries)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)
            return f"[OK] {output_filename}"
        except Exception as e:
            error_msg = str(e)
            await _record_failure(subject, doc_id, window_paths, error_msg, "API_ERROR", save_root, failed_jobs_lock)
            return f"[FAIL:API_ERROR] {output_filename}: {error_msg}"


async def _record_failure(subject, doc_id, window_paths, error_msg, error_type, save_root, lock):
    """Append a failed job entry to failed_jobs.jsonl in an async-safe way."""
    entry = json.dumps({
        "subject": subject,
        "doc_id": doc_id,
        "window_paths": window_paths,
        "error": error_msg,
        "error_type": error_type,
    })
    failed_jobs_path = os.path.join(save_root, FAILED_JOBS_FILE)
    async with lock:
        with open(failed_jobs_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def load_failed_jobs(save_root):
    """Load retryable work items from a previous failed_jobs.jsonl.

    Only API_ERROR entries are returned for retry — MISSING_PAGE failures
    are permanent and require fixing the data, not re-running the job.

    The file is renamed to .processing so the original is preserved if the
    run is interrupted (Ctrl+C, OOM, etc.). It is deleted only after the
    run completes — see the finally block in main().
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

    retryable = [e for e in entries if e["error_type"] == "API_ERROR"]
    permanent = [e for e in entries if e["error_type"] != "API_ERROR"]

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

    # Rename to .processing so the original is preserved if the run is interrupted.
    processing_path = path + ".processing"
    os.rename(path, processing_path)

    return [(e["subject"], e["doc_id"], e["window_paths"]) for e in retryable], processing_path


async def run(args, save_root):
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
        doc_entries = []  # (subject, doc_id, images_path)
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

    semaphore = asyncio.Semaphore(args.threads)
    failed_jobs_lock = asyncio.Lock()

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                process_window(
                    session, semaphore, subject, doc_id, window_paths, save_root, args.model,
                    args.prompt, overwrite=args.debug,
                    failed_jobs_lock=failed_jobs_lock,
                    max_retries=args.max_retries,
                )
                for subject, doc_id, window_paths in work_items
            ]

            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
                results.append(await coro)

    except KeyboardInterrupt:
        # Restore the .processing file so the job list isn't lost.
        if processing_path and os.path.exists(processing_path):
            os.rename(processing_path, os.path.join(save_root, FAILED_JOBS_FILE))
            print(f"\n[INFO] Run interrupted — original failed_jobs.jsonl restored. "
                  f"Re-run with --retry-failed to resume.")
        raise

    # Run completed cleanly — safe to discard the .processing file.
    if processing_path and os.path.exists(processing_path):
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


def main():
    parser = argparse.ArgumentParser(description="Multi-page sliding-window Gemini inference for document images")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root folder (e.g. /root/close_source_benchmark or /root/open_source) containing subject/doc_id/images/ subfolders")
    parser.add_argument("--save_root", type=str, required=True,
                        help="Folder to save markdown results")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Model name")
    parser.add_argument("--threads", type=int, default=10,
                        help="Max number of concurrent async requests")
    parser.add_argument("--num_pages", type=int, default=5,
                        help="Sliding window size: number of consecutive pages per inference call")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Path to a .txt file containing the prompt. "
                             "Use {num_images} anywhere in the file to insert the window size.")
    parser.add_argument("--max-retries", type=int, default=20,
                        help="Max retries for transient errors (429, 5xx, connection) with "
                             "exponential backoff. Permanent errors (e.g. 400) are never retried.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only the first 10 documents, overwrite existing results")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only the jobs recorded in failed_jobs.jsonl from a previous run. "
                             "Renames the file to .processing before starting so new failures are "
                             "tracked fresh; the file is restored on KeyboardInterrupt.")
    add_subjects_arg(parser)
    args = parser.parse_args()

    save_root = args.save_root
    if args.model not in save_root:
        print(f"Model name '{args.model}' not found in save_root path '{save_root}'. "
              "Please ensure the model name is included in the save_root path for clarity.")
        if re.search(r"gemini[^/]*", save_root):
            save_root = re.sub(r"gemini[^/]*", args.model, save_root)
            print(f"Updated save_root to: {save_root}")
        else:
            print("Unable to automatically update save_root. "
                  "Please manually ensure the model name is included in the save_root path.")
            return

    save_root = save_root.rstrip("/") + f"_{args.num_pages}p"

    if "close" in args.image_root:
        save_root = os.path.join(save_root, "close_source")
    elif "open" in args.image_root:
        save_root = os.path.join(save_root, "open_source")
    elif "sell" in args.image_root:
        save_root = os.path.join(save_root, "sell")

    asyncio.run(run(args, save_root))


if __name__ == "__main__":
    main()
