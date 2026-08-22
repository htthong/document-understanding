"""
Dataset class for evaluating multi-page sliding-window predictions.

Prediction files are named `{doc_id}_page_{start}-{end}.md` and live under
`pred_root/{subject}/`.

Two GT layouts are supported, detected automatically per document:

1. Per-page GT (normal case):
   `gt_root/{subject}/{doc_id}/mds/{doc_id}_{N}.md` — one file per page.
   Each prediction window is matched against the concatenation of its covered pages.

2. Whole-doc GT (e.g. MUSIC subject):
   `gt_root/{subject}/{doc_id}/mds/{doc_id}.md` — a single file for the entire document.
   Used when the content cannot be split by page (e.g. MusicXML, which is a continuous
   stream with no meaningful page boundaries aligned to the scanned images).

   For whole-doc GT, predictions are:
   - Deduplicated to a non-overlapping subset (handles both stride-1 and stride-N inference)
   - Merged in page order
   - Compared against the whole-doc GT via normalized character-level edit distance

   MusicXML-specific: non-visual metadata present in the GT but explicitly omitted by the
   inference prompt (<print>, <divisions>, <midi-device>, <midi-instrument>, etc.) is
   stripped from the GT before comparison to avoid penalizing correct predictions.

   Known limitation: formula, table, and reading-order metrics are not computed for
   whole-doc subjects — only text-block edit distance is reported.
"""

import os
import re
from collections import defaultdict
from functools import partial

from tqdm import tqdm

from registry.registry import DATASET_REGISTRY
from dataset.recog_dataset import *
from utils.extract import md_tex_filter
from utils.match_quick import match_gt2pred_quick
from utils.match import match_gt2pred_simple, match_gt2pred_no_split
from utils.read_files import read_md_file
import Levenshtein

PRED_RE = re.compile(r'^(.+)_page_(\d+)-(\d+)\.md$')
GT_RE   = re.compile(r'^(.+)_(\d+)\.md$')

# --- Whole-doc (MusicXML) helpers ---

_MUSICXML_FENCE_RE = re.compile(r'```musicxml\n(.*?)(?:```|$)', re.DOTALL)

_MUSICXML_META_PATTERNS = [
    re.compile(r'<print\b[^>]*>.*?</print>', re.DOTALL),
    re.compile(r'<print\b[^>]*/>', re.DOTALL),
    re.compile(r'<divisions>[^<]*</divisions>'),
    re.compile(r'<midi-device[^>]*/?>'),
    re.compile(r'<midi-instrument\b[^>]*>.*?</midi-instrument>', re.DOTALL),
    re.compile(r'<volume>[^<]*</volume>'),
    re.compile(r'<pan>[^<]*</pan>'),
    re.compile(r'<elevation>[^<]*</elevation>'),
]


def _extract_musicxml(text: str) -> str:
    """Return concatenated content of all ```musicxml fenced blocks."""
    parts = _MUSICXML_FENCE_RE.findall(text)
    return '\n'.join(p.strip() for p in parts if p.strip())


def _strip_musicxml_metadata(xml: str) -> str:
    """Remove non-visual MusicXML elements that models are told to omit."""
    for pat in _MUSICXML_META_PATTERNS:
        xml = pat.sub('', xml)
    return re.sub(r'\n{3,}', '\n\n', xml).strip()


def _has_per_page_gt(mds_dir: str, doc_id: str) -> bool:
    """Return True if mds_dir contains at least one {doc_id}_{N}.md file."""
    pat = re.compile(r'^' + re.escape(doc_id) + r'_\d+\.md$')
    try:
        return any(pat.match(f) for f in os.listdir(mds_dir))
    except OSError:
        return False


def _select_nonoverlapping(pred_files: list) -> list:
    """Return the largest non-overlapping subset of prediction windows, sorted by start page."""
    entries = []
    for f in pred_files:
        parsed = _parse_pred_filename(f)
        if parsed:
            entries.append((parsed[1], parsed[2], f))
    entries.sort(key=lambda x: x[0])
    selected, next_start = [], 0
    for start, end, filename in entries:
        if start >= next_start:
            selected.append(filename)
            next_start = end + 1
    return selected


def _parse_pred_filename(filename):
    """Return (doc_id, start, end) from `{doc_id}_page_{start}-{end}.md`, or None."""
    m = PRED_RE.match(filename)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _merge_gt_pages(doc_id, start, end, mds_dir):
    """Concatenate GT files for pages start..end.

    Warns if any page in the range has no GT file (non-consecutive GT).
    """
    parts = []
    missing = []
    for page_no in range(start, end + 1):
        path = os.path.join(mds_dir, f"{doc_id}_{page_no}.md")
        if not os.path.exists(path):
            missing.append(page_no)
            continue
        content = read_md_file(path)
        if content:
            parts.append(content)
    if missing:
        print(f"[WARN] Non-consecutive GT: missing pages {missing} in {start}-{end} for {doc_id}")
    return "\n\n".join(parts)


@DATASET_REGISTRY.register("multipage_md2md_dataset")
class MultipageMd2MdDataset:
    def __init__(self, cfg_task):
        self.gt_root          = cfg_task['dataset']['ground_truth']['data_path']
        self.pred_root        = cfg_task['dataset']['prediction']['data_path']
        self.match_method     = cfg_task['dataset'].get('match_method', 'quick_match')
        self.include_subjects = set(cfg_task['dataset'].get('include_subjects', []))
        self.exclude_subjects = set(cfg_task['dataset'].get('exclude_subjects', []))

        self.samples = self._get_matched_elements()

    def _get_matched_elements(self):
        plain_text_match      = []
        display_formula_match = []
        html_table_match      = []
        latex_table_match     = []
        order_match           = []

        if self.match_method == 'simple_match':
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            match_gt2pred = partial(match_gt2pred_quick, skip_truncated=True)

        # Group pred files by (subject, doc_id) to detect whole-doc GT
        doc_groups = defaultdict(list)
        for subject in sorted(os.listdir(self.pred_root)):
            subject_pred_dir = os.path.join(self.pred_root, subject)
            if not os.path.isdir(subject_pred_dir):
                continue
            if self.include_subjects and subject not in self.include_subjects:
                continue
            if subject in self.exclude_subjects:
                continue
            for f in sorted(os.listdir(subject_pred_dir)):
                if f.endswith('.md'):
                    parsed = _parse_pred_filename(f)
                    if parsed is None:
                        print(f"[SKIP] Cannot parse prediction filename: {f}")
                        continue
                    doc_groups[(subject, parsed[0])].append(f)

        per_page_pairs  = []
        whole_doc_groups = {}

        for (subject, doc_id), pred_files in doc_groups.items():
            mds_dir = os.path.join(self.gt_root, subject, doc_id, 'mds')
            if not os.path.isdir(mds_dir):
                print(f"[WARN] GT mds dir not found: {mds_dir}")
                continue
            if _has_per_page_gt(mds_dir, doc_id):
                per_page_pairs.extend((subject, f) for f in pred_files)
            elif os.path.exists(os.path.join(mds_dir, f'{doc_id}.md')):
                whole_doc_groups[(subject, doc_id)] = pred_files
            else:
                print(f"[WARN] No usable GT for {subject}/{doc_id}")

        # --- Per-page matching (existing behavior) ---
        for subject, pred_file in tqdm(per_page_pairs, desc="Matching (per-page)"):
            parsed = _parse_pred_filename(pred_file)
            doc_id, start, end = parsed

            mds_dir = os.path.join(self.gt_root, subject, doc_id, "mds")
            gt_content = _merge_gt_pages(doc_id, start, end, mds_dir)
            if not gt_content:
                print(f"[WARN] Empty merged GT for {pred_file}")
                continue

            pred_content = read_md_file(os.path.join(self.pred_root, subject, pred_file))
            img_name = pred_file.replace('.md', '.jpg')

            gt_data   = md_tex_filter(gt_content)
            pred_data = md_tex_filter(pred_content)

            display_formula_match_s = []
            plain_text_match_clean  = []

            if gt_data.get('text_all'):
                plain_text_match_s = match_gt2pred(
                    gt_data['text_all'], pred_data.get('text_all', []), 'text', img_name)
                plain_text_match_clean = plain_text_match_s
                plain_text_match.extend(plain_text_match_s)

            if gt_data.get('equation_isolated'):
                display_formula_match_s = match_gt2pred(
                    gt_data['equation_isolated'],
                    pred_data.get('equation_isolated', []), 'formula', img_name)
                display_formula_match_s = [
                    x for x in display_formula_match_s
                    if x['gt_idx'] != [''] and x.get('gt_category_type') != 'equation_inline'
                ]
                display_formula_match.extend(display_formula_match_s)

            if gt_data.get('latex_table') and pred_data.get('latex_table'):
                table_match_s = match_gt2pred(
                    gt_data['latex_table'], pred_data['latex_table'], 'latex_table', img_name)
                latex_table_match.extend(x for x in table_match_s if x['gt_idx'] != [''])
            elif gt_data.get('html_table') and pred_data.get('html_table'):
                table_match_s = match_gt2pred(
                    gt_data['html_table'], pred_data['html_table'], 'html_table', img_name)
                html_table_match.extend(x for x in table_match_s if x['gt_idx'] != [''])

            order_match_s = self._get_order_paired(plain_text_match_clean, img_name)
            if order_match_s:
                order_match.append(order_match_s)

        # --- Whole-doc matching (e.g. MusicXML subjects with a single GT file) ---
        for (subject, doc_id), pred_files in tqdm(whole_doc_groups.items(), desc="Matching (whole-doc)"):
            whole_doc_gt_path = os.path.join(self.gt_root, subject, doc_id, 'mds', f'{doc_id}.md')
            raw_gt = read_md_file(whole_doc_gt_path)
            if not raw_gt:
                print(f"[WARN] Empty whole-doc GT for {subject}/{doc_id}")
                continue

            selected_files = _select_nonoverlapping(pred_files)
            pred_parts = [
                read_md_file(os.path.join(self.pred_root, subject, f))
                for f in selected_files
            ]
            pred_parts = [p for p in pred_parts if p]
            if not pred_parts:
                print(f"[WARN] No prediction content for {subject}/{doc_id}")
                continue

            merged_pred = "\n\n".join(pred_parts)

            gt_xml   = _strip_musicxml_metadata(_extract_musicxml(raw_gt))
            pred_xml = _extract_musicxml(merged_pred)

            if not gt_xml:
                print(f"[WARN] No MusicXML content found in GT for {subject}/{doc_id}")
                continue

            img_name = f"{doc_id}_wholedoc.jpg"
            edit = Levenshtein.distance(gt_xml, pred_xml) / max(len(gt_xml), len(pred_xml), 1)
            plain_text_match.append({
                'gt_idx':             [0],
                'gt':                 gt_xml,
                'norm_gt':            gt_xml,
                'gt_category_type':   'text_block',
                'gt_position':        [0],
                'gt_attribute':       [{}],
                'pred':               pred_xml,
                'pred_position':      0,
                'pred_category_type': 'text_block',
                'img_id':             img_name,
                'edit':               edit,
            })

        table_match  = latex_table_match if latex_table_match else html_table_match
        table_format = 'latex' if latex_table_match else 'html'

        return {
            'text_block':      DATASET_REGISTRY.get('recogition_end2end_base_dataset')(plain_text_match),
            'display_formula': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(display_formula_match),
            'table':           DATASET_REGISTRY.get('recogition_end2end_table_dataset')(table_match, table_format),
            'reading_order':   DATASET_REGISTRY.get('recogition_end2end_base_dataset')(order_match),
        }

    def _get_order_paired(self, order_match_s, img_name):
        matched = [
            (item['gt_position'], item['pred_position'])
            for item in order_match_s
            if item['gt_position'] != [''] and item['pred_position'] != ''
        ]
        gt_idx_all = [item['gt_position'] for item in order_match_s if item['gt_position'] != ['']]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]
        read_order_gt   = sum(gt_idx_all, [])
        read_order_gt   = [x for x in read_order_gt if x]
        gt_sorted = sorted(read_order_gt)
        pred_flat = sum(read_order_pred, [])
        pred_flat = [x for x in pred_flat if x]
        if pred_flat or gt_sorted:
            edit = Levenshtein.distance(gt_sorted, pred_flat) / max(len(pred_flat), len(gt_sorted), 1)
            return {'gt': gt_sorted, 'pred': pred_flat, 'img_id': img_name, 'edit': edit}
        return {}
