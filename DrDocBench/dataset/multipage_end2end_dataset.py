"""
Dataset class for evaluating multi-page sliding-window predictions against
per-page JSON ground truth (the rich annotated format used by end2end_dataset).

GT layout: gt_root/{subject}/{doc_id}/json/{doc_id}_page_{N}.json
Pred layout: pred_root/{subject}/{doc_id}_page_{start}-{end}.md

For each prediction window [start, end]:
  - The per-page JSONs are loaded and their layout_dets are merged.
  - anno_ids are made globally unique across pages.
  - order values are offset so elements sort correctly across pages.
  - Intra-page truncation relations are remapped and preserved.

The matched elements are then fed into the same end2end eval pipeline
as End2EndDataset, giving per-category scores (text, formula, table,
reading order) with proper GT labels — not heuristic markdown parsing.
"""

import json
import os
import re
import time
import traceback
from functools import partial

from tqdm import tqdm
from pylatexenc.latex2text import LatexNodes2Text

from registry.registry import DATASET_REGISTRY
from dataset.end2end_dataset import End2EndDataset
from utils.extract import md_tex_filter
from utils.match import match_gt2pred_simple, match_gt2pred_no_split
from utils.match_quick import match_gt2pred_quick
from utils.read_files import read_md_file
from utils.data_preprocess import clean_string

PRED_RE = re.compile(r'^(.+)_page_(\d+)-(\d+)\.md$')


@DATASET_REGISTRY.register("multipage_end2end_dataset")
class MultipageEnd2EndDataset(End2EndDataset):
    """Like End2EndDataset but GT comes from per-page JSONs merged per window."""

    def __init__(self, cfg_task):
        self.gt_root          = cfg_task['dataset']['ground_truth']['data_path']
        self.pred_root        = cfg_task['dataset']['prediction']['data_path']
        self.match_method     = cfg_task['dataset'].get('match_method', 'quick_match')
        self.include_subjects = set(cfg_task['dataset'].get('include_subjects', []))
        self.exclude_subjects = set(cfg_task['dataset'].get('exclude_subjects', []))
        self.samples = self._get_matched_elements_multipage()

    # ------------------------------------------------------------------ helpers

    def _load_page_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return raw[0] if raw else None

    def _merge_window_json(self, doc_id, start, end, json_dir):
        """Merge per-page JSONs for pages [start..end] into one combined sample.

        Returns a dict with 'layout_dets' and 'extra' (same shape as a single
        page JSON entry), suitable for passing directly to get_page_elements().
        Returns None if no pages were found.
        """
        all_dets = []
        all_relations = []
        page_info_map = {}   # page_no → page_attribute (for filling 'page' aggregation later)
        missing = []

        for page_no in range(start, end + 1):
            path = os.path.join(json_dir, f"{doc_id}_page_{page_no}.json")
            if not os.path.exists(path):
                missing.append(page_no)
                continue

            page_data = self._load_page_json(path)
            if page_data is None or not page_data.get('layout_dets'):
                missing.append(page_no)
                continue

            # Capture this page's attributes for the window-level page_info_map
            page_attr = page_data.get('page_info', {}).get('page_attribute', {})
            if page_attr:
                page_info_map[page_no] = page_attr

            # Make anno_ids globally unique by embedding page_no
            id_map = {
                det['anno_id']: page_no * 100000 + det['anno_id']
                for det in page_data['layout_dets']
            }

            # Shift order values so pages sort correctly: page 0 → 0..9999, page 1 → 10000..19999
            page_order_offset = (page_no - start) * 10000

            for det in page_data['layout_dets']:
                new_det = dict(det)
                new_det['anno_id'] = id_map[det['anno_id']]
                if new_det.get('order') is not None:
                    new_det['order'] = page_order_offset + new_det['order']

                # Tag the block with its source page (used by get_page_split to look up
                # this page's attributes from page_info_map, so we never need to "pick
                # a window-level label" for layouts that vary across pages).
                new_attr = dict(new_det.get('attribute') or {})
                new_attr['_source_page_no'] = page_no
                new_det['attribute'] = new_attr

                all_dets.append(new_det)

            for rel in page_data.get('extra', {}).get('relation', []):
                src, tgt = rel['source_anno_id'], rel['target_anno_id']
                if src in id_map and tgt in id_map:
                    all_relations.append({
                        'source_anno_id': id_map[src],
                        'target_anno_id': id_map[tgt],
                        'relation_type': rel['relation_type'],
                    })

        if missing:
            print(f"[WARN] Non-consecutive GT: missing pages {missing} for {doc_id} [{start}-{end}]")

        if not all_dets:
            return None

        return {
            'layout_dets': all_dets,
            'extra': {'relation': all_relations},
            'page_info_map': page_info_map,
        }

    # ------------------------------------------------------------------ main loop

    def _get_matched_elements_multipage(self):
        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []
        save_time = time.time()

        # Maps {window_img_name → {page_no → page_attribute}}.
        # End2EndEval reads this off the dataset to fill the page-level
        # aggregation in metric_result.json without needing an external
        # aggregated page_info JSON file.
        self.page_info_aggregated = {}

        pred_pairs = []
        for subject in sorted(os.listdir(self.pred_root)):
            subject_dir = os.path.join(self.pred_root, subject)
            if not os.path.isdir(subject_dir):
                continue
            if self.include_subjects and subject not in self.include_subjects:
                continue
            if subject in self.exclude_subjects:
                continue
            for f in sorted(os.listdir(subject_dir)):
                if f.endswith('.md'):
                    pred_pairs.append((subject, f))

        for subject, pred_file in tqdm(pred_pairs, desc="Matching"):
            m = PRED_RE.match(pred_file)
            if not m:
                print(f"[SKIP] Cannot parse prediction filename: {pred_file}")
                continue
            doc_id, start, end = m.group(1), int(m.group(2)), int(m.group(3))

            json_dir = os.path.join(self.gt_root, subject, doc_id, 'json')
            if not os.path.isdir(json_dir):
                print(f"[WARN] No json dir: {json_dir} — skipping {pred_file}")
                continue

            merged = self._merge_window_json(doc_id, start, end, json_dir)
            if merged is None:
                print(f"[WARN] Empty merged GT for {pred_file}")
                continue

            pred_content = read_md_file(os.path.join(self.pred_root, subject, pred_file))
            img_name = pred_file.replace('.md', '.jpg')

            # Stash this window's page→attr map keyed by the same img_name stem
            # that get_page_split will derive from sample['img_id'] later.
            self.page_info_aggregated[img_name[:-4]] = merged.pop('page_info_map', {}) or {}

            result = self.process_get_matched_elements(merged, pred_content, img_name, save_time)
            plain_text_clean, formula_match, latex_tbl, html_tbl, order_single = result

            if order_single:
                order_match.append(order_single)
            if plain_text_clean:
                plain_text_match.extend(plain_text_clean)
            if formula_match:
                display_formula_match.extend(formula_match)
            if latex_tbl:
                latex_table_match.extend(latex_tbl)
            if html_tbl:
                html_table_match.extend(html_tbl)

        # Post-processing identical to End2EndDataset.get_matched_elements
        display_formula_match_clean, display_formula_match_others = [], []
        for item in display_formula_match:
            pred_category_type = item.get('pred_category_type', None)
            if pred_category_type not in ['equation_inline', 'equation_isolated', '']:
                gt = item.get('gt', None)
                try:
                    item['gt'] = LatexNodes2Text().latex_to_text(gt)
                except ValueError:
                    item['gt'] = gt
                item['norm_gt'] = clean_string(item['gt'])
                display_formula_match_others.append(item)
            else:
                display_formula_match_clean.append(item)
        display_formula_match = display_formula_match_clean
        if display_formula_match_others and plain_text_match:
            plain_text_match.extend(display_formula_match_others)

        if latex_table_match:
            latex_to_html = []
            for latex_table in latex_table_match:
                for k in list(latex_table.keys()):
                    if 'pred' in k:
                        latex_table[k] = ""
                latex_table['edit'] = 1
                latex_to_html.append(latex_table)
            html_table_match.extend(latex_to_html)

        if len(latex_table_match) > len(html_table_match):
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'

        return {
            'text_block':      DATASET_REGISTRY.get('recogition_end2end_base_dataset')(plain_text_match),
            'display_formula': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(display_formula_match),
            'table':           DATASET_REGISTRY.get('recogition_end2end_table_dataset')(table_match, table_format),
            'reading_order':   DATASET_REGISTRY.get('recogition_end2end_base_dataset')(order_match),
        }

    def process_get_matched_elements(self, sample, pred_content, img_name, save_time):
        """Override to remove the 30s timeout — multipage windows are too large for it."""
        if self.match_method == 'simple_match':
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            match_gt2pred = partial(match_gt2pred_quick, skip_truncated=True)

        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)

        pred_dataset_mix = []
        for category in pred_dataset:
            if category not in ['html_table', 'latex_table', 'md2html_table']:
                pred_dataset_mix.extend(pred_dataset[category])

        gt_mix = self.get_page_elements_list(gt_page_elements, [
            'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference',
            'equation_caption', 'figure_caption', 'figure_footnote', 'table_caption',
            'table_footnote', 'code_algorithm', 'code_algorithm_caption',
            'header', 'footer', 'page_footnote', 'page_number', 'equation_isolated',
        ])
        if gt_mix:
            gt_mix = self.get_sorted_text_list(gt_mix)

        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []
        unmatch_table_pred = None

        if gt_page_elements.get('table'):
            gt_table = self.get_sorted_text_list(gt_page_elements['table'])
            latex_len = len(pred_dataset['latex_table']) if pred_dataset.get('latex_table') else 0
            html_len = len(pred_dataset['html_table']) if pred_dataset.get('html_table') else 0
            if latex_len == html_len == 0:
                html_table_match_s, unmatch_table_pred = match_gt2pred_simple(
                    gt_table, [], 'html_table', img_name)
            elif latex_len > html_len:
                latex_table_match_s, unmatch_table_pred = match_gt2pred_simple(
                    gt_table, pred_dataset['latex_table'], 'latex_table', img_name)
            else:
                html_table_match_s, unmatch_table_pred = match_gt2pred_simple(
                    gt_table, pred_dataset['html_table'], 'html_table', img_name)
            html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]
            latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != [""]]
            if unmatch_table_pred:
                pred_dataset_mix.extend(unmatch_table_pred)

        try:
            match = match_gt2pred(gt_mix, pred_dataset_mix, 'text_all', img_name)
            if isinstance(match, tuple):
                match = match[0]
        except Exception:
            print(traceback.format_exc())
            match, _ = match_gt2pred_simple(gt_mix, pred_dataset_mix, 'text_all', img_name)

        plain_text_match_s = []
        for item in match:
            gt_category = item.get('gt_category_type', None)
            if gt_category in [
                'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference',
                'equation_caption', 'figure_caption', 'figure_footnote', 'table_caption',
                'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                'header', 'footer', 'page_footnote', 'page_number',
            ]:
                plain_text_match_s.append(item)
            elif gt_category == 'equation_isolated':
                display_formula_match_s.append(item)

        display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]

        if plain_text_match_s:
            plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, [
                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote',
                'code_algorithm', 'code_algorithm_caption', 'header', 'footer',
                'page_footnote', 'page_number', 'equation_caption',
            ])

        if plain_text_match_clean:
            order_match_single = self.get_order_paired(plain_text_match_clean, img_name)
            if order_match_single:
                # Only carry _source_page_no — not block-level attrs like
                # text_language — so get_page_split can resolve page-level
                # attributes (data_source, layout, etc.) without polluting
                # the reading_order group breakdown.
                order_match_single['gt_attribute'] = [
                    {'_source_page_no': item['gt_attribute'][0]['_source_page_no']}
                    for item in plain_text_match_clean
                    if item.get('gt_attribute')
                    and item['gt_attribute'][0].get('_source_page_no') is not None
                ]

        return [plain_text_match_clean, display_formula_match_s,
                latex_table_match_s, html_table_match_s, order_match_single]
