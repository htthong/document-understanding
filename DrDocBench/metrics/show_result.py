from collections import defaultdict
from tabulate import tabulate
import pandas as pd
import pdb

def show_result(results):
    for metric_name in results.keys():
        print(f'{metric_name}:')
        score_table = [[k,v] for k,v in results[metric_name].items()]
        print(tabulate(score_table))
        print('='*100)

def sort_nested_dict(d):
    # If it's a dictionary, recursively sort it
    if isinstance(d, dict):
        # Sort the current dictionary
        sorted_dict = {k: sort_nested_dict(v) for k, v in sorted(d.items())}
        return sorted_dict
    # If not a dictionary, return directly
    return d

def get_full_labels_results(samples):
    if not samples:
        return {}
    label_group_dict = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        label_list = []
        if not sample.get("gt_attribute"):
            continue
        for anno in sample["gt_attribute"]:
            for k,v in anno.items():
                if k == '_source_page_no':
                    # internal tag for multipage attribution; not a block-level attr
                    continue
                label_list.append(k+": "+str(v))
        for label_name in list(set(label_list)):  # Currently if there are merged cases, calculate based on the set of all labels involved after merging
            for metric, score in sample['metric'].items():
                label_group_dict[label_name][metric].append(score)

    print('----Anno Attribute---------------')
    result = {}
    result['sample_count'] = {}
    for attribute in label_group_dict.keys():
        for metric, scores in label_group_dict[attribute].items():
            mean_score = sum(scores) / len(scores)
            if not result.get(metric):
                result[metric] = {}
            result[metric][attribute] = mean_score
            result['sample_count'][attribute] = len(scores)
    result = sort_nested_dict(result)
    show_result(result)
    return result

# def get_page_split(samples, page_info):    # Sample level metric
#     if not page_info:
#         return {}
#     page_split_dict = defaultdict(lambda: defaultdict(list)) 
#     for sample in samples:
#         img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') else '_'.join(sample['img_id'].split('_')[:-1])
#         page_info_s = page_info[img_name]
#         if not sample.get('metric'):
#             continue
#         for metric, score in sample['metric'].items():
#             for k,v in page_info_s.items():
#                 if isinstance(v, list): # special issue
#                     for special_issue in v:
#                         if 'table' not in special_issue:  # Table-related special fields have duplicates
#                             page_split_dict[metric][special_issue].append(score)
#                 else:
#                     page_split_dict[metric][k+": "+str(v)].append(score)
    
#     print('----Page Attribute---------------')
#     result = {}
#     result['sample_count'] = {}
#     for metric in page_split_dict.keys():
#         for attribute, scores in page_split_dict[metric].items():
#             mean_score = sum(scores) / len(scores)
#             if not result.get(metric):
#                 result[metric] = {}
#             result[metric][attribute] = mean_score
#             result['sample_count'][attribute] = len(scores)
#     result = sort_nested_dict(result)
#     show_result(result)
#     return result

def get_page_split(samples, page_info):   # Page level metric
    if not page_info:
        return {}

    # Detect multipage shape: {window_img_name → {page_no → page_attribute}}.
    # In that case each block carries `_source_page_no` in its gt_attribute,
    # so we look up source-page attrs per block (no need to pick a single
    # "window-level label" when pages within a window have different layouts).
    multipage_mode = any(
        isinstance(v, dict) and v and all(isinstance(k, int) for k in v.keys())
        for v in page_info.values()
    )

    result_list = defaultdict(list)
    for sample in samples:
        img_name = sample['img_id'][:-4] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
        if not sample.get('metric'):
            continue

        if multipage_mode:
            window_map = page_info.get(img_name) or {}
            if not window_map:
                continue

            # Collect distinct page-attribute (key, value) pairs across all matched
            # GTs in this sample. If a sample matches GTs from pages with different
            # layouts (e.g., one block from single_column, another from multi_column),
            # the sample contributes its score to BOTH layout buckets — same semantics
            # as get_full_labels_results uses for block-level attributes.
            seen_keys = set()
            for anno in (sample.get('gt_attribute') or []):
                page_no = anno.get('_source_page_no')
                if page_no is None:
                    continue
                page_attrs = window_map.get(page_no) or {}
                for k, v in page_attrs.items():
                    if isinstance(v, list):
                        for item in v:
                            if 'table' not in str(item):
                                seen_keys.add(('SPECIAL', item))
                    else:
                        seen_keys.add((k, v))

            for metric, score in sample['metric'].items():
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                result_list[metric].append({
                    'image_name': img_name, 'metric': metric, 'attribute': 'ALL',
                    'score': score, 'upper_len': max(len(gt), len(pred))
                })
                for kt in seen_keys:
                    label = kt[1] if kt[0] == 'SPECIAL' else f'{kt[0]}: {kt[1]}'
                    result_list[metric].append({
                        'image_name': img_name, 'metric': metric,
                        'attribute': label, 'score': score,
                        'upper_len': max(len(gt), len(pred))
                    })
            continue

        # ---- legacy single-page path (unchanged) ----
        page_info_s = page_info[img_name]
        for metric, score in sample['metric'].items():
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            result_list[metric].append({
                'image_name': img_name,
                'metric': metric,
                'attribute': 'ALL',
                'score': score,
                'upper_len': max(len(gt), len(pred))
            })
            for k,v in page_info_s.items():
                if isinstance(v, list): # special issue
                    for special_issue in v:
                        if 'table' not in special_issue:  # Table-related special fields have duplicates
                            result_list[metric].append({
                                'image_name': img_name,
                                'metric': metric,
                                'attribute': special_issue,
                                'score': score,
                                'upper_len': max(len(gt), len(pred))
                            })
                else:
                    result_list[metric].append({
                        'image_name': img_name,
                        'metric': metric,
                        'attribute': k+": "+str(v),
                        'score': score,
                        'upper_len': max(len(gt), len(pred))
                    })
    
    # Page level logic, accumulation is only done within pages, and mean operation is performed between pages
    result = {}
    if result_list.get('Edit_dist'):   # 只有Edit_dist需要进行page level的计算
        df = pd.DataFrame(result_list['Edit_dist'])
        up_total_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()).groupby('attribute').mean()  # At page level, accumulate edits, denominator is sum of max(gt, pred) from each sample
        # up_total_avg = df.groupby(["attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()) # whole_level
        result['Edit_dist'] = up_total_avg.to_dict()
    for metric in result_list.keys():
        if metric == 'Edit_dist':
            continue
        df = pd.DataFrame(result_list[metric])
        page_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: x["score"].mean()).groupby('attribute').mean() # 页面内部平均以后，再页面间的平均
        result[metric] = page_avg.to_dict()

    result = sort_nested_dict(result)
    # print('----Page Attribute---------------')
    show_result(result)
    return result