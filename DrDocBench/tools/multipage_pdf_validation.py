# coding: utf-8
import os
import argparse
import sys
import io
import json
import pathlib
import time
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from registry.registry import EVAL_TASK_REGISTRY, DATASET_REGISTRY, METRIC_REGISTRY
import dataset  # registers all dataset classes including MultipageMd2MdDataset
import task
import metrics


def process_args(args):
    parser = argparse.ArgumentParser(
        description='Evaluate multi-page sliding-window predictions against ground truth.')
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/multipage_end2end.yaml')
    parser.add_argument('--gt_root', type=str, default=None,
                        help='Override ground_truth.data_path in config '
                             '(e.g. /root/close_source_benchmark or /root/open_source)')
    parser.add_argument('--pred_path', type=str, default=None,
                        help='Override prediction.data_path in config')
    parser.add_argument('--save_name', type=str, default=None,
                        help='Override the result file prefix (default: derived from pred_path parent+basename)')
    parser.add_argument('--save_matched', type=str, default=None, metavar='DIR',
                        help='Directory to save matched {gt, pred} pairs as {save_name}_matched.json. '
                             'The filename is derived from save_name so close/open source is encoded automatically. '
                             'The JSON can be fed into pdf_validation.py for recognition-style re-evaluation.')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subjects to evaluate (e.g. LAW,MEDICAL). '
                             'Maps to include_subjects in the dataset config.')
    return parser.parse_args(args)


if __name__ == '__main__':
    total_start = time.time()
    parameters = process_args(sys.argv[1:])
    config_path = parameters.config

    if isinstance(config_path, (str, pathlib.Path)):
        with io.open(os.path.abspath(config_path), 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise TypeError("Unexpected file type")

    if cfg is not None and not isinstance(cfg, (list, dict, str)):
        raise IOError(f"Invalid loaded object type: {type(cfg).__name__}")

    for task_name in cfg.keys():
        if not cfg.get(task_name):
            print(f'No config for task {task_name}')
            continue

        # Apply CLI overrides
        if parameters.gt_root:
            cfg[task_name]['dataset']['ground_truth']['data_path'] = parameters.gt_root
        if parameters.pred_path:
            cfg[task_name]['dataset']['prediction']['data_path'] = parameters.pred_path

        dataset_name  = cfg[task_name]['dataset']['dataset_name']
        metrics_list  = cfg[task_name]['metrics']
        val_dataset   = DATASET_REGISTRY.get(dataset_name)(cfg[task_name])
        val_task      = EVAL_TASK_REGISTRY.get(task_name)

        if parameters.save_name:
            save_name = parameters.save_name
        elif cfg[task_name]['dataset']['prediction'].get('data_path'):
            pred_path = cfg[task_name]['dataset']['prediction']['data_path'].rstrip('/')
            parent    = os.path.basename(os.path.dirname(pred_path))
            basename  = os.path.basename(pred_path)
            match_method = cfg[task_name]['dataset'].get('match_method', 'quick_match')
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(pred_path)))
            if grandparent:
                save_name = f"{grandparent}_{parent}_{basename}_{match_method}"
            elif parent:
                save_name = f"{parent}_{basename}_{match_method}"
            else:
                save_name = f"{basename}_{match_method}"
        else:
            save_name = os.path.basename(
                cfg[task_name]['dataset']['ground_truth']['data_path']).split('.')[0]

        print('###### Process: ', save_name)
        task_start = time.time()

        if parameters.save_matched:
            matched_out = {
                key: dataset_obj.samples
                for key, dataset_obj in val_dataset.samples.items()
            }
            os.makedirs(parameters.save_matched, exist_ok=True)
            out_path = os.path.join(parameters.save_matched, f'{save_name}_matched.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(matched_out, f, ensure_ascii=False, indent=2)
            print(f'[INFO] Matched pairs saved to: {out_path}')

        if cfg[task_name]['dataset']['ground_truth'].get('page_info'):
            val_task(val_dataset, metrics_list,
                     cfg[task_name]['dataset']['ground_truth']['page_info'], save_name)
        else:
            val_task(val_dataset, metrics_list,
                     cfg[task_name]['dataset']['ground_truth']['data_path'], save_name)

        task_elapsed = time.time() - task_start
        print(f'###### Task "{task_name}" finished in {task_elapsed:.1f}s')

    total_elapsed = time.time() - total_start
    print(f'###### Total time: {total_elapsed:.1f}s')
