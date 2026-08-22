# DrDocBench
A Benchmark for Expert-Level and Difficult Document Parsing

## Overview

DrDocBench is an evaluation framework for multi-page document parsing, targeting difficult and expert-level documents across diverse subject areas. It extends the [OmniDocBench](https://github.com/opendatalab/OmniDocBench) evaluation framework with a sliding-window multipage inference pipeline and subject-level granularity.

Supported evaluation dimensions:
- **Text** — Normalized Edit Distance
- **Display Formulas** — Edit Distance, CDM
- **Tables** — TEDS, Edit Distance
- **Reading Order** — Normalized Edit Distance

## Repository Structure

```
DrDocBench/
├── configs/                    # YAML configs for each eval task
│   ├── end2end.yaml
│   ├── multipage_end2end.yaml
│   ├── multipage_md2md.yaml
│   └── ...
├── dataset/                    # Dataset loaders (single-page and multipage)
├── metrics/                    # Metric implementations (Edit, TEDS, CDM, BLEU, METEOR)
├── registry/                   # Registry for tasks, datasets, and metrics
├── task/                       # Eval task classes (end2end, detection, recognition)
├── tools/
│   ├── multipage_pdf_validation.py   # Entry point for multipage evaluation
│   └── model_infer/                  # Inference scripts per model
└── utils/                      # Matching, preprocessing, and I/O utilities
```

## Environment Setup

```bash
conda create -n drdocbench python=3.10
conda activate drdocbench

# DrDocBench reuses OmniDocBench's base dependency set.
git clone https://github.com/opendatalab/OmniDocBench.git /path/to/OmniDocBench
pip install -r /path/to/OmniDocBench/requirements.txt
```

For CDM formula evaluation, set up the [CDM environment](https://github.com/opendatalab/UniMERNet/tree/main/cdm) separately. For LaTeX-format table evaluation, install [LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/).

Several single-page example configs inherit `./demo_data/...` paths from OmniDocBench. The demo data is not vendored in this repository; copy or symlink `demo_data/` from OmniDocBench if you want to run those examples. For DrDocBench multipage runs, set the config or CLI paths to your actual benchmark and prediction directories.

## Evaluation

### Multipage sliding-window evaluation

Configure `configs/multipage_end2end.yaml`, then run:

```bash
python tools/multipage_pdf_validation.py --config configs/multipage_end2end.yaml \
    --gt_root /path/to/ground_truth \
    --pred_path /path/to/predictions \
    --save_name my_model_run
```

CLI overrides (`--gt_root`, `--pred_path`, `--save_name`, `--subjects`) take precedence over values in the config file.

The default sliding window is **2 pages** (`--num_pages 2`). MUSIC transcription is evaluated separately via `multipage_md2md.yaml`.

Results are written to `./result/` as JSON files:
```
result/
├── <save_name>_metric_result.json         # Aggregated metrics
├── <save_name>_<element>_result.json      # Per-sample matched pairs
└── <save_name>_<element>_per_page_edit.json
```

### Matching methods

Configure `match_method` in the dataset section of any config:

| Method | Description |
|---|---|
| `quick_match` | Paragraph segmentation + truncation/merge via adjacency search; recommended |
| `simple_match` | Paragraph segmentation only, direct one-to-one match |
| `no_split` | Concatenate all blocks into a single markdown before scoring |

## Model Inference

Inference scripts for supported models are in `tools/model_infer/`:

| Script | Model |
|---|---|
| `claude_multipage_inf.py` | Claude (Anthropic API) |
| `gemini_multipage_inf.py` | Gemini (Google API) |
| `gpt_multipage_inf.py` | GPT-4o (OpenAI API) |
| `qwen_multipage_inf.py` | Qwen-VL |
| `kimi_multipage_inf.py` | Kimi (Moonshot API) |
| `doubao_multipage_inf.py` | Doubao (ByteDance API) |
| `nemotron_multipage_inf.py` | Nemotron |
| `mineru2.5_multipage_inf.py` | MinerU 2.5 |
| `paddleOCR_multipage_inf.py` | PaddleOCR |

All multipage inference scripts require explicit `--image_root` and `--save_root` arguments; no dataset or output path is hardcoded. Prompt-based model scripts also require `--prompt` and load that file directly.

Example:

```bash
cd tools/model_infer
python gpt_multipage_inf.py \
    --image_root /path/to/close_source_benchmark \
    --save_root /path/to/results/gpt-5.5-multipage \
    --prompt /path/to/prompt.txt \
    --model gpt-5.5 \
    --num_pages 5
```

API-backed scripts still read credentials from local credential files such as `api_key.txt` / `base_url.txt` where applicable.

## Citation

If you use DrDocBench repository in your work, please cite us and the original OmniDocBench paper:

```bibtex
@article{yangandyu2026drdocbenchcomprehensivebenchmark,
  title={Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing},
  author={Minglai Yang and Xinyan Velocity Yu and Pengyuan Li and Xinyu Guo and Zhenting Qi and Konwoo Kim and Longtian Ye and Xiaolong Luo and Jinhe Bi and Henry Zhang and Haris Riaz and Xuan Zhang and Yunze Xiao and Bangya Liu and Tom Tang and Yunfei Zhao and Qunshu Lin and Zihan Wang and Minghao Liu and Michael Lingzhi Li and Yilun Du and Jesse Thomason and Rogerio Feris and Alex Pentland and Zexue He},
  journal={arXiv preprint arXiv:2606.01393},
  year={2026},
  url={https://arxiv.org/abs/2606.01393}, 
}

@article{ouyang2024omnidocbench,
  title={OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations},
  author={Ouyang, Linke and others},
  journal={arXiv preprint arXiv:2412.07626},
  year={2024}
}
```
