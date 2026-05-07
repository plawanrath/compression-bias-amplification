# Compression Bias Amplification

Does model compression amplify social biases? This repository measures how post-training quantization (from BF16 down to 3-bit) affects stereotype reliance in large language models. We evaluate three instruction-tuned models (LLaMA 3.1 8B, Mistral 7B v0.3, Phi-3.5 Mini) on ambiguous questions from the [BBQ benchmark](https://github.com/nyu-mll/BBQ) across five bias categories: Age, Gender Identity, Race/Ethnicity, Religion, and Socioeconomic Status.

## Hardware Requirements

- **Model download & quantization**: Any machine with internet access and ~50 GB disk space
- **Inference**: Apple Silicon Mac with 32+ GB unified memory (tested on Mac Studio M2 Ultra)
- **Dataset preparation & analysis**: Any machine (no GPU required)

## Setup

```bash
git clone https://github.com/<your-username>/compression-bias-amplification.git
cd compression-bias-amplification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pre-built Quantized Models (HuggingFace)

All 15 MLX-format model artifacts are grouped in a single HuggingFace Collection:

**[Quantization Undoes Alignment — IEEE Cloud Summit 2026](https://huggingface.co/collections/plawanrath/quantization-undoes-alignment-ieee-cloud-summit-2026-69fc1cdfe3d46535cc2ba579)**

You can also access each variant directly. Either way you can skip `scripts/01_download_models.py` and load these directly:

| Base | BF16 | Q8 | Q6 | Q4 | Q3 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | [bf16](https://huggingface.co/plawanrath/qwen2.5-7b-instruct-bf16-mlx-cba) | [q8](https://huggingface.co/plawanrath/qwen2.5-7b-instruct-q8-mlx-cba) | [q6](https://huggingface.co/plawanrath/qwen2.5-7b-instruct-q6-mlx-cba) | [q4](https://huggingface.co/plawanrath/qwen2.5-7b-instruct-q4-mlx-cba) | [q3](https://huggingface.co/plawanrath/qwen2.5-7b-instruct-q3-mlx-cba) |
| Mistral-7B-Instruct-v0.3 | [bf16](https://huggingface.co/plawanrath/mistral-7b-instruct-v0.3-bf16-mlx-cba) | [q8](https://huggingface.co/plawanrath/mistral-7b-instruct-v0.3-q8-mlx-cba) | [q6](https://huggingface.co/plawanrath/mistral-7b-instruct-v0.3-q6-mlx-cba) | [q4](https://huggingface.co/plawanrath/mistral-7b-instruct-v0.3-q4-mlx-cba) | [q3](https://huggingface.co/plawanrath/mistral-7b-instruct-v0.3-q3-mlx-cba) |
| Phi-3.5-mini-Instruct | [bf16](https://huggingface.co/plawanrath/phi-3.5-mini-instruct-bf16-mlx-cba) | [q8](https://huggingface.co/plawanrath/phi-3.5-mini-instruct-q8-mlx-cba) | [q6](https://huggingface.co/plawanrath/phi-3.5-mini-instruct-q6-mlx-cba) | [q4](https://huggingface.co/plawanrath/phi-3.5-mini-instruct-q4-mlx-cba) | [q3](https://huggingface.co/plawanrath/phi-3.5-mini-instruct-q3-mlx-cba) |

```python
from mlx_lm import load, generate
model, tokenizer = load("plawanrath/qwen2.5-7b-instruct-q4-mlx-cba")
print(generate(model, tokenizer, prompt="Hello", max_tokens=64))
```

These are the **exact** artifacts used in §4.3 of the paper. Quantization config: `mlx_lm.convert` with `--q-group-size 64`, affine mode. License is inherited from each base model (Qwen2.5: Apache-2.0, Mistral-v0.3: Apache-2.0, Phi-3.5-mini: MIT).

## Execution

Run the numbered scripts in order:

```bash
# 1. Download models from HuggingFace and create quantized variants
python scripts/01_download_models.py

# 2. Prepare the BBQ dataset (filter to ambiguous condition + target categories)
python scripts/02_prepare_dataset.py

# 3. Run inference across all model x quantization combinations
python scripts/03_run_inference.py

# 4. (Optional) Re-parse responses if you've updated the parser logic
python scripts/04_parse_responses.py

# 5. Compute metrics, run statistical tests, generate figures
python scripts/05_analyze.py
```

## Expected Outputs

| Path | Contents |
|------|----------|
| `data/processed/bbq_ambiguous.jsonl` | Filtered BBQ items with answer metadata |
| `results/raw/*.jsonl` | Per-generation logs (model, quant, item, seed, raw response) |
| `results/aggregated/metrics_summary.csv` | SRS, USR, parse failure rates per model x quant x category |
| `results/figures/srs_vs_bitwidth.pdf` | Main result: stereotype reliance vs. bit-width |
| `results/figures/usr_vs_bitwidth.pdf` | Unknown selection rate decline under compression |

## Key Metrics

- **SRS (Stereotype Reliance Score)**: Fraction of responses selecting the stereotypical answer
- **USR (Unknown Selection Rate)**: Fraction of responses selecting "unknown" / "can't be determined"
- **Parse Failure Rate**: Fraction of model outputs that couldn't be mapped to A/B/C

## Configuration

All experiment hyperparameters live in `config.yaml`: models, quantization levels, dataset filters, inference settings, and output paths.

## Dataset Citation

This project uses the BBQ (Bias Benchmark for QA) dataset:

> Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P.M., & Bowman, S.R. (2022). BBQ: A Hand-Built Bias Benchmark for Question Answering. *Findings of ACL 2022*.

## License

MIT
