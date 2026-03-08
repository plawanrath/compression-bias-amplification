"""Compute perplexity for all model x quantization combinations.

Uses mlx_lm's eval_ppl() to measure standard language-model quality,
establishing that aggregate metrics barely change even as item-level
biases emerge under compression.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx_lm
import numpy as np
import yaml
from mlx_lm.perplexity import eval_ppl, load_data


def compute_perplexity(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    ppl_cfg = cfg["perplexity"]

    out_dir = Path("results/perplexity")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "perplexity_results.json"
    csv_path = out_dir / "perplexity_results.csv"

    # Load existing results for resume support
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    else:
        existing = []

    done_keys = {(r["model"], r["quant"]) for r in existing}

    for model_cfg in cfg["models"]:
        for bits in cfg["quantization"]["bit_widths"]:
            if (model_cfg["name"], bits) in done_keys:
                print(f"  Skipping {model_cfg['name']} Q{bits} (already computed)")
                continue

            if bits == 16:
                model_dir = (
                    f"{cfg['quantization']['models_dir']}/"
                    f"{model_cfg['name']}-bf16"
                )
            else:
                model_dir = (
                    f"{cfg['quantization']['models_dir']}/"
                    f"{model_cfg['name']}-q{bits}"
                )

            print(f"\n{'=' * 60}")
            print(f"Computing perplexity: {model_cfg['name']} @ Q{bits}")
            print(f"Model dir: {model_dir}")

            # Set seeds for reproducibility
            np.random.seed(ppl_cfg["seed"])
            mx.random.seed(ppl_cfg["seed"])

            t0 = time.time()
            model, tokenizer = mlx_lm.load(model_dir)

            # Load evaluation data (tokenized)
            data = load_data(
                tokenizer,
                data_path=ppl_cfg["data_path"],
                num_samples=ppl_cfg["num_samples"],
                sequence_length=ppl_cfg["sequence_length"],
            )

            ppl, ppl_se = eval_ppl(
                model, data, batch_size=ppl_cfg["batch_size"]
            )
            elapsed = time.time() - t0

            result = {
                "model": model_cfg["name"],
                "quant": bits,
                "perplexity": round(float(ppl), 4),
                "ppl_se": round(float(ppl_se), 4),
                "elapsed_s": round(elapsed, 1),
            }
            existing.append(result)
            print(
                f"  PPL = {ppl:.4f} +/- {ppl_se:.4f}  ({elapsed:.1f}s)"
            )

            # Write incrementally (resume-safe)
            json_path.write_text(json.dumps(existing, indent=2))

            del model  # Free memory before next variant

    # Write final CSV
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "quant", "perplexity", "ppl_se", "elapsed_s"],
        )
        writer.writeheader()
        for r in existing:
            writer.writerow(r)

    print(f"\nSaved {len(existing)} entries to {json_path} and {csv_path}")


if __name__ == "__main__":
    compute_perplexity()
