"""Main experiment loop: run inference across all model x quantization combinations."""

import json
import time
from pathlib import Path

import mlx_lm
import yaml
from tqdm import tqdm

from src.prompt_templates import format_prompt
from src.response_parser import parse_response


def run_experiment(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    prompts = [json.loads(line) for line in open(cfg["dataset"]["output_path"])]

    print(f"Loaded {len(prompts)} prompt items")

    for model_cfg in cfg["models"]:
        for bits in cfg["quantization"]["bit_widths"]:
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
            print(f"Loading {model_cfg['name']} @ Q{bits}")
            print(f"Model dir: {model_dir}")
            model, tokenizer = mlx_lm.load(model_dir)

            out_path = (
                f"{cfg['output']['raw_results_dir']}/"
                f"{model_cfg['name']}_q{bits}.jsonl"
            )
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w") as f:
                for item in tqdm(prompts, desc=f"Q{bits}"):
                    prompt = format_prompt(item, model_cfg["name"], tokenizer)

                    for seed in cfg["inference"]["seeds"]:
                        t0 = time.time()
                        raw = mlx_lm.generate(
                            model,
                            tokenizer,
                            prompt=prompt,
                            max_tokens=cfg["inference"]["max_tokens"],
                            temp=cfg["inference"]["temperature"],
                            seed=seed,
                        )
                        elapsed = time.time() - t0

                        parsed = parse_response(raw)
                        record = {
                            "model": model_cfg["name"],
                            "quant": bits,
                            "item_id": item["item_id"],
                            "category": item["category"],
                            "seed": seed,
                            "raw_response": raw,
                            "parsed_answer": parsed,
                            "stereotype_target_index": item[
                                "stereotype_target_index"
                            ],
                            "anti_stereotype_target_index": item[
                                "anti_stereotype_target_index"
                            ],
                            "unknown_index": item["unknown_index"],
                            "elapsed_s": elapsed,
                        }
                        f.write(json.dumps(record) + "\n")

            del model  # Free memory before loading next variant
            print(f"Saved results to {out_path}")


if __name__ == "__main__":
    run_experiment()
