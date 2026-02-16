"""Download models from HuggingFace and convert to all quantization variants."""

import os
import subprocess
import yaml


def download_and_convert(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    models_dir = cfg["quantization"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    for model_cfg in cfg["models"]:
        hf_path = model_cfg["hf_path"]
        name = model_cfg["name"]

        for bits in cfg["quantization"]["bit_widths"]:
            if bits == 16:
                out = f"{models_dir}/{name}-bf16"
                cmd = [
                    "python", "-m", "mlx_lm.convert",
                    "--hf-path", hf_path,
                    "--mlx-path", out,
                ]
            else:
                out = f"{models_dir}/{name}-q{bits}"
                cmd = [
                    "python", "-m", "mlx_lm.convert",
                    "--hf-path", hf_path,
                    "--mlx-path", out,
                    "--quantize",
                    "--q-bits", str(bits),
                    "--q-group-size",
                    str(cfg["quantization"]["group_size"]),
                ]

            if os.path.exists(out):
                print(f"Skipping {out} (already exists)")
                continue

            print(f"Converting {name} -> Q{bits}...")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    download_and_convert()
