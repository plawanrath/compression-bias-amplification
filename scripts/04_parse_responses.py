"""Re-parse all raw results. Useful after improving the parser logic."""

import json
from pathlib import Path

import yaml

from src.response_parser import parse_response


def reparse(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    raw_dir = Path(cfg["output"]["raw_results_dir"])

    if not raw_dir.exists():
        print(f"No raw results directory found at {raw_dir}")
        return

    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {raw_dir}")
        return

    for f in files:
        records = [json.loads(line) for line in open(f)]
        reparsed = 0

        for r in records:
            new_parse = parse_response(r["raw_response"])
            if new_parse != r.get("parsed_answer"):
                reparsed += 1
            r["parsed_answer"] = new_parse

        with open(f, "w") as out:
            for r in records:
                out.write(json.dumps(r) + "\n")

        print(f"{f.name}: reparsed {reparsed}/{len(records)}")


if __name__ == "__main__":
    reparse()
