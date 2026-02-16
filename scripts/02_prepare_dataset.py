"""Load BBQ dataset, filter to ambiguous condition and target categories, export as JSONL."""

import json
from pathlib import Path

import yaml
from datasets import load_dataset


def _extract_answer_info(example: dict) -> dict:
    """Parse BBQ answer metadata to identify stereotype/anti-stereotype/unknown indices.

    BBQ encodes answer roles in the 'answer_info' field. Each answer (ans0, ans1, ans2)
    has metadata indicating whether it refers to a specific demographic group or is the
    'unknown' option. The 'label' field indicates which answer index is correct, and
    'question_polarity' + 'context_condition' determine which answer is stereotypical.

    In the ambiguous condition:
    - The correct answer is always 'unknown'
    - 'question_polarity' tells us which of the two non-unknown answers aligns with
      the stereotype (neg polarity -> the targeted group is the stereotypical answer)
    """
    answer_choices = [example["ans0"], example["ans1"], example["ans2"]]

    # Find the unknown answer index — it's the one containing "unknown",
    # "can't be determined", "not answerable", "not enough info", "undetermined",
    # or "cannot be determined"
    unknown_idx = None
    unknown_markers = [
        "unknown", "can't be determined", "not answerable",
        "not enough info", "undetermined", "cannot be determined",
        "not known", "not enough information",
    ]
    for i, ans in enumerate(answer_choices):
        if any(marker in ans.lower() for marker in unknown_markers):
            unknown_idx = i
            break

    if unknown_idx is None:
        # Fallback: use the label (correct answer) as unknown in ambiguous condition
        unknown_idx = example["label"]

    # The other two indices are the demographic-group answers.
    # 'question_polarity' determines which one is stereotypical:
    #   - "neg" means the question asks about a negative attribute
    #   - "nonneg" means the question asks about a non-negative attribute
    # The 'target_loc' field tells us where the stereotyped group is placed
    # (ans0, ans1, ans2), but more reliably we use answer_info.
    #
    # BBQ answer_info format: {"ans0": ["text", "group_label"], ...}
    # We need to figure out which non-unknown answer is the stereotype target.

    answer_info = example.get("answer_info", {})
    non_unknown = [i for i in range(3) if i != unknown_idx]

    # Determine stereotype target from question_polarity and additional_metadata
    # In BBQ: for "neg" polarity questions, the stereotyped group is the one
    # society would negatively associate with the attribute asked about.
    # 'target_loc' directly tells us where the targeted group answer is.
    target_loc = example.get("target_loc", "")

    if target_loc == "ans0":
        stereotype_idx = 0
    elif target_loc == "ans1":
        stereotype_idx = 1
    elif target_loc == "ans2":
        stereotype_idx = 2
    else:
        # Fallback: first non-unknown index
        stereotype_idx = non_unknown[0]

    # Anti-stereotype is the remaining non-unknown answer
    anti_stereotype_idx = [i for i in non_unknown if i != stereotype_idx]
    anti_stereotype_idx = anti_stereotype_idx[0] if anti_stereotype_idx else non_unknown[-1]

    return {
        "answer_choices": answer_choices,
        "stereotype_target_index": stereotype_idx,
        "anti_stereotype_target_index": anti_stereotype_idx,
        "unknown_index": unknown_idx,
    }


def prepare_dataset(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    ds_name = cfg["dataset"]["name"]
    categories = cfg["dataset"]["bias_categories"]
    condition = cfg["dataset"]["condition"]
    output_path = Path(cfg["dataset"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {ds_name}")
    dataset = load_dataset(ds_name)

    # BBQ may come as a single split or multiple; flatten to a list
    all_examples = []
    if isinstance(dataset, dict):
        for split_name in dataset:
            all_examples.extend(dataset[split_name])
    else:
        all_examples.extend(dataset)

    print(f"Total examples loaded: {len(all_examples)}")

    # Filter to ambiguous condition and target categories
    filtered = []
    for ex in all_examples:
        if ex.get("context_condition") != condition:
            continue
        if ex.get("category") not in categories:
            continue
        filtered.append(ex)

    print(f"After filtering (condition={condition}, categories={categories}): {len(filtered)}")

    # Export as JSONL
    count = 0
    with open(output_path, "w") as f:
        for ex in filtered:
            answer_info = _extract_answer_info(ex)
            record = {
                "item_id": ex.get("example_id", count),
                "category": ex["category"],
                "context": ex["context"],
                "question": ex["question"],
                "answer_choices": answer_info["answer_choices"],
                "stereotype_target_index": answer_info["stereotype_target_index"],
                "anti_stereotype_target_index": answer_info["anti_stereotype_target_index"],
                "unknown_index": answer_info["unknown_index"],
                "question_polarity": ex.get("question_polarity", ""),
                "label": ex.get("label"),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Exported {count} items to {output_path}")


if __name__ == "__main__":
    prepare_dataset()
