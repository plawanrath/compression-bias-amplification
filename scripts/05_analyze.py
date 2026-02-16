"""Full analysis pipeline: compute metrics, statistical tests, and generate figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from src.metrics import compute_metrics
from src.stats import chi_squared_test, logistic_regression_trend


def analyze(config_path="config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    raw_dir = Path(cfg["output"]["raw_results_dir"])

    # Load all results
    rows = []
    for f in raw_dir.glob("*.jsonl"):
        rows.extend(json.loads(line) for line in open(f))

    if not rows:
        print(f"No results found in {raw_dir}")
        return

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} records from {raw_dir}")

    # Compute aggregated metrics
    metrics = compute_metrics(df)
    agg_path = Path(cfg["output"]["aggregated_dir"])
    agg_path.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(agg_path / "metrics_summary.csv", index=False)
    print("\n=== Aggregated Metrics ===")
    print(metrics.to_string())

    # --- Figure 1: SRS vs Bit-width (main result) ---
    fig_dir = Path(cfg["output"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=metrics,
        x="quant",
        y="srs",
        hue="model",
        style="category",
        markers=True,
        ax=ax,
    )
    ax.invert_xaxis()  # Higher bits on left
    ax.set_xlabel("Quantization Bit-width")
    ax.set_ylabel("Stereotype Reliance Score (SRS)")
    ax.set_title("Bias Amplification Under Compression")
    ax.axhline(y=1 / 3, color="gray", linestyle="--", label="Random baseline")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "srs_vs_bitwidth.pdf", dpi=300)
    fig.savefig(fig_dir / "srs_vs_bitwidth.png", dpi=300)
    print(f"\nSaved SRS figure to {fig_dir}/srs_vs_bitwidth.{{pdf,png}}")

    # --- Figure 2: Unknown Selection Rate decline ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=metrics,
        x="quant",
        y="usr",
        hue="model",
        markers=True,
        ax=ax2,
    )
    ax2.invert_xaxis()
    ax2.set_xlabel("Quantization Bit-width")
    ax2.set_ylabel("Unknown Selection Rate (USR)")
    ax2.set_title("Decline in 'Unknown' Selection Under Compression")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "usr_vs_bitwidth.pdf", dpi=300)
    fig2.savefig(fig_dir / "usr_vs_bitwidth.png", dpi=300)
    print(f"Saved USR figure to {fig_dir}/usr_vs_bitwidth.{{pdf,png}}")

    # --- Statistical tests: BF16 vs Q4, BF16 vs Q3 ---
    print("\n=== Statistical Tests ===")
    for model_name in metrics["model"].unique():
        for cat in metrics["category"].unique():
            base = metrics[
                (metrics["model"] == model_name)
                & (metrics["category"] == cat)
                & (metrics["quant"] == 16)
            ]
            if base.empty:
                continue
            for target_q in [4, 3]:
                comp = metrics[
                    (metrics["model"] == model_name)
                    & (metrics["category"] == cat)
                    & (metrics["quant"] == target_q)
                ]
                if comp.empty:
                    continue
                result = chi_squared_test(
                    base.iloc[0]["srs"],
                    int(base.iloc[0]["n"]),
                    comp.iloc[0]["srs"],
                    int(comp.iloc[0]["n"]),
                )
                print(
                    f"{model_name} | {cat} | BF16->Q{target_q}: "
                    f"chi2={result['chi2']:.2f}, p={result['p']:.4f}, "
                    f"h={result['cohens_h']:.3f}"
                )

    # --- Logistic regression trend ---
    print("\n=== Logistic Regression (trend test) ===")
    valid_df = df[df["parsed_answer"].notna()]
    if len(valid_df) > 0:
        summary = logistic_regression_trend(valid_df)
        print(summary)
    else:
        print("No valid parsed answers for logistic regression.")

    plt.close("all")


if __name__ == "__main__":
    analyze()
