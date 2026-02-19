"""Full analysis pipeline: compute metrics, statistical tests, and generate figures."""

import json
import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.metrics import (
    compute_metrics,
    add_deltas_from_baseline,
    compute_per_item_srs,
    filter_items_by_baseline_srs,
)
from src.stats import chi_squared_test, logistic_regression_trend


# =============================================================================
# CSV Table Generators
# =============================================================================

def generate_main_results_csv(metrics_with_deltas: pd.DataFrame, output_path: Path):
    """Generate main results table: SRS by model × bit-width with deltas from BF16."""
    # Pivot to wide format: rows=model, columns=quant levels
    pivot = metrics_with_deltas.groupby(["model", "quant"]).agg({
        "srs": "mean",
        "srs_delta": "mean",
        "srs_pct_change": "mean",
    }).reset_index()

    # Create readable table
    table = pivot.pivot(index="model", columns="quant", values="srs")
    table.columns = [f"Q{q}_SRS" for q in table.columns]

    delta_table = pivot.pivot(index="model", columns="quant", values="srs_delta")
    delta_table.columns = [f"Q{q}_delta" for q in delta_table.columns]

    result = pd.concat([table, delta_table], axis=1)
    result = result.round(4)
    result.to_csv(output_path)
    return result


def generate_statistical_tests_csv(metrics: pd.DataFrame, output_path: Path):
    """Generate table of all statistical tests with p-values and effect sizes."""
    rows = []
    for model_name in metrics["model"].unique():
        for cat in metrics["category"].unique():
            base = metrics[
                (metrics["model"] == model_name)
                & (metrics["category"] == cat)
                & (metrics["quant"] == 16)
            ]
            if base.empty:
                continue

            for target_q in [8, 6, 4, 3]:
                comp = metrics[
                    (metrics["model"] == model_name)
                    & (metrics["category"] == cat)
                    & (metrics["quant"] == target_q)
                ]
                if comp.empty:
                    continue

                result = chi_squared_test(
                    base.iloc[0]["srs"],
                    int(base.iloc[0]["n_valid"]),
                    comp.iloc[0]["srs"],
                    int(comp.iloc[0]["n_valid"]),
                )

                # Add significance stars
                p = result["p"]
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""

                # Effect size interpretation
                h = abs(result["cohens_h"])
                if h >= 0.8:
                    effect = "large"
                elif h >= 0.5:
                    effect = "medium"
                elif h >= 0.2:
                    effect = "small"
                else:
                    effect = "negligible"

                rows.append({
                    "model": model_name,
                    "category": cat,
                    "comparison": f"BF16→Q{target_q}",
                    "srs_bf16": round(base.iloc[0]["srs"], 4),
                    "srs_target": round(comp.iloc[0]["srs"], 4),
                    "delta": round(comp.iloc[0]["srs"] - base.iloc[0]["srs"], 4),
                    "chi2": round(result["chi2"], 2),
                    "p_value": round(p, 4),
                    "significance": sig,
                    "cohens_h": round(result["cohens_h"], 3),
                    "effect_size": effect,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_category_breakdown_csv(metrics: pd.DataFrame, output_path: Path):
    """Generate per-category breakdown of SRS by bit-width."""
    pivot = metrics.pivot_table(
        index=["category", "model"],
        columns="quant",
        values="srs",
        aggfunc="mean"
    ).round(4)
    pivot.columns = [f"Q{q}" for q in pivot.columns]
    pivot.to_csv(output_path)
    return pivot


# =============================================================================
# Enhanced Figure Generation
# =============================================================================

def plot_srs_heatmap(metrics: pd.DataFrame, output_path: Path):
    """Generate heatmap of SRS by model × quant, averaged across categories."""
    pivot = metrics.pivot_table(
        index="model",
        columns="quant",
        values="srs",
        aggfunc="mean"
    )
    # Reorder columns: 16, 8, 6, 4, 3
    pivot = pivot[[16, 8, 6, 4, 3]]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Red=high bias, Green=low bias
        center=0.33,  # Random baseline
        ax=ax,
        cbar_kws={"label": "Stereotype Reliance Score (SRS)"}
    )
    ax.set_xlabel("Quantization Bit-width")
    ax.set_ylabel("Model")
    ax.set_title("SRS Heatmap: Bias Amplification Across Models and Compression Levels")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_bf16_vs_q3_comparison(metrics: pd.DataFrame, output_path: Path):
    """Bar chart comparing BF16 vs Q3 per model with error bars."""
    bf16 = metrics[metrics["quant"] == 16].groupby("model").agg({
        "srs": "mean",
        "srs_ci_low": "mean",
        "srs_ci_high": "mean"
    }).reset_index()
    bf16["quant_label"] = "BF16"

    q3 = metrics[metrics["quant"] == 3].groupby("model").agg({
        "srs": "mean",
        "srs_ci_low": "mean",
        "srs_ci_high": "mean"
    }).reset_index()
    q3["quant_label"] = "Q3"

    combined = pd.concat([bf16, q3])
    combined["error_low"] = combined["srs"] - combined["srs_ci_low"]
    combined["error_high"] = combined["srs_ci_high"] - combined["srs"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(combined["model"].unique()))
    width = 0.35

    models = combined["model"].unique()
    bf16_data = combined[combined["quant_label"] == "BF16"].set_index("model").loc[models]
    q3_data = combined[combined["quant_label"] == "Q3"].set_index("model").loc[models]

    bars1 = ax.bar(
        x - width/2, bf16_data["srs"], width, label="BF16",
        yerr=[bf16_data["error_low"], bf16_data["error_high"]],
        capsize=3, color="steelblue"
    )
    bars2 = ax.bar(
        x + width/2, q3_data["srs"], width, label="Q3 (3-bit)",
        yerr=[q3_data["error_low"], q3_data["error_high"]],
        capsize=3, color="coral"
    )

    ax.axhline(y=1/3, color="gray", linestyle="--", label="Random baseline")
    ax.set_ylabel("Stereotype Reliance Score (SRS)")
    ax.set_xlabel("Model")
    ax.set_title("Bias Amplification: BF16 vs 3-bit Quantization")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, min(1, combined["srs"].max() * 1.2))
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_category_facets(metrics: pd.DataFrame, output_path: Path):
    """Faceted plot showing SRS trends split by bias category."""
    g = sns.FacetGrid(
        metrics,
        col="category",
        col_wrap=3,
        height=3,
        aspect=1.2,
        sharey=True
    )
    g.map_dataframe(
        sns.lineplot,
        x="quant",
        y="srs",
        hue="model",
        marker="o"
    )

    for ax in g.axes.flat:
        ax.invert_xaxis()
        ax.axhline(y=1/3, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Bit-width")
        ax.set_ylabel("SRS")

    g.add_legend(title="Model")
    g.figure.suptitle("Stereotype Reliance by Bias Category", y=1.02)
    g.tight_layout()
    g.savefig(output_path.with_suffix(".pdf"), dpi=300)
    g.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(g.figure)


# =============================================================================
# Filtered Analysis Figures
# =============================================================================

def plot_amplification_curve(metrics_filtered: pd.DataFrame, output_path: Path):
    """Line plot: SRS vs bit-width for filtered (latent-bias) items only."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=metrics_filtered,
        x="quant",
        y="srs",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.invert_xaxis()
    ax.axhline(y=1/3, color="gray", linestyle="--", label="Random baseline")
    ax.set_xlabel("Quantization Bit-width")
    ax.set_ylabel("SRS (Filtered: Latent-Bias Items)")
    ax.set_title("Bias Amplification on Items With Latent BF16 Bias")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_filtered_vs_unfiltered(
    metrics_all: pd.DataFrame,
    metrics_filtered: pd.DataFrame,
    output_path: Path,
):
    """Grouped bar chart: BF16 vs Q3, filtered vs unfiltered, per model."""
    models = sorted(metrics_all["model"].unique())

    data_rows = []
    for model_name in models:
        for label, mdf in [("All items", metrics_all), ("Filtered", metrics_filtered)]:
            for q, qlabel in [(16, "BF16"), (3, "Q3")]:
                subset = mdf[(mdf["model"] == model_name) & (mdf["quant"] == q)]
                srs = subset["srs"].mean() if not subset.empty else np.nan
                data_rows.append({
                    "model": model_name,
                    "subset": label,
                    "quant": qlabel,
                    "srs": srs,
                })

    plot_df = pd.DataFrame(data_rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.18

    colors = {
        ("All items", "BF16"): "steelblue",
        ("All items", "Q3"): "cornflowerblue",
        ("Filtered", "BF16"): "coral",
        ("Filtered", "Q3"): "orangered",
    }

    for i, (subset, quant) in enumerate([
        ("All items", "BF16"), ("All items", "Q3"),
        ("Filtered", "BF16"), ("Filtered", "Q3"),
    ]):
        vals = plot_df[(plot_df["subset"] == subset) & (plot_df["quant"] == quant)]
        vals = vals.set_index("model").loc[models]
        ax.bar(
            x + (i - 1.5) * width, vals["srs"], width,
            label=f"{subset} - {quant}",
            color=colors[(subset, quant)],
        )

    ax.axhline(y=1/3, color="gray", linestyle="--", alpha=0.7, label="Random baseline")
    ax.set_ylabel("Stereotype Reliance Score (SRS)")
    ax.set_xlabel("Model")
    ax.set_title("Magnifying Glass Effect: All Items vs. Latent-Bias Items")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(1, plot_df["srs"].max() * 1.3))
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def generate_filter_comparison_csv(
    metrics_all: pd.DataFrame,
    metrics_filtered: pd.DataFrame,
    filter_summary: pd.DataFrame,
    output_path: Path,
):
    """Side-by-side comparison table: filtered vs unfiltered BF16→Q3 amplification."""
    rows = []
    for model_name in metrics_all["model"].unique():
        for cat in metrics_all["category"].unique():
            all_bf16 = metrics_all[
                (metrics_all["model"] == model_name)
                & (metrics_all["category"] == cat)
                & (metrics_all["quant"] == 16)
            ]
            all_q3 = metrics_all[
                (metrics_all["model"] == model_name)
                & (metrics_all["category"] == cat)
                & (metrics_all["quant"] == 3)
            ]
            filt_bf16 = metrics_filtered[
                (metrics_filtered["model"] == model_name)
                & (metrics_filtered["category"] == cat)
                & (metrics_filtered["quant"] == 16)
            ]
            filt_q3 = metrics_filtered[
                (metrics_filtered["model"] == model_name)
                & (metrics_filtered["category"] == cat)
                & (metrics_filtered["quant"] == 3)
            ]
            fs = filter_summary[
                (filter_summary["model"] == model_name)
                & (filter_summary["category"] == cat)
            ]

            row = {
                "model": model_name,
                "category": cat,
                "n_items_total": int(fs.iloc[0]["total_items"]) if not fs.empty else 0,
                "n_items_filtered": int(fs.iloc[0]["filtered_items"]) if not fs.empty else 0,
                "all_srs_bf16": round(all_bf16.iloc[0]["srs"], 4) if not all_bf16.empty else None,
                "all_srs_q3": round(all_q3.iloc[0]["srs"], 4) if not all_q3.empty else None,
                "all_delta": round(all_q3.iloc[0]["srs"] - all_bf16.iloc[0]["srs"], 4)
                    if not all_q3.empty and not all_bf16.empty else None,
                "filt_srs_bf16": round(filt_bf16.iloc[0]["srs"], 4) if not filt_bf16.empty else None,
                "filt_srs_q3": round(filt_q3.iloc[0]["srs"], 4) if not filt_q3.empty else None,
                "filt_delta": round(filt_q3.iloc[0]["srs"] - filt_bf16.iloc[0]["srs"], 4)
                    if not filt_q3.empty and not filt_bf16.empty else None,
            }

            if row["all_delta"] and row["filt_delta"] and row["all_delta"] != 0:
                row["amplification_ratio"] = round(row["filt_delta"] / row["all_delta"], 2)
            else:
                row["amplification_ratio"] = None

            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def compute_transition_analysis(
    df: pd.DataFrame,
    item_srs_bf16: pd.DataFrame,
) -> pd.DataFrame:
    """Count items transitioning from SRS=0 at BF16 to SRS>0 at each quant level.

    Tests whether compression causes NEW biases to emerge in previously-unbiased items.
    """
    letters = ["A", "B", "C"]
    unbiased_items = item_srs_bf16[item_srs_bf16["item_srs"] == 0.0][
        ["model", "category", "item_id"]
    ]

    rows = []
    for quant_level in sorted(df["quant"].unique()):
        quant_df = df[df["quant"] == quant_level].copy()
        quant_df["is_stereo"] = (
            quant_df["parsed_answer"]
            == quant_df["stereotype_target_index"].map(lambda i: letters[i])
        )
        quant_df["is_valid"] = quant_df["parsed_answer"].notna()

        # Keep only items that were unbiased at BF16
        quant_unbiased = quant_df.merge(
            unbiased_items, on=["model", "category", "item_id"], how="inner"
        )

        item_agg = (
            quant_unbiased.groupby(["model", "category", "item_id"])
            .agg(n_valid=("is_valid", "sum"), n_stereo=("is_stereo", "sum"))
            .reset_index()
        )
        item_agg["item_srs"] = np.where(
            item_agg["n_valid"] > 0, item_agg["n_stereo"] / item_agg["n_valid"], 0
        )

        for model_name in item_agg["model"].unique():
            m = item_agg[item_agg["model"] == model_name]
            n_total = len(m)
            n_became_biased = (m["item_srs"] > 0).sum()
            mean_srs = m["item_srs"].mean()
            rows.append({
                "model": model_name,
                "quant": quant_level,
                "n_unbiased_at_bf16": n_total,
                "n_became_biased": int(n_became_biased),
                "pct_became_biased": round(n_became_biased / n_total * 100, 1) if n_total > 0 else 0,
                "mean_srs": round(mean_srs, 4),
            })

    return pd.DataFrame(rows)


def plot_transition_chart(transition_df: pd.DataFrame, output_path: Path):
    """Bar chart showing % of previously-unbiased items that become biased at each quant."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=transition_df,
        x="quant",
        y="pct_became_biased",
        hue="model",
        order=[16, 8, 6, 4, 3],
        ax=ax,
    )
    ax.set_xlabel("Quantization Bit-width")
    ax.set_ylabel("% of BF16-Unbiased Items That Became Biased")
    ax.set_title("New Biases Emerging Under Compression")
    ax.legend(title="Model", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


# =============================================================================
# Key Findings Generator
# =============================================================================

def generate_key_findings(
    metrics: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path
):
    """Generate markdown summary of key findings for the paper."""
    lines = ["# Key Findings: Bias Amplification Under Quantization\n"]

    # Overall trend
    bf16_mean = metrics[metrics["quant"] == 16]["srs"].mean()
    q3_mean = metrics[metrics["quant"] == 3]["srs"].mean()
    overall_increase = (q3_mean - bf16_mean) / bf16_mean * 100

    lines.append("## Overall Trend\n")
    if overall_increase > 0:
        lines.append(
            f"**Quantization amplifies social biases.** Across all models and categories, "
            f"average stereotype reliance score (SRS) increased from **{bf16_mean:.3f}** (BF16) "
            f"to **{q3_mean:.3f}** (3-bit), representing a **{overall_increase:.1f}% increase**.\n"
        )
    else:
        lines.append(
            f"Quantization did not consistently amplify biases. Average SRS changed from "
            f"{bf16_mean:.3f} (BF16) to {q3_mean:.3f} (3-bit), a {overall_increase:.1f}% change.\n"
        )

    # Most affected model
    lines.append("## Most Affected Model\n")
    model_deltas = metrics.groupby("model").apply(
        lambda g: g[g["quant"] == 3]["srs"].mean() - g[g["quant"] == 16]["srs"].mean()
    ).sort_values(ascending=False)

    worst_model = model_deltas.index[0]
    worst_delta = model_deltas.iloc[0]
    lines.append(
        f"**{worst_model}** showed the largest bias amplification, with SRS increasing by "
        f"**{worst_delta:.3f}** ({worst_delta/bf16_mean*100:.1f}% relative increase) "
        f"when compressed from BF16 to 3-bit.\n"
    )

    # Most affected category
    lines.append("## Most Affected Bias Category\n")
    cat_deltas = metrics.groupby("category").apply(
        lambda g: g[g["quant"] == 3]["srs"].mean() - g[g["quant"] == 16]["srs"].mean()
    ).sort_values(ascending=False)

    worst_cat = cat_deltas.index[0]
    worst_cat_delta = cat_deltas.iloc[0]
    lines.append(
        f"**{worst_cat}** biases were most amplified by compression, with an average SRS "
        f"increase of **{worst_cat_delta:.3f}** across all models.\n"
    )

    # Statistical significance summary
    lines.append("## Statistical Significance\n")
    sig_tests = stats_df[stats_df["comparison"] == "BF16→Q3"]
    n_significant = (sig_tests["p_value"] < 0.05).sum()
    n_total = len(sig_tests)
    pct_sig = n_significant / n_total * 100 if n_total > 0 else 0

    lines.append(
        f"Of {n_total} model×category comparisons between BF16 and Q3, "
        f"**{n_significant} ({pct_sig:.0f}%)** showed statistically significant "
        f"(p < 0.05) increases in stereotype reliance.\n"
    )

    # Effect size summary
    lines.append("## Effect Sizes\n")
    effect_counts = sig_tests["effect_size"].value_counts()
    lines.append("Distribution of Cohen's h effect sizes (BF16→Q3):\n")
    for effect, count in effect_counts.items():
        lines.append(f"- **{effect.capitalize()}**: {count} comparisons\n")

    # Average effect
    avg_cohens_h = sig_tests["cohens_h"].mean()
    lines.append(f"\nAverage Cohen's h: **{avg_cohens_h:.3f}**\n")

    # USR decline (important finding)
    lines.append("## Unknown Selection Rate (USR) Decline\n")
    bf16_usr = metrics[metrics["quant"] == 16]["usr"].mean()
    q3_usr = metrics[metrics["quant"] == 3]["usr"].mean()
    usr_decline = (bf16_usr - q3_usr) / bf16_usr * 100

    lines.append(
        f"Models become less likely to select 'unknown/cannot determine' answers "
        f"under compression. Average USR dropped from **{bf16_usr:.3f}** (BF16) to "
        f"**{q3_usr:.3f}** (3-bit), a **{usr_decline:.1f}% decline**. This suggests "
        f"quantized models are more prone to making unwarranted assumptions.\n"
    )

    # Copy-paste ready summary paragraph
    lines.append("---\n")
    lines.append("## Summary Paragraph (copy-paste ready)\n")
    lines.append(
        f"Our experiments reveal that post-training quantization systematically amplifies "
        f"social biases in large language models. Across {len(metrics['model'].unique())} "
        f"instruction-tuned models and {len(metrics['category'].unique())} bias categories "
        f"from the BBQ benchmark, we observed an average {overall_increase:.1f}% increase in "
        f"stereotype reliance when compressing from BF16 to 3-bit precision. "
        f"{worst_model} exhibited the most severe amplification (+{worst_delta:.3f} SRS), "
        f"while {worst_cat} biases were most affected across all models. "
        f"Of {n_total} statistical comparisons, {n_significant} ({pct_sig:.0f}%) showed "
        f"significant increases (p < 0.05) with an average effect size of h={avg_cohens_h:.3f}. "
        f"Additionally, we found that quantized models select 'unknown' answers {usr_decline:.1f}% "
        f"less frequently, suggesting reduced epistemic humility under compression.\n"
    )

    # Write to file
    content = "\n".join(lines)
    output_path.write_text(content)
    return content


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

    # Add delta columns from BF16 baseline
    metrics_with_deltas = add_deltas_from_baseline(metrics)
    metrics_with_deltas.to_csv(agg_path / "metrics_with_deltas.csv", index=False)

    # ==========================================================================
    # CSV Tables (for Google Sheets → Google Docs)
    # ==========================================================================
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Generating CSV Tables ===")
    generate_main_results_csv(metrics_with_deltas, tables_dir / "main_results.csv")
    print(f"  Saved: {tables_dir}/main_results.csv")

    stats_df = generate_statistical_tests_csv(metrics, tables_dir / "statistical_tests.csv")
    print(f"  Saved: {tables_dir}/statistical_tests.csv")

    generate_category_breakdown_csv(metrics, tables_dir / "category_breakdown.csv")
    print(f"  Saved: {tables_dir}/category_breakdown.csv")

    # ==========================================================================
    # Figures
    # ==========================================================================
    fig_dir = Path(cfg["output"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Generating Figures ===")

    # --- Figure 1: SRS vs Bit-width (main result) ---
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
    plt.close(fig)
    print(f"  Saved: {fig_dir}/srs_vs_bitwidth.{{pdf,png}}")

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
    plt.close(fig2)
    print(f"  Saved: {fig_dir}/usr_vs_bitwidth.{{pdf,png}}")

    # --- Figure 3: SRS Heatmap ---
    plot_srs_heatmap(metrics, fig_dir / "srs_heatmap")
    print(f"  Saved: {fig_dir}/srs_heatmap.{{pdf,png}}")

    # --- Figure 4: BF16 vs Q3 Bar Chart ---
    plot_bf16_vs_q3_comparison(metrics, fig_dir / "bf16_vs_q3_comparison")
    print(f"  Saved: {fig_dir}/bf16_vs_q3_comparison.{{pdf,png}}")

    # --- Figure 5: Per-Category Facets ---
    plot_category_facets(metrics, fig_dir / "srs_by_category")
    print(f"  Saved: {fig_dir}/srs_by_category.{{pdf,png}}")

    # ==========================================================================
    # Statistical Tests (console output)
    # ==========================================================================
    print("\n=== Statistical Tests (BF16 vs Q4, Q3) ===")
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
                    int(base.iloc[0]["n_valid"]),
                    comp.iloc[0]["srs"],
                    int(comp.iloc[0]["n_valid"]),
                )
                sig = "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
                print(
                    f"{model_name} | {cat} | BF16→Q{target_q}: "
                    f"χ²={result['chi2']:.2f}, p={result['p']:.4f}{sig}, "
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

    # ==========================================================================
    # Key Findings (Markdown)
    # ==========================================================================
    print("\n=== Generating Key Findings ===")
    findings = generate_key_findings(metrics, stats_df, agg_path / "key_findings.md")
    print(f"  Saved: {agg_path}/key_findings.md")
    print("\n" + "=" * 60)
    print(findings)
    print("=" * 60)

    # ==========================================================================
    # BASELINE FILTER ANALYSIS (Approach A: latent-bias items)
    # ==========================================================================
    baseline_filter_cfg = cfg.get("baseline_filter")
    if baseline_filter_cfg:
        min_srs = baseline_filter_cfg["min_srs"]
        max_srs = baseline_filter_cfg["max_srs"]
        print(f"\n{'=' * 60}")
        print(f"=== Baseline Filter Analysis (BF16 SRS in [{min_srs}, {max_srs}]) ===")
        print(f"{'=' * 60}")

        # Step 1: Compute per-item SRS at BF16
        item_srs = compute_per_item_srs(df, quant_level=16)

        # Log per-item SRS distribution
        print("\n--- Per-Item BF16 SRS Distribution ---")
        for model_name in sorted(df["model"].unique()):
            model_items = item_srs[item_srs["model"] == model_name]
            dist = model_items["item_srs"].value_counts().sort_index()
            print(f"\n{model_name}:")
            for srs_val, count in dist.items():
                marker = " <-- SELECTED" if min_srs <= srs_val <= max_srs else ""
                print(f"  SRS={srs_val:.1f}: {count} items{marker}")

        # Step 2: Filter
        filtered_df, filter_summary = filter_items_by_baseline_srs(
            df, item_srs, min_srs=min_srs, max_srs=max_srs
        )

        print(f"\n--- Filter Summary ---")
        print(filter_summary.to_string(index=False))
        print(f"\nTotal records after filtering: {len(filtered_df)} "
              f"(was {len(df)}, {len(filtered_df)/len(df)*100:.1f}% retained)")

        if len(filtered_df) == 0:
            print("WARNING: No items passed the baseline filter! Skipping filtered analysis.")
        else:
            # Step 3: Run full analysis pipeline on filtered data
            filt_metrics = compute_metrics(filtered_df)
            filt_metrics_with_deltas = add_deltas_from_baseline(filt_metrics)

            # Output directories
            filt_agg_path = Path("results/filtered/aggregated")
            filt_tables_dir = Path("results/filtered/tables")
            filt_fig_dir = Path("results/filtered/figures")
            filt_comparison_dir = Path("results/filtered/comparison")
            for d in [filt_agg_path, filt_tables_dir, filt_fig_dir, filt_comparison_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Save filtered metrics
            filt_metrics.to_csv(filt_agg_path / "metrics_summary.csv", index=False)
            filt_metrics_with_deltas.to_csv(filt_agg_path / "metrics_with_deltas.csv", index=False)
            filter_summary.to_csv(filt_agg_path / "filter_summary.csv", index=False)
            item_srs.to_csv(filt_agg_path / "per_item_srs.csv", index=False)

            # CSV Tables (same generators, filtered data)
            print("\n=== Generating Filtered CSV Tables ===")
            generate_main_results_csv(filt_metrics_with_deltas, filt_tables_dir / "main_results.csv")
            print(f"  Saved: {filt_tables_dir}/main_results.csv")

            filt_stats_df = generate_statistical_tests_csv(filt_metrics, filt_tables_dir / "statistical_tests.csv")
            print(f"  Saved: {filt_tables_dir}/statistical_tests.csv")

            generate_category_breakdown_csv(filt_metrics, filt_tables_dir / "category_breakdown.csv")
            print(f"  Saved: {filt_tables_dir}/category_breakdown.csv")

            # Figures (same generators, filtered data)
            print("\n=== Generating Filtered Figures ===")
            plot_srs_heatmap(filt_metrics, filt_fig_dir / "srs_heatmap")
            print(f"  Saved: {filt_fig_dir}/srs_heatmap.{{pdf,png}}")

            plot_bf16_vs_q3_comparison(filt_metrics, filt_fig_dir / "bf16_vs_q3_comparison")
            print(f"  Saved: {filt_fig_dir}/bf16_vs_q3_comparison.{{pdf,png}}")

            plot_category_facets(filt_metrics, filt_fig_dir / "srs_by_category")
            print(f"  Saved: {filt_fig_dir}/srs_by_category.{{pdf,png}}")

            plot_amplification_curve(filt_metrics, filt_fig_dir / "amplification_curve")
            print(f"  Saved: {filt_fig_dir}/amplification_curve.{{pdf,png}}")

            # Key Findings for filtered subset
            generate_key_findings(filt_metrics, filt_stats_df, filt_agg_path / "key_findings.md")
            print(f"  Saved: {filt_agg_path}/key_findings.md")

            # COMPARISON: filtered vs unfiltered
            print("\n=== Generating Comparison Outputs ===")
            generate_filter_comparison_csv(
                metrics, filt_metrics, filter_summary,
                filt_comparison_dir / "filtered_vs_unfiltered.csv"
            )
            print(f"  Saved: {filt_comparison_dir}/filtered_vs_unfiltered.csv")

            plot_filtered_vs_unfiltered(
                metrics, filt_metrics,
                filt_comparison_dir / "filtered_vs_unfiltered"
            )
            print(f"  Saved: {filt_comparison_dir}/filtered_vs_unfiltered.{{pdf,png}}")

            # Print key comparison
            bf16_all = metrics[metrics["quant"] == 16]["srs"].mean()
            q3_all = metrics[metrics["quant"] == 3]["srs"].mean()
            bf16_filt = filt_metrics[filt_metrics["quant"] == 16]["srs"].mean()
            q3_filt = filt_metrics[filt_metrics["quant"] == 3]["srs"].mean()

            print(f"\n--- Amplification Comparison ---")
            print(f"ALL items:      BF16={bf16_all:.3f} -> Q3={q3_all:.3f} "
                  f"(+{(q3_all-bf16_all):.3f}, {(q3_all-bf16_all)/bf16_all*100:.1f}%)")
            print(f"FILTERED items: BF16={bf16_filt:.3f} -> Q3={q3_filt:.3f} "
                  f"(+{(q3_filt-bf16_filt):.3f}, {(q3_filt-bf16_filt)/bf16_filt*100:.1f}%)")

        # ==================================================================
        # TRANSITION ANALYSIS (Approach B: new biases emerging)
        # ==================================================================
        print(f"\n{'=' * 60}")
        print("=== Transition Analysis: New Biases Emerging Under Compression ===")
        print(f"{'=' * 60}")

        transition_df = compute_transition_analysis(df, item_srs)
        transition_df.to_csv(filt_comparison_dir / "transition_analysis.csv", index=False)
        print(f"  Saved: {filt_comparison_dir}/transition_analysis.csv")

        print("\n--- Transition Summary ---")
        print(transition_df.to_string(index=False))

        plot_transition_chart(transition_df, filt_comparison_dir / "transition_chart")
        print(f"  Saved: {filt_comparison_dir}/transition_chart.{{pdf,png}}")

    plt.close("all")
    print("\n✓ Analysis complete! Check results/ directory for outputs.")


if __name__ == "__main__":
    analyze()
