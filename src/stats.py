"""Statistical tests for bias amplification analysis."""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def chi_squared_test(
    srs_base: float, n_base: int, srs_comp: float, n_comp: int
) -> dict:
    """Compare two SRS proportions using chi-squared test.

    Args:
        srs_base: Stereotype reliance score for the baseline (e.g., BF16).
        n_base: Sample size for the baseline.
        srs_comp: Stereotype reliance score for the comparison (e.g., Q4).
        n_comp: Sample size for the comparison.

    Returns:
        Dict with keys 'chi2', 'p', and 'cohens_h' (effect size).
    """
    table = np.array([
        [int(srs_base * n_base), n_base - int(srs_base * n_base)],
        [int(srs_comp * n_comp), n_comp - int(srs_comp * n_comp)],
    ])

    # Guard against zero rows/columns which make chi2 undefined
    if table.min() < 0 or table.sum() == 0:
        return {"chi2": np.nan, "p": np.nan, "cohens_h": np.nan}

    chi2, p, dof, _ = stats.chi2_contingency(table)

    # Cohen's h for effect size between two proportions
    h = 2 * (np.arcsin(np.sqrt(srs_comp)) - np.arcsin(np.sqrt(srs_base)))

    return {"chi2": chi2, "p": p, "cohens_h": h}


def logistic_regression_trend(df: pd.DataFrame):
    """Logistic regression: stereotype_chosen ~ bit_width + category.

    Tests whether lower bit-width predicts higher stereotype reliance.

    Args:
        df: DataFrame with columns 'parsed_answer', 'stereotype_target_index',
            'quant', and 'category'. Should be pre-filtered to non-null
            parsed_answer rows.

    Returns:
        statsmodels Summary2 object with regression results.
    """
    df = df.copy()
    letters = "ABC"
    df["stereotype_chosen"] = (
        df["parsed_answer"]
        == df["stereotype_target_index"].map(lambda i: letters[i])
    ).astype(int)

    X = pd.get_dummies(
        df[["quant", "category"]], columns=["category"], drop_first=True,
        dtype=float,
    )
    X = sm.add_constant(X)
    y = df["stereotype_chosen"]

    model = sm.Logit(y, X).fit(disp=0)
    return model.summary2()
