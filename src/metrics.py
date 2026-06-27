from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUANTILES
from .utils import quantile_column

CENTRAL_INTERVALS = {
    0.50: (0.25, 0.75),
    0.80: (0.10, 0.90),
    0.90: (0.05, 0.95),
    0.95: (0.025, 0.975),
}


def interval_score(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    score = upper - lower
    score += (2 / alpha) * (lower - y) * (y < lower)
    score += (2 / alpha) * (y - upper) * (y > upper)
    return score


def wis(frame: pd.DataFrame, target: str = "target") -> pd.Series:
    y = frame[target].to_numpy(dtype=float)
    median_error = np.abs(
        y - frame[quantile_column(0.5)].to_numpy(dtype=float)
    )
    weighted = 0.5 * median_error
    for coverage, (lower_q, upper_q) in CENTRAL_INTERVALS.items():
        alpha = 1 - coverage
        score = interval_score(
            y,
            frame[quantile_column(lower_q)].to_numpy(dtype=float),
            frame[quantile_column(upper_q)].to_numpy(dtype=float),
            alpha,
        )
        weight = alpha / 2
        weighted += weight * score
    denominator = len(CENTRAL_INTERVALS) + 0.5
    return pd.Series(weighted / denominator, index=frame.index, name="wis")


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions.dropna(subset=["target"]).copy()
    valid["wis"] = wis(valid)
    valid["absolute_error"] = (
        valid["target"] - valid[quantile_column(0.5)]
    ).abs()
    records = []
    for name, group in valid.groupby("round", observed=True):
        record: dict[str, object] = {
            "round": name,
            "rows": len(group),
            "wis": group["wis"].mean(),
            "mae_median": group["absolute_error"].mean(),
            "bias_median": (
                group[quantile_column(0.5)] - group["target"]
            ).mean(),
        }
        for coverage, (lower_q, upper_q) in CENTRAL_INTERVALS.items():
            inside = group["target"].between(
                group[quantile_column(lower_q)],
                group[quantile_column(upper_q)],
            )
            record[f"coverage_{int(coverage * 100)}"] = inside.mean()
        records.append(record)
    return pd.DataFrame(records)


def compare_models(predictions: pd.DataFrame) -> pd.DataFrame:
    if "round_complete" in predictions:
        predictions = predictions.loc[predictions["round_complete"]]
    valid = predictions.dropna(subset=["target"]).copy()
    valid["wis"] = wis(valid)
    valid["residual"] = valid[quantile_column(0.5)] - valid["target"]
    records = []
    for model, group in valid.groupby("model", observed=True):
        round_wis = group.groupby("round", observed=True)["wis"].mean()
        record = {
            "model": model,
            "rows": len(group),
            "rounds": round_wis.size,
            "wis": group["wis"].mean(),
            "wis_round_std": round_wis.std(ddof=0),
            "wis_worst_round": round_wis.max(),
            "mae": group["residual"].abs().mean(),
            "bias": group["residual"].mean(),
            "rmse": np.sqrt(np.mean(group["residual"] ** 2)),
            "residual_std": group["residual"].std(),
            "residual_q05": group["residual"].quantile(0.05),
            "residual_q95": group["residual"].quantile(0.95),
        }
        coverage_errors = []
        for coverage, (lower, upper) in CENTRAL_INTERVALS.items():
            suffix = int(coverage * 100)
            observed = (
                group["target"]
                .between(
                    group[quantile_column(lower)],
                    group[quantile_column(upper)],
                )
                .mean()
            )
            record[f"coverage_{suffix}"] = observed
            record[f"width_{suffix}"] = (
                group[quantile_column(upper)] - group[quantile_column(lower)]
            ).mean()
            coverage_errors.append(abs(observed - coverage))
        record["coverage_mae"] = np.mean(coverage_errors)
        records.append(record)

    comparison = pd.DataFrame(records)
    if comparison.empty:
        return comparison
    baseline = comparison.loc[comparison["model"].eq("baseline"), "wis"]
    comparison["wis_relative_baseline"] = (
        comparison["wis"] / baseline.iloc[0] if not baseline.empty else np.nan
    )
    comparison["rank_wis"] = comparison["wis"].rank(method="min").astype(int)
    return comparison.sort_values(
        ["rank_wis", "coverage_mae", "model"]
    ).reset_index(drop=True)


def enforce_quantile_order(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    columns = [quantile_column(value) for value in QUANTILES]
    values = np.sort(
        output[columns].clip(lower=0).to_numpy(dtype=float), axis=1
    )
    output[columns] = values
    return output
