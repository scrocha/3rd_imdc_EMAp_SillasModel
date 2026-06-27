from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR
from .metrics import CENTRAL_INTERVALS, wis
from .utils import quantile_column

INTERVAL_COLORS = {
    0.95: "#dbeafe",
    0.90: "#bfdbfe",
    0.80: "#93c5fd",
    0.50: "#60a5fa",
}


def plot_validation_forecasts(
    predictions: pd.DataFrame,
    challenge_name: str,
    output_root: Path | None = None,
) -> list[Path]:
    root = output_root or RESULTS_DIR / "plots" / "validation"
    paths = []
    for (model, round_name), frame in predictions.groupby(
        ["model", "round"], observed=True
    ):
        quantile_columns = [
            quantile_column(value)
            for bounds in CENTRAL_INTERVALS.values()
            for value in bounds
        ]
        weekly = (
            frame.assign(date=pd.to_datetime(frame["date"]))
            .groupby("date", observed=True)[["target", *quantile_columns]]
            .sum(min_count=1)
            .sort_index()
        )
        median = quantile_column(0.5)
        if median not in weekly:
            weekly[median] = (
                frame.assign(date=pd.to_datetime(frame["date"]))
                .groupby("date", observed=True)[median]
                .sum(min_count=1)
            )

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        for coverage in sorted(CENTRAL_INTERVALS, reverse=True):
            lower, upper = CENTRAL_INTERVALS[coverage]
            ax.fill_between(
                weekly.index,
                weekly[quantile_column(lower)],
                weekly[quantile_column(upper)],
                color=INTERVAL_COLORS[coverage],
                label=f"Intervalo {int(coverage * 100)}% (aprox.)",
            )
        ax.plot(
            weekly.index,
            weekly[median],
            color="#2563eb",
            linewidth=2,
            label="Mediana prevista",
        )
        ax.plot(
            weekly.index,
            weekly["target"],
            color="#111827",
            linewidth=1.8,
            label="Casos observados",
        )
        valid = frame.dropna(subset=["target"])
        score = float(wis(valid).mean()) if not valid.empty else float("nan")
        ax.set(
            title=(
                f"{challenge_name} — {model} — {round_name}\n"
                f"Total semanal; intervalos somam marginais | WIS={score:.2f}"
            ),
            xlabel="Semana",
            ylabel="Número de casos",
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        path = root / challenge_name / str(model) / f"{round_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
