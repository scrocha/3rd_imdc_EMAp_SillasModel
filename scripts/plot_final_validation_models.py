from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CHALLENGES, RESULTS_DIR
from src.config import PROCESSED_DIR
from src.features import feature_columns
from src.metrics import wis
from src.utils import quantile_column

ROUND_NAME = "validation_3"
EXTERNAL_MODELS = ("3rd_imdc_emap_lstm",)
BASELINE_MODEL = "baseline"
MODEL_LABELS = {
    "xgb_quantile": "XGB quantile",
    "xgb_quantile_log1p": "XGB log1p",
    "xgb_residual": "XGB residual",
    "baseline": "Baseline (mediana histórica)",
    "3rd_imdc_emap_lstm": "LSTM EMAp",
}
MODEL_ORDER = (
    "xgb_quantile",
    "xgb_quantile_log1p",
    "xgb_residual",
)
COLORS = {
    "truth": "#000000",
    "xgb_quantile": "#0072B2",
    "xgb_quantile_log1p": "#009E73",
    "xgb_residual": "#D55E00",
    "baseline": "#999999",
    "3rd_imdc_emap_lstm": "#56B4E9",
}
LINESTYLES = {
    "3rd_imdc_emap_lstm": (0, (2, 1)),
    "baseline": (0, (4, 2)),
}
XGB_MODELS = MODEL_ORDER


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_prediction(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    if "date" not in frame.columns and "target_end_date" in frame.columns:
        frame = frame.rename(columns={"target_end_date": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.copy()


def _load_model(root: Path, challenge: str, model: str, round_name: str) -> pd.DataFrame | None:
    base = root / challenge / model / round_name
    for path in (base.with_suffix(".parquet"), base.with_suffix(".csv")):
        if path.exists():
            frame = _read_prediction(path)
            if "target" in frame.columns:
                frame = frame.dropna(subset=["target"])
            return frame
    return None


def _load_predictions(challenge: str, round_name: str, external_dir: Path) -> dict[str, pd.DataFrame]:
    root = RESULTS_DIR / "predictions"
    forecasts = {}
    for model in MODEL_ORDER:
        frame = _load_model(root, challenge, model, round_name)
        if frame is not None:
            forecasts[model] = frame
    frame = _load_model(root, challenge, BASELINE_MODEL, round_name)
    if frame is not None:
        forecasts[BASELINE_MODEL] = frame
    for model in EXTERNAL_MODELS:
        frame = _load_model(external_dir, challenge, model, round_name)
        if frame is not None:
            forecasts[model] = frame
    return forecasts


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [quantile_column(q) for q in (0.025, 0.5, 0.975)]
    available = [column for column in ["target", *columns] if column in frame.columns]
    return frame.groupby("date", observed=True)[available].sum().reset_index().sort_values("date")


def _truth_source(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    for model in (*MODEL_ORDER, BASELINE_MODEL, *EXTERNAL_MODELS):
        frame = forecasts.get(model)
        if frame is not None and "target" in frame.columns:
            return frame
    return None


def _score_rows(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = ["target", *(quantile_column(q) for q in (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975))]
    eligible = {}
    common_index = None
    for model, frame in forecasts.items():
        if not {"location_code", "date", *required}.issubset(frame.columns):
            continue
        valid = frame.dropna(subset=["target"]).copy()
        if valid.empty:
            continue
        valid.index = pd.MultiIndex.from_frame(valid[["location_code", "date"]])
        eligible[model] = valid
        common_index = valid.index if common_index is None else common_index.intersection(valid.index)

    rows = []
    if common_index is None or common_index.empty:
        return pd.DataFrame(columns=["model", "label", "rows", "wis", "mae", "bias"])
    for model, frame in eligible.items():
        valid = frame.loc[common_index]
        if valid.empty:
            continue
        error = valid[quantile_column(0.5)] - valid["target"]
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "rows": len(valid),
                "wis": wis(valid).mean(),
                "mae": error.abs().mean(),
                "bias": error.mean(),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["model", "label", "rows", "wis", "mae", "bias"])
    return pd.DataFrame(rows).sort_values("wis")


def _best_model(scores: pd.DataFrame, forecasts: dict[str, pd.DataFrame]) -> str | None:
    for model in scores["model"].tolist():
        if model in forecasts:
            return model
    return None


def _location_wis(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["target"]).copy()
    valid["wis"] = wis(valid)
    keys = ["location_code", "location"]
    return (
        valid.groupby(keys, observed=True)["wis"]
        .mean()
        .reset_index()
        .sort_values("wis")
    )


def write_feature_importance(round_name: str) -> Path | None:
    rows = []
    for challenge in CHALLENGES:
        for model_name in XGB_MODELS:
            path = RESULTS_DIR / "models" / challenge.name / round_name / f"{model_name}.pkl"
            if not path.exists():
                continue
            with path.open("rb") as file:
                saved = pickle.load(file)
            model = saved.get("model") if isinstance(saved, dict) else saved
            booster = getattr(model, "model", None)
            columns = getattr(model, "columns", None)
            if booster is None or columns is None or not hasattr(booster, "feature_importances_"):
                continue
            rows.extend(
                {
                    "challenge": challenge.name,
                    "model": model_name,
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(columns, booster.feature_importances_)
            )
    if not rows:
        return None
    output = RESULTS_DIR / "figures" / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"feature_importance_{round_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"feature importance: {path}")
    return path


def _set_y_axis(ax: plt.Axes, truth: pd.DataFrame, aggregates: dict[str, pd.DataFrame]) -> None:
    truth_max = float(truth["target"].max()) if "target" in truth else 0.0
    median_max = max(
        float(table[quantile_column(0.5)].max())
        for table in aggregates.values()
        if quantile_column(0.5) in table.columns
    )
    cap = max(5.0, truth_max * 1.35)
    if median_max > cap * 1.5:
        ax.set_ylim(0, cap)
        ax.text(
            0.99,
            0.96,
            "y-axis capped",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            color="#666666",
        )
        ax.set_ylabel("Weekly cases, summed over locations (capped)")
    else:
        ax.set_ylabel("Weekly cases, summed over locations")


def plot_challenge(challenge: str, round_name: str, external_dir: Path, tag: str) -> Path | None:
    forecasts = _load_predictions(challenge, round_name, external_dir)
    if not forecasts:
        print(f"{challenge}: sem {round_name} local.")
        return None

    scores = _score_rows(forecasts)
    truth_frame = _truth_source(forecasts)
    if truth_frame is None or scores.empty:
        print(f"{challenge}: {round_name} sem target/score para plotar.")
        return None
    truth = _aggregate(truth_frame)
    aggregates = {model: _aggregate(frame) for model, frame in forecasts.items()}

    fig = plt.figure(figsize=(9.2, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.1])
    ax = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])

    ax.plot(
        truth["date"],
        truth["target"],
        color=COLORS["truth"],
        lw=2.8,
        label="Real",
        zorder=10,
    )
    plot_models = list(MODEL_ORDER)
    for extra in (BASELINE_MODEL, *EXTERNAL_MODELS):
        if extra not in plot_models and extra in aggregates:
            plot_models.append(extra)
    for model in plot_models:
        if model not in aggregates or quantile_column(0.5) not in aggregates[model].columns:
            continue
        table = aggregates[model]
        ls = LINESTYLES.get(model, "-")
        lw = 1.2 if ls == "-" else 1.3
        ax.plot(
            table["date"],
            table[quantile_column(0.5)],
            color=COLORS[model],
            lw=lw,
            alpha=0.9,
            linestyle=ls,
            label=MODEL_LABELS.get(model, model),
        )

    _set_y_axis(ax, truth, aggregates)
    ax.set_title(f"{challenge}: {round_name} aggregate curve")
    ax.set_xlabel("Week")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#dddddd", lw=0.5)

    labels = scores["label"].tolist()
    values = scores["wis"].to_numpy()
    colors = [COLORS.get(m, "#666666") for m in scores["model"]]
    ax_bar.barh(labels, values, color=colors)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean WIS")
    ax_bar.set_title("Model score")
    for index, row in enumerate(scores.itertuples(index=False)):
        ax_bar.text(row.wis, index, f" {row.wis:.1f}", va="center", fontsize=7)
    ax_bar.grid(axis="x", color="#dddddd", lw=0.5)

    output = RESULTS_DIR / "figures" / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{challenge}_{tag}.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    scores.to_csv(output / f"{challenge}_{tag}_scores.csv", index=False)
    print(f"{challenge}: {path}")
    return path


def plot_diagnostics(challenge: str, round_name: str, external_dir: Path) -> Path | None:
    forecasts = _load_predictions(challenge, round_name, external_dir)
    scores = _score_rows(forecasts)
    if scores.empty:
        return None
    models = scores["model"].tolist()
    fig, (ax_scatter, ax_resid) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.2),
        constrained_layout=True,
    )
    max_value = 0.0
    residual_rows = []
    for model in models:
        frame = forecasts[model].dropna(subset=["target"]).copy()
        if frame.empty:
            continue
        predicted = frame[quantile_column(0.5)].to_numpy(dtype=float)
        observed = frame["target"].to_numpy(dtype=float)
        max_value = max(max_value, float(np.nanmax([predicted.max(), observed.max()])))
        ax_scatter.scatter(
            observed,
            predicted,
            s=8,
            alpha=0.28,
            color=COLORS.get(model, "#666666"),
            label=MODEL_LABELS.get(model, model),
            linewidths=0,
        )
        residual_rows.append(
            pd.DataFrame(
                {
                    "model": MODEL_LABELS.get(model, model),
                    "residual": predicted - observed,
                    "color": COLORS.get(model, "#666666"),
                }
            )
        )
    if not residual_rows:
        return None
    limit = max(1.0, max_value)
    ax_scatter.plot([0, limit], [0, limit], color="#333333", lw=1, ls="--", label="Ideal")
    ax_scatter.set_xlim(0, limit)
    ax_scatter.set_ylim(0, limit)
    ax_scatter.set_xlabel("Observed weekly cases")
    ax_scatter.set_ylabel("Predicted median")
    ax_scatter.set_title(f"{challenge}: predicted vs observed")
    ax_scatter.legend(frameon=False, fontsize=6)
    ax_scatter.grid(color="#dddddd", lw=0.4)

    residuals = pd.concat(residual_rows, ignore_index=True)
    groups = [group["residual"].to_numpy(dtype=float) for _, group in residuals.groupby("model", sort=False)]
    labels = [name for name, _ in residuals.groupby("model", sort=False)]
    colors = [
        residuals.loc[residuals["model"].eq(label), "color"].iloc[0]
        for label in labels
    ]
    parts = ax_resid.violinplot(groups, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    parts["cmedians"].set_color("#111111")
    parts["cmedians"].set_linewidth(1)
    ax_resid.axhline(0, color="#333333", lw=1, ls="--")
    low, high = np.nanpercentile(residuals["residual"].to_numpy(dtype=float), [1, 99])
    bound = max(abs(low), abs(high), 1.0)
    ax_resid.set_ylim(-bound, bound)
    clipped = int(residuals["residual"].abs().gt(bound).sum())
    if clipped:
        ax_resid.text(
            0.98,
            0.96,
            f"central 98%; {clipped} clipped",
            ha="right",
            va="top",
            transform=ax_resid.transAxes,
            fontsize=6,
            color="#666666",
        )
    ax_resid.set_xticks(range(1, len(labels) + 1))
    ax_resid.set_xticklabels(labels, rotation=35, ha="right")
    ax_resid.set_ylabel("Residual (predicted - observed)")
    ax_resid.set_title("Residual distribution")
    ax_resid.grid(axis="y", color="#dddddd", lw=0.4)

    output = RESULTS_DIR / "figures" / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{challenge}_{round_name}_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    residuals.drop(columns="color").to_csv(
        output / f"{challenge}_{round_name}_residuals.csv",
        index=False,
    )
    print(f"{challenge}: {path}")
    return path


def plot_location_examples(challenge: str, round_name: str, external_dir: Path) -> Path | None:
    forecasts = _load_predictions(challenge, round_name, external_dir)
    scores = _score_rows(forecasts)
    model = _best_model(scores, forecasts)
    truth_frame = _truth_source(forecasts)
    if model is None or truth_frame is None:
        return None
    locations = _location_wis(forecasts[model])
    if locations.empty:
        return None
    selected = pd.concat([locations.head(2), locations.tail(2)], ignore_index=True).drop_duplicates(
        "location_code"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    axes = axes.ravel()
    for ax, row in zip(axes, selected.itertuples(index=False)):
        code = row.location_code
        truth = truth_frame.loc[truth_frame["location_code"].eq(code)].sort_values("date")
        ax.plot(truth["date"], truth["target"], color=COLORS["truth"], lw=2.2, label="Real", zorder=10)
        median_max = float(truth["target"].max())
        plot_models = list(MODEL_ORDER)
        for extra in (BASELINE_MODEL, *EXTERNAL_MODELS):
            if extra not in plot_models:
                plot_models.append(extra)
        for current_model in plot_models:
            frame = forecasts.get(current_model)
            if frame is None:
                continue
            table = frame.loc[frame["location_code"].eq(code)].sort_values("date")
            if table.empty or quantile_column(0.5) not in table:
                continue
            median_max = max(median_max, float(table[quantile_column(0.5)].max()))
            ls = LINESTYLES.get(current_model, "-")
            lw = 1.8 if current_model == model else (1.3 if ls != "-" else 1.1)
            ax.plot(
                table["date"],
                table[quantile_column(0.5)],
                color=COLORS.get(current_model, "#666666"),
                lw=lw,
                linestyle=ls,
                alpha=0.9,
                label=MODEL_LABELS.get(current_model, current_model),
            )
        ymax = max(5.0, float(truth["target"].max()) * 1.35)
        ax.set_ylim(0, ymax)
        if median_max > ymax:
            ax.text(
                0.99,
                0.94,
                "y-axis capped",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
                color="#666666",
            )
        ax.set_title(f"{row.location} | WIS={row.wis:.1f}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Weekly cases")
        ax.grid(axis="y", color="#dddddd", lw=0.4)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{challenge}: 2 best and 2 worst locations by {MODEL_LABELS.get(model, model)} WIS")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)

    output = RESULTS_DIR / "figures" / "model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{challenge}_{round_name}_location_examples.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    selected.assign(selection_model=model).to_csv(
        output / f"{challenge}_{round_name}_location_examples.csv", index=False
    )
    print(f"{challenge}: {path}")
    return path


def plot_variable_analysis(challenge: str, round_name: str) -> Path | None:
    importance_path = RESULTS_DIR / "figures" / "model_comparison" / "feature_importance_validation_3.csv"
    matrix_path = PROCESSED_DIR / f"model_matrix_{challenge}_final.parquet"
    if not importance_path.exists() or not matrix_path.exists():
        return None
    importance = pd.read_csv(importance_path)
    importance = importance.loc[importance["challenge"].eq(challenge)]
    if importance.empty:
        return None
    challenge_config = next(item for item in CHALLENGES if item.name == challenge)
    model = challenge_config.model_names[0]
    top = importance.loc[importance["model"].eq(model)].nlargest(12, "importance")
    matrix = pd.read_parquet(matrix_path)
    candidate_columns = [column for column in top["feature"] if column in feature_columns(matrix)]
    if len(candidate_columns) < 2:
        return None
    values = matrix[candidate_columns].replace([np.inf, -np.inf], np.nan)
    numeric = values.dropna(axis=1, how="all")
    variable_columns = [
        column
        for column in numeric.columns
        if numeric[column].nunique(dropna=True) > 1
    ]
    corr = numeric[variable_columns].corr() if len(variable_columns) >= 2 else pd.DataFrame()
    if len(variable_columns) >= 2:
        clean_corr = corr.fillna(0).to_numpy(dtype=float)
        inverse = np.linalg.pinv(clean_corr, hermitian=True)
        vif = pd.DataFrame(
            {"feature": variable_columns, "vif": np.maximum(np.diag(inverse), 1)}
        ).sort_values("vif", ascending=False)
    else:
        vif = pd.DataFrame({"feature": candidate_columns, "vif": np.nan})

    fig, (ax_bar, ax_vif) = plt.subplots(
        1, 2, figsize=(10.2, 4.8), constrained_layout=True, width_ratios=[1.2, 1.0]
    )
    ordered = top.sort_values("importance")
    ax_bar.barh(ordered["feature"], ordered["importance"], color="#0072B2")
    ax_bar.set_xlabel("XGBoost importance")
    ax_bar.set_title("Top predictors in final model")
    ax_bar.grid(axis="x", color="#dddddd", lw=0.4)

    vif_plot = vif.dropna(subset=["vif"]).head(12).sort_values("vif")
    if vif_plot.empty:
        ax_vif.text(
            0.5,
            0.5,
            "VIF unavailable:\nconstant predictors in subset",
            ha="center",
            va="center",
            transform=ax_vif.transAxes,
            color="#666666",
        )
        ax_vif.set_axis_off()
    else:
        colors = np.where(vif_plot["vif"].gt(10), "#D55E00", "#009E73")
        ax_vif.barh(vif_plot["feature"], vif_plot["vif"], color=colors)
        ax_vif.axvline(5, color="#666666", lw=0.8, ls="--")
        ax_vif.axvline(10, color="#333333", lw=0.8, ls=":")
        ax_vif.set_xlabel("VIF among non-constant top predictors")
        ax_vif.set_title("Collinearity diagnostic")
        ax_vif.grid(axis="x", color="#dddddd", lw=0.4)
    fig.suptitle(
        f"{challenge}: predictors from final {MODEL_LABELS.get(model, model)}",
        fontsize=11,
        fontweight="bold",
    )

    output = RESULTS_DIR / "figures" / "model_comparison"
    path = output / f"{challenge}_{round_name}_variable_analysis.png"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(output / f"{challenge}_{round_name}_top_feature_correlation.csv")
    vif.to_csv(output / f"{challenge}_{round_name}_top_feature_vif.csv", index=False)
    print(f"{challenge}: {path}")
    return path


def write_consolidated_scores(round_name: str, tag: str) -> Path | None:
    output = RESULTS_DIR / "figures" / "model_comparison"
    rows = []
    for path in sorted(output.glob(f"*_{tag}_scores.csv")):
        challenge = path.name.removesuffix(f"_{tag}_scores.csv")
        table = pd.read_csv(path)
        table.insert(0, "challenge", challenge)
        rows.append(table)
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    combined["rank"] = combined.groupby("challenge", observed=True)["wis"].rank(method="min")
    path = output / f"all_challenges_{round_name}_model_scores.csv"
    combined.sort_values(["challenge", "rank", "model"]).to_csv(path, index=False)
    print(f"all challenges: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=RESULTS_DIR / "predictions_external",
        help="Folder with external model files: <challenge>/<model>/<validation_n>.parquet|csv",
    )
    args = parser.parse_args()

    _style()
    write_feature_importance(ROUND_NAME)
    for challenge in CHALLENGES:
        plot_challenge(challenge.name, ROUND_NAME, args.external_dir, f"{ROUND_NAME}_models")
        plot_diagnostics(challenge.name, ROUND_NAME, args.external_dir)
        plot_location_examples(challenge.name, ROUND_NAME, args.external_dir)
        plot_variable_analysis(challenge.name, ROUND_NAME)
    write_consolidated_scores(ROUND_NAME, f"{ROUND_NAME}_models")


if __name__ == "__main__":
    main()
