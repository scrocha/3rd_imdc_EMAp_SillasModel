from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import numpy as np
import pandas as pd

from .build_dataset import write_outputs
from .config import (
    CHALLENGES,
    FIRST_ORIGIN_YEAR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    ROUNDS,
    Challenge,
    ensure_directories,
)
from .features import (
    build_official_like_training_matrix,
    build_origin_matrix,
    feature_columns,
    official_cutoff_date,
)
from .metrics import CENTRAL_INTERVALS, summarize_metrics, wis
from .models import (
    XGBoostQuantileForecaster,
    XGBoostResidualForecaster,
    accelerator_device,
    historical_quantile_baseline,
)
from .plots import plot_validation_forecasts
from .utils import LOGGER, quantile_column, setup_logging, timed_step

BASELINE_NAME = "baseline"
MODEL_NAMES = ("xgb_quantile", "xgb_quantile_log1p", "xgb_residual")
OFFICIAL_ORIGIN_WEEK = 25


def _compact_training(matrix: pd.DataFrame) -> pd.DataFrame:
    keep = [
        *feature_columns(matrix),
        "origin_date",
        "target_end_date",
        "date",
        "target",
    ]
    output = matrix.loc[:, list(dict.fromkeys(keep))].copy()
    floats = output.select_dtypes(include="float64").columns
    integers = output.select_dtypes(include="int64").columns
    output = output.astype(
        {
            **dict.fromkeys(floats, "float32"),
            **dict.fromkeys(integers, "int32"),
        }
    )
    return output


def available_training_rows(
    matrix: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Rows whose full forecast window is observable by cutoff."""
    cutoff = pd.Timestamp(cutoff)
    date_column = (
        "target_end_date" if "target_end_date" in matrix.columns else "date"
    )
    output = matrix.loc[pd.to_datetime(matrix[date_column]).le(cutoff)]
    return (
        output.dropna(subset=["target"])
        if "target" in output.columns
        else output
    )


def load_inputs(
    challenge: Challenge,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    panel_path = PROCESSED_DIR / f"panel_{challenge.name}_week.parquet"
    if not panel_path.exists():
        write_outputs()
    panel = pd.read_parquet(panel_path)
    mapbiomas_name = (
        "mapbiomas_uf_year.parquet"
        if challenge.level == "uf"
        else "mapbiomas_muni_year.parquet"
        if challenge.level == "city"
        else None
    )
    mapbiomas_path = PROCESSED_DIR / mapbiomas_name if mapbiomas_name else None
    mapbiomas = (
        pd.read_parquet(mapbiomas_path)
        if mapbiomas_path is not None and mapbiomas_path.exists()
        else None
    )
    if mapbiomas is not None:
        source_key = "uf_code" if challenge.level == "uf" else "geocode"
        mapbiomas = mapbiomas.rename(columns={source_key: "location_code"})
    return panel, mapbiomas


def load_training_matrix(
    challenge: Challenge,
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    origin_year: int,
    mapbiomas: pd.DataFrame | None,
) -> pd.DataFrame:
    cache_path = (
        INTERIM_DIR
        / f"official_ew{OFFICIAL_ORIGIN_WEEK}_training_v1_{challenge.name}_{origin_year}_{pd.Timestamp(cutoff).date()}.parquet"
    )
    if cache_path.exists():
        with timed_step(
            f"{challenge.name} | carregar treino EW{OFFICIAL_ORIGIN_WEEK}"
        ):
            train = pd.read_parquet(cache_path)
    else:
        with timed_step(f"{challenge.name} | treino EW{OFFICIAL_ORIGIN_WEEK}"):
            train = build_official_like_training_matrix(
                panel,
                cutoff,
                mapbiomas,
                origin_weeks=[OFFICIAL_ORIGIN_WEEK],
                years=range(FIRST_ORIGIN_YEAR, origin_year),
            )
        with timed_step(
            f"{challenge.name} | salvar treino EW{OFFICIAL_ORIGIN_WEEK}"
        ):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            train.to_parquet(cache_path, index=False)
    train = available_training_rows(train, cutoff)
    if "target_end_date" in train:
        max_target_end = pd.to_datetime(train["target_end_date"]).max()
        if pd.notna(max_target_end) and max_target_end > pd.Timestamp(cutoff):
            raise ValueError(
                f"{challenge.name}: treino vazaria alvo ate {max_target_end.date()} "
                f"para cutoff {pd.Timestamp(cutoff).date()}."
            )
    if "origin_year" in train and train["origin_year"].max() >= origin_year:
        raise ValueError(
            f"{challenge.name}: treino inclui origem >= validation/final {origin_year}."
        )
    return _compact_training(train)


def _save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inner = model.get("model") if isinstance(model, dict) else model
    booster_model = getattr(inner, "model", None)
    if booster_model is not None and hasattr(booster_model, "get_booster"):
        booster_model.get_booster().set_param({"device": "cpu"})
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _fit_or_load(model: object, train: pd.DataFrame, path: Path) -> object:
    n_features = len(feature_columns(train))
    n_dates = train["date"].nunique() if "date" in train else 0
    target_hash = (
        hashlib.sha256(
            train["target"].fillna(0).to_numpy(dtype=float).tobytes()
        ).hexdigest()
        if "target" in train
        else ""
    )
    signature_data = (
        type(model).__qualname__,
        repr(model),
        tuple(feature_columns(train)),
        len(train),
        str(train["date"].max()) if "date" in train else "",
        target_hash,
    )
    signature = hashlib.sha256(repr(signature_data).encode()).hexdigest()
    if path.exists():
        with path.open("rb") as file:
            cached = pickle.load(file)
        if (
            isinstance(cached, dict)
            and cached.get("signature") == signature
            and "model" in cached
        ):
            LOGGER.info("modelo carregado | %s", path)
            return cached["model"]
        LOGGER.info("cache de modelo obsoleto; retreinando | %s", path)
    LOGGER.info(
        "treinando | %s | linhas=%s features=%s datas_alvo=%s",
        path,
        f"{len(train):,}",
        n_features,
        n_dates,
    )
    fitted = model.fit(train)
    _save_model({"signature": signature, "model": fitted}, path)
    LOGGER.info("modelo salvo | %s", path)
    return fitted


def _predict_component(
    challenge: Challenge,
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_directory: Path,
) -> pd.DataFrame:
    if name == BASELINE_NAME:
        return historical_quantile_baseline(test)
    params_path = (
        RESULTS_DIR / "models" / challenge.name / f"{name}_params.json"
    )
    params = (
        json.loads(params_path.read_text()) if params_path.exists() else {}
    )
    if name == "xgb_quantile":
        model = XGBoostQuantileForecaster(
            use_extreme_features=True, params=params
        )
    elif name == "xgb_quantile_log1p":
        model = XGBoostQuantileForecaster(
            log_target=True,
            use_extreme_features=True,
            params=params,
        )
    elif name == "xgb_residual":
        model = XGBoostResidualForecaster(
            use_extreme_features=True, params=params
        )
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return _fit_or_load(model, train, model_directory / f"{name}.pkl").predict(
        test
    )


def backtest_model_names() -> tuple[str, ...]:
    return (BASELINE_NAME, *MODEL_NAMES)


def predict_components(
    challenge: Challenge,
    train: pd.DataFrame,
    test: pd.DataFrame,
    context: str,
    model_directory: Path,
    timings: list[dict[str, object]] | None = None,
    round_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    names = backtest_model_names()
    device = accelerator_device()
    LOGGER.info(
        "%s | device=%s | modelos=%d | treino=%s teste=%s",
        context,
        device,
        len(names),
        f"{len(train):,}",
        f"{len(test):,}",
    )
    output = {}
    for name in names:
        started = time.perf_counter()
        with timed_step(f"{context} | {name}"):
            output[name] = _predict_component(
                challenge, name, train, test, model_directory
            )
        if timings is not None:
            timings.append(
                {
                    "round": round_name,
                    "model": name,
                    "seconds": time.perf_counter() - started,
                }
            )
    return output


def _write_metrics(challenge: Challenge, predictions: pd.DataFrame) -> None:
    directory = RESULTS_DIR / "metrics" / challenge.name
    directory.mkdir(parents=True, exist_ok=True)
    valid = predictions.dropna(subset=["target"]).copy()
    valid["wis"] = wis(valid)
    valid["absolute_error"] = (
        valid["target"] - valid[quantile_column(0.5)]
    ).abs()
    valid["bias"] = valid[quantile_column(0.5)] - valid["target"]
    valid.groupby(["model", "round"], observed=True).agg(
        wis=("wis", "mean"),
        mae_median=("absolute_error", "mean"),
        bias_median=("bias", "mean"),
        rows=("target", "size"),
    ).reset_index().to_csv(directory / "wis_by_model.csv", index=False)
    valid.groupby(["model", "location_code", "location"], observed=True)[
        "wis"
    ].mean().reset_index().to_csv(
        directory / "wis_by_location.csv", index=False
    )
    valid.groupby(["model", "epiweek"], observed=True)[
        "wis"
    ].mean().reset_index().to_csv(directory / "wis_by_week.csv", index=False)
    coverage_rows = []
    for (model, round_name), group in valid.groupby(
        ["model", "round"], observed=True
    ):
        for coverage, (lower, upper) in CENTRAL_INTERVALS.items():
            inside = group["target"].between(
                group[quantile_column(lower)], group[quantile_column(upper)]
            )
            coverage_rows.append(
                {
                    "model": model,
                    "round": round_name,
                    "interval": coverage,
                    "coverage": inside.mean(),
                    "mean_width": (
                        group[quantile_column(upper)]
                        - group[quantile_column(lower)]
                    ).mean(),
                }
            )
    pd.DataFrame(coverage_rows).to_csv(
        directory / "coverage_by_interval.csv", index=False
    )
    valid.groupby(["model", "round"], observed=True)["bias"].agg(
        mean="mean",
        std="std",
        median="median",
        q05=lambda values: values.quantile(0.05),
        q95=lambda values: values.quantile(0.95),
        rmse=lambda values: np.sqrt(np.mean(values**2)),
    ).reset_index().to_csv(directory / "residuals_by_model.csv", index=False)


def run_challenge_backtests(challenge: Challenge) -> pd.DataFrame:
    with timed_step(f"backtest | {challenge.name} | carregar"):
        panel, mapbiomas = load_inputs(challenge)
    all_predictions: list[pd.DataFrame] = []
    timing_rows: list[dict[str, object]] = []
    with timed_step(f"backtest | {challenge.name} | matriz oficial"):
        official_matrix = build_origin_matrix(
            panel,
            [round_spec.origin_year for round_spec in ROUNDS[:-1]],
            mapbiomas,
            include_target=True,
        )
    LOGGER.info(
        "backtest | %s | treino=EW%s | oficial=%s linhas",
        challenge.name,
        OFFICIAL_ORIGIN_WEEK,
        f"{len(official_matrix):,}",
    )
    for round_spec in ROUNDS[:-1]:
        cutoff = official_cutoff_date(round_spec.origin_year)
        train = load_training_matrix(
            challenge,
            panel,
            cutoff,
            round_spec.origin_year,
            mapbiomas,
        )
        test = official_matrix.loc[
            official_matrix["origin_year"].eq(round_spec.origin_year)
        ].copy()
        complete_round = bool(test["target"].notna().all())
        LOGGER.info(
            "backtest | %s | %s | treino=%s teste=%s",
            challenge.name,
            round_spec.name,
            f"{len(train):,}",
            f"{len(test):,}",
        )
        components = predict_components(
            challenge,
            train,
            test,
            f"backtest | {challenge.name} | {round_spec.name}",
            RESULTS_DIR / "models" / challenge.name / round_spec.name,
            timing_rows,
            round_spec.name,
        )

        for name, prediction in components.items():
            prediction = prediction.copy()
            prediction["model"] = name
            prediction["round"] = round_spec.name
            prediction["origin_year"] = round_spec.origin_year
            prediction["training_cutoff"] = cutoff
            prediction["training_target_end_max"] = pd.to_datetime(
                train["target_end_date"]
            ).max()
            prediction["round_complete"] = complete_round
            all_predictions.append(prediction)
            directory = RESULTS_DIR / "predictions" / challenge.name / name
            directory.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(
                directory / f"{round_spec.name}.parquet", index=False
            )
    combined = pd.concat(all_predictions, ignore_index=True)
    evaluated = combined.loc[combined["round_complete"]]
    _write_metrics(challenge, evaluated)
    model_root = RESULTS_DIR / "models" / challenge.name
    model_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timing_rows).to_csv(
        RESULTS_DIR / "metrics" / challenge.name / "model_timings.csv",
        index=False,
    )
    summary = []
    for name, group in evaluated.groupby("model", observed=True):
        table = summarize_metrics(group)
        table.insert(0, "model", name)
        summary.append(table)
    summary_frame = pd.concat(summary, ignore_index=True)
    summary_frame.insert(0, "challenge", challenge.name)
    summary_frame.to_csv(
        RESULTS_DIR / "metrics" / challenge.name / "backtest_summary.csv",
        index=False,
    )
    plot_validation_forecasts(combined, challenge.name)
    print(summary_frame.to_string(index=False))
    return combined


def run_all_backtests(
    challenges: tuple[Challenge, ...] = CHALLENGES,
) -> pd.DataFrame:
    ensure_directories()
    outputs = [run_challenge_backtests(challenge) for challenge in challenges]
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_parquet(
        RESULTS_DIR / "predictions" / "backtests_all.parquet", index=False
    )
    return combined


def generate_report_figures() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_final_validation_models.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge",
        nargs="+",
        choices=[challenge.name for challenge in CHALLENGES],
        default=[challenge.name for challenge in CHALLENGES],
    )
    args = parser.parse_args()
    selected = tuple(
        challenge
        for challenge in CHALLENGES
        if challenge.name in set(args.challenge)
    )
    run_all_backtests(selected)
    generate_report_figures()


if __name__ == "__main__":
    main()
