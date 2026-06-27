from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    CHALLENGES,
    RESULTS_DIR,
    SUBMISSIONS_DIR,
    ensure_directories,
)
from .forecast import generate_all_forecasts
from .metrics import enforce_quantile_order
from .utils import LOGGER, quantile_column, setup_logging, timed_step

SUBMISSION_COLUMNS = (
    "date",
    "pred",
    "lower_50",
    "upper_50",
    "lower_80",
    "upper_80",
    "lower_90",
    "upper_90",
    "lower_95",
    "upper_95",
)


def to_submission_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = enforce_quantile_order(predictions)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(predictions["date"]).dt.strftime(
                "%Y-%m-%d"
            ),
            "pred": predictions[quantile_column(0.5)],
            "lower_50": predictions[quantile_column(0.25)],
            "upper_50": predictions[quantile_column(0.75)],
            "lower_80": predictions[quantile_column(0.10)],
            "upper_80": predictions[quantile_column(0.90)],
            "lower_90": predictions[quantile_column(0.05)],
            "upper_90": predictions[quantile_column(0.95)],
            "lower_95": predictions[quantile_column(0.025)],
            "upper_95": predictions[quantile_column(0.975)],
        }
    )


def validate_submission(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Colunas inválidas: {list(frame.columns)}")
    if len(frame) not in {52, 53}:
        raise ValueError(
            f"Uma temporada deve ter 52/53 semanas; recebeu {len(frame)}."
        )
    dates = pd.to_datetime(frame["date"])
    if not dates.dt.dayofweek.eq(6).all():
        raise ValueError("Todas as datas semanais devem ser domingos.")
    if not dates.sort_values().diff().dropna().dt.days.eq(7).all():
        raise ValueError("As datas devem ser contínuas, sem lacunas.")
    if frame.drop(columns="date").lt(0).any().any():
        raise ValueError("Previsões não podem ser negativas.")
    ordered = [
        "lower_95",
        "lower_90",
        "lower_80",
        "lower_50",
        "pred",
        "upper_50",
        "upper_80",
        "upper_90",
        "upper_95",
    ]
    if (frame[ordered].diff(axis=1).iloc[:, 1:] < 0).any().any():
        raise ValueError("Intervalos preditivos não estão aninhados.")


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def make_submission_files() -> None:
    ensure_directories()
    if not (
        RESULTS_DIR / "predictions" / "final_forecasts_all.parquet"
    ).exists():
        generate_all_forecasts()
    if SUBMISSIONS_DIR.exists():
        shutil.rmtree(SUBMISSIONS_DIR)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for challenge in CHALLENGES:
        LOGGER.info("submission | desafio=%s", challenge.name)
        root = RESULTS_DIR / "predictions" / challenge.name
        models = challenge.model_names
        sources = [
            (
                model,
                f"validation_{index}",
                root / model / f"validation_{index}.parquet",
            )
            for model in models
            for index in range(1, 5)
        ]
        sources.extend(
            (model, "final", root / model / "final_forecast.parquet")
            for model in models
        )
        missing = [path for _, _, path in sources if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{challenge.name}: previsões ausentes: {missing}"
            )
        for model, round_name, source in sources:
            predictions = pd.read_parquet(source)
            for (code, location), group in predictions.groupby(
                ["location_code", "location"], observed=True
            ):
                submission = to_submission_rows(group.sort_values("date"))
                validate_submission(submission)
                directory = (
                    SUBMISSIONS_DIR / challenge.name / model / round_name
                )
                directory.mkdir(parents=True, exist_ok=True)
                submission.to_csv(
                    directory / f"{int(code)}_{_safe_name(location)}.csv",
                    index=False,
                )
                enriched = submission.copy()
                enriched.insert(0, "location", location)
                enriched.insert(0, "location_code", int(code))
                enriched.insert(0, "round", round_name)
                enriched.insert(0, "model", model)
                enriched.insert(0, "challenge", challenge.name)
                combined.append(enriched)
    all_forecasts = pd.concat(combined, ignore_index=True)
    print(
        f"Submissões: {all_forecasts['challenge'].nunique()} desafios, "
        f"{len(all_forecasts):,} linhas."
    )


def main() -> None:
    setup_logging()
    with timed_step("submission | gerar arquivos"):
        make_submission_files()


if __name__ == "__main__":
    main()
