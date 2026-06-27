from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .backtest import (
    OFFICIAL_ORIGIN_WEEK,
    load_inputs,
    load_training_matrix,
    predict_components,
)
from .config import (
    CHALLENGES,
    PROCESSED_DIR,
    RESULTS_DIR,
    Challenge,
    ensure_directories,
)
from .features import _matrix_for_origins, official_cutoff_date
from .utils import LOGGER, season_epiweeks, setup_logging, timed_step

FINAL_ORIGIN_YEAR = 2026


def generate_challenge_forecast(challenge: Challenge) -> pd.DataFrame:
    with timed_step(f"forecast | {challenge.name} | carregar"):
        panel, mapbiomas = load_inputs(challenge)
    observed_max = pd.to_datetime(panel["date"]).max()
    official_cutoff = official_cutoff_date(FINAL_ORIGIN_YEAR)
    cutoff = min(observed_max, official_cutoff)
    if cutoff < official_cutoff:
        LOGGER.warning(
            "forecast | %s | usando cutoff disponível %s; cutoff oficial seria %s",
            challenge.name,
            cutoff.date(),
            official_cutoff.date(),
        )
    with timed_step(
        f"forecast | {challenge.name} | treino EW{OFFICIAL_ORIGIN_WEEK}"
    ):
        train = load_training_matrix(
            challenge,
            panel,
            cutoff,
            FINAL_ORIGIN_YEAR,
            mapbiomas,
        )
    with timed_step(f"forecast | {challenge.name} | matriz futura"):
        future = _matrix_for_origins(
            panel,
            [(cutoff, season_epiweeks(FINAL_ORIGIN_YEAR))],
            mapbiomas,
            include_target=False,
        )
    LOGGER.info(
        "forecast | %s | treino=%s futuro=%s",
        challenge.name,
        f"{len(train):,}",
        f"{len(future):,}",
    )
    components = predict_components(
        challenge,
        train,
        future,
        f"forecast | {challenge.name}",
        RESULTS_DIR / "models" / challenge.name / "final",
    )
    root = RESULTS_DIR / "predictions" / challenge.name
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, component in components.items():
        component = component.copy()
        component["model"] = name
        component["round"] = "final"
        component["origin_year"] = FINAL_ORIGIN_YEAR
        directory = root / name
        directory.mkdir(exist_ok=True)
        component.to_parquet(directory / "final_forecast.parquet", index=False)
        outputs.append(component)
    future.to_parquet(
        PROCESSED_DIR / f"model_matrix_{challenge.name}_final.parquet",
        index=False,
    )
    combined = pd.concat(outputs, ignore_index=True)
    print(
        f"{challenge.name}: {len(combined):,} previsões finais em {len(outputs)} modelos."
    )
    return combined


def generate_all_forecasts() -> pd.DataFrame:
    ensure_directories()
    outputs = [
        generate_challenge_forecast(challenge) for challenge in CHALLENGES
    ]
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_parquet(
        RESULTS_DIR / "predictions" / "final_forecasts_all.parquet",
        index=False,
    )
    return combined


def main() -> None:
    setup_logging()
    generate_all_forecasts()


if __name__ == "__main__":
    main()
