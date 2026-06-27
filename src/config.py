from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FTP_DIR = RAW_DIR / "ftp"
EXTERNAL_DIR = RAW_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

FTP_HOST = "info.dengue.mat.br"
FTP_REMOTE_DIR = "data_imdc_2026"
FTP_FILE_ALIASES = {
    "dengue.csv.gz": ("dengue.csv.gz",),
    "chikungunya.csv.gz": ("chikungunya.csv.gz",),
    "climate.csv.gz": ("climate.csv.gz",),
    # The FTP currently uses forecasting_climate.csv.gz, while the challenge
    # documentation refers to climate_forecast.csv.gz.
    "climate_forecast.csv.gz": (
        "climate_forecast.csv.gz",
        "forecasting_climate.csv.gz",
    ),
    "datasus_population_2001_2025.csv.gz": (
        "datasus_population_2001_2025.csv.gz",
    ),
    "environ_vars.csv.gz": ("environ_vars.csv.gz",),
    "ocean_climate_oscillations.csv.gz": (
        "ocean_climate_oscillations.csv.gz",
        "enso.csv.gz",
    ),
    "shape_muni.gpkg": ("shape_muni.gpkg",),
    "shape_regional_health.gpkg": ("shape_regional_health.gpkg",),
    "shape_macroregional_health.gpkg": ("shape_macroregional_health.gpkg",),
    "map_regional_health.csv": ("map_regional_health.csv",),
}
FTP_FILES = tuple(FTP_FILE_ALIASES)
ESSENTIAL_FTP_FILES = (
    "dengue.csv.gz",
    "climate.csv.gz",
    "climate_forecast.csv.gz",
    "datasus_population_2001_2025.csv.gz",
)

QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
EXCLUDED_UFS = ("ES",)
FIRST_ORIGIN_YEAR = 2010
DENGUE_CITY_CODES = (
    2931350,
    2933307,
    2302503,
    3119401,
    3549805,
    3541406,
    1200401,
    1200203,
    1716109,
    4113700,
    4103701,
    4104808,
    5201405,
    5102637,
    5215231,
)
CHIKUNGUNYA_CITY_CODES = (
    2211001,
    2931350,
    3143302,
    3119401,
    1721000,
    1716109,
    4104808,
    4219507,
    5103403,
    5102637,
)


@dataclass(frozen=True)
class ForecastRound:
    name: str
    origin_year: int


@dataclass(frozen=True)
class Challenge:
    name: str
    disease: str
    level: str
    city_codes: tuple[int, ...] = ()
    model_names: tuple[str, ...] = ()


CHALLENGES = (
    Challenge("dengue_uf", "dengue", "uf", model_names=("xgb_residual",)),
    Challenge(
        "dengue_city",
        "dengue",
        "city",
        DENGUE_CITY_CODES,
        model_names=("xgb_residual",),
    ),
    Challenge(
        "chikungunya_uf",
        "chikungunya",
        "uf",
        model_names=("xgb_quantile_log1p",),
    ),
    Challenge(
        "chikungunya_city",
        "chikungunya",
        "city",
        CHIKUNGUNYA_CITY_CODES,
        model_names=("xgb_quantile",),
    ),
)


ROUNDS = (
    ForecastRound("validation_1", 2022),
    ForecastRound("validation_2", 2023),
    ForecastRound("validation_3", 2024),
    ForecastRound("validation_4", 2025),
    ForecastRound("final", 2026),
)


def ensure_directories() -> None:
    for path in (
        FTP_DIR,
        EXTERNAL_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        SUBMISSIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
