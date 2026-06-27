from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    ESSENTIAL_FTP_FILES,
    FTP_DIR,
    INTERIM_DIR,
    ensure_directories,
)
from .utils import write_json

SCHEMAS = {
    "dengue.csv.gz": {
        "required": {"date", "epiweek", "geocode", "casos", "uf", "uf_code"},
        "key": ["geocode", "epiweek"],
    },
    "chikungunya.csv.gz": {
        "required": {"date", "epiweek", "geocode", "casos", "uf", "uf_code"},
        "key": ["geocode", "epiweek"],
    },
    "climate.csv.gz": {
        "required": {"date", "epiweek", "geocode", "temp_med", "precip_med"},
        "key": ["geocode", "epiweek"],
    },
    "climate_forecast.csv.gz": {
        "required": {
            "geocode",
            "reference_month",
            "forecast_months_ahead",
            "temp_med",
            "precip_tot",
        },
        "key": ["geocode", "reference_month", "forecast_months_ahead"],
    },
    "datasus_population_2001_2025.csv.gz": {
        "required": {"geocode", "year", "population"},
        "key": ["geocode", "year"],
    },
}


def validate_file(path: Path) -> dict[str, object]:
    specification = SCHEMAS[path.name]
    frame = pd.read_csv(path)
    missing_columns = sorted(specification["required"] - set(frame.columns))
    if path.name == "climate_forecast.csv.gz" and not {
        "rel_humid_med",
        "umid_med",
    }.intersection(frame.columns):
        missing_columns.append("rel_humid_med|umid_med")
    duplicate_rows = int(frame.duplicated(specification["key"]).sum())
    report: dict[str, object] = {
        "path": str(path),
        "rows": len(frame),
        "missing_columns": missing_columns,
        "duplicate_keys": duplicate_rows,
        "valid": not missing_columns and duplicate_rows == 0,
    }
    if "epiweek" in frame:
        weeks = pd.to_numeric(frame["epiweek"], errors="coerce")
        report.update(
            min_epiweek=int(weeks.min()), max_epiweek=int(weeks.max())
        )
    return report


def validate_all(strict: bool = True) -> dict[str, dict[str, object]]:
    ensure_directories()
    missing_files = [
        name for name in ESSENTIAL_FTP_FILES if not (FTP_DIR / name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Execute o download; faltam: {missing_files}")
    report = {name: validate_file(FTP_DIR / name) for name in SCHEMAS}
    write_json(INTERIM_DIR / "raw_schema_report.json", report)
    invalid = [name for name, item in report.items() if not item["valid"]]
    if strict and invalid:
        raise ValueError(f"Falha na validação dos arquivos: {invalid}")
    return report


def main() -> None:
    validate_all()
    print("Schemas e chaves validados.")


if __name__ == "__main__":
    main()
