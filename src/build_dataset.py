from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.parquet import read_schema

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    CHALLENGES,
    CHIKUNGUNYA_CITY_CODES,
    DENGUE_CITY_CODES,
    EXCLUDED_UFS,
    FTP_DIR,
    PROCESSED_DIR,
    ensure_directories,
)
from .download import ensure_downloads
from .mapbiomas import build_mapbiomas, download_mapbiomas
from .spatial import build_uf_geometries, build_uf_weights
from .utils import LOGGER, setup_logging, timed_step
from .validate_raw_data import validate_all

CLIMATE_COLUMNS = (
    "temp_min",
    "temp_med",
    "temp_max",
    "precip_min",
    "precip_med",
    "precip_max",
    "precip_tot",
    "pressure_min",
    "pressure_med",
    "pressure_max",
    "rel_humid_min",
    "rel_humid_med",
    "rel_humid_max",
    "thermal_range",
    "rainy_days",
)
FLAG_COLUMNS = tuple(
    f"{kind}_{index}" for kind in ("train", "target") for index in range(1, 5)
)


def _read(name: str, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(FTP_DIR / name, **kwargs)


def _normalize_epi(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["epiweek"] = pd.to_numeric(
        output["epiweek"], errors="coerce"
    ).astype("Int64")
    output["geocode"] = pd.to_numeric(
        output["geocode"], errors="coerce"
    ).astype("Int64")
    output["year"] = output["epiweek"] // 100
    output["week"] = output["epiweek"] % 100
    return output


def load_population() -> pd.DataFrame:
    population = _read("datasus_population_2001_2025.csv.gz")
    for column in ("geocode", "year", "population"):
        population[column] = pd.to_numeric(population[column], errors="coerce")
    population = population.dropna(subset=["geocode", "year"]).copy()
    population["geocode"] = population["geocode"].astype("Int64")
    population["year"] = population["year"].astype(int)
    population["population"] = population["population"].clip(lower=0)
    return population.drop_duplicates(["geocode", "year"], keep="last")


def _common_covariates() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    climate = _normalize_epi(_read("climate.csv.gz"))
    if "precip_tot" not in climate and "precip_med" in climate:
        climate["precip_tot"] = (
            pd.to_numeric(climate["precip_med"], errors="coerce") * 168
        )
    available = [column for column in CLIMATE_COLUMNS if column in climate]
    climate[available] = climate[available].apply(
        pd.to_numeric, errors="coerce"
    )
    climate = climate[["geocode", "epiweek", *available]].drop_duplicates(
        ["geocode", "epiweek"], keep="last"
    )

    environment = _read("environ_vars.csv.gz")
    environment["geocode"] = pd.to_numeric(
        environment["geocode"], errors="coerce"
    ).astype("Int64")
    environment = environment[["geocode", "koppen", "biome"]].drop_duplicates(
        "geocode"
    )

    ocean = _read("ocean_climate_oscillations.csv.gz")
    ocean["epiweek"] = pd.to_numeric(ocean["epiweek"], errors="coerce").astype(
        "Int64"
    )
    ocean = ocean[["epiweek", "enso", "iod", "pdo"]].drop_duplicates("epiweek")

    municipalities = pd.read_csv(FTP_DIR / "map_regional_health.csv")
    municipalities["geocode"] = pd.to_numeric(
        municipalities["geocode"], errors="coerce"
    ).astype("Int64")
    name_column = next(
        (
            column
            for column in ("geocode_name", "name_muni", "municipality")
            if column in municipalities
        ),
        None,
    )
    names_table = (
        municipalities[["geocode", name_column]].rename(
            columns={name_column: "location"}
        )
        if name_column
        else municipalities[["geocode"]].assign(
            location=lambda frame: frame["geocode"].astype(str)
        )
    )
    return climate, environment, ocean, names_table.drop_duplicates("geocode")


def build_municipal_panel(
    disease: str,
    climate: pd.DataFrame,
    environment: pd.DataFrame,
    ocean: pd.DataFrame,
) -> pd.DataFrame:
    cases = _normalize_epi(_read(f"{disease}.csv.gz"))
    cases["casos"] = pd.to_numeric(cases["casos"], errors="coerce").clip(
        lower=0
    )
    for column in FLAG_COLUMNS:
        if column in cases:
            cases[column] = cases[column].fillna(False).astype(bool)
    panel = cases.merge(
        climate, on=["geocode", "epiweek"], how="left", validate="one_to_one"
    )
    population = load_population()
    panel = panel.merge(
        population, on=["geocode", "year"], how="left", validate="many_to_one"
    )
    latest_population = (
        population.sort_values("year").groupby("geocode")["population"].last()
    )
    panel["population"] = panel["population"].fillna(
        panel["geocode"].map(latest_population)
    )
    panel["incidence_100k"] = np.where(
        panel["population"].gt(0),
        panel["casos"] / panel["population"] * 100_000,
        np.nan,
    )
    panel = panel.merge(
        environment, on="geocode", how="left", validate="many_to_one"
    )
    panel = panel.merge(
        ocean, on="epiweek", how="left", validate="many_to_one"
    )
    mapping = _read("map_regional_health.csv")
    mapping["geocode"] = pd.to_numeric(
        mapping["geocode"], errors="coerce"
    ).astype("Int64")
    mapping_columns = [
        "geocode",
        "regional_name",
        "macroregional_name",
        "uf_name",
    ]
    panel = panel.merge(
        mapping[mapping_columns].drop_duplicates("geocode"),
        on="geocode",
        how="left",
        validate="many_to_one",
    )
    for key, prefix in (
        ("regional_geocode", "regional"),
        ("macroregional_geocode", "macroregional"),
        ("uf_code", "uf"),
    ):
        groups = panel.groupby([key, "date"], observed=True)
        panel[f"{prefix}_cases"] = groups["casos"].transform("sum")
        panel[f"{prefix}_population"] = groups["population"].transform("sum")
    panel["population_share_region"] = np.where(
        panel["regional_population"].gt(0),
        panel["population"] / panel["regional_population"],
        np.nan,
    )
    target_codes = set(DENGUE_CITY_CODES) | set(CHIKUNGUNYA_CITY_CODES)
    panel["target_city"] = panel["geocode"].isin(target_codes)
    return panel.sort_values(["geocode", "date"]).reset_index(drop=True)


def build_area_panel(
    municipal: pd.DataFrame,
    code_column: str,
    name_column: str,
    level: str,
) -> pd.DataFrame:
    identifiers = [
        code_column,
        name_column,
        "date",
        "epiweek",
        "year",
        "week",
        "disease",
    ]
    climate_columns = [
        column for column in CLIMATE_COLUMNS if column in municipal
    ]
    grouped = municipal.groupby(
        identifiers, observed=True, sort=False, dropna=False
    )
    aggregations: dict[str, str] = {
        "casos": "sum",
        "population": "sum",
        "geocode": "nunique",
        **{column: "max" for column in FLAG_COLUMNS},
        **{column: "mean" for column in ("enso", "iod", "pdo")},
    }
    panel = grouped.agg(aggregations).rename(
        columns={"geocode": "municipalities_reporting"}
    )
    weights = pd.to_numeric(municipal["population"], errors="coerce")
    group_keys = [municipal[column] for column in identifiers]
    for column in climate_columns:
        values = pd.to_numeric(municipal[column], errors="coerce")
        valid_weights = weights.where(values.notna() & weights.gt(0))
        numerator = (
            (values * valid_weights)
            .groupby(group_keys, observed=True, sort=False, dropna=False)
            .sum(min_count=1)
        )
        denominator = valid_weights.groupby(
            group_keys, observed=True, sort=False, dropna=False
        ).sum(min_count=1)
        weighted = numerator / denominator
        panel[column] = weighted.fillna(grouped[column].mean())
    panel = panel.reset_index()
    panel = panel.rename(
        columns={code_column: "location_code", name_column: "location"}
    )
    panel["level"] = level
    panel["incidence_100k"] = np.where(
        panel["population"].gt(0),
        panel["casos"] / panel["population"] * 100_000,
        np.nan,
    )
    return panel.sort_values(["location_code", "date"]).reset_index(drop=True)


def _valid_panel(path, required: set[str]) -> bool:
    if not path.exists():
        return False
    return required.issubset(read_schema(path).names)


def _standardize_challenge(
    challenge,
    municipal: pd.DataFrame,
    uf_panel: pd.DataFrame,
    names: pd.DataFrame,
) -> pd.DataFrame:
    if challenge.level == "uf":
        panel = uf_panel
    else:
        panel = municipal.loc[
            municipal["geocode"].isin(challenge.city_codes)
        ].merge(names, on="geocode", how="left", validate="many_to_one")
        panel = panel.rename(columns={"geocode": "location_code"})
    panel = panel.copy()
    panel["challenge"] = challenge.name
    panel["level"] = challenge.level
    return panel.sort_values(["location_code", "date"]).reset_index(drop=True)


def write_outputs() -> None:
    ensure_directories()
    with timed_step("dataset | downloads FTP"):
        ensure_downloads()
    with timed_step("dataset | downloads MapBiomas"):
        download_mapbiomas()
    mapbiomas_path = PROCESSED_DIR / "mapbiomas_muni_year.parquet"
    mapbiomas_years = (
        set(pd.read_parquet(mapbiomas_path, columns=["year"])["year"].unique())
        if mapbiomas_path.exists()
        else set()
    )
    if mapbiomas_years != set(range(2010, 2025)):
        with timed_step("dataset | processar MapBiomas"):
            build_mapbiomas()
    with timed_step("dataset | validar dados brutos"):
        validate_all()
    with timed_step("dataset | carregar covariáveis"):
        climate, environment, ocean, names = _common_covariates()
    for disease in ("dengue", "chikungunya"):
        LOGGER.info("dataset | doença=%s", disease)
        municipal_path = PROCESSED_DIR / f"panel_{disease}_muni_week.parquet"
        if _valid_panel(
            municipal_path,
            {
                "regional_cases",
                "macroregional_cases",
                "uf_cases",
                "population_share_region",
                "target_city",
            },
        ):
            municipal = pd.read_parquet(municipal_path)
        else:
            municipal = build_municipal_panel(
                disease, climate, environment, ocean
            )
            municipal.to_parquet(municipal_path, index=False)

        area_panels = {}
        for level, code, name in (
            ("uf", "uf_code", "uf"),
            ("regional", "regional_geocode", "regional_name"),
            ("macroregional", "macroregional_geocode", "macroregional_name"),
        ):
            path = PROCESSED_DIR / f"panel_{disease}_{level}_week.parquet"
            if _valid_panel(
                path, {"location_code", "location", "level", "temp_min"}
            ):
                area_panels[level] = pd.read_parquet(path)
            else:
                source = municipal.loc[~municipal["uf"].isin(EXCLUDED_UFS)]
                area_panels[level] = build_area_panel(
                    source, code, name, level
                )
                area_panels[level].to_parquet(path, index=False)
        uf_panel = area_panels["uf"]
        for challenge in (
            item for item in CHALLENGES if item.disease == disease
        ):
            panel = _standardize_challenge(
                challenge, municipal, uf_panel, names
            )
            expected = (
                (27 - len(EXCLUDED_UFS))
                if challenge.level == "uf"
                else len(challenge.city_codes)
            )
            if panel["location_code"].nunique() != expected:
                raise ValueError(
                    f"{challenge.name}: esperado {expected} locais, "
                    f"encontrado {panel['location_code'].nunique()}."
                )
            panel.to_parquet(
                PROCESSED_DIR / f"panel_{challenge.name}_week.parquet",
                index=False,
            )
            print(
                f"{challenge.name}: {len(panel):,} linhas, "
                f"{panel['location_code'].nunique()} locais."
            )
        del municipal, area_panels, uf_panel
    build_uf_weights(build_uf_geometries())


def main() -> None:
    setup_logging()
    write_outputs()


if __name__ == "__main__":
    main()
