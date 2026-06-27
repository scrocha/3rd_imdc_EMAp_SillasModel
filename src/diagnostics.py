from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda import Moran
from libpysal.weights import KNN

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .backtest import load_inputs
from .build_dataset import write_outputs
from .config import CHALLENGES, FTP_DIR, PROCESSED_DIR, RESULTS_DIR, Challenge
from .features import (
    build_origin_matrix,
    feature_columns,
)
from .utils import LOGGER, setup_logging, timed_step

TARGET_LAGS = (1, 2, 3, 4, 6, 8, 12, 26, 52, 104)
CLIMATE_COLUMNS = (
    "temp_med",
    "precip_tot",
    "rel_humid_med",
    "enso",
    "iod",
    "pdo",
)


def _directory(challenge: Challenge):
    path = RESULTS_DIR / "diagnostics" / challenge.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _geometries(challenge: Challenge) -> gpd.GeoDataFrame:
    municipalities = gpd.read_file(FTP_DIR / "shape_muni.gpkg")
    municipalities["geometry"] = municipalities.geometry.make_valid()
    if challenge.level == "city":
        return municipalities.loc[
            municipalities["geocode"].isin(challenge.city_codes)
        ].rename(
            columns={"geocode": "location_code", "geocode_name": "location"}
        )[["location_code", "location", "geometry"]]
    return municipalities.dissolve(
        by=["uf_code", "uf"], as_index=False
    ).rename(columns={"uf_code": "location_code", "uf": "location"})[
        ["location_code", "location", "geometry"]
    ]


def run_temporal_diagnostics(
    challenge: Challenge, panel: pd.DataFrame
) -> None:
    try:
        from statsmodels.tsa.stattools import pacf
    except ModuleNotFoundError:
        pacf = None

    acf_rows, pacf_rows, cross_rows = [], [], []
    for (code, location), group in panel.groupby(
        ["location_code", "location"], observed=True
    ):
        group = group.sort_values("date")
        values = group["casos"].astype(float)
        nlags = min(104, max(len(values) // 2 - 1, 1))
        partial = pacf(values, nlags=nlags, method="ywm") if pacf else None
        for lag in TARGET_LAGS:
            if lag > nlags:
                continue
            acf_rows.append(
                {
                    "location_code": code,
                    "location": location,
                    "lag": lag,
                    "autocorrelation": values.autocorr(lag=lag),
                }
            )
            if partial is not None:
                pacf_rows.append(
                    {
                        "location_code": code,
                        "location": location,
                        "lag": lag,
                        "partial_autocorrelation": partial[lag],
                    }
                )
        for variable in CLIMATE_COLUMNS:
            if variable not in group:
                continue
            maximum_lag = 24 if variable in {"enso", "iod", "pdo"} else 16
            for lag in range(maximum_lag + 1):
                cross_rows.append(
                    {
                        "location_code": code,
                        "location": location,
                        "variable": variable,
                        "lag": lag,
                        "correlation": group[variable].shift(lag).corr(values),
                    }
                )
    directory = _directory(challenge)
    acf = pd.DataFrame(acf_rows)
    acf.to_csv(directory / "acf_target.csv", index=False)
    pd.DataFrame(pacf_rows).to_csv(directory / "pacf_target.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(
        directory / "cross_correlation_climate_target.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for location in (
        panel.groupby("location", observed=True)["casos"]
        .mean()
        .nlargest(6)
        .index
    ):
        table = acf.loc[acf["location"].eq(location)]
        ax.plot(
            table["lag"], table["autocorrelation"], marker="o", label=location
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Lag (semanas)", ylabel="ACF", title=challenge.name)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(directory / "acf_target.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_covariate_diagnostics(
    challenge: Challenge,
    panel: pd.DataFrame,
    mapbiomas: pd.DataFrame | None,
) -> None:
    matrix = build_origin_matrix(
        panel,
        range(2022, 2026),
        mapbiomas,
        include_target=True,
    )
    LOGGER.info(
        "%s | matriz diagnóstica | linhas=%s colunas=%d memória=%.1f MB",
        challenge.name,
        f"{len(matrix):,}",
        len(matrix.columns),
        matrix.memory_usage(deep=True).sum() / 1024**2,
    )
    numeric = matrix[feature_columns(matrix)].replace(
        [np.inf, -np.inf], np.nan
    )
    directory = _directory(challenge)
    numeric.corr().to_csv(directory / "feature_correlation.csv")
    sample = (
        numeric.dropna(axis=1, how="all").fillna(numeric.median()).fillna(0)
    )
    sample = sample.loc[:, sample.nunique() > 1]
    if len(sample) > 10_000:
        sample = sample.sample(10_000, random_state=42)
    standardized = (sample - sample.mean()) / sample.std(ddof=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = standardized.corr().to_numpy(dtype=float)
        inverse = np.linalg.pinv(correlation, hermitian=True)
    vif = pd.DataFrame(
        {
            "feature": standardized.columns,
            "vif": np.maximum(np.diag(inverse), 1),
        }
    ).sort_values("vif", ascending=False)
    vif.to_csv(directory / "vif_report.csv", index=False)
    vif.loc[vif["vif"].gt(30), ["feature", "vif"]].to_csv(
        directory / "features_to_review.csv", index=False
    )


def run_extreme_diagnostics(challenge: Challenge, panel: pd.DataFrame) -> None:
    output = panel.sort_values(["location_code", "week", "date"]).copy()
    grouped = output.groupby(["location_code", "week"], observed=True)["casos"]
    for quantile in (0.75, 0.90, 0.95):
        label = int(quantile * 100)
        output[f"historical_q{label}"] = grouped.transform(
            lambda values: values.shift().expanding().quantile(quantile)
        )
        output[f"excess_q{label}"] = (
            output["casos"] - output[f"historical_q{label}"]
        ).clip(lower=0)
    output["is_extreme"] = output["casos"].gt(output["historical_q90"])
    columns = [
        "location_code",
        "location",
        "date",
        "epiweek",
        "historical_q75",
        "historical_q90",
        "historical_q95",
        "excess_q75",
        "excess_q90",
        "excess_q95",
        "is_extreme",
    ]
    output[columns].to_parquet(
        PROCESSED_DIR / f"extreme_features_{challenge.name}.parquet",
        index=False,
    )
    output.groupby(["location_code", "location", "week"], observed=True)[
        ["historical_q75", "historical_q90", "historical_q95"]
    ].last().reset_index().to_csv(
        _directory(challenge) / "extreme_thresholds.csv", index=False
    )


def run_spatial_diagnostics(challenge: Challenge, panel: pd.DataFrame) -> None:
    geometry = (
        _geometries(challenge)
        .sort_values("location_code")
        .reset_index(drop=True)
    )
    metric = geometry.to_crs("EPSG:5880")
    points = np.column_stack(
        (metric.geometry.centroid.x, metric.geometry.centroid.y)
    )
    weights = KNN.from_array(points, k=min(3, len(points) - 1))
    weights.transform = "R"
    order = geometry["location_code"].astype(int)
    rows = []
    for epiweek in sorted(panel["epiweek"].unique())[::13]:
        week = panel.loc[panel["epiweek"].eq(epiweek)].set_index(
            "location_code"
        )
        for variable in ("casos", "incidence_100k", "temp_med", "precip_tot"):
            values = pd.to_numeric(
                week[variable].reindex(order), errors="coerce"
            )
            values = values.fillna(values.mean())
            if values.nunique() < 2:
                continue
            statistic = Moran(values.to_numpy(), weights, permutations=0)
            rows.append(
                {
                    "epiweek": int(epiweek),
                    "variable": variable,
                    "moran_i": statistic.I,
                    "p_normal": statistic.p_norm,
                }
            )
    pd.DataFrame(rows).to_csv(
        _directory(challenge) / "moran_variables.csv", index=False
    )


def _zero_summary(
    frame: pd.DataFrame, keys: list[str], level: str
) -> pd.DataFrame:
    weekly = (
        frame.groupby([*keys, "date"], observed=True)["casos"]
        .sum()
        .reset_index()
        if keys
        else frame.groupby("date", observed=True)["casos"].sum().reset_index()
    )
    if not keys:
        weekly["scope"] = "Brazil"
        keys = ["scope"]
    summary = (
        weekly.groupby(keys, observed=True)["casos"]
        .agg(
            weeks="size",
            zero_weeks=lambda values: values.eq(0).sum(),
            total_cases="sum",
            mean_cases="mean",
            median_cases="median",
            p95_cases=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    summary["zero_rate"] = summary["zero_weeks"] / summary["weeks"]
    summary.insert(0, "level", level)
    return summary


def run_zero_diagnostics() -> None:
    root = RESULTS_DIR / "diagnostics" / "zeros"
    root.mkdir(parents=True, exist_ok=True)
    for disease in ("dengue", "chikungunya"):
        path = PROCESSED_DIR / f"panel_{disease}_muni_week.parquet"
        with timed_step(f"diagnostics | zeros | carregar {disease}"):
            panel = pd.read_parquet(path)
        LOGGER.info(
            "diagnostics | zeros | %s | linhas=%s", disease, f"{len(panel):,}"
        )
        summaries = [
            _zero_summary(panel, ["geocode", "uf"], "city"),
            _zero_summary(panel, ["regional_geocode", "uf"], "health_region"),
            _zero_summary(
                panel, ["macroregional_geocode", "uf"], "health_macroregion"
            ),
            _zero_summary(panel, ["uf_code", "uf"], "uf"),
            _zero_summary(panel, [], "country"),
        ]
        pd.concat(summaries, ignore_index=True, sort=False).to_csv(
            root / f"{disease}_zero_summary.csv", index=False
        )
        target = panel.loc[panel["target_city"]].copy()
        target["region_positive_city_zero"] = target["casos"].eq(0) & target[
            "regional_cases"
        ].gt(0)
        target["population_allocated_cases"] = (
            target["regional_cases"] * target["population_share_region"]
        )
        target_summary = (
            target.groupby(["geocode", "uf"], observed=True)
            .agg(
                weeks=("casos", "size"),
                zero_rate=("casos", lambda values: values.eq(0).mean()),
                region_positive_city_zero_rate=(
                    "region_positive_city_zero",
                    "mean",
                ),
                population_share_region=("population_share_region", "median"),
            )
            .reset_index()
        )
        target_summary.merge(
            target.assign(
                allocation_absolute_error=lambda frame: (
                    frame["casos"] - frame["population_allocated_cases"]
                ).abs()
            )
            .groupby("geocode", observed=True)["allocation_absolute_error"]
            .mean()
            .rename("population_allocation_mae"),
            on="geocode",
        ).to_csv(
            root / f"{disease}_target_city_zero_analysis.csv", index=False
        )


def run_all_diagnostics() -> None:
    if not all(
        (PROCESSED_DIR / f"panel_{challenge.name}_week.parquet").exists()
        for challenge in CHALLENGES
    ):
        write_outputs()
    with timed_step("diagnostics | zeros"):
        run_zero_diagnostics()
    for challenge in CHALLENGES:
        with timed_step(f"diagnostics | {challenge.name} | carregar"):
            panel, mapbiomas = load_inputs(challenge)
        LOGGER.info(
            "diagnostics | %s | painel=%s linhas, %d locais",
            challenge.name,
            f"{len(panel):,}",
            panel["location_code"].nunique(),
        )
        with timed_step(f"diagnostics | {challenge.name} | temporal"):
            run_temporal_diagnostics(challenge, panel)
        with timed_step(f"diagnostics | {challenge.name} | covariáveis"):
            run_covariate_diagnostics(challenge, panel, mapbiomas)
        with timed_step(f"diagnostics | {challenge.name} | extremos"):
            run_extreme_diagnostics(challenge, panel)
        with timed_step(f"diagnostics | {challenge.name} | espacial"):
            run_spatial_diagnostics(challenge, panel)


def main() -> None:
    setup_logging()
    run_all_diagnostics()


if __name__ == "__main__":
    main()
