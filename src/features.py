from __future__ import annotations

import numpy as np
import pandas as pd
from epiweeks import Week

from .config import FIRST_ORIGIN_YEAR
from .spatial import load_neighbor_map
from .utils import LOGGER, season_epiweeks

CASE_LAGS = (1, 2, 3, 4, 6, 8, 12, 26, 52)
ROLLING_WINDOWS = (4, 8, 12)
CLIMATE_COLUMNS = (
    "temp_med",
    "precip_tot",
    "rel_humid_med",
)
OCEAN_COLUMNS = ("enso", "iod", "pdo")
FORECAST_GAP_WEEKS = 16
FORECAST_WEEKS = 52


def _season_position(week: int) -> int:
    return week - 40 if week >= 41 else week + 12


def official_cutoff_date(origin_year: int) -> pd.Timestamp:
    return pd.Timestamp(Week(origin_year, 25).startdate())


def _origin_features(
    history: pd.DataFrame, cutoff: pd.Timestamp
) -> dict[str, float]:
    known = history.loc[history["date"].le(cutoff)].sort_values("date")

    # Extract cases array to avoid Pandas Series index overhead
    cases_arr = known["casos"].to_numpy(dtype=float)
    n_cases = len(cases_arr)

    output: dict[str, float] = {
        "origin_year": float(cutoff.year),
        "origin_week": float(Week.fromdate(cutoff.date()).week),
        "origin_distance_ew25": float(
            min(
                abs(Week.fromdate(cutoff.date()).week - 25),
                52 - abs(Week.fromdate(cutoff.date()).week - 25),
            )
        ),
    }

    for lag in CASE_LAGS:
        output[f"cases_lag_{lag}"] = (
            float(cases_arr[-lag]) if n_cases >= lag else np.nan
        )

    for window in ROLLING_WINDOWS:
        if n_cases >= window:
            tail = cases_arr[-window:]
            output[f"cases_roll_mean_{window}"] = float(np.mean(tail))
            output[f"cases_roll_max_{window}"] = float(np.max(tail))
        else:
            output[f"cases_roll_mean_{window}"] = np.nan
            output[f"cases_roll_max_{window}"] = np.nan

    for window in (4, 8):
        if n_cases >= window:
            tail = np.log1p(cases_arr[-window:])
            output[f"cases_log_trend_{window}"] = float(
                np.polyfit(np.arange(len(tail)), tail, 1)[0]
            )
        else:
            output[f"cases_log_trend_{window}"] = np.nan

    recent_tail = cases_arr[-8:] if n_cases >= 8 else cases_arr
    recent_mean = np.mean(recent_tail) if len(recent_tail) > 0 else 0.0

    recent_weeks = known["week"].tail(8).tolist()
    historical_recent_df = known.loc[
        known["week"].isin(recent_weeks)
        & known["date"].lt(
            known["date"].tail(8).min()
            if n_cases >= 8
            else known["date"].min()
        )
    ]
    historical_mean = (
        historical_recent_df["casos"].mean()
        if not historical_recent_df.empty
        else 0.0
    )
    output["recent_vs_historical"] = float(
        (recent_mean + 1) / (historical_mean + 1)
    )

    # --- Epidemic dynamics features ---
    # Log-acceleration: second derivative of log1p(cases) — captures epidemic phase transitions
    if n_cases >= 4:
        log_cases = np.log1p(cases_arr[-8:] if n_cases >= 8 else cases_arr)
        first_diff = np.diff(log_cases)
        if len(first_diff) >= 2:
            second_diff = np.diff(first_diff)
            output["log_acceleration_4"] = (
                float(np.mean(second_diff[-4:]))
                if len(second_diff) >= 4
                else float(second_diff[-1])
            )
        else:
            output["log_acceleration_4"] = 0.0
        output["log_trend_sign"] = (
            float(np.sign(first_diff[-1])) if len(first_diff) > 0 else 0.0
        )
    else:
        output["log_acceleration_4"] = np.nan
        output["log_trend_sign"] = np.nan

    # Momentum: ratio between last 2 weeks and prior 2 weeks
    if n_cases >= 4:
        output["cases_momentum_4"] = float(
            (np.mean(cases_arr[-2:]) + 1) / (np.mean(cases_arr[-4:-2]) + 1)
        )
    else:
        output["cases_momentum_4"] = np.nan

    extra_geo_cols = (
        "regional_cases",
        "macroregional_cases",
        "uf_cases",
        "regional_cases_max",
        "macroregional_cases_max",
        "uf_cases_max",
        "regional_incidence_max",
        "macroregional_incidence_max",
        "uf_incidence_max",
        "macroregion_cases_sum",
        "macroregion_cases_max",
        "macroregion_incidence_mean",
        "macroregion_incidence_max",
    )

    for column in extra_geo_cols:
        if column not in known:
            continue
        values = known[column].to_numpy(dtype=float)
        n_val = len(values)
        for lag in (1, 2, 4, 8, 12, 52):
            output[f"{column}_lag_{lag}"] = (
                float(values[-lag]) if n_val >= lag else np.nan
            )
        output[f"{column}_roll_mean_4"] = (
            float(np.mean(values[-4:])) if n_val >= 4 else np.nan
        )
        output[f"{column}_roll_mean_8"] = (
            float(np.mean(values[-8:])) if n_val >= 8 else np.nan
        )
        output[f"{column}_roll_max_4"] = (
            float(np.max(values[-4:])) if n_val >= 4 else np.nan
        )

    if "population_share_region" in known:
        output["population_share_region"] = float(
            known["population_share_region"].iloc[-1]
        )

    for column in CLIMATE_COLUMNS:
        if column not in known:
            continue
        values = known[column].to_numpy(dtype=float)
        n_val = len(values)
        for window in (4, 8, 12):
            output[f"{column}_roll_mean_{window}"] = (
                float(np.mean(values[-window:])) if n_val >= window else np.nan
            )

    # Weekly climate lags
    for column in ("temp_med", "precip_tot", "rel_humid_med"):
        if column not in known:
            continue
        values = known[column].to_numpy(dtype=float)
        n_val = len(values)
        for lag in (1, 2, 4, 8):
            output[f"{column}_lag_{lag}"] = (
                float(values[-lag]) if n_val >= lag else np.nan
            )

    # Aggregated climate features: last_month, last_semester, last_year
    # Replaces the previous 36 monthly lag features (12 per variable)
    # to reduce colinearity while retaining multi-scale climate signal.
    for col, prefix, agg in [
        ("temp_med", "temp", "mean"),
        ("rel_humid_med", "humid", "mean"),
        ("precip_tot", "precip", "sum"),
    ]:
        if col not in known:
            continue
        vals = known[col].to_numpy(dtype=float)
        n = len(vals)
        if n >= 4:
            output[f"{prefix}_last_month"] = float(
                np.sum(vals[-4:]) if agg == "sum" else np.mean(vals[-4:])
            )
        else:
            output[f"{prefix}_last_month"] = np.nan
        if n >= 24:
            semester = vals[-24:-4]
            if agg == "sum":
                output[f"{prefix}_last_semester"] = float(
                    np.mean(
                        [
                            np.sum(semester[i * 4 : (i + 1) * 4])
                            for i in range(5)
                        ]
                    )
                )
            else:
                output[f"{prefix}_last_semester"] = float(np.mean(semester))
        elif n >= 8:
            output[f"{prefix}_last_semester"] = float(np.mean(vals[-8:-4]))
        else:
            output[f"{prefix}_last_semester"] = np.nan
        if n >= 48:
            year_ago = vals[-48:-24]
            if agg == "sum":
                output[f"{prefix}_last_year"] = float(
                    np.mean(
                        [
                            np.sum(year_ago[i * 4 : (i + 1) * 4])
                            for i in range(6)
                        ]
                    )
                )
            else:
                output[f"{prefix}_last_year"] = float(np.mean(year_ago))
        elif n >= 28:
            output[f"{prefix}_last_year"] = float(np.mean(vals[-28:-24]))
        else:
            output[f"{prefix}_last_year"] = np.nan

    # Monthly Ocean/ENSO lags (1 to 12 months)
    for column in OCEAN_COLUMNS:
        if column not in known:
            continue
        values = known[column].to_numpy(dtype=float)
        n_val = len(values)
        for m in range(1, 13):
            pos = 4 * m
            output[f"{column}_month_lag_{m}"] = (
                float(values[-pos]) if n_val >= pos else np.nan
            )

    for column in known.columns:
        if column.startswith(("share_biome_", "share_koppen_")):
            output[column] = float(known[column].iloc[-1])
    return output


def _prepare_extreme_history(history: pd.DataFrame) -> pd.DataFrame:
    output = history.sort_values("date").copy()
    output["historical_q90"] = output.groupby("week", observed=True)[
        "casos"
    ].transform(lambda values: values.shift().expanding().quantile(0.90))
    output["is_extreme"] = output["casos"].gt(output["historical_q90"])
    return output


def _extreme_origin_features(
    extreme_history: pd.DataFrame, cutoff: pd.Timestamp
) -> dict[str, float]:
    known = extreme_history.loc[extreme_history["date"].le(cutoff)]
    recent = known.tail(12)
    last = known.iloc[-1]
    return {
        "is_above_q90_at_cutoff": float(last["is_extreme"]),
        "cases_excess_q90_at_cutoff": float(
            max(last["casos"] - last["historical_q90"], 0)
        )
        if pd.notna(last["historical_q90"])
        else np.nan,
        "extreme_weeks_last_8": float(recent.tail(8)["is_extreme"].sum()),
        "extreme_weeks_last_12": float(recent["is_extreme"].sum()),
    }


def _prepare_neighbor_history(
    panel: pd.DataFrame,
    neighbor_map: dict[int, list[int]],
    location_code: int,
) -> pd.DataFrame:
    neighbors = neighbor_map.get(int(location_code), [])
    known = panel.loc[panel["location_code"].isin(neighbors)]
    if known.empty:
        return pd.DataFrame()
    return known.groupby("date", observed=True).agg(
        neighbor_cases=("casos", "mean"),
        neighbor_cases_max=("casos", "max"),
        neighbor_incidence=("incidence_100k", "mean"),
        neighbor_incidence_max=("incidence_100k", "max"),
        neighbor_temp=("temp_med", "mean"),
        neighbor_precip=("precip_tot", "mean"),
        neighbor_humidity=("rel_humid_med", "mean"),
    )


def _neighbor_origin_features(
    neighbor_history: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> dict[str, float]:
    output: dict[str, float] = {}
    if neighbor_history.empty:
        return output
    weekly = neighbor_history.loc[neighbor_history.index <= cutoff]
    if weekly.empty:
        return output
    for lag in (1, 2, 4):
        output[f"neighbor_cases_lag_{lag}"] = (
            float(weekly["neighbor_cases"].iloc[-lag])
            if len(weekly) >= lag
            else np.nan
        )
        output[f"neighbor_cases_max_lag_{lag}"] = (
            float(weekly["neighbor_cases_max"].iloc[-lag])
            if len(weekly) >= lag
            else np.nan
        )
    output["neighbor_incidence_lag_1"] = float(
        weekly["neighbor_incidence"].iloc[-1]
    )
    output["neighbor_incidence_max_lag_1"] = float(
        weekly["neighbor_incidence_max"].iloc[-1]
    )
    output["neighbor_temp_lag_4"] = (
        float(weekly["neighbor_temp"].iloc[-4]) if len(weekly) >= 4 else np.nan
    )
    output["neighbor_humidity_lag_4"] = (
        float(weekly["neighbor_humidity"].iloc[-4])
        if len(weekly) >= 4
        else np.nan
    )
    output["neighbor_precip_roll_8"] = float(
        weekly["neighbor_precip"].tail(8).mean()
    )
    return output


def _mapbiomas_features(
    mapbiomas: pd.DataFrame | None, location_code: int, cutoff: pd.Timestamp
) -> dict[str, float]:
    if mapbiomas is None or mapbiomas.empty:
        return {}
    # Annual structural data must be published before the forecasting cutoff.
    allowed_year = cutoff.year - 1
    available = mapbiomas.loc[
        mapbiomas["location_code"].eq(location_code)
        & mapbiomas["year"].le(allowed_year)
    ].sort_values("year")
    if available.empty:
        return {}
    row = available.iloc[-1]
    return {
        column: float(row[column])
        for column in available.columns
        if column.startswith("mapbiomas_") and pd.notna(row[column])
    }


def _seasonal_table(
    history: pd.DataFrame, cutoff: pd.Timestamp
) -> dict[int, dict[str, float]]:
    known = history.loc[history["date"].le(cutoff)]
    labels = {
        0.05: "05",
        0.10: "10",
        0.25: "25",
        0.75: "75",
        0.90: "90",
        0.95: "95",
    }
    output: dict[int, dict[str, float]] = {}
    peak_week = int(
        known.groupby("week", observed=True)["casos"].mean().idxmax()
    )
    peak_position = _season_position(peak_week)
    for week, group in known.groupby("week", observed=True):
        probabilities = list(labels)
        cases = group["casos"].to_numpy(dtype=float)
        quantiles = np.nanquantile(cases, probabilities)
        regional_quantiles = (
            np.nanquantile(
                group["regional_cases"].to_numpy(dtype=float), probabilities
            )
            if "regional_cases" in group
            else None
        )
        features = {
            "historical_mean": float(np.mean(cases)),
            "historical_median": float(np.median(cases)),
            "seasonal_naive": float(cases[-1]),
            "historical_peak_week": float(peak_week),
            "weeks_to_historical_peak": float(
                peak_position - _season_position(int(week))
            ),
        }
        for quantile_index, (_, label) in enumerate(labels.items()):
            features[f"historical_q{label}"] = float(quantiles[quantile_index])
            if regional_quantiles is not None:
                features[f"regional_historical_q{label}"] = float(
                    regional_quantiles[quantile_index]
                )
        output[int(week)] = features
    return output


def _rolling_target_dates(cutoff: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(
        cutoff + pd.Timedelta(weeks=FORECAST_GAP_WEEKS),
        periods=FORECAST_WEEKS,
        freq="W-SUN",
    )
    weeks = [Week.fromdate(value.date()) for value in dates]
    return pd.DataFrame(
        {
            "date": dates,
            "epiweek": [week.year * 100 + week.week for week in weeks],
            "target_year": [week.year for week in weeks],
            "target_week": [week.week for week in weeks],
            "horizon": np.arange(1, FORECAST_WEEKS + 1, dtype=int),
        }
    )


def _matrix_for_origins(
    panel: pd.DataFrame,
    origins: list[tuple[pd.Timestamp, pd.DataFrame]],
    mapbiomas: pd.DataFrame | None,
    include_target: bool,
    training_only: bool = False,
) -> pd.DataFrame:
    panel = panel.copy()
    if "location_code" not in panel:
        panel = panel.rename(
            columns={"uf_code": "location_code", "uf": "location"}
        )
    panel["date"] = pd.to_datetime(panel["date"])
    is_uf = panel.get("level", pd.Series(["uf"])).iloc[0] == "uf"

    # Calculate new geographical lag columns
    if is_uf:
        panel["macroregion_cases_sum"] = panel.groupby(
            [panel["location_code"] // 10, "date"]
        )["casos"].transform("sum")
        panel["macroregion_cases_max"] = panel.groupby(
            [panel["location_code"] // 10, "date"]
        )["casos"].transform("max")
        panel["macroregion_incidence_mean"] = panel.groupby(
            [panel["location_code"] // 10, "date"]
        )["incidence_100k"].transform("mean")
        panel["macroregion_incidence_max"] = panel.groupby(
            [panel["location_code"] // 10, "date"]
        )["incidence_100k"].transform("max")
    else:
        panel["regional_cases_max"] = panel.groupby(
            ["regional_geocode", "date"]
        )["casos"].transform("max")
        panel["macroregional_cases_max"] = panel.groupby(
            ["macroregional_geocode", "date"]
        )["casos"].transform("max")
        panel["uf_cases_max"] = panel.groupby(["uf_code", "date"])[
            "casos"
        ].transform("max")

        panel["regional_incidence_max"] = panel.groupby(
            ["regional_geocode", "date"]
        )["incidence_100k"].transform("max")
        panel["macroregional_incidence_max"] = panel.groupby(
            ["macroregional_geocode", "date"]
        )["incidence_100k"].transform("max")
        panel["uf_incidence_max"] = panel.groupby(["uf_code", "date"])[
            "incidence_100k"
        ].transform("max")

    extra_geo_cols = (
        "regional_cases",
        "macroregional_cases",
        "uf_cases",
        "regional_cases_max",
        "macroregional_cases_max",
        "uf_cases_max",
        "regional_incidence_max",
        "macroregional_incidence_max",
        "uf_incidence_max",
        "macroregion_cases_sum",
        "macroregion_cases_max",
        "macroregion_incidence_mean",
        "macroregion_incidence_max",
    )
    for col in extra_geo_cols:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").astype(
                "float32"
            )
    if is_uf:
        try:
            neighbor_map = load_neighbor_map()
        except (FileNotFoundError, ImportError):
            neighbor_map = {}
    else:
        neighbor_map = {}
    frames: list[pd.DataFrame] = []
    groups = panel.groupby(["location_code", "location"], observed=True)
    total_locations = groups.ngroups
    LOGGER.info(
        "matriz | locais=%d origens=%d linhas/local=%d",
        total_locations,
        len(origins),
        sum(len(targets) for _, targets in origins),
    )
    for index, ((location_code, location), history) in enumerate(groups, 1):
        if (
            index == 1
            or index == total_locations
            or index % max(total_locations // 10, 1) == 0
        ):
            LOGGER.info(
                "matriz | local %d/%d | %s", index, total_locations, location
            )
        history = history.sort_values("date")
        extreme_history = _prepare_extreme_history(history)
        neighbor_history = _prepare_neighbor_history(
            panel, neighbor_map, int(location_code)
        )
        observed = history.set_index("epiweek")["casos"]
        population = history.set_index("date")["population"]
        rows: list[dict[str, object]] = []
        for cutoff, targets in origins:
            if history["date"].min() > cutoff:
                continue
            base = _origin_features(history, cutoff)
            extreme = _extreme_origin_features(extreme_history, cutoff)
            neighbor = _neighbor_origin_features(neighbor_history, cutoff)
            mapbiomas_features = _mapbiomas_features(
                mapbiomas, int(location_code), cutoff
            )
            seasonal = _seasonal_table(history, cutoff)
            cutoff_population = population.loc[population.index <= cutoff]
            for target in targets.itertuples(index=False):
                row: dict[str, object] = {
                    "location_code": int(location_code),
                    "origin_date": cutoff,
                    "origin_year": cutoff.year,
                    "target_year": target.target_year,
                    "target_week": target.target_week,
                    "horizon": target.horizon,
                    "target_end_date": targets["date"].iloc[-1],
                    "week_sin": np.sin(2 * np.pi * target.target_week / 52.0),
                    "week_cos": np.cos(2 * np.pi * target.target_week / 52.0),
                    "phase_start": float(target.target_week >= 41),
                    "phase_peak": float(target.target_week <= 16),
                    "phase_tail": float(17 <= target.target_week <= 40),
                    "population": float(cutoff_population.iloc[-1])
                    if len(cutoff_population)
                    else np.nan,
                    **base,
                    **extreme,
                    **neighbor,
                    **seasonal.get(int(target.target_week), {}),
                    **mapbiomas_features,
                }
                # --- Interaction features (depend on both origin and target context) ---
                weeks_to_peak = row.get("weeks_to_historical_peak", np.nan)
                if pd.notna(weeks_to_peak):
                    row["horizon_x_weeks_to_peak"] = float(
                        target.horizon * weeks_to_peak
                    )
                    row["horizon_x_phase_peak"] = float(
                        target.horizon * row.get("phase_peak", 0)
                    )
                # Epidemic intensity: how extreme is the current level vs historical quantiles?
                roll_max_4 = row.get("cases_roll_max_4", np.nan)
                hist_q90 = row.get("historical_q90", np.nan)
                if (
                    pd.notna(roll_max_4)
                    and pd.notna(hist_q90)
                    and hist_q90 > 0
                ):
                    row["epidemic_intensity"] = float(
                        roll_max_4 / (hist_q90 + 1)
                    )
                else:
                    row["epidemic_intensity"] = np.nan
                row.update(date=target.date, epiweek=target.epiweek)
                if not training_only:
                    row.update(location=location)
                    for column in ("challenge", "disease", "level"):
                        if column in history:
                            row[column] = history[column].iloc[0]
                if include_target:
                    row["target"] = float(observed.get(target.epiweek, np.nan))
                rows.append(row)

        frame = pd.DataFrame(rows)
        floats = frame.select_dtypes(include="float64").columns
        integers = frame.select_dtypes(include="int64").columns
        frame = frame.astype(
            {
                **dict.fromkeys(floats, "float32"),
                **dict.fromkeys(integers, "int32"),
            }
        )
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if training_only:
        return output
    return output.sort_values(
        ["origin_date", "location_code", "date"]
    ).reset_index(drop=True)


def build_origin_matrix(
    panel: pd.DataFrame,
    origin_years: list[int] | range,
    mapbiomas: pd.DataFrame | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    origins = [
        (official_cutoff_date(year), season_epiweeks(year))
        for year in origin_years
        if year >= FIRST_ORIGIN_YEAR
    ]
    return _matrix_for_origins(panel, origins, mapbiomas, include_target)


def build_rolling_training_matrix(
    panel: pd.DataFrame,
    evaluation_cutoff: pd.Timestamp,
    mapbiomas: pd.DataFrame | None = None,
    origin_stride_weeks: int = 1,
) -> pd.DataFrame:
    panel_dates = pd.to_datetime(panel["date"])
    labels_available_until = min(
        pd.Timestamp(evaluation_cutoff), panel_dates.max()
    )
    latest_origin = labels_available_until - pd.Timedelta(
        weeks=FORECAST_GAP_WEEKS + FORECAST_WEEKS - 1
    )
    earliest_origin = panel_dates.min() + pd.Timedelta(
        weeks=max(CASE_LAGS) - 1
    )
    cutoffs = pd.date_range(earliest_origin, latest_origin, freq="W-SUN")[
        ::origin_stride_weeks
    ]
    origins = [(cutoff, _rolling_target_dates(cutoff)) for cutoff in cutoffs]
    matrix = _matrix_for_origins(
        panel,
        origins,
        mapbiomas,
        include_target=True,
        training_only=True,
    )
    return (
        matrix
        if matrix["target"].notna().all()
        else matrix.dropna(subset=["target"])
    )


def build_official_like_training_matrix(
    panel: pd.DataFrame,
    evaluation_cutoff: pd.Timestamp,
    mapbiomas: pd.DataFrame | None = None,
    origin_weeks: list[int] | range = range(21, 30),
    years: list[int] | range | None = None,
) -> pd.DataFrame:
    panel_dates = pd.to_datetime(panel["date"])
    labels_available_until = min(
        pd.Timestamp(evaluation_cutoff), panel_dates.max()
    )
    earliest_origin = panel_dates.min() + pd.Timedelta(
        weeks=max(CASE_LAGS) - 1
    )
    years = years or range(FIRST_ORIGIN_YEAR, labels_available_until.year + 1)
    origins = []
    for year in years:
        for week in origin_weeks:
            cutoff = pd.Timestamp(Week(int(year), int(week)).startdate())
            targets = _rolling_target_dates(cutoff)
            if (
                cutoff >= earliest_origin
                and targets["date"].max() <= labels_available_until
            ):
                origins.append((cutoff, targets))
    matrix = _matrix_for_origins(
        panel,
        origins,
        mapbiomas,
        include_target=True,
        training_only=True,
    )
    return (
        matrix
        if matrix.empty or matrix["target"].notna().all()
        else matrix.dropna(subset=["target"])
    )


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    excluded = {
        "location",
        "challenge",
        "disease",
        "level",
        "origin_date",
        "target_end_date",
        "date",
        "epiweek",
        "target",
        "horizon_sq",
        "horizon_sqrt",
        "temp_min_roll_mean_4",
        "temp_min_roll_mean_8",
        "temp_min_roll_mean_12",
        "temp_max_roll_mean_4",
        "temp_max_roll_mean_8",
        "temp_max_roll_mean_12",
        "rel_humid_min_roll_mean_4",
        "rel_humid_min_roll_mean_8",
        "rel_humid_min_roll_mean_12",
        "rel_humid_max_roll_mean_4",
        "rel_humid_max_roll_mean_8",
        "rel_humid_max_roll_mean_12",
        "thermal_range_roll_mean_4",
        "thermal_range_roll_mean_8",
        "thermal_range_roll_mean_12",
        "rainy_days_roll_mean_4",
        "rainy_days_roll_mean_8",
        "rainy_days_roll_mean_12",
    }
    return [
        column
        for column in matrix.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(matrix[column])
    ]
