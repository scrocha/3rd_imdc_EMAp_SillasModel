from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from esda import Moran, Moran_Local
from libpysal.weights import KNN
from rasterio.errors import WindowError
from rasterio.features import geometry_window, rasterize
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    EXTERNAL_DIR,
    FTP_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_directories,
)
from .utils import setup_logging, sha256_file, timed_step, utc_now, write_json

MAPBIOMAS_YEARS = tuple(range(2010, 2025))
MAPBIOMAS_URL = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    "collection_10/lulc/coverage/brazil_coverage_{year}.tif"
)
CLASS_GROUPS = {
    "forest_share": {1, 3, 4, 5, 6, 49},
    "native_vegetation_share": {1, 3, 4, 5, 6, 10, 11, 12, 29, 32, 49, 50},
    "pasture_share": {15},
    "agriculture_share": {9, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62},
    "urban_land_share": {24},
    "water_share": {26, 33},
    "wetland_share": {11, 32},
    "non_vegetated_area_share": {22, 23, 25, 29, 30, 31},
}


def raster_path(year: int) -> Path:
    return EXTERNAL_DIR / "mapbiomas" / f"brazil_coverage_{year}.tif"


def download_mapbiomas(
    years: tuple[int, ...] = MAPBIOMAS_YEARS, force: bool = False
) -> None:
    ensure_directories()
    directory = EXTERNAL_DIR / "mapbiomas"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    for year in years:
        path = raster_path(year)
        if path.exists() and not force:
            print(f"MapBiomas {year} já existe; pulando.")
        else:
            temporary = path.with_suffix(".tif.part")
            print(f"Baixando MapBiomas {year}...")
            with requests.get(
                MAPBIOMAS_URL.format(year=year), stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(
                        chunk_size=8 * 1024 * 1024
                    ):
                        stream.write(chunk)
            temporary.replace(path)
        manifest[path.name] = {
            "source": "mapbiomas_collection_10",
            "url": MAPBIOMAS_URL.format(year=year),
            "path": str(path.relative_to(RAW_DIR)),
            "downloaded_at": utc_now(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "year": year,
            "status": "downloaded",
        }
    write_json(manifest_path, manifest)


def _summarize_classes(values: np.ndarray) -> dict[str, float]:
    if values.size:
        classes, counts = np.unique(values.astype(int), return_counts=True)
        shares = counts / counts.sum()
        class_share = dict(
            zip(classes.tolist(), shares.tolist(), strict=False)
        )
    else:
        shares = np.asarray([], dtype=float)
        class_share = {}
    output = {
        f"mapbiomas_{name}": float(
            sum(class_share.get(code, 0.0) for code in classes)
        )
        for name, classes in CLASS_GROUPS.items()
    }
    output["mapbiomas_agro_land_share"] = (
        output["mapbiomas_agriculture_share"]
        + output["mapbiomas_pasture_share"]
    )
    positive = shares[shares > 0]
    output["mapbiomas_land_use_diversity"] = float(
        -np.sum(positive * np.log(positive))
    )
    native = output["mapbiomas_native_vegetation_share"]
    agro = output["mapbiomas_agro_land_share"]
    urban = output["mapbiomas_urban_land_share"]
    output["mapbiomas_urban_native_ratio"] = urban / (native + 1e-6)
    output["mapbiomas_agro_native_ratio"] = agro / (native + 1e-6)
    output["mapbiomas_urban_agro_ratio"] = urban / (agro + 1e-6)
    output["mapbiomas_non_natural_share"] = (
        urban + agro + output["mapbiomas_non_vegetated_area_share"]
    )
    return output


def _pixels_touching_geometry(
    source: rasterio.io.DatasetReader, geometry: object
) -> np.ndarray:
    """Read only source pixels whose cells touch the municipal geometry."""
    try:
        window = geometry_window(source, [geometry], pad_x=0, pad_y=0)
    except WindowError:
        return np.asarray([], dtype=source.dtypes[0])
    data = source.read(1, window=window, masked=True)
    if data.size == 0:
        return np.asarray([], dtype=source.dtypes[0])
    touched = rasterize(
        [(geometry, 1)],
        out_shape=data.shape,
        transform=source.window_transform(window),
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    valid = touched & ~np.ma.getmaskarray(data)
    if source.nodata is not None:
        valid &= np.asarray(data) != source.nodata
    return np.asarray(data)[valid]


def extract_municipal_year(
    year: int, municipalities: gpd.GeoDataFrame
) -> pd.DataFrame:
    path = raster_path(year)
    if not path.exists():
        raise FileNotFoundError(f"Raster MapBiomas ausente: {path}")
    with rasterio.open(path) as source:
        aligned_municipalities = municipalities.to_crs(source.crs)
        records = []
        iterator = tqdm(
            aligned_municipalities.itertuples(index=False),
            total=len(aligned_municipalities),
            desc=f"MapBiomas {year}: municípios",
        )
        for row in iterator:
            values = _pixels_touching_geometry(source, row.geometry)
            record = {
                "geocode": int(row.geocode),
                "uf_code": int(row.uf_code),
                "year": year,
                "mapbiomas_pixels_touching": int(values.size),
            }
            record.update(_summarize_classes(values))
            records.append(record)
    return pd.DataFrame(records)


def add_changes(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.sort_values(["geocode", "year"]).copy()
    bases = (
        "urban_land_share",
        "agro_land_share",
        "native_vegetation_share",
        "water_share",
    )
    for base in bases:
        column = f"mapbiomas_{base}"
        output[f"mapbiomas_{base}_change_1y"] = output.groupby("geocode")[
            column
        ].diff(1)
        output[f"mapbiomas_{base}_change_2y"] = output.groupby("geocode")[
            column
        ].diff(2)
    return output


def aggregate_uf(municipal: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in municipal
        if column.startswith("mapbiomas_")
        and column != "mapbiomas_pixels_touching"
    ]
    rows = []
    for (uf_code, year), group in municipal.groupby(
        ["uf_code", "year"], observed=True
    ):
        weights = group["mapbiomas_pixels_touching"].astype(float)
        row: dict[str, float | int] = {
            "uf_code": int(uf_code),
            "year": int(year),
            "mapbiomas_pixels_touching": int(weights.sum()),
        }
        for column in feature_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            valid = values.notna() & weights.gt(0)
            row[column] = (
                float(np.average(values[valid], weights=weights[valid]))
                if valid.any()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_mapbiomas_diagnostics(
    municipal: pd.DataFrame, municipalities: gpd.GeoDataFrame
) -> None:
    directory = PROCESSED_DIR.parent / "results" / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    panel_path = PROCESSED_DIR / "panel_dengue_muni_week.parquet"
    feature_columns = [
        column for column in municipal if column.startswith("mapbiomas_")
    ]
    if panel_path.exists():
        dengue = pd.read_parquet(
            panel_path, columns=["geocode", "year", "casos"]
        )
        annual = (
            dengue.groupby(["geocode", "year"], observed=True)["casos"]
            .sum()
            .reset_index()
        )
        joined = municipal.merge(annual, on=["geocode", "year"], how="left")
        correlations = [
            {
                "feature": column,
                "correlation_with_cases": joined[column].corr(joined["casos"]),
            }
            for column in feature_columns
        ]
        pd.DataFrame(correlations).to_csv(
            directory / "mapbiomas_correlation.csv", index=False
        )

    geometry = (
        municipalities.set_index("geocode")
        .loc[municipal["geocode"].drop_duplicates()]
        .reset_index()
    )
    metric = geometry.to_crs("EPSG:5880")
    points = np.column_stack(
        (metric.geometry.centroid.x, metric.geometry.centroid.y)
    )
    weights = KNN.from_array(points, k=8)
    weights.transform = "R"
    moran_rows = []
    for year, table in municipal.groupby("year", observed=True):
        indexed = table.set_index("geocode")
        for column in feature_columns:
            values = indexed[column].reindex(geometry["geocode"]).astype(float)
            values = values.fillna(values.mean())
            if values.nunique() < 2:
                continue
            statistic = Moran(values.to_numpy(), weights, permutations=0)
            moran_rows.append(
                {
                    "year": int(year),
                    "feature": column,
                    "moran_i": statistic.I,
                    "p_normal": statistic.p_norm,
                }
            )
    pd.DataFrame(moran_rows).to_csv(
        directory / "mapbiomas_moran.csv", index=False
    )
    latest = municipal.loc[
        municipal["year"].eq(municipal["year"].max())
    ].set_index("geocode")
    values = (
        latest["mapbiomas_urban_land_share"]
        .reindex(geometry["geocode"])
        .fillna(0)
    )
    lisa = Moran_Local(values.to_numpy(), weights, permutations=99, seed=42)
    labels = np.array(
        ["not_significant", "high_high", "low_high", "low_low", "high_low"]
    )
    output = geometry.copy()
    output["urban_land_share"] = values.to_numpy()
    output["lisa_cluster"] = np.where(
        lisa.p_sim < 0.05, labels[lisa.q], labels[0]
    )
    output["lisa_i"] = lisa.Is
    output["lisa_p"] = lisa.p_sim
    output.to_file(directory / "mapbiomas_lisa.gpkg", driver="GPKG")


def build_mapbiomas(years: tuple[int, ...] = MAPBIOMAS_YEARS) -> None:
    municipalities = (
        gpd.read_file(FTP_DIR / "shape_muni.gpkg")
        .sort_values("geocode")
        .reset_index(drop=True)
    )
    annual = []
    for year in years:
        cache = INTERIM_DIR / f"mapbiomas_muni_{year}.parquet"
        if cache.exists():
            table = pd.read_parquet(cache)
        else:
            table = extract_municipal_year(year, municipalities)
            table.to_parquet(cache, index=False)
        annual.append(table)
    municipal = add_changes(pd.concat(annual, ignore_index=True))
    municipal.to_parquet(
        PROCESSED_DIR / "mapbiomas_muni_year.parquet", index=False
    )
    aggregate_uf(municipal).to_parquet(
        PROCESSED_DIR / "mapbiomas_uf_year.parquet", index=False
    )
    write_mapbiomas_diagnostics(municipal, municipalities)


def main() -> None:
    setup_logging()
    with timed_step("mapbiomas | download"):
        download_mapbiomas()
    with timed_step("mapbiomas | processamento"):
        build_mapbiomas()


if __name__ == "__main__":
    main()
