from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
from libpysal.weights import KNN, Queen, W

from .config import EXCLUDED_UFS, FTP_DIR, PROCESSED_DIR


def build_uf_geometries() -> gpd.GeoDataFrame:
    municipalities = gpd.read_file(FTP_DIR / "shape_muni.gpkg")
    municipalities = municipalities.loc[
        ~municipalities["uf"].isin(EXCLUDED_UFS)
    ].copy()
    municipalities["geometry"] = municipalities.geometry.make_valid()
    ufs = municipalities.dissolve(
        by=["uf_code", "uf"], as_index=False, aggfunc="first"
    )
    ufs = ufs[["uf_code", "uf", "geometry"]].sort_values("uf_code")
    ufs.to_file(PROCESSED_DIR / "uf_geometry.gpkg", driver="GPKG")
    return ufs


def _row_standardized_union(contiguity: W, knn: W) -> W:
    neighbors: dict[int, list[int]] = {}
    weights: dict[int, list[float]] = {}
    for index in contiguity.id_order:
        linked = set(contiguity.neighbors.get(index, []))
        if not linked:
            linked.update(knn.neighbors[index])
        neighbors[index] = sorted(linked)
        weights[index] = [1.0] * len(neighbors[index])
    output = W(
        neighbors, weights, id_order=contiguity.id_order, silence_warnings=True
    )
    output.transform = "R"
    return output


def build_uf_weights(
    ufs: gpd.GeoDataFrame | None = None,
) -> tuple[W, dict[int, int]]:
    if ufs is None:
        geometry_path = PROCESSED_DIR / "uf_geometry.gpkg"
        ufs = (
            gpd.read_file(geometry_path)
            if geometry_path.exists()
            else build_uf_geometries()
        )
    ufs = ufs.sort_values("uf_code").reset_index(drop=True)
    queen = Queen.from_dataframe(ufs, use_index=False, silence_warnings=True)
    metric = ufs.to_crs("EPSG:5880")
    centroids = np.column_stack(
        (metric.geometry.centroid.x, metric.geometry.centroid.y)
    )
    knn = KNN.from_array(centroids, k=3)
    weights = _row_standardized_union(queen, knn)
    index_to_code = dict(enumerate(ufs["uf_code"].astype(int)))
    payload = {
        str(index_to_code[index]): [
            int(index_to_code[neighbor])
            for neighbor in weights.neighbors[index]
        ]
        for index in weights.id_order
    }
    path = PROCESSED_DIR / "uf_neighbors.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return weights, index_to_code


def load_neighbor_map() -> dict[int, list[int]]:
    path = PROCESSED_DIR / "uf_neighbors.json"
    if not path.exists():
        build_uf_weights()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(code): [int(value) for value in values]
        for code, values in payload.items()
    }
