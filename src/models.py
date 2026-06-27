from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import QUANTILES
from .features import feature_columns
from .metrics import enforce_quantile_order
from .utils import LOGGER, quantile_column

DEFAULT_N_JOBS = os.cpu_count() or 2
# Fraction of training data used as temporal validation for early stopping
_EARLY_STOP_FRAC = 0.15
_EARLY_STOP_ROUNDS = 50
PREDICTION_COLUMNS = (
    "location_code",
    "location",
    "challenge",
    "disease",
    "level",
    "date",
    "epiweek",
    "horizon",
    "origin_year",
    "target",
)


def _complete_rows(matrix: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    valid = matrix[columns].notna().all(axis=1)
    return matrix if valid.all() else matrix.loc[valid]


def _prediction_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix[
        [column for column in PREDICTION_COLUMNS if column in matrix]
    ].copy()


@lru_cache
def accelerator_device() -> str:
    requested = os.getenv(
        "MODEL_DEVICE", os.getenv("XGBOOST_DEVICE", "auto")
    ).lower()
    if requested == "cpu":
        return "cpu"
    wants_gpu = requested in {"cuda", "gpu"}
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return "cuda"
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        if wants_gpu:
            raise RuntimeError(
                "GPU solicitada, mas o processo não consegue acessar o driver NVIDIA. "
                "Verifique nvidia-smi, /dev/nvidia* e a exposição da GPU no container/WSL."
            ) from exc
        return "cpu"


def _xgboost_device_args() -> dict[str, str]:
    return {"device": accelerator_device(), "tree_method": "hist"}


def _xgb_predict(model: XGBRegressor, x: pd.DataFrame) -> np.ndarray:
    # ponytail: training is the expensive GPU step; CPU inference avoids a CuPy dependency.
    model.get_booster().set_param({"device": "cpu"})
    return model.predict(x)


@dataclass
class XGBoostQuantileForecaster:
    random_state: int = 42
    use_extreme_features: bool = False
    log_target: bool = False
    extreme_weight_q95: float = 6.0
    extreme_weight_q90: float = 4.0
    extreme_weight_q75: float = 2.0
    params: dict[str, Any] = field(default_factory=dict)
    model: XGBRegressor | None = None
    columns: list[str] = field(default_factory=list)
    medians: pd.Series | None = None

    def _apply_transform(self, y: pd.Series) -> pd.Series:
        return np.log1p(y) if self.log_target else y

    def _inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y) if self.log_target else y

    def fit(self, matrix: pd.DataFrame) -> "XGBoostQuantileForecaster":
        training = _complete_rows(matrix, ["target"])
        self.columns = feature_columns(training)
        self.medians = (
            training[self.columns].median(numeric_only=True).fillna(0)
        )
        x = (
            training[self.columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        arguments = {
            **_xgboost_device_args(),
            "objective": "reg:quantileerror",
            "quantile_alpha": np.asarray(QUANTILES),
            "n_estimators": 600,
            "learning_rate": 0.04,
            "max_depth": 7,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "random_state": self.random_state,
            "n_jobs": DEFAULT_N_JOBS,
            **self.params,
        }
        y_cases = training["target"].clip(lower=0)
        y = self._apply_transform(y_cases)
        sample_weight = np.ones(len(training), dtype=float)
        if "origin_distance_ew25" in training:
            distance = training["origin_distance_ew25"].to_numpy(dtype=float)
            sample_weight *= 0.3 + 0.7 * np.exp(-distance / 6)
        if self.use_extreme_features:
            has_q95 = "historical_q95" in training.columns
            has_q90 = "historical_q90" in training.columns
            has_q75 = "historical_q75" in training.columns
            if has_q90:
                sample_weight *= np.select(
                    [
                        y_cases.gt(training["historical_q95"])
                        if has_q95
                        else False,
                        y_cases.gt(training["historical_q90"]),
                        y_cases.gt(training["historical_q75"])
                        if has_q75
                        else False,
                    ],
                    [
                        self.extreme_weight_q95,
                        self.extreme_weight_q90,
                        self.extreme_weight_q75,
                    ],
                    default=1.0,
                )
        # Temporal early stopping: hold out the most recent target dates.
        # Uses the individual `date` column per row so that overlapping origins
        # placing the same calendar week in both train and val are avoided.
        if "date" in training.columns:
            unique_dates = np.sort(training["date"].unique())
            n_val_dates = max(1, int(len(unique_dates) * _EARLY_STOP_FRAC))
            val_dates = set(unique_dates[-n_val_dates:])
            val_mask = training["date"].isin(val_dates).to_numpy()
            LOGGER.info(
                "early_stop | datas=%d val_datas=%d treino=%s val=%s",
                len(unique_dates),
                n_val_dates,
                f"{(~val_mask).sum():,}",
                f"{val_mask.sum():,}",
            )
        else:
            n_val = min(max(1, int(len(training) * _EARLY_STOP_FRAC)), 50_000)
            val_mask = np.zeros(len(training), dtype=bool)
            val_mask[-n_val:] = True
            LOGGER.info(
                "early_stop | fallback | linhas=%s val=%s",
                f"{len(training):,}",
                f"{n_val:,}",
            )
        x_tr, x_va = x.iloc[~val_mask], x.iloc[val_mask]
        y_tr, y_va = y.iloc[~val_mask], y.iloc[val_mask]
        sw_tr, sw_va = sample_weight[~val_mask], sample_weight[val_mask]
        self.model = XGBRegressor(
            **arguments, early_stopping_rounds=_EARLY_STOP_ROUNDS
        )
        self.model.fit(
            x_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(x_va, y_va)],
            sample_weight_eval_set=[sw_va],
            verbose=False,
        )
        return self

    def predict(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.medians is None:
            raise RuntimeError(
                "O modelo precisa ser ajustado antes da previsão."
            )
        x = (
            matrix.reindex(columns=self.columns)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        values = np.asarray(_xgb_predict(self.model, x))
        values = self._inverse_transform(values)
        output = _prediction_frame(matrix)
        for index, quantile in enumerate(QUANTILES):
            output[quantile_column(quantile)] = np.maximum(values[:, index], 0)
        return enforce_quantile_order(output)


def historical_quantile_baseline(matrix: pd.DataFrame) -> pd.DataFrame:
    output = _prediction_frame(matrix)
    median = (
        matrix["historical_median"].fillna(matrix["seasonal_naive"]).fillna(0)
    )
    spread = (
        (matrix["historical_q75"] - matrix["historical_q25"])
        .clip(lower=1)
        .fillna(1)
    )
    NORMAL_SCORES = {
        0.025: -1.96,
        0.05: -1.645,
        0.10: -1.282,
        0.25: -0.674,
        0.50: 0.0,
        0.75: 0.674,
        0.90: 1.282,
        0.95: 1.645,
        0.975: 1.96,
    }
    sigma = spread / 1.349
    horizon_scale = np.sqrt(1 + matrix["horizon"].to_numpy(dtype=float) / 52)
    for quantile in QUANTILES:
        output[quantile_column(quantile)] = np.maximum(
            median + NORMAL_SCORES[quantile] * sigma * horizon_scale, 0
        )
    return enforce_quantile_order(output)


@dataclass
class XGBoostResidualForecaster:
    """Experimento 1: residual sobre baseline.

    Hipótese: o baseline já captura sazonalidade e escala;
    o XGB deve aprender correção, não casos absolutos.

    target = log1p(y) - log1p(historical_median)
    q0.5_final = expm1(log1p(historical_median) + pred_delta)

    Para quantis: escala proporcional ao baseline iqr.
    """

    random_state: int = 42
    use_extreme_features: bool = False
    extreme_weight_q95: float = 6.0
    extreme_weight_q90: float = 4.0
    extreme_weight_q75: float = 2.0
    params: dict[str, Any] = field(default_factory=dict)
    model: XGBRegressor | None = None
    columns: list[str] = field(default_factory=list)
    medians: pd.Series | None = None

    def fit(self, matrix: pd.DataFrame) -> "XGBoostResidualForecaster":
        training = _complete_rows(matrix, ["target", "historical_median"])
        self.columns = feature_columns(training)
        self.medians = (
            training[self.columns].median(numeric_only=True).fillna(0)
        )
        x = (
            training[self.columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        arguments = {
            **_xgboost_device_args(),
            "objective": "reg:quantileerror",
            "quantile_alpha": np.asarray(QUANTILES),
            "n_estimators": 600,
            "learning_rate": 0.04,
            "max_depth": 7,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "random_state": self.random_state,
            "n_jobs": DEFAULT_N_JOBS,
            **self.params,
        }
        y_cases = training["target"].clip(lower=0)
        y_baseline = np.log1p(
            training["historical_median"].fillna(1).clip(lower=1)
        )
        y_log = np.log1p(y_cases)
        y_residual = y_log - y_baseline
        sample_weight = np.ones(len(training), dtype=float)
        if "origin_distance_ew25" in training:
            distance = training["origin_distance_ew25"].to_numpy(dtype=float)
            sample_weight *= 0.3 + 0.7 * np.exp(-distance / 6)
        if self.use_extreme_features:
            has_q95 = "historical_q95" in training.columns
            has_q90 = "historical_q90" in training.columns
            has_q75 = "historical_q75" in training.columns
            if has_q90:
                sample_weight *= np.select(
                    [
                        y_cases.gt(training["historical_q95"])
                        if has_q95
                        else False,
                        y_cases.gt(training["historical_q90"]),
                        y_cases.gt(training["historical_q75"])
                        if has_q75
                        else False,
                    ],
                    [
                        self.extreme_weight_q95,
                        self.extreme_weight_q90,
                        self.extreme_weight_q75,
                    ],
                    default=1.0,
                )
        if "date" in training.columns:
            unique_dates = np.sort(training["date"].unique())
            n_val_dates = max(1, int(len(unique_dates) * _EARLY_STOP_FRAC))
            val_dates = set(unique_dates[-n_val_dates:])
            val_mask = training["date"].isin(val_dates).to_numpy()
        else:
            n_val = min(max(1, int(len(training) * _EARLY_STOP_FRAC)), 50_000)
            val_mask = np.zeros(len(training), dtype=bool)
            val_mask[-n_val:] = True
        x_tr, x_va = x.iloc[~val_mask], x.iloc[val_mask]
        y_tr, y_va = y_residual.iloc[~val_mask], y_residual.iloc[val_mask]
        sw_tr, sw_va = sample_weight[~val_mask], sample_weight[val_mask]
        self.model = XGBRegressor(
            **arguments, early_stopping_rounds=_EARLY_STOP_ROUNDS
        )
        self.model.fit(
            x_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(x_va, y_va)],
            sample_weight_eval_set=[sw_va],
            verbose=False,
        )
        return self

    def predict(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.medians is None:
            raise RuntimeError(
                "O modelo precisa ser ajustado antes da previsão."
            )
        x = (
            matrix.reindex(columns=self.columns)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        values = np.asarray(_xgb_predict(self.model, x))
        y_baseline = np.log1p(
            matrix["historical_median"]
            .fillna(1)
            .clip(lower=1)
            .to_numpy(dtype=float)
        )
        output = _prediction_frame(matrix)
        for idx, quantile in enumerate(QUANTILES):
            pred = np.expm1(y_baseline + values[:, idx])
            output[quantile_column(quantile)] = np.maximum(pred, 0)
        return enforce_quantile_order(output)
