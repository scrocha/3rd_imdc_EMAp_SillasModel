from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mosqlient as mosq

from src.config import CHALLENGES, RESULTS_DIR, ROUNDS
from src.utils import quantile_column

DISEASE_CODES = {"dengue": "A90", "chikungunya": "A92.0"}
DEFAULT_MODELS = ("3rd_imdc_emap_lstm",)
PREDICTION_COLUMNS = {
    "lower_95": quantile_column(0.025),
    "lower_90": quantile_column(0.05),
    "lower_80": quantile_column(0.10),
    "lower_50": quantile_column(0.25),
    "pred": quantile_column(0.5),
    "upper_50": quantile_column(0.75),
    "upper_80": quantile_column(0.90),
    "upper_90": quantile_column(0.95),
    "upper_95": quantile_column(0.975),
}


def _safe_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.rsplit("/", 1)[-1]).strip("_")


def _target_table(challenge_name: str, round_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / "predictions" / challenge_name / "baseline" / f"{round_name}.parquet"
    frame = pd.read_parquet(path, columns=["date", "location_code", "location", "target"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _prediction_frame(prediction, challenge, round_name: str) -> pd.DataFrame:
    frame = prediction.to_dataframe()
    if frame.empty:
        return frame
    frame = frame.rename(columns=PREDICTION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["location_code"] = prediction.adm_2 if challenge.level == "city" else prediction.adm_1
    frame["round"] = round_name
    frame["model"] = _safe_model_name(prediction.model.repository)
    keep = ["date", "location_code", "round", "model", *PREDICTION_COLUMNS.values()]
    return frame[keep]


def fetch_model(api_key: str, model: str, output_dir: Path) -> None:
    saved = 0
    for challenge in CHALLENGES:
        for round_info in ROUNDS:
            if round_info.name == "final":
                continue
            target = _target_table(challenge.name, round_info.name)
            print(f"buscando {challenge.name} {round_info.name} {model}", flush=True)
            preds = mosq.get_predictions(
                api_key=api_key,
                model_name=model,
                disease=DISEASE_CODES[challenge.disease],
                adm_level=1 if challenge.level == "uf" else 2,
                model_time_resolution="week",
                start=target["date"].min().date(),
                end=target["date"].max().date(),
            )
            frames = [
                _prediction_frame(pred, challenge, round_info.name)
                for pred in preds
                if pred.published
            ]
            frames = [frame for frame in frames if not frame.empty]
            if not frames:
                print(f"{challenge.name} {round_info.name} {model}: sem dados")
                continue
            output = pd.concat(frames, ignore_index=True)
            if challenge.city_codes:
                output = output.loc[output["location_code"].isin(challenge.city_codes)]
            output = output.merge(target, on=["date", "location_code"], how="inner")
            model_name = _safe_model_name(model)
            path = output_dir / challenge.name / model_name / f"{round_info.name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, index=False)
            saved += 1
            print(f"{challenge.name} {round_info.name} {model_name}: {len(output):,} linhas -> {path}")
    print(f"{model}: {saved} arquivos salvos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("MOSQLIMATE_API_KEY"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "predictions_external")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Defina MOSQLIMATE_API_KEY ou passe --api-key.")
    for model in args.models:
        fetch_model(args.api_key, model, args.output_dir)


if __name__ == "__main__":
    main()
