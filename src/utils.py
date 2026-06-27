from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone as _tz

    UTC = _tz.utc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from epiweeks import Week

LOGGER = logging.getLogger("imdc")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def timed_step(name: str):
    start = time.perf_counter()
    LOGGER.info("start | %s", name)
    try:
        yield
    finally:
        LOGGER.info("done | %s | %.1fs", name, time.perf_counter() - start)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def season_epiweeks(origin_year: int) -> pd.DataFrame:
    start = pd.Timestamp(Week(origin_year, 41).startdate())
    end = pd.Timestamp(Week(origin_year + 1, 40).startdate())
    dates = pd.date_range(start, end, freq="W-SUN")
    weeks = [Week.fromdate(value.date()) for value in dates]
    return pd.DataFrame(
        {
            "date": dates,
            "epiweek": [week.year * 100 + week.week for week in weeks],
            "target_year": [week.year for week in weeks],
            "target_week": [week.week for week in weeks],
            "horizon": np.arange(1, len(dates) + 1, dtype=int),
        }
    )


def quantile_column(quantile: float) -> str:
    return f"q{quantile:.3f}".rstrip("0").rstrip(".")
