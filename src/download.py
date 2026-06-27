from __future__ import annotations

import ftplib
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from .config import (
    ESSENTIAL_FTP_FILES,
    FTP_DIR,
    FTP_FILE_ALIASES,
    FTP_FILES,
    FTP_HOST,
    FTP_REMOTE_DIR,
    RAW_DIR,
    ensure_directories,
)
from .utils import setup_logging, sha256_file, timed_step, utc_now, write_json


def _download_one(ftp: ftplib.FTP, filename: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as stream:
        ftp.retrbinary(f"RETR {filename}", stream.write)
    temporary.replace(destination)


def download_ftp(force: bool = False) -> dict[str, dict[str, object]]:
    ensure_directories()
    previous_path = RAW_DIR / "manifest.json"
    manifest = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else {}
    )

    with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
        ftp.login()
        ftp.cwd(FTP_REMOTE_DIR)
        available = set(ftp.nlst())
        for filename in FTP_FILES:
            remote_name = next(
                (
                    candidate
                    for candidate in FTP_FILE_ALIASES[filename]
                    if candidate in available
                ),
                None,
            )
            if remote_name is None:
                continue
            path = FTP_DIR / filename
            if force or not path.exists():
                print(f"Baixando {remote_name} como {filename}")
                _download_one(ftp, remote_name, path)
            manifest[filename] = inspect_file(path, remote_name=remote_name)

    missing = [
        name for name in ESSENTIAL_FTP_FILES if not (FTP_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Arquivos essenciais ausentes no FTP: {missing}"
        )
    write_json(previous_path, manifest)
    return manifest


def ensure_downloads() -> None:
    missing = [name for name in FTP_FILES if not (FTP_DIR / name).exists()]
    if missing:
        download_ftp()
    else:
        print("Dados oficiais já existem; download ignorado.")


def inspect_file(
    path: Path, remote_name: str | None = None
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": "ftp",
        "remote_name": remote_name or path.name,
        "path": str(path.relative_to(path.parents[2])),
        "downloaded_at": utc_now(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "downloaded",
    }
    if path.name.endswith((".csv", ".csv.gz")):
        rows = 0
        min_epiweek: int | None = None
        max_epiweek: int | None = None
        for chunk in pd.read_csv(path, chunksize=250_000):
            rows += len(chunk)
            metadata.setdefault("columns", list(chunk.columns))
            if "epiweek" in chunk:
                weeks = pd.to_numeric(
                    chunk["epiweek"], errors="coerce"
                ).dropna()
                if not weeks.empty:
                    chunk_min, chunk_max = int(weeks.min()), int(weeks.max())
                    min_epiweek = (
                        chunk_min
                        if min_epiweek is None
                        else min(min_epiweek, chunk_min)
                    )
                    max_epiweek = (
                        chunk_max
                        if max_epiweek is None
                        else max(max_epiweek, chunk_max)
                    )
        metadata["rows"] = rows
        if min_epiweek is not None:
            metadata["min_epiweek"] = min_epiweek
            metadata["max_epiweek"] = max_epiweek
    return metadata


def main() -> None:
    setup_logging()
    with timed_step("download | FTP"):
        ensure_downloads()
    from .mapbiomas import download_mapbiomas

    with timed_step("download | MapBiomas"):
        download_mapbiomas()


if __name__ == "__main__":
    main()
