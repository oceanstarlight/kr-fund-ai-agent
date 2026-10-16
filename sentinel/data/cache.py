"""File cache with a provenance manifest.

Layout: ``<CACHE_DIR>/<source>/<id>.csv`` plus ``manifest.json`` recording, for each
entry, the source URL, license note, fetch time and row count. This is the record the
contest rules ask for (source, collection date, license, preprocessing).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentinel import config
from sentinel.data import sources


def _path(source: str, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return config.CACHE_DIR / source / f"{safe}.csv"


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _record(source: str, key: str, url: str, rows: int, note: str = "") -> None:
    manifest = load_manifest()
    manifest[f"{source}:{key}"] = {
        "source": source,
        "id": key,
        "url": url,
        "license": sources.LICENSES.get(source, ""),
        "fetched_at": sources.now_utc(),
        "rows": rows,
        "preprocessing": note,
    }
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def _fresh(path: Path, max_age_hours: float | None) -> bool:
    if not path.exists():
        return False
    if max_age_hours is None:
        return True
    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age < max_age_hours * 3600


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _write(path: Path, obj: pd.Series | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.to_csv(path)


def get_series(
    source: str, key: str, max_age_hours: float | None = 24, offline: bool = False
) -> pd.Series:
    """Return a cached series, refreshing it from the network when stale.

    On network failure a stale cache entry is used if present (snapshot fallback).
    """
    path = _path(source, key)
    if offline or _fresh(path, max_age_hours):
        if not path.exists():
            raise sources.FetchError(f"No cached data for {source}:{key} (offline mode)")
        return _read(path).iloc[:, 0].rename(key)
    try:
        if source == "fred":
            s, url = sources.fetch_fred(key)
            note = "결측('.') 제거"
        elif source == "yahoo":
            s, url = sources.fetch_yahoo(key)
            note = "수정종가(배당·분할 반영), 뉴욕 기준 일자"
        else:
            raise ValueError(f"Unknown series source: {source}")
    except sources.FetchError:
        if path.exists():
            return _read(path).iloc[:, 0].rename(key)
        raise
    _write(path, s)
    _record(source, key, url, len(s), note)
    return s


def get_gpr(max_age_hours: float | None = 24, offline: bool = False) -> pd.DataFrame:
    path = _path("gpr", "daily")
    if offline or _fresh(path, max_age_hours):
        if not path.exists():
            raise sources.FetchError("No cached GPR data (offline mode)")
        return _read(path)
    try:
        df, url = sources.fetch_gpr_daily()
    except sources.FetchError:
        if path.exists():
            return _read(path)
        raise
    _write(path, df)
    _record("gpr", "daily", url, len(df), "GPRD, GPRD_ACT, GPRD_THREAT 컬럼만 사용")
    return df
