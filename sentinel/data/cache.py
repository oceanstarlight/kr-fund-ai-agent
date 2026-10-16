"""File cache with a provenance manifest and a committed fallback snapshot.

Layout: ``<CACHE_DIR>/<source>/<id>.csv`` plus ``manifest.json`` recording, for each
entry, the source URL, license note, fetch time and row count. This is the record the
contest rules ask for (source, collection date, license, preprocessing).

Lookup order for a series: fresh cache -> network -> committed snapshot (public-domain
sources only, ``data/snapshot``) -> stale cache. Every call reports which one was used, so
the app can keep running and show what is missing instead of failing.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentinel import config
from sentinel.data import sources

SNAPSHOT_DIR = config.DATA_DIR / "snapshot"
_manifest_lock = threading.Lock()

FETCHERS = {
    "yahoo": (sources.fetch_yahoo, "수정종가(배당·분할 반영), 뉴욕 기준 일자"),
    "eia": (sources.fetch_eia, "Data 1 시트, 결측 제거"),
    "fed": (sources.fetch_fed, "Data Download Program CSV, 'ND'(미발표) 제거"),
    "cboe": (sources.fetch_cboe, "종가(CLOSE) 열"),
}


@dataclass
class SeriesStatus:
    source: str
    key: str
    origin: str  # "cache" | "network" | "snapshot" | "stale-cache" | "failed"
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.origin != "failed"


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def _path(source: str, key: str, root: Path | None = None) -> Path:
    return (root or config.CACHE_DIR) / source / f"{_safe(key)}.csv"


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _record(source: str, key: str, url: str, rows: int, note: str = "") -> None:
    with _manifest_lock:
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


def fetch_series(
    source: str, key: str, max_age_hours: float | None = 24, offline: bool = False
) -> tuple[pd.Series | None, SeriesStatus]:
    """Never raises: returns (series or None, status)."""
    path = _path(source, key)
    snap = _path(source, key, SNAPSHOT_DIR)
    if _fresh(path, max_age_hours) or (offline and path.exists()):
        return _read(path).iloc[:, 0].rename(key), SeriesStatus(source, key, "cache")
    error = "offline mode"
    if not offline:
        fetch, note = FETCHERS[source]
        try:
            s, url = fetch(key)
            _write(path, s)
            _record(source, key, url, len(s), note)
            return s.rename(key), SeriesStatus(source, key, "network")
        except Exception as exc:  # noqa: BLE001 - one failing source must not stop the app
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
    if snap.exists():
        return _read(snap).iloc[:, 0].rename(key), SeriesStatus(source, key, "snapshot", error)
    if path.exists():
        return _read(path).iloc[:, 0].rename(key), SeriesStatus(source, key, "stale-cache", error)
    return None, SeriesStatus(source, key, "failed", error)


def get_series(source: str, key: str, max_age_hours: float | None = 24, offline: bool = False) -> pd.Series:
    """Raising variant kept for scripts that need a hard failure."""
    s, status = fetch_series(source, key, max_age_hours, offline)
    if s is None:
        raise sources.FetchError(f"{source}:{key} unavailable ({status.error})")
    return s


def get_gpr(max_age_hours: float | None = 24, offline: bool = False) -> tuple[pd.DataFrame | None, SeriesStatus]:
    path = _path("gpr", "daily")
    if _fresh(path, max_age_hours) or (offline and path.exists()):
        return _read(path), SeriesStatus("gpr", "daily", "cache")
    error = "offline mode"
    if not offline:
        try:
            df, url = sources.fetch_gpr_daily()
            _write(path, df)
            _record("gpr", "daily", url, len(df), "GPRD, GPRD_ACT, GPRD_THREAT 컬럼만 사용")
            return df, SeriesStatus("gpr", "daily", "network")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
    if path.exists():
        return _read(path), SeriesStatus("gpr", "daily", "stale-cache", error)
    return None, SeriesStatus("gpr", "daily", "failed", error)


def write_snapshot(source: str, key: str) -> bool:
    """Copy a cached public-domain series into the committed snapshot directory."""
    if source not in sources.REDISTRIBUTABLE:
        return False
    path = _path(source, key)
    if not path.exists():
        return False
    _write(_path(source, key, SNAPSHOT_DIR), _read(path).iloc[:, 0])
    return True


def cache_exists() -> bool:
    return (config.CACHE_DIR / "yahoo").exists() and any((config.CACHE_DIR / "yahoo").glob("*.csv"))
