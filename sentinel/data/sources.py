"""Raw data fetchers.

Every fetcher returns a pandas object and records provenance (URL, license note, fetch
time) in the cache manifest. Network access is throttled and retried; callers normally go
through ``sentinel.data.cache`` instead of calling these directly.
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from sentinel import config

LICENSES = {
    "fred": "FRED (St. Louis Fed). 출처 표기, 시리즈별 저작권 고지 유지. "
    "AI/ML 개발·학습 이용 금지 조항에 따라 원시 데이터는 수치 계산에만 사용.",
    "yahoo": "Yahoo Finance chart endpoint (비공식). 연구·비상업 시연 목적, 재배포하지 않음.",
    "gpr": "Caldara & Iacoviello Geopolitical Risk Index. "
    "인용: 'Data downloaded from https://www.matteoiacoviello.com/gpr.htm on <date>'.",
    "gdelt": "The GDELT Project. 출처 표기.",
}


class FetchError(RuntimeError):
    pass


_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    wait = _last_call.get(host, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def _get(
    url: str, host: str, min_interval: float, retries: int = 4, browser_ua: bool = False
) -> requests.Response:
    # FRED rejects unknown custom User-Agents; Yahoo rejects non-browser ones.
    headers = {"User-Agent": config.BROWSER_USER_AGENT} if browser_ua else {}
    delay = 2.0
    for attempt in range(retries):
        _throttle(host, min_interval)
        try:
            resp = requests.get(
                url, headers=headers, timeout=config.HTTP_TIMEOUT
            )
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise FetchError(f"{url}: {exc}") from exc
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise FetchError(f"{url}: HTTP {resp.status_code}")
        time.sleep(delay)
        delay *= 2
    raise FetchError(url)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_fred(series_id: str) -> tuple[pd.Series, str]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = _get(url, "fred", 0.5)
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index("date")["value"].rename(series_id)
    if s.empty:
        raise FetchError(f"FRED {series_id}: empty series")
    return s, url


def fetch_yahoo(ticker: str, start: str = "2000-01-01") -> tuple[pd.Series, str]:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(time.time())
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div,splits&includeAdjustedClose=true"
    )
    resp = _get(url, "yahoo", 1.5, browser_ua=True)
    try:
        result = resp.json()["chart"]["result"][0]
        ts = result["timestamp"]
        adj = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise FetchError(f"Yahoo {ticker}: unexpected payload") from exc
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize()
    s = pd.Series(adj, index=idx.tz_localize(None), name=ticker, dtype="float64").dropna()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        raise FetchError(f"Yahoo {ticker}: empty series")
    return s, url.split("?")[0]


def fetch_gpr_daily() -> tuple[pd.DataFrame, str]:
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    resp = _get(url, "gpr", 1.0)
    df = pd.read_excel(io.BytesIO(resp.content))
    df["date"] = pd.to_datetime(df["DAY"].astype(int).astype(str), format="%Y%m%d")
    out = df.set_index("date")[["GPRD", "GPRD_ACT", "GPRD_THREAT"]].astype(float)
    return out, url


def fetch_gdelt_timeline(query: str, timespan: str = "7d") -> tuple[pd.Series, str]:
    """Article-volume share timeline for a query (live monitoring only)."""
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={requests.utils.quote(query)}&mode=timelinevol&format=json&timespan={timespan}"
    )
    resp = _get(url, "gdelt", 6.0)
    try:
        data = resp.json()["timeline"][0]["data"]
    except (KeyError, IndexError, ValueError) as exc:
        raise FetchError("GDELT: unexpected payload") from exc
    s = pd.Series(
        [d["value"] for d in data],
        index=pd.to_datetime([d["date"] for d in data], format="%Y%m%dT%H%M%SZ"),
        name=query,
    )
    return s, url.split("&timespan")[0]
