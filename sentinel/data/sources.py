"""Raw data fetchers.

Every fetcher returns a pandas object and records provenance (URL, license note, fetch
time) in the cache manifest. Network access is throttled and retried; callers normally go
through ``sentinel.data.cache`` instead of calling these directly.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from sentinel import config

LICENSES = {
    "eia": "U.S. Energy Information Administration. 미국 정부 공공 데이터(public domain), 출처 표기.",
    "fed": "Board of Governors of the Federal Reserve System, H.10/H.15. 미국 정부 공공 데이터, 출처 표기.",
    "cboe": "Cboe Global Markets, VIX Index History. 무료 다운로드, 재배포 조건 확인 필요.",
    "yahoo": "Yahoo Finance chart endpoint (비공식). 연구·비상업 시연 목적, 재배포하지 않음.",
    "gpr": "Caldara & Iacoviello Geopolitical Risk Index. "
    "인용: 'Data downloaded from https://www.matteoiacoviello.com/gpr.htm on <date>'.",
    "gdelt": "The GDELT Project. 출처 표기.",
}
# Sources whose data may be committed to the public repository as a fallback snapshot.
REDISTRIBUTABLE = {"eia", "fed"}


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
    # Some hosts reject unknown custom User-Agents; Yahoo needs a browser-like one.
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


def fetch_eia(series_id: str = "RBRTE") -> tuple[pd.Series, str]:
    url = f"https://www.eia.gov/dnav/pet/hist_xls/{series_id}d.xls"
    resp = _get(url, "eia", 0.5)
    df = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "value"]
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["date"], errors="coerce"), name=series_id).dropna()
    s = s[s.index.notna()]
    if s.empty:
        raise FetchError(f"EIA {series_id}: empty series")
    return s.sort_index(), url


# History from 1998 so that events from 2000 on have the two years of weekly data the betas need.
_FED_PACKAGES = {  # series id -> (release, Data Download Program package id, first date)
    "JRXWTFB_N.B": ("H10", "122e3bcb627e8e53f1bf72a1a09cfb81", "2006-01-01"),  # nominal broad dollar
    "RXI_N.B.KO": ("H10", "60f32914ab61dfab590e0e470153e3ae", "1998-01-01"),  # KRW per USD
    "RIFLGFCY10_N.B": ("H15", "bf17364827e38702b42a58cf8eaa3f78", "1998-01-01"),  # 10y constant maturity
}


def fetch_fed(series_id: str) -> tuple[pd.Series, str]:
    """Federal Reserve Board Data Download Program (H.10 exchange rates, H.15 yields)."""
    rel, pkg, start = _FED_PACKAGES[series_id]
    frm = pd.Timestamp(start).strftime("%m/%d/%Y")
    to = pd.Timestamp.today().strftime("%m/%d/%Y")
    url = (
        f"https://www.federalreserve.gov/datadownload/Output.aspx?rel={rel}"
        f"&series={pkg}&lastobs=&from={frm}&to={to}&filetype=csv&label=include&layout=seriescolumn&type=package"
    )
    resp = _get(url, "fed", 0.5)
    df = pd.read_csv(io.StringIO(resp.text), skiprows=5)
    if series_id not in df.columns:
        raise FetchError(f"Fed {rel} {series_id}: column missing")
    s = pd.Series(pd.to_numeric(df[series_id], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"), name=series_id).dropna()
    if s.empty:
        raise FetchError(f"Fed {rel} {series_id}: empty series")
    return s.sort_index(), url.split("&from")[0]


def fetch_cboe(index: str = "VIX") -> tuple[pd.Series, str]:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
    resp = _get(url, "cboe", 0.5)
    df = pd.read_csv(io.StringIO(resp.text))
    s = pd.Series(pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce"), name=index).dropna()
    if s.empty:
        raise FetchError(f"Cboe {index}: empty series")
    return s.sort_index(), url


def fetch_yahoo(ticker: str, start: str = "1998-01-01") -> tuple[pd.Series, str]:
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
