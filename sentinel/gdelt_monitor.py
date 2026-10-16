"""GDELT 2.0 15-minute event-file monitor (no LLM, no API quota).

Every 15 minutes GDELT publishes an events export. For each file we compute, per action
country, the share of global news mentions that refer to conflict events (QuadClass 3
verbal conflict, 4 material conflict). Using a share instead of a raw count removes the
day/night and weekday swings in total news volume. A country triggers when its latest
share is far above its own recent history (robust z-score on median/MAD) and the absolute
volume is large enough to matter.
"""

from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

from sentinel import config

BASE_URL = "https://data.gdeltproject.org/gdeltv2"
SLOT = pd.Timedelta(minutes=15)

# GDELT 2.0 event export column positions.
COL = {"actor1": 6, "actor2": 16, "root": 28, "quad": 29, "goldstein": 30, "mentions": 31,
       "country": 53, "place": 52, "added": 59, "url": 60}

# FIPS 10-4 codes used by GDELT for countries most relevant to institutional portfolios.
WATCH = {
    "US": "미국", "CH": "중국", "TW": "대만", "JA": "일본", "KS": "한국", "KN": "북한",
    "RS": "러시아", "UP": "우크라이나", "IR": "이란", "IS": "이스라엘", "SA": "사우디아라비아",
    "IZ": "이라크", "YM": "예멘", "SY": "시리아", "LE": "레바논", "TU": "튀르키예", "IN": "인도",
    "PK": "파키스탄", "UK": "영국", "GM": "독일", "FR": "프랑스", "VE": "베네수엘라", "EG": "이집트",
}
CAMEO_ROOT = {
    "11": "비난", "13": "위협", "14": "시위", "15": "무력 과시", "16": "관계 축소", "17": "강압",
    "18": "공격", "19": "교전", "20": "대량 폭력", "10": "요구", "12": "거부",
}

# Calibrated on two events (2025-06-13 Israel-Iran, 2024-12-03 Korea martial law) and two
# quiet days: requiring two consecutive slots cut quiet-day alerts from about ten per
# twelve hours to at most one, while still detecting both events within about ninety minutes.
GDELT_Z = 4.0
GDELT_PERSIST = 2  # consecutive 15-minute slots above the threshold
GDELT_MIN_MENTIONS = 60  # conflict mentions in the slot, to ignore tiny countries
GDELT_MIN_HISTORY = 80  # slots (about a day; GDELT skips some files) before a country can trigger
GDELT_HISTORY_SLOTS = 672  # keep seven days
COOLDOWN = pd.Timedelta(hours=6)


def slot_of(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).floor("15min")


def export_url(slot: pd.Timestamp) -> str:
    return f"{BASE_URL}/{slot.strftime('%Y%m%d%H%M%S')}.export.CSV.zip"


def latest_slot() -> pd.Timestamp:
    txt = requests.get(f"{BASE_URL}/lastupdate.txt", timeout=config.HTTP_TIMEOUT).text
    name = txt.splitlines()[0].split()[-1].rsplit("/", 1)[-1]
    return pd.Timestamp(name[:14])


def fetch_slot(slot: pd.Timestamp) -> pd.DataFrame | None:
    """Events of one 15-minute file, or None if the file is missing (GDELT skips some)."""
    try:
        resp = requests.get(export_url(slot), timeout=config.HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str, quoting=3,
                     usecols=list(COL.values()), on_bad_lines="skip")
    # usecols returns columns in file order, so map by position rather than dict order.
    df = df.rename(columns={idx: name for name, idx in COL.items()})
    df["quad"] = pd.to_numeric(df["quad"], errors="coerce")
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce").fillna(0)
    df["goldstein"] = pd.to_numeric(df["goldstein"], errors="coerce")
    df["slot"] = slot
    return df


def fetch_slots(slots: list[pd.Timestamp], workers: int = 8) -> dict[pd.Timestamp, pd.DataFrame]:
    """Download several 15-minute files in parallel (static files, no API rate limit)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = dict(zip(slots, ex.map(fetch_slot, slots)))
    return {k: v for k, v in out.items() if v is not None}


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country conflict mentions and their share of all mentions in the slot."""
    total = float(df["mentions"].sum()) or 1.0
    conf = df[df["quad"].isin([3, 4]) & df["country"].notna()]
    g = conf.groupby("country").agg(conflict_mentions=("mentions", "sum"),
                                     material=("quad", lambda q: float((q == 4).mean())),
                                     goldstein=("goldstein", "mean"))
    g["share"] = g["conflict_mentions"] / total
    g["slot"] = df["slot"].iloc[0]
    return g.reset_index()


@dataclass
class GdeltAlert:
    country: str
    name: str
    slot: str
    share: float
    z: float
    conflict_mentions: float
    top_actions: list[str]
    top_actors: list[str]
    sources: list[str]

    def event_text(self) -> str:
        return (
            f"GDELT news spike: conflict-related coverage of {self.name} ({self.country}) jumped "
            f"(robust z {self.z:.1f}). Main actions: {', '.join(self.top_actions)}. "
            f"Main actors: {', '.join(self.top_actors)}. Sample sources: {', '.join(self.sources)}."
        )

    def as_dict(self) -> dict:
        return asdict(self)


def _robust_z(x: float, hist: np.ndarray) -> float:
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med))) * 1.4826
    if mad <= 0:
        mad = float(hist.std()) or 1e-9
    return (x - med) / mad


def _describe(df: pd.DataFrame, country: str) -> tuple[list[str], list[str], list[str]]:
    sub = df[(df["country"] == country) & df["quad"].isin([3, 4])]
    roots = sub.groupby("root")["mentions"].sum().sort_values(ascending=False).head(3)
    actions = [CAMEO_ROOT.get(str(r).zfill(2)[:2], f"CAMEO {r}") for r in roots.index]
    actors = pd.concat([sub["actor1"], sub["actor2"]]).dropna()
    top_actors = actors.value_counts().head(4).index.tolist()
    urls = sub.sort_values("mentions", ascending=False)["url"].dropna().drop_duplicates().head(3)
    domains = [u.split("/")[2] if "//" in u else u for u in urls]
    return actions, top_actors, domains


def detect(history: pd.DataFrame, latest: pd.DataFrame, latest_raw: pd.DataFrame,
           last_alerts: dict[str, str], streak: dict[str, int]) -> tuple[list[GdeltAlert], dict[str, int]]:
    """Compare the latest slot with each watched country's own history.

    Returns (alerts, updated streak). ``streak`` counts consecutive slots above the threshold.
    """
    alerts, new_streak = [], {}
    if latest.empty:
        return alerts, new_streak
    now = pd.Timestamp(latest["slot"].iloc[0])
    n_slots = history.loc[history["slot"] < now, "slot"].nunique() if len(history) else 0
    if n_slots < GDELT_MIN_HISTORY:
        return alerts, new_streak
    for _, row in latest.iterrows():
        c = row["country"]
        if c not in WATCH or row["conflict_mentions"] < GDELT_MIN_MENTIONS:
            continue
        hist = history.loc[(history["country"] == c) & (history["slot"] < now), "share"].to_numpy()
        # Slots where the country had no conflict coverage count as zero share.
        hist = np.concatenate([hist, np.zeros(max(0, n_slots - len(hist)))])
        z = _robust_z(float(row["share"]), hist)
        if z < GDELT_Z:
            continue
        new_streak[c] = streak.get(c, 0) + 1
        if new_streak[c] < GDELT_PERSIST:
            continue
        last = last_alerts.get(c)
        if last and now - pd.Timestamp(last) < COOLDOWN:
            continue
        actions, actors, sources = _describe(latest_raw, c)
        alerts.append(GdeltAlert(c, WATCH[c], str(now), float(row["share"]), float(z),
                                 float(row["conflict_mentions"]), actions, actors, sources))
    return sorted(alerts, key=lambda a: -a.z), new_streak


def trim_history(history: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    return history[history["slot"] > now - GDELT_HISTORY_SLOTS * SLOT].reset_index(drop=True)


def country_scores(history: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    """Robust z of the latest conflict share for every watched country (for the live map).

    Unlike ``detect`` this keeps sub-threshold countries; ``z`` is NaN while history is short.
    """
    cols = ["country", "name", "share", "z", "conflict_mentions", "goldstein"]
    if latest is None or latest.empty:
        return pd.DataFrame(columns=cols)
    now = pd.Timestamp(latest["slot"].iloc[0])
    past = history[history["slot"] < now] if len(history) else history
    n_slots = past["slot"].nunique() if len(past) else 0
    cur = latest.set_index("country")
    rows = []
    for c, name in WATCH.items():
        share = float(cur.at[c, "share"]) if c in cur.index else 0.0
        mentions = float(cur.at[c, "conflict_mentions"]) if c in cur.index else 0.0
        gold = float(cur.at[c, "goldstein"]) if c in cur.index else float("nan")
        z = float("nan")
        if n_slots >= GDELT_MIN_HISTORY:
            hist = past.loc[past["country"] == c, "share"].to_numpy()
            hist = np.concatenate([hist, np.zeros(max(0, n_slots - len(hist)))])
            z = _robust_z(share, hist)
        rows.append({"country": c, "name": name, "share": share, "z": z,
                     "conflict_mentions": mentions, "goldstein": gold})
    return pd.DataFrame(rows, columns=cols)
