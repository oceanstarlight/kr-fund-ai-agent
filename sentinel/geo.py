"""Geography for the Shock Map: country coordinates, asset -> country weights, per-country P&L.

Country weights of each ETF come from the issuers' published holdings (see ``ASSET_COUNTRY_SOURCE``)
and are refreshed by hand; the part not listed is kept as ``OTHER`` (not drawn on the map but
counted in totals). Weights are fixed, so the map shows today's geographic mix even when an
older event is replayed.
"""

from __future__ import annotations

import pandas as pd

# ISO 3166 alpha-3 -> (Korean name, lat, lon) of a representative point (capital or centroid).
COUNTRIES: dict[str, tuple[str, float, float]] = {
    "USA": ("미국", 39.0, -98.0), "CAN": ("캐나다", 56.0, -106.0), "MEX": ("멕시코", 23.6, -102.5),
    "BRA": ("브라질", -10.0, -52.0), "CHL": ("칠레", -33.4, -70.6), "COL": ("콜롬비아", 4.6, -74.1),
    "PER": ("페루", -12.0, -77.0), "ARG": ("아르헨티나", -34.6, -58.4), "VEN": ("베네수엘라", 10.5, -66.9),
    "DOM": ("도미니카공화국", 18.5, -69.9), "PAN": ("파나마", 9.0, -79.5), "URY": ("우루과이", -34.9, -56.2),
    "ECU": ("에콰도르", -0.2, -78.5), "GBR": ("영국", 52.5, -1.5), "IRL": ("아일랜드", 53.3, -7.7),
    "FRA": ("프랑스", 46.6, 2.4), "DEU": ("독일", 51.2, 10.4), "NLD": ("네덜란드", 52.2, 5.3),
    "BEL": ("벨기에", 50.6, 4.6), "CHE": ("스위스", 46.8, 8.2), "AUT": ("오스트리아", 47.6, 14.1),
    "ITA": ("이탈리아", 42.8, 12.6), "ESP": ("스페인", 40.2, -3.7), "PRT": ("포르투갈", 39.6, -8.0),
    "DNK": ("덴마크", 56.0, 9.5), "SWE": ("스웨덴", 62.0, 15.0), "NOR": ("노르웨이", 61.0, 9.0),
    "FIN": ("핀란드", 62.5, 26.0), "POL": ("폴란드", 52.0, 19.4), "ROU": ("루마니아", 45.9, 24.9),
    "HUN": ("헝가리", 47.2, 19.5), "CZE": ("체코", 49.8, 15.5), "GRC": ("그리스", 39.0, 22.0),
    "TUR": ("튀르키예", 39.0, 35.2), "UKR": ("우크라이나", 49.0, 31.4), "RUS": ("러시아", 56.0, 38.0),
    "ISR": ("이스라엘", 31.5, 34.9), "LBN": ("레바논", 33.9, 35.9), "SYR": ("시리아", 35.0, 38.5),
    "IRQ": ("이라크", 33.2, 43.7), "IRN": ("이란", 32.4, 53.7), "SAU": ("사우디아라비아", 24.0, 45.0),
    "ARE": ("아랍에미리트", 24.3, 54.4), "QAT": ("카타르", 25.3, 51.2), "KWT": ("쿠웨이트", 29.3, 47.5),
    "OMN": ("오만", 21.5, 56.0), "BHR": ("바레인", 26.0, 50.5), "YEM": ("예멘", 15.6, 48.0),
    "EGY": ("이집트", 26.8, 30.8), "NGA": ("나이지리아", 9.1, 8.7), "ZAF": ("남아프리카공화국", -29.0, 25.0),
    "KEN": ("케냐", 0.0, 37.9), "MAR": ("모로코", 31.8, -7.1), "IND": ("인도", 22.0, 79.0),
    "PAK": ("파키스탄", 30.4, 69.3), "CHN": ("중국", 35.0, 103.0), "HKG": ("홍콩", 22.3, 114.2),
    "TWN": ("대만", 23.7, 121.0), "KOR": ("한국", 36.5, 127.8), "PRK": ("북한", 40.3, 127.4),
    "JPN": ("일본", 36.2, 138.3), "IDN": ("인도네시아", -2.5, 118.0), "MYS": ("말레이시아", 4.2, 102.0),
    "THA": ("태국", 15.9, 100.9), "PHL": ("필리핀", 12.9, 121.8), "VNM": ("베트남", 14.1, 108.3),
    "SGP": ("싱가포르", 1.35, 103.8), "AUS": ("호주", -25.3, 133.8), "NZL": ("뉴질랜드", -41.0, 174.0),
}

# GDELT uses FIPS 10-4 country codes.
FIPS_TO_ISO3 = {
    "US": "USA", "CH": "CHN", "TW": "TWN", "JA": "JPN", "KS": "KOR", "KN": "PRK", "RS": "RUS",
    "UP": "UKR", "IR": "IRN", "IS": "ISR", "SA": "SAU", "IZ": "IRQ", "YM": "YEM", "SY": "SYR",
    "LE": "LBN", "TU": "TUR", "IN": "IND", "PK": "PAK", "UK": "GBR", "GM": "DEU", "FR": "FRA",
    "VE": "VEN", "EG": "EGY",
}

# Percent of fund by issuer country. The unlisted remainder becomes OTHER (not drawn).
_US = {"USA": 100.0}
ASSET_COUNTRIES: dict[str, dict[str, float]] = {
    "EFA": {"JPN": 24.08, "GBR": 14.74, "CHE": 9.47, "FRA": 9.32, "DEU": 8.67, "AUS": 6.46, "NLD": 5.66,
            "ESP": 3.93, "SWE": 3.44, "ITA": 3.12, "HKG": 1.79, "SGP": 1.78},
    "EEM": {"TWN": 29.64, "CHN": 20.05, "KOR": 19.13, "IND": 10.98, "BRA": 4.79, "ZAF": 2.93, "SAU": 2.29,
            "MEX": 1.62, "ARE": 1.24, "POL": 1.17},
    "EMB": {"SAU": 6.44, "MEX": 6.25, "TUR": 4.93, "IDN": 4.32, "BRA": 3.95, "POL": 3.62, "DOM": 3.49,
            "PHL": 3.45, "COL": 3.39, "CHL": 3.36, "HUN": 3.36, "ROU": 3.33, "CHN": 3.32, "ZAF": 3.27,
            "ARG": 3.25, "PAN": 3.22, "BHR": 3.15, "PER": 3.05, "EGY": 3.05, "NGA": 2.61, "OMN": 2.54,
            "URY": 2.16, "MYS": 2.01, "ECU": 1.82, "UKR": 1.69},
    "IGF": {"USA": 37.40, "ESP": 9.01, "CAN": 7.73, "MEX": 6.92, "FRA": 6.72, "AUS": 6.38, "CHN": 5.09,
            "NZL": 3.52, "DEU": 3.47, "GBR": 3.45, "ITA": 2.79, "CHE": 1.99, "SGP": 1.82, "JPN": 1.47},
    "PSP": {"USA": 40.0, "GBR": 21.0, "SWE": 7.0},
    "EWY": {"KOR": 100.0}, "KRW_CASH": {"KOR": 100.0},
    "SPY": _US, "VNQ": _US, "IEF": _US, "TLT": _US, "TIP": _US, "BIL": _US,
    "LQD": _US, "HYG": _US, "BKLN": _US,
    # GLD (physical gold) and QAI (hedge fund replication) have no meaningful country: OTHER.
}
ASSET_COUNTRY_SOURCE: dict[str, str] = {
    "EFA": "iShares 상품 페이지 국가 비중 (2026-10-15)",
    "EEM": "iShares 상품 페이지 국가 비중 (2026-10-14)",
    "EMB": "iShares 상품 페이지 발행자 국가 비중, 상위 25개국 (2026-10-14)",
    "IGF": "iShares 상품 페이지 국가 비중 (2026-10-14)",
    "PSP": "제3자 자료(Pensions & Investments, 2026-07-17), 운용사 원자료 미확인 → 근사치",
    "SPY/VNQ/IEF/TLT/TIP/BIL": "지수 정의상 미국 (국채·미국 주식·미국 리츠)",
    "LQD/HYG/BKLN": "미국 달러 회사채·대출 지수. 해외 발행자가 일부 포함되나 미국으로 근사",
    "EWY/KRW_CASH": "한국",
    "GLD/QAI": "국가 없음(금, 헤지펀드 복제 전략) → 지도에 표시하지 않음",
}


def country_weights(ticker: str) -> dict[str, float]:
    """Fractions summing to one, including OTHER for the unlisted remainder."""
    raw = ASSET_COUNTRIES.get(ticker)
    if not raw:
        return {"OTHER": 1.0}
    total = sum(raw.values())
    out = {c: v / 100.0 for c, v in raw.items()}
    if total < 100.0:
        out["OTHER"] = out.get("OTHER", 0.0) + (100.0 - total) / 100.0
    s = sum(out.values())
    return {c: v / s for c, v in out.items()}


def exposure_matrix(tickers: list[str]) -> pd.DataFrame:
    """tickers x countries matrix of fractions (rows sum to one)."""
    return pd.DataFrame({t: country_weights(t) for t in tickers}).T.fillna(0.0)


def country_exposure(weights: dict[str, float]) -> pd.Series:
    """Portfolio weight held in each country."""
    m = exposure_matrix(list(weights))
    return (m.mul(pd.Series(weights), axis=0)).sum().sort_values(ascending=False)


def country_pnl(weights: dict[str, float], asset_returns: pd.Series, aum: float = 1.0) -> pd.Series:
    """P&L by country: sum over assets of weight x return x country share (x AUM)."""
    r = asset_returns.reindex(list(weights)).fillna(0.0)
    m = exposure_matrix(list(weights))
    return m.mul(pd.Series(weights) * r * aum, axis=0).sum()


def located(series: pd.Series) -> pd.DataFrame:
    """Attach names and coordinates; drops OTHER and unknown codes."""
    rows = [{"iso3": c, "name": COUNTRIES[c][0], "lat": COUNTRIES[c][1], "lon": COUNTRIES[c][2], "value": float(v)}
            for c, v in series.items() if c in COUNTRIES]
    return pd.DataFrame(rows, columns=["iso3", "name", "lat", "lon", "value"])
