"""Shock Map figures (Plotly scattergeo): live tension, event replay, event library.

Design rules (docs/ideas.md): marker AREA, not radius, is proportional to the value; shocks
travel through exposures, not geography, so arcs link the event to the countries the
portfolio holds; the map stays plain (borders and a flat background only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sentinel import geo

# Diverging polarity (gain blue / loss red) and status colors, always shown with labels.
GAIN, LOSS = "#2a78d6", "#e34948"
CRITICAL, WARNING, NEUTRAL = "#d03b3b", "#fab219", "#8a8984"
EVENT = "#eb6834"
MAX_PX = 46  # diameter of the largest marker
MAX_ARCS = 12
MAX_COUNTRIES = 15  # largest country P&L circles drawn; the rest stay in the table

# Event types folded into six groups, fixed categorical order.
TYPE_GROUP = {
    "war": "무력 충돌", "military_strike": "무력 충돌", "trade": "무역", "energy_supply": "에너지",
    "financial_stress": "금융·환율", "fx_shock": "금융·환율", "political": "정치",
    "pandemic": "재난·보건", "disaster": "재난·보건",
}
GROUP_COLOR = {"무력 충돌": "#e34948", "에너지": "#eda100", "무역": "#2a78d6",
               "금융·환율": "#6250d6", "정치": "#1baf7a", "재난·보건": "#e87ba4"}

# Representative point of a planner region when the event has no coordinates (Ask mode).
REGION_POINT = {
    "middle_east": (29.0, 47.0), "europe": (50.0, 15.0), "east_asia": (33.0, 120.0),
    "korea": (37.6, 127.0), "north_america": (39.0, -98.0), "emerging": (5.0, 20.0),
}


def _sizeref(values: np.ndarray) -> float:
    m = float(np.nanmax(np.abs(values))) if len(values) and np.isfinite(values).any() else 0.0
    return 2.0 * m / MAX_PX**2 if m > 0 else 1.0


def _layout(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#c9c8c2",
                    showcoastlines=False, showland=True, landcolor="#ecebe7", showocean=False,
                    showframe=False, bgcolor="rgba(0,0,0,0)", lataxis_range=[-50, 75])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.02, x=0, bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
                      hoverlabel=dict(bgcolor="white"))
    return fig


def money(v: float, currency: str = "USD") -> str:
    a = abs(v)
    s = f"{a / 1e9:,.2f}B" if a >= 1e9 else f"{a / 1e6:,.1f}M" if a >= 1e6 else f"{a:,.0f}"
    return f"{'-' if v < 0 else '+'}{currency} {s}"


# ------------------------------------------------------------------------------ traces
def pnl_traces(pnl: pd.Series, origin: tuple[float, float] | None, sizeref: float, currency: str,
               label: str, hollow: bool = False, show_legend: bool = True) -> list[go.Scattergeo]:
    """Country P&L circles (area ~ |P&L|) and arcs from the event origin (width ~ |P&L|)."""
    df = geo.located(pnl)
    df = df[df["value"].abs() > 0].sort_values("value", key=abs, ascending=False).head(MAX_COUNTRIES)
    traces: list[go.Scattergeo] = []
    top = float(df["value"].abs().max()) if len(df) else 1.0
    if origin is not None and not hollow:
        # Always MAX_ARCS traces (padded) so animation frames keep the same trace layout.
        big = df.sort_values("value", key=abs, ascending=False).head(MAX_ARCS)
        for _, r in big.iterrows():
            traces.append(go.Scattergeo(
                lat=[origin[0], r["lat"]], lon=[origin[1], r["lon"]], mode="lines", hoverinfo="skip",
                line=dict(width=0.8 + 4.0 * abs(r["value"]) / top, color=LOSS if r["value"] < 0 else GAIN),
                opacity=0.45, showlegend=False))
        traces += [go.Scattergeo(lat=[], lon=[], mode="lines", showlegend=False)] * (MAX_ARCS - len(big))
    for sign, color, name in [(-1, LOSS, f"{label} 손실"), (1, GAIN, f"{label} 이익")]:
        part = df[np.sign(df["value"]) == sign]
        traces.append(go.Scattergeo(
            lat=part["lat"], lon=part["lon"], mode="markers", name=name, showlegend=show_legend,
            marker=dict(size=part["value"].abs(), sizemode="area", sizeref=sizeref, sizemin=3,
                        color="rgba(0,0,0,0)" if hollow else color,
                        line=dict(width=2 if hollow else 1, color=color if hollow else "white"),
                        opacity=1.0 if hollow else 0.8),
            text=[f"{n}: {money(v, currency)}" for n, v in zip(part["name"], part["value"])],
            hovertemplate="%{text}<extra>" + label + "</extra>"))
    return traces


def ripple(lat: float, lon: float, strength: float, name: str, color: str = EVENT) -> list[go.Scattergeo]:
    """Concentric rings at the event location; ring area grows with signal strength."""
    base = 10 + 6 * np.sqrt(max(strength, 0.0))
    out = [go.Scattergeo(lat=[lat], lon=[lon], mode="markers", hoverinfo="skip", showlegend=False,
                         marker=dict(size=base * k, color="rgba(0,0,0,0)", line=dict(width=1.5, color=color),
                                     opacity=op)) for k, op in [(2.2, 0.25), (1.6, 0.45)]]
    out.append(go.Scattergeo(lat=[lat], lon=[lon], mode="markers", name=name,
                             marker=dict(size=base, color=color, line=dict(width=1, color="white")),
                             text=[name], hovertemplate="%{text}<extra>사건</extra>"))
    return out


# ------------------------------------------------------------------------------ figures
def live_map(scores: pd.DataFrame, alerts: list, arcs: dict[str, pd.Series] | None = None,
             z_threshold: float = 4.0, min_mentions: float = 60, currency: str = "USD") -> go.Figure:
    """Current conflict-news tension per watched country plus exposure arcs of alerted ones."""
    fig = go.Figure()
    alerted = {a.country for a in alerts}
    df = scores.copy()
    df["iso3"] = df["country"].map(geo.FIPS_TO_ISO3)
    df = df[df["iso3"].isin(geo.COUNTRIES)]
    df["lat"] = df["iso3"].map(lambda c: geo.COUNTRIES[c][1])
    df["lon"] = df["iso3"].map(lambda c: geo.COUNTRIES[c][2])
    df["zc"] = df["z"].clip(lower=0).fillna(0) + 0.3

    def status(r):
        if r["country"] in alerted:
            return "경보"
        if r["z"] >= z_threshold and r["conflict_mentions"] < min_mentions:
            return "급등(보도량 적음)"
        if r["z"] >= z_threshold:
            return "급등(지속 확인 중)"
        return "평상"

    df["status"] = df.apply(status, axis=1) if len(df) else []
    for country, series in (arcs or {}).items():
        iso = geo.FIPS_TO_ISO3.get(country)
        if iso in geo.COUNTRIES and len(series):
            o = geo.COUNTRIES[iso]
            for t in pnl_traces(series, (o[1], o[2]), _sizeref(series.to_numpy()), currency,
                                f"{o[0]} 경보 시 예상", show_legend=True):
                fig.add_trace(t)
    # Fixed scale (largest circle at z = twice the threshold), so a quiet day looks quiet.
    sref = _sizeref(np.append(df["zc"].to_numpy(), 2 * z_threshold))
    for st_name, color in [("평상", NEUTRAL), ("급등(보도량 적음)", WARNING), ("급등(지속 확인 중)", WARNING),
                           ("경보", CRITICAL)]:
        part = df[df["status"] == st_name] if len(df) else df
        if part.empty:
            continue
        if st_name == "경보":
            for _, r in part.iterrows():
                for t in ripple(r["lat"], r["lon"], r["z"], r["name"], CRITICAL)[:-1]:
                    fig.add_trace(t)
        fig.add_trace(go.Scattergeo(
            lat=part["lat"], lon=part["lon"], mode="markers", name=st_name,
            marker=dict(size=part["zc"], sizemode="area", sizeref=sref, sizemin=4, color=color,
                        opacity=0.85 if st_name != "평상" else 0.5, line=dict(width=1, color="white"),
                        symbol="circle" if st_name != "급등(보도량 적음)" else "circle-dot"),
            text=[f"{r['name']}<br>z {r['z']:+.1f} · 분쟁 보도 {r['conflict_mentions']:.0f}건"
                  if np.isfinite(r["z"]) else f"{r['name']}<br>이력 축적 중" for _, r in part.iterrows()],
            hovertemplate="%{text}<extra>" + st_name + "</extra>"))
    return _layout(fig)


def replay_frames(md, weights: dict[str, float], t0: pd.Timestamp, horizon: int = 20) -> dict[int, pd.Series]:
    """Realized per-asset return from the base close to each of D+1..D+horizon (if observed)."""
    out = {}
    for d in range(1, horizon + 1):
        w = md.window(t0, d)
        if w is None:
            break
        base, end = w
        out[d] = md.asset_returns(base, end).reindex(list(weights))
    return out


def replay_map(state: dict, md, aum: float, currency: str) -> go.Figure | None:
    """Animated replay: realized country P&L day by day, prediction rings at D+1/D+5/D+20."""
    ev, weights = state.get("event"), state["weights"]
    origin = event_origin(state)
    preds = {h: geo.country_pnl(weights, p.asset_blend_median, aum) for h, p in state["impacts"].items()}
    days = replay_frames(md, weights, ev.t0, max(state["impacts"])) if ev is not None else {}
    if not days:
        return static_map(state, aum, currency)
    real = {d: geo.country_pnl(weights, r, aum) for d, r in days.items()}
    allv = np.concatenate([s.to_numpy() for s in list(real.values()) + list(preds.values())])
    sref = _sizeref(allv)
    strength = signal_strength(state)
    hs = sorted(preds)

    def frame_traces(d: int) -> list:
        h = max([x for x in hs if x <= d], default=hs[0])
        tr = []
        if origin:
            tr += ripple(origin[0], origin[1], strength, origin[2])
        tr += pnl_traces(real[d], origin[:2] if origin else None, sref, currency, f"실측 D+{d}")
        tr += pnl_traces(preds[h], None, sref, currency, f"예측 D+{h}", hollow=True)
        return tr

    first = frame_traces(1)
    fig = go.Figure(data=first)
    frames = []
    for d in real:
        frames.append(go.Frame(data=frame_traces(d), name=f"D+{d}"))
    fig.frames = frames
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for f in frames]
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0, y=1.08, xanchor="left", showactive=False, buttons=[
            dict(label="▶ 재생", method="animate",
                 args=[None, dict(frame=dict(duration=450, redraw=True), fromcurrent=True)])])],
        sliders=[dict(active=0, steps=steps, x=0.08, len=0.9, y=0, currentvalue=dict(prefix="거래일 "))])
    _layout(fig, 540)
    fig.update_layout(legend=dict(y=1.07, x=0.1, yanchor="bottom"), margin=dict(l=0, r=0, t=50, b=40))
    return fig


def static_map(state: dict, aum: float, currency: str, horizon: int | None = None) -> go.Figure:
    """Predicted country P&L at one horizon (Ask mode, or when nothing is realized yet)."""
    from sentinel import config

    h = horizon or config.PRIMARY_HORIZON
    p = state["impacts"][h]
    pnl = geo.country_pnl(state["weights"], p.asset_blend_median, aum)
    origin = event_origin(state)
    fig = go.Figure()
    if origin:
        for t in ripple(origin[0], origin[1], signal_strength(state), origin[2]):
            fig.add_trace(t)
    for t in pnl_traces(pnl, origin[:2] if origin else None, _sizeref(pnl.to_numpy()), currency, f"예측 D+{h}"):
        fig.add_trace(t)
    return _layout(fig)


def library_map(rows: pd.DataFrame) -> go.Figure:
    """Event library: one circle per event, area ~ |realized portfolio return|, color = type group.

    ``rows`` columns: id, name, date, type, lat, lon, realized (fraction).
    """
    fig = go.Figure()
    df = rows.dropna(subset=["lat", "lon"]).copy()
    df["group"] = df["type"].map(TYPE_GROUP).fillna("정치")
    df["mag"] = df["realized"].abs().fillna(0) + 1e-4
    sref = _sizeref(df["mag"].to_numpy())
    for grp, color in GROUP_COLOR.items():
        part = df[df["group"] == grp]
        if part.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=part["lat"], lon=part["lon"], mode="markers", name=grp, customdata=part["id"],
            marker=dict(size=part["mag"], sizemode="area", sizeref=sref, sizemin=5, color=color, opacity=0.8,
                        line=dict(width=1.5, color=[LOSS if v < 0 else GAIN for v in part["realized"].fillna(0)])),
            text=[f"{r['date']} {r['name']}<br>당시 포트폴리오 실측 {r['realized']:+.2%}"
                  if pd.notna(r["realized"]) else f"{r['date']} {r['name']}<br>실측 없음" for _, r in part.iterrows()],
            hovertemplate="%{text}<extra>" + grp + "</extra>"))
    return _layout(fig, 480)


# ------------------------------------------------------------------------------ helpers
def event_origin(state: dict) -> tuple[float, float, str] | None:
    ev = state.get("event")
    if ev is not None and ev.lat is not None:
        return float(ev.lat), float(ev.lon), ev.name
    cls = state.get("classification") or {}
    pt = REGION_POINT.get(cls.get("region", ""))
    if pt:
        return pt[0], pt[1], cls.get("headline_ko") or "입력 사건"
    return None


def signal_strength(state: dict) -> float:
    """Largest |z| among detection signals (0 when nothing was measured, e.g. Ask mode)."""
    det = state.get("detection") or {}
    zs = [abs(s.get("z", 0.0)) for s in det.get("signals", []) if s.get("z") is not None and np.isfinite(s["z"])]
    return float(max(zs)) if zs and state.get("mode") != "ask" else 0.0


def library_rows(md, weights: dict[str, float], horizon: int = 5) -> pd.DataFrame:
    """Event library with the current portfolio's realized return over ``horizon`` days."""
    from sentinel.events.library import load_events
    from sentinel.quant.engines import realized

    rows = []
    for e in load_events():
        r = realized(md, weights, e.t0, horizon) if md.trading_day_before(e.t0) is not None else None
        rows.append({"id": e.id, "name": e.name, "date": str(pd.Timestamp(e.date).date()), "type": e.type,
                     "lat": e.lat, "lon": e.lon, "realized": r["portfolio"] if r else float("nan")})
    return pd.DataFrame(rows)
