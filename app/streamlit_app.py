"""MACRO SENTINEL - Streamlit UI.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import config  # noqa: E402
from sentinel.agents.llm import llm_available  # noqa: E402
from sentinel.agents.pipeline import run  # noqa: E402
from sentinel.backtest import run_backtest, summarize  # noqa: E402
from sentinel.data.cache import load_manifest  # noqa: E402
from sentinel.data.store import MarketData  # noqa: E402
from sentinel.data.universe import ASSETS, FACTORS, PRESETS, normalize  # noqa: E402
from sentinel.events.library import load_events  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1-3, text inks.
C_A, C_B, C_REAL = "#2a78d6", "#eb6834", "#1baf7a"
C_GAIN, C_LOSS = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"

st.set_page_config(page_title="MACRO SENTINEL", layout="wide")


@st.cache_resource(show_spinner="시장 데이터 불러오는 중...")
def get_data(offline: bool) -> MarketData:
    return MarketData.load(offline=offline, max_age_hours=None if offline else 24)


def pct(x: float) -> str:
    return "-" if x != x else f"{x:+.2%}"


# ------------------------------------------------------------------------ sidebar
st.sidebar.title("MACRO SENTINEL")
st.sidebar.caption("지정학·매크로 충격 감시 및 포트폴리오 영향 추정 Agent (프로토타입)")

offline = st.sidebar.toggle("캐시 데이터만 사용 (오프라인)", value=True)
try:
    md = get_data(offline)
except Exception as exc:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했어요: {exc}\n\n`python scripts/build_snapshot.py` 로 캐시를 먼저 만들어 주세요.")
    st.stop()
st.sidebar.caption(f"데이터 기준일: {md.last_date.date()}")

preset_key = st.sidebar.selectbox(
    "포트폴리오", list(PRESETS) + ["upload"],
    format_func=lambda k: PRESETS[k].name if k in PRESETS else "CSV 업로드 (ticker, weight)",
)
if preset_key == "upload":
    up = st.sidebar.file_uploader("CSV", type="csv")
    if up is None:
        st.sidebar.info(f"사용 가능 자산: {', '.join(ASSETS)}")
        st.stop()
    df_up = pd.read_csv(io.BytesIO(up.getvalue()))
    try:
        weights = normalize(dict(zip(df_up.iloc[:, 0].astype(str).str.upper(), df_up.iloc[:, 1].astype(float))))
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    preset_note = "사용자 업로드"
else:
    weights = PRESETS[preset_key].weights
    preset_note = f"{PRESETS[preset_key].source}. {PRESETS[preset_key].note}"

currency = st.sidebar.radio("통화", ["USD", "KRW"], horizontal=True)
aum = st.sidebar.number_input(
    "운용자산 규모", min_value=0.0,
    value=232e9 if currency == "USD" else 1684.2e12, step=1e9, format="%.0f",
)
with st.sidebar.expander("포트폴리오 구성"):
    st.dataframe(
        pd.DataFrame({"자산": [ASSETS[t].name for t in weights], "티커": list(weights), "비중": [f"{w:.1%}" for w in weights.values()]}),
        hide_index=True,
    )
    st.caption(preset_note)
st.sidebar.markdown(
    f"**LLM**: {'연결됨 (분류 ' + config.MODEL_FAST + ', 브리프 ' + config.MODEL_SMART + ')' if llm_available() else '미연결 → 규칙 기반 분류 + 템플릿 브리프'}"
)


# ------------------------------------------------------------------------ views
def render_trace(state: dict) -> None:
    icons = {"orchestrator": "0", "sentinel": "1", "analyst": "2", "scenario_builder": "3",
             "quant_engine": "4", "verifier": "6", "brief": "7", "evaluate": "8"}
    for s in state["trace"]:
        st.markdown(f"`{icons.get(s['node'], '-')}. {s['node']}` {s['summary']}")


def horizon_chart(state: dict) -> go.Figure:
    hs = list(state["impacts"])
    labels = [f"D+{h}" for h in hs]
    fig = go.Figure()
    for name, key, color, off in (("엔진A 팩터모델", "engine_a", C_A, -0.18), ("엔진B 사건재현", "engine_b", C_B, 0.0)):
        fig.add_trace(go.Scatter(
            x=[i + off for i in range(len(hs))], y=[state["impacts"][h].__dict__[key]["q50"] for h in hs],
            mode="markers", name=name, marker=dict(size=10, color=color),
            hovertemplate="%{y:+.2%}<extra>" + name + "</extra>",
        ))
    blend = [state["impacts"][h].blend for h in hs]
    fig.add_trace(go.Scatter(
        x=[i + 0.18 for i in range(len(hs))], y=[b["q50"] for b in blend], mode="markers",
        name="통합 예측 (하위~상위 25%)", marker=dict(size=12, color=INK2, symbol="diamond"),
        error_y=dict(type="data", symmetric=False, array=[b["q75"] - b["q50"] for b in blend],
                     arrayminus=[b["q50"] - b["q25"] for b in blend], color=INK2, thickness=2),
        hovertemplate="%{y:+.2%}<extra>통합</extra>",
    ))
    real = state.get("realized") or {}
    if real:
        fig.add_trace(go.Scatter(
            x=[i + 0.36 for i, h in enumerate(hs) if h in real], y=[real[h]["realized"] for h in hs if h in real],
            mode="markers", name="실측", marker=dict(size=12, color=C_REAL, symbol="star"),
            hovertemplate="%{y:+.2%}<extra>실측</extra>",
        ))
    fig.add_hline(y=0, line_color="#c3c2b7", line_width=1)
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat="+.1%",
        xaxis=dict(tickmode="array", tickvals=list(range(len(hs))), ticktext=labels),
        legend=dict(orientation="h", y=1.12), plot_bgcolor="#fcfcfb", font=dict(color=INK),
    )
    return fig


def contribution_chart(state: dict) -> go.Figure:
    p = state["impacts"].get(config.PRIMARY_HORIZON) or next(iter(state["impacts"].values()))
    c = p.contribution
    c = c.reindex(c.abs().sort_values(ascending=False).index[:8]).sort_values()
    fig = go.Figure(go.Bar(
        x=c.values, y=[ASSETS[t].name for t in c.index], orientation="h",
        marker_color=[C_LOSS if v < 0 else C_GAIN for v in c.values],
        hovertemplate="%{y}: %{x:+.2%}<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat="+.2%",
                      plot_bgcolor="#fcfcfb", font=dict(color=INK))
    return fig


def render_result(state: dict) -> None:
    if "impacts" not in state:
        st.info("임계값을 넘는 이상 신호가 없어 분석을 진행하지 않았어요.")
        render_trace(state)
        return
    h = config.PRIMARY_HORIZON
    p = state["impacts"][h]
    real = (state.get("realized") or {}).get(h)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"D+{h} 예상 영향 (통합 중앙값)", pct(p.blend["q50"]))
    k2.metric("불리 시나리오 (하위 25%)", pct(p.blend["q25"]))
    k3.metric(f"D+{h} 실측", pct(real["realized"]) if real else "미확정")
    k4.metric("검증 등급", state["verification"]["grade"])

    with st.expander("Agent 실행 경로 (파이프라인 트레이스)", expanded=True):
        render_trace(state)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("기간별 예측 vs 실측")
        st.plotly_chart(horizon_chart(state), use_container_width=True)
        st.caption("D+20은 백테스트상 0% 기준선보다 오차가 커서 참고용으로만 표시함.")
    with c2:
        st.subheader(f"자산별 기여도 (D+{h})")
        st.plotly_chart(contribution_chart(state), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("유사 과거 사건")
        per = p.per_analog.copy()
        if not per.empty:
            per = per[["name", "similarity", "weight", "vol_scale", "A", "B", "blend"]]
            per.columns = ["사건", "유사도", "가중치", "변동성 보정", "엔진A", "엔진B", "통합"]
            st.dataframe(per.style.format({"유사도": "{:.2f}", "가중치": "{:.0%}", "변동성 보정": "{:.2f}x",
                                            "엔진A": "{:+.2%}", "엔진B": "{:+.2%}", "통합": "{:+.2%}"}), hide_index=True)
    with c4:
        st.subheader(f"충격 시나리오 (D+{h}, 유사 사건 실측 분포)")
        t = state["scenario"][h]["table"]
        if not t.empty:
            rows = []
            for k, r in t.iterrows():
                f = FACTORS[k]
                fmt = (lambda v: f"{v:+.1%}") if f.transform == "logret" else (lambda v, u=f.unit: f"{v:+.2f}{u}")
                rows.append({"변수": f.name, "불리(25%)": fmt(r["q25"]), "기본(50%)": fmt(r["q50"]), "유리(75%)": fmt(r["q75"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader(f"Impact Brief ({'LLM 작성·수치 감사 통과' if state['brief_engine'] == 'llm' else '템플릿'})")
    st.markdown(state["brief"])

    with st.expander("검증 상세 / 감지 신호"):
        st.dataframe(pd.DataFrame(state["verification"]["checks"]), hide_index=True)
        det = state.get("detection")
        if det:
            sig = pd.DataFrame(det["signals"])[["name", "date", "value", "z", "triggered"]]
            st.dataframe(sig.style.format({"value": "{:.4f}", "z": "{:+.2f}"}), hide_index=True)
        if p.notes:
            st.write("\n".join(f"- {n}" for n in p.notes))


# ------------------------------------------------------------------------ tabs
tab_replay, tab_ask, tab_bt, tab_data = st.tabs(["Replay (과거 사건 재현)", "Ask (사건 질의)", "Backtest", "데이터·출처"])

with tab_replay:
    events = [e for e in load_events() if md.trading_day_before(e.t0) is not None]
    ev_id = st.selectbox(
        "재현할 사건", [e.id for e in reversed(events)],
        format_func=lambda i: next(f"{e.t0.date()}  {e.name}" for e in events if e.id == i),
    )
    ev = next(e for e in events if e.id == ev_id)
    st.caption(f"{ev.description}" + (f"  \n※ {ev.note}" if ev.note else ""))
    if st.button("Replay 실행", type="primary"):
        st.session_state["replay"] = run(md, weights, mode="replay", event_id=ev_id, aum=aum, currency=currency)
    if st.session_state.get("replay") and st.session_state["replay"].get("event") and st.session_state["replay"]["event"].id == ev_id:
        render_result(st.session_state["replay"])

with tab_ask:
    st.write(f"최신 종가({md.last_date.date()}) 기준으로, 입력한 사건이 지금 일어났다고 가정하고 영향을 추정해요.")
    text = st.text_area("사건 설명 (한국어/영어)", placeholder="예: Iran closes the Strait of Hormuz after US strikes / 대만 해협 봉쇄")
    if st.button("분석", type="primary", disabled=not text.strip()):
        st.session_state["ask"] = run(md, weights, mode="ask", event_text=text, aum=aum, currency=currency)
    if st.session_state.get("ask"):
        cls = st.session_state["ask"]["classification"]
        st.caption(f"분류: {cls['type']} / {cls['region']} / {', '.join(cls['channels'])} ({st.session_state['ask']['classifier']}) - {cls.get('rationale_ko', '')}")
        render_result(st.session_state["ask"])

with tab_bt:
    st.write("사건 라이브러리 전체에 대한 Leave-one-out 백테스트 (사건 직전 데이터만 사용).")
    if st.button("현재 포트폴리오로 백테스트 실행"):
        with st.spinner("백테스트 실행 중..."):
            st.session_state["bt"] = run_backtest(md, weights)
    bt = st.session_state.get("bt")
    if bt is None and preset_key in PRESETS and (config.REPORT_DIR / f"backtest_{preset_key}.csv").exists():
        bt = pd.read_csv(config.REPORT_DIR / f"backtest_{preset_key}.csv")
    if bt is not None and not bt.empty:
        s = summarize(bt)
        show = s[["horizon", "events", "direction_hit", "mae_blend", "mae_zero_baseline", "skill_vs_zero", "iqr_coverage", "rank_corr"]]
        show.columns = ["기간", "사건 수", "방향 적중", "MAE", "MAE(0% 기준선)", "기준선 대비 개선", "구간 포함률", "순위상관"]
        st.dataframe(show.style.format({"방향 적중": "{:.0%}", "MAE": "{:.2%}", "MAE(0% 기준선)": "{:.2%}",
                                         "기준선 대비 개선": "{:+.0%}", "구간 포함률": "{:.0%}", "순위상관": "{:+.2f}"}), hide_index=True)
        hsel = st.radio("기간", sorted(bt["horizon"].unique()), horizontal=True, format_func=lambda h: f"D+{h}")
        g = bt[bt["horizon"] == hsel]
        lim = max(g["pred"].abs().max(), g["realized"].abs().max()) * 1.1
        fig = go.Figure(go.Scatter(
            x=g["pred"], y=g["realized"], mode="markers", text=g["name"],
            marker=dict(size=10, color=C_A, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{text}<br>예측 %{x:+.2%}<br>실측 %{y:+.2%}<extra></extra>",
        ))
        fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line=dict(color="#c3c2b7", dash="dot"))
        fig.update_layout(height=420, xaxis_title="예측", yaxis_title="실측", xaxis_tickformat="+.1%",
                          yaxis_tickformat="+.1%", plot_bgcolor="#fcfcfb", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(g, hide_index=True)

with tab_data:
    st.write("수집 데이터의 출처, 수집 시각, 이용 조건, 전처리 내역 (공모전 실증 요구 대응).")
    man = load_manifest()
    if man:
        st.dataframe(pd.DataFrame(man.values())[["source", "id", "fetched_at", "rows", "license", "preprocessing", "url"]], hide_index=True)
    st.markdown(
        "- 대체자산·일부 ETF는 상장 이전 기간을 문서화된 프록시로 보완함 (`sentinel/data/universe.py`의 backfill)\n"
        "- 이벤트 라이브러리: `sentinel/events/events.json` (t0 = 시장이 처음 반응할 수 있었던 미국 거래일)"
    )
