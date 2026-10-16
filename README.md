# MACRO SENTINEL (가칭)

지정학·매크로 충격을 감지하고, 포트폴리오 영향을 **과거 실측 데이터 기반**으로 추정한 뒤, 실제 결과와 비교해 검증하는 관찰형 AI Agent입니다. 「국부펀드 KIC, AI 에이전트 공모전」 출품용 프로토타입입니다.

- 기획서 원고: [`docs/proposal.md`](docs/proposal.md)
- 파이프라인 그림: [`docs/images/pipeline.png`](docs/images/pipeline.png)
- 백테스트 결과: [`reports/`](reports)

## 구현 범위 (P0)

| 구성 | 위치 | 내용 |
|---|---|---|
| 데이터 수집·캐시 | `sentinel/data/` | FRED, Yahoo 시세, GPR 일간 지수, GDELT. 캐시와 출처 manifest(출처, 수집 시각, 이용 조건, 전처리) |
| 사건 라이브러리 | `sentinel/events/events.json` | 2008~2025년 22개 사건 (유형, 지역, 전이 경로, 시장 반응 첫 거래일 t0) |
| 정량 엔진 | `sentinel/quant/` | 엔진 A 팩터모델(주간 Ridge 회귀), 엔진 B 과거사건 재현, 변동성 국면 보정, VaR/CVaR, 자산별 기여도 |
| Sentinel | `sentinel/monitor.py` | 시장 변수 일간 z-score, GPR 지수 z-score 이상 감지 |
| 도구 레이어 | `sentinel/tools.py` | 모든 도구가 값 + 근거(evidence) + 알려진 한계(unknowns)를 함께 반환 |
| Agent 파이프라인 | `sentinel/agents/pipeline.py` | LangGraph: Orchestrator → Sentinel(코드) → Planner(LLM) → Executor(도구) → Verifier(코드, 실패 시 사유를 붙여 재계획, 최대 2회) → Reviewer(LLM) → Brief(코드) → Evaluate |
| LLM | `sentinel/agents/llm.py` | Planner: 사건 분류 + 유사 사건 검색 조건 결정. Reviewer: 근거·한계를 보고 확인/주의 판정(주의면 등급 하향) + 해석 작성. 사건당 2회 호출(재계획 시 +1). 숫자는 자리표시자로만 쓰고 감사 후 치환. Gemini(현재 기본) 또는 Claude |
| MCP 서버 | `sentinel/mcp_server.py` | 도구를 MCP로 공개해 외부 에이전트·사내 시스템이 호출 가능 (`python -m sentinel.mcp_server`) |
| Replay / 백테스트 | `sentinel/backtest.py` | 사건별 Leave-one-out, 사건 직전 데이터만 사용 |
| UI | `app/streamlit_app.py` | Replay, Ask, Backtest, 데이터·출처 탭 |

## 실행 방법

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python scripts/build_snapshot.py          # 데이터 수집 → data/cache (약 10초)
python scripts/run_backtest.py --offline  # 백테스트 → reports/
streamlit run app/streamlit_app.py        # 대시보드
python -m pytest -q                       # 테스트 (네트워크 불필요)
```

LLM 사용은 선택입니다. `GEMINI_API_KEY`(현재 기본) 또는 `ANTHROPIC_API_KEY`를 설정하면 Planner·Reviewer에 LLM을 쓰고, 키가 없으면 규칙 기반 계획과 템플릿 브리프로 동작합니다. 키는 git에 올리지 않는 `.env`에 둡니다 (`.env.example` 참고).

## 설계 원칙

1. **LLM은 계산하지 않는다.** 모든 수치는 Python 정량 엔진이 만들고, LLM 브리프는 `[[fact_key]]` 자리표시자로만 숫자를 인용합니다. 자리표시자 밖에 숫자가 있으면 감사에서 탈락하고 템플릿으로 대체됩니다.
2. **미래 정보 누수 차단.** Replay·백테스트에서 민감도와 시장 데이터는 사건 직전 종가까지만 쓰고, 유사 사건은 관측 구간(최대 D+20)이 예측 기준일 이전에 끝난 사건만 씁니다.
3. **교차 검증.** 독립된 두 엔진(팩터모델 / 사건 재현)의 차이가 허용치를 넘거나 유사 사건이 부족하면 Verifier가 유사 사건 선정을 바꿔 재분석합니다.

## 백테스트 요약 (국부펀드형, 사건 21건)

| 기간 | 실측 하락 비율 | 방향 적중 | 방향 적중(충격 기준선) | MAE | MAE(0% 기준선) | MAE(충격 기준선) |
|---|---|---|---|---|---|---|
| D+1 | 86% | 86% | 86% | 1.08% | 1.31% | 1.39% |
| D+5 | 67% | 67% | 62% | 1.57% | 1.73% | 1.70% |
| D+20 | 57% | 48% | 57% | 3.72% | 2.23% | 3.71% |

- 충격 기준선 = 과거 라이브러리 사건 전체의 실측 중앙값(유사도·변동성 보정 없음).
- **방향 적중률은 기준선과 거의 같습니다.** 충격 사건은 대체로 하락하기 때문에 높은 적중률 자체는 판별력의 증거가 아닙니다.
- 오차(MAE)는 D+1에서 두 기준선보다 작고, D+5는 근소하게 작으며, **D+20은 0% 기준선보다 나쁩니다.**
- 예측 구간(하위~상위 25%)의 실측 포함률은 19~38%로 명목 50%보다 낮아 구간이 좁습니다.
- 상세는 `reports/backtest_*.md`.

## 한계

- 사건 수(21건)가 적고 일부 기간이 겹쳐 통계적 검정력이 낮습니다. 사건 라이브러리는 사후적으로 선별해 선택 편향이 있습니다.
- 대체자산은 상장 프록시로, 연기금형의 국내채권은 원/달러 환율 효과만 반영하는 프록시로 평가합니다.
- Yahoo 시세는 비공식 엔드포인트입니다. 원시 데이터는 저장소에 포함하지 않고 각자 내려받습니다.
- FRED 이용약관(AI/ML 개발·학습 관련 이용 금지)을 고려해 FRED 원시 시계열은 LLM에 입력하지 않고, 계산된 시나리오 통계만 브리프 작성에 전달합니다. 이 해석이 약관에 부합하는지는 추가 확인이 필요합니다.

## 데이터 출처

FRED(St. Louis Fed), Yahoo Finance, Caldara & Iacoviello Geopolitical Risk Index (https://www.matteoiacoviello.com/gpr.htm), The GDELT Project. 수집 시각과 이용 조건은 앱의 "데이터·출처" 탭과 `data/cache/manifest.json`에 기록됩니다.
