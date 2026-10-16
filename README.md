# MACRO SENTINEL (가칭)

지정학·매크로 충격을 감지하고, 포트폴리오 영향을 **과거 실측 데이터 기반**으로 추정한 뒤, 실제 결과와 비교해 검증하는 관찰형 AI Agent입니다. 「국부펀드 KIC, AI 에이전트 공모전」 출품용 프로토타입입니다.

- 계획·미정 사항: [`PLANS.md`](PLANS.md)
- 기획서 원고: [`docs/proposal.md`](docs/proposal.md)
- 파이프라인 그림: [`docs/images/pipeline.png`](docs/images/pipeline.png)
- 백테스트 결과: [`reports/`](reports)

## 구현 범위 (P0)

| 구성 | 위치 | 내용 |
|---|---|---|
| 데이터 수집·캐시 | `sentinel/data/` | 팩터: EIA(브렌트유), 연준 H.10(달러지수·원달러)·H.15(10년물), Cboe(VIX), ETF 파생(신용). 시세: Yahoo. GPR 일간 지수, GDELT. 소스별 병렬 수집, 실패 시 저장소 스냅샷·오래된 캐시로 대체하고 실패 항목을 보고(앱은 계속 동작). 출처 manifest 기록 |
| 사건 라이브러리 | `sentinel/events/events.json` | 2000~2026년 40개 사건 (유형, 지역, 전이 경로, 시장 반응 첫 거래일 t0, 선정 경로). 초기 선별 22건 + GPR 급등 규칙 12건(`scripts/gpr_candidates.py`) + 공식 발표 6건 |
| 정량 엔진 | `sentinel/quant/` | 엔진 A 팩터모델(주간 Ridge 회귀), 엔진 B 과거사건 재현, 변동성 국면 보정, GARCH 기반 예측 구간(`arch`), VaR/CVaR, 자산별 기여도 |
| Sentinel | `sentinel/monitor.py` | 시장 변수 일간 z-score, GPR 지수 z-score 이상 감지 |
| GDELT 감시 | `sentinel/gdelt_monitor.py`, `scripts/monitor.py` | GDELT 15분 원본 파일로 국가별 분쟁 보도 비중의 급증 감지(2구간 연속, robust z ≥ 4). 신호가 뜰 때만 에이전트 실행 |
| 상시 구동 | `.github/workflows/monitor.yml` | GitHub Actions 예약 실행, 상태는 Actions 캐시에 저장, 경보는 GitHub Issue로 알림 |
| 도구 레이어 | `sentinel/tools.py` | 모든 도구가 값 + 근거(evidence) + 알려진 한계(unknowns)를 함께 반환 |
| Agent 파이프라인 | `sentinel/agents/pipeline.py` | LangGraph: Orchestrator → Sentinel(코드) → Planner(LLM) → Executor(도구) → Verifier(코드, 실패 시 사유를 붙여 재계획, 최대 2회) → Reviewer(LLM) → Brief(코드) → Evaluate |
| LLM | `sentinel/agents/llm.py` | Planner: 사건 분류 + 유사 사건 검색 조건 결정. Reviewer: 근거·한계를 보고 확인/주의 판정(주의면 등급 하향) + 해석 작성. 사건당 2회 호출(재계획 시 +1). 숫자는 자리표시자로만 쓰고 감사 후 치환. 같은 요청은 디스크 캐시(`diskcache`)로 재사용. Gemini(현재 기본) 또는 Claude |
| MCP 서버 | `sentinel/mcp_server.py` | 도구를 MCP로 공개해 외부 에이전트·사내 시스템이 호출 가능 (`python -m sentinel.mcp_server`) |
| Replay / 백테스트 | `sentinel/backtest.py` | 사건별 Leave-one-out, 사건 직전 데이터만 사용. 감지 시점(첫날 종가) 백테스트 포함 |
| 감지 시점 갱신 예측 | `sentinel/quant/reaction.py` | 첫날 종가에서 첫날 팩터 반응이 닮은 과거 사건에 가중해 남은 기간 재추정 (Replay) |
| UI | `app/streamlit_app.py` | Replay, Ask, Backtest, 데이터·출처 탭 |

## 실행 방법

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python scripts/build_snapshot.py          # 데이터 수집 → data/cache (약 30초, 소스별 결과 출력)
python scripts/run_backtest.py --offline  # 백테스트 → reports/
streamlit run app/streamlit_app.py        # 대시보드
python -m pytest -q                       # 테스트 (네트워크 불필요)
python scripts/monitor.py                 # 감시 1회 실행 (GitHub Actions가 주기적으로 실행)
```

LLM 사용은 선택입니다. `GEMINI_API_KEY`(현재 기본) 또는 `ANTHROPIC_API_KEY`를 설정하면 Planner·Reviewer에 LLM을 쓰고, 키가 없으면 규칙 기반 계획과 템플릿 브리프로 동작합니다. 키는 git에 올리지 않는 `.env`에 둡니다 (`.env.example` 참고).

## 설계 원칙

1. **LLM은 계산하지 않는다.** 모든 수치는 Python 정량 엔진이 만들고, LLM 브리프는 `[[fact_key]]` 자리표시자로만 숫자를 인용합니다. 자리표시자 밖에 숫자가 있으면 감사에서 탈락하고 템플릿으로 대체됩니다.
2. **미래 정보 누수 차단.** Replay·백테스트에서 민감도와 시장 데이터는 사건 직전 종가까지만 쓰고, 유사 사건은 관측 구간(최대 D+20)이 예측 기준일 이전에 끝난 사건만 씁니다.
3. **교차 검증.** 독립된 두 엔진(팩터모델 / 사건 재현)의 차이가 허용치를 넘거나 유사 사건이 부족하면 Verifier가 유사 사건 선정을 바꿔 재분석합니다.

## 백테스트 요약 (국부펀드형, 사건 39건)

사건 40건 중 가장 이른 1건은 그 이전 유사 사건이 없어 제외됩니다.

| 기간 | 실측 하락 비율 | 방향 적중 | 방향 적중(충격 기준선) | MAE | MAE(0% 기준선) | MAE(충격 기준선) | 구간 포함률(명목 50%) |
|---|---|---|---|---|---|---|---|
| D+1 | 77% | 72% | 77% | 0.88% | 1.10% | 0.96% | 49% |
| D+5 | 64% | 64% | 64% | 1.67% | 1.70% | 1.70% | 56% |
| D+20 | 56% | 36% | 46% | 3.47% | 2.92% | 3.11% | 51% |

- 충격 기준선 = 과거 라이브러리 사건 전체의 실측 중앙값(유사도·변동성 보정 없음).
- **D+1은 0% 기준선보다 오차가 약 20% 작고, 부트스트랩 90% 신뢰구간(+5% ~ +32%)이 0을 넘습니다.** 충격 기준선 대비 개선(+9%)은 신뢰구간이 0을 걸칩니다.
- D+5는 두 기준선과 비슷하고, **D+20은 두 기준선보다 나쁩니다(참고용).**
- **방향 적중률은 기준선보다 높지 않습니다.** 충격 사건은 대체로 하락하기 때문에 적중률 자체는 판별력의 증거가 아닙니다.
- 사건을 21건에서 40건으로 늘리자 기존 21건의 오차도 줄었습니다(유사 사건 후보 증가): D+1 1.10% → 0.98%, D+5 1.59% → 1.52%, D+20 3.86% → 2.71%.
- 새로 추가한 18건(규칙 기반 선정)만 보면 D+1 우위가 거의 없습니다. 시장 반응이 작은 사건(외교 합의, 정책 발표 등)이 포함되어 0% 기준선이 강하기 때문입니다.
- 리포트에는 개선율의 부트스트랩 90% 신뢰구간과 선정 경로별(초기 선별 / 규칙 기반) 결과가 함께 나옵니다.
- 상세는 `reports/backtest_*.md`.

### 감지 시점 백테스트 (첫날 반응을 본 뒤 남은 움직임 평가)

위 결과는 사건 직전 종가 기준이라, 실제 감지 시점(첫날 종가)에는 이미 아는 첫날 움직임이 점수에 섞입니다. 그래서 첫날 종가에서 예측해 **남은 수익률(첫날 종가 → D+5, D+20)**로 다시 평가했습니다(`reports/detection_*.md`, 사건 37건).

| 기간 | 방식 | MAE | MAE(0% 기준선) | 0% 대비 (90% CI) | 기존 방식 대비 (90% CI) |
|---|---|---|---|---|---|
| D+5 | 기존 방식(유사도 가중) | 1.57% | 1.48% | -6% (-24% ~ +8%) | - |
| D+5 | 첫날 반응 가중 | 1.64% | 1.48% | -10% (-31% ~ +8%) | -4% (-22% ~ +11%) |
| D+20 | 기존 방식(유사도 가중) | 3.88% | 3.05% | -27% (-44% ~ -14%) | - |
| D+20 | 첫날 반응 가중 | 3.38% | 3.05% | -11% (-25% ~ +2%) | **+13% (+4% ~ +23%)** |

- **첫날 반응을 본 뒤의 남은 움직임은 어떤 방식도 0%(첫날 수준 유지)보다 잘 맞히지 못했습니다.** 세 프리셋, 감지된 사건만 본 경우 모두 같습니다.
- 첫날 팩터 반응이 닮은 과거 사건에 가중하면 D+20 오차가 기존 방식보다 유의하게 줄었습니다(세 프리셋 모두 신뢰구간이 0 이상).
- 그래서 Replay의 **"감지 시점 갱신 예측"**은 첫날 반응 가중을 씁니다. 사건 직전 예측은 그대로 두고, 0% 기준선보다 낫다는 근거는 없다는 점을 함께 표시합니다. 코드만으로 계산하며 LLM 비용은 0입니다.

## 한계

- 사건 수(40건)가 여전히 적어 신뢰구간이 넓습니다. 초기 22건은 사후적으로 선별했고, 이후 18건은 시장 반응을 보지 않는 규칙(GPR 급등, 공식 발표)으로 추가했습니다. GPR 급등일을 사건에 연결하는 과정에는 판단이 들어가며, 신뢰도가 낮은 경우 사건 note에 표시했습니다.
- 2006년 이전 사건은 달러지수 데이터가 없고, 일부 ETF는 상장 전이라 프록시로 보완합니다.
- 대체자산은 상장 프록시로, 연기금형의 국내채권은 원/달러 환율 효과만 반영하는 프록시로 평가합니다.
- Yahoo 시세는 비공식 엔드포인트입니다. 시세는 저장소에 포함하지 않고 실행 시 수집합니다. 저장소에는 재배포 가능한 공공 데이터(EIA, 연준) 팩터 스냅샷(`data/snapshot`)만 포함해 수집 실패 시 대체합니다.

## 데이터 출처

U.S. EIA, Federal Reserve Board(H.10, H.15), Cboe Global Markets, Yahoo Finance, Caldara & Iacoviello Geopolitical Risk Index (https://www.matteoiacoviello.com/gpr.htm), The GDELT Project. 수집 시각과 이용 조건은 앱의 "데이터·출처" 탭과 `data/cache/manifest.json`에 기록됩니다.
