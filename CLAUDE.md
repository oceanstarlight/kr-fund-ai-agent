# 작업 원칙 (최우선)

1. **애매하거나 모르는 부분은 반드시 질문한다.** 작업 중 판단이 애매하거나 모르는 부분이 생기면 추측으로 진행하지 말고 사용자에게 먼저 질문한다.
2. **최소 비용으로 최고 성능을 내도록 설계한다.**
   - 결정적 코드(스크립트, 통계, 규칙)로 해결되는 일은 코드로 처리하고, LLM 호출은 꼭 필요한 곳에만 쓴다.
   - 무료·오픈소스 도구와 무료 API 등급을 우선 검토한다.
   - 설계안을 낼 때는 비용과 성능의 트레이드오프를 함께 밝힌다.

# 프로젝트 메모 (MACRO SENTINEL)

- 테스트: `python -m pytest -q` (네트워크·API 키 불필요)
- 데이터 수집: `python scripts/build_snapshot.py`, 백테스트: `python scripts/run_backtest.py --offline`
- 앱: `streamlit run app/streamlit_app.py`
- LLM은 Planner(분류·검색 조건)와 Reviewer(검토·해석)에만 쓴다. 모든 수치는 `sentinel/quant`·`sentinel/tools.py`의 Python 코드가 계산한다.
- 도구는 값과 함께 근거(evidence)와 한계(unknowns)를 반환한다. unknowns 메시지에는 숫자를 쓰지 않는다(LLM 입력용).
- API 키는 git에 올리지 않는 `.env`에 둔다.
- 코드 주석에 이모티콘을 쓰지 않는다.
