"""Global configuration: paths, model names, thresholds."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local, git-ignored .env file (existing env vars win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(os.environ.get("SENTINEL_CACHE_DIR", DATA_DIR / "cache"))
REPORT_DIR = ROOT / "reports"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# LLM routing. Provider "auto" picks Gemini when a Google key is set, then Anthropic, else none.
# LLMs only classify free-text events and write the brief; they are called only after the
# deterministic sentinel triggers (or a user asks), so per-event cost stays near zero.
LLM_PROVIDER = os.environ.get("SENTINEL_LLM_PROVIDER", "auto")  # auto | gemini | anthropic | none
GEMINI_MODEL_FAST = os.environ.get("SENTINEL_GEMINI_MODEL_FAST", "gemini-3.5-flash-lite")
GEMINI_MODEL_SMART = os.environ.get("SENTINEL_GEMINI_MODEL_SMART", "gemini-3.5-flash-lite")
ANTHROPIC_MODEL_FAST = os.environ.get("SENTINEL_MODEL_FAST", "claude-haiku-5-5")
ANTHROPIC_MODEL_SMART = os.environ.get("SENTINEL_MODEL_SMART", "claude-sonnet-5-5")

# Disk cache of LLM responses keyed by (provider, model, schema, prompt). Identical requests
# are never paid for twice. Set SENTINEL_LLM_CACHE=0 to disable.
LLM_CACHE = os.environ.get("SENTINEL_LLM_CACHE", "1") != "0"
LLM_CACHE_DIR = Path(os.environ.get("SENTINEL_LLM_CACHE_DIR", DATA_DIR / "llm_cache"))

# Forecast horizons in trading days (D+1 = event day close vs. prior close).
HORIZONS = (1, 5, 20)
PRIMARY_HORIZON = 5

# Factor sensitivity estimation.
BETA_WINDOW_WEEKS = 156
BETA_MIN_WEEKS = 52
# Ridge penalty on the standardized (correlation) scale: a number, or "cv" to pick it per
# asset by leave-one-out error over RIDGE_ALPHAS (efficient closed form, like sklearn RidgeCV).
# Backtest (21 events, 3 presets): CV mostly picks weaker penalties (0.01-0.03) and leaves MAE
# unchanged within noise (D+1 slightly worse, D+5 mixed), so the fixed 0.1 stays the default.
_ridge_env = os.environ.get("SENTINEL_RIDGE_ALPHA", "0.1")
RIDGE_ALPHA: float | str = _ridge_env if _ridge_env == "cv" else float(_ridge_env)
RIDGE_ALPHAS = tuple(float(a) for a in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0))

# Volatility regime scaling and prediction intervals: "garch" (arch GARCH(1,1) forecast)
# or the simple versions ("realized" 60-day vol, "analog" spread across analogs).
# Backtest (21 events, 3 presets): GARCH intervals moved IQR coverage from 14-38% to 48-71%
# (nominal 50%), so they are the default. GARCH vol scaling gave a negligible D+1/D+5 gain and a
# worse D+20, so the realized-vol ratio stays the default.
VOL_METHOD = os.environ.get("SENTINEL_VOL_METHOD", "realized")
INTERVAL_METHOD = os.environ.get("SENTINEL_INTERVAL_METHOD", "garch")

# Analog selection.
ANALOG_K = 5
ANALOG_MIN = 3
ANALOG_MIN_SIMILARITY = 0.25
MAX_VERIFY_RETRIES = 2

# Verifier: allowed gap between engine A and engine B (portfolio, primary horizon).
ENGINE_GAP_ABS = 0.0075
ENGINE_GAP_REL = 0.5

# Sentinel thresholds.
SENTINEL_Z_MARKET = 3.0
SENTINEL_Z_GPR = 2.5
SENTINEL_LOOKBACK_DAYS = 252

# HTTP.
HTTP_TIMEOUT = 30
# Yahoo answers 429 to many full browser UA strings but accepts the bare token (checked 2026-10).
BROWSER_USER_AGENT = "Mozilla/5.0"
