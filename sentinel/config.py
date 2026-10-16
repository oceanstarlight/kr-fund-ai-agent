"""Global configuration: paths, model names, thresholds."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(os.environ.get("SENTINEL_CACHE_DIR", DATA_DIR / "cache"))
REPORT_DIR = ROOT / "reports"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# LLM routing. Provider "auto" picks Gemini when a Google key is set, then Anthropic, else none.
# LLMs only classify free-text events and write the brief; they are called only after the
# deterministic sentinel triggers (or a user asks), so per-event cost stays near zero.
LLM_PROVIDER = os.environ.get("SENTINEL_LLM_PROVIDER", "auto")  # auto | gemini | anthropic | none
GEMINI_MODEL_FAST = os.environ.get("SENTINEL_GEMINI_MODEL_FAST", "gemini-flash-lite-latest")
GEMINI_MODEL_SMART = os.environ.get("SENTINEL_GEMINI_MODEL_SMART", "gemini-flash-latest")
ANTHROPIC_MODEL_FAST = os.environ.get("SENTINEL_MODEL_FAST", "claude-haiku-5-5")
ANTHROPIC_MODEL_SMART = os.environ.get("SENTINEL_MODEL_SMART", "claude-sonnet-5-5")

# Forecast horizons in trading days (D+1 = event day close vs. prior close).
HORIZONS = (1, 5, 20)
PRIMARY_HORIZON = 5

# Factor sensitivity estimation.
BETA_WINDOW_WEEKS = 156
BETA_MIN_WEEKS = 52
RIDGE_ALPHA = 0.1  # penalty on the standardized (correlation) scale

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
BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
