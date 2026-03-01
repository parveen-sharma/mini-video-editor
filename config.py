"""
config.py — Configuration loader for mini-video-editor.

Reads subtitle.config.json from the project root, validates all values,
and exposes named constants used throughout process.py.

Separation of concerns:
  config.py   → what the settings are (loading, validation, constants)
  process.py  → what the pipeline does (ffmpeg, OpenCV, Whisper)

Adding a new setting:
  1. Add the key to subtitle.config.json
  2. Load it here with a sensible default
  3. Reference the constant in process.py — never access CFG directly there
"""

import json
from pathlib import Path

# =========================
# LOAD
# =========================

ROOT        = Path(".")
CONFIG_PATH = ROOT / "subtitle.config.json"

if not CONFIG_PATH.exists():
    raise RuntimeError(
        f"subtitle.config.json not found in: {ROOT.resolve()}\n"
        f"Copy the sample config to this directory and re-run."
    )

with open(CONFIG_PATH, "r", encoding="utf8") as _f:
    CFG = json.load(_f)

# =========================
# VALIDATE HELPERS
# =========================

def _require(d: dict, *keys):
    """Raise a clear error if any key is missing from d."""
    for k in keys:
        if k not in d:
            raise KeyError(f"subtitle.config.json is missing required key: {k!r}")

def _clamp(val: float, lo: float, hi: float, name: str) -> float:
    if not (lo <= val <= hi):
        raise ValueError(f"Config '{name}' must be between {lo} and {hi}, got {val}")
    return val

# =========================
# FONT & SUBTITLE
# =========================

_require(CFG, "font", "highlight", "max_words_per_line", "max_lines",
         "margin_horizontal_px", "margin_vertical_px",
         "extend_last_word_ms", "pause_threshold_ms")

FONT               = str(CFG["font"]["family"])
FONT_SIZE          = int(CFG["font"]["size"])

MAX_WORDS_PER_LINE = int(CFG["max_words_per_line"])
MAX_LINES          = int(CFG["max_lines"])

MARGIN_H           = int(CFG["margin_horizontal_px"])
MARGIN_V           = int(CFG["margin_vertical_px"])

ACTIVE_COLOR       = str(CFG["highlight"]["text_color"])
INACTIVE_COLOR     = str(CFG["font"]["inactive_color"])

BG_COLOR           = str(CFG["highlight"]["background_color"])
BG_OPACITY         = _clamp(float(CFG["highlight"]["background_opacity"]), 0.0, 1.0,
                             "highlight.background_opacity")

PAD_X              = int(CFG["highlight"]["padding_x"])
PAD_Y              = int(CFG["highlight"]["padding_y"])

HL_ENABLED         = bool(CFG["highlight"].get("enabled", True))
EXTEND_LAST_WORD_SEC = float(CFG["extend_last_word_ms"]) / 1000.0
PAUSE_THRESHOLD_SEC  = float(CFG.get("pause_threshold_ms", 400)) / 1000.0

ALIGNMENT          = 8      # bottom-center (ASS standard)
BG_BORD            = PAD_Y + 6

# =========================
# OUTPUT DIMENSIONS
# =========================

# Portrait output — must match ASS PlayRes values exactly.
OUT_W, OUT_H = 1080, 1920

# =========================
# PRE-CROP
# =========================

# Separate anchor keys (preferred):
#   horizontal_anchor → left | center | right
#   vertical_anchor   → top  | middle | bottom
#
# Backward compat: if the new keys are absent, the old combined
#   "anchor": "vertical-horizontal"  string is parsed as a fallback.

_pc     = CFG.get("precrop", {})
PC_W    = _clamp(float(_pc.get("horizontal_keep_pct", 0.95)), 0.1, 1.0,
                 "precrop.horizontal_keep_pct")
PC_H    = _clamp(float(_pc.get("vertical_keep_pct",   0.95)), 0.1, 1.0,
                 "precrop.vertical_keep_pct")

_h_anc  = str(_pc.get("horizontal_anchor", "")).lower().strip()
_v_anc  = str(_pc.get("vertical_anchor",   "")).lower().strip()
if not _h_anc or not _v_anc:
    _old = str(_pc.get("anchor", "middle-center")).lower().strip().split("-")
    _v_anc = _v_anc or (_old[0] if len(_old) > 0 else "middle")
    _h_anc = _h_anc or (_old[1] if len(_old) > 1 else "center")

_VALID_H = {"left", "center", "right"}
_VALID_V = {"top", "middle", "bottom"}
if _h_anc not in _VALID_H:
    raise ValueError(f"Invalid precrop horizontal_anchor '{_h_anc}'. Must be: {_VALID_H}")
if _v_anc not in _VALID_V:
    raise ValueError(f"Invalid precrop vertical_anchor '{_v_anc}'. Must be: {_VALID_V}")

PC_H_ANCHOR = _h_anc   # left | center | right
PC_V_ANCHOR = _v_anc   # top  | middle | bottom

# =========================
# ZOOM-OUT OUTRO
# =========================

# Duration of the zoom-out outro appended to every scene.
# Set to 0.0 to disable. Capped at 30% of scene duration for short clips.
ZOOM_OUT_DURATION = float(CFG.get("zoom_out_duration_sec", 1.5))

# =========================
# SCENE SNAPPING
# =========================

# Maximum seconds a scene boundary can move to align with a word end.
SNAP_TOLERANCE_SEC = float(CFG.get("snap_tolerance_sec", 1.0))

# =========================
# LOGO
# =========================

# Logo settings are read as a dict by _build_logo_filter() since the
# whole block is optional. Expose it as a named constant so process.py
# never imports CFG directly.
LOGO_CFG = CFG.get("logo", {})

