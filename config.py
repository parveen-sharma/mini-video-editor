"""
config.py — Configuration loader for mini-video-editor.

Reads editor.config.json from the project root, validates all values,
and exposes named constants used throughout process.py.

Separation of concerns:
  config.py   → what the settings are (loading, validation, constants)
  process.py  → what the pipeline does (ffmpeg, OpenCV, Whisper)

Adding a new setting:
  1. Add the key to editor.config.json
  2. Load it here with a sensible default
  3. Reference the constant in process.py — never access CFG directly there
"""

import json
from pathlib import Path

# =========================
# LOAD
# =========================

ROOT        = Path(".")
CONFIG_PATH = ROOT / "editor.config.json"

if not CONFIG_PATH.exists():
    raise RuntimeError(
        f"editor.config.json not found in: {ROOT.resolve()}\n"
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
            raise KeyError(f"editor.config.json is missing required key: {k!r}")

def _clamp(val: float, lo: float, hi: float, name: str) -> float:
    if not (lo <= val <= hi):
        raise ValueError(f"Config '{name}' must be between {lo} and {hi}, got {val}")
    return val

# =========================
# FONT & SUBTITLE
# =========================

# =========================
# FONT & SUBTITLE
# =========================
_require(CFG, "highlight", "max_words_per_line", "max_lines",
         "margin_horizontal_px", "margin_vertical_px",
         "extend_last_word_ms", "pause_threshold_ms",
         "subtitle_preset")

MAX_WORDS_PER_LINE = int(CFG["max_words_per_line"])
MAX_LINES          = int(CFG["max_lines"])
MARGIN_H           = int(CFG["margin_horizontal_px"])
MARGIN_V           = int(CFG["margin_vertical_px"])

ACTIVE_COLOR       = str(CFG["highlight"]["text_color"])

BG_COLOR           = str(CFG["highlight"]["background_color"])
BG_OPACITY         = _clamp(float(CFG["highlight"]["background_opacity"]), 0.0, 1.0,
                             "highlight.background_opacity")

PAD_X              = int(CFG["highlight"]["padding_x"])
PAD_Y              = int(CFG["highlight"]["padding_y"])

HL_ENABLED         = bool(CFG["highlight"].get("enabled", True))
EXTEND_LAST_WORD_SEC = float(CFG["extend_last_word_ms"]) / 1000.0
PAUSE_THRESHOLD_SEC  = float(CFG.get("pause_threshold_ms", 400)) / 1000.0

# --- DYNAMIC ALIGNMENT FIX ---
_raw_align = str(CFG.get("alignment", "bottom_center")).lower()
align_lookup = {
    "bottom_center": 2,
    "center": 5,
    "top_center": 8
}
ALIGNMENT = align_lookup.get(_raw_align, 2) # Default to 2 if something goes wrong

# ALIGNMENT          = 8      # bottom-center (ASS standard)
BG_BORD            = PAD_Y + 6

# =========================
# SUBTITLE PRESET
# =========================

SUBTITLE_PRESET = str(CFG.get("subtitle_preset", "classic")).lower()

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
PC_W    = _clamp(float(_pc.get("horizontal_keep_pct", 0.88)), 0.1, 1.0,
                 "precrop.horizontal_keep_pct")
PC_H    = _clamp(float(_pc.get("vertical_keep_pct",   0.84)), 0.1, 1.0,
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
ZOOM_OUT_DURATION     = float(CFG.get("zoom_out_duration_sec", 1.5))

# =========================
# PROCESSING
# =========================
# All values that were previously hardcoded inside process.py functions.
# Grouped by the step they control.

_proc = CFG.get("processing", {})

# ── Face Detection ───────────────────────────────────────────────────────────
_fd = _proc.get("face_detection", {})
# Set "enabled": false in editor.config.json to ignore face detection
FACE_DETECTION_ENABLED = bool(_fd.get("enabled", True))

# Add this to the Processing / Face Detection section
# 0.1 = very slow/smooth, 1.0 = instant/jerky
CAMERA_SMOOTHING = 0.15

# ── Scene detection ───────────────────────────────────────────────────────────
# How many times per second to sample frames when diffing for cuts.
# Replaces the hardcoded idx%6 — now derived from actual source fps at runtime
# so it behaves consistently regardless of whether source is 30fps or 120fps.
_sd = _proc.get("scene_detection", {})
SCENE_SAMPLE_FPS        = float(_sd.get("sample_fps",            5.0))
SCENE_DIFF_THRESHOLD    = float(_sd.get("diff_threshold",        22.0))
SCENE_MIN_DURATION_SEC  = float(_sd.get("min_duration_sec",       1.0))

# ── Motion / saliency analysis ────────────────────────────────────────────────
_ma = _proc.get("motion_analysis", {})
MOTION_THRESHOLD        = float(_ma.get("motion_threshold",       8.0))
ANALYSIS_SAMPLE_FPS     = float(_ma.get("sample_fps",             2.0))
MOTION_PIXEL_THRESHOLD  = int(  _ma.get("pixel_diff_threshold",  15  ))
SALIENCY_CENTER_DEADBAND= float(_ma.get("saliency_center_deadband", 0.04))

# ── Background blur ───────────────────────────────────────────────────────────
_bl = _proc.get("background_blur", {})
BLUR_LUMA_RADIUS  = int(_bl.get("ffmpeg_luma_radius", 40))
BLUR_LUMA_POWER   = int(_bl.get("ffmpeg_luma_power",   3))
BLUR_OPENCV_KSIZE = int(_bl.get("opencv_kernel_size", 81))
# Kernel size must be odd for GaussianBlur
if BLUR_OPENCV_KSIZE % 2 == 0:
    BLUR_OPENCV_KSIZE += 1

# ── Zoom-out outro ────────────────────────────────────────────────────────────
_zo = _proc.get("zoom_out", {})
ZOOM_OUT_ENABLED      = bool(_zo.get("enabled",           True))
ZOOM_OUT_MAX_PCT      = float(_zo.get("max_pct_of_scene", 0.30))

# ── Encode quality ────────────────────────────────────────────────────────────
_enc = _proc.get("encode", {})
ENCODE_CRF_PRECROP  = int( _enc.get("crf_precrop",    16))
ENCODE_CRF_SCENES   = int( _enc.get("crf_scenes",     18))
ENCODE_AUDIO_BR     = str( _enc.get("audio_bitrate", "192k"))
ENCODE_PRESET       = str( _enc.get("preset",        "fast"))

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
