import argparse, json, subprocess, gc, re, hashlib, random
from pathlib import Path
from tqdm import tqdm
import whisper, cv2, numpy as np

import sys
import io

# Force UTF-8 for Windows console to handle emojis correctly
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import (
    CFG, LOGO_CFG,
    MAX_WORDS_PER_LINE, MAX_LINES,
    MARGIN_H, MARGIN_V,
    ACTIVE_COLOR,
    BG_COLOR, BG_OPACITY, PAD_X, PAD_Y,
    HL_ENABLED, EXTEND_LAST_WORD_SEC, PAUSE_THRESHOLD_SEC,
    ALIGNMENT, BG_BORD,
    OUT_W, OUT_H,
    PC_W, PC_H, PC_H_ANCHOR, PC_V_ANCHOR,
    ZOOM_OUT_DURATION, SNAP_TOLERANCE_SEC,
    SCENE_SAMPLE_FPS, SCENE_DIFF_THRESHOLD, SCENE_MIN_DURATION_SEC,
    MOTION_THRESHOLD, ANALYSIS_SAMPLE_FPS,
    MOTION_PIXEL_THRESHOLD, SALIENCY_CENTER_DEADBAND,
    BLUR_LUMA_RADIUS, BLUR_LUMA_POWER, BLUR_OPENCV_KSIZE,
    ZOOM_OUT_ENABLED, ZOOM_OUT_MAX_PCT,
    ENCODE_CRF_PRECROP, ENCODE_CRF_SCENES, ENCODE_AUDIO_BR, ENCODE_PRESET,
    FACE_DETECTION_ENABLED,
    SUBTITLE_PRESET,
)

# =========================
# LANGUAGE → FONT MAP
# =========================

LANG_FONT_MAP = {
    "hi": "Noto Sans Devanagari",
    "en": "Inter SemiBold",
    "ar": "Noto Sans Arabic",
    "zh": "Noto Sans SC",
    "ja": "Noto Sans JP",
    "ko": "Noto Sans KR",
}

DEFAULT_FONT = "Noto Sans"

# =========================
# MODULE-LEVEL FACE CASCADE
# Loaded once at import time — NOT per-frame or per-scene.
# Previously re-loaded inside a tight loop, costing ~120 disk reads per scene.
# =========================

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE  = cv2.CascadeClassifier(_CASCADE_PATH)

# =========================
# MULTI-LINGUAL HELPERS
# =========================

def tokenize_text(text):
    return text.split()


def words_to_text(scene_words):
    return " ".join(w["w"] for w in scene_words)


def summarize_text_simple(text):
    sentences = re.split(r'[.!?।]', text)
    for s in sentences:
        s = s.strip()
        if len(s) > 10:
            return s
    return text[:80]


def extract_keywords_simple(text, top_k=5):
    words = tokenize_text(text)
    freq = {}
    for w in words:
        w = w.strip()
        if len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_k]


HOOK_TEMPLATES = [
    "Nobody talks about {}",
    "This changes how you see {}",
    "The truth about {}",
    "Why {} matters more than you think",
    "Stop doing this if you care about {}",
]


def generate_hook_simple(summary):
    keywords = extract_keywords_simple(summary)
    topic = " ".join(keywords[:2]) if keywords else summary[:30]
    # FIX (low): use random.choice instead of numpy for a simple list pick
    return random.choice(HOOK_TEMPLATES).format(topic)


def clean_filename_unicode(text):
    text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    text = text.strip().replace(' ', '_')
    short = text[:40]
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:6]
    return f"{short}_{h}"


# =========================
# CONTENT-AWARE SPLITTING
# =========================

def split_by_speech(words, min_dur=35, max_dur=60, pause_thresh=1.1):
    """
    Create scenes based on speech pauses and sentence boundaries.

    Rules:
      1. Split on long pauses (> pause_thresh)
      2. Split on sentence-ending punctuation
      3. Hard cap at max_dur seconds
    """
    if not words:
        return []

    scenes = []
    current = []
    start_time = words[0]["start"]

    for w in words:
        if not current:
            current.append(w)
            continue

        gap      = w["start"] - current[-1]["end"]
        duration = w["end"] - start_time
        word_text = current[-1].get("word") or current[-1].get("w", "")

        if gap > pause_thresh and duration >= min_dur:
            scenes.append({"start": start_time, "end": current[-1]["end"]})
            current = []
            start_time = w["start"]
        elif word_text.endswith((".", "?", "!", "।")) and duration >= min_dur:
            scenes.append({"start": start_time, "end": current[-1]["end"]})
            current = []
            start_time = w["start"]
        elif duration >= max_dur:
            scenes.append({"start": start_time, "end": current[-1]["end"]})
            current = []
            start_time = w["start"]

        current.append(w)

    if current:
        scenes.append({"start": start_time, "end": current[-1]["end"]})

    for i, s in enumerate(scenes, 1):
        s["id"] = i

    return scenes


# =========================
# HYBRID SPLITTING (VISUAL + SPEECH)
# =========================

def hybrid_split(words, visual_scenes, min_dur=35, max_dur=65, pause_thresh=1.1):
    """Combine visual scene boundaries with speech-based splitting."""
    final_scenes = []

    for vs in visual_scenes:
        vs_start = vs["start"]
        vs_end   = vs["end"]

        scene_words = [
            w for w in words
            if w["end"] > vs_start and w["start"] < vs_end
        ]

        if not scene_words:
            continue

        local_words = [
            {
                "word":  w.get("word", ""),
                "start": w["start"] - vs_start,
                "end":   w["end"]   - vs_start,
            }
            for w in scene_words
        ]

        sub_scenes = split_by_speech(
            local_words,
            min_dur=min_dur,
            max_dur=max_dur,
            pause_thresh=pause_thresh,
        )

        for s in sub_scenes:
            final_scenes.append({
                "start": s["start"] + vs_start,
                "end":   s["end"]   + vs_start,
            })

    for i, s in enumerate(final_scenes, 1):
        s["id"] = i

    return final_scenes


# =========================
# MERGE WEAK SCENES
# =========================

def merge_short_scenes(scenes, min_duration=20):
    merged = []
    buffer = None

    for s in scenes:
        dur = s["end"] - s["start"]

        if dur < min_duration:
            if buffer:
                buffer["end"] = s["end"]
            else:
                buffer = s.copy()
        else:
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(s)

    if buffer:
        merged.append(buffer)

    for i, s in enumerate(merged, 1):
        s["id"] = i

    return merged


# =========================
# FONTS
# =========================

FONTS_DIR = str(Path(__file__).resolve().parent / "fonts").replace("\\", "/").replace(":", "\\:")


# =========================
# UTILS
# =========================

def escape_ffmpeg_path(p) -> str:
    """
    Escape a path for use inside an ffmpeg filter string.
    Converts backslashes to forward slashes, then escapes colons.
    Single source of truth — used by FONTS_DIR, ASS paths, and logo paths.
    """
    return str(Path(p).resolve()).replace("\\", "/").replace(":", "\\:")


def run(cmd):
    """
    Run an ffmpeg (or any) subprocess, capturing stderr.
    FIX (high): previously swallowed all stderr, making failures invisible.
    Now surfaces the full ffmpeg error output on failure.
    """
    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        cmd_str = " ".join(str(c) for c in cmd)
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{cmd_str}\n\n"
            f"--- stderr ---\n{stderr}"
        )


def ts(t):
    """Seconds → ASS timestamp  H:MM:SS.cc"""
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def escape_ass(text):
    return re.sub(r"[{}\\]", "", text)


def ass_color(hex_color, alpha: int = 0) -> str:
    """#RRGGBB → ASS &HAABBGGRR& (alpha 0 = opaque, 255 = transparent)"""
    rgb = hex_color.lstrip("#")
    aa  = f"{alpha:02X}"
    return f"&H{aa}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&"


def opacity_to_alpha(opacity: float) -> int:
    """Config opacity 0-1 → ASS alpha 0-255 (inverted)."""
    return int(255 * (1.0 - max(0.0, min(1.0, opacity))))


# =========================
# SCENE DETECTION
# =========================

def detect_scenes(video):
    """
    Detect scene cuts by diffing sampled grayscale frames.

    Uses SCENE_SAMPLE_FPS to derive a framerate-agnostic sample interval,
    so behaviour is consistent whether source is 24, 30, 60, or 120 fps.

    Returns (fps, scenes) where fps is the actual source framerate.
    """
    cap   = cv2.VideoCapture(video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval = max(1, int(fps / SCENE_SAMPLE_FPS))

    prev  = None
    idx   = 0
    cuts  = [0]

    pbar = tqdm(total=total, desc="🔍 Detecting scenes")

    while True:
        ok, frm = cap.read()
        if not ok:
            break

        if idx % sample_interval == 0:
            g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = np.mean(cv2.absdiff(g, prev))
                if diff > SCENE_DIFF_THRESHOLD:
                    cuts.append(idx)
            prev = g

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    cuts.append(total)

    min_frames = fps * SCENE_MIN_DURATION_SEC
    scenes = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        if b - a > min_frames:
            scenes.append({"id": i, "start": a / fps, "end": b / fps})

    print(f"   sample_interval={sample_interval} frames  "
          f"({fps:.0f}fps ÷ {SCENE_SAMPLE_FPS} samples/s)  "
          f"diff_threshold={SCENE_DIFF_THRESHOLD}")
    return fps, scenes


# =========================
# SCENE ANALYSIS — unified face + motion + saliency in one pass
# FIX (medium): previously opened the video twice per scene (analyze_scene +
# analyze_scene_dynamic). Now a single pass collects gray frames, face path,
# and a representative middle frame, cutting I/O and compute roughly in half.
# =========================

def _motion_center(gray_frames: list) -> tuple:
    """
    Compute the median centroid of changed pixels across consecutive frame pairs.
    Returns (cx_frac, cy_frac, mean_motion_energy).
    """
    cx_samples, cy_samples, energies = [], [], []
    h, w = gray_frames[0].shape

    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        energies.append(float(diff.mean()))

        _, mask = cv2.threshold(diff, MOTION_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.dilate(mask, kernel, iterations=1)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx_samples.append(M["m10"] / M["m00"] / w)
            cy_samples.append(M["m01"] / M["m00"] / h)

    if not cx_samples:
        return 0.5, 0.5, 0.0

    return (
        float(np.median(cx_samples)),
        float(np.median(cy_samples)),
        float(np.mean(energies)),
    )


def _face_center(frame_bgr: np.ndarray) -> tuple:
    """
    Return (cx_frac, cy_frac) of the largest face in frame_bgr, or None.
    Uses the module-level _FACE_CASCADE — no per-call disk load.
    minNeighbors=8 reduces false detections on slides/text.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 8, minSize=(120, 120))

    if len(faces) == 0:
        return None

    largest = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest
    img_h, img_w = frame_bgr.shape[:2]
    return (x + w / 2) / img_w, (y + h / 2) / img_h


def _face_path_from_frames(color_frames: list, timestamps: list, img_h: int, img_w: int) -> list:
    """
    Run face detection on every sampled color frame and return a list of
    {"t", "x", "y"} dicts for faces large enough to be a real person
    (min dimension ≥ 15% of frame height).
    Uses the module-level cascade — no per-frame disk load.
    """
    min_dim = int(img_h * 0.15)
    path = []
    for frame, t in zip(color_frames, timestamps):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 6, minSize=(min_dim, min_dim))
        for (x, y, w, h) in faces:
            path.append({
                "t": t,
                "x": (x + w / 2) / img_w,
                "y": (y + h / 2) / img_h,
            })
    return path


def _saliency_center(frame_bgr: np.ndarray) -> tuple:
    """
    Use OpenCV's SpectralResidual saliency to find the most visually prominent
    region. Useful for static scenes (title cards, diagrams).
    Returns (cx_frac, cy_frac).
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = saliency.computeSaliency(frame_bgr)

    if not ok:
        return 0.5, 0.5

    sal_8 = (sal_map * 255).astype("uint8")
    M     = cv2.moments(sal_8)

    if M["m00"] > 0:
        h, w = frame_bgr.shape[:2]
        return M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h

    return 0.5, 0.5


def analyze_scene(video_path: str, scene_start: float, scene_end: float) -> dict:
    """
    Single-pass analysis: opens the video ONCE and collects everything needed
    for both strategy selection and crop positioning.

    Returns a dict with:
      src_w, src_h     — source frame dimensions
      cx_frac, cy_frac — normalised crop center (0-1)
      strategy         — "face" | "motion" | "saliency" | "center"
      motion_energy    — mean motion signal (for slide detection)
      face_path        — list of {"t", "x", "y"} timed face detections
    """
    cap    = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(fps / ANALYSIS_SAMPLE_FPS))
    start_frame    = int(scene_start * fps)
    end_frame      = int(scene_end   * fps)
    mid_frame      = start_frame + (end_frame - start_frame) // 2

    gray_frames   = []
    color_frames  = []
    timestamps    = []          # relative timestamps for face path
    middle_region = None

    for fi in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break

        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        color_frames.append(frame)
        timestamps.append((fi / fps) - scene_start)

        if middle_region is None or abs(fi - mid_frame) < frame_interval:
            middle_region = frame.copy()

    cap.release()

    # Build face path from ALL sampled frames (replaces analyze_scene_dynamic)
    face_path = _face_path_from_frames(color_frames, timestamps, src_h, src_w)

    # ── Tier 0: Face Detection (center from best/largest face in mid frame) ──
    if FACE_DETECTION_ENABLED and middle_region is not None:
        face_res = _face_center(middle_region)
        if face_res:
            cx, cy = face_res
            return {
                "src_w": src_w, "src_h": src_h,
                "cx_frac": cx, "cy_frac": cy,
                "strategy": "face", "motion_energy": 0.0,
                "face_path": face_path,
            }

    # ── Tier 1: Motion ───────────────────────────────────────────────────────
    energy = 0.0
    if len(gray_frames) >= 2:
        cx, cy, energy = _motion_center(gray_frames)
        if energy >= MOTION_THRESHOLD:
            return {
                "src_w": src_w, "src_h": src_h,
                "cx_frac": cx, "cy_frac": cy,
                "strategy": "motion", "motion_energy": energy,
                "face_path": face_path,
            }

    # ── Tier 2: Saliency ─────────────────────────────────────────────────────
    if middle_region is not None:
        cx, cy = _saliency_center(middle_region)
        if not (0.5 - SALIENCY_CENTER_DEADBAND < cx < 0.5 + SALIENCY_CENTER_DEADBAND):
            return {
                "src_w": src_w, "src_h": src_h,
                "cx_frac": cx, "cy_frac": cy,
                "strategy": "saliency", "motion_energy": energy,
                "face_path": face_path,
            }

    # ── Tier 3: Center fallback ───────────────────────────────────────────────
    return {
        "src_w": src_w, "src_h": src_h,
        "cx_frac": 0.5, "cy_frac": 0.5,
        "strategy": "center", "motion_energy": energy,
        "face_path": face_path,
    }


# =========================
# PORTRAIT FILTER BUILDER
# =========================

def build_portrait_filter(info: dict, scene_duration: float, strategy_type: str):
    """
    Generates dynamic FFmpeg crop expressions.
    - Slides: slow Ken Burns pan across the full slide width.
    - Action/Face: static window anchored to the detected subject center.
    """
    src_w    = info["src_w"]
    src_h    = info["src_h"]
    port_w   = int(src_h * 9 / 16)
    pan_range = max(0, src_w - port_w)

    bg = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        f"boxblur=luma_radius={BLUR_LUMA_RADIUS}:luma_power={BLUR_LUMA_POWER}"
    )

    if strategy_type == "slides":
        x_expr    = f"({pan_range}*t/{scene_duration})"
        crop_x_end = pan_range
        print(f"   🎥 Strategy: SLIDE (Panning 0 -> {pan_range}px)")
    else:
        target_cx  = info["cx_frac"] * src_w
        static_x   = max(0, min(src_w - port_w, int(target_cx - port_w // 2)))
        x_expr     = str(static_x)
        crop_x_end = static_x
        print(f"   👤 Strategy: ACTION/FACE (Anchored x={static_x}px)")

    fg = f"crop={port_w}:{src_h}:{x_expr}:0,scale={OUT_W}:{OUT_H}"

    filt = (
        f"[0:v]split[bg_src][fg_src];"
        f"[bg_src]{bg}[blurbg];"
        f"[fg_src]{fg}[portrait];"
        f"[blurbg][portrait]overlay=0:0[out]"
    )
    return filt, crop_x_end


def _composite_frame(content_crop: np.ndarray, full_frame: np.ndarray) -> np.ndarray:
    """
    Place content_crop (any aspect ratio) onto an OUT_W×OUT_H canvas
    with a blurred version of full_frame as background.
    Aspect ratio of content is always preserved (fit, not fill).
    """
    bg = cv2.resize(full_frame, (OUT_W, OUT_H))
    bg = cv2.GaussianBlur(bg, (BLUR_OPENCV_KSIZE, BLUR_OPENCV_KSIZE), 0)

    h, w  = content_crop.shape[:2]
    scale = min(OUT_W / w, OUT_H / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    fg    = cv2.resize(content_crop, (new_w, new_h))

    x0 = (OUT_W - new_w) // 2
    y0 = (OUT_H - new_h) // 2
    canvas = bg.copy()
    canvas[y0:y0 + new_h, x0:x0 + new_w] = fg

    return canvas


def render_zoomout_opencv(
    video: str,
    scene_start: float,
    scene_end: float,
    zoom_duration: float,
    info: dict,
    crop_x_end: int,
    out_path: Path,
) -> None:
    """
    Render the zoom-out outro entirely in Python using OpenCV.

    ffmpeg's crop filter cannot animate width/height via expressions — only x
    and y are runtime-dynamic. OpenCV gives per-frame control with no
    expression parser.

    Source frame: middle of the scene (most representative/stable content).
    Animation: progress 0.0 = portrait-width crop at crop_x_end,
               progress 1.0 = full source width, x=0.
    """
    src_w  = info["src_w"]
    src_h  = info["src_h"]
    port_w = int(src_h * 9 / 16)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    mid_frame_idx = int((scene_start + (scene_end - scene_start) / 2) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ok, source_frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(scene_start * fps))
        _, source_frame = cap.read()
    cap.release()

    n_frames = max(1, int(zoom_duration * fps))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))

    for i in range(n_frames):
        progress = i / max(n_frames - 1, 1)

        cw = int(port_w + (src_w - port_w) * progress)
        cx = int(crop_x_end * (1 - progress))
        ch = src_h

        cw = max(1, min(cw, src_w - cx))

        content_crop = source_frame[0:ch, cx:cx + cw]
        out_frame    = _composite_frame(content_crop, source_frame)
        writer.write(out_frame)

    writer.release()


# =========================
# STEP 0 — PRE-CROP
# =========================

def _probe_dimensions(src: str) -> tuple:
    """
    Use ffprobe to read pixel dimensions of a video file.
    Returns (width, height) as integers.
    """
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(src),
    ], text=True).strip()

    for line in out.splitlines():
        parts = [p for p in line.strip().split(",") if p.strip()]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1])

    raise ValueError(f"Could not parse video dimensions from ffprobe output:\n{out}")


def _detect_orientation(src_w: int, src_h: int) -> str:
    """
    Classify source video orientation.
    Returns "portrait" | "landscape" | "square".
    FIX (critical): removed ~60 lines of dead unreachable code that previously
    sat after the return statements inside this function body.
    """
    if src_h > src_w:
        return "portrait"
    elif src_w > src_h:
        return "landscape"
    else:
        return "square"


def _precrop_offsets(src_w, src_h):
    """
    Compute (cw, ch, x0, y0) pixel values for the ffmpeg pre-crop filter,
    derived from PC_W/PC_H/PC_H_ANCHOR/PC_V_ANCHOR config constants.
    """
    valid_h = {"left", "center", "right"}
    valid_v = {"top", "middle", "bottom"}

    if PC_H_ANCHOR not in valid_h:
        raise ValueError(f"Invalid horizontal_anchor '{PC_H_ANCHOR}'. Must be: {sorted(valid_h)}")
    if PC_V_ANCHOR not in valid_v:
        raise ValueError(f"Invalid vertical_anchor '{PC_V_ANCHOR}'. Must be: {sorted(valid_v)}")

    cw = int(src_w * PC_W)
    ch = int(src_h * PC_H)

    if PC_H_ANCHOR == "left":
        x0 = 0
    elif PC_H_ANCHOR == "right":
        x0 = src_w - cw
    else:
        x0 = (src_w - cw) // 2

    if PC_V_ANCHOR == "top":
        y0 = 0
    elif PC_V_ANCHOR == "bottom":
        y0 = src_h - ch
    else:
        y0 = (src_h - ch) // 2

    return cw, ch, x0, y0


def precrop_video(src: str, out_path: Path) -> tuple:
    """
    Step 0: probe source dimensions, detect orientation, apply border-trim crop.

    Returns (out_path, orientation).
    Cached: delete _precrop.mp4 to force a re-crop with new settings.
    """
    src_w, src_h = _probe_dimensions(str(src))
    orientation  = _detect_orientation(src_w, src_h)

    if out_path.exists():
        print(f"⏭️  Pre-crop exists, reusing: {out_path.name}  [{orientation}]")
        print(f"   (delete it to re-crop with updated settings)")
        return out_path, orientation

    cw, ch, x0, y0 = _precrop_offsets(src_w, src_h)
    vf = f"crop={cw}:{ch}:{x0}:{y0}"

    print(f"✂️  Step 0: pre-cropping  [{orientation} source]")
    print(f"   source    : {src_w}×{src_h} px")
    print(f"   h_anchor  : {PC_H_ANCHOR}   keep {PC_W*100:.0f}% → {cw}px wide"
          f"   x0={x0}  x1={x0+cw}  (trim L:{x0} R:{src_w-x0-cw})")
    print(f"   v_anchor  : {PC_V_ANCHOR}   keep {PC_H*100:.0f}% → {ch}px tall"
          f"   y0={y0}  y1={y0+ch}  (trim T:{y0} B:{src_h-y0-ch})")
    print(f"   output    : {cw}×{ch} px")

    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_PRECROP),
        "-c:a", "copy",
        str(out_path),
    ])
    print(f"   ✅ Saved: {out_path}")
    return out_path, orientation


# =========================
# SPLIT VIDEO
# =========================

def _ffmpeg_encode(src, ss, duration, filt, out):
    """Single ffmpeg encode call with filter_complex. Used by split_video."""
    run([
        "ffmpeg", "-y",
        "-ss", f"{ss:.6f}",
        "-t",  f"{duration:.6f}",
        "-i",  str(src),
        "-filter_complex", filt,
        "-map", "[out]",
        "-map", "0:a?",
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
        "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
        str(out),
    ])


def _split_portrait(video, scenes, out_dir):
    """
    Portrait-source path: trim each scene and scale to OUT_W×OUT_H.
    No pan animation or zoom-out — those are landscape-only effects.
    """
    print("   📱 Portrait source — using direct trim+scale (no crop window)")
    for s in tqdm(scenes, desc="🎬 Splitting"):
        # FIX (critical): consistent scene filename — no hook suffix in scenes/
        out = out_dir / f"scene_{s['id']:02d}.mp4"
        if out.exists():
            continue

        run([
            "ffmpeg", "-y",
            "-ss", f"{s['start']:.6f}",
            "-t",  f"{s['end'] - s['start']:.6f}",
            "-i",  str(video),
            "-vf", f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                   f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
            "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
            str(out),
        ])


def split_video(video, scenes, out_dir, words, orientation: str = "landscape"):
    """
    Convert each scene to a portrait clip.
    FIX (critical): scene files are always named scene_NN.mp4 (no hook suffix).
    Hooks belong only in the final/ directory — keeping scenes/ clean allows
    the burn loop to reliably locate files by ID.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if orientation == "portrait":
        _split_portrait(video, scenes, out_dir)
        return

    cap   = cv2.VideoCapture(str(video))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    for s in tqdm(scenes, desc="🎬 Splitting"):
        # Consistent naming — scene ID only, no hook in the scenes/ dir
        out = out_dir / f"scene_{s['id']:02d}.mp4"
        if out.exists():
            continue

        scene_start, scene_end = s["start"], s["end"]
        eff_duration = scene_end - scene_start

        zoom_dur      = min(ZOOM_OUT_DURATION, eff_duration * ZOOM_OUT_MAX_PCT) if ZOOM_OUT_ENABLED else 0.0
        main_duration = eff_duration - zoom_dur

        # Single-pass analysis — replaces two separate video-open calls
        tier_info = analyze_scene(video, scene_start, scene_end)

        energy      = tier_info.get("motion_energy", 0.0)
        face_points = tier_info.get("face_path", [])

        spread = 0.0
        if len(face_points) > 1:
            x_coords = [p["x"] for p in face_points]
            spread   = max(x_coords) - min(x_coords)

        is_slide = (energy < MOTION_THRESHOLD) and (
            len(face_points) == 0 or len(face_points) > 10
        )

        if len(face_points) > 0 and len(face_points) <= 10:
            strategy_type = "action"
        elif is_slide:
            strategy_type = "slides"
        else:
            strategy_type = "action"

        print(f"   🔍 Scene {s['id']:02d}: energy={energy:.2f}  "
              f"detections={len(face_points)}  spread={spread:.2f}  "
              f"→ {strategy_type}")

        main_filt, crop_x_end = build_portrait_filter(tier_info, main_duration, strategy_type)

        tmp_main      = out_dir / f"_tmp_main_{s['id']:02d}.mp4"
        tmp_zoom      = out_dir / f"_tmp_zoom_{s['id']:02d}.mp4"
        tmp_zoom_enc  = out_dir / f"_tmp_zoom_enc_{s['id']:02d}.mp4"
        concat_txt    = out_dir / f"_tmp_concat_{s['id']:02d}.txt"

        try:
            _ffmpeg_encode(video, scene_start, main_duration, main_filt, tmp_main)

            if zoom_dur > 0:
                render_zoomout_opencv(
                    video, scene_start, scene_end, zoom_dur,
                    tier_info, int(crop_x_end), tmp_zoom,
                )

                run([
                    "ffmpeg", "-y",
                    "-i", str(tmp_zoom),
                    "-ss", f"{scene_start + main_duration:.6f}",
                    "-t",  f"{zoom_dur:.6f}",
                    "-i",  str(video),
                    "-map", "0:v", "-map", "1:a?",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
                    "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
                    str(tmp_zoom_enc),
                ])

                with open(concat_txt, "w", encoding="utf-8") as cf:
                    cf.write(f"file '{tmp_main.resolve().as_posix()}'\n")
                    cf.write(f"file '{tmp_zoom_enc.resolve().as_posix()}'\n")

                run([
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_txt),
                    "-c", "copy",
                    str(out),
                ])

            else:
                tmp_main.replace(out)

        finally:
            for f in [tmp_main, tmp_zoom, tmp_zoom_enc, concat_txt]:
                if f and Path(f).exists():
                    Path(f).unlink()


# =========================
# KARAOKE SUBTITLE LOGIC
# =========================

def words_for_scene(words, start, end):
    return [
        {
            "w": escape_ass(w["word"].strip()),
            "s": w["start"] - start,
            "e": w["end"]   - start,
        }
        for w in words if w["end"] > start and w["start"] < end
    ]


def chunk_words(words):
    if not words:
        return []

    max_chunk     = MAX_WORDS_PER_LINE * MAX_LINES
    gap_threshold = PAUSE_THRESHOLD_SEC

    chunks, current = [], [words[0]]

    for w in words[1:]:
        silence    = w["s"] - current[-1]["e"]
        over_limit = len(current) >= max_chunk

        if over_limit or silence >= gap_threshold:
            chunks.append(current)
            current = []

        current.append(w)

    if current:
        chunks.append(current)

    return chunks


# =========================
# SCENE BOUNDARY SNAPPING
# =========================

def snap_scenes_to_words(scenes: list, words: list, tolerance: float = 1.0) -> list:
    """
    Adjust scene boundaries so they fall on word ends rather than mid-sentence.
    Only internal boundaries are moved — first/last are hard edges.
    """
    if not words or len(scenes) < 2:
        return scenes

    word_ends = sorted(w["end"] for w in words)
    snapped   = [dict(s) for s in scenes]

    for i in range(len(snapped) - 1):
        original = snapped[i]["end"]
        lo, hi   = original - tolerance, original + tolerance
        candidates = [t for t in word_ends if lo <= t <= hi]

        if not candidates:
            print(f"   ⚠️  Scene {snapped[i]['id']}→{snapped[i+1]['id']}: "
                  f"no word end within ±{tolerance}s of {original:.2f}s — kept")
            continue

        best  = min(candidates, key=lambda t: abs(t - original))
        delta = best - original

        snapped[i]["end"]       = best
        snapped[i + 1]["start"] = best

        sign = "+" if delta >= 0 else ""
        print(f"   ✂️  Scene {snapped[i]['id']}→{snapped[i+1]['id']}: "
              f"{original:.3f}s → {best:.3f}s  ({sign}{delta:.3f}s)")

    return snapped


# =========================
# SUBTITLE PRESETS
# =========================

def get_subtitle_style(detected_lang: str = "en") -> dict:
    """
    Return the subtitle style dict for the current preset.
    FIX (high): language is now passed in as a parameter instead of being read
    from globals(). This ensures correct font selection even when words are
    loaded from cache (--regen-words not passed).
    """
    presets = {
        "classic": {
            "font_name": "Inter SemiBold",
            "size": 92,
            "color": "#FFFFFF",
            "outline": 2,
            "shadow": 1,
            "uppercase": False,
        },
        "loud_clear": {
            "font_name": "Montserrat ExtraBold",
            "size": 92,
            "color": "#FFFFFF",
            "outline": 4,
            "shadow": 0,
            "uppercase": True,
        },
        "hype_mode": {
            "font_name": "Anton",
            "size": 105,
            "color": "#FF0000",
            "outline": 6,
            "shadow": 0,
            "uppercase": True,
        },
        "vlog_pop": {
            "font_name": "Poppins Bold",
            "size": 92,
            "color": "#00CFFF",
            "outline": 3,
            "shadow": 1,
            "uppercase": False,
        },
        "talk_show": {
            "font_name": "Roboto Bold",
            "size": 92,
            "color": "#FFFFFF",
            "outline": 2,
            "shadow": 3,
            "uppercase": False,
        },
    }

    style = presets.get(SUBTITLE_PRESET, presets["classic"]).copy()

    NON_ENGLISH_FONT_MAP = {
        "hi": "Noto Sans Devanagari",
        "ar": "Noto Sans Arabic",
        "zh": "Noto Sans SC",
        "ja": "Noto Sans JP",
        "ko": "Noto Sans KR",
    }

    lang = detected_lang.split("-")[0].lower()
    if lang != "en":
        style["font_name"] = NON_ENGLISH_FONT_MAP.get(lang, "Noto Sans")

    print(f"🎨 Font selected: {style['font_name']} (lang={lang})")
    return style


# =========================
# ASS FILE WRITING
# =========================

def _ass_header(f, detected_lang: str = "en"):
    style     = get_subtitle_style(detected_lang)
    font_name = style["font_name"]
    font_size = style["size"]
    color     = ass_color(style["color"])
    outline   = style["outline"]
    shadow    = style["shadow"]

    f.write(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {OUT_W}\n"
        f"PlayResY: {OUT_H}\n\n"

        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"

        f"Style: Base,{font_name},{font_size},"
        f"{color},{color},&H00000000&,&H00000000&,"
        f"0,0,0,0,100,100,0,0,"
        f"1,{outline},{shadow},"
        f"{ALIGNMENT},{MARGIN_H},{MARGIN_H},{MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def render_karaoke_chunk(f, chunk, detected_lang: str = "en"):
    """
    Emit one dialogue event per word.  Each event spans word[i].start →
    word[i+1].start (last word: end + extend), shows ALL words in the chunk,
    and highlights the active word with a coloured pill background.
    """
    style    = get_subtitle_style(detected_lang)
    inact_c  = ass_color(style["color"])
    act_c    = ass_color(ACTIVE_COLOR)
    bg_alpha = opacity_to_alpha(BG_OPACITY)
    bg_c     = ass_color(BG_COLOR, alpha=bg_alpha)

    if HL_ENABLED:
        open_tag  = "{" + "\\1c" + act_c + "\\3c" + bg_c + "\\bord" + str(BG_BORD) + "\\blur0}"
    else:
        open_tag  = "{" + "\\1c" + act_c + "}"
    close_tag = "{" + "\\1c" + inact_c + "\\bord2\\blur0}"

    for i, word in enumerate(chunk):
        ev_start = word["s"]
        ev_end   = (chunk[i + 1]["s"] - 0.05) if i + 1 < len(chunk) else word["e"] + EXTEND_LAST_WORD_SEC

        parts = []
        for j, w in enumerate(chunk):
            raw = w["w"]
            if style["uppercase"]:
                raw = raw.upper()
            if j == i:
                parts.append(open_tag + raw + close_tag)
            else:
                parts.append(raw)

        text = " ".join(parts)
        f.write(f"Dialogue:0,{ts(ev_start)},{ts(ev_end)},Base,,0,0,0,,{text}\n")


def write_ass(path, scene_words, detected_lang: str = "en"):
    """
    Write an ASS subtitle file for a scene.
    detected_lang drives font selection — passed through from main() so it is
    correct even when words are loaded from cache.
    """
    with open(path, "w", encoding="utf8") as f:
        _ass_header(f, detected_lang)
        if not scene_words:
            return
        for chunk in chunk_words(scene_words):
            render_karaoke_chunk(f, chunk, detected_lang)


# =========================
# LOGO + BURN
# =========================

def _build_logo_filter(ass_escaped: str, logo_path: str) -> tuple:
    """
    Build the ffmpeg -filter_complex and input args for logo + subtitle burn.

    CLI  → which files (video, logo)
    Config → how to render (size, position, opacity)

    Returns (extra_inputs, filter_complex, map_arg).
    filter_complex is None if logo is disabled — caller uses plain -vf then.
    """
    logo_cfg = LOGO_CFG

    if not logo_cfg.get("enabled", False):
        return [], None, None

    if not logo_path or not Path(logo_path).exists():
        print(f"   ⚠️  Logo enabled but file not found: '{logo_path}' — skipping logo")
        return [], None, None

    corner    = str(logo_cfg.get("corner", "bottom-right")).lower()
    width_pct = float(logo_cfg.get("width_pct", 0.12))
    margin_x  = int(logo_cfg.get("margin_x", 30))
    margin_y  = int(logo_cfg.get("margin_y", 30))
    opacity   = float(logo_cfg.get("opacity", 1.0))
    logo_w    = int(OUT_W * width_pct)

    ox = f"W-w-{margin_x}" if "right"  in corner else str(margin_x)
    oy = f"H-h-{margin_y}" if "bottom" in corner else str(margin_y)

    # FIX (low): use shared escape utility — consistent with ASS and FONTS_DIR
    logo_path_esc = escape_ffmpeg_path(logo_path)

    filter_complex = (
        f"[0:v]subtitles='{ass_escaped}':fontsdir='{FONTS_DIR}'[subbed];"
        f"[1:v]scale={logo_w}:-1[logo_scaled];"
        f"[subbed][logo_scaled]overlay={ox}:{oy}:format=auto,"
        f"colorchannelmixer=aa={opacity:.3f}[out]"
    )

    return ["-i", logo_path], filter_complex, "[out]"


def burn(src, ass, out, logo_path: str = "", force: bool = False):
    """
    Burn subtitles (and optional logo) onto a scene clip.
    """
    if out.exists() and not force:
        return

    ass_escaped = escape_ffmpeg_path(ass)
    extra_inputs, filter_complex, map_out = _build_logo_filter(ass_escaped, logo_path)

    if filter_complex:
        run([
            "ffmpeg", "-y",
            "-i", str(src),
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", map_out,
            "-map", "0:a?",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
            "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
            str(out),
        ])
    else:
        run([
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", f"subtitles='{ass_escaped}':fontsdir='{FONTS_DIR}'",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
            "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
            str(out),
        ])


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser(
        description="Convert horizontal video to vertical shorts with karaoke subtitles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each step only runs if its output is missing. Use flags to force re-runs.

Steps and flags:
  Step 0  pre-crop      runs if _precrop.mp4 missing; delete to re-crop
  Step 1  transcribe    --regen-words   (re-runs Whisper, overwrites words.json)
  Step 2  scene detect  runs if visual_scenes.json missing
  Step 3  hybrid split  always re-derives from cached visual scenes + words
  Step 4  split scenes  --regen-splits
  Step 5  gen subtitles --regen-subs
  Step 6  burn          --force-burn

Common workflows:
  First run:
    python process.py video.mp4 logo.png --regen-words --regen-subs --regen-splits --force-burn

  Change subtitle style only:
    python process.py video.mp4 logo.png --regen-subs --force-burn

  Change logo / burn settings only:
    python process.py video.mp4 logo.png --force-burn
        """,
    )
    ap.add_argument("video", help="Path to the source video file")
    ap.add_argument("logo",  help="Path to logo image (PNG). Placement controlled via config.")
    ap.add_argument("--regen-words",  action="store_true", help="Re-run Whisper; overwrite words.json")
    ap.add_argument("--regen-subs",   action="store_true", help="Re-write ASS files from words.json")
    ap.add_argument("--regen-splits", action="store_true", help="Re-split and re-encode all scene files")
    ap.add_argument("--force-burn",   action="store_true", help="Re-burn final files even if they exist")
    args = ap.parse_args()

    name        = Path(args.video).stem
    base        = Path("output") / name
    scenes_dir  = base / "scenes"
    subs_dir    = base / "subtitles"
    final_dir   = base / "final"
    meta        = base / "scenes.json"
    precropped  = base / f"{name}_precrop.mp4"
    words_cache = base / "words.json"
    lang_cache  = base / "lang.txt"      # FIX (high): persist language separately

    for d in (base, scenes_dir, subs_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬  Processing: {args.video}")
    print(f"    output   → {base}")
    print(f"    logo     → {args.logo}")

    # ── Step 0: Pre-crop ──────────────────────────────────────────────
    precropped, orientation = precrop_video(args.video, precropped)
    print(f"    orientation: {orientation}")

    # ── Step 1: Transcribe ────────────────────────────────────────────
    # FIX (critical): single Whisper call — language detected from the same
    # res dict, not a separate full transcribe pass that was previously thrown away.
    # FIX (high): language persisted to lang.txt so font selection is correct
    # even on cache hits (when --regen-words is not passed).
    if args.regen_words or not words_cache.exists():
        print("🧠  Transcribing with Whisper…")
        src_for_whisper = str(precropped) if precropped.exists() else args.video
        model = whisper.load_model("medium")

        res  = model.transcribe(src_for_whisper, word_timestamps=True)
        lang = res.get("language", "en")
        print(f"   detected language: {lang}")

        del model
        gc.collect()

        words = [w for seg in res["segments"] for w in seg["words"]]

        # FIX (high): use context manager — no leaked file handles on Windows
        with open(words_cache, "w", encoding="utf8") as f:
            json.dump(words, f, indent=2)
        with open(lang_cache, "w", encoding="utf8") as f:
            f.write(lang)

        print(f"   💾 {len(words)} words → {words_cache.name}")
        print(f"   💾 language '{lang}' → {lang_cache.name}")

    else:
        # FIX (high): use context manager for cache load too
        with open(words_cache, encoding="utf8") as f:
            words = json.load(f)

        lang = "en"
        if lang_cache.exists():
            with open(lang_cache, encoding="utf8") as f:
                lang = f.read().strip() or "en"

        print(f"   📖 Loaded {len(words)} words from cache  (lang={lang})")

    # ── Step 2: Visual scene detection ────────────────────────────────
    visual_cache = base / "visual_scenes.json"

    if visual_cache.exists() and not args.regen_splits:
        with open(visual_cache, encoding="utf8") as f:
            visual_scenes = json.load(f)
        print(f"   📖 Loaded visual scenes from cache")
    else:
        # FIX (high): capture real fps from detect_scenes — no more hardcoded 25
        print("🔍  Detecting visual scenes…")
        source_fps, visual_scenes = detect_scenes(str(precropped))
        with open(visual_cache, "w", encoding="utf8") as f:
            json.dump(visual_scenes, f, indent=2)
        print(f"   💾 {len(visual_scenes)} visual scenes → {visual_cache.name}")

    # ── Step 3: Hybrid scene splitting ────────────────────────────────
    print("🧠  Creating hybrid scenes (visual + speech)…")
    scenes = hybrid_split(words, visual_scenes, min_dur=35, max_dur=65, pause_thresh=1.1)
    scenes = merge_short_scenes(scenes, min_duration=25)

    # Scene burst protection
    if len(scenes) > 40:
        print("⚠️  Too many scenes — auto-adjusting thresholds")
        scenes = hybrid_split(words, visual_scenes, min_dur=45, max_dur=75, pause_thresh=1.3)
        scenes = merge_short_scenes(scenes, min_duration=30)

    # FIX (critical): single guard — the duplicate was unreachable and misleading
    if not scenes:
        raise ValueError("No scenes generated. Adjust thresholds in editor.config.json.")

    duration = scenes[-1]["end"]

    # FIX (high): use actual source fps, not hardcoded 25
    # source_fps may not be set if visual scenes came from cache — probe if needed
    if "source_fps" not in dir():
        cap = cv2.VideoCapture(str(precropped))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

    with open(meta, "w", encoding="utf8") as f:
        json.dump(
            {
                "video":       str(precropped),
                "fps":         source_fps,
                "duration":    duration,
                "orientation": orientation,
                "scenes":      scenes,
            },
            f,
            indent=2,
        )

    print(f"   💾 {len(scenes)} hybrid scenes → {meta.name}")

    # ── Step 4: Split scenes ──────────────────────────────────────────
    if args.regen_splits:
        print("🗑️  Clearing existing scene files for re-split…")
        for f in scenes_dir.glob("scene_*.mp4"):
            f.unlink()

    split_video(str(precropped), scenes, scenes_dir, words, orientation=orientation)

    # ── Step 5: Generate subtitle ASS files ──────────────────────────
    if words:
        subs_needed = [
            s for s in scenes
            if args.regen_subs or not (subs_dir / f"scene_{s['id']:02d}.ass").exists()
        ]

        if subs_needed:
            print(f"📝  Writing subtitles ({len(subs_needed)} scenes)…")
            for s in subs_needed:
                sw = words_for_scene(words, s["start"], s["end"])
                # FIX (high): pass detected_lang explicitly — no more globals()
                write_ass(subs_dir / f"scene_{s['id']:02d}.ass", sw, detected_lang=lang)
        else:
            print("📝  All subtitle files present — skipping (use --regen-subs to regenerate)")
    else:
        print("   ⚠️  No words available — subtitles skipped")
        print("        Run with --regen-words to transcribe first")

    # ── Step 6: Burn subtitles + logo onto final clips ────────────────
    print("🎞️   Burning…")
    burned, skipped, missing = 0, 0, 0

    for s in tqdm(scenes, desc="🔥 Burning"):
        base_name   = f"scene_{s['id']:02d}"
        scene_path  = scenes_dir / f"{base_name}.mp4"
        ass_path    = subs_dir   / f"{base_name}.ass"

        # FIX (critical): hook and summary always initialized before use.
        # Previously caused NameError when scene had no transcribed words.
        hook    = ""
        summary = ""
        scene_words = words_for_scene(words, s["start"], s["end"])

        if scene_words:
            scene_text = words_to_text(scene_words)
            summary    = summarize_text_simple(scene_text)
            hook       = generate_hook_simple(summary)

        safe_hook  = clean_filename_unicode(hook) if hook else "clip"
        final_name = f"{base_name}_{safe_hook}.mp4"
        final_path = final_dir / final_name

        # Save per-scene metadata
        meta_path = final_dir / f"{base_name}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"hook": hook, "summary": summary, "start": s["start"], "end": s["end"]},
                f,
                indent=2,
            )

        if not scene_path.exists():
            print(f"   ⚠️  {base_name}: scene MP4 missing — skipped")
            missing += 1
            continue
        if not ass_path.exists():
            print(f"   ⚠️  {base_name}: ASS file missing — skipped")
            missing += 1
            continue
        if final_path.exists() and not args.force_burn:
            skipped += 1
            continue

        burn(scene_path, ass_path, final_path, logo_path=args.logo, force=args.force_burn)
        burned += 1

    print(f"   ✅  burned={burned}  skipped={skipped}  missing={missing}")
    print("\n✅  Done.")


if __name__ == "__main__":
    main()
