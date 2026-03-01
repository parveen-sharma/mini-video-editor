import argparse, json, subprocess, gc, re
from pathlib import Path
from tqdm import tqdm
import whisper, cv2, numpy as np

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
    # Processing constants (formerly hardcoded in functions)
    SCENE_SAMPLE_FPS, SCENE_DIFF_THRESHOLD, SCENE_MIN_DURATION_SEC,
    MOTION_THRESHOLD, ANALYSIS_SAMPLE_FPS,
    MOTION_PIXEL_THRESHOLD, SALIENCY_CENTER_DEADBAND,
    BLUR_LUMA_RADIUS, BLUR_LUMA_POWER, BLUR_OPENCV_KSIZE,
    ZOOM_OUT_ENABLED, ZOOM_OUT_MAX_PCT,
    ENCODE_CRF_PRECROP, ENCODE_CRF_SCENES, ENCODE_AUDIO_BR, ENCODE_PRESET,
    FACE_DETECTION_ENABLED,
    SUBTITLE_PRESET   
)

# =========================
# FONTS
# =========================

# Absolute path to the fonts/ folder next to this script.
# Using __file__ makes it work regardless of which directory you run from.
# Passed to ffmpeg subtitles filter as fontsdir= so custom preset fonts load.
FONTS_DIR = str(Path(__file__).resolve().parent / "fonts").replace("\\", "/").replace(":", "\\:")

# =========================
# UTILS
# =========================

def run(cmd):
    subprocess.run(cmd, check=True)

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

def _get_face_bounds(frame_bgr):
    """Returns the normalized center (x, y) of all detected faces."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    if len(faces) == 0:
        return 0.5, 0.5 # Default to center
        
    # Calculate the average center of all faces found
    avg_x = sum([x + w/2 for (x, y, w, h) in faces]) / len(faces)
    avg_y = sum([y + h/2 for (x, y, w, h) in faces]) / len(faces)
    
    img_h, img_w = frame_bgr.shape[:2]
    return avg_x / img_w, avg_y / img_h
    
# =========================
# SCENE DETECTION
# =========================

def detect_scenes(video):
    """
    Detect scene cuts by diffing sampled grayscale frames.

    Previously hardcoded idx%6 and threshold=22 meant behaviour changed
    completely with source framerate — at 120fps it sampled every 0.05s
    (4× too fast) producing many false cuts from minor motion/flicker.

    Now uses SCENE_SAMPLE_FPS (default 5) to compute sample_interval from
    actual source fps, so we always diff ~5 times/second regardless of
    whether the source is 24, 30, 60, or 120fps.

    Config constants (processing.scene_detection in editor.config.json):
      SCENE_SAMPLE_FPS       — target samples per second (default 5.0)
      SCENE_DIFF_THRESHOLD   — mean pixel diff to trigger a cut (default 22.0)
      SCENE_MIN_DURATION_SEC — discard scenes shorter than this (default 1.0)
    """
    cap   = cv2.VideoCapture(video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Derive sample interval from actual fps so behaviour is framerate-agnostic.
    # e.g. 30fps / 5 samples/s = check every 6th frame
    #      120fps / 5 samples/s = check every 24th frame  (was every 6th — 4× too fast)
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
# SCENE ANALYSIS — motion centroid + saliency fallback
# =========================

def _motion_center(gray_frames: list) -> tuple:
    """
    Compute the median centroid of changed pixels across consecutive frame pairs.

    Works by diffing adjacent sampled frames, thresholding to a binary mask of
    changed pixels, then computing the centroid of that mask via image moments.
    Returns (cx_frac, cy_frac, mean_motion_energy).

    cx_frac / cy_frac are normalised 0-1 relative to the frame dimensions.
    mean_motion_energy is the average mean diff across all pairs — used by the
    caller to decide whether there was enough motion to trust this signal.
    """
    cx_samples, cy_samples, energies = [], [], []
    h, w = gray_frames[0].shape

    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        energies.append(float(diff.mean()))

        _, mask = cv2.threshold(diff, MOTION_PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Dilate slightly to merge nearby changed regions into blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.dilate(mask, kernel, iterations=1)

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
    Expert Refinement: Filters out small, likely-false detections (text/icons).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Increase minNeighbors to 8 to reduce 'hallucinated' faces on slides
    faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(120, 120))
    
    if len(faces) == 0:
        return None
        
    # Pick the largest face (most likely the subject)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    img_h, img_w = frame_bgr.shape[:2]
    return (x + w/2) / img_w, (y + h/2) / img_h

def _saliency_center(frame_bgr: np.ndarray) -> tuple:
    """
    Use OpenCV's SpectralResidual saliency on a single frame to find the most
    visually prominent region. Good for static scenes (title cards, diagrams)
    where motion gives no signal.

    Returns (cx_frac, cy_frac).
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    ok, sal_map = saliency.computeSaliency(frame_bgr)

    if not ok:
        return 0.5, 0.5

    sal_8 = (sal_map * 255).astype("uint8")
    M = cv2.moments(sal_8)

    if M["m00"] > 0:
        h, w = frame_bgr.shape[:2]
        return M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h

    return 0.5, 0.5

def analyze_scene(video_path: str, scene_start: float, scene_end: float) -> dict:
    """
    Tiered strategy: Face Detection -> Motion -> Saliency -> Center
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(fps / ANALYSIS_SAMPLE_FPS))
    start_frame    = int(scene_start * fps)
    end_frame      = int(scene_end   * fps)
    mid_frame      = start_frame + (end_frame - start_frame) // 2

    gray_frames   = []
    middle_region = None

    # Sample frames for analysis
    for fi in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok: break
        
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if middle_region is None or abs(fi - mid_frame) < frame_interval:
            middle_region = frame.copy()

    cap.release()

    # ── Tier 0: Face Detection ───────────────────────────────────────
    if FACE_DETECTION_ENABLED and middle_region is not None:
        face_res = _face_center(middle_region)
        if face_res:
            cx, cy = face_res
            return {"src_w": src_w, "src_h": src_h, "cx_frac": cx, "cy_frac": cy,
                    "strategy": "face", "motion_energy": 0.0}

    # ── Tier 1: Motion (Existing) ────────────────────────────────────
    if len(gray_frames) >= 2:
        cx, cy, energy = _motion_center(gray_frames)
        if energy >= MOTION_THRESHOLD:
            return {"src_w": src_w, "src_h": src_h, "cx_frac": cx, "cy_frac": cy,
                    "strategy": "motion", "motion_energy": energy}
    else:
        energy = 0.0

    # ── Tier 2: Saliency (Existing) ──────────────────────────────────
    if middle_region is not None:
        cx, cy = _saliency_center(middle_region)
        if not (0.5 - SALIENCY_CENTER_DEADBAND < cx < 0.5 + SALIENCY_CENTER_DEADBAND):
            return {"src_w": src_w, "src_h": src_h, "cx_frac": cx, "cy_frac": cy,
                    "strategy": "saliency", "motion_energy": energy}

    # ── Tier 3: Center Fallback ──────────────────────────────────────
    return {"src_w": src_w, "src_h": src_h, "cx_frac": 0.5, "cy_frac": 0.5,
            "strategy": "center", "motion_energy": energy}

def analyze_scene_dynamic(video_path, start_s, end_s):
    """
    Expert Refinement: Only counts a 'Face' if it is large enough to be a human.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    path = []
    for t in np.arange(start_s, end_s, 0.5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # minSize ensures we don't count tiny text-artifacts as faces
        # 0.15 * height is a good threshold for a person in a video
        min_dim = int(img_h * 0.15)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(min_dim, min_dim))
        
        for (x, y, w, h) in faces:
            path.append({"t": t - start_s, "x": (x + w/2) / frame.shape[1], "y": (y + h/2) / img_h})
    
    cap.release()
    return path
    
def build_portrait_filter(info: dict, scene_duration: float, strategy_type: str):
    """
    Expert Refinement: Generates dynamic FFmpeg crop expressions.
    - Slides: Slow Pan 0 -> max_range to keep static content engaging.
    - Action/Face: Anchors the 9:16 window to the identified subject center.
    """
    src_w = info["src_w"]
    src_h = info["src_h"]
    port_w = int(src_h * 9 / 16)
    pan_range = max(0, src_w - port_w)
    
    # ── Background Blur (Standard Pipeline) ──────────────────────────
    bg = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        f"boxblur=luma_radius={BLUR_LUMA_RADIUS}:luma_power={BLUR_LUMA_POWER}"
    )

    # ── Foreground Logic ─────────────────────────────────────────────
    if strategy_type == "slides":
        # Professional Ken Burns effect: Pan across the slide
        # t = current timestamp in the filter, duration = scene length
        x_expr = f"({pan_range}*t/{scene_duration})"
        crop_x_end = pan_range
        print(f"   🎥 Strategy: SLIDE (Panning 0 -> {pan_range}px)")
    else:
        # Real-world: Anchor the 1080px window to the detected center
        target_cx = info["cx_frac"] * src_w
        static_x = max(0, min(src_w - port_w, int(target_cx - port_w // 2)))
        x_expr = str(static_x)
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
    Place content_crop (any aspect ratio) onto a OUT_W x OUT_H canvas.

    Strategy:
      1. Scale content_crop to fit WITHIN OUT_W x OUT_H preserving aspect ratio.
         This is the key fix — never stretch or squish the image.
      2. Fill the canvas with the full_frame scaled+blurred to OUT_W x OUT_H.
         This gives an organic background instead of black bars.
      3. Overlay the scaled content centered on the blurred background.

    ┌────────────────────────────────────────────┐
    │  blurred full frame (OUT_W x OUT_H)        │
    │  ┌──────────────────────────────────────┐  │
    │  │  content_crop scaled to fit,         │  │
    │  │  aspect ratio preserved              │  │
    │  └──────────────────────────────────────┘  │
    └────────────────────────────────────────────┘
    """
    # ── Background: full frame blurred to fill canvas ─────────────────
    bg = cv2.resize(full_frame, (OUT_W, OUT_H))
    bg = cv2.GaussianBlur(bg, (BLUR_OPENCV_KSIZE, BLUR_OPENCV_KSIZE), 0)

    # ── Foreground: scale to fit, preserve aspect ratio ───────────────
    h, w = content_crop.shape[:2]
    scale   = min(OUT_W / w, OUT_H / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    fg      = cv2.resize(content_crop, (new_w, new_h))

    # ── Center fg on bg ───────────────────────────────────────────────
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

    WHY NOT FFMPEG FILTER EXPRESSIONS:
      ffmpeg's crop filter only supports dynamic expressions for x and y —
      not w and h. Those are evaluated once at filter-graph init time, so
      any expression involving t fails immediately with "Error when evaluating".
      OpenCV gives us per-frame control with no expression parser involved.

    SOURCE FRAME — middle of scene:
      Rather than reading live frames from zoom_start (which may be a
      transition frame), we grab the MIDDLE frame of the scene. This is
      the most representative, stable content of the slide. We then
      repeat/hold that single frame for the entire zoom-out duration.

      This also avoids any "last frame looks incomplete" issue — the
      middle frame always has fully-rendered slide content.

    ASPECT RATIO:
      The expanding crop region is composited via _composite_frame() which
      scales to fit (not fill) and blurs the full frame as background.
      This preserves the horizontal/vertical proportions at all zoom levels.

    ANIMATION:
      progress 0.0 → portrait-width crop at crop_x_end (matches main clip end)
      progress 1.0 → full source width, x=0 (entire slide visible)
    """
    src_w  = info["src_w"]
    src_h  = info["src_h"]
    port_w = int(src_h * 9 / 16)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ── Grab the middle frame of the scene ────────────────────────────
    mid_frame_idx = int((scene_start + (scene_end - scene_start) / 2) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ok, source_frame = cap.read()
    if not ok:
        # Fallback: first frame of zoom window
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(scene_start * fps))
        _, source_frame = cap.read()
    cap.release()

    n_frames = max(1, int(zoom_duration * fps))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (OUT_W, OUT_H))

    for i in range(n_frames):
        progress = i / max(n_frames - 1, 1)   # 0.0 → 1.0

        # Expanding crop window across the source frame
        cw = int(port_w + (src_w - port_w) * progress)
        cx = int(crop_x_end * (1 - progress))
        ch = src_h

        # Clamp — crop window must stay within frame bounds
        cw = max(1, min(cw, src_w - cx))

        content_crop = source_frame[0:ch, cx:cx + cw]

        # Composite with aspect-ratio-preserving scale + blurred background
        out_frame = _composite_frame(content_crop, source_frame)
        writer.write(out_frame)

    writer.release()



# =========================
# STEP 0 — PRE-CROP
# =========================

def _probe_dimensions(src: str) -> tuple:
    """
    Use ffprobe to read the exact pixel dimensions of a video file.
    Returns (width, height) as integers.
    """
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(src),
    ], text=True).strip()
    # MOV files (e.g. iPhone) can have multiple streams (video + timecode track).
    # ffprobe returns one line per stream — take only the first non-empty line
    # with exactly two comma-separated integer values (width,height).
    for line in out.splitlines():
        # Strip trailing commas/whitespace, filter empty tokens.
        # iPhone MOV files produce "1920,1080," (trailing comma) which
        # splits to ["1920", "1080", ""] — filtering empties handles it.
        parts = [p for p in line.strip().split(",") if p.strip()]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1])
    raise ValueError(
        f"Could not parse video dimensions from ffprobe output:\n{out}"
    )


def _detect_orientation(src_w: int, src_h: int) -> str:
    """
    Classify source video orientation based on aspect ratio.

    Returns:
      "portrait"  — H > W (e.g. 1080×1920, iPhone selfie/video)
      "landscape" — W > H (e.g. 1920×1080, screen recording, camera wide)
      "square"    — W == H

    Implication for the pipeline:
      landscape → full portrait conversion (crop window, pan, zoom-out)
      portrait  → already correct shape; just trim + scale, no crop window needed
      square    → treated as landscape (will be pillarboxed to portrait)
    """
    if src_h > src_w:
        return "portrait"
    elif src_w > src_h:
        return "landscape"
    else:
        return "square"
    """
    Compute concrete pixel values for the pre-crop step.

    Takes the actual source dimensions (probed via ffprobe) and the
    config values PC_W, PC_H, PC_H_ANCHOR, PC_V_ANCHOR, returning
    integer pixel values ready to embed in an ffmpeg crop filter.

    Config keys (editor.config.json → "precrop"):
      horizontal_keep_pct  — fraction of WIDTH  to retain  (0.0–1.0)
      vertical_keep_pct    — fraction of HEIGHT to retain  (0.0–1.0)
      horizontal_anchor    — where to anchor horizontally: left | center | right
      vertical_anchor      — where to anchor vertically:   top  | middle | bottom

    Returns (cw, ch, x0, y0) as integers.

    Offset table for a W×H source:

      h_anchor   x0                    right edge
      ────────   ────────────────────  ──────────
      left       0                     cw
      center     (W - cw) // 2         (W - cw)//2 + cw
      right      W - cw                W

      v_anchor   y0                    bottom edge
      ────────   ────────────────────  ──────────
      top        0                     ch
      middle     (H - ch) // 2         (H - ch)//2 + ch
      bottom     H - ch                H
    """
    valid_h = {"left", "center", "right"}
    valid_v = {"top", "middle", "bottom"}

    if PC_H_ANCHOR not in valid_h:
        raise ValueError(
            f"Invalid horizontal_anchor '{PC_H_ANCHOR}'. "
            f"Must be one of: {sorted(valid_h)}"
        )
    if PC_V_ANCHOR not in valid_v:
        raise ValueError(
            f"Invalid vertical_anchor '{PC_V_ANCHOR}'. "
            f"Must be one of: {sorted(valid_v)}"
        )

    cw = int(src_w * PC_W)
    ch = int(src_h * PC_H)

    # ── Horizontal offset ────────────────────────────────────────────
    if PC_H_ANCHOR == "left":
        x0 = 0
    elif PC_H_ANCHOR == "right":
        x0 = src_w - cw
    else:   # center
        x0 = (src_w - cw) // 2

    # ── Vertical offset ──────────────────────────────────────────────
    if PC_V_ANCHOR == "top":
        y0 = 0
    elif PC_V_ANCHOR == "bottom":
        y0 = src_h - ch
    else:   # middle
        y0 = (src_h - ch) // 2

    return cw, ch, x0, y0

def _precrop_offsets(src_w, src_h):
    """
    Compute concrete pixel values for the pre-crop step.
    """
    valid_h = {"left", "center", "right"}
    valid_v = {"top", "middle", "bottom"}

    if PC_H_ANCHOR not in valid_h:
        raise ValueError(
            f"Invalid horizontal_anchor '{PC_H_ANCHOR}'. "
            f"Must be one of: {sorted(valid_h)}"
        )
    if PC_V_ANCHOR not in valid_v:
        raise ValueError(
            f"Invalid vertical_anchor '{PC_V_ANCHOR}'. "
            f"Must be one of: {sorted(valid_v)}"
        )

    cw = int(src_w * PC_W)
    ch = int(src_h * PC_H)

    # ── Horizontal offset ────────────────────────────────────────────
    if PC_H_ANCHOR == "left":
        x0 = 0
    elif PC_H_ANCHOR == "right":
        x0 = src_w - cw
    else:   # center
        x0 = (src_w - cw) // 2

    # ── Vertical offset ──────────────────────────────────────────────
    if PC_V_ANCHOR == "top":
        y0 = 0
    elif PC_V_ANCHOR == "bottom":
        y0 = src_h - ch
    else:   # middle
        y0 = (src_h - ch) // 2

    return cw, ch, x0, y0

def precrop_video(src: str, out_path: Path) -> tuple:
    """
    Step 0: probe source dimensions, detect orientation, apply border-trim crop,
    save a clean intermediate used by all downstream steps.

    Returns (out_path, orientation) where orientation is "portrait" | "landscape" | "square".

    Orientation drives the split strategy:
      landscape → portrait conversion (crop window + pan + zoom-out)
      portrait  → already vertical; just trim borders, then scale in split

    Cached: delete _precrop.mp4 to force re-crop with new settings.
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
    print(f"   filter    : {vf}")

    run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", vf,
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_PRECROP),
        "-c:a", "copy",
        str(out_path),
    ])
    print(f"   ✅ Saved: {out_path}")
    return out_path, orientation

# =========================
# SPLIT VIDEO
# =========================

def _ffmpeg_encode(src, ss, duration, filt, out):
    """Single ffmpeg encode call with filter_complex, used by split_video."""
    run([
        "ffmpeg", "-y",
        "-ss", f"{ss:.6f}",
        "-t",  f"{duration:.6f}",   # -t (duration) not -to (end time) — more precise
        "-i",  str(src),
        "-filter_complex", filt,
        "-map", "[out]",
        "-map", "0:a?",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
        "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
        str(out),
    ])


def _split_portrait(video, scenes, out_dir):
    """
    Split path for portrait-source videos (e.g. iPhone selfie/video).

    The source is already vertical — no portrait crop window needed.
    We just trim each scene and scale to OUT_W×OUT_H.

    No pan animation, no zoom-out outro — those are landscape-specific effects
    that make no sense on talking-head or portrait-shot content.
    """
    print("   📱 Portrait source — using direct trim+scale (no crop window)")
    for s in tqdm(scenes, desc="🎬 Splitting"):
        out = out_dir / f"scene_{s['id']:02d}.mp4"
        if out.exists():
            continue

        start    = s["start"]
        duration = s["end"] - s["start"]

        run([
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-t",  f"{duration:.6f}",
            "-i",  str(video),
            "-vf", f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                   f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2",
            "-pix_fmt", "yuv420p", "-c:v", "libx264",
            "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
            "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
            str(out),
        ])

def split_video(video, scenes, out_dir, orientation: str = "landscape"):
    """
    Refined Splitter: Automatically detects Slides vs. Real-World content
    and applies the appropriate camera behavior for vertical conversion.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if orientation == "portrait":
        _split_portrait(video, scenes, out_dir)
        return

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frame_dur = 1.0 / fps

    for s in tqdm(scenes, desc="🎬 Splitting"):
        out = out_dir / f"scene_{s['id']:02d}.mp4"
        if out.exists(): continue

        scene_start, scene_end = s["start"], s["end"]
        eff_duration = scene_end - scene_start
        
        # Determine zoom vs main duration based on config
        zoom_dur = min(ZOOM_OUT_DURATION, eff_duration * ZOOM_OUT_MAX_PCT) if ZOOM_OUT_ENABLED else 0.0
        main_duration = eff_duration - zoom_dur

        # ── Step 1: Multi-Tier Content Analysis ─────────────────────
        # 1. Face Tracking path
        path_data = analyze_scene_dynamic(video, scene_start, scene_end)
        # 2. Motion/Saliency path
        tier_info = analyze_scene(video, scene_start, scene_end)
        
        # ── Step 2: Behavior Selection (The Group vs. Slide Logic) ──
        energy = tier_info.get("motion_energy", 0.0)
        face_points = path_data
        
        # Calculate how 'spread out' the detections are
        if len(face_points) > 1:
            x_coords = [p['x'] for p in face_points]
            # Spread = difference between furthest left and right 'face'
            spread = max(x_coords) - min(x_coords)
        else:
            spread = 0

        # Expert Decision Logic:
        # 1. If energy is ultra-low (0.14) -> Slide.
        # 2. If face count is massive (>10) but they don't move -> Slide.
        # 3. If faces are grouped (spread < 0.5) -> Group Video.
        
        is_slide = (energy < MOTION_THRESHOLD) and (len(face_points) == 0 or len(face_points) > 10)
        
        if len(face_points) > 0 and len(face_points) <= 10:
            strategy_type = "action" # Trust the group tracking
        elif is_slide:
            strategy_type = "slides" # Trigger the Ken Burns Pan
        else:
            strategy_type = "action" # Fallback to motion/saliency

        print(f"   🔍 Diagnostic: Energy={energy:.2f}, Detections={len(face_points)}, Spread={spread:.2f}")
        
        
        # ── Step 3: Call Updated Filter Builder ─────────────────────
        # This matches the new 3-argument signature defined above
        main_filt, crop_x_end = build_portrait_filter(tier_info, main_duration, strategy_type)

        tmp_main = out_dir / f"_tmp_main_{s['id']:02d}.mp4"
        tmp_zoom = out_dir / f"_tmp_zoom_{s['id']:02d}.mp4"
        concat_txt = out_dir / f"_tmp_concat_{s['id']:02d}.txt"

        try:
            # ── Clip A: Main behavior (Pan or Anchor) ───────────────
            _ffmpeg_encode(video, scene_start, main_duration, main_filt, tmp_main)

            if zoom_dur > 0:
                # ── Clip B: Zoom-out (Starts from final pan/anchor x) ─
                render_zoomout_opencv(video, scene_start, scene_end, zoom_dur, tier_info, int(crop_x_end), tmp_zoom)

                # Standard h264 re-encode for concat compatibility
                tmp_zoom_enc = out_dir / f"_tmp_zoom_enc_{s['id']:02d}.mp4"
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(tmp_zoom), "-ss", f"{scene_start + main_duration:.6f}",
                    "-t", f"{zoom_dur:.6f}", "-i", str(video), "-map", "0:v", "-map", "1:a?",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
                    "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR, str(tmp_zoom_enc)
                ], check=True, capture_output=True)

                with open(concat_txt, "w") as cf:
                    cf.write(f"file '{tmp_main.resolve().as_posix()}'\n")
                    cf.write(f"file '{tmp_zoom_enc.resolve().as_posix()}'\n")

                subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(out)], check=True)
            else:
                tmp_main.replace(out)
        finally:
            # Cleanup intermediate files
            for f in [tmp_main, tmp_zoom, out_dir / f"_tmp_zoom_enc_{s['id']:02d}.mp4", concat_txt]:
                if f and Path(f).exists(): Path(f).unlink()


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

def snap_scenes_to_words(scenes: list, words: list,
                          tolerance: float = 1.0) -> list:
    """
    Adjust scene boundaries so they fall on word ends rather than mid-sentence.

    WHY:
      Scene detection is purely visual — it finds frame-difference spikes.
      Those cut points have no relationship to speech rhythm, so a boundary
      can land mid-word or mid-sentence.  When the clip is split there, the
      audio is hard-cut at that exact timestamp → audible truncation.

    HOW (Option B):
      For each *internal* scene boundary (end of scene N = start of scene N+1):
        1. Collect all word ends that fall within [scene_end - tolerance,
           scene_end + tolerance].
        2. Pick the word end closest to the original boundary.
        3. Snap scene[N].end  and scene[N+1].start to that word end.

      The first scene start and last scene end are never moved — those are
      the hard edges of the source video.

    TOLERANCE:
      1.0 second default.  Increase if sentences are long and boundaries are
      far from natural pauses.  Decrease if you want stricter alignment.

    Returns a new list of scenes (originals are not mutated).
    """
    if not words or len(scenes) < 2:
        return scenes

    # Build a flat list of word-end timestamps for fast lookup
    word_ends = sorted(w["end"] for w in words)

    snapped = [dict(s) for s in scenes]   # shallow copies

    for i in range(len(snapped) - 1):
        original_boundary = snapped[i]["end"]

        lo = original_boundary - tolerance
        hi = original_boundary + tolerance

        # Word ends that fall inside the search window
        candidates = [t for t in word_ends if lo <= t <= hi]

        if not candidates:
            print(f"   ⚠️  Scene {snapped[i]['id']}→{snapped[i+1]['id']}: "
                  f"no word end within ±{tolerance}s of {original_boundary:.2f}s — boundary kept")
            continue

        best = min(candidates, key=lambda t: abs(t - original_boundary))
        delta = best - original_boundary

        snapped[i]["end"]        = best
        snapped[i + 1]["start"]  = best

        sign = "+" if delta >= 0 else ""
        print(f"   ✂️  Scene {snapped[i]['id']}→{snapped[i+1]['id']}: "
              f"boundary {original_boundary:.3f}s → {best:.3f}s  ({sign}{delta:.3f}s)")

    return snapped

# =========================
# ASS FILE WRITING
# =========================

# =========================
# SUBTITLE PRESETS
# =========================

from config import SUBTITLE_PRESET

def get_subtitle_style():
    presets = {

        "classic": {
            "font_name": "Inter SemiBold",
            "size": 72,
            "color": "#FFFFFF",
            "outline": 2,
            "shadow": 1,
            "uppercase": False
        },

        "loud_clear": {
            "font_name": "Montserrat ExtraBold",
            "size": 92,
            "color": "#FFFFFF",
            "outline": 4,
            "shadow": 0,
            "uppercase": True
        },

        "hype_mode": {
            "font_name": "Anton",
            "size": 105,
            "color": "#FF0000",
            "outline": 6,
            "shadow": 0,
            "uppercase": True
        },

        "vlog_pop": {
            "font_name": "Poppins Bold",
            "size": 82,
            "color": "#00CFFF",
            "outline": 3,
            "shadow": 1,
            "uppercase": False
        },

        "talk_show": {
            "font_name": "Roboto Bold",
            "size": 78,
            "color": "#FFFFFF",
            "outline": 0,
            "shadow": 3,
            "uppercase": False
        }
    }

    return presets.get(SUBTITLE_PRESET, presets["classic"])

def _ass_header(f):
    style = get_subtitle_style()

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
        f"2,{MARGIN_H},{MARGIN_H},{MARGIN_V},1\n\n"

        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

def render_karaoke_chunk(f, chunk):
    """
    Karaoke word-by-word highlight renderer.

    Emits one dialogue event per word. Each event:
      - spans word[i].start → word[i+1].start (last word: end + extend)
      - shows ALL words in the chunk
      - active word:   \1c<active_color> + optional pill background (\3c + \bord)
      - inactive words: \1c<preset_base_color>  (reset after active word)

    Active colour  = highlight.text_color  in editor.config.json  (e.g. #FFD700)
    Inactive colour= preset style["color"]                         (e.g. #FFFFFF)
    Pill background= highlight.background_color + opacity + padding when HL_ENABLED
    """
    style    = get_subtitle_style()
    inact_c  = ass_color(style["color"])
    act_c    = ass_color(ACTIVE_COLOR)
    bg_alpha = opacity_to_alpha(BG_OPACITY)
    bg_c     = ass_color(BG_COLOR, alpha=bg_alpha)

    # Build ASS override tags using string concatenation — NOT f-strings.
    # f-strings cause backslash escaping confusion:
    #   \1  in f-string source = Python octal SOH (0x01)   — wrong
    #   \\1 in f-string source = double backslash + 1      — also wrong
    # Concatenation: "\\1c" in source = one backslash + "1c" in memory, unambiguously.
    if HL_ENABLED:
        open_tag  = "{" + "\\1c" + act_c + "\\3c" + bg_c + "\\bord" + str(BG_BORD) + "\\blur0}"
    else:
        open_tag  = "{" + "\\1c" + act_c + "}"
    close_tag = "{" + "\\1c" + inact_c + "\\bord2\\blur0}"

    for i, word in enumerate(chunk):
        ev_start = word["s"]
        ev_end   = chunk[i + 1]["s"] if i + 1 < len(chunk) else word["e"] + EXTEND_LAST_WORD_SEC

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


def write_ass(path, scene_words):
    """
    Write an ASS subtitle file for a scene using karaoke word-by-word highlight.

    subtitle_preset  → controls font, size, outline, uppercase, base (inactive) colour
    highlight.text_color → active word colour (overrides preset per word via \1c tag)
    highlight.background_color / opacity → pill behind active word when enabled
    """
    with open(path, "w", encoding="utf8") as f:
        _ass_header(f)

        if not scene_words:
            return

        for chunk in chunk_words(scene_words):
            render_karaoke_chunk(f, chunk)

# =========================
# LOGO + BURN
# =========================

def _build_logo_filter(ass_escaped: str, logo_path: str) -> tuple:
    """
    Build the ffmpeg -filter_complex and input args needed to composite
    a logo + subtitles onto a video, driven by the config "logo" block.

    logo_path is passed in from args.logo (the CLI argument) — it is NOT
    read from config. Config only controls how the logo is displayed, not
    which file is used. This keeps a clean separation:
      CLI  → what files to process  (video, logo)
      Config → how to render them   (size, position, opacity)

    Config keys (editor.config.json → "logo"):
      enabled      — bool, set false to skip logo without changing your command
      corner       — top-left | top-right | bottom-left | bottom-right
      width_pct    — logo width as fraction of output frame (e.g. 0.12)
      margin_x     — horizontal margin in pixels from the chosen edge
      margin_y     — vertical margin in pixels from the chosen edge
      opacity      — 0.0 (invisible) → 1.0 (fully opaque)

    Returns (extra_inputs, filter_complex, map_arg) where:
      extra_inputs   — list of additional ffmpeg args to add the logo input
      filter_complex — the complete -filter_complex string, or None if
                       logo is disabled (use plain -vf subtitles= instead)
      map_arg        — output pad name to -map, or None

    Filter graph (when logo enabled):
      [0:v] → subtitles → [subbed]
      [1:v] → scale to logo_w × -1 → [logo_scaled]
      [subbed][logo_scaled] → overlay at (ox, oy) with opacity → [out]
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

    # Overlay position — ffmpeg overlay filter uses (x, y) of top-left corner
    # of the logo. W/H refer to main video dimensions, w/h to overlay.
    if "right" in corner:
        ox = f"W-w-{margin_x}"
    else:
        ox = str(margin_x)

    if "bottom" in corner:
        oy = f"H-h-{margin_y}"
    else:
        oy = str(margin_y)

    logo_path_esc = str(Path(logo_path).resolve()).replace("\\", "/").replace(":", "\\:")

    filter_complex = (
        f"[0:v]subtitles='{ass_escaped}':fontsdir='{FONTS_DIR}'[subbed];"
        f"[1:v]scale={logo_w}:-1[logo_scaled];"
        f"[subbed][logo_scaled]overlay={ox}:{oy}:format=auto,"
        f"colorchannelmixer=aa={opacity:.3f}[out]"
    )

    extra_inputs = ["-i", logo_path]
    return extra_inputs, filter_complex, "[out]"


def burn(src, ass, out, logo_path: str = "", force: bool = False):
    """
    Burn subtitles (and optional logo) onto a scene clip.

    logo_path comes from args.logo (CLI argument). Config controls
    how the logo is displayed — size, position, opacity, enabled toggle.

    If logo is enabled in config, uses a filter_complex that:
      1. Renders ASS subtitles onto the video
      2. Scales the logo to the configured width
      3. Overlays it at the configured corner with the configured opacity

    If logo is disabled or logo_path is empty, uses plain -vf subtitles=.
    """
    if out.exists() and not force:
        return   # caller handles skip counting

    ass_escaped = str(ass.resolve()).replace("\\", "/").replace(":", "\\:")

    extra_inputs, filter_complex, map_out = _build_logo_filter(ass_escaped, logo_path)

    if filter_complex:
        # Logo + subtitles via filter_complex
        run([
            "ffmpeg", "-y",
            "-i", str(src),
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", map_out,
            "-map", "0:a?",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
            "-c:a", "aac", "-b:a", ENCODE_AUDIO_BR,
            str(out),
        ])
    else:
        # Subtitles only
        run([
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", f"subtitles='{ass_escaped}':fontsdir='{FONTS_DIR}'",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", ENCODE_PRESET, "-crf", str(ENCODE_CRF_SCENES),
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
Each step is independently controllable. Default behaviour (no flags) runs
only what is missing — nothing is re-run if its output already exists.

Steps and their controlling flags:
  Step 0  pre-crop         runs if _precrop.mp4 missing; delete it to re-crop
  Step 1  scene detect     runs if scenes.json missing
  Step 2  transcribe       --regen-words   (re-runs Whisper, overwrites words.json)
  Step 3  snap boundaries  runs after Step 2 if words available
  Step 4  split scenes     --regen-splits  (re-splits even if scene files exist)
  Step 5  gen subtitles    --regen-subs    (re-writes ASS from words.json)
  Step 6  burn             --force-burn    (re-burns even if final files exist)

Common workflows:
  First run (everything):
    python process.py video.mp4 logo.png --regen-words --regen-subs --regen-splits --force-burn

  Fix wrong transcription (edit words.json then):
    python process.py video.mp4 logo.png --regen-subs --force-burn

  Change subtitle formatting only (edit editor.config.json then):
    python process.py video.mp4 logo.png --regen-subs --force-burn

  Change logo / burn settings only:
    python process.py video.mp4 logo.png --force-burn
        """,
    )
    ap.add_argument("video",  help="Path to the source video file")
    ap.add_argument("logo",   help="Path to logo image (PNG). Placement controlled via config.")
    ap.add_argument("--regen-words",  action="store_true",
                    help="Re-run Whisper and overwrite words.json")
    ap.add_argument("--regen-subs",   action="store_true",
                    help="Re-write ASS subtitle files from words.json (no re-transcribe)")
    ap.add_argument("--regen-splits", action="store_true",
                    help="Re-split and re-encode all scene files")
    ap.add_argument("--force-burn",   action="store_true",
                    help="Re-burn final files even if they already exist")
    args = ap.parse_args()

    name       = Path(args.video).stem
    base       = Path("output") / name
    scenes_dir = base / "scenes"
    subs_dir   = base / "subtitles"
    final_dir  = base / "final"
    meta       = base / "scenes.json"
    precropped = base / f"{name}_precrop.mp4"
    words_cache = base / "words.json"

    for d in (base, scenes_dir, subs_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬  Processing: {args.video}")
    print(f"    output   → {base}")
    print(f"    logo     → {args.logo}")

    # ── Step 0: Pre-crop ─────────────────────────────────────────────
    precropped, orientation = precrop_video(args.video, precropped)
    print(f"   orientation: {orientation}")

    # ── Step 1: Scene detection ──────────────────────────────────────
    if not meta.exists():
        src_for_detection = str(precropped) if precropped.exists() else args.video
        fps, scenes = detect_scenes(src_for_detection)
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            src_for_detection
        ]))
        json.dump(
            {"video": str(precropped), "fps": fps, "duration": duration,
             "orientation": orientation, "scenes": scenes},
            open(meta, "w", encoding="utf8"),
            indent=2,
        )
        print(f"   💾 Detected {len(scenes)} scenes → {meta.name}")
    else:
        print(f"   📖 Loaded scenes from cache ({meta.name})")

    data        = json.load(open(meta, encoding="utf8"))
    scenes      = data["scenes"]
    orientation = data.get("orientation", orientation)

    # ── Step 2: Transcribe ───────────────────────────────────────────
    if args.regen_words or not words_cache.exists():
        print("🧠  Transcribing with Whisper…")
        model = whisper.load_model("base")
        src_for_whisper = str(precropped) if precropped.exists() else args.video
        res   = model.transcribe(src_for_whisper, word_timestamps=True)
        del model
        gc.collect()

        words = [w for seg in res["segments"] for w in seg["words"]]
        json.dump(words, open(words_cache, "w", encoding="utf8"), indent=2)
        print(f"   💾 {len(words)} words → {words_cache.name}")

    else:
        words = json.load(open(words_cache, encoding="utf8"))
        print(f"   📖 Loaded {len(words)} words from cache ({words_cache.name})")

    # ── Step 3: Snap scene boundaries to word ends ───────────────────
    # Runs automatically whenever words are available and splits haven't
    # been locked yet. Re-runs after --regen-words to use new timestamps.
    if words and (args.regen_words or args.regen_splits or not meta.exists()):
        print("✂️  Snapping scene boundaries to word ends…")
        snap_tolerance = SNAP_TOLERANCE_SEC
        scenes = snap_scenes_to_words(scenes, words, tolerance=snap_tolerance)
        data["scenes"] = scenes
        json.dump(data, open(meta, "w", encoding="utf8"), indent=2)

    # ── Step 4: Split scenes ─────────────────────────────────────────
    # --regen-splits → delete existing scene files and re-split
    if args.regen_splits:
        print("🗑️  Clearing existing scene files for re-split…")
        for f in scenes_dir.glob("scene_*.mp4"):
            f.unlink()

    split_video(str(precropped), scenes, scenes_dir, orientation=orientation)

    # ── Step 5: Generate subtitle ASS files ──────────────────────────
    # --regen-subs → force-regenerate all ASS from words.json
    # default      → generate only missing ASS files
    if words:
        subs_needed = []
        for s in scenes:
            ass_path = subs_dir / f"scene_{s['id']:02d}.ass"
            if args.regen_subs or not ass_path.exists():
                subs_needed.append(s)

        if subs_needed:
            print(f"📝  Writing subtitles ({len(subs_needed)} scenes)…")
            for s in subs_needed:
                sw = words_for_scene(words, s["start"], s["end"])
                write_ass(subs_dir / f"scene_{s['id']:02d}.ass", sw)
        else:
            print("📝  All subtitle files present — skipping (use --regen-subs to regenerate)")
    else:
        print("   ⚠️  No words available — subtitles skipped")
        print("        Run with --regen-words to transcribe first")

    # ── Step 6: Burn subtitles + logo onto final clips ───────────────
    print("🎞️   Burning…")
    burned, skipped, missing = 0, 0, 0
    for s in tqdm(scenes, desc="🔥 Burning"):
        ass_path   = subs_dir   / f"scene_{s['id']:02d}.ass"
        scene_path = scenes_dir / f"scene_{s['id']:02d}.mp4"
        final_path = final_dir  / f"scene_{s['id']:02d}.mp4"

        if not scene_path.exists():
            print(f"   ⚠️  scene_{s['id']:02d}: scene MP4 missing — skipped")
            missing += 1
            continue
        if not ass_path.exists():
            print(f"   ⚠️  scene_{s['id']:02d}: ASS file missing — skipped")
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

