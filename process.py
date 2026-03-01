import argparse, json, subprocess, gc, re
from pathlib import Path
from tqdm import tqdm
import whisper, cv2, numpy as np

from config import (
    CFG, LOGO_CFG,
    FONT, FONT_SIZE,
    MAX_WORDS_PER_LINE, MAX_LINES,
    MARGIN_H, MARGIN_V,
    ACTIVE_COLOR, INACTIVE_COLOR,
    BG_COLOR, BG_OPACITY, PAD_X, PAD_Y,
    HL_ENABLED, EXTEND_LAST_WORD_SEC, PAUSE_THRESHOLD_SEC,
    ALIGNMENT, BG_BORD,
    OUT_W, OUT_H,
    PC_W, PC_H, PC_H_ANCHOR, PC_V_ANCHOR,
    ZOOM_OUT_DURATION, SNAP_TOLERANCE_SEC,
)

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

# =========================
# SCENE DETECTION
# =========================

def detect_scenes(video):
    cap   = cv2.VideoCapture(video)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev  = None
    idx   = 0
    cuts  = [0]

    pbar = tqdm(total=total, desc="🔍 Detecting scenes")

    while True:
        ok, frm = cap.read()
        if not ok:
            break

        if idx % 6 == 0:
            g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diff = np.mean(cv2.absdiff(g, prev))
                if diff > 22:
                    cuts.append(idx)
            prev = g

        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    cuts.append(total)

    scenes = []
    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        if b - a > fps:
            scenes.append({"id": i, "start": a / fps, "end": b / fps})

    return fps, scenes

# =========================
# SCENE ANALYSIS — motion centroid + saliency fallback
# =========================

# Minimum mean pixel difference across a scene to be considered "animated".
# Below this the scene is treated as static (title card, frozen slide, etc.)
# and saliency is used instead of motion.
MOTION_THRESHOLD = 8.0

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

        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Optional: dilate slightly to merge nearby changed regions into blobs
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


def analyze_scene(video_path: str, scene_start: float, scene_end: float,
                  sample_fps: float = 2.0) -> dict:
    """
    Sample frames from [scene_start, scene_end], apply the 95% pre-crop, then
    determine the best portrait crop center using a 3-tier strategy:

      Tier 1 — Motion centroid
        Diffs consecutive samples to find where animation is happening.
        Best signal for animated explainer content where drawn elements
        move into frame, grow, highlight, etc.
        Used when mean motion energy > MOTION_THRESHOLD.

      Tier 2 — Spectral saliency
        Finds the most visually prominent region of the middle frame.
        Used when the scene is mostly static (title card, frozen slide).

      Tier 3 — Geometric center fallback
        Pure center crop. Used when both above signals are flat or fail.

    Returns a dict:
      src_w, src_h  — original video dimensions (pre-crop)
      cx_frac       — crop center X as fraction of pre-cropped width  (0-1)
      cy_frac       — crop center Y as fraction of pre-cropped height (0-1)
      strategy      — "motion" | "saliency" | "center"
      motion_energy — mean pixel diff energy (diagnostic)
    """
    # NOTE: video_path is the PRE-CROPPED intermediate — no manual crop needed.
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(1, int(fps / sample_fps))
    start_frame    = int(scene_start * fps)
    end_frame      = int(scene_end   * fps)

    gray_frames   = []
    middle_region = None

    mid_frame = start_frame + (end_frame - start_frame) // 2

    for fi in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break

        # Frame is already pre-cropped — read it directly
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)

        if middle_region is None or abs(fi - mid_frame) < abs(fi - mid_frame - frame_interval):
            middle_region = frame.copy()

    cap.release()

    if not gray_frames:
        return {"src_w": src_w, "src_h": src_h,
                "cx_frac": 0.5, "cy_frac": 0.5,
                "strategy": "center", "motion_energy": 0.0}

    # ── Tier 1: motion ───────────────────────────────────────────────
    if len(gray_frames) >= 2:
        cx, cy, energy = _motion_center(gray_frames)
        if energy >= MOTION_THRESHOLD:
            return {"src_w": src_w, "src_h": src_h,
                    "cx_frac": cx, "cy_frac": cy,
                    "strategy": "motion", "motion_energy": energy}
    else:
        energy = 0.0

    # ── Tier 2: saliency ─────────────────────────────────────────────
    if middle_region is not None:
        cx, cy = _saliency_center(middle_region)
        # Only trust saliency if it produced a non-trivial result
        # (i.e. not just snapping to dead center, which means it found nothing)
        if not (0.48 < cx < 0.52 and 0.48 < cy < 0.52):
            return {"src_w": src_w, "src_h": src_h,
                    "cx_frac": cx, "cy_frac": cy,
                    "strategy": "saliency", "motion_energy": energy}

    # ── Tier 3: center fallback ──────────────────────────────────────
    return {"src_w": src_w, "src_h": src_h,
            "cx_frac": 0.5, "cy_frac": 0.5,
            "strategy": "center", "motion_energy": energy}


def build_portrait_filter(info: dict, scene_duration: float = 0.0):
    """
    Build the ffmpeg -filter_complex chain for the MAIN pan/motion portion of
    a scene.

    Returns: (filter_str, crop_x_end)
      filter_str  — the complete filter_complex string
      crop_x_end  — the horizontal crop-left pixel position at end of clip,
                    used by build_zoomout_filter to start the zoom from the
                    correct position so there is no visual jump.
    """
    src_w    = info["src_w"]
    src_h    = info["src_h"]
    cx_frac  = info["cx_frac"]
    strategy = info["strategy"]
    energy   = info.get("motion_energy", 0.0)

    port_h = src_h
    port_w = int(src_h * 9 / 16)

    bg = (
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},"
        f"boxblur=luma_radius=40:luma_power=3"
    )

    if strategy == "motion":
        desired_cx = int(cx_frac * src_w)
        crop_x     = max(0, min(src_w - port_w, desired_cx - port_w // 2))
        fg         = f"crop={port_w}:{port_h}:{crop_x}:0,scale={OUT_W}:{OUT_H}"
        crop_x_end = crop_x
        print(f"   📐 strategy=motion   energy={energy:.1f}  "
              f"cx={cx_frac:.2f}  window=[{crop_x}:{crop_x+port_w}]/{src_w}px")
    else:
        pan_range = max(0, src_w - port_w)
        if pan_range > 0 and scene_duration > 0:
            x_expr = f"{pan_range:.4f}*t/{scene_duration:.4f}"
            fg     = f"crop={port_w}:{port_h}:{x_expr}:0,scale={OUT_W}:{OUT_H}"
            crop_x_end = pan_range   # pan ends at the right edge
            print(f"   📐 strategy={strategy} (pan)  "
                  f"energy={energy:.1f}  range=0→{pan_range}px  dur={scene_duration:.1f}s")
        else:
            crop_x     = max(0, (src_w - port_w) // 2)
            fg         = f"crop={port_w}:{port_h}:{crop_x}:0,scale={OUT_W}:{OUT_H}"
            crop_x_end = crop_x
            print(f"   📐 strategy={strategy} (center-fixed)  "
                  f"energy={energy:.1f}  crop_x={crop_x}")

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
    bg = cv2.GaussianBlur(bg, (81, 81), 0)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def _precrop_offsets(src_w: int, src_h: int) -> tuple:
    """
    Compute concrete pixel values for the pre-crop step.

    Takes the actual source dimensions (probed via ffprobe) and the
    config values PC_W, PC_H, PC_H_ANCHOR, PC_V_ANCHOR, returning
    integer pixel values ready to embed in an ffmpeg crop filter.

    Config keys (subtitle.config.json → "precrop"):
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


def precrop_video(src: str, out_path: Path) -> Path:
    """
    Step 0: probe the source video dimensions, compute concrete pixel
    crop values from config, and save a clean intermediate file used
    by all downstream steps.

    Config (subtitle.config.json → "precrop"):
      horizontal_keep_pct  — fraction of WIDTH  to keep  (e.g. 0.95)
      vertical_keep_pct    — fraction of HEIGHT to keep  (e.g. 0.85)
      horizontal_anchor    — left | center | right
      vertical_anchor      — top  | middle | bottom

    Anchor grid:
      top-left      top-center      top-right
      middle-left   middle-center   middle-right
      bottom-left   bottom-center   bottom-right

    Dimensions are probed live from the source video via ffprobe so
    the crop values are always exact integers, not runtime expressions.

    Cached: delete the _precrop.mp4 to force re-crop with new settings.
    """
    if out_path.exists():
        print(f"⏭️  Pre-crop exists, reusing: {out_path.name}")
        print(f"   (delete it to re-crop with updated settings)")
        return out_path

    src_w, src_h = _probe_dimensions(str(src))
    cw, ch, x0, y0 = _precrop_offsets(src_w, src_h)
    vf = f"crop={cw}:{ch}:{x0}:{y0}"

    print(f"✂️  Step 0: pre-cropping")
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
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-c:a", "copy",
        str(out_path),
    ])
    print(f"   ✅ Saved: {out_path}")
    return out_path

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
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ])


def split_video(video, scenes, out_dir):
    """
    For each scene:
      1. Analyse for crop strategy (motion / saliency / center).
      2. Encode MAIN clip  → pan or fixed crop, minus last frame and zoom duration.
      3. Encode ZOOM clip  → zoom-out outro that reveals the full slide.
      4. Concatenate MAIN + ZOOM → final scene file.
      5. Clean up temp files.

    Drop-last-frame:
      We subtract one frame duration (1/fps) from the end of every scene.
      The last frame of a scene is typically a frozen or half-transitioned frame
      that reads as an incomplete cut. Dropping it gives clean endings.

    Zoom-out outro:
      The zoom clip animates crop_w from portrait-width → full-width over
      ZOOM_OUT_DURATION seconds. crop_x simultaneously contracts back to 0.
      Rendered via OpenCV — avoids ffmpeg crop w/h expression limitation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read fps from the pre-cropped video (needed for frame-drop calculation)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    frame_dur = 1.0 / fps

    print("🔎 Analysing scenes for crop…")
    for s in tqdm(scenes, desc="🎬 Splitting"):
        out = out_dir / f"scene_{s['id']:02d}.mp4"
        if out.exists():
            continue

        scene_start    = s["start"]
        scene_end      = s["end"]
        raw_duration   = scene_end - scene_start

        # Effective duration after dropping the last frame
        eff_end        = scene_end - frame_dur
        eff_duration   = eff_end - scene_start

        # Cap zoom duration at 30% of the scene so short clips aren't all outro
        zoom_dur       = min(ZOOM_OUT_DURATION, eff_duration * 0.30)
        main_duration  = eff_duration - zoom_dur

        info           = analyze_scene(video, scene_start, scene_end)
        main_filt, crop_x_end = build_portrait_filter(info, scene_duration=main_duration)

        tmp_main   = out_dir / f"_tmp_main_{s['id']:02d}.mp4"
        tmp_zoom   = out_dir / f"_tmp_zoom_{s['id']:02d}.mp4"
        concat_txt = out_dir / f"_tmp_concat_{s['id']:02d}.txt"

        try:
            # ── Clip A: main pan / motion (ffmpeg) ────────────────────
            _ffmpeg_encode(video, scene_start, main_duration, main_filt, tmp_main)

            # ── Clip B: zoom-out outro (OpenCV — avoids ffmpeg crop
            #    w/h expression limitation) ────────────────────────────
            zoom_start = scene_start + main_duration
            render_zoomout_opencv(
                video, scene_start, scene_end, zoom_dur, info, crop_x_end, tmp_zoom
            )

            # ── Re-encode zoom clip so codec matches main clip ─────────
            # OpenCV writes mp4v — a quick ffmpeg pass normalises to h264.
            #
            # Option A: carry REAL audio for the zoom window instead of
            # silence (-an). Two inputs:
            #   [0] tmp_zoom  — video from OpenCV
            #   [1] source video at zoom_start — audio only
            # This means speech that falls inside the zoom window is audible,
            # so sentences are never silently cut off mid-word.
            tmp_zoom_enc = out_dir / f"_tmp_zoom_enc_{s['id']:02d}.mp4"
            run([
                "ffmpeg", "-y",
                "-i", str(tmp_zoom),            # input 0: opencv video
                "-ss", f"{zoom_start:.6f}",     # input 1: audio from source
                "-t",  f"{zoom_dur:.6f}",
                "-i",  str(video),
                "-map", "0:v",                  # video  ← opencv frames
                "-map", "1:a?",                 # audio  ← source (? = ok if none)
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                str(tmp_zoom_enc),
            ])

            # ── Concat A + B ──────────────────────────────────────────
            with open(concat_txt, "w", encoding="utf8") as cf:
                cf.write(f"file '{tmp_main.resolve().as_posix()}'\n")
                cf.write(f"file '{tmp_zoom_enc.resolve().as_posix()}'\n")

            run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_txt),
                "-c", "copy",
                str(out),
            ])

        finally:
            for f in (tmp_main, tmp_zoom,
                      out_dir / f"_tmp_zoom_enc_{s['id']:02d}.mp4",
                      concat_txt):
                if f.exists():
                    f.unlink()


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

def _ass_header(f, inact_c: str):
    f.write(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "Collisions: Normal\n"
        f"PlayResX: {OUT_W}\n"
        f"PlayResY: {OUT_H}\n\n"

        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"

        f"Style: Base,{FONT},{FONT_SIZE},"
        f"{inact_c},{inact_c},&H00000000&,&H00000000&,"
        f"0,0,0,0,100,100,0,0,"
        f"1,0,0,"
        f"{ALIGNMENT},{MARGIN_H},{MARGIN_H},{MARGIN_V},1\n\n"

        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

def write_ass(path, scene_words):
    """
    Single-layer karaoke rendering.

    WHY we dropped the two-layer approach:
      Layer 0 (plain text) and Layer 1 (highlight overlay) each contain the
      full chunk text. libass lays out every dialogue event independently, so
      when Layer 1 has invisible words interspersed the rendered line width
      differs from Layer 0 — libass then places the two lines at different
      horizontal positions, producing the double-text / offset subtitle bug.

    NEW APPROACH — one event per word, single layer:
      • Each event covers from this word's start → next word's start
        (last word in chunk covers to chunk_end). This means subtitles
        are always visible with no gaps between words.
      • ALL words are rendered in every event, so the line width is always
        identical and libass always positions the line in the same place.
      • The active word gets highlight colour + background box.
      • Inactive words get the inactive colour with no border.
      • Result: one coherent subtitle line, correct position, no duplication.
    """
    inact_c    = ass_color(INACTIVE_COLOR)
    act_c      = ass_color(ACTIVE_COLOR)
    bg_alpha   = opacity_to_alpha(BG_OPACITY)
    bg_c       = ass_color(BG_COLOR, alpha=bg_alpha)
    extend     = EXTEND_LAST_WORD_SEC
    hl_enabled = HL_ENABLED

    with open(path, "w", encoding="utf8") as f:
        _ass_header(f, inact_c)

        for chunk in chunk_words(scene_words):
            if not chunk:
                continue

            chunk_end = chunk[-1]["e"] + extend

            if not hl_enabled:
                # No highlight — single static event for the whole chunk
                base_text = " ".join(w["w"] for w in chunk)
                f.write(
                    f"Dialogue:0,{ts(chunk[0]['s'])},{ts(chunk_end)},"
                    f"Base,,0,0,0,,{base_text}\n"
                )
                continue

            # One event per word — each lasts from this word start → next word start
            for i, active_w in enumerate(chunk):
                word_start = active_w["s"]
                # Hold until next word begins (seamless), or chunk end for last word
                word_end   = chunk[i + 1]["s"] if i + 1 < len(chunk) else chunk_end

                parts = []
                for j, w in enumerate(chunk):
                    if j == i:
                        # Active word: highlight colour + background pill box
                        # \3c  = outline/border colour  → becomes the pill fill
                        # \bord = border thickness      → pill size
                        parts.append(
                            f"{{\1c{act_c}\3c{bg_c}\bord{BG_BORD}\blur0}}{w['w']}"
                        )
                    else:
                        # Inactive word: plain colour, no border
                        parts.append(
                            f"{{\1c{inact_c}\bord0}}{w['w']}"
                        )

                f.write(
                    f"Dialogue:0,{ts(word_start)},{ts(word_end)},"
                    f"Base,,0,0,0,,{' '.join(parts)}\n"
                )

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

    Config keys (subtitle.config.json → "logo"):
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
        f"[0:v]subtitles='{ass_escaped}'[subbed];"
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
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ])
    else:
        # Subtitles only
        run([
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", f"subtitles='{ass_escaped}'",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
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

  Change subtitle formatting only (edit subtitle.config.json then):
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
    precrop_video(args.video, precropped)

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
            {"video": str(precropped), "fps": fps, "duration": duration, "scenes": scenes},
            open(meta, "w", encoding="utf8"),
            indent=2,
        )
        print(f"   💾 Detected {len(scenes)} scenes → {meta.name}")
    else:
        print(f"   📖 Loaded scenes from cache ({meta.name})")

    data   = json.load(open(meta, encoding="utf8"))
    scenes = data["scenes"]

    # ── Step 2: Transcribe ───────────────────────────────────────────
    # --regen-words  → re-run Whisper even if words.json exists
    # default        → load from words.json if present, skip if not
    if args.regen_words:
        print("🧠  Transcribing with Whisper…")
        model = whisper.load_model("base")
        src_for_whisper = str(precropped) if precropped.exists() else args.video
        res   = model.transcribe(src_for_whisper, word_timestamps=True)
        del model
        gc.collect()
        words = [w for seg in res["segments"] for w in seg["words"]]
        json.dump(words, open(words_cache, "w", encoding="utf8"), indent=2)
        print(f"   💾 {len(words)} words → {words_cache.name}")
    elif words_cache.exists():
        words = json.load(open(words_cache, encoding="utf8"))
        print(f"   📖 Loaded {len(words)} words from cache ({words_cache.name})")
    else:
        words = []
        print("   ⚠️  No words.json — run with --regen-words to transcribe")

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

    split_video(str(precropped), scenes, scenes_dir)

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

