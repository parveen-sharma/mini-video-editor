"""
app.py — AI Video Reframer · Streamlit UI

Architecture:
  · st.set_page_config is the very first Streamlit call (required by Streamlit)
  · Config is loaded from disk into st.session_state ONCE — not on every rerun
  · All widget mutations go through st.session_state["config"]
  · Unsaved changes are tracked via st.session_state["config_dirty"]
  · Sidebar: live 9:16 preview + Save/Reset buttons + dirty indicator only
  · Main area: tabbed layout covering every config section + Process + Output
"""

import streamlit as st

# ── MUST be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Reframer Pro",
    layout="wide",
    page_icon="🎥",
    initial_sidebar_state="expanded",
)

import json
import os
import io
import sys
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# =========================
# UTF-8 CONSOLE (Windows)
# =========================
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# =========================
# CONFIG I/O
# =========================
CONFIG_FILE = "editor.config.json"

DEFAULTS = {
    "enabled": True,
    "mode": "karaoke",
    "subtitle_preset": "loud_clear",
    "max_words_per_line": 4,
    "max_lines": 2,
    "alignment": "top_center",
    "margin_vertical_px": 150,
    "margin_horizontal_px": 120,
    "line_spacing_px": 12,
    "word_spacing_px": 8,
    "pause_threshold_ms": 400,
    "extend_last_word_ms": 200,
    "min_word_duration_ms": 120,
    "max_word_duration_ms": 900,
    "enable_fade": True,
    "fade_in_ms": 120,
    "fade_out_ms": 120,
    "highlight": {
        "enabled": True,
        "text_color": "#FFD700",
        "background_color": "#0A1A2F",
        "background_opacity": 0.9,
        "padding_x": 14,
        "padding_y": 6,
        "corner_radius_px": 8,
        "transition_ms": 80,
    },
    "highlight_pop": {"enabled": True, "scale": 1.05, "duration_ms": 90},
    "logo": {
        "enabled": True,
        "corner": "bottom-right",
        "width_pct": 0.12,
        "margin_x": 30,
        "margin_y": 30,
        "opacity": 0.85,
    },
    "processing": {
        "scene_detection": {"sample_fps": 6.0, "diff_threshold": 22.0, "min_duration_sec": 1.0},
        "motion_analysis": {"motion_threshold": 15.0, "sample_fps": 20.0, "pixel_diff_threshold": 100, "saliency_center_deadband": 0.0},
        "background_blur": {"ffmpeg_luma_radius": 40, "ffmpeg_luma_power": 3, "opencv_kernel_size": 81},
        "zoom_out": {"enabled": True, "max_pct_of_scene": 0.1},
        "encode": {"crf_precrop": 16, "crf_scenes": 18, "audio_bitrate": "192k", "preset": "fast"},
        "face_detection": {"enabled": True},
    },
    "precrop": {
        "horizontal_keep_pct": 0.95,
        "vertical_keep_pct": 0.9,
        "horizontal_anchor": "center",
        "vertical_anchor": "top",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, preserving nested structure from base for missing keys."""
    result = base.copy()
    for k, v in override.items():
        if k.startswith("//"):
            continue  # skip comment keys
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return _deep_merge({}, DEFAULTS)
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        return _deep_merge(DEFAULTS, on_disk)
    except Exception as e:
        st.error(f"⚠️ Failed to load {CONFIG_FILE}: {e}. Using defaults.")
        return _deep_merge({}, DEFAULTS)


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        st.session_state["config_dirty"] = False
        st.toast("✅ Configuration saved to disk.", icon="💾")
    except Exception as e:
        st.error(f"Failed to save config: {e}")


# =========================
# SESSION STATE — init once
# =========================
if "config" not in st.session_state:
    st.session_state["config"]       = load_config()
    st.session_state["config_dirty"] = False
    st.session_state["v_path"]       = ""
    st.session_state["l_path"]       = ""
    st.session_state["f_path"]       = ""
    st.session_state["last_stdout"]  = ""
    st.session_state["last_stderr"]  = ""
    st.session_state["last_rc"]      = None
    st.session_state["active_proc"]  = None

cfg = st.session_state["config"]


def mark_dirty():
    st.session_state["config_dirty"] = True


# =========================
# OS NATIVE FILE PICKERS
# =========================
def pick_file(file_types):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(filetypes=file_types)
    root.destroy()
    return path or ""


def pick_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory()
    root.destroy()
    return path or ""


# =========================
# HELPERS
# =========================
def _cfg_get(key, default=None):
    """Flat top-level key lookup with fallback."""
    return cfg.get(key, default)


def _nested(section: str, key: str, default=None):
    return cfg.get(section, {}).get(key, default)


def _safe_index(lst, value, fallback=0):
    try:
        return lst.index(value)
    except ValueError:
        return fallback


# =========================
# SIDEBAR — preview + save
# =========================
with st.sidebar:
    # Dirty indicator
    if st.session_state["config_dirty"]:
        st.warning("⚠️ Unsaved changes", icon="🟡")
    else:
        st.success("Config saved", icon="✅")

    col_save, col_reset = st.columns(2)
    if col_save.button("💾 Save", use_container_width=True, type="primary"):
        save_config(cfg)
        st.rerun()
    if col_reset.button("↺ Defaults", use_container_width=True):
        st.session_state["config"] = _deep_merge({}, DEFAULTS)
        st.session_state["config_dirty"] = True
        st.rerun()

    st.divider()

    # ── Live 9:16 Phone Preview ──────────────────────────────────────
    st.subheader("📱 Live Preview")

    p_bg      = cfg.get("highlight", {}).get("background_color", "#0A1A2F")
    p_text    = cfg.get("highlight", {}).get("text_color", "#FFD700")
    p_align   = cfg.get("alignment", "bottom_center")
    p_v_marg  = int(cfg.get("margin_vertical_px", 150))
    p_preset  = cfg.get("subtitle_preset", "classic").replace("_", " ").title()

    if "top"    in p_align: flex_pos = "flex-start"
    elif "center" in p_align and "top" not in p_align and "bottom" not in p_align:
        flex_pos = "center"
    else:
        flex_pos = "flex-end"

    scaled_v = max(4, p_v_marg // 6)

    st.markdown(
        f"""
        <div style="border:2px solid #555;border-radius:18px;width:140px;height:248px;
                    background:#111;margin:0 auto;display:flex;flex-direction:column;
                    justify-content:{flex_pos};align-items:center;overflow:hidden;
                    position:relative;">
          <div style="position:absolute;top:8px;left:50%;transform:translateX(-50%);
                      width:36px;height:4px;background:#333;border-radius:4px;"></div>
          <div style="padding:{scaled_v}px 8px;text-align:center;width:100%;">
            <div style="font-family:sans-serif;color:#aaa;font-size:9px;margin-bottom:4px;">
              {p_preset}
            </div>
            <span style="background:{p_bg};color:{p_text};display:inline-block;
                         padding:2px 7px;border-radius:4px;font-weight:800;
                         font-size:11px;letter-spacing:0.4px;">ACTIVE</span>
            <div style="color:#ddd;font-size:10px;margin-top:3px;">word word word</div>
          </div>
        </div>
        <p style="text-align:center;font-size:10px;color:#888;margin-top:6px;">9:16 frame</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Quick stats ──────────────────────────────────────────────────
    st.caption(
        f"Words/line: **{cfg.get('max_words_per_line', 4)}** · "
        f"Lines: **{cfg.get('max_lines', 2)}** · "
        f"Align: **{cfg.get('alignment', '—')}**"
    )
    enc = cfg.get("processing", {}).get("encode", {})
    st.caption(
        f"CRF: **{enc.get('crf_scenes', 18)}** · "
        f"Preset: **{enc.get('preset', 'fast')}** · "
        f"Audio: **{enc.get('audio_bitrate', '192k')}**"
    )


# =========================
# MAIN AREA — tabbed layout
# =========================
st.title("🎥 AI Video Reframer Pro")

(
    tab_process,
    tab_subtitles,
    tab_framing,
    tab_ai,
    tab_encode,
    tab_logo,
    tab_output,
) = st.tabs([
    "▶ Process",
    "🔡 Subtitles",
    "✂️ Framing",
    "🧠 AI & Scenes",
    "⚙️ Encode",
    "🖼️ Logo",
    "📂 Output",
])


# ── TAB: Process ─────────────────────────────────────────────────────────────
with tab_process:
    st.subheader("File Selection")
    col_v, col_l = st.columns(2)

    with col_v:
        v_inp, v_btn = st.columns([0.85, 0.15])
        typed_v = v_inp.text_input(
            "Source Video",
            value=st.session_state["v_path"],
            placeholder="/path/to/video.mp4",
        )
        if typed_v != st.session_state["v_path"]:
            st.session_state["v_path"] = typed_v
        if v_btn.button("📁", key="pick_v", help="Browse for video file"):
            res = pick_file([("Video", "*.mp4 *.mov *.mkv *.avi")])
            if res:
                st.session_state["v_path"] = res
                st.rerun()
        if st.session_state["v_path"]:
            p = Path(st.session_state["v_path"])
            if p.exists():
                size_mb = p.stat().st_size / 1_048_576
                st.caption(f"✅ {p.name}  ·  {size_mb:.1f} MB")
            else:
                st.caption("⚠️ File not found")

    with col_l:
        l_inp, l_btn = st.columns([0.85, 0.15])
        typed_l = l_inp.text_input(
            "Overlay Logo (PNG)",
            value=st.session_state["l_path"],
            placeholder="/path/to/logo.png",
        )
        if typed_l != st.session_state["l_path"]:
            st.session_state["l_path"] = typed_l
        if l_btn.button("🖼️", key="pick_l", help="Browse for logo image"):
            res = pick_file([("Image", "*.png *.jpg *.jpeg")])
            if res:
                st.session_state["l_path"] = res
                st.rerun()
        if st.session_state["l_path"]:
            p = Path(st.session_state["l_path"])
            st.caption("✅ " + p.name if p.exists() else "⚠️ File not found")

    st.divider()

    # ── Single vs Batch ──────────────────────────────────────────────
    proc_mode = st.radio("Mode", ["Single file", "Batch folder"], horizontal=True)

    if proc_mode == "Batch folder":
        b_inp, b_btn = st.columns([0.85, 0.15])
        typed_f = b_inp.text_input(
            "Source Folder",
            value=st.session_state["f_path"],
            placeholder="/path/to/videos/",
        )
        if typed_f != st.session_state["f_path"]:
            st.session_state["f_path"] = typed_f
        if b_btn.button("📂", key="pick_f", help="Browse for folder"):
            res = pick_folder()
            if res:
                st.session_state["f_path"] = res
                st.rerun()
        if st.session_state["f_path"]:
            p = Path(st.session_state["f_path"])
            if p.is_dir():
                vids = list(p.glob("*.mp4")) + list(p.glob("*.mov")) + list(p.glob("*.mkv"))
                st.caption(f"✅ {p.name}  ·  {len(vids)} video(s) found")
            else:
                st.caption("⚠️ Folder not found")

    st.divider()

    # ── Run Mode flags ───────────────────────────────────────────────
    st.subheader("Run Mode")
    st.caption("Each flag forces re-running one pipeline step even if cached output exists.")

    fc1, fc2, fc3, fc4 = st.columns(4)
    r_words  = fc1.checkbox("Regen Words",  help="Re-run Whisper transcription. Use when audio quality was poor or language changed.")
    r_splits = fc2.checkbox("Regen Splits", help="Re-split all scenes. Required after changing framing or scene detection settings.")
    r_subs   = fc3.checkbox("Regen Subs",   help="Re-write subtitle ASS files. Use after changing any subtitle style settings.")
    f_burn   = fc4.checkbox("Force Burn",   help="Re-burn all final clips. Use after changing logo or subtitle appearance.")

    # ── Run Button ───────────────────────────────────────────────────
    st.markdown("")

    if st.session_state["config_dirty"]:
        st.info("💡 You have unsaved config changes. Save before processing to use the latest settings.", icon="ℹ️")

    can_run = bool(st.session_state["v_path"] or (proc_mode == "Batch folder" and st.session_state["f_path"]))

    run_label = "🚀 Start Processing" if proc_mode == "Single file" else "📦 Start Batch"
    if st.button(run_label, use_container_width=True, type="primary", disabled=not can_run):
        if proc_mode == "Single file" and not st.session_state["v_path"]:
            st.error("Select a source video first.")
        elif proc_mode == "Batch folder" and not st.session_state["f_path"]:
            st.error("Select a source folder first.")
        elif not st.session_state["l_path"]:
            st.error("Select a logo image first.")
        else:
            flags = []
            if r_words:  flags.append("--regen-words")
            if r_splits: flags.append("--regen-splits")
            if r_subs:   flags.append("--regen-subs")
            if f_burn:   flags.append("--force-burn")

            if proc_mode == "Single file":
                cmd = [sys.executable, "process.py", st.session_state["v_path"], st.session_state["l_path"]] + flags
            else:
                cmd = [sys.executable, "batch_process.py", st.session_state["f_path"], st.session_state["l_path"]] + flags

            # Use lists as mutable containers so the stderr thread can append
            # without needing `nonlocal` (which requires an enclosing function scope,
            # not an if-block scope).
            stdout_buf = []
            stderr_buf = []

            with st.status("🎬 Pipeline running…", expanded=True) as status:
                log_box  = st.empty()
                err_area = st.empty()

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
                st.session_state["active_proc"] = proc

                # Collect stderr in a background thread while stdout streams live
                def _collect_stderr(buf, pipe):
                    for line in iter(pipe.readline, ""):
                        buf.append(line)

                t = threading.Thread(
                    target=_collect_stderr,
                    args=(stderr_buf, proc.stderr),
                    daemon=True,
                )
                t.start()

                for line in iter(proc.stdout.readline, ""):
                    stdout_buf.append(line)
                    log_box.code(
                        "".join(stdout_buf[-20:]), language=""
                    )

                proc.wait()
                t.join(timeout=2)

                stdout_acc = "".join(stdout_buf)
                stderr_acc = "".join(stderr_buf)

                st.session_state["active_proc"] = None
                st.session_state["last_stdout"] = stdout_acc
                st.session_state["last_stderr"] = stderr_acc
                st.session_state["last_rc"]     = proc.returncode

                if proc.returncode == 0:
                    status.update(label="✅ Done!", state="complete")
                    st.balloons()
                else:
                    status.update(label="❌ Pipeline failed", state="error")
                    if stderr_acc.strip():
                        err_area.error(f"**stderr output:**\n```\n{stderr_acc[-2000:]}\n```")

    # Full log expander (always shown after a run)
    if st.session_state["last_stdout"]:
        with st.expander("📋 Full log from last run", expanded=False):
            rc = st.session_state["last_rc"]
            st.caption(f"Exit code: **{rc}** — {'✅ success' if rc == 0 else '❌ failed'}")
            st.code(st.session_state["last_stdout"], language="")
            if st.session_state["last_stderr"].strip():
                st.subheader("stderr")
                st.code(st.session_state["last_stderr"], language="")


# ── TAB: Subtitles ───────────────────────────────────────────────────────────
with tab_subtitles:

    st.subheader("Style Preset")
    presets = ["classic", "loud_clear", "hype_mode", "vlog_pop", "talk_show"]
    preset_desc = {
        "classic":    "Clean white text, light shadow — good for talking heads",
        "loud_clear": "Bold uppercase, heavy outline — social media default",
        "hype_mode":  "Large red Anton font — high-energy content",
        "vlog_pop":   "Cyan Poppins, subtle outline — lifestyle/vlog feel",
        "talk_show":  "White Roboto, strong shadow — interview style",
    }
    new_preset = st.selectbox(
        "Preset",
        presets,
        index=_safe_index(presets, cfg.get("subtitle_preset", "loud_clear")),
        format_func=lambda x: f"{x.replace('_', ' ').title()} — {preset_desc[x]}",
        on_change=mark_dirty,
    )
    cfg["subtitle_preset"] = new_preset

    st.divider()
    st.subheader("Highlight Colours")

    hl = cfg.setdefault("highlight", {})

    hc1, hc2, hc3 = st.columns(3)
    hl["enabled"] = hc1.checkbox(
        "Enable word highlight",
        value=bool(hl.get("enabled", True)),
        on_change=mark_dirty,
        help="Draws a coloured pill around the currently-spoken word.",
    )
    hl["text_color"] = hc2.color_picker(
        "Active word colour",
        value=hl.get("text_color", "#FFD700"),
        on_change=mark_dirty,
    )
    hl["background_color"] = hc3.color_picker(
        "Pill background",
        value=hl.get("background_color", "#0A1A2F"),
        on_change=mark_dirty,
    )
    hl["background_opacity"] = st.slider(
        "Pill opacity",
        0.0, 1.0,
        float(hl.get("background_opacity", 0.9)),
        step=0.05,
        on_change=mark_dirty,
        help="0 = invisible pill, 1 = fully opaque.",
    )

    pad_c1, pad_c2, rad_c = st.columns(3)
    hl["padding_x"] = pad_c1.number_input(
        "Pill padding X (px)", 0, 40, int(hl.get("padding_x", 14)), on_change=mark_dirty
    )
    hl["padding_y"] = pad_c2.number_input(
        "Pill padding Y (px)", 0, 30, int(hl.get("padding_y", 6)), on_change=mark_dirty
    )
    hl["corner_radius_px"] = rad_c.number_input(
        "Corner radius (px)", 0, 30, int(hl.get("corner_radius_px", 8)), on_change=mark_dirty
    )

    st.divider()
    st.subheader("Position & Alignment")

    align_map = {"Top": "top_center", "Middle": "center", "Bottom": "bottom_center"}
    align_ui  = st.radio(
        "Vertical alignment",
        list(align_map.keys()),
        index=_safe_index(list(align_map.values()), cfg.get("alignment", "top_center")),
        horizontal=True,
        on_change=mark_dirty,
    )
    cfg["alignment"] = align_map[align_ui]

    mc1, mc2 = st.columns(2)
    cfg["margin_vertical_px"] = mc1.slider(
        "Vertical margin (px)", 50, 800,
        int(cfg.get("margin_vertical_px", 150)),
        on_change=mark_dirty,
        help="Distance from the top or bottom edge depending on alignment.",
    )
    cfg["margin_horizontal_px"] = mc2.slider(
        "Horizontal margin (px)", 20, 400,
        int(cfg.get("margin_horizontal_px", 120)),
        on_change=mark_dirty,
        help="Left/right safe zone — keeps text away from the frame edge.",
    )

    st.divider()
    st.subheader("Layout & Word Timing")

    lc1, lc2 = st.columns(2)
    cfg["max_words_per_line"] = lc1.number_input(
        "Words per line", 1, 10, int(cfg.get("max_words_per_line", 4)),
        on_change=mark_dirty,
    )
    cfg["max_lines"] = lc2.number_input(
        "Max lines visible", 1, 5, int(cfg.get("max_lines", 2)),
        on_change=mark_dirty,
    )

    tc1, tc2, tc3 = st.columns(3)
    cfg["pause_threshold_ms"] = tc1.number_input(
        "Pause gap (ms)", 100, 2000, int(cfg.get("pause_threshold_ms", 400)),
        on_change=mark_dirty,
        help="Silence longer than this starts a new subtitle chunk.",
    )
    cfg["extend_last_word_ms"] = tc2.number_input(
        "Last word hold (ms)", 0, 1000, int(cfg.get("extend_last_word_ms", 200)),
        on_change=mark_dirty,
        help="Extra display time after the final word in a chunk.",
    )
    cfg["min_word_duration_ms"] = tc3.number_input(
        "Min word duration (ms)", 40, 500, int(cfg.get("min_word_duration_ms", 120)),
        on_change=mark_dirty,
        help="Floor to prevent fast words from flickering.",
    )

    cfg["max_word_duration_ms"] = st.slider(
        "Max word duration (ms)", 200, 2000,
        int(cfg.get("max_word_duration_ms", 900)),
        on_change=mark_dirty,
        help="Ceiling for very slow speech — prevents subtitle from hanging.",
    )

    st.divider()
    st.subheader("Animation")

    fa_c1, fa_c2, fa_c3 = st.columns(3)
    cfg["enable_fade"] = fa_c1.checkbox(
        "Fade in/out", bool(cfg.get("enable_fade", True)), on_change=mark_dirty
    )
    cfg["fade_in_ms"]  = fa_c2.number_input(
        "Fade-in (ms)", 0, 500, int(cfg.get("fade_in_ms", 120)),
        disabled=not cfg["enable_fade"], on_change=mark_dirty,
    )
    cfg["fade_out_ms"] = fa_c3.number_input(
        "Fade-out (ms)", 0, 500, int(cfg.get("fade_out_ms", 120)),
        disabled=not cfg["enable_fade"], on_change=mark_dirty,
    )

    pop = cfg.setdefault("highlight_pop", {})
    pc1, pc2, pc3 = st.columns(3)
    pop["enabled"] = pc1.checkbox(
        "Word pop animation", bool(pop.get("enabled", True)),
        on_change=mark_dirty,
        help="Subtle scale-up on the active word to draw the eye.",
    )
    pop["scale"] = pc2.slider(
        "Pop scale", 1.0, 1.3,
        float(pop.get("scale", 1.05)), step=0.01,
        disabled=not pop["enabled"], on_change=mark_dirty,
    )
    pop["duration_ms"] = pc3.number_input(
        "Pop duration (ms)", 20, 300, int(pop.get("duration_ms", 90)),
        disabled=not pop["enabled"], on_change=mark_dirty,
    )


# ── TAB: Framing ─────────────────────────────────────────────────────────────
with tab_framing:
    st.subheader("Pre-Crop")
    st.caption(
        "Trims the source video borders before any portrait conversion. "
        "Useful for removing letterboxing, watermarks, or screen-recording chrome."
    )

    pc = cfg.setdefault("precrop", {})
    pc_c1, pc_c2 = st.columns(2)
    pc["horizontal_keep_pct"] = pc_c1.slider(
        "Keep width %", 0.1, 1.0,
        float(pc.get("horizontal_keep_pct", 0.95)), step=0.01,
        on_change=mark_dirty,
        help="Fraction of frame width to retain. 1.0 = no horizontal trim.",
    )
    pc["vertical_keep_pct"] = pc_c2.slider(
        "Keep height %", 0.1, 1.0,
        float(pc.get("vertical_keep_pct", 0.9)), step=0.01,
        on_change=mark_dirty,
        help="Fraction of frame height to retain. 1.0 = no vertical trim.",
    )

    h_anchors = ["left", "center", "right"]
    v_anchors = ["top", "middle", "bottom"]
    anc_c1, anc_c2 = st.columns(2)
    pc["horizontal_anchor"] = anc_c1.selectbox(
        "Horizontal anchor",
        h_anchors,
        index=_safe_index(h_anchors, pc.get("horizontal_anchor", "center")),
        on_change=mark_dirty,
        help="Which edge to keep when trimming horizontally.",
    )
    pc["vertical_anchor"] = anc_c2.selectbox(
        "Vertical anchor",
        v_anchors,
        index=_safe_index(v_anchors, pc.get("vertical_anchor", "top")),
        on_change=mark_dirty,
        help="Which edge to keep when trimming vertically.",
    )

    kept_w = pc["horizontal_keep_pct"] * 1920
    kept_h = pc["vertical_keep_pct"] * 1080
    st.caption(
        f"Effective crop on a 1920×1080 source: **{kept_w:.0f} × {kept_h:.0f} px** "
        f"(anchor: {pc['horizontal_anchor']} / {pc['vertical_anchor']})"
    )

    st.divider()
    st.subheader("Zoom-Out Outro")
    st.caption(
        "Appends a short expanding-window animation at the end of each landscape scene, "
        "revealing the full slide or background. Portrait sources skip this automatically."
    )

    zo = cfg.setdefault("processing", {}).setdefault("zoom_out", {})
    zo_c1, zo_c2 = st.columns(2)
    zo["enabled"] = zo_c1.checkbox(
        "Enable zoom-out outro", bool(zo.get("enabled", True)), on_change=mark_dirty
    )
    zo["max_pct_of_scene"] = zo_c2.slider(
        "Max % of scene duration", 0.0, 0.5,
        float(zo.get("max_pct_of_scene", 0.1)), step=0.01,
        disabled=not zo["enabled"], on_change=mark_dirty,
        help="Caps zoom duration so short scenes don't become mostly zoom.",
    )


# ── TAB: AI & Scenes ─────────────────────────────────────────────────────────
with tab_ai:
    processing = cfg.setdefault("processing", {})

    st.subheader("Scene Detection")
    st.caption(
        "Controls how the pipeline finds visual cut points. "
        "Changes here require **--regen-splits** to take effect."
    )

    sd = processing.setdefault("scene_detection", {})
    sd_c1, sd_c2, sd_c3 = st.columns(3)
    sd["sample_fps"] = sd_c1.number_input(
        "Sample rate (fps)", 1.0, 30.0,
        float(sd.get("sample_fps", 6.0)), step=0.5,
        on_change=mark_dirty,
        help="How many times per second frames are compared. Framerate-agnostic.",
    )
    sd["diff_threshold"] = sd_c2.slider(
        "Cut sensitivity", 5.0, 80.0,
        float(sd.get("diff_threshold", 22.0)), step=0.5,
        on_change=mark_dirty,
        help="Mean pixel diff to trigger a scene cut. Lower = more sensitive.",
    )
    sd["min_duration_sec"] = sd_c3.number_input(
        "Min scene duration (s)", 0.2, 10.0,
        float(sd.get("min_duration_sec", 1.0)), step=0.1,
        on_change=mark_dirty,
        help="Scenes shorter than this are discarded — filters flash cuts.",
    )

    st.divider()
    st.subheader("Motion & Crop Analysis")
    st.caption("Determines whether a scene is 'slide-like' or has real motion.")

    ma = processing.setdefault("motion_analysis", {})
    ma_c1, ma_c2, ma_c3 = st.columns(3)
    ma["motion_threshold"] = ma_c1.slider(
        "Motion threshold", 0.0, 50.0,
        float(ma.get("motion_threshold", 15.0)), step=0.5,
        on_change=mark_dirty,
        help="Energy above this → scene treated as motion (not a slide). Pan disabled.",
    )
    ma["sample_fps"] = ma_c2.number_input(
        "Analysis sample rate (fps)", 1.0, 30.0,
        float(ma.get("sample_fps", 20.0)), step=1.0,
        on_change=mark_dirty,
    )
    ma["pixel_diff_threshold"] = ma_c3.number_input(
        "Pixel diff threshold", 5, 255,
        int(ma.get("pixel_diff_threshold", 100)),
        on_change=mark_dirty,
        help="Per-pixel diff to count a pixel as 'moving' during motion analysis.",
    )
    ma["saliency_center_deadband"] = st.slider(
        "Saliency center deadband", 0.0, 0.4,
        float(ma.get("saliency_center_deadband", 0.0)), step=0.01,
        on_change=mark_dirty,
        help="If saliency returns a crop center within this fraction of dead-center, "
             "it falls back to center crop. Prevents tiny off-center artifacts from dominating.",
    )

    st.divider()
    st.subheader("Face Detection")

    fd = processing.setdefault("face_detection", {})
    fd["enabled"] = st.checkbox(
        "Enable face tracking",
        bool(fd.get("enabled", True)),
        on_change=mark_dirty,
        help="When enabled, the crop window is anchored to the largest detected face. "
             "Disable for screen recordings or non-human content.",
    )

    st.divider()
    st.subheader("Background Blur")
    st.caption("Blur applied to the letterboxed background fill on landscape sources.")

    bl = processing.setdefault("background_blur", {})
    bl_c1, bl_c2, bl_c3 = st.columns(3)
    bl["ffmpeg_luma_radius"] = bl_c1.number_input(
        "ffmpeg luma radius", 1, 100, int(bl.get("ffmpeg_luma_radius", 40)),
        on_change=mark_dirty,
    )
    bl["ffmpeg_luma_power"] = bl_c2.number_input(
        "ffmpeg luma power", 1, 10, int(bl.get("ffmpeg_luma_power", 3)),
        on_change=mark_dirty,
    )
    raw_k = int(bl.get("opencv_kernel_size", 81))
    # Ensure odd
    raw_k = raw_k if raw_k % 2 == 1 else raw_k + 1
    new_k = bl_c3.number_input(
        "OpenCV kernel size (odd)", 3, 201, raw_k, step=2,
        on_change=mark_dirty,
        help="Must be odd. Larger = heavier blur on the background panel.",
    )
    bl["opencv_kernel_size"] = new_k if new_k % 2 == 1 else new_k + 1


# ── TAB: Encode ──────────────────────────────────────────────────────────────
with tab_encode:
    st.subheader("Video & Audio Quality")
    st.caption(
        "These settings affect file size and quality for all encoded outputs. "
        "Changes only take effect on the next run — existing files are not re-encoded "
        "unless you pass **--regen-splits** or **--force-burn**."
    )

    enc = cfg.setdefault("processing", {}).setdefault("encode", {})

    ec1, ec2 = st.columns(2)
    enc["crf_precrop"] = ec1.slider(
        "CRF — pre-crop intermediate", 0, 51,
        int(enc.get("crf_precrop", 16)),
        on_change=mark_dirty,
        help="Lower = higher quality, larger file. 16–18 recommended for intermediate.",
    )
    enc["crf_scenes"] = ec2.slider(
        "CRF — scenes & final clips", 0, 51,
        int(enc.get("crf_scenes", 18)),
        on_change=mark_dirty,
        help="Quality for split scenes and burned final outputs. 18–22 is typical.",
    )

    crf_note = enc["crf_scenes"]
    if crf_note <= 17:
        st.info("🔵 Very high quality — large files. Good for archival.", icon="ℹ️")
    elif crf_note <= 23:
        st.success("🟢 Good balance of quality and file size.", icon="✅")
    else:
        st.warning("🟡 Lower quality — files will be smaller but visible compression may appear.", icon="⚠️")

    ffmpeg_presets = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
    enc["preset"] = st.select_slider(
        "ffmpeg encode preset",
        ffmpeg_presets,
        value=enc.get("preset", "fast"),
        on_change=mark_dirty,
        help="Slower presets produce smaller files at the same CRF, but take longer to encode.",
    )

    audio_options = ["96k", "128k", "160k", "192k", "256k", "320k"]
    enc["audio_bitrate"] = st.select_slider(
        "Audio bitrate (AAC)",
        audio_options,
        value=enc.get("audio_bitrate", "192k"),
        on_change=mark_dirty,
    )

    st.divider()
    st.subheader("Estimated impact")
    preset_idx   = ffmpeg_presets.index(enc["preset"])
    speed_pct    = int((preset_idx / (len(ffmpeg_presets) - 1)) * 100)
    quality_note = "smaller files" if preset_idx > 4 else ("larger files" if preset_idx < 3 else "balanced")

    col_spd, col_qual, col_aud = st.columns(3)
    col_spd.metric("Encode speed", f"{'Faster' if speed_pct < 50 else 'Slower'}", f"{quality_note}")
    col_qual.metric("Visual CRF (final)", enc["crf_scenes"])
    col_aud.metric("Audio bitrate", enc["audio_bitrate"])


# ── TAB: Logo ────────────────────────────────────────────────────────────────
with tab_logo:
    st.subheader("Logo Overlay")
    st.caption(
        "Logo path is set in the **Process** tab (CLI argument). "
        "Settings here control how it is composited onto each clip."
    )

    logo = cfg.setdefault("logo", {})

    logo["enabled"] = st.checkbox(
        "Show logo on clips",
        bool(logo.get("enabled", True)),
        on_change=mark_dirty,
        help="Disable to produce clips without any logo watermark.",
    )

    if logo["enabled"]:
        corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
        logo["corner"] = st.radio(
            "Placement corner",
            corners,
            index=_safe_index(corners, logo.get("corner", "bottom-right")),
            horizontal=True,
            on_change=mark_dirty,
        )

        lc1, lc2 = st.columns(2)
        logo["width_pct"] = lc1.slider(
            "Width (% of frame)", 0.03, 0.40,
            float(logo.get("width_pct", 0.12)), step=0.01,
            on_change=mark_dirty,
            help="Fraction of the 1080px output width. 0.12 ≈ 130px wide.",
            format="%.2f",
        )
        logo["opacity"] = lc2.slider(
            "Opacity", 0.0, 1.0,
            float(logo.get("opacity", 0.85)), step=0.05,
            on_change=mark_dirty,
        )

        mx_c, my_c = st.columns(2)
        logo["margin_x"] = mx_c.number_input(
            "Margin X (px)", 0, 200, int(logo.get("margin_x", 30)), on_change=mark_dirty,
            help="Horizontal distance from the chosen edge.",
        )
        logo["margin_y"] = my_c.number_input(
            "Margin Y (px)", 0, 200, int(logo.get("margin_y", 30)), on_change=mark_dirty,
            help="Vertical distance from the chosen edge.",
        )

        logo_w_px = int(logo["width_pct"] * 1080)
        st.caption(
            f"Logo will be **{logo_w_px}px** wide, placed in the **{logo['corner']}** "
            f"corner at {int(logo['opacity']*100)}% opacity, "
            f"{logo['margin_x']}px from side, {logo['margin_y']}px from edge."
        )

        # Mini visual placement diagram
        corners_css = {
            "top-left":     ("8px", "auto", "auto", "8px"),
            "top-right":    ("8px", "8px",  "auto", "auto"),
            "bottom-left":  ("auto","auto", "8px",  "8px"),
            "bottom-right": ("auto","8px",  "8px",  "auto"),
        }
        t, r, b, l = corners_css.get(logo["corner"], ("auto","8px","8px","auto"))
        st.markdown(
            f"""
            <div style="width:120px;height:80px;border:1.5px solid #555;border-radius:8px;
                        background:#111;position:relative;margin:8px 0;">
              <div style="position:absolute;top:{t};right:{r};bottom:{b};left:{l};
                          width:18px;height:12px;background:#aaa;border-radius:2px;
                          opacity:{logo['opacity']};"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Logo overlay is disabled. All output clips will have no watermark.", icon="ℹ️")


# ── TAB: Output Browser ──────────────────────────────────────────────────────
with tab_output:
    st.subheader("Output Files")

    output_root = Path("output")
    if not output_root.exists():
        st.info("No output folder found yet. Process a video first.", icon="📂")
    else:
        video_jobs = sorted([d for d in output_root.iterdir() if d.is_dir()])

        if not video_jobs:
            st.info("Output folder is empty.", icon="📂")
        else:
            selected_job = st.selectbox(
                "Job",
                video_jobs,
                format_func=lambda p: p.name,
            )

            if selected_job:
                final_dir = selected_job / "final"
                scenes_dir = selected_job / "scenes"

                col_stats1, col_stats2, col_stats3 = st.columns(3)
                final_clips   = list(final_dir.glob("*.mp4"))  if final_dir.exists()  else []
                scene_clips   = list(scenes_dir.glob("*.mp4")) if scenes_dir.exists() else []
                total_mb      = sum(f.stat().st_size for f in final_clips) / 1_048_576

                col_stats1.metric("Final clips",   len(final_clips))
                col_stats2.metric("Scene clips",   len(scene_clips))
                col_stats3.metric("Total size",    f"{total_mb:.1f} MB")

                if final_clips:
                    st.markdown("**Final clips:**")
                    for clip in sorted(final_clips):
                        size_mb = clip.stat().st_size / 1_048_576
                        mj = selected_job / "final" / (clip.stem + ".json")
                        hook = ""
                        if mj.exists():
                            try:
                                with open(mj, encoding="utf-8") as jf:
                                    meta = json.load(jf)
                                hook = meta.get("hook", "")
                            except Exception:
                                pass
                        label = f"🎬 {clip.name}  ·  {size_mb:.1f} MB"
                        if hook:
                            label += f"  ·  *{hook}*"
                        st.markdown(label)

                # scenes.json summary
                meta_file = selected_job / "scenes.json"
                if meta_file.exists():
                    with st.expander("scenes.json", expanded=False):
                        with open(meta_file, encoding="utf-8") as jf:
                            meta = json.load(jf)
                        st.write(f"FPS: {meta.get('fps')}  |  Duration: {meta.get('duration', 0):.1f}s  |  Orientation: {meta.get('orientation')}")
                        st.write(f"Scenes: {len(meta.get('scenes', []))}")
                        st.json(meta.get("scenes", []), expanded=False)
