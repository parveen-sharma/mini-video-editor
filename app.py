import streamlit as st
import json
import subprocess
import os
import sys
import io
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# ==============================================================================
# 🛠️ SYSTEM CORE: ENCODING & RE-DIRECTION
# ==============================================================================

# Force UTF-8 encoding for the terminal/stdout to prevent Unicode crashes on Windows.
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==============================================================================
# 🧠 PERSISTENT STATE MANAGEMENT
# ==============================================================================

if "v_path" not in st.session_state: 
    st.session_state.v_path = ""
if "l_path" not in st.session_state: 
    st.session_state.l_path = ""
if "f_path" not in st.session_state: 
    st.session_state.f_path = ""

# ==============================================================================
# 📁 OS INTERFACE: NATIVE PICKERS
# ==============================================================================

def select_file(file_types):
    """Triggers the Windows File Explorer to pick a specific file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=file_types)
    root.destroy()
    return file_path

def select_folder():
    """Triggers the Windows Folder Browser to select a directory."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def open_directory(path):
    """Opens a folder in the Windows File Explorer."""
    path = os.path.realpath(path)
    if os.path.exists(path):
        os.startfile(path)

# ==============================================================================
# ⚙️ CONFIGURATION DATA ENGINE
# ==============================================================================

CONFIG_FILE = "editor.config.json"

def load_config():
    """Reads settings from JSON. Creates empty structure if file is missing."""
    if not os.path.exists(CONFIG_FILE):
        return {
            "highlight": {},
            "processing": {"motion_analysis": {}, "face_detection": {}, "zoom_out": {}},
            "precrop": {}
        }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return {}

def save_config(config_data):
    """Writes all UI modifications back to the disk."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        st.toast("Settings saved to disk!")
    except Exception as e:
        st.error(f"Failed to save config: {e}")

# Load the working configuration
config = load_config()

# ==============================================================================
# 📱 SIDEBAR: LIVE VISUAL CONTROL PANEL
# ==============================================================================

st.set_page_config(page_title="AI Video Reframer Pro", layout="wide", page_icon="🎥")

with st.sidebar:
    st.title("⚙️ Global Settings")
    
    # --- LIVE SUBTITLE PREVIEW ---
    st.subheader("📱 Live Subtitle Preview")
    
    # Extract values for the CSS-based virtual phone
    p_bg = config.get("highlight", {}).get("background_color", "#0A1A2F")
    p_text = config.get("highlight", {}).get("text_color", "#FFD700")
    p_align = config.get("alignment", "bottom_center")
    p_v_margin = config.get("margin_vertical_px", 200)
    
    # Calculate CSS Flexbox positioning based on alignment
    if "top" in p_align: f_pos = "flex-start"
    elif "center" in p_align: f_pos = "center"
    else: f_pos = "flex-end"
    
    # Scale the vertical margin for the small preview window (1/6th scale)
    scaled_v = p_v_margin / 6 

    # FIX: Corrected parameter name to unsafe_allow_html
    st.markdown(f"""
    <div style="border: 3px solid #444; border-radius: 20px; width: 160px; height: 284px; 
                background-color: #000; margin: 0 auto; display: flex; flex-direction: column;
                justify-content: {f_pos}; align-items: center; position: relative; overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
        <div style="padding: {scaled_v}px 10px; text-align: center; width: 100%; font-family: 'Segoe UI', Tahoma, sans-serif;">
            <div style="background-color: {p_bg}; color: {p_text}; 
                        display: inline-block; padding: 2px 6px; border-radius: 4px; 
                        font-weight: 800; font-size: 12px; letter-spacing: 0.5px;">
                ACTIVE
            </div>
            <div style="color: white; font-size: 12px; margin-top: 2px;">WORD</div>
        </div>
        <div style="position: absolute; top: 10px; width: 40px; height: 5px; background: #333; border-radius: 10px;"></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("<center>Virtual 9:16 Frame Preview</center>", unsafe_allow_html=True)

    st.divider()

    # --- CATEGORY: SUBTITLE AESTHETICS ---
    st.header("🔡 Subtitle Aesthetics")
    preset_options = ["classic", "loud_clear", "hype_mode", "vlog_pop", "talk_show"]
    config["subtitle_preset"] = st.selectbox(
        "Style Preset", preset_options, 
        index=preset_options.index(config.get("subtitle_preset", "classic"))
    )
    
    if "highlight" not in config: config["highlight"] = {}
    c_col1, c_col2 = st.columns(2)
    config["highlight"]["text_color"] = c_col1.color_picker("Text", config["highlight"].get("text_color", "#FFD700"))
    config["highlight"]["background_color"] = c_col2.color_picker("Box", config["highlight"].get("background_color", "#0A1A2F"))
    config["highlight"]["background_opacity"] = st.slider("Box Opacity", 0.0, 1.0, float(config["highlight"].get("background_opacity", 0.9)))
    
    # --- CATEGORY: POSITION & ALIGNMENT ---
    st.subheader("📍 Position & Alignment")
    alignment_mapping = {"Bottom": "bottom_center", "Middle": "center", "Top": "top_center"}
    current_align_key = config.get("alignment", "bottom_center")
    
    try:
        align_idx = list(alignment_mapping.values()).index(current_align_key)
    except:
        align_idx = 0
        
    align_ui_choice = st.selectbox("Alignment", list(alignment_mapping.keys()), index=align_idx)
    config["alignment"] = alignment_mapping[align_ui_choice]

    config["margin_vertical_px"] = st.slider("Vertical Margin", 50, 800, int(config.get("margin_vertical_px", 200)))
    config["margin_horizontal_px"] = st.slider("Horizontal Margin", 20, 400, int(config.get("margin_horizontal_px", 120)))

    # --- CATEGORY: LAYOUT & TIMING ---
    st.subheader("📏 Layout & Timing")
    lay_col1, lay_col2 = st.columns(2)
    config["max_words_per_line"] = lay_col1.number_input("Words/Line", 1, 10, int(config.get("max_words_per_line", 4)))
    config["max_lines"] = lay_col2.number_input("Max Lines", 1, 5, int(config.get("max_lines", 2)))
    
    config["pause_threshold_ms"] = st.number_input("Pause Gap (ms)", 100, 2000, int(config.get("pause_threshold_ms", 400)))
    config["extend_last_word_ms"] = st.number_input("Outro Hold (ms)", 0, 1000, int(config.get("extend_last_word_ms", 200)))

    st.divider()
    
    # --- CATEGORY: AI & PROCESSING ---
    st.header("🧠 Content AI")
    if "processing" not in config: 
        config["processing"] = {"motion_analysis": {}, "face_detection": {}, "zoom_out": {}}
    
    proc_ma = config["processing"].get("motion_analysis", {})
    config["processing"]["motion_analysis"]["motion_threshold"] = st.slider(
        "Motion Sensitivity", 0.0, 50.0, float(proc_ma.get("motion_threshold", 15.0))
    )
    
    proc_fd = config["processing"].get("face_detection", {})
    config["processing"]["face_detection"]["enabled"] = st.checkbox(
        "Enable Face Tracking", bool(proc_fd.get("enabled", True))
    )
    
    proc_zo = config["processing"].get("zoom_out", {})
    config["processing"]["zoom_out"]["enabled"] = st.checkbox(
        "Enable Zoom-Out Outro", bool(proc_zo.get("enabled", True))
    )

    st.divider()

    # --- CATEGORY: PRE-CROP SETTINGS ---
    st.header("✂️ Framing & Pre-Crop")
    if "precrop" not in config: config["precrop"] = {}

    config["precrop"]["horizontal_keep_pct"] = st.slider(
        "Horizontal Keep %", 0.1, 1.0, float(config["precrop"].get("horizontal_keep_pct", 0.88))
    )
    config["precrop"]["vertical_keep_pct"] = st.slider(
        "Vertical Keep %", 0.1, 1.0, float(config["precrop"].get("vertical_keep_pct", 0.84))
    )

    anc_col1, anc_col2 = st.columns(2)
    h_anchors, v_anchors = ["left", "center", "right"], ["top", "middle", "bottom"]
    
    config["precrop"]["horizontal_anchor"] = anc_col1.selectbox(
        "H-Anchor", h_anchors, 
        index=h_anchors.index(config["precrop"].get("horizontal_anchor", "center"))
    )
    config["precrop"]["vertical_anchor"] = anc_col2.selectbox(
        "V-Anchor", v_anchors, 
        index=v_anchors.index(config["precrop"].get("vertical_anchor", "middle"))
    )

    st.divider()
    
    if st.button("💾 SAVE ALL CONFIGURATIONS", use_container_width=True, type="primary"):
        save_config(config)
        st.success("Configuration Synced!")

# ==============================================================================
# 🚀 MAIN WORKSPACE
# ==============================================================================

st.title("🎥 AI Video Reframer Dashboard")

tab_single, tab_batch = st.tabs(["🎯 Single Processing", "🗂️ Batch Processing"])

with tab_single:
    row1_c1, row1_c2 = st.columns(2)
    
    # Video Selection
    v_in_col, v_pk_col = row1_c1.columns([0.85, 0.15])
    v_display = v_in_col.text_input("Source Video File", value=st.session_state.v_path)
    if v_pk_col.button("📁", key="single_v"):
        res = select_file([("Video Files", "*.mp4 *.mov *.mkv *.avi")])
        if res:
            st.session_state.v_path = res
            st.rerun()

    # Logo Selection
    l_in_col, l_pk_col = row1_c2.columns([0.85, 0.15])
    l_display = l_in_col.text_input("Overlay Logo (PNG)", value=st.session_state.l_path)
    if l_pk_col.button("🖼️", key="single_l"):
        res = select_file([("Image Files", "*.png *.jpg *.jpeg")])
        if res:
            st.session_state.l_path = res
            st.rerun()

    st.subheader("🛠️ Run Mode")
    f_c1, f_c2, f_c3, f_c4 = st.columns(4)
    r_words = f_c1.checkbox("Regen Words")
    r_splits = f_c2.checkbox("Regen Splits")
    r_subs = f_c3.checkbox("Regen Subs")
    f_burn = f_c4.checkbox("Force Burn")

    if st.button("🚀 INITIATE PROCESSING", use_container_width=True):
        if not st.session_state.v_path or not st.session_state.l_path:
            st.error("Please select files.")
        else:
            flags = []
            if r_words: flags.append("--regen-words")
            if r_splits: flags.append("--regen-splits")
            if r_subs: flags.append("--regen-subs")
            if f_burn: flags.append("--force-burn")
            
            cmd = ["python", "process.py", st.session_state.v_path, st.session_state.l_path] + flags
            
            with st.status("🎬 Processing...", expanded=True) as status:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, bufsize=1, encoding="utf-8", errors="replace"
                )
                log_box = st.empty()
                log_acc = ""
                for line in iter(proc.stdout.readline, ""):
                    log_acc += line
                    log_box.code("\n".join(log_acc.splitlines()[-15:]))
                proc.wait()
                if proc.returncode == 0:
                    status.update(label="✅ Success!", state="complete")
                    st.balloons()
                else: status.update(label="❌ Failed", state="error")

with tab_batch:
    st.subheader("🗂️ Batch Processing")
    b_in_col, b_pk_col = st.columns([0.85, 0.15])
    b_display = b_in_col.text_input("Source Directory", value=st.session_state.f_path)
    if b_pk_col.button("📂", key="batch_dir"):
        res = select_folder()
        if res:
            st.session_state.f_path = res
            st.rerun()

    if st.button("📦 EXECUTE BATCH", use_container_width=True):
        if not st.session_state.f_path or not st.session_state.l_path:
            st.error("Select folder/logo first.")
        else:
            cmd = ["python", "batch_process.py", st.session_state.f_path, st.session_state.l_path]
            with st.status("📦 Batching...", expanded=True) as status:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, encoding="utf-8", errors="replace"
                )
                log_box = st.empty()
                log_acc = ""
                for line in iter(proc.stdout.readline, ""):
                    log_acc += line
                    log_box.code("\n".join(log_acc.splitlines()[-15:]))
                proc.wait()
                status.update(label="✅ Finished!", state="complete")
