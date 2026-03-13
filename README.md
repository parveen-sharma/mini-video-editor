# 🎬 Mini Video Editor
# 🎥 AI Video Reframer Pro

### Automated Horizontal → Vertical Shorts Pipeline with Smart Intelligence

AI Video Reframer Pro is a high-performance suite designed to transform landscape videos into viral 9:16 vertical content. It features a tiered intelligence system (Face -> Motion -> Saliency) and a professional Streamlit dashboard for real-time configuration.

---

## ✨ Key Features

* **🖥️ Pro Dashboard:** A unified UI to manage subtitles, AI thresholds, and framing.
* **📱 Live Subtitle Preview:** Visualize alignment, margins, and colors in a virtual 9:16 frame before rendering.
* **📁 Native OS Pickers:** Integrated Windows File/Folder explorers for a seamless workflow.
* **👤 Multi-Tier Tracking:** 1. **Face Tracking:** Centers the speaker automatically.
    2. **Motion Analysis:** Follows action if no faces are present.
    3. **Saliency Mapping:** Identifies prominent objects (diagrams/slides).
* **🎞️ Smart Pan (Ken Burns):** Automatically detects static slides and applies cinematic pans.
* **🔤 Karaoke Subtitles:** High-impact, word-level highlights with 5 pro presets.
* **⚙️ 7-Stage Pipeline:** Fully incremental—only runs the steps you change.

---

## 🧠 System Architecture

1.  **Step 0: Pre-Crop:** Trims edges to remove static watermarks or black bars.
2.  **Step 1: Scene Detection:** Diff-based analysis to find natural cuts.
3.  **Step 2: Transcription:** AI-powered word-level timestamps via OpenAI Whisper.
4.  **Step 3: Boundary Snapping:** Aligns visual cuts with the end of spoken words to prevent audio glitches.
5.  **Step 4: Portrait Conversion:** Applies cropping, tracking, and optional OpenCV zoom-out.
6.  **Step 5: ASS Generation:** Builds advanced subtitle files based on UI presets.
7.  **Step 6: Burn & Overlay:** Final FFmpeg render combining video, subtitles, and logo.

---

## 🚀 Getting Started

### Installation
1. Install Python 3.10+ and FFmpeg.
2. Clone this repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

### AI-Powered Horizontal → Vertical Shorts Generator

<p align="center">
  <b>Face-aware framing · Karaoke subtitles · Smart slide detection · Logo overlays · Fully configurable</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/ffmpeg-required-green.svg" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" />
  <img src="https://img.shields.io/badge/status-active-success.svg" />
</p>

---

## 🚀 What This Does

Convert long-form videos into **upload-ready vertical shorts** with:

* 🎥 Smart 9:16 portrait conversion
* 👤 Face tracking (with motion + saliency fallback)
* 🎞 Slide detection + Ken Burns pan
* 🔤 Word-level karaoke subtitles
* 🎨 Preset-based subtitle styles
* 🖼 Logo watermark overlay
* 🧠 Whisper transcription (editable cache)
* ♻ Incremental, restartable pipeline

Built for YouTube Shorts, Instagram Reels, TikTok.

---

# 🧠 Architecture Overview

```text
Source Video
   ↓
[0] Pre-crop (border trimming)
   ↓
[1] Scene detection (diff-based, fps aware)
   ↓
[2] Whisper transcription (word timestamps)
   ↓
[3] Scene boundary snapping to word ends
   ↓
[4] Portrait conversion
      Face → Motion → Saliency → Center
      Slides → Ken Burns pan
      Optional zoom-out (OpenCV)
   ↓
[5] ASS subtitle generation (preset-based)
   ↓
[6] Burn subtitles + logo (FFmpeg)
```

Each stage is:

* Cache-aware
* Re-runnable
* Independent
* Deterministic

---

# 📦 Installation

## Requirements

* Python 3.10+
* `ffmpeg` and `ffprobe` installed
* Dependencies (see `requirements.txt` )

```bash
git clone https://github.com/parveen-sharma/mini-video-editor.git
cd mini-video-editor
pip install -r requirements.txt
```

---

# ▶ Quick Start

Full first run:

```bash
python process.py videos/MyVideo.mp4 logo.png \
  --regen-words --regen-subs --regen-splits --force-burn
```

After that, it runs incrementally.

---

# 🎛 CLI Flags

| Flag             | What it does                     |
| ---------------- | -------------------------------- |
| `--regen-words`  | Re-run Whisper transcription     |
| `--regen-subs`   | Regenerate subtitle `.ass` files |
| `--regen-splits` | Re-split portrait scene clips    |
| `--force-burn`   | Re-burn final output videos      |

---

# 🔤 Subtitle Engine

Controlled via `editor.config.json` 

## Presets

| Preset     | Font                 | Style         |
| ---------- | -------------------- | ------------- |
| classic    | Inter SemiBold       | Balanced      |
| loud_clear | Montserrat ExtraBold | High-impact   |
| hype_mode  | Anton                | Bold social   |
| vlog_pop   | Poppins Bold         | Clean creator |
| talk_show  | Roboto Bold          | Broadcast     |

Fonts load from local `/fonts` directory (not system fonts).

Defined in `process.py` 

---

# 🔤 Karaoke Subtitles Example

![Image](https://simplified.co/siteimages/video-editor/subtitle-styles.png)

Active word highlight:

* Custom color
* Optional pill background
* Fade-in/out
* Pop animation
* Pause-aware chunking

---

# 🎥 Smart Framing Logic

Landscape videos go through multi-tier analysis:

1. 👤 Face detection
2. 🏃 Motion centroid
3. 🔎 Saliency map
4. 🎯 Center fallback

Slides trigger slow pan animation.

Zoom-out outro rendered via OpenCV (FFmpeg cannot animate dynamic crop width).

---

# 🖼 Logo Overlay

Logo file passed via CLI:

```bash
python process.py video.mp4 logo.png
```

Display configured in `editor.config.json`.

* Corner placement
* Opacity
* Relative scaling
* Margin control

---

# 🧠 Editable Transcription

Whisper output cached in:

```
output/MyVideo/words.json
```

You can edit incorrect words manually.

Rebuild subtitles only:

```bash
python process.py video.mp4 logo.png --regen-subs --force-burn
```

No re-transcription required.

---

# 📐 Output Specs

* Resolution: 1080×1920
* Codec: H.264
* Audio: AAC 192k
* Each scene exported as independent MP4

Upload-ready.

---

# 🏗 Design Principles

* Config > Code
* Deterministic outputs
* Portable fonts
* No hidden state
* Modular architecture
* Extensible pipeline

---

# 🧩 Project Structure

```
mini-video-editor/
├── process.py              ← main script
├── batch_process.py        ← batch processing script - use this to process all video files in a folder 
├── config.py
├── editor.config.json      ← all tuning knobs (font, colour, crop, logo…) - edit this file
├── fonts/
├── videos/                 ← folder that contains videos to be processed
│   └── MyVideo.mp4         ← specific video can be processed, or complete folder for processing
├── logo.png                ← logo (PNG with transparency recommended)
└── output/
    └── MyVideo/
        ├── MyVideo_precrop.mp4     ← Step 0 intermediate (cached)
        ├── scenes.json             ← scene list with snapped boundaries (cached)
        ├── words.json              ← Whisper word timestamps (cached, hand-editable)
        ├── scenes/
        │   └── scene_01.mp4 …     ← portrait-converted scene clips
        ├── subtitles/
        │   └── scene_01.ass …     ← karaoke subtitle files
        └── final/
            └── scene_01.mp4 …     ← finished clips with subtitles + logo
```

---

# 🤝 Quick Start Commands

Processing: Single File
```bash
python process.py video.mp4 logo.png
```

Processing: Folder (with multiple files)
```bash
 python batch_process.py videos/subfolder01 "logo.png"
```

---

# 🤝 Contributions

Ideas:
* Please reach-out at https://www.linkedin.com/in/parveensharma/

