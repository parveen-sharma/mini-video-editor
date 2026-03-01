# 🎬 Mini Video Editor

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

![Image](https://async.com/blog/content/images/2025/02/Choose-the-style.webp)

![Image](https://blitzcutai.com/_next/image?q=75\&url=%2Fblog%2Fcaption-fonts-tiktok.png\&w=3840)

![Image](https://www.notta.ai/pictures/how-to-add-subtitles-on-tiktok-cover.png)

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
├── config.py
├── editor.config.json      ← all tuning knobs (font, colour, crop, logo…)
├── fonts/
├── videos/
│   └── MyVideo.mp4
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

# 🤝 Contributions

Ideas:
* Please reach-out at https://www.linkedin.com/in/parveensharma/

