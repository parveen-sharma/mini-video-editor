# Mini Video Editor — Horizontal → Vertical Shorts

Converts a horizontal video into portrait-format (9:16) short clips with
karaoke-style word-by-word highlighted subtitles and an optional logo overlay.

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) and `ffprobe` on your PATH
- Python packages:

```bash
pip install openai-whisper opencv-python numpy tqdm
```

---

## Project layout

```
mini-video-editor/
├── process.py              ← main script
├── subtitle.config.json    ← all tuning knobs (font, colour, crop, logo…)
├── videos/
│   └── MyVideo.mp4
├── numberDesign.png        ← logo (PNG with transparency recommended)
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

## Quick start

**First run — everything from scratch:**

```bash
python process.py videos/MyVideo.mp4 numberDesign.png \
  --regen-words --regen-subs --regen-splits --force-burn
```

This runs the full pipeline end-to-end. Subsequent runs are incremental — only
missing steps are executed unless you explicitly force a step to re-run.

---

## Flags — each step is independently controlled

| Flag | Controls | Re-runs even if output exists |
|---|---|---|
| `--regen-words` | Step 2 — Whisper transcription | Yes — overwrites `words.json` |
| `--regen-subs` | Step 5 — ASS subtitle generation | Yes — overwrites all `.ass` files |
| `--regen-splits` | Step 4 — scene splitting & portrait encode | Yes — deletes and re-creates scene MP4s |
| `--force-burn` | Step 6 — burn subtitles + logo onto finals | Yes — overwrites all `final/` MP4s |

**Default behaviour (no flags):** only runs steps whose output files are missing.
Nothing is re-processed if it already exists.

### Common workflows

```bash
# First full run
python process.py videos/MyVideo.mp4 logo.png \
  --regen-words --regen-subs --regen-splits --force-burn

# Fix wrong words in transcription (edit words.json first, then):
python process.py videos/MyVideo.mp4 logo.png --regen-subs --force-burn

# Change subtitle font / colour / size (edit subtitle.config.json, then):
python process.py videos/MyVideo.mp4 logo.png --regen-subs --force-burn

# Change logo position / opacity only:
python process.py videos/MyVideo.mp4 logo.png --force-burn

# Change pre-crop settings (edit config, delete _precrop.mp4, then):
python process.py videos/MyVideo.mp4 logo.png --regen-splits --force-burn

# Re-transcribe with no other changes:
python process.py videos/MyVideo.mp4 logo.png --regen-words
```

---

## Pipeline steps

### Step 0 — Pre-crop
Trims borders/watermarks from the source video and saves a clean intermediate
(`_precrop.mp4`). Controlled by the `precrop` block in config.

**Cached.** Delete `_precrop.mp4` to force re-crop when you change crop settings.

### Step 1 — Scene detection
Finds visual cut points by diffing consecutive frames. Saves `scenes.json`.

**Cached.** Delete `scenes.json` to re-detect scenes.

### Step 2 — Transcription (`--regen-words`)
Runs OpenAI Whisper on the pre-cropped video and saves word-level timestamps
to `words.json`.

**`words.json` is designed to be hand-edited** — see section below.

### Step 3 — Boundary snapping
Adjusts each scene boundary to the nearest word end within a tolerance window,
so audio never cuts mid-sentence. Runs automatically after Step 2.

### Step 4 — Scene splitting (`--regen-splits`)
Splits the pre-cropped video into portrait (9:16) clips. Each scene gets:
- Motion-based or saliency-based crop centering
- Pan animation for static slides
- Zoom-out outro revealing the full slide

### Step 5 — Subtitle generation (`--regen-subs`)
Writes one `.ass` karaoke subtitle file per scene from `words.json`.
Use `--regen-subs` to regenerate after editing `words.json` or changing
subtitle styling in config.

### Step 6 — Burn (`--force-burn`)
Burns subtitles and the logo overlay onto each scene, writing to `final/`.

---

## Editing `words.json` (fixing transcription errors)

Whisper occasionally mishears words, especially proper nouns, acronyms, or
domain-specific terms. You can correct these without re-running Whisper.

**File location:** `output/MyVideo/words.json`

**Format — each entry:**
```json
{ "word": " hello", "start": 1.23, "end": 1.56 }
```

- `word` — the transcribed word. Note the **leading space** — Whisper includes
  it as part of the word token. Keep it when editing to preserve correct spacing
  in subtitles.
- `start` — word start time in **seconds relative to the full video** (not the scene).
- `end` — word end time in seconds relative to the full video.

**Example fix:** Whisper wrote `" recieve"`, you change it to `" receive"`.
Only edit the `word` field. Do not change `start`/`end` unless you know the
correct timestamp.

**After editing:**
```bash
python process.py videos/MyVideo.mp4 logo.png --regen-subs --force-burn
```

---

## Configuration — `subtitle.config.json`

All visual and behavioural settings live here. No code changes needed.

### Pre-crop

Removes borders, letterboxing, and watermarks before processing.

```json
"precrop": {
  "horizontal_keep_pct": 0.95,
  "vertical_keep_pct": 0.85,
  "horizontal_anchor": "center",
  "vertical_anchor": "top"
}
```

| Key | Values | Effect |
|---|---|---|
| `horizontal_keep_pct` | 0.0–1.0 | Fraction of frame **width** to keep. `0.95` removes 2.5% from each side (for `center` anchor). |
| `vertical_keep_pct` | 0.0–1.0 | Fraction of frame **height** to keep. `0.85` removes 15% vertically. |
| `horizontal_anchor` | `left` `center` `right` | Which horizontal edge the kept region is anchored to. |
| `vertical_anchor` | `top` `middle` `bottom` | Which vertical edge the kept region is anchored to. |

**Anchor reference — what gets trimmed:**

```
vertical_anchor=top, horizontal_anchor=center, V=0.85, H=0.95

┌─────────────────────────────────┐  ← top of frame
│  2.5%   ┌───────────────┐  2.5% │  ← kept region starts (flush top)
│  trim   │               │  trim │
│         │  kept region  │       │
│         │  95%W × 85%H  │       │
│         │               │       │
│         └───────────────┘       │  ← kept region ends
│         15% trimmed from bottom │
└─────────────────────────────────┘
```

**Delete `_precrop.mp4`** after changing these settings so the intermediate is regenerated.

### Subtitles

```json
"font": {
  "family": "Arial",
  "size": 70,
  "inactive_color": "#FFFFFF"
},
"highlight": {
  "enabled": true,
  "text_color": "#FFD700",
  "background_color": "#0A1F2A",
  "background_opacity": 0.6,
  "padding_x": 8,
  "padding_y": 6
},
"max_words_per_line": 4,
"max_lines": 2,
"margin_horizontal_px": 120,
"margin_vertical_px": 200,
"extend_last_word_ms": 300,
"pause_threshold_ms": 400
```

| Key | Effect |
|---|---|
| `font.size` | Font size in pixels on the 1080×1920 canvas. Recommended: 60–80. |
| `font.inactive_color` | Colour of words not currently being spoken. |
| `highlight.text_color` | Colour of the active (currently spoken) word. |
| `highlight.background_color` | Pill background behind the active word. |
| `highlight.background_opacity` | 0 = transparent pill, 1 = fully opaque. |
| `max_words_per_line` | Line wrapping. 4 words/line × 2 lines = 8 words max on screen. |
| `pause_threshold_ms` | Gap between words (ms) that starts a new subtitle chunk. |
| `extend_last_word_ms` | How long the last word of a chunk stays highlighted after it ends. |

After changing any of these: `--regen-subs --force-burn`

### Logo overlay

The **logo file path** is provided as the second CLI argument — not in config.
Config only controls how it is displayed.

```bash
python process.py videos/MyVideo.mp4 numberDesign.png --force-burn
#                                    ^^^^^^^^^^^^^^^^
#                                    this is the logo path
```

```json
"logo": {
  "enabled": true,
  "corner": "bottom-right",
  "width_pct": 0.12,
  "margin_x": 30,
  "margin_y": 30,
  "opacity": 0.85
}
```

| Key | Values | Effect |
|---|---|---|
| `enabled` | `true` / `false` | Toggle logo on/off without changing your command. |
| `corner` | `top-left` `top-right` `bottom-left` `bottom-right` | Which corner to place the logo. |
| `width_pct` | 0.0–1.0 | Logo width as fraction of output frame (1080px). `0.12` ≈ 130px wide. Height scales proportionally. |
| `margin_x` | pixels | Distance from the chosen horizontal edge. |
| `margin_y` | pixels | Distance from the chosen vertical edge. |
| `opacity` | 0.0–1.0 | `1.0` = fully opaque, `0.5` = semi-transparent watermark. |

After changing logo settings: `--force-burn`

### Scene boundary snapping

```json
"snap_tolerance_sec": 1.0
```

When snapping scene cut points to word ends, this is the maximum distance
(in seconds) the boundary can move. Increase to `1.5` if long sentences frequently
land outside the window. Decrease to `0.5` for tighter alignment.

---

## Output format

- Resolution: **1080 × 1920** (9:16 portrait)
- Video codec: H.264 (libx264), CRF 18
- Audio codec: AAC, 192 kbps
- Each scene is one self-contained MP4 — ready to upload to YouTube Shorts,
  Instagram Reels, or TikTok

---

## Troubleshooting

**Subtitles not visible**
Check that `subtitle.config.json` has `"highlight": { "enabled": true }` and
that `font.inactive_color` contrasts with the video background (white text on
a white slide will be invisible).

**Logo not appearing**
Verify `"enabled": true` in the `logo` block and that the `path` points to an
existing file relative to where you run the script.

**Audio cuts mid-sentence**
The scene boundary snapper requires `words.json` to be present. Run with
`--regen-words` at least once so Whisper generates the timestamps.

**Pre-crop is cutting into slide content**
Increase `horizontal_keep_pct` or `vertical_keep_pct` closer to `1.0`, or
change the anchor so the content-rich edge is not trimmed. Delete
`_precrop.mp4` and re-run with `--regen-splits --force-burn`.

**Wrong words in subtitles**
Edit `output/MyVideo/words.json` directly (keep leading spaces, only change
the `word` field), then run `--regen-subs --force-burn`.

**Scene detection splits too aggressively / not enough**
Edit the threshold in `detect_scenes()` — currently `diff > 22`. Lower it
(e.g. `18`) for more splits, raise it (e.g. `28`) for fewer. Delete
`scenes.json` and re-run.
